#pragma once

#include "../engine/action_space.h"
#include "../engine/game_runner.h"
#include "../engine/agents.h"
#include "../game/game_state.h"
#include "../game/cards/base_cards.h"
#include "../game/cards/level_1_cards.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <stdexcept>

// A "steppable" wrapper around GameRunner that pauses at each decision point,
// letting an external caller (Python) provide actions one at a time.
//
// Internally it runs the game loop on a background thread. Whenever the game
// needs a decision, the thread blocks and the main thread gets the
// DecisionPoint. The main thread calls step() with the chosen action indices,
// which wakes the game thread to continue.
//
// This avoids rewriting the deeply-nested game loop (action phase → card
// effects → sub-decisions) into a flat state machine.

class SteppableGame {
public:
    SteppableGame(int num_players, const std::vector<std::string>& kingdom_cards)
        : num_players_(num_players)
        , kingdom_cards_(kingdom_cards)
        , ai_agents_(num_players)           // nullptr = human-controlled
    {}

    ~SteppableGame() {
        shutdown();
    }

    // Assign an AI agent to a player slot. That player's decisions will be
    // auto-resolved in C++ instead of being sent to Python.
    // agent_type: "random", "big_money", "heuristic", "engine_bot"
    void set_ai_agent(int player_id, const std::string& agent_type) {
        if (player_id < 0 || player_id >= num_players_) {
            throw std::runtime_error("Invalid player_id");
        }
        if (agent_type == "random") {
            ai_agents_[player_id] = std::make_unique<RandomAgent>();
        } else if (agent_type == "big_money") {
            ai_agents_[player_id] = std::make_unique<BigMoneyAgent>();
        } else if (agent_type == "heuristic") {
            ai_agents_[player_id] = std::make_unique<HeuristicAgent>();
        } else if (agent_type == "engine_bot") {
            ai_agents_[player_id] = std::make_unique<EngineBot>();
        } else if (agent_type == "" || agent_type == "human") {
            ai_agents_[player_id] = nullptr;
        } else {
            throw std::runtime_error("Unknown agent type: " + agent_type);
        }
    }

    // Check if a player slot is AI-controlled
    bool is_ai_player(int player_id) const {
        return player_id >= 0 && player_id < num_players_ && ai_agents_[player_id] != nullptr;
    }

    // --- Main API ---

    // Start a new game. Returns the first DecisionPoint.
    const DecisionPoint& reset() {
        // If a previous game thread is running, shut it down
        shutdown();

        game_over_ = false;
        has_pending_decision_ = false;

        // Launch the game loop on a background thread
        game_thread_ = std::thread([this]() { run_game(); });

        // Wait for the first decision point (or game over)
        std::unique_lock<std::mutex> lock(mtx_);
        cv_main_.wait(lock, [this]() { return has_pending_decision_ || game_over_; });

        if (game_over_) {
            throw std::runtime_error("Game ended before any decision");
        }
        return pending_dp_;
    }

    // Provide the chosen action indices for the current decision point.
    // Returns the next DecisionPoint.
    const DecisionPoint& step(const std::vector<int>& action) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (game_over_) {
                throw std::runtime_error("Game is already over");
            }
            if (!has_pending_decision_) {
                throw std::runtime_error("No pending decision to respond to");
            }
            // Store the response and wake the game thread
            response_ = action;
            has_pending_decision_ = false;
        }
        cv_game_.notify_one();

        // Wait for the next decision point (or game over)
        std::unique_lock<std::mutex> lock(mtx_);
        cv_main_.wait(lock, [this]() { return has_pending_decision_ || game_over_; });

        if (game_over_ && !has_pending_decision_) {
            // Return a sentinel decision point
            return pending_dp_;
        }
        return pending_dp_;
    }

    // --- State queries ---

    bool is_game_over() const { return game_over_; }

    const GameResult& get_result() const {
        if (!game_over_) throw std::runtime_error("Game not over yet");
        return result_;
    }

    // Access the underlying game state (read-only) for observation building
    const GameState& get_state() const {
        if (!runner_) throw std::runtime_error("Game not started");
        return runner_->state();
    }

    int num_players() const { return num_players_; }

private:
    int num_players_;
    std::vector<std::string> kingdom_cards_;
    std::vector<std::unique_ptr<Agent>> ai_agents_;  // nullptr = human

    // Synchronization between main thread and game thread
    std::mutex mtx_;
    std::condition_variable cv_main_;  // wake main thread (decision ready)
    std::condition_variable cv_game_;  // wake game thread (action provided)

    // Shared state (protected by mtx_)
    DecisionPoint pending_dp_;
    std::vector<int> response_;
    bool has_pending_decision_ = false;
    bool game_over_ = false;
    bool shutdown_requested_ = false;

    GameResult result_;
    std::unique_ptr<GameRunner> runner_;
    std::thread game_thread_;

    // The game loop — runs on the background thread
    void run_game() {
        runner_ = std::make_unique<GameRunner>(num_players_, kingdom_cards_);

        // Create a "proxy agent" that, instead of computing an action,
        // posts the DecisionPoint to the main thread and waits for a response.
        auto proxy = std::make_unique<ProxyAgent>(this);
        std::vector<Agent*> agents(num_players_, proxy.get());

        result_ = runner_->run(agents);

        // Signal game over
        {
            std::lock_guard<std::mutex> lock(mtx_);
            game_over_ = true;
        }
        cv_main_.notify_one();
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            shutdown_requested_ = true;
        }
        cv_game_.notify_one();

        if (game_thread_.joinable()) {
            game_thread_.join();
        }
        shutdown_requested_ = false;
    }

    // Agent implementation that bridges to the main thread
    class ProxyAgent : public Agent {
    public:
        ProxyAgent(SteppableGame* game) : game_(game) {}

        std::vector<int> decide(const DecisionPoint& dp, const GameState& state) override {
            // If this player has an AI agent, resolve immediately without
            // involving the Python/main thread at all
            if (game_->ai_agents_[dp.player_id]) {
                return game_->ai_agents_[dp.player_id]->decide(dp, state);
            }

            std::unique_lock<std::mutex> lock(game_->mtx_);

            // Check if we should abort (shutdown or game over via max turns)
            if (game_->shutdown_requested_) {
                // Return a pass/minimal action to let the game loop end
                return {static_cast<int>(dp.options.size()) - 1};
            }

            // Post the decision point to the main thread
            game_->pending_dp_ = dp;
            game_->has_pending_decision_ = true;
            game_->cv_main_.notify_one();

            // Wait for the main thread to provide a response
            game_->cv_game_.wait(lock, [this]() {
                return !game_->has_pending_decision_ || game_->shutdown_requested_;
            });

            if (game_->shutdown_requested_) {
                return {static_cast<int>(dp.options.size()) - 1};
            }

            return game_->response_;
        }

    private:
        SteppableGame* game_;
    };
};
