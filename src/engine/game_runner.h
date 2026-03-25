#pragma once

#include "action_space.h"
#include "../game/game_state.h"
#include "../game/cards/base_cards.h"
#include "../game/cards/base_kingdom.h"

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

class Agent {
public:
    virtual ~Agent() = default;
    virtual std::vector<int> decide(const DecisionPoint& dp, const GameState& state) = 0;
};

class RandomAgent : public Agent {
public:
    explicit RandomAgent(uint64_t seed = 42);
    std::vector<int> decide(const DecisionPoint& dp, const GameState& state) override;
private:
    std::mt19937 rng_;
};

class BigMoneyAgent : public Agent {
public:
    std::vector<int> decide(const DecisionPoint& dp, const GameState& state) override;
};

// Plays all actions by priority (villages first, then cantrips, then terminals).
// Buys the most expensive card it can afford (Province > other).
// Makes reasonable sub-decisions (trash junk, discard victory cards, etc.)
class HeuristicAgent : public Agent {
public:
    std::vector<int> decide(const DecisionPoint& dp, const GameState& state) override;
};

struct GameResult {
    std::vector<int> scores;
    int winner;
    int total_turns;
    int total_decisions;
};

// Observer callback for game events (used by interactive mode)
using GameObserver = std::function<void(const std::string& message)>;

class GameRunner {
public:
    GameRunner(int num_players, const std::vector<std::string>& kingdom_cards);

    GameResult run(std::vector<Agent*> agents);
    const GameState& state() const;

    // Set an observer to receive narration of game events
    void set_observer(GameObserver observer);

private:
    GameState state_;
    std::vector<std::string> kingdom_cards_;
    int total_decisions_;
    std::vector<Agent*> agents_;   // stored during run() for sub-decision routing
    GameObserver observer_;

    // Routes sub-decisions to the correct player's agent
    DecisionFn make_decision_fn();
    void run_turn(int pid);
    void run_action_phase(int pid);
    void run_treasure_phase(int pid);
    void run_buy_phase(int pid);

    void observe(const std::string& msg);
};
