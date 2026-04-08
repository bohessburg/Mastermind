#pragma once

#include "action_space.h"
#include "agents.h"
#include "../game/game_state.h"
#include "../game/cards/base_cards.h"
#include "../game/cards/level_1_cards.h"

#include <functional>
#include <string>
#include <vector>

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

    // Build a runner around a pre-existing state (used by MCTS rollouts).
    // Setup (supply, starting decks, start_game) is skipped — caller is
    // responsible for the state being mid-game.
    explicit GameRunner(GameState state);

    GameResult run(std::vector<Agent*> agents);

    // Resume mid-game from `start_phase` of the current player's turn, then
    // continue normal turn order until terminal or timeout. Used by MCTS to
    // play out a cloned state from a top-level decision point.
    GameResult resume(std::vector<Agent*> agents, Phase start_phase);

    const GameState& state() const;

    void set_observer(GameObserver observer);

private:
    GameState state_;
    std::vector<std::string> kingdom_cards_;
    int total_decisions_;
    std::vector<Agent*> agents_;
    GameObserver observer_;

    DecisionFn make_decision_fn();
    void run_turn(int pid);
    void run_action_phase(int pid);
    void run_treasure_phase(int pid);
    void run_buy_phase(int pid);

    void observe(const std::string& msg);
};
