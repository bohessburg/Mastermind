#pragma once

#include "agents.h"
#include "../game/game_state.h"

#include <cstdint>
#include <random>
#include <vector>

// Rollout policy used by MCTS rollouts. Same logic as BetterRandomAgent
// (always plays-all-treasures, always buys Province if affordable, otherwise
// uniform random over non-pass options) but owned by the MCTS instance so it
// has its own RNG state.
class BetterRandomRolloutPolicy : public Agent {
public:
    explicit BetterRandomRolloutPolicy(uint64_t seed);
    std::vector<int> decide(const DecisionPoint& dp, const GameState& state) override;
private:
    std::mt19937 rng_;
};

class HeuristicRolloutPolicy : public Agent {
    public:
        explicit HeuristicRolloutPolicy(uint64_t seed);
        std::vector<int> decide(const DecisionPoint& dp, const GameState& state) override;
    private:
        std::mt19937 rng_;
};

// Open-loop, shallow MCTS over the options at a single top-level decision
// point. The "tree" is one level deep — children are option choices, no
// further expansion. Each iteration:
//   1. Pick a child via UCB1.
//   2. Clone the game state.
//   3. Drive a fresh GameRunner from the clone with a scripted-then-rollout
//      agent that plays the chosen option once, then defers to the rollout
//      policy. The opponent uses the rollout policy throughout.
//   4. Backprop +1 / -1 / 0 (win / loss / tie or timeout) into the chosen
//      child's stats.
//
// MCTS is used only for PLAY_ACTION / PLAY_TREASURE / BUY_CARD decisions.
// Sub-decisions inside card effects are handled by MCTSBot's fallback.
struct MCTSChild {
    int visits = 0;
    double total_value = 0.0;  // sum of returns from MCTS player's POV
};

class MCTS {
public:
    MCTS(int num_simulations, double exploration_c, uint64_t seed);

    // Run search from the given state at the given decision point. Returns
    // the option index of the most-visited child.
    int search(const GameState& state, const DecisionPoint& dp, int mcts_player_id);

private:
    int num_simulations_;
    double exploration_c_;
    std::mt19937 rng_;
    BetterRandomRolloutPolicy rollout_policy_;
};
