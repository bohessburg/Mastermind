#pragma once

#include "action_space.h"
#include "../game/game_state.h"

#include <cstdint>
#include <memory>
#include <random>
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

class BetterRandomAgent : public Agent {
public:
    explicit BetterRandomAgent(uint64_t seed = 42);
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

// Engine builder: kingdom-aware strategy selection.
// FULL_ENGINE: Chapel + Village + terminal draw → build engine, then green
// BM_PLUS_X: Good terminal but no village → BigMoney + 1-2 key actions
// PURE_BM: Nothing useful → falls back to BigMoney
class EngineBot : public Agent {
public:
    std::vector<int> decide(const DecisionPoint& dp, const GameState& state) override;
};

// Forward decl — defined in mcts.h. Kept opaque so agents.h doesn't pull
// the MCTS implementation into every translation unit.
class MCTS;

// Shallow open-loop MCTS at top-level decisions (PLAY_ACTION /
// PLAY_TREASURE / BUY_CARD). Sub-decisions inside card effects are
// handled by an internal BetterRandomAgent fallback (the same logic
// used by the rollout policy, for consistency).
class MCTSBot : public Agent {
public:
    // num_simulations: rollouts per top-level decision (default 200)
    // exploration_c:   UCB1 exploration constant (default sqrt(2))
    // seed:            seeds both the MCTS RNG and the sub-decision fallback
    explicit MCTSBot(int num_simulations = 200,
                     double exploration_c = 1.41421356237,
                     uint64_t seed = 42);

    // Overload for plugging in a custom rollout policy. Ownership is taken.
    MCTSBot(int num_simulations,
            double exploration_c,
            uint64_t seed,
            std::unique_ptr<Agent> rollout_policy);

    ~MCTSBot() override;

    std::vector<int> decide(const DecisionPoint& dp, const GameState& state) override;

private:
    std::unique_ptr<MCTS> mcts_;
    BetterRandomAgent fallback_;  // for sub-decisions inside card effects
};