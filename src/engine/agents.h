#pragma once

#include "action_space.h"
#include "../game/game_state.h"

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
