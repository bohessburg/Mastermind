#include "mcts.h"
#include "game_runner.h"
#include "card_ids.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

// ═══════════════════════════════════════════════════════════════════
// BetterRandomRolloutPolicy — copy of BetterRandomAgent's logic with
// an independent RNG. Kept separate so MCTSBot can be tuned without
// touching the existing baseline agent.
// ═══════════════════════════════════════════════════════════════════

BetterRandomRolloutPolicy::BetterRandomRolloutPolicy(uint64_t seed) : rng_(seed) {}

std::vector<int> BetterRandomRolloutPolicy::decide(const DecisionPoint& dp,
                                                   const GameState& /*state*/) {
    if (dp.options.empty()) return {};

    if (dp.type == DecisionType::PLAY_TREASURE) {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            if (dp.options[i].is_play_all) return {i};
        }
    }

    if (dp.type == DecisionType::BUY_CARD) {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            if (dp.options[i].def_id == CardIds::Province) return {i};
        }
    }

    if (dp.min_choices > 1) {
        std::vector<int> all_indices;
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            all_indices.push_back(i);
        }
        std::shuffle(all_indices.begin(), all_indices.end(), rng_);
        int pick = std::min(dp.min_choices, static_cast<int>(all_indices.size()));
        return {all_indices.begin(), all_indices.begin() + pick};
    }

    std::vector<int> candidates;
    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
        if (!dp.options[i].is_pass) candidates.push_back(i);
    }

    if (candidates.empty()) {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            if (dp.options[i].is_pass) return {i};
        }
        return {0};
    }

    std::uniform_int_distribution<int> dist(0, static_cast<int>(candidates.size()) - 1);
    return {candidates[dist(rng_)]};
}

HeuristicRolloutPolicy::HeuristicRolloutPolicy(uint64_t seed) : rng_(seed) {}

std::vector<int> HeuristicRolloutPolicy::decide(const DecisionPoint& dp, const GameState&) {
    if (dp.options.empty()) return {};

    if (dp.type == DecisionType::PLAY_TREASURE) {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            if (dp.options[i].is_play_all) return {i};
        }
    }

    if (dp.type == DecisionType::BUY_CARD) {
        
    }
}

// ═══════════════════════════════════════════════════════════════════
// ScriptedThenRolloutAgent — internal helper. Returns a fixed first
// choice, then defers all subsequent decide() calls to a fallback
// agent (the rollout policy). One instance per MCTS simulation.
// ═══════════════════════════════════════════════════════════════════

namespace {

class ScriptedThenRolloutAgent : public Agent {
public:
    ScriptedThenRolloutAgent(int first_choice, Agent* fallback)
        : first_choice_(first_choice), fallback_(fallback), used_(false) {}

    std::vector<int> decide(const DecisionPoint& dp, const GameState& state) override {
        if (!used_) {
            used_ = true;
            // Defensive: if the cloned engine somehow asks a different
            // question first (shouldn't happen for top-level decisions),
            // fall through to the rollout policy instead of returning a
            // bogus index.
            if (first_choice_ >= 0 && first_choice_ < static_cast<int>(dp.options.size())) {
                return {first_choice_};
            }
        }
        return fallback_->decide(dp, state);
    }

private:
    int first_choice_;
    Agent* fallback_;
    bool used_;
};

Phase phase_for_decision(DecisionType type) {
    switch (type) {
        case DecisionType::PLAY_ACTION:   return Phase::ACTION;
        case DecisionType::PLAY_TREASURE: return Phase::TREASURE;
        case DecisionType::BUY_CARD:      return Phase::BUY;
        default:                          return Phase::ACTION;
    }
}

} // namespace

// ═══════════════════════════════════════════════════════════════════
// MCTS
// ═══════════════════════════════════════════════════════════════════

MCTS::MCTS(int num_simulations,
           double exploration_c,
           uint64_t seed,
           std::unique_ptr<Agent> rollout_policy)
    : num_simulations_(num_simulations)
    , exploration_c_(exploration_c)
    , rng_(seed)
    , rollout_policy_(std::move(rollout_policy))
{
}

int MCTS::search(const GameState& state, const DecisionPoint& dp, int mcts_player_id) {
    int num_options = static_cast<int>(dp.options.size());
    if (num_options == 0) return 0;
    if (num_options == 1) return 0;

    std::vector<MCTSChild> children(num_options);
    Phase start_phase = phase_for_decision(dp.type);

    int total_visits = 0;
    for (int sim = 0; sim < num_simulations_; sim++) {
        // --- Selection (UCB1, with unvisited children prioritized) ---
        int chosen = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < num_options; i++) {
            double score;
            if (children[i].visits == 0) {
                score = std::numeric_limits<double>::infinity();
            } else {
                double mean = children[i].total_value / children[i].visits;
                double explore = exploration_c_ *
                    std::sqrt(std::log(static_cast<double>(total_visits + 1)) /
                              children[i].visits);
                score = mean + explore;
            }
            if (score > best_score) {
                best_score = score;
                chosen = i;
            }
        }

        // --- Simulation ---
        GameState clone = state;
        // Detach any log handler the original state inherited from an
        // outer GameRunner — rollouts must not emit user-visible output.
        clone.set_log(nullptr);

        ScriptedThenRolloutAgent scripted(chosen, rollout_policy_.get());
        std::vector<Agent*> rollout_agents(clone.num_players(), rollout_policy_.get());
        rollout_agents[mcts_player_id] = &scripted;

        GameRunner runner(std::move(clone));
        runner.resume(rollout_agents, start_phase);

        // --- Value (from MCTS player's POV) ---
        double value;
        const GameState& terminal = runner.state();
        if (!terminal.is_game_over()) {
            // Timed out (max turns / max decisions reached). Treat as draw.
            value = 0.0;
        } else {
            std::vector<int> scores = terminal.calculate_scores();
            int my_score = scores[mcts_player_id];
            int best_other = std::numeric_limits<int>::min();
            for (int p = 0; p < static_cast<int>(scores.size()); p++) {
                if (p != mcts_player_id && scores[p] > best_other) best_other = scores[p];
            }
            if (my_score > best_other) value = 1.0;
            else if (my_score < best_other) value = -1.0;
            else value = 0.0;
        }

        // --- Backprop ---
        children[chosen].visits++;
        children[chosen].total_value += value;
        total_visits++;
    }

    // Return most-visited child (standard MCTS final-move selection).
    int best = 0;
    for (int i = 1; i < num_options; i++) {
        if (children[i].visits > children[best].visits) best = i;
    }
    return best;
}
