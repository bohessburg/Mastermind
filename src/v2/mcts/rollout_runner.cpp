#include "v2/mcts/rollout_runner.h"

#include "v2/core/determinize.h"
#include "v2/core/game.h"
#include "v2/core/rng.h"
#include "v2/core/score.h"
#include "v2/core/turns.h"
#include "v2/mcts/tree.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

// A playout that needs more engine steps than this is treated as stuck and
// scored as a truncated game; real games finish in a few thousand steps.
constexpr std::uint32_t MAX_ROLLOUT_STEPS = 50'000U;

[[nodiscard]] PlayerId rollout_decision_player(const GameState& state) noexcept {
    if (state.decision.player < state.num_players) {
        return state.decision.player;
    }
    return current_player(state);
}

[[nodiscard]] bool is_over(const GameState& state) noexcept {
    return state.phase == static_cast<std::uint8_t>(Phase::Over);
}

[[nodiscard]] int mask_count(const ActionMask& mask) noexcept {
    int count = 0;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        count += mask.test(action) ? 1 : 0;
    }
    return count;
}

} // namespace

struct RolloutRunner::Slot {
    GameState state{};
    ActionMask mask{};
    Xoshiro256pp rng{};
    std::uint32_t tag = 0;
    std::uint32_t steps = 0;
    PlayerId perspective = NONE;
    bool active = false;
};

RolloutRunner::RolloutRunner(const RolloutConfig& config) : config_(config) {
    if (config_.n_slots == 0U) {
        throw std::invalid_argument("RolloutConfig.n_slots must be positive");
    }
    if (!is_valid_obs_version(config_.obs_version)) {
        throw std::invalid_argument("RolloutConfig.obs_version must be V1, V2, or V3");
    }
    if (!(config_.temperature >= 0.0F) || !std::isfinite(config_.temperature)) {
        throw std::invalid_argument("RolloutConfig.temperature must be finite and non-negative");
    }
    obs_size_ = obs_size_for(config_.obs_version);
    slots_.reset(new Slot[config_.n_slots]);
    leaf_obs_.reset(new float[static_cast<std::size_t>(config_.n_slots) * obs_size_]);
    leaf_masks_.reset(new bool[static_cast<std::size_t>(config_.n_slots) * ACTION_SPACE_SIZE]);
    leaf_slots_.reset(new std::uint32_t[config_.n_slots]);
}

RolloutRunner::~RolloutRunner() = default;

std::uint32_t RolloutRunner::add_position(const GameState& state) {
    if (is_over(state)) {
        throw std::invalid_argument("RolloutRunner position is already terminal");
    }
    positions_.push_back(state);
    return static_cast<std::uint32_t>(positions_.size() - 1U);
}

void RolloutRunner::add_job(const RolloutJob& job) {
    if (job.position >= positions_.size()) {
        throw std::invalid_argument("RolloutJob.position is out of range");
    }
    ActionMask legal{};
    (void)Game::legal_actions(positions_[job.position], legal);
    if (job.action >= ACTION_SPACE_SIZE || !legal.test(job.action)) {
        throw std::invalid_argument("RolloutJob.action is not legal in its position");
    }
    jobs_.push_back(job);
}

std::uint32_t RolloutRunner::active_rollouts() const noexcept {
    std::uint32_t active = 0;
    for (std::uint32_t i = 0; i < config_.n_slots; ++i) {
        active += slots_[i].active ? 1U : 0U;
    }
    return active;
}

void RolloutRunner::step_slot(Slot& slot, Action action) noexcept {
    (void)Game::step(slot.state, action);
    if (config_.max_turns != 0U && slot.state.turn_counter >= config_.max_turns) {
        slot.state.truncated = 1U;
        slot.state.phase = static_cast<std::uint8_t>(Phase::Over);
    }
    ++slot.steps;
}

bool RolloutRunner::start_next(Slot& slot) noexcept {
    slot.active = false;
    if (next_job_ >= jobs_.size()) {
        return false;
    }
    const RolloutJob& job = jobs_[next_job_];
    ++next_job_;
    slot.state = positions_[job.position];
    slot.perspective = rollout_decision_player(slot.state);
    if (job.determinize) {
        determinize(slot.state, slot.perspective, job.seed);
    } else {
        // Keep the true hidden zones; only future shuffles become independent.
        slot.state.rng = Xoshiro256pp::seeded(job.seed ^ 0xD6E8'FEB8'6659'FD93ULL);
    }
    slot.rng = Xoshiro256pp::seeded(job.seed ^ 0x524F'4C4C'4F55'5421ULL);
    slot.tag = job.tag;
    slot.steps = 0;
    slot.active = true;
    step_slot(slot, job.action);
    return true;
}

void RolloutRunner::finish(Slot& slot) noexcept {
    RolloutResult result{};
    result.tag = slot.tag;
    const PlayerId opponent = static_cast<PlayerId>(slot.perspective == 0U ? 1U : 0U);
    const PlayerId winner = game_winner(slot.state);
    result.outcome = winner == NONE ? 0.5F : (winner == slot.perspective ? 1.0F : 0.0F);
    const int margin = static_cast<int>(score(slot.state, slot.perspective))
        - static_cast<int>(score(slot.state, opponent));
    result.margin = static_cast<std::int16_t>(std::clamp(margin, -32768, 32767));
    result.turns = slot.state.turn_counter;
    result.truncated = slot.state.truncated != 0U;
    results_.push_back(result);
    slot.active = false;
}

bool RolloutRunner::advance(Slot& slot) noexcept {
    for (;;) {
        if (!slot.active && !start_next(slot)) {
            return false;
        }
        if (slot.steps >= MAX_ROLLOUT_STEPS && !is_over(slot.state)) {
            slot.state.truncated = 1U;
            slot.state.phase = static_cast<std::uint8_t>(Phase::Over);
        }
        if (is_over(slot.state)) {
            finish(slot);
            continue;
        }
        ActionMask legal{};
        const int legal_count = Game::legal_actions(slot.state, legal);
        if (legal_count <= 0) {
            slot.state.truncated = 1U;
            slot.state.phase = static_cast<std::uint8_t>(Phase::Over);
            continue;
        }
        if (config_.auto_play_treasures) {
            const Action treasure = mcts_canonical_treasure_play(
                mcts_filter_treasure_plays(slot.state, legal));
            if (treasure != A_PASS) {
                step_slot(slot, treasure);
                continue;
            }
        }
        ActionMask mask = config_.prune_treasure_plays
            ? mcts_filter_treasure_plays(slot.state, legal)
            : legal;
        int count = mask_count(mask);
        if (count <= 0) {
            mask = legal;
            count = legal_count;
        }
        if (count == 1) {
            step_slot(slot, mask.nth_set(0U));
            continue;
        }
        slot.mask = mask;
        return true;
    }
}

std::uint32_t RolloutRunner::collect_leaves() noexcept {
    if (leaf_count_ != 0U) {
        return leaf_count_;
    }
    for (std::uint32_t index = 0; index < config_.n_slots; ++index) {
        Slot& slot = slots_[index];
        if (!advance(slot)) {
            continue;
        }
        encode(
            slot.state,
            rollout_decision_player(slot.state),
            leaf_obs_.get() + (static_cast<std::size_t>(leaf_count_) * obs_size_),
            config_.obs_version);
        bool* mask = leaf_masks_.get() + (static_cast<std::size_t>(leaf_count_) * ACTION_SPACE_SIZE);
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            mask[action] = slot.mask.test(action);
        }
        leaf_slots_[leaf_count_] = index;
        ++leaf_count_;
    }
    return leaf_count_;
}

void RolloutRunner::provide_policies(const float* logits, std::uint32_t count) {
    if (count != leaf_count_) {
        throw std::invalid_argument("RolloutRunner.provide_policies count does not match pending leaves");
    }
    for (std::uint32_t leaf = 0; leaf < count; ++leaf) {
        Slot& slot = slots_[leaf_slots_[leaf]];
        const float* row = logits + (static_cast<std::size_t>(leaf) * ACTION_SPACE_SIZE);
        Action best = A_END;
        float best_logit = 0.0F;
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            if (!slot.mask.test(action)) {
                continue;
            }
            const float value = std::isfinite(row[action]) ? row[action] : -1.0e30F;
            if (best == A_END || value > best_logit) {
                best = action;
                best_logit = value;
            }
        }
        Action chosen = best;
        if (config_.temperature > 0.0F) {
            // Softmax over the legal logits, stabilised by the maximum.
            float weights[ACTION_SPACE_SIZE]{};
            double total = 0.0;
            for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
                if (!slot.mask.test(action)) {
                    continue;
                }
                const float value = std::isfinite(row[action]) ? row[action] : -1.0e30F;
                weights[action] = std::exp((value - best_logit) / config_.temperature);
                total += static_cast<double>(weights[action]);
            }
            double sample = (static_cast<double>(slot.rng.next() >> 11U) * 0x1.0p-53) * total;
            for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
                if (!slot.mask.test(action)) {
                    continue;
                }
                sample -= static_cast<double>(weights[action]);
                if (sample <= 0.0) {
                    chosen = action;
                    break;
                }
            }
        }
        step_slot(slot, chosen);
        ++decisions_;
    }
    leaf_count_ = 0;
}

std::vector<RolloutResult> RolloutRunner::take_results() {
    std::vector<RolloutResult> out;
    out.swap(results_);
    return out;
}
