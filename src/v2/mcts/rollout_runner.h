#pragma once

#include "v2/core/actions.h"
#include "v2/core/state.h"
#include "v2/encode/encoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// Batched policy playouts for decision-level diagnostics (2026-11 drawing
// board). A job takes a stored root position, samples a world for the root's
// deciding player, applies one candidate action, and then plays the game out
// with an externally evaluated policy for BOTH seats. The mean outcome over
// many jobs is an estimate of the candidate's value that never consults a
// value head and, with determinize set, never sees hidden information.
//
// This is an analysis tool: it allocates freely and is not used by training.
struct RolloutConfig {
    // Concurrent playouts; also the maximum leaf batch.
    std::uint32_t n_slots = 1024;
    ObsVersion obs_version = ObsVersion::V1;
    // Mirror the self-play runner's move conventions so a playout follows the
    // same decision stream the search and the recorded policy targets use.
    bool auto_play_treasures = false;
    bool prune_treasure_plays = false;
    // Same semantics as SelfPlayConfig::selfplay_max_turns (0 = engine cap).
    std::uint16_t max_turns = 0;
    // 0 plays the argmax of the provided logits; >0 samples softmax(logits/T).
    float temperature = 0.0F;
};

struct RolloutJob {
    std::uint32_t position = 0;
    Action action = A_PASS;
    // Seeds the sampled world and the playout's own sampling stream. Jobs that
    // share a position and seed play the SAME world (common random numbers),
    // which is what makes a paired comparison of two actions cheap.
    std::uint64_t seed = 0;
    std::uint32_t tag = 0;
    // true: honest world (root player's deck order and the opponent's hand /
    // deck resampled). false: keep the true hidden state and only reseed
    // future shuffles (a clairvoyant-world playout).
    bool determinize = true;
};

struct RolloutResult {
    std::uint32_t tag = 0;
    // For the root position's deciding player: 1 win, 0.5 tie, 0 loss.
    float outcome = 0.5F;
    std::int16_t margin = 0;
    std::uint16_t turns = 0;
    bool truncated = false;
};

class RolloutRunner {
public:
    explicit RolloutRunner(const RolloutConfig& config);
    ~RolloutRunner();

    // Returns the position id used by RolloutJob::position.
    [[nodiscard]] std::uint32_t add_position(const GameState& state);
    // Throws when the position id is unknown or the action is illegal there.
    void add_job(const RolloutJob& job);

    // Advances every playout to its next policy decision and returns how many
    // need an evaluation. Returning zero means all queued work is finished.
    [[nodiscard]] std::uint32_t collect_leaves() noexcept;
    // One row of ACTION_SPACE_SIZE logits per pending leaf, in leaf order.
    void provide_policies(const float* logits, std::uint32_t count);

    [[nodiscard]] const float* leaf_observations() const noexcept { return leaf_obs_.get(); }
    [[nodiscard]] const bool* leaf_legal_masks() const noexcept { return leaf_masks_.get(); }
    [[nodiscard]] std::uint32_t leaf_count() const noexcept { return leaf_count_; }
    [[nodiscard]] std::size_t observation_size() const noexcept { return obs_size_; }
    [[nodiscard]] std::size_t queued_jobs() const noexcept { return jobs_.size() - next_job_; }
    [[nodiscard]] std::uint32_t active_rollouts() const noexcept;
    [[nodiscard]] std::uint64_t decisions_evaluated() const noexcept { return decisions_; }
    std::vector<RolloutResult> take_results();

private:
    struct Slot;

    [[nodiscard]] bool start_next(Slot& slot) noexcept;
    // Plays forced moves and auto-played treasures until the playout needs a
    // policy evaluation (returns true) or the slot runs out of work (false).
    [[nodiscard]] bool advance(Slot& slot) noexcept;
    void finish(Slot& slot) noexcept;
    void step_slot(Slot& slot, Action action) noexcept;

    RolloutConfig config_{};
    std::size_t obs_size_ = OBS_SIZE_V1;
    std::vector<GameState> positions_;
    std::vector<RolloutJob> jobs_;
    std::size_t next_job_ = 0;
    std::unique_ptr<Slot[]> slots_;
    std::unique_ptr<float[]> leaf_obs_;
    std::unique_ptr<bool[]> leaf_masks_;
    std::unique_ptr<std::uint32_t[]> leaf_slots_;
    std::uint32_t leaf_count_ = 0;
    std::uint64_t decisions_ = 0;
    std::vector<RolloutResult> results_;
};
