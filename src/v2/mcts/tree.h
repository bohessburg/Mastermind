#pragma once

#include "v2/core/actions.h"
#include "v2/core/state.h"
#include "v2/core/types.h"

#include <cstdint>
#include <memory>

inline constexpr std::uint32_t MCTS_NULL = 0xFFFF'FFFFU;

using MctsPriorFn = float (*)(
    const GameState& state,
    PlayerId player,
    Action action,
    void* user) noexcept;

enum class MctsRolloutPolicy : std::uint8_t {
    Random,
    Heuristic,
    EngineLike,
    External,
};

struct MctsConfig {
    std::uint32_t sims_per_move = 100;
    float c_puct = 1.25F;
    std::uint8_t determinizations = 2;
    std::uint64_t rollout_seed = 0x4D435453ULL;
    std::uint16_t rollout_step_cap = 1024U;
    std::uint32_t max_tree_nodes = 4096;
    MctsRolloutPolicy rollout_policy = MctsRolloutPolicy::EngineLike;
    float rollout_epsilon = 0.10F;
    MctsPriorFn prior_fn = nullptr;
    void* prior_user = nullptr;
    bool prune_treasure_plays = false;
    // Keeps the highest-prior legal actions plus one random wildcard when
    // expanding a non-root node. Zero preserves full-width expansion; roots
    // always retain every legal action.
    std::uint8_t expand_top_k = 0;
    // Internal owner opt-in for cross-decision re-rooting. The runner is the
    // only current owner that enables it.
    bool tree_reuse = false;
    // When an adopted root already has visits, this guarantees a small
    // amount of new exploration after root-noise priors are applied.
    std::uint16_t min_new_sims = 64U;
};

struct MctsNode {
    std::uint32_t parent = MCTS_NULL;
    std::uint32_t first_child = MCTS_NULL;
    std::uint32_t next_sibling = MCTS_NULL;
    std::uint32_t state_index = MCTS_NULL;
    Action action_from_parent = A_PASS;
    PlayerId player = 0;
    std::uint32_t visits = 0;
    float value_sum = 0.0F;
    float prior = 0.0F;
    float virtual_loss = 0.0F;
    bool expanded = false;
    bool terminal = false;
};

inline constexpr std::uint8_t MCTS_MAX_PATH = 96;

struct MctsPendingLeaf {
    std::uint32_t node = MCTS_NULL;
    std::uint32_t state_index = MCTS_NULL;
    std::uint32_t path[MCTS_MAX_PATH]{};
    std::uint8_t depth = 0;
    PlayerId player = 0;
    ActionMask legal{};
    int legal_count = 0;
    bool valid = false;
};

class Mcts {
public:
    explicit Mcts(const MctsConfig& config);

    [[nodiscard]] Action choose(const GameState& root, PlayerId perspective) noexcept;
    // Allows serialized callers to reuse one scratch tree across independently
    // seeded games without reallocating its node/state buffers.
    void set_rollout_seed(std::uint64_t rollout_seed) noexcept { config_.rollout_seed = rollout_seed; }
    // Lets a reusable scratch tree switch between opening and endgame budgets
    // without reallocating its fixed node/state buffers.
    void set_sims_per_move(std::uint32_t sims_per_move) noexcept { config_.sims_per_move = sims_per_move; }
    void reset(const GameState& root, PlayerId perspective) noexcept;
    void run_simulations(std::uint32_t simulations, Xoshiro256pp& rng) noexcept;

    // Retain one selected root child until the runner knows whether its
    // post-action state is exactly the next searched decision.  Reuse is
    // intentionally limited to one determinization: a K>1 root aggregates
    // incompatible hidden-information worlds below its children.
    [[nodiscard]] bool retain_root_child(Action action, std::uint64_t post_action_hash) noexcept;
    [[nodiscard]] bool adopt_retained_root(const GameState& root, PlayerId perspective) noexcept;
    void clear_retained_root() noexcept;
    // The caller owns its per-decision started/completed counters. Pass true
    // only after a successful cross-decision adoption; fresh roots keep the
    // historical sims_per_move new-simulation budget.
    [[nodiscard]] std::uint32_t new_simulation_target(bool adopted_root) const noexcept;
    // Re-normalizes priors only across the existing root children.  The
    // runner uses this when re-applying root exploration noise after reuse.
    void set_root_priors(const float* priors) noexcept;

    void add_virtual_loss(std::uint32_t node, float amount = 1.0F) noexcept;
    void revert_virtual_loss(std::uint32_t node, float amount = 1.0F) noexcept;

    [[nodiscard]] const MctsNode& node(std::uint32_t index) const noexcept;
    [[nodiscard]] std::uint32_t node_count() const noexcept;
    [[nodiscard]] bool exhausted() const noexcept;
    [[nodiscard]] Action best_root_action() const noexcept;
    [[nodiscard]] std::uint32_t root_visits_for(Action action) const noexcept;
    [[nodiscard]] float root_value_for(Action action) const noexcept;
    [[nodiscard]] const GameState& state_for(std::uint32_t state_index) const noexcept;
    [[nodiscard]] bool collect_external_leaf(MctsPendingLeaf& leaf) noexcept;
    void provide_external_evaluation(
        const MctsPendingLeaf& leaf,
        float value,
        const float* priors) noexcept;
    void provide_external_evaluation(
        const MctsPendingLeaf& leaf,
        float value,
        const float* priors,
        Xoshiro256pp& rng) noexcept;
    void root_visit_policy(float* out, float temperature) const noexcept;
    [[nodiscard]] Action sample_root_action(float temperature, Xoshiro256pp& rng) const noexcept;
    [[nodiscard]] float total_virtual_loss() const noexcept;

private:
    [[nodiscard]] std::uint32_t allocate_node() noexcept;
    [[nodiscard]] bool compact_subtree_to_root(
        std::uint32_t retained_root,
        PlayerId perspective) noexcept;
    [[nodiscard]] bool expand(std::uint32_t node_index, Xoshiro256pp& rng) noexcept;
    [[nodiscard]] bool expand_with_priors(
        std::uint32_t node_index,
        const float* priors,
        Xoshiro256pp& rng) noexcept;
    [[nodiscard]] std::uint32_t select_child(std::uint32_t node_index) const noexcept;
    void rollout(GameState& state, Xoshiro256pp& rng) const noexcept;
    void backpropagate(const std::uint32_t* path, std::uint8_t depth, const GameState& terminal) noexcept;
    void backpropagate_value(
        const std::uint32_t* path,
        std::uint8_t depth,
        PlayerId value_player,
        float value) noexcept;
    void apply_virtual_loss(const std::uint32_t* path, std::uint8_t depth, float amount) noexcept;
    [[nodiscard]] float terminal_value_for(PlayerId player, const GameState& terminal) const noexcept;
    [[nodiscard]] float child_value_for_parent(const MctsNode& parent, const MctsNode& child) const noexcept;
    [[nodiscard]] PlayerId next_player_for_child(const GameState& child_state, PlayerId parent_player) const noexcept;

    MctsConfig config_{};
    std::unique_ptr<MctsNode[]> nodes_;
    std::unique_ptr<GameState[]> states_;
    // Fixed remap scratch keeps re-root compaction allocation-free.
    std::unique_ptr<std::uint32_t[]> remap_;
    std::uint32_t capacity_ = 0;
    std::uint32_t node_count_ = 0;
    PlayerId root_perspective_ = 0;
    bool exhausted_ = false;
    std::uint32_t retained_root_ = MCTS_NULL;
    // Recorded against states_[retained_root_] when the runner commits an
    // action; adoption verifies both that stored child state and the live one.
    std::uint64_t retained_root_state_hash_ = 0;
    // Deterministic fallback for external-policy callers that do not pass
    // their search RNG into the tree.
    Xoshiro256pp expansion_rng_ = Xoshiro256pp::seeded(0x4D435453ULL);
};

[[nodiscard]] std::uint64_t mcts_state_hash(const GameState& state) noexcept;
// Return the top (k - 1) legal actions by prior plus one uniformly selected
// action from the excluded remainder. Invalid inputs fall back to the full
// legal mask, preserving a selectable legal action.
[[nodiscard]] ActionMask mcts_top_k_actions(
    const ActionMask& legal,
    int legal_count,
    const float* priors,
    std::uint8_t top_k,
    Xoshiro256pp& rng) noexcept;
// Mix Dirichlet exploration into an action set. Root callers always pass the
// complete legal mask so all legal moves receive exploration support.
void mcts_add_dirichlet_noise(
    float* priors,
    const ActionMask& actions,
    int action_count,
    float alpha,
    float frac,
    Xoshiro256pp& rng) noexcept;
[[nodiscard]] float mcts_terminal_value(const GameState& state, PlayerId player) noexcept;
[[nodiscard]] ActionMask mcts_filter_treasure_plays(
    const GameState& state,
    const ActionMask& legal) noexcept;
[[nodiscard]] Action mcts_canonical_treasure_play(const ActionMask& legal) noexcept;
// The EngineLike buying branch used by MCTS rollouts. Its pile-clock guard is
// shared with the separate Engine chart.
[[nodiscard]] Action mcts_engine_like_rollout_buy(
    const GameState& state,
    const ActionMask& legal) noexcept;
[[nodiscard]] Action mcts_choose(
    const GameState& state,
    PlayerId perspective,
    const MctsConfig& config) noexcept;
