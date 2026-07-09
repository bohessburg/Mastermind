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

struct MctsConfig {
    std::uint32_t sims_per_move = 100;
    float c_puct = 1.25F;
    std::uint8_t determinizations = 8;
    std::uint64_t rollout_seed = 0x4D435453ULL;
    std::uint32_t max_tree_nodes = 4096;
    MctsPriorFn prior_fn = nullptr;
    void* prior_user = nullptr;
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

class Mcts {
public:
    explicit Mcts(const MctsConfig& config);

    [[nodiscard]] Action choose(const GameState& root, PlayerId perspective) noexcept;
    void reset(const GameState& root, PlayerId perspective) noexcept;
    void run_simulations(std::uint32_t simulations, Xoshiro256pp& rng) noexcept;

    void add_virtual_loss(std::uint32_t node, float amount = 1.0F) noexcept;
    void revert_virtual_loss(std::uint32_t node, float amount = 1.0F) noexcept;

    [[nodiscard]] const MctsNode& node(std::uint32_t index) const noexcept;
    [[nodiscard]] std::uint32_t node_count() const noexcept;
    [[nodiscard]] bool exhausted() const noexcept;
    [[nodiscard]] Action best_root_action() const noexcept;
    [[nodiscard]] std::uint32_t root_visits_for(Action action) const noexcept;
    [[nodiscard]] float root_value_for(Action action) const noexcept;

private:
    [[nodiscard]] std::uint32_t allocate_node() noexcept;
    [[nodiscard]] bool expand(std::uint32_t node_index) noexcept;
    [[nodiscard]] std::uint32_t select_child(std::uint32_t node_index) const noexcept;
    void rollout(GameState& state, Xoshiro256pp& rng) const noexcept;
    void backpropagate(const std::uint32_t* path, std::uint8_t depth, const GameState& terminal) noexcept;
    [[nodiscard]] float terminal_value_for(PlayerId player, const GameState& terminal) const noexcept;
    [[nodiscard]] float child_value_for_parent(const MctsNode& parent, const MctsNode& child) const noexcept;
    [[nodiscard]] PlayerId next_player_for_child(const GameState& child_state, PlayerId parent_player) const noexcept;

    MctsConfig config_{};
    std::unique_ptr<MctsNode[]> nodes_;
    std::unique_ptr<GameState[]> states_;
    std::uint32_t capacity_ = 0;
    std::uint32_t node_count_ = 0;
    PlayerId root_perspective_ = 0;
    bool exhausted_ = false;
};

[[nodiscard]] float mcts_terminal_value(const GameState& state, PlayerId player) noexcept;
[[nodiscard]] Action mcts_choose(
    const GameState& state,
    PlayerId perspective,
    const MctsConfig& config) noexcept;
