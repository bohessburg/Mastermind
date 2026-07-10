#pragma once

#include "v2/core/actions.h"
#include "v2/core/setup.h"
#include "v2/core/types.h"
#include "v2/encode/encoder.h"
#include "v2/mcts/selfplay.h"
#include "v2/mcts/tree.h"

#include <cstdint>
#include <memory>
#include <vector>

enum class EvalScriptedBotKind : std::uint8_t {
    Engine,
    BigMoney,
    Heuristic,
    Random,
    Mcts,
};

struct EvalRunnerConfig {
    std::uint32_t n_games = 64;
    std::uint32_t sims_per_move = 400;
    float c_puct = 1.25F;
    std::uint32_t max_batch = 512;
    std::uint64_t seed = 0x4556414CULL;
    std::uint32_t target_games = 0;
    SelfPlayKingdomMode kingdom_mode = SelfPlayKingdomMode::Random;
    Setup fixed_setup{};
    std::uint32_t max_tree_nodes = 4096;
    EvalScriptedBotKind opponent = EvalScriptedBotKind::Engine;
    bool retain_finished_games = false;
};

struct EvalRunnerResult {
    std::uint32_t games = 0;
    std::uint32_t nn_wins = 0;
    std::uint32_t scripted_wins = 0;
    std::uint32_t ties = 0;
    std::uint32_t truncated = 0;
};

class EvalRunner {
public:
    explicit EvalRunner(const EvalRunnerConfig& config);
    ~EvalRunner();

    [[nodiscard]] std::uint32_t collect_leaves(std::uint32_t max_batch) noexcept;
    void provide_evaluations(const float* values, const float* policies, std::uint32_t count) noexcept;
    [[nodiscard]] const float* leaf_observations() const noexcept;
    [[nodiscard]] const bool* leaf_legal_masks() const noexcept;
    [[nodiscard]] std::uint32_t leaf_count() const noexcept;
    [[nodiscard]] EvalRunnerResult result() const noexcept;
    [[nodiscard]] std::uint64_t games_completed() const noexcept;
    [[nodiscard]] float total_virtual_loss() const noexcept;
    [[nodiscard]] PlayerId active_nn_player(std::uint32_t index) const noexcept;
    [[nodiscard]] std::uint64_t active_sequence(std::uint32_t index) const noexcept;
    [[nodiscard]] Action last_scripted_action() const noexcept;
    std::vector<GameState> take_finished_games();

private:
    struct GameSlot;
    struct PendingLeaf;

    void reset_game(std::uint32_t index) noexcept;
    void start_search(GameSlot& game) noexcept;
    void drive_scripted(GameSlot& game) noexcept;
    void maybe_finish_move(GameSlot& game) noexcept;
    void finish_game(GameSlot& game) noexcept;
    [[nodiscard]] Setup setup_for(std::uint64_t sequence) const noexcept;
    void normalize_policy(const float* logits, const ActionMask& legal, int legal_count, float* out) const noexcept;
    [[nodiscard]] bool resolve_scripted_tree_leaf(GameSlot& game, const MctsPendingLeaf& leaf) noexcept;

    EvalRunnerConfig config_{};
    MctsConfig mcts_config_{};
    MctsConfig scaffold_mcts_config_{};
    std::unique_ptr<GameSlot[]> games_;
    std::unique_ptr<PendingLeaf[]> pending_;
    std::unique_ptr<float[]> leaf_obs_;
    std::unique_ptr<bool[]> leaf_masks_;
    std::unique_ptr<float[]> normalized_policy_;
    EvalRunnerResult result_{};
    std::uint32_t pending_count_ = 0;
    std::uint32_t next_collect_game_ = 0;
    std::uint64_t next_sequence_ = 0;
    Action last_scripted_action_ = A_PASS;
    std::vector<GameState> finished_games_;
};

[[nodiscard]] Action eval_scripted_action(
    const GameState& state,
    const ActionMask& legal,
    int legal_count,
    EvalScriptedBotKind kind,
    Xoshiro256pp& rng) noexcept;
