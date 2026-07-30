#pragma once

#include "v2/core/actions.h"
#include "v2/core/setup.h"
#include "v2/core/types.h"
#include "v2/encode/encoder.h"
#include "v2/mcts/selfplay.h"
#include "v2/mcts/tree.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

enum class EvalScriptedBotKind : std::uint8_t {
    Engine = 0,
    BigMoney = 1,
    Heuristic = 2,
    Random = 3,
    Mcts = 4,
    EngineV2 = 5,
    EngineV3 = 6,
    Thinner = 7,
};

struct EvalRunnerConfig {
    std::uint32_t n_games = 64;
    std::uint32_t sims_per_move = 400;
    float c_puct = 1.25F;
    MctsCPuctSchedule c_puct_schedule = MctsCPuctSchedule::Fixed;
    float c_puct_init = 1.25F;
    float c_puct_base = 19652.0F;
    std::uint32_t max_batch = 512;
    std::uint64_t seed = 0x4556414CULL;
    ObsVersion obs_version = ObsVersion::V1;
    std::uint32_t target_games = 0;
    SelfPlayKingdomMode kingdom_mode = SelfPlayKingdomMode::Random;
    Setup fixed_setup{};
    std::uint32_t max_tree_nodes = 4096;
    EvalScriptedBotKind opponent = EvalScriptedBotKind::Engine;
    bool retain_finished_games = false;
    bool auto_play_treasures = false;
    bool prune_treasure_plays = false;
    // Opt-in honest hidden-information root sampling for the NN seat.  The
    // scripted opponent continues to act on the live game state.
    SelfPlayDeterminizeMode determinize = SelfPlayDeterminizeMode::Off;
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
    [[nodiscard]] std::size_t observation_size() const noexcept;
    [[nodiscard]] EvalRunnerResult result() const noexcept;
    [[nodiscard]] std::uint64_t games_completed() const noexcept;
    [[nodiscard]] float total_virtual_loss() const noexcept;
    [[nodiscard]] PlayerId active_nn_player(std::uint32_t index) const noexcept;
    [[nodiscard]] std::uint64_t active_sequence(std::uint32_t index) const noexcept;
    // Read-only diagnostics for runner invariants. The live state is the
    // authoritative game; the search root can be a sampled hidden world.
    [[nodiscard]] const GameState* active_state(std::uint32_t index) const noexcept;
    [[nodiscard]] const GameState* active_search_root(std::uint32_t index) const noexcept;
    [[nodiscard]] Action last_scripted_action() const noexcept;
    std::vector<GameState> take_finished_games();

private:
    struct GameSlot;
    struct PendingLeaf;

    void reset_game(std::uint32_t index) noexcept;
    void start_search(GameSlot& game) noexcept;
    void auto_play_treasures(GameSlot& game) noexcept;
    void drive_scripted(GameSlot& game) noexcept;
    void maybe_finish_move(GameSlot& game) noexcept;
    void finish_game(GameSlot& game) noexcept;
    [[nodiscard]] Setup setup_for(std::uint64_t sequence) const noexcept;
    void normalize_policy(const float* logits, const ActionMask& legal, int legal_count, float* out) const noexcept;
    [[nodiscard]] bool resolve_scripted_tree_leaf(GameSlot& game, const MctsPendingLeaf& leaf) noexcept;

    EvalRunnerConfig config_{};
    std::size_t obs_size_ = OBS_SIZE_V1;
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

// Shared by the eval yardstick and training-time Scaffold opponent so their
// rollout settings and forced-treasure fast path stay identical.
[[nodiscard]] MctsConfig make_scaffold_mcts_config(
    std::uint32_t sims_per_move,
    float c_puct,
    std::uint32_t max_tree_nodes,
    bool prune_treasure_plays,
    std::uint8_t determinizations = 2U,
    MctsCPuctSchedule c_puct_schedule = MctsCPuctSchedule::Fixed,
    float c_puct_init = 1.25F,
    float c_puct_base = 19652.0F) noexcept;
[[nodiscard]] std::uint64_t scaffold_rollout_seed(std::uint64_t game_seed) noexcept;
[[nodiscard]] Action eval_scaffold_mcts_action(
    Mcts& search,
    const GameState& state,
    const ActionMask& legal,
    int legal_count) noexcept;
