#pragma once

#include "v2/core/setup.h"
#include "v2/core/types.h"
#include "v2/encode/encoder.h"
#include "v2/mcts/tree.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

enum class SelfPlayKingdomMode : std::uint8_t {
    Fixed,
    Random,
};

// Training-only opponent mode. The policy implementation is shared with
// EvalRunner so scripted data and eval use identical BigMoney/Engine/Random
// behavior. None preserves the existing NN-vs-NN self-play path exactly.
enum class SelfPlayScriptedBotKind : std::uint8_t {
    None,
    BigMoney,
    Engine,
    Random,
};

struct SelfPlayConfig {
    std::uint32_t n_games = 64;
    std::uint32_t sims_per_move = 64;
    float c_puct = 1.25F;
    float dirichlet_alpha = 0.30F;
    float dirichlet_frac = 0.25F;
    std::uint16_t temp_moves = 12;
    std::uint32_t max_batch = 256;
    std::uint64_t seed = 0x545241494EULL;
    SelfPlayKingdomMode kingdom_mode = SelfPlayKingdomMode::Random;
    Setup fixed_setup{};
    std::uint16_t max_recorded_moves = 512;
    std::uint32_t max_tree_nodes = 4096;
    SelfPlayScriptedBotKind scripted_bot = SelfPlayScriptedBotKind::None;
    PlayerId scripted_nn_player = 0U;
};

struct SelfPlayRecord {
    std::vector<float> observations;
    std::vector<float> policy_targets;
    std::vector<float> values;
    std::vector<PlayerId> players;
    DefId kingdom[MAX_KINGDOM_DEFS]{};
    std::uint8_t kingdom_count = 0;
    std::uint64_t seed = 0;
    std::uint16_t moves = 0;
    // Read-only outcome metadata for Python-side head-to-head evaluation.
    // It is not consumed by self-play search or replay generation.
    PlayerId winner = NONE;
    // None for NN-vs-NN games. Scripted-game records contain only decisions
    // from this NN player, which lets Python derive cheap per-generation
    // head-to-head outcomes without inspecting private engine state.
    PlayerId scripted_nn_player = NONE;
};

class SelfPlayRunner {
public:
    explicit SelfPlayRunner(const SelfPlayConfig& config);
    ~SelfPlayRunner();

    [[nodiscard]] std::uint32_t collect_leaves(std::uint32_t max_batch) noexcept;
    void provide_evaluations(const float* values, const float* policies, std::uint32_t count) noexcept;
    [[nodiscard]] const float* leaf_observations() const noexcept;
    [[nodiscard]] const bool* leaf_legal_masks() const noexcept;
    [[nodiscard]] const PlayerId* leaf_players() const noexcept;
    [[nodiscard]] std::uint32_t leaf_count() const noexcept;
    [[nodiscard]] std::uint64_t games_completed() const noexcept;
    [[nodiscard]] float total_virtual_loss() const noexcept;
    [[nodiscard]] const std::vector<SelfPlayRecord>& finished_games() const noexcept;
    std::vector<SelfPlayRecord> take_finished_games();

private:
    struct GameSlot;
    struct PendingLeaf;

    void reset_game(std::uint32_t index) noexcept;
    void start_search(GameSlot& game) noexcept;
    void drive_scripted(GameSlot& game) noexcept;
    [[nodiscard]] bool game_has_pending(std::uint32_t index) const noexcept;
    void maybe_finish_move(GameSlot& game) noexcept;
    void record_decision(GameSlot& game, const float* policy) noexcept;
    void finish_game(GameSlot& game) noexcept;
    [[nodiscard]] Setup setup_for(std::uint32_t index, std::uint64_t generation) const noexcept;
    [[nodiscard]] float terminal_value_for(const GameState& state, PlayerId player) const noexcept;
    void normalize_policy(
        const float* logits,
        const ActionMask& legal,
        int legal_count,
        bool add_root_noise,
        float* out,
        Xoshiro256pp& rng) noexcept;
    [[nodiscard]] bool resolve_scripted_tree_leaf(GameSlot& game, const MctsPendingLeaf& leaf) noexcept;

    SelfPlayConfig config_{};
    MctsConfig mcts_config_{};
    std::unique_ptr<GameSlot[]> games_;
    std::unique_ptr<PendingLeaf[]> pending_;
    std::unique_ptr<float[]> leaf_obs_;
    std::unique_ptr<bool[]> leaf_masks_;
    std::unique_ptr<PlayerId[]> leaf_players_;
    std::unique_ptr<float[]> normalized_policy_;
    std::vector<SelfPlayRecord> finished_;
    std::uint32_t pending_count_ = 0;
    std::uint32_t next_collect_game_ = 0;
    std::uint64_t completed_ = 0;
};
