#pragma once

#include "v2/core/setup.h"
#include "v2/core/types.h"
#include "v2/encode/encoder.h"
#include "v2/mcts/tree.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

enum class SelfPlayKingdomMode : std::uint8_t {
    Fixed,
    Random,
};

enum class SelfPlayValueTarget : std::uint8_t {
    Outcome,
    Margin,
};

// Training-only opponent mode. The policy implementation is shared with
// EvalRunner so scripted data and eval use identical BigMoney/Engine/Random
// behavior. None preserves the existing NN-vs-NN self-play path exactly.
enum class SelfPlayScriptedBotKind : std::uint8_t {
    None,
    BigMoney,
    Engine,
    Random,
    Scaffold,
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
    std::uint32_t scaffold_sims = 400;
    // Zero keeps the endgame budget for the whole game. A positive value is
    // used before a supply pile empties or Provinces fall to four or fewer.
    std::uint32_t scaffold_sims_opening = 0;
    std::uint8_t scaffold_determinizations = 2;
    // Zero keeps Scaffold synchronous, which is useful for deterministic
    // reference runs. Positive values create that many Scaffold workers.
    std::uint8_t scripted_threads = 2;
    SelfPlayValueTarget value_target = SelfPlayValueTarget::Outcome;
    float margin_scale = 20.0F;
    SelfPlayScriptedBotKind scripted_bot = SelfPlayScriptedBotKind::None;
    PlayerId scripted_nn_player = 0U;
    bool auto_play_treasures = false;
    bool prune_treasure_plays = false;
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
    // Final scores are read-only metadata for diagnostics and Python-side
    // value-target validation; self-play search never consumes them.
    std::int16_t scores[MAX_PLAYERS]{};
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
    // Async Scaffold jobs preserve each seed's trajectory, but cross-slot
    // completion timing can still change the arrival order of these records.
    [[nodiscard]] const std::vector<SelfPlayRecord>& finished_games() const noexcept;
    std::vector<SelfPlayRecord> take_finished_games();

private:
    struct GameSlot;
    struct PendingLeaf;
    struct ScriptedPool;

    void reset_game(std::uint32_t index) noexcept;
    void start_search(GameSlot& game) noexcept;
    void auto_play_treasures(GameSlot& game) noexcept;
    void drive_scripted(GameSlot& game) noexcept;
    void drive_scaffold(GameSlot& game, Mcts& scratch) noexcept;
    // Returns true while an offloaded Scaffold job owns this slot (including
    // the pass on which the job is queued).
    [[nodiscard]] bool offload_scripted_slot(std::uint32_t index) noexcept;
    void drain_scripted_ready() noexcept;
    void run_scripted_job(std::uint32_t index, Mcts& scratch) noexcept;
    [[nodiscard]] std::uint32_t scaffold_sims_for(
        const GameState& state,
        const ActionMask& legal) const noexcept;
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
    MctsConfig scaffold_mcts_config_{};
    // Used only for scripted_threads=0. Async Scaffold jobs use the
    // per-worker scratch trees in scripted_pool_.
    std::optional<Mcts> scaffold_mcts_{};
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
    std::unique_ptr<ScriptedPool> scripted_pool_{};
};
