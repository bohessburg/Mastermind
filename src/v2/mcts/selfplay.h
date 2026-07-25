#pragma once

#include "v2/core/setup.h"
#include "v2/core/types.h"
#include "v2/encode/encoder.h"
#include "v2/mcts/tree.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

// The training curriculum can restrict random kingdoms to a larger candidate
// pool. Keep this fixed-size so individual slot descriptors remain cheap to
// copy across the runner/pybind boundary.
constexpr std::uint8_t MAX_SELFPLAY_KINGDOM_POOL = 32U;
constexpr std::uint8_t SELFPLAY_OPENING_TEMPLATE_COUNT = 7U;
constexpr std::uint8_t SELFPLAY_UNCONSTRAINED_TEMPLATE = 0U;
constexpr std::uint8_t SELFPLAY_OPENING_TELEMETRY_CARD_COUNT = 4U;

enum class SelfPlayKingdomMode : std::uint8_t {
    Fixed,
    Random,
};

enum class SelfPlayValueTarget : std::uint8_t {
    Outcome,
    Margin,
    MarginBlend,
};

// Training-only opponent mode. The policy implementation is shared with
// EvalRunner so scripted data and eval use identical BigMoney/Engine/Random
// behavior. EngineV3 keeps its per-game policy state in ScriptedPool's slot
// storage. None preserves the existing NN-vs-NN self-play path exactly.
enum class SelfPlayScriptedBotKind : std::uint8_t {
    None,
    BigMoney,
    Engine,
    Random,
    Scaffold,
    EngineV3,
    Thinner,
};

// Per-game attributes for a heterogeneous self-play runner.  A non-empty
// SelfPlayConfig::slot_manifest creates exactly one game for every entry;
// unlike the legacy path, completed manifest slots are retired rather than
// immediately reset into a new game stream.  game_index is deliberately
// independent of the slot's storage position so worker-side segment splitting
// or prioritization cannot perturb the game's seed.
struct SelfPlaySlotConfig {
    std::uint64_t game_index = 0U;
    std::uint32_t seat0_model_id = 0U;
    std::uint32_t seat1_model_id = 0U;
    SelfPlayKingdomMode kingdom_mode = SelfPlayKingdomMode::Random;
    // Empty means the complete implemented random-kingdom pool.  The pool is
    // ignored for Fixed mode, exactly as SelfPlayConfig::kingdom_pool is.
    DefId kingdom_pool[MAX_SELFPLAY_KINGDOM_POOL]{};
    std::uint8_t kingdom_pool_count = 0U;
    // Zero inherits SelfPlayConfig::sims_per_move; a positive value is a
    // deep-search override for this slot only.
    std::uint32_t sims_override = 0U;
    SelfPlayScriptedBotKind scripted_bot = SelfPlayScriptedBotKind::None;
    PlayerId scripted_nn_player = 0U;
};

static_assert(std::is_trivially_copyable_v<SelfPlaySlotConfig>);

struct SelfPlayConfig {
    std::uint32_t n_games = 64;
    std::uint32_t sims_per_move = 64;
    float c_puct = 1.25F;
    MctsCPuctSchedule c_puct_schedule = MctsCPuctSchedule::Fixed;
    float c_puct_init = 1.25F;
    float c_puct_base = 19652.0F;
    float dirichlet_alpha = 0.30F;
    float dirichlet_frac = 0.25F;
    std::uint16_t temp_moves = 12;
    std::uint32_t max_batch = 256;
    std::uint64_t seed = 0x545241494EULL;
    ObsVersion obs_version = ObsVersion::V1;
    SelfPlayKingdomMode kingdom_mode = SelfPlayKingdomMode::Random;
    Setup fixed_setup{};
    // Empty means sample random kingdoms from every implemented card. A
    // non-empty pool is used only by Random mode and must contain at least ten
    // distinct implemented kingdom defs.
    DefId kingdom_pool[MAX_SELFPLAY_KINGDOM_POOL]{};
    std::uint8_t kingdom_pool_count = 0;
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
    float margin_blend_alpha = 0.6F;
    SelfPlayScriptedBotKind scripted_bot = SelfPlayScriptedBotKind::None;
    PlayerId scripted_nn_player = 0U;
    bool auto_play_treasures = false;
    bool prune_treasure_plays = false;
    // Training-only search-depth controls. Both remain opt-in so default
    // self-play and gate/eval behavior keep their established full trees.
    bool tree_reuse = false;
    std::uint16_t min_new_sims = 64U;
    std::uint8_t expand_top_k = 0;
    // Opening templates are a self-play-runner concern, deliberately kept
    // out of GameState so search/state frames remain POD.  The Python train
    // loop supplies opening_lambda and weights anew for every generation.
    bool opening_templates_enabled = false;
    float opening_lambda = 0.6F;
    std::int32_t opening_turn_window = 8;
    // Index zero is Unconstrained; one through six are the static opening
    // archetypes implemented in selfplay.cpp.
    float template_weights[SELFPLAY_OPENING_TEMPLATE_COUNT] = {
        0.3F,
        0.11666667F,
        0.11666667F,
        0.11666667F,
        0.11666667F,
        0.11666667F,
        0.11666667F,
    };
    // Empty preserves the historical homogeneous, continuously-reset runner
    // behavior.  A populated manifest makes every GameSlot independently
    // configured and lets a worker keep all segment games live together.
    std::vector<SelfPlaySlotConfig> slot_manifest;
};

[[nodiscard]] bool is_selfplay_implemented_kingdom(DefId def) noexcept;

// Computes the non-tie MarginBlend target after terminal scoring has
// established a score margin. Kept separate from SelfPlayRunner so the exact
// target transform is unit-testable without constructing a terminal game.
[[nodiscard]] float selfplay_margin_blend_value(
    float margin,
    float margin_scale,
    float margin_blend_alpha) noexcept;

// Resolve an opening-template preference for the current root decision. A_END
// means that the template has no legal preference (including after its window
// expires). This narrow, allocation-free helper is public for focused runner
// tests; normal callers should leave action choice to MCTS.
[[nodiscard]] Action selfplay_opening_preferred_action(
    const GameState& state,
    PlayerId player,
    std::uint8_t template_id,
    std::int32_t opening_turn_window,
    const ActionMask& legal) noexcept;

// Mix an already-normalized root prior with a one-hot preferred action. The
// operation is a strict no-op for lambda == 0 or an unavailable preference.
void selfplay_mix_opening_prior(
    float* priors,
    const ActionMask& legal,
    int legal_count,
    Action preferred_action,
    float lambda) noexcept;

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
    // Slot metadata is surfaced with the record because heterogeneous slots
    // complete interleaved. Python uses it for per-kind/per-opponent counters
    // without relying on completion order.
    std::uint64_t game_index = 0;
    std::uint32_t seat0_model_id = 0;
    std::uint32_t seat1_model_id = 0;
    SelfPlayScriptedBotKind scripted_bot = SelfPlayScriptedBotKind::None;
    std::uint32_t sims_override = 0;
    // Opening-template telemetry. Both seats are assigned independently;
    // zero denotes an unconstrained seat.
    std::uint8_t seat_template_ids[MAX_PLAYERS]{};
    std::uint16_t cards_trashed = 0U;
    // Counts actual buy actions while the opening window is active. Keeping
    // this fixed-size makes the integration smoke inspectable without adding
    // a per-move allocation path.
    std::uint16_t opening_buy_counts[ACTION_DEF_COUNT]{};
    // Chapel, Sentry, Moneylender, Village in that order. Only buys made by
    // seats assigned the unconstrained template contribute.
    std::uint16_t unconstrained_buy_counts[SELFPLAY_OPENING_TELEMETRY_CARD_COUNT]{};
};

// Read-only per-slot search counters, primarily useful when profiling batched
// self-play. sims_started/sims_completed count only work begun for the active
// decision; root_visits includes any inherited tree-reuse evidence.
struct SelfPlaySearchStats {
    std::uint32_t sims_target = 0;
    std::uint32_t sims_started = 0;
    std::uint32_t sims_completed = 0;
    std::uint32_t root_visits = 0;
    bool search_active = false;
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
    // Aligned with leaf_observations()/leaf_players().  It identifies the
    // worker-model-table entry that must evaluate this pending leaf.
    [[nodiscard]] const std::uint32_t* leaf_model_ids() const noexcept;
    // Aligned with the pending leaves and stable for the manifest slot's
    // prescribed game.  It lets Python retain per-slot routing metrics while
    // evaluations themselves are grouped by model id.
    [[nodiscard]] const std::uint64_t* leaf_game_indices() const noexcept;
    [[nodiscard]] std::uint32_t leaf_count() const noexcept;
    [[nodiscard]] std::size_t observation_size() const noexcept;
    [[nodiscard]] std::uint64_t games_completed() const noexcept;
    [[nodiscard]] float total_virtual_loss() const noexcept;
    [[nodiscard]] SelfPlaySearchStats search_stats(std::uint32_t index) const noexcept;
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
    [[nodiscard]] Action choose_scripted_action(
        GameSlot& game,
        const GameState& state,
        const ActionMask& legal,
        int legal_count) noexcept;
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
    [[nodiscard]] Setup setup_for(const GameSlot& game) const noexcept;
    [[nodiscard]] float terminal_value_for(const GameState& state, PlayerId player) const noexcept;
    void normalize_policy(
        const float* logits,
        const ActionMask& legal,
        int legal_count,
        bool add_root_noise,
        Action opening_preference,
        float* out,
        Xoshiro256pp& rng) noexcept;
    void reapply_root_noise(GameSlot& game) noexcept;
    [[nodiscard]] bool resolve_scripted_tree_leaf(GameSlot& game, const MctsPendingLeaf& leaf) noexcept;

    SelfPlayConfig config_{};
    std::uint32_t slot_count_ = 0;
    bool manifest_mode_ = false;
    std::size_t obs_size_ = OBS_SIZE_V1;
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
    std::unique_ptr<std::uint32_t[]> leaf_model_ids_;
    std::unique_ptr<std::uint64_t[]> leaf_game_indices_;
    std::unique_ptr<float[]> normalized_policy_;
    std::vector<SelfPlayRecord> finished_;
    std::uint32_t pending_count_ = 0;
    std::uint32_t next_collect_game_ = 0;
    std::uint64_t completed_ = 0;
    std::unique_ptr<ScriptedPool> scripted_pool_{};
};
