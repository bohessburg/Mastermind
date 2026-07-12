#include "v2/mcts/selfplay.h"

#include "v2/core/actions.h"
#include "v2/core/game.h"
#include "v2/core/score.h"
#include "v2/core/turns.h"
#include "v2/mcts/eval_runner.h"
#include "v2/mcts/pile_clock.h"

#include <algorithm>
#include <cassert>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {

constexpr DefId IMPLEMENTED_KINGDOMS[] = {
    DEF_CELLAR,
    DEF_CHAPEL,
    DEF_VILLAGE,
    DEF_SMITHY,
    DEF_WORKSHOP,
    DEF_REMODEL,
    DEF_MINE,
    DEF_MERCHANT,
    DEF_MILITIA,
    DEF_WITCH,
    DEF_MOAT,
    DEF_BUREAUCRAT,
    DEF_MARKET,
    DEF_FESTIVAL,
    DEF_LABORATORY,
    DEF_GARDENS,
    DEF_MONEYLENDER,
    DEF_POACHER,
    DEF_VASSAL,
    DEF_HARBINGER,
    DEF_THRONE_ROOM,
    DEF_COUNCIL_ROOM,
    DEF_ARTISAN,
    DEF_BANDIT,
    DEF_LIBRARY,
    DEF_SENTRY,
};

constexpr std::uint8_t IMPLEMENTED_KINGDOM_COUNT =
    static_cast<std::uint8_t>(sizeof(IMPLEMENTED_KINGDOMS) / sizeof(IMPLEMENTED_KINGDOMS[0]));

[[nodiscard]] bool implemented_kingdom(DefId def) noexcept {
    for (const DefId implemented : IMPLEMENTED_KINGDOMS) {
        if (implemented == def) {
            return true;
        }
    }
    return false;
}

enum class ScriptedSlotStatus : std::uint8_t {
    Idle,
    ScriptedPending,
    ScriptedRunning,
    ScriptedReady,
};

[[nodiscard]] PlayerId decision_player(const GameState& state) noexcept {
    if (state.decision.player < state.num_players) {
        return state.decision.player;
    }
    return current_player(state);
}

[[nodiscard]] int pile_count(const Pile& pile) noexcept {
    return pile.mixed_len > 0U ? static_cast<int>(pile.mixed_len) : static_cast<int>(pile.count);
}

[[nodiscard]] DefId pile_top_def(const GameState& state, const Pile& pile) noexcept {
    const Slot slot = pile.mixed_len > 0U ? pile.mixed[pile.mixed_len - 1U] : pile.base;
    return slot < state.num_slots ? state.slot_to_def[slot] : DEF_COPPER;
}

[[nodiscard]] bool scaffold_endgame_armed(
    const GameState& state,
    const ActionMask& legal) noexcept {
    // analyze_pile_clock is the shared, stack-only supply analysis used by
    // the Engine rollout chart. Its empty-pile count is independent of which
    // buys happen to be legal at this decision.
    if (analyze_pile_clock(state, legal).empty_piles > 0) {
        return true;
    }
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        const Pile& pile = state.piles[i];
        if (pile_top_def(state, pile) == DEF_PROVINCE) {
            return pile_count(pile) <= 4;
        }
    }
    return false;
}

[[nodiscard]] bool scripted_mode(const SelfPlayConfig& config) noexcept {
    return config.scripted_bot != SelfPlayScriptedBotKind::None;
}

[[nodiscard]] std::size_t checked_obs_size(ObsVersion version) {
    if (!is_valid_obs_version(version)) {
        throw std::invalid_argument("SelfPlayConfig.obs_version must be V1 or V2");
    }
    return obs_size_for(version);
}

[[nodiscard]] EvalScriptedBotKind eval_scripted_kind(SelfPlayScriptedBotKind kind) noexcept {
    switch (kind) {
    case SelfPlayScriptedBotKind::BigMoney:
        return EvalScriptedBotKind::BigMoney;
    case SelfPlayScriptedBotKind::Engine:
        return EvalScriptedBotKind::Engine;
    case SelfPlayScriptedBotKind::Random:
        return EvalScriptedBotKind::Random;
    case SelfPlayScriptedBotKind::Scaffold:
    case SelfPlayScriptedBotKind::None:
        break;
    }
    return EvalScriptedBotKind::BigMoney;
}

[[nodiscard]] Setup default_fixed_setup() noexcept {
    Setup setup{};
    setup.num_players = 2U;
    constexpr DefId FIXED[] = {
        DEF_VILLAGE,
        DEF_SMITHY,
        DEF_MARKET,
        DEF_FESTIVAL,
        DEF_LABORATORY,
        DEF_CELLAR,
        DEF_CHAPEL,
        DEF_MILITIA,
        DEF_WITCH,
        DEF_MOAT,
    };
    setup.kingdom_count = static_cast<std::uint8_t>(sizeof(FIXED) / sizeof(FIXED[0]));
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        setup.kingdom[i] = FIXED[i];
    }
    return setup;
}

[[nodiscard]] std::uint64_t game_seed(
    const SelfPlayConfig& config,
    std::uint32_t index,
    std::uint64_t generation) noexcept {
    return config.seed
        + (generation * 0x9E37'79B9'7F4A'7C15ULL)
        + (static_cast<std::uint64_t>(index) * 0xD1B5'4A32'D192'ED03ULL);
}

[[nodiscard]] PlayerId winner_for(const GameState& state) noexcept {
    const std::int16_t score0 = score(state, 0U);
    const std::int16_t score1 = score(state, 1U);
    if (score0 > score1) {
        return 0U;
    }
    if (score1 > score0) {
        return 1U;
    }
    return NONE;
}

} // namespace

struct SelfPlayRunner::GameSlot {
    GameState state{};
    Setup setup{};
    Mcts mcts;
    Xoshiro256pp rng{};
    std::vector<float> observations;
    std::vector<float> policy_targets;
    std::vector<PlayerId> players;
    std::uint64_t seed = 0;
    std::uint64_t generation = 0;
    std::uint32_t sims_started = 0;
    std::uint32_t sims_completed = 0;
    std::uint32_t pending = 0;
    std::uint16_t move_index = 0;
    PlayerId nn_player = NONE;
    bool search_active = false;
    // GameSlot is runner bookkeeping (it already owns Mcts and vectors), so
    // the atomic leaves the POD GameState/frame types untouched.
    std::atomic<ScriptedSlotStatus> scripted_status{ScriptedSlotStatus::Idle};

    GameSlot() : mcts(MctsConfig{}) {}
};

struct SelfPlayRunner::PendingLeaf {
    MctsPendingLeaf leaf{};
    std::uint32_t game = 0;
    bool root = false;
};

struct SelfPlayRunner::ScriptedPool {
    ScriptedPool(
        SelfPlayRunner& runner,
        std::uint32_t slot_count,
        std::uint8_t thread_count,
        const MctsConfig& scratch_config)
        : runner_(runner),
          queue_(new std::uint32_t[slot_count]),
          queue_capacity_(slot_count) {
        scratches_.reserve(thread_count);
        for (std::uint32_t i = 0; i < thread_count; ++i) {
            scratches_.emplace_back(scratch_config);
        }

        workers_.reserve(thread_count);
        try {
            for (std::uint32_t i = 0; i < thread_count; ++i) {
                workers_.emplace_back([this, i]() noexcept { worker_loop(i); });
            }
        } catch (...) {
            stop();
            throw;
        }
    }

    ~ScriptedPool() {
        stop();
    }

    [[nodiscard]] bool enqueue(std::uint32_t slot) noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_size_ >= queue_capacity_ || stopping_) {
                return false;
            }
            queue_[queue_tail_] = slot;
            queue_tail_ = (queue_tail_ + 1U) % queue_capacity_;
            ++queue_size_;
        }
        ready_.notify_one();
        return true;
    }

    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    void worker_loop(std::uint32_t worker_index) noexcept {
        while (true) {
            std::uint32_t slot = 0U;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this]() { return stopping_ || queue_size_ != 0U; });
                if (queue_size_ == 0U) {
                    return;
                }
                slot = queue_[queue_head_];
                queue_head_ = (queue_head_ + 1U) % queue_capacity_;
                --queue_size_;
            }
            runner_.run_scripted_job(slot, scratches_[worker_index]);
        }
    }

    SelfPlayRunner& runner_;
    std::vector<Mcts> scratches_;
    std::vector<std::thread> workers_;
    std::unique_ptr<std::uint32_t[]> queue_;
    std::uint32_t queue_capacity_ = 0U;
    std::uint32_t queue_head_ = 0U;
    std::uint32_t queue_tail_ = 0U;
    std::uint32_t queue_size_ = 0U;
    std::mutex mutex_;
    std::condition_variable ready_;
    bool stopping_ = false;
};

bool is_selfplay_implemented_kingdom(DefId def) noexcept {
    return implemented_kingdom(def);
}

SelfPlayRunner::SelfPlayRunner(const SelfPlayConfig& config)
    : config_(config),
      obs_size_(checked_obs_size(config.obs_version)),
      mcts_config_(),
      games_(new GameSlot[std::max(1U, config.n_games)]),
      pending_(new PendingLeaf[std::max(1U, config.max_batch)]),
      leaf_obs_(new float[static_cast<std::size_t>(std::max(1U, config.max_batch)) * obs_size_]),
      leaf_masks_(new bool[static_cast<std::size_t>(std::max(1U, config.max_batch)) * ACTION_SPACE_SIZE]),
      leaf_players_(new PlayerId[std::max(1U, config.max_batch)]),
      normalized_policy_(new float[static_cast<std::size_t>(std::max(1U, config.max_batch)) * ACTION_SPACE_SIZE]) {
    if (config_.n_games == 0U) {
        throw std::invalid_argument("SelfPlayConfig.n_games must be positive");
    }
    if (config_.max_batch == 0U) {
        throw std::invalid_argument("SelfPlayConfig.max_batch must be positive");
    }
    if (config_.sims_per_move == 0U) {
        throw std::invalid_argument("SelfPlayConfig.sims_per_move must be positive");
    }
    if (config_.kingdom_pool_count > MAX_SELFPLAY_KINGDOM_POOL) {
        throw std::invalid_argument("SelfPlayConfig.kingdom_pool has too many cards");
    }
    if (config_.kingdom_pool_count > 0U && config_.kingdom_pool_count < 10U) {
        throw std::invalid_argument("SelfPlayConfig.kingdom_pool must contain at least 10 cards");
    }
    for (std::uint8_t i = 0; i < config_.kingdom_pool_count; ++i) {
        const DefId def = config_.kingdom_pool[i];
        if (!implemented_kingdom(def)) {
            throw std::invalid_argument("SelfPlayConfig.kingdom_pool contains an unimplemented kingdom card");
        }
        for (std::uint8_t previous = 0; previous < i; ++previous) {
            if (config_.kingdom_pool[previous] == def) {
                throw std::invalid_argument("SelfPlayConfig.kingdom_pool contains duplicate cards");
            }
        }
    }
    if (!(config_.margin_scale > 0.0F) || !std::isfinite(config_.margin_scale)) {
        throw std::invalid_argument("SelfPlayConfig.margin_scale must be finite and positive");
    }
    if (config_.scripted_bot == SelfPlayScriptedBotKind::Scaffold && config_.scaffold_sims == 0U) {
        throw std::invalid_argument("SelfPlayConfig.scaffold_sims must be positive for Scaffold");
    }
    if (scripted_mode(config_) && config_.scripted_nn_player >= 2U) {
        throw std::invalid_argument("SelfPlayConfig.scripted_nn_player must be zero or one");
    }
    if (config_.fixed_setup.num_players == 0U) {
        config_.fixed_setup = default_fixed_setup();
    }
    config_.fixed_setup.num_players = 2U;
    mcts_config_.sims_per_move = config_.sims_per_move;
    mcts_config_.c_puct = config_.c_puct;
    mcts_config_.determinizations = 1U;
    mcts_config_.max_tree_nodes = config_.max_tree_nodes == 0U ? 4096U : config_.max_tree_nodes;
    mcts_config_.rollout_policy = MctsRolloutPolicy::External;
    mcts_config_.prune_treasure_plays = config_.prune_treasure_plays;
    mcts_config_.expand_top_k = config_.expand_top_k;
    mcts_config_.tree_reuse = config_.tree_reuse;
    if (mcts_config_.tree_reuse && mcts_config_.determinizations != 1U) {
        // A reused subtree belongs to one root-sampled hidden-information
        // world; K>1 trees aggregate different worlds below the root.
        throw std::invalid_argument("SelfPlayConfig.tree_reuse requires determinizations == 1");
    }
    if (config_.tree_reuse && config_.scripted_bot == SelfPlayScriptedBotKind::Scaffold
        && config_.scaffold_determinizations > 1U) {
        // Keep the self-play/scaffold configuration unambiguous: subtree
        // reuse is valid only for a single determinized world, never a
        // root-level aggregate of K hidden-information samples.
        throw std::invalid_argument(
            "SelfPlayConfig.tree_reuse requires scaffold_determinizations <= 1");
    }

    if (config_.scripted_bot == SelfPlayScriptedBotKind::Scaffold) {
        scaffold_mcts_config_ = make_scaffold_mcts_config(
            config_.scaffold_sims,
            config_.c_puct,
            mcts_config_.max_tree_nodes,
            config_.prune_treasure_plays,
            config_.scaffold_determinizations);
        if (config_.scripted_threads == 0U) {
            scaffold_mcts_.emplace(scaffold_mcts_config_);
        } else {
            scripted_pool_ = std::make_unique<ScriptedPool>(
                *this,
                config_.n_games,
                config_.scripted_threads,
                scaffold_mcts_config_);
        }
    }

    finished_.reserve(config_.n_games);
    for (std::uint32_t i = 0; i < config_.n_games; ++i) {
        games_[i].mcts = Mcts(mcts_config_);
        games_[i].observations.reserve(static_cast<std::size_t>(config_.max_recorded_moves) * obs_size_);
        games_[i].policy_targets.reserve(
            static_cast<std::size_t>(config_.max_recorded_moves) * ACTION_SPACE_SIZE);
        games_[i].players.reserve(config_.max_recorded_moves);
        reset_game(i);
    }
}

SelfPlayRunner::~SelfPlayRunner() {
    if (scripted_pool_) {
        scripted_pool_->stop();
    }
}

std::uint32_t SelfPlayRunner::collect_leaves(std::uint32_t max_batch) noexcept {
    if (pending_count_ != 0U) {
        return pending_count_;
    }
    // Ready jobs never touch shared runner counters. Reap them on the main
    // thread in slot order before building the next NN batch, which makes the
    // finished-record order less timing-dependent without coupling searches.
    drain_scripted_ready();
    const std::uint32_t limit = std::min(
        max_batch == 0U ? config_.max_batch : max_batch,
        config_.max_batch);
    pending_count_ = 0;
    std::uint32_t idle = 0;
    while (pending_count_ < limit && idle < config_.n_games) {
        const std::uint32_t index = next_collect_game_;
        next_collect_game_ = (next_collect_game_ + 1U) % config_.n_games;
        GameSlot& game = games_[index];

        if (game.pending != 0U) {
            ++idle;
            continue;
        }
        if (offload_scripted_slot(index)) {
            ++idle;
            continue;
        }
        drive_scripted(game);
        if (game.pending != 0U) {
            ++idle;
            continue;
        }
        if (scripted_mode(config_) && decision_player(game.state) != game.nn_player) {
            ++idle;
            continue;
        }
        maybe_finish_move(game);
        if (game.pending != 0U) {
            ++idle;
            continue;
        }
        if (scripted_mode(config_) && decision_player(game.state) != game.nn_player) {
            ++idle;
            continue;
        }
        if (!game.search_active) {
            auto_play_treasures(game);
        }
        if (game.pending != 0U) {
            ++idle;
            continue;
        }
        if (scripted_mode(config_) && decision_player(game.state) != game.nn_player) {
            ++idle;
            continue;
        }
        if (!game.search_active) {
            start_search(game);
        }
        if (game.sims_started >= config_.sims_per_move) {
            maybe_finish_move(game);
            ++idle;
            continue;
        }

        MctsPendingLeaf leaf{};
        const bool need_eval = game.mcts.collect_external_leaf(leaf);
        ++game.sims_started;
        if (!need_eval) {
            ++game.sims_completed;
            maybe_finish_move(game);
            idle = 0;
            continue;
        }
        if (scripted_mode(config_) && leaf.player != game.nn_player) {
            if (resolve_scripted_tree_leaf(game, leaf)) {
                ++game.sims_completed;
            }
            maybe_finish_move(game);
            idle = 0;
            continue;
        }

        PendingLeaf& pending = pending_[pending_count_];
        pending.leaf = leaf;
        pending.game = index;
        pending.root = leaf.node == 0U;
        leaf_players_[pending_count_] = leaf.player;
        encode(
            game.mcts.state_for(leaf.state_index),
            leaf.player,
            leaf_obs_.get() + (pending_count_ * obs_size_),
            config_.obs_version);
        bool* mask = leaf_masks_.get() + (pending_count_ * ACTION_SPACE_SIZE);
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            mask[action] = leaf.legal.test(action);
        }
        ++game.pending;
        ++pending_count_;
        idle = 0;
    }
    if (pending_count_ == 0U && scripted_pool_) {
        // A caller may immediately poll again while every slot is in a
        // scripted job. Give those CPU workers a scheduling opportunity
        // without delaying a non-empty NN batch.
        std::this_thread::yield();
    }
    return pending_count_;
}

void SelfPlayRunner::provide_evaluations(const float* values, const float* policies, std::uint32_t count) noexcept {
    const std::uint32_t n = std::min(count, pending_count_);
    for (std::uint32_t i = 0; i < n; ++i) {
        PendingLeaf& pending = pending_[i];
        GameSlot& game = games_[pending.game];
        float* normalized = normalized_policy_.get() + (i * ACTION_SPACE_SIZE);
        normalize_policy(
            policies == nullptr ? nullptr : policies + (i * ACTION_SPACE_SIZE),
            pending.leaf.legal,
            pending.leaf.legal_count,
            pending.root,
            normalized,
            game.rng);
        const float value = values == nullptr ? 0.0F : values[i];
        game.mcts.provide_external_evaluation(pending.leaf, value, normalized, game.rng);
        if (game.pending > 0U) {
            --game.pending;
        }
        ++game.sims_completed;
        maybe_finish_move(game);
    }
    pending_count_ = 0;
}

const float* SelfPlayRunner::leaf_observations() const noexcept {
    return leaf_obs_.get();
}

const bool* SelfPlayRunner::leaf_legal_masks() const noexcept {
    return leaf_masks_.get();
}

const PlayerId* SelfPlayRunner::leaf_players() const noexcept {
    return leaf_players_.get();
}

std::uint32_t SelfPlayRunner::leaf_count() const noexcept {
    return pending_count_;
}

std::size_t SelfPlayRunner::observation_size() const noexcept {
    return obs_size_;
}

std::uint64_t SelfPlayRunner::games_completed() const noexcept {
    return completed_;
}

float SelfPlayRunner::total_virtual_loss() const noexcept {
    float total = 0.0F;
    for (std::uint32_t i = 0; i < config_.n_games; ++i) {
        total += games_[i].mcts.total_virtual_loss();
    }
    return total;
}

const std::vector<SelfPlayRecord>& SelfPlayRunner::finished_games() const noexcept {
    return finished_;
}

std::vector<SelfPlayRecord> SelfPlayRunner::take_finished_games() {
    std::vector<SelfPlayRecord> out = std::move(finished_);
    finished_.clear();
    finished_.reserve(config_.n_games);
    return out;
}

void SelfPlayRunner::reset_game(std::uint32_t index) noexcept {
    GameSlot& game = games_[index];
    game.scripted_status.store(ScriptedSlotStatus::Idle, std::memory_order_relaxed);
    game.mcts.clear_retained_root();
    game.setup = setup_for(index, game.generation);
    game.seed = game_seed(config_, index, game.generation);
    game.state = Game::new_game(game.setup, game.seed);
    game.rng = Xoshiro256pp::seeded(game.seed ^ 0x53E1'F019'0000'0001ULL);
    game.observations.clear();
    game.policy_targets.clear();
    game.players.clear();
    game.sims_started = 0;
    game.sims_completed = 0;
    game.pending = 0;
    game.move_index = 0;
    game.nn_player = scripted_mode(config_) ? config_.scripted_nn_player : NONE;
    game.search_active = false;
}

void SelfPlayRunner::start_search(GameSlot& game) noexcept {
    /*
     * Phase T.1 uses perfect-information self-play at search time. The
     * generated observations still pass through the public encoder, so the
     * model never sees opponent private zones; determinized search remains
     * available for later imperfect-information evaluation.
     */
    assert(game.pending == 0U);
    const PlayerId player = decision_player(game.state);
    const bool reused = config_.tree_reuse
        && game.mcts.adopt_retained_root(game.state, player);
    if (reused) {
        reapply_root_noise(game);
    } else {
        game.mcts.reset(game.state, player);
    }
    game.sims_started = 0;
    game.sims_completed = 0;
    game.pending = 0;
    game.search_active = true;
}

void SelfPlayRunner::reapply_root_noise(GameSlot& game) noexcept {
    if (config_.dirichlet_frac <= 0.0F || game.mcts.node_count() == 0U) {
        return;
    }

    float priors[ACTION_SPACE_SIZE]{};
    ActionMask retained{};
    float sum = 0.0F;
    int count = 0;
    const MctsNode& root = game.mcts.node(0U);
    // A reused interior node is deliberately marked unexpanded so a fresh
    // root evaluation restores every legal child before Dirichlet noise.
    if (!root.expanded) {
        return;
    }
    for (std::uint32_t child_index = root.first_child; child_index != MCTS_NULL;
         child_index = game.mcts.node(child_index).next_sibling) {
        const MctsNode& child = game.mcts.node(child_index);
        if (child.action_from_parent >= ACTION_SPACE_SIZE) {
            continue;
        }
        retained.set(child.action_from_parent);
        priors[child.action_from_parent] = child.prior > 0.0F ? child.prior : 0.0F;
        sum += priors[child.action_from_parent];
        ++count;
    }
    if (count <= 1) {
        return;
    }
    if (sum <= 0.0F) {
        const float uniform = 1.0F / static_cast<float>(count);
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            if (retained.test(action)) {
                priors[action] = uniform;
            }
        }
    } else {
        const float inv_sum = 1.0F / sum;
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            if (retained.test(action)) {
                priors[action] *= inv_sum;
            }
        }
    }
    mcts_add_dirichlet_noise(
        priors,
        retained,
        count,
        config_.dirichlet_alpha,
        config_.dirichlet_frac,
        game.rng);
    game.mcts.set_root_priors(priors);
}

void SelfPlayRunner::auto_play_treasures(GameSlot& game) noexcept {
    if (!config_.auto_play_treasures) {
        return;
    }

    std::uint16_t guard = 0;
    while (game.state.phase != static_cast<std::uint8_t>(Phase::Over) && guard < 512U) {
        ActionMask legal{};
        (void)Game::legal_actions(game.state, legal);
        const Action action = mcts_canonical_treasure_play(
            mcts_filter_treasure_plays(game.state, legal));
        if (action == A_PASS) {
            return;
        }
        const bool done = Game::step(game.state, action);
        game.mcts.clear_retained_root();
        ++guard;
        if (done || game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            finish_game(game);
            return;
        }
    }
}

void SelfPlayRunner::drive_scripted(GameSlot& game) noexcept {
    if (!scripted_mode(config_)) {
        return;
    }

    if (config_.scripted_bot == SelfPlayScriptedBotKind::Scaffold) {
        // Async Scaffold jobs own their slots through scripted_pool_. The
        // synchronous reference path keeps one runner-local scratch tree.
        if (!scaffold_mcts_.has_value()) {
            return;
        }
        drive_scaffold(game, *scaffold_mcts_);
        if (game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            finish_game(game);
        }
        return;
    }

    std::uint16_t guard = 0;
    while (game.state.phase != static_cast<std::uint8_t>(Phase::Over)
           && decision_player(game.state) != game.nn_player
           && guard < 512U) {
        ActionMask legal{};
        const int legal_count = Game::legal_actions(game.state, legal);
        if (legal_count <= 0) {
            break;
        }
        Action action = A_PASS;
        action = eval_scripted_action(
            game.state,
            legal,
            legal_count,
            eval_scripted_kind(config_.scripted_bot),
            game.rng);
        if (!legal.test(action)) {
            action = legal.nth_set(0U);
        }
        const bool done = Game::step(game.state, action);
        game.mcts.clear_retained_root();
        ++guard;
        if (done || game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            finish_game(game);
            break;
        }
    }
}

void SelfPlayRunner::drive_scaffold(GameSlot& game, Mcts& scratch) noexcept {
    std::uint16_t guard = 0;
    while (game.state.phase != static_cast<std::uint8_t>(Phase::Over)
           && decision_player(game.state) != game.nn_player
           && guard < 512U) {
        ActionMask legal{};
        const int legal_count = Game::legal_actions(game.state, legal);
        if (legal_count <= 0) {
            break;
        }

        // Re-seed every search from the game seed exactly as the serialized
        // path did. The scratch's worker identity therefore cannot influence
        // a game's trajectory.
        scratch.set_rollout_seed(scaffold_rollout_seed(game.seed));
        scratch.set_sims_per_move(scaffold_sims_for(game.state, legal));
        Action action = eval_scaffold_mcts_action(scratch, game.state, legal, legal_count);
        if (!legal.test(action)) {
            action = legal.nth_set(0U);
        }
        const bool done = Game::step(game.state, action);
        game.mcts.clear_retained_root();
        ++guard;
        if (done || game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            break;
        }
    }
}

bool SelfPlayRunner::offload_scripted_slot(std::uint32_t index) noexcept {
    if (!scripted_pool_) {
        return false;
    }

    GameSlot& game = games_[index];
    const ScriptedSlotStatus status = game.scripted_status.load(std::memory_order_acquire);
    if (status == ScriptedSlotStatus::ScriptedPending
        || status == ScriptedSlotStatus::ScriptedRunning) {
        return true;
    }
    if (status == ScriptedSlotStatus::ScriptedReady) {
        // The acquire load above pairs with the worker's Ready store, so the
        // main thread now owns all of the job's GameSlot writes.
        game.scripted_status.store(ScriptedSlotStatus::Idle, std::memory_order_release);
        if (game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            finish_game(game);
        }
        return false;
    }

    if (game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
        finish_game(game);
        return false;
    }
    if (decision_player(game.state) == game.nn_player) {
        return false;
    }

    game.scripted_status.store(ScriptedSlotStatus::ScriptedPending, std::memory_order_release);
    if (!scripted_pool_->enqueue(index)) {
        // This should be unreachable: every slot can be queued at most once,
        // and the ring has one entry per slot. Keep the slot retryable rather
        // than letting a failed enqueue strand it in Pending.
        game.scripted_status.store(ScriptedSlotStatus::Idle, std::memory_order_release);
        return false;
    }
    return true;
}

void SelfPlayRunner::drain_scripted_ready() noexcept {
    if (!scripted_pool_) {
        return;
    }
    for (std::uint32_t index = 0; index < config_.n_games; ++index) {
        GameSlot& game = games_[index];
        if (game.scripted_status.load(std::memory_order_acquire) != ScriptedSlotStatus::ScriptedReady) {
            continue;
        }
        game.scripted_status.store(ScriptedSlotStatus::Idle, std::memory_order_release);
        if (game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            finish_game(game);
        }
    }
}

void SelfPlayRunner::run_scripted_job(std::uint32_t index, Mcts& scratch) noexcept {
    if (index >= config_.n_games) {
        return;
    }
    GameSlot& game = games_[index];
    game.scripted_status.store(ScriptedSlotStatus::ScriptedRunning, std::memory_order_release);
    // This is deliberately limited to the slot and the worker's scratch.
    // finish_game(), completed_, and finished_ remain main-thread-only.
    drive_scaffold(game, scratch);
    game.scripted_status.store(ScriptedSlotStatus::ScriptedReady, std::memory_order_release);
}

std::uint32_t SelfPlayRunner::scaffold_sims_for(
    const GameState& state,
    const ActionMask& legal) const noexcept {
    if (config_.scaffold_sims_opening == 0U || scaffold_endgame_armed(state, legal)) {
        return config_.scaffold_sims;
    }
    return config_.scaffold_sims_opening;
}

bool SelfPlayRunner::game_has_pending(std::uint32_t index) const noexcept {
    return index < config_.n_games && games_[index].pending != 0U;
}

void SelfPlayRunner::maybe_finish_move(GameSlot& game) noexcept {
    if (!game.search_active || game.pending != 0U || game.sims_completed < config_.sims_per_move) {
        return;
    }

    float policy[ACTION_SPACE_SIZE]{};
    game.mcts.root_visit_policy(policy, 1.0F);
    record_decision(game, policy);

    const float temperature = game.move_index < config_.temp_moves ? 1.0F : 0.0F;
    Action action = game.mcts.sample_root_action(temperature, game.rng);
    ActionMask legal{};
    const int legal_count = Game::legal_actions(game.state, legal);
    if (legal_count <= 0 || !legal.test(action)) {
        action = legal_count > 0 ? legal.nth_set(0U) : A_PASS;
    }

    const bool done = Game::step(game.state, action);
    if (config_.tree_reuse && !done
        && game.state.phase != static_cast<std::uint8_t>(Phase::Over)) {
        // Store the runner's post-step hash with the selected child. The next
        // search adopts it only when this exact state still survives all
        // intervening engine/scripted work.
        (void)game.mcts.retain_root_child(action, mcts_state_hash(game.state));
    } else {
        game.mcts.clear_retained_root();
    }
    ++game.move_index;
    game.search_active = false;
    if (done || game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
        finish_game(game);
    }
}

void SelfPlayRunner::record_decision(GameSlot& game, const float* policy) noexcept {
    const std::uint16_t recorded = static_cast<std::uint16_t>(game.players.size());
    if (recorded >= config_.max_recorded_moves) {
        return;
    }
    const PlayerId player = decision_player(game.state);
    if (scripted_mode(config_) && player != game.nn_player) {
        return;
    }
    const std::size_t obs_offset = game.observations.size();
    game.observations.resize(obs_offset + obs_size_);
    encode(game.state, player, game.observations.data() + obs_offset, config_.obs_version);

    const std::size_t policy_offset = game.policy_targets.size();
    game.policy_targets.resize(policy_offset + ACTION_SPACE_SIZE);
    std::memcpy(
        game.policy_targets.data() + policy_offset,
        policy,
        sizeof(float) * ACTION_SPACE_SIZE);
    game.players.push_back(player);
}

void SelfPlayRunner::finish_game(GameSlot& game) noexcept {
    SelfPlayRecord record{};
    record.observations = game.observations;
    record.policy_targets = game.policy_targets;
    record.players = game.players;
    record.moves = static_cast<std::uint16_t>(game.players.size());
    record.values.resize(record.moves);
    for (std::uint16_t i = 0; i < record.moves; ++i) {
        record.values[i] = terminal_value_for(game.state, game.players[i]);
    }
    record.seed = game.seed;
    record.winner = winner_for(game.state);
    for (PlayerId player = 0U; player < game.state.num_players; ++player) {
        record.scores[player] = score(game.state, player);
    }
    record.scripted_nn_player = game.nn_player;
    record.kingdom_count = game.setup.kingdom_count;
    for (std::uint8_t i = 0; i < game.setup.kingdom_count; ++i) {
        record.kingdom[i] = game.setup.kingdom[i];
    }
    finished_.push_back(std::move(record));

    ++completed_;
    ++game.generation;
    reset_game(static_cast<std::uint32_t>(&game - games_.get()));
}

bool SelfPlayRunner::resolve_scripted_tree_leaf(GameSlot& game, const MctsPendingLeaf& leaf) noexcept {
    const GameState& leaf_state = game.mcts.state_for(leaf.state_index);
    Action action = A_PASS;
    if (config_.scripted_bot == SelfPlayScriptedBotKind::Scaffold) {
        // Actual moves keep full Scaffold fidelity in drive_scripted; charting
        // its tree leaves as Engine avoids batch stalls (2026-07-11: 36+ min
        // for ~10 Scaffold games in 1,024, versus a ~5 min baseline).
        action = eval_scripted_action(
            leaf_state,
            leaf.legal,
            leaf.legal_count,
            EvalScriptedBotKind::Engine,
            game.rng);
    } else {
        action = eval_scripted_action(
            leaf_state,
            leaf.legal,
            leaf.legal_count,
            eval_scripted_kind(config_.scripted_bot),
            game.rng);
    }
    if (!leaf.legal.test(action)) {
        action = leaf.legal_count > 0 ? leaf.legal.nth_set(0U) : A_PASS;
    }
    float priors[ACTION_SPACE_SIZE]{};
    priors[action] = 1.0F;
    game.mcts.provide_external_evaluation(leaf, 0.0F, priors, game.rng);
    return true;
}

Setup SelfPlayRunner::setup_for(std::uint32_t index, std::uint64_t generation) const noexcept {
    if (config_.kingdom_mode == SelfPlayKingdomMode::Fixed) {
        Setup setup = config_.fixed_setup;
        setup.num_players = 2U;
        return setup;
    }

    Setup setup{};
    setup.num_players = 2U;
    setup.kingdom_count = 10U;
    const DefId* kingdom_pool = IMPLEMENTED_KINGDOMS;
    std::uint8_t kingdom_pool_count = IMPLEMENTED_KINGDOM_COUNT;
    if (config_.kingdom_pool_count > 0U) {
        kingdom_pool = config_.kingdom_pool;
        kingdom_pool_count = config_.kingdom_pool_count;
    }
    DefId defs[MAX_SELFPLAY_KINGDOM_POOL]{};
    for (std::uint8_t i = 0; i < kingdom_pool_count; ++i) {
        defs[i] = kingdom_pool[i];
    }
    Xoshiro256pp rng = Xoshiro256pp::seeded(
        config_.seed
        ^ (static_cast<std::uint64_t>(index + 1U) * 0xBADC'0FFE'1234'5678ULL)
        ^ (generation * 0x9E37'79B9'7F4A'7C15ULL));
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        const std::uint32_t offset = rng.uniform(static_cast<std::uint32_t>(kingdom_pool_count - i));
        const std::uint8_t swap_index = static_cast<std::uint8_t>(i + offset);
        const DefId selected = defs[swap_index];
        defs[swap_index] = defs[i];
        defs[i] = selected;
        setup.kingdom[i] = selected;
    }
    return setup;
}

float SelfPlayRunner::terminal_value_for(const GameState& state, PlayerId player) const noexcept {
    const PlayerId winner = winner_for(state);
    if (config_.value_target == SelfPlayValueTarget::Margin) {
        // Truncated games have no training outcome. Keep record.winner based
        // on the final board for counters and gates, but do not turn that
        // partial score into a value target.
        if (state.truncated != 0U || winner == NONE) {
            return 0.0F;
        }
        const PlayerId opponent = static_cast<PlayerId>(player == 0U ? 1U : 0U);
        const float margin = static_cast<float>(
            static_cast<int>(score(state, player)) - static_cast<int>(score(state, opponent)));
        if (margin == 0.0F) {
            return 0.0F;
        }
        // Sign-preserving hybrid: every win is worth at least +0.5 so narrow
        // wins (a legal, often correct outcome — e.g. a well-executed pile
        // race) never train as near-ties; margin adds gradient WITHIN the
        // win/loss categories so crushes teach more than squeakers.
        const float scale = config_.margin_scale;
        const float sign = margin > 0.0F ? 1.0F : -1.0F;
        const float graded = std::clamp(std::abs(margin), 0.0F, scale) / scale;
        return sign * (0.5F + 0.5F * graded);
    }
    if (winner == NONE) {
        return 0.0F;
    }
    return winner == player ? 1.0F : -1.0F;
}

void SelfPlayRunner::normalize_policy(
    const float* logits,
    const ActionMask& legal,
    int legal_count,
    bool add_root_noise,
    float* out,
    Xoshiro256pp& rng) noexcept {
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        out[action] = 0.0F;
    }
    if (legal_count <= 0) {
        out[A_PASS] = 1.0F;
        return;
    }

    float max_logit = -3.4e38F;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (legal.test(action)) {
            const float logit = logits == nullptr ? 0.0F : logits[action];
            if (logit > max_logit) {
                max_logit = logit;
            }
        }
    }

    double sum = 0.0;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (!legal.test(action)) {
            continue;
        }
        const float logit = logits == nullptr ? 0.0F : logits[action];
        const double value = std::exp(static_cast<double>(logit - max_logit));
        out[action] = static_cast<float>(value);
        sum += value;
    }
    if (sum <= 0.0) {
        const float uniform = 1.0F / static_cast<float>(legal_count);
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            out[action] = legal.test(action) ? uniform : 0.0F;
        }
    } else {
        const float inv_sum = static_cast<float>(1.0 / sum);
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            out[action] *= inv_sum;
        }
    }

    if (!add_root_noise || config_.dirichlet_frac <= 0.0F || legal_count <= 1) {
        return;
    }

    // Roots are always full-width. Noise must cover the same full legal set
    // so policy targets retain the original Dirichlet exploration guarantee.
    mcts_add_dirichlet_noise(
        out,
        legal,
        legal_count,
        config_.dirichlet_alpha,
        config_.dirichlet_frac,
        rng);
}
