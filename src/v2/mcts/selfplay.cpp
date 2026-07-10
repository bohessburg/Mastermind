#include "v2/mcts/selfplay.h"

#include "v2/core/actions.h"
#include "v2/core/game.h"
#include "v2/core/score.h"
#include "v2/core/turns.h"
#include "v2/mcts/eval_runner.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

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

[[nodiscard]] PlayerId decision_player(const GameState& state) noexcept {
    if (state.decision.player < state.num_players) {
        return state.decision.player;
    }
    return current_player(state);
}

[[nodiscard]] bool scripted_mode(const SelfPlayConfig& config) noexcept {
    return config.scripted_bot != SelfPlayScriptedBotKind::None;
}

[[nodiscard]] EvalScriptedBotKind eval_scripted_kind(SelfPlayScriptedBotKind kind) noexcept {
    switch (kind) {
    case SelfPlayScriptedBotKind::BigMoney:
        return EvalScriptedBotKind::BigMoney;
    case SelfPlayScriptedBotKind::Engine:
        return EvalScriptedBotKind::Engine;
    case SelfPlayScriptedBotKind::Random:
        return EvalScriptedBotKind::Random;
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

[[nodiscard]] float uniform_float(Xoshiro256pp& rng) noexcept {
    const double value = static_cast<double>(rng.next() >> 11U) * 0x1.0p-53;
    return static_cast<float>(value <= 0.0 ? 0x1.0p-24 : value);
}

[[nodiscard]] double normal01(Xoshiro256pp& rng) noexcept {
    const double u1 = static_cast<double>(uniform_float(rng));
    const double u2 = static_cast<double>(uniform_float(rng));
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * PI * u2);
}

[[nodiscard]] double gamma_sample(double alpha, Xoshiro256pp& rng) noexcept {
    if (alpha <= 0.0) {
        return 0.0;
    }
    if (alpha < 1.0) {
        const double boosted = gamma_sample(alpha + 1.0, rng);
        return boosted * std::pow(static_cast<double>(uniform_float(rng)), 1.0 / alpha);
    }

    const double d = alpha - (1.0 / 3.0);
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (int attempt = 0; attempt < 32; ++attempt) {
        const double x = normal01(rng);
        const double v_base = 1.0 + (c * x);
        if (v_base <= 0.0) {
            continue;
        }
        const double v = v_base * v_base * v_base;
        const double u = static_cast<double>(uniform_float(rng));
        if (u < 1.0 - (0.0331 * x * x * x * x)) {
            return d * v;
        }
        if (std::log(u) < (0.5 * x * x) + (d * (1.0 - v + std::log(v)))) {
            return d * v;
        }
    }
    return alpha;
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

    GameSlot() : mcts(MctsConfig{}) {}
};

struct SelfPlayRunner::PendingLeaf {
    MctsPendingLeaf leaf{};
    std::uint32_t game = 0;
    bool root = false;
};

SelfPlayRunner::SelfPlayRunner(const SelfPlayConfig& config)
    : config_(config),
      mcts_config_(),
      games_(new GameSlot[std::max(1U, config.n_games)]),
      pending_(new PendingLeaf[std::max(1U, config.max_batch)]),
      leaf_obs_(new float[static_cast<std::size_t>(std::max(1U, config.max_batch)) * OBS_SIZE]),
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

    finished_.reserve(config_.n_games);
    for (std::uint32_t i = 0; i < config_.n_games; ++i) {
        games_[i].mcts = Mcts(mcts_config_);
        games_[i].observations.reserve(static_cast<std::size_t>(config_.max_recorded_moves) * OBS_SIZE);
        games_[i].policy_targets.reserve(
            static_cast<std::size_t>(config_.max_recorded_moves) * ACTION_SPACE_SIZE);
        games_[i].players.reserve(config_.max_recorded_moves);
        reset_game(i);
    }
}

SelfPlayRunner::~SelfPlayRunner() = default;

std::uint32_t SelfPlayRunner::collect_leaves(std::uint32_t max_batch) noexcept {
    if (pending_count_ != 0U) {
        return pending_count_;
    }
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
        encode(game.mcts.state_for(leaf.state_index), leaf.player, leaf_obs_.get() + (pending_count_ * OBS_SIZE));
        bool* mask = leaf_masks_.get() + (pending_count_ * ACTION_SPACE_SIZE);
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            mask[action] = leaf.legal.test(action);
        }
        ++game.pending;
        ++pending_count_;
        idle = 0;
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
        game.mcts.provide_external_evaluation(pending.leaf, value, normalized);
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
    game.mcts.reset(game.state, decision_player(game.state));
    game.sims_started = 0;
    game.sims_completed = 0;
    game.pending = 0;
    game.search_active = true;
}

void SelfPlayRunner::drive_scripted(GameSlot& game) noexcept {
    if (!scripted_mode(config_)) {
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
        Action action = eval_scripted_action(
            game.state,
            legal,
            legal_count,
            eval_scripted_kind(config_.scripted_bot),
            game.rng);
        if (!legal.test(action)) {
            action = legal.nth_set(0U);
        }
        const bool done = Game::step(game.state, action);
        ++guard;
        if (done || game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            finish_game(game);
            break;
        }
    }
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
    game.observations.resize(obs_offset + OBS_SIZE);
    encode(game.state, player, game.observations.data() + obs_offset);

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
    Action action = eval_scripted_action(
        leaf_state,
        leaf.legal,
        leaf.legal_count,
        eval_scripted_kind(config_.scripted_bot),
        game.rng);
    if (!leaf.legal.test(action)) {
        action = leaf.legal_count > 0 ? leaf.legal.nth_set(0U) : A_PASS;
    }
    float priors[ACTION_SPACE_SIZE]{};
    priors[action] = 1.0F;
    game.mcts.provide_external_evaluation(leaf, 0.0F, priors);
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
    DefId defs[IMPLEMENTED_KINGDOM_COUNT]{};
    for (std::uint8_t i = 0; i < IMPLEMENTED_KINGDOM_COUNT; ++i) {
        defs[i] = IMPLEMENTED_KINGDOMS[i];
    }
    Xoshiro256pp rng = Xoshiro256pp::seeded(
        config_.seed
        ^ (static_cast<std::uint64_t>(index + 1U) * 0xBADC'0FFE'1234'5678ULL)
        ^ (generation * 0x9E37'79B9'7F4A'7C15ULL));
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        const std::uint32_t offset = rng.uniform(static_cast<std::uint32_t>(IMPLEMENTED_KINGDOM_COUNT - i));
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
    float noise[ACTION_SPACE_SIZE]{};
    double noise_sum = 0.0;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (legal.test(action)) {
            const double sample = gamma_sample(static_cast<double>(config_.dirichlet_alpha), rng);
            noise[action] = static_cast<float>(sample);
            noise_sum += sample;
        }
    }
    if (noise_sum <= 0.0) {
        return;
    }
    float frac = config_.dirichlet_frac;
    if (frac > 1.0F) {
        frac = 1.0F;
    }
    const float keep = 1.0F - frac;
    const float noise_scale = static_cast<float>(frac / noise_sum);
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (legal.test(action)) {
            out[action] = (out[action] * keep) + (noise[action] * noise_scale);
        }
    }
}
