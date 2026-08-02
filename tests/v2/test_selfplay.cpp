#include "v2/mcts/selfplay.h"

#include "v2/core/actions.h"
#include "v2/core/game.h"
#include "v2/core/interp.h"
#include "v2/core/setup.h"
#include "v2/core/turns.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <thread>
#include <vector>

namespace {

[[nodiscard]] SelfPlayConfig fixed_config(
    std::uint32_t n_games,
    std::uint32_t sims,
    std::uint32_t max_batch,
    std::uint64_t seed) {
    SelfPlayConfig config{};
    config.n_games = n_games;
    config.sims_per_move = sims;
    config.max_batch = max_batch;
    config.seed = seed;
    config.kingdom_mode = SelfPlayKingdomMode::Fixed;
    config.dirichlet_frac = 0.0F;
    config.temp_moves = 4U;
    config.max_recorded_moves = 256U;
    config.max_tree_nodes = 2048U;
    return config;
}

void provide_zero_eval(SelfPlayRunner& runner, std::uint32_t count) {
    std::vector<float> values(count, 0.0F);
    std::vector<float> policies(static_cast<std::size_t>(count) * ACTION_SPACE_SIZE, 0.0F);
    runner.provide_evaluations(values.data(), policies.data(), count);
}

[[nodiscard]] std::uint64_t drive_mock(SelfPlayRunner& runner, std::uint32_t max_batch, std::uint64_t target_leaves) {
    std::uint64_t leaves = 0;
    std::uint32_t idle = 0;
    while (leaves < target_leaves && idle < 1000U) {
        const std::uint32_t count = runner.collect_leaves(max_batch);
        if (count == 0U) {
            ++idle;
            continue;
        }
        idle = 0;
        provide_zero_eval(runner, count);
        leaves += count;
    }
    return leaves;
}

[[nodiscard]] std::vector<SelfPlayRecord> run_until_finished(const SelfPlayConfig& config, std::uint32_t wanted) {
    SelfPlayRunner runner(config);
    std::uint32_t guard = 0;
    while (runner.games_completed() < wanted && guard < 20000U) {
        const std::uint32_t count = runner.collect_leaves(config.max_batch);
        if (count > 0U) {
            provide_zero_eval(runner, count);
        } else {
            // Async Scaffold jobs can temporarily own every slot. A tiny
            // wait keeps this generic helper valid for both sync and async
            // runs without busy-spinning the test process.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ++guard;
    }
    REQUIRE(runner.games_completed() >= wanted);
    return runner.take_finished_games();
}

[[nodiscard]] Setup opening_setup(std::initializer_list<DefId> kingdom) {
    Setup setup{};
    setup.num_players = 2U;
    setup.kingdom_count = static_cast<std::uint8_t>(kingdom.size());
    std::uint8_t index = 0U;
    for (const DefId def : kingdom) {
        setup.kingdom[index] = def;
        ++index;
    }
    return setup;
}

[[nodiscard]] GameState opening_buy_state(
    std::initializer_list<DefId> kingdom,
    std::int16_t coins) {
    GameState state = Game::new_game(opening_setup(kingdom), 0x0E31'1A6ULL);
    state.phase = static_cast<std::uint8_t>(Phase::Buy);
    state.actions = 0U;
    state.buys = 1U;
    state.coins = coins;
    state.decision = PendingDecision{
        current_player(state),
        static_cast<std::uint8_t>(DecisionKind::PhaseBuy),
        0U,
        0U,
        0U,
        static_cast<std::uint8_t>(SelectSemantic::None),
    };
    return state;
}

[[nodiscard]] Pile* opening_pile(GameState& state, DefId def) {
    for (std::uint8_t index = 0U; index < state.num_piles; ++index) {
        Pile& pile = state.piles[index];
        if (pile.mixed_len == 0U && state.slot_to_def[pile.base] == def) {
            return &pile;
        }
    }
    return nullptr;
}

void add_owned_card(GameState& state, PlayerId player, DefId def, std::uint8_t count = 1U) {
    const Slot slot = slot_of(state, def);
    REQUIRE(slot != NONE);
    PlayerState& owner = state.players[player];
    for (std::uint8_t index = 0U; index < count; ++index) {
        REQUIRE(owner.discard.size < MAX_DECK_CARDS);
        owner.discard.cards[owner.discard.size] = slot;
        ++owner.discard.size;
    }
}

void clear_owned_cards(GameState& state, PlayerId player) {
    PlayerState& owner = state.players[player];
    for (Slot slot = 0U; slot < MAX_SLOTS; ++slot) {
        owner.hand[slot] = 0U;
        owner.exile[slot] = 0U;
        owner.tavern[slot] = 0U;
        owner.island_mat[slot] = 0U;
    }
    owner.deck = OrderedZone{};
    owner.discard = OrderedZone{};
    owner.set_aside = OrderedZone{};
    owner.in_play_size = 0U;
}

void set_own_trash_decision(GameState& state, PlayerId player, DefId source, std::uint8_t min_left) {
    const Slot slot = slot_of(state, source);
    REQUIRE(slot != NONE);
    PlayerState& owner = state.players[player];
    REQUIRE(owner.in_play_size < MAX_IN_PLAY);
    owner.in_play[owner.in_play_size] = InPlayEntry{slot, slot, 0U};
    ++owner.in_play_size;
    state.decision = PendingDecision{
        player,
        static_cast<std::uint8_t>(DecisionKind::Choose),
        source,
        min_left,
        4U,
        static_cast<std::uint8_t>(SelectSemantic::Trash),
    };
    state.effect_depth = 1U;
    state.effect_stack[0] = EffectFrame{};
    state.effect_stack[0].source = source;
    state.effect_stack[0].player = player;
}

void require_same_records(const std::vector<SelfPlayRecord>& lhs, const std::vector<SelfPlayRecord>& rhs) {
    REQUIRE(lhs.size() == rhs.size());
    for (std::size_t index = 0U; index < lhs.size(); ++index) {
        REQUIRE(lhs[index].seed == rhs[index].seed);
        REQUIRE(lhs[index].moves == rhs[index].moves);
        REQUIRE(lhs[index].turn_counter == rhs[index].turn_counter);
        REQUIRE(lhs[index].truncated == rhs[index].truncated);
        REQUIRE(lhs[index].winner == rhs[index].winner);
        REQUIRE(lhs[index].kingdom_count == rhs[index].kingdom_count);
        REQUIRE(std::memcmp(lhs[index].kingdom, rhs[index].kingdom, sizeof(lhs[index].kingdom)) == 0);
        REQUIRE(std::memcmp(lhs[index].scores, rhs[index].scores, sizeof(lhs[index].scores)) == 0);
        REQUIRE(lhs[index].scripted_nn_player == rhs[index].scripted_nn_player);
        REQUIRE(lhs[index].game_index == rhs[index].game_index);
        REQUIRE(lhs[index].seat0_model_id == rhs[index].seat0_model_id);
        REQUIRE(lhs[index].seat1_model_id == rhs[index].seat1_model_id);
        REQUIRE(lhs[index].scripted_bot == rhs[index].scripted_bot);
        REQUIRE(lhs[index].sims_override == rhs[index].sims_override);
        REQUIRE(lhs[index].observations == rhs[index].observations);
        REQUIRE(lhs[index].policy_targets == rhs[index].policy_targets);
        REQUIRE(lhs[index].sampled_actions == rhs[index].sampled_actions);
        REQUIRE(lhs[index].legal_mask_words == rhs[index].legal_mask_words);
        REQUIRE(lhs[index].players == rhs[index].players);
        REQUIRE(lhs[index].values == rhs[index].values);
        REQUIRE(lhs[index].margins == rhs[index].margins);
        REQUIRE(lhs[index].cards_trashed == rhs[index].cards_trashed);
        REQUIRE(std::memcmp(
                    lhs[index].seat_template_ids,
                    rhs[index].seat_template_ids,
                    sizeof(lhs[index].seat_template_ids)) == 0);
        REQUIRE(std::memcmp(
                    lhs[index].opening_buy_counts,
                    rhs[index].opening_buy_counts,
                    sizeof(lhs[index].opening_buy_counts)) == 0);
        REQUIRE(std::memcmp(
                    lhs[index].unconstrained_buy_counts,
                    rhs[index].unconstrained_buy_counts,
                    sizeof(lhs[index].unconstrained_buy_counts)) == 0);
    }
}

void require_temperature_sampling_mode(float temperature, bool expect_argmax) {
    GameState state = opening_buy_state({DEF_CHAPEL}, 3);
    MctsConfig config{};
    config.rollout_policy = MctsRolloutPolicy::External;
    config.determinizations = 1U;
    config.max_tree_nodes = 256U;
    Mcts search(config);
    search.reset(state, 0U);
    MctsPendingLeaf leaf{};
    REQUIRE(search.collect_external_leaf(leaf));
    float priors[ACTION_SPACE_SIZE]{};
    for (Action action = 0U; action < ACTION_SPACE_SIZE; ++action) {
        priors[action] = leaf.legal.test(action) ? 1.0F : 0.0F;
    }
    Xoshiro256pp expansion_rng = Xoshiro256pp::seeded(0x7E4D'0001ULL);
    search.provide_external_evaluation(leaf, 0.0F, priors, expansion_rng);

    Xoshiro256pp argmax_rng = Xoshiro256pp::seeded(0x7E4D'0002ULL);
    const Action argmax = search.sample_root_action(0.0F, argmax_rng);
    if (expect_argmax) {
        Xoshiro256pp choice_rng = Xoshiro256pp::seeded(0x7E4D'0003ULL);
        REQUIRE(search.sample_root_action(temperature, choice_rng) == argmax);
        return;
    }

    bool sampled_non_argmax = false;
    for (std::uint64_t seed = 1U; seed <= 64U; ++seed) {
        Xoshiro256pp choice_rng = Xoshiro256pp::seeded(seed);
        sampled_non_argmax = sampled_non_argmax
            || search.sample_root_action(temperature, choice_rng) != argmax;
    }
    REQUIRE(sampled_non_argmax);
}

} // namespace

TEST_CASE("v2 selfplay virtual loss clears after evaluations", "[v2][selfplay]") {
    SelfPlayRunner runner(fixed_config(4U, 8U, 8U, 0x5E1F'0001ULL));

    const std::uint32_t count = runner.collect_leaves(8U);
    REQUIRE(count > 0U);
    REQUIRE(runner.total_virtual_loss() > 0.0F);

    provide_zero_eval(runner, count);
    REQUIRE(runner.total_virtual_loss() == Catch::Approx(0.0F));
}

TEST_CASE("v2 selfplay runner uses the selected observation-version stride", "[v2][selfplay][encode]") {
    SelfPlayConfig config = fixed_config(2U, 4U, 4U, 0x5E1F'0101ULL);
    config.obs_version = ObsVersion::V2;
    SelfPlayRunner runner(config);

    const std::uint32_t count = runner.collect_leaves(config.max_batch);
    REQUIRE(count > 0U);
    REQUIRE(runner.observation_size() == OBS_SIZE_V2);
    const float* observations = runner.leaf_observations();
    for (std::uint32_t index = 0; index < count; ++index) {
        const float* observation = observations + (static_cast<std::size_t>(index) * OBS_SIZE_V2);
        CHECK(observation[OBS_V2_META_OFFSET] == static_cast<float>(ObsVersion::V2));
        CHECK(observation[OBS_V2_META_OFFSET + 1U] == static_cast<float>(OBS_SIZE_V2));
    }
    provide_zero_eval(runner, count);
}

TEST_CASE("v2 selfplay runner accepts the v3 observation stride", "[v2][selfplay][encode]") {
    SelfPlayConfig config = fixed_config(2U, 4U, 4U, 0x5E1F'0102ULL);
    config.obs_version = ObsVersion::V3;
    SelfPlayRunner runner(config);

    const std::uint32_t count = runner.collect_leaves(config.max_batch);
    REQUIRE(count > 0U);
    REQUIRE(runner.observation_size() == OBS_SIZE_V3);
    const float* observations = runner.leaf_observations();
    for (std::uint32_t index = 0; index < count; ++index) {
        const float* observation = observations + (static_cast<std::size_t>(index) * OBS_SIZE_V3);
        CHECK(observation[OBS_V2_META_OFFSET] == static_cast<float>(ObsVersion::V3));
        CHECK(observation[OBS_V2_META_OFFSET + 1U] == static_cast<float>(OBS_SIZE_V3));
    }
    provide_zero_eval(runner, count);
}

TEST_CASE("v2 external MCTS policy targets are legal-action masked", "[v2][selfplay][mcts]") {
    GameState state = Game::new_game(Setup{}, 0x5E1F'0002ULL);
    MctsConfig config{};
    config.rollout_policy = MctsRolloutPolicy::External;
    config.determinizations = 1U;
    config.max_tree_nodes = 1024U;
    Mcts search(config);
    search.reset(state, 0U);

    MctsPendingLeaf leaf{};
    REQUIRE(search.collect_external_leaf(leaf));
    std::vector<float> priors(ACTION_SPACE_SIZE, 1.0F);
    search.provide_external_evaluation(leaf, 0.0F, priors.data());
    REQUIRE(search.total_virtual_loss() == Catch::Approx(0.0F));

    float target[ACTION_SPACE_SIZE]{};
    search.root_visit_policy(target, 1.0F);
    float sum = 0.0F;
    bool illegal_zero = true;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (!leaf.legal.test(action)) {
            illegal_zero = illegal_zero && target[action] == 0.0F;
        }
        sum += target[action];
    }
    REQUIRE(illegal_zero);
    REQUIRE(sum == Catch::Approx(1.0F).margin(0.0001F));
}

TEST_CASE("v2 selfplay temperature schedule decouples seats and decision kinds", "[v2][selfplay][temperature]") {
    SelfPlayConfig config{};
    REQUIRE(config.temp_mode == SelfPlayTempMode::Legacy);
    REQUIRE(config.temp_moves == 20U);
    REQUIRE(selfplay_temperature_for(
                config, DecisionKind::Choose, 19U, 999U, 999U)
            == 1.0F);
    REQUIRE(selfplay_temperature_for(
                config, DecisionKind::Choose, 20U, 0U, 1U)
            == 0.0F);

    config.temp_mode = SelfPlayTempMode::PerSeatBuy;
    config.temp_buy_turns = 2U;
    config.temp_action_plies = 2U;
    config.temp_effect_plies = 2U;
    config.temp_final = 0.0F;

    // turns.cpp increments after cleanup: 0/1 are the players' first turns.
    REQUIRE(selfplay_seat_turn_number(0U) == 1U);
    REQUIRE(selfplay_seat_turn_number(1U) == 1U);
    REQUIRE(selfplay_seat_turn_number(2U) == 2U);
    REQUIRE(selfplay_seat_turn_number(3U) == 2U);

    const float buy_early = selfplay_temperature_for(
        config, DecisionKind::PhaseBuy, 999U, 99U, 2U);
    const float buy_late = selfplay_temperature_for(
        config, DecisionKind::PhaseBuy, 0U, 0U, 3U);
    const float action_seat_one_early = selfplay_temperature_for(
        config, DecisionKind::PhaseAction, 0U, 1U, 99U);
    const float action_seat_zero_late = selfplay_temperature_for(
        config, DecisionKind::PhaseAction, 0U, 2U, 99U);
    const float effect_early = selfplay_temperature_for(
        config, DecisionKind::ChooseGain, 0U, 1U, 99U);
    const float effect_late = selfplay_temperature_for(
        config, DecisionKind::ReactWindow, 0U, 2U, 99U);

    REQUIRE(buy_early == 1.0F);
    REQUIRE(buy_late == 0.0F);
    REQUIRE(action_seat_one_early == 1.0F);
    REQUIRE(action_seat_zero_late == 0.0F);
    REQUIRE(effect_early == 1.0F);
    REQUIRE(effect_late == 0.0F);

    // A positive temperature samples visit mass; zero is the root argmax.
    require_temperature_sampling_mode(buy_early, false);
    require_temperature_sampling_mode(buy_late, true);
    require_temperature_sampling_mode(action_seat_one_early, false);
    require_temperature_sampling_mode(action_seat_zero_late, true);
    require_temperature_sampling_mode(effect_early, false);
    require_temperature_sampling_mode(effect_late, true);

    config.temp_final = 0.25F;
    REQUIRE(selfplay_temperature_for(
                config, DecisionKind::ChooseOption, 0U, 2U, 99U)
            == Catch::Approx(0.25F));
}

TEST_CASE("v2 legacy temperature mode preserves seeded action streams", "[v2][selfplay][temperature]") {
    SelfPlayConfig historical = fixed_config(1U, 8U, 8U, 0x5E1F'7A20ULL);
    historical.auto_play_treasures = true;
    historical.prune_treasure_plays = true;

    SelfPlayConfig mode_off = historical;
    // These fields must be inert while legacy mode retains the global clock.
    mode_off.temp_mode = SelfPlayTempMode::Legacy;
    mode_off.temp_buy_turns = 0U;
    mode_off.temp_action_plies = 0U;
    mode_off.temp_effect_plies = 0U;
    mode_off.temp_final = 0.0F;

    const std::vector<SelfPlayRecord> expected = run_until_finished(historical, 1U);
    const std::vector<SelfPlayRecord> actual = run_until_finished(mode_off, 1U);
    REQUIRE_FALSE(expected.empty());
    REQUIRE_FALSE(expected.front().sampled_actions.empty());
    require_same_records(expected, actual);
}

TEST_CASE("v2 opening templates resolve legal price-band preferences", "[v2][selfplay][opening]") {
    // A kingdom without Chapel must fall through T1's first band to Silver.
    GameState missing_chapel = opening_buy_state({DEF_VILLAGE}, 3);
    ActionMask legal{};
    REQUIRE(Game::legal_actions(missing_chapel, legal) > 0);
    REQUIRE(selfplay_opening_preferred_action(missing_chapel, 0U, 1U, 8, legal) == buy_action(DEF_SILVER));

    // An empty Chapel pile has the same legal-mask fallthrough behavior.
    GameState empty_chapel = opening_buy_state({DEF_CHAPEL}, 3);
    Pile* chapel_pile = opening_pile(empty_chapel, DEF_CHAPEL);
    REQUIRE(chapel_pile != nullptr);
    chapel_pile->count = 0U;
    REQUIRE(Game::legal_actions(empty_chapel, legal) > 0);
    REQUIRE(selfplay_opening_preferred_action(empty_chapel, 0U, 1U, 8, legal) == buy_action(DEF_SILVER));

    // T1's Chapel cap falls through within its active price band.
    GameState capped_chapel = opening_buy_state({DEF_CHAPEL}, 3);
    add_owned_card(capped_chapel, 0U, DEF_CHAPEL);
    REQUIRE(Game::legal_actions(capped_chapel, legal) > 0);
    REQUIRE(selfplay_opening_preferred_action(capped_chapel, 0U, 1U, 8, legal) == buy_action(DEF_SILVER));

    // T2 owns Villages only while it does not outnumber the terminal count.
    GameState t2_ratio = opening_buy_state({DEF_VILLAGE, DEF_SMITHY}, 3);
    REQUIRE(Game::legal_actions(t2_ratio, legal) > 0);
    REQUIRE(selfplay_opening_preferred_action(t2_ratio, 0U, 2U, 8, legal) == buy_action(DEF_VILLAGE));
    add_owned_card(t2_ratio, 0U, DEF_VILLAGE);
    REQUIRE(selfplay_opening_preferred_action(t2_ratio, 0U, 2U, 8, legal) == buy_action(DEF_SILVER));

    t2_ratio.coins = 4;
    REQUIRE(Game::legal_actions(t2_ratio, legal) > 0);
    REQUIRE(selfplay_opening_preferred_action(t2_ratio, 0U, 2U, 8, legal) == buy_action(DEF_SMITHY));
    add_owned_card(t2_ratio, 0U, DEF_SMITHY, 3U);
    REQUIRE(selfplay_opening_preferred_action(t2_ratio, 0U, 2U, 8, legal) == buy_action(DEF_SILVER));

    GameState province = opening_buy_state({DEF_CHAPEL}, 8);
    REQUIRE(Game::legal_actions(province, legal) > 0);
    for (std::uint8_t template_id = 1U; template_id < SELFPLAY_OPENING_TEMPLATE_COUNT; ++template_id) {
        REQUIRE(selfplay_opening_preferred_action(province, 0U, template_id, 8, legal) == buy_action(DEF_PROVINCE));
    }
}

TEST_CASE("v2 opening prior mixing is legal masked and normalized", "[v2][selfplay][opening]") {
    ActionMask legal{};
    legal.set(A_PASS);
    legal.set(buy_action(DEF_CHAPEL));
    legal.set(buy_action(DEF_SILVER));
    constexpr int LEGAL_COUNT = 3;
    float priors[ACTION_SPACE_SIZE]{};
    priors[A_PASS] = 0.2F;
    priors[buy_action(DEF_CHAPEL)] = 0.3F;
    priors[buy_action(DEF_SILVER)] = 0.5F;
    priors[buy_action(DEF_GOLD)] = 99.0F;

    float no_op[ACTION_SPACE_SIZE]{};
    std::memcpy(no_op, priors, sizeof(priors));
    selfplay_mix_opening_prior(no_op, legal, LEGAL_COUNT, buy_action(DEF_CHAPEL), 0.0F);
    REQUIRE(std::memcmp(no_op, priors, sizeof(priors)) == 0);

    float concentrated[ACTION_SPACE_SIZE]{};
    std::memcpy(concentrated, priors, sizeof(priors));
    selfplay_mix_opening_prior(concentrated, legal, LEGAL_COUNT, buy_action(DEF_CHAPEL), 1.0F);
    for (Action action = 0U; action < ACTION_SPACE_SIZE; ++action) {
        const float expected = action == buy_action(DEF_CHAPEL) ? 1.0F : 0.0F;
        REQUIRE(concentrated[action] == Catch::Approx(expected));
    }

    float mixed[ACTION_SPACE_SIZE]{};
    std::memcpy(mixed, priors, sizeof(priors));
    selfplay_mix_opening_prior(mixed, legal, LEGAL_COUNT, buy_action(DEF_CHAPEL), 0.5F);
    REQUIRE(mixed[A_PASS] == Catch::Approx(0.1F));
    REQUIRE(mixed[buy_action(DEF_CHAPEL)] == Catch::Approx(0.65F));
    REQUIRE(mixed[buy_action(DEF_SILVER)] == Catch::Approx(0.25F));
    REQUIRE(mixed[buy_action(DEF_GOLD)] == 0.0F);
    float sum = 0.0F;
    for (Action action = 0U; action < ACTION_SPACE_SIZE; ++action) {
        sum += mixed[action];
    }
    REQUIRE(sum == Catch::Approx(1.0F));
}

TEST_CASE("v2 opening trash guard protects the treasure floor", "[v2][selfplay][opening]") {
    GameState state = opening_buy_state({DEF_CHAPEL}, 0);
    clear_owned_cards(state, 0U);
    set_own_trash_decision(state, 0U, DEF_CHAPEL, 0U);
    ActionMask legal{};
    legal.set(A_PASS);
    legal.set(select_action(DEF_CURSE));
    legal.set(select_action(DEF_ESTATE));
    legal.set(select_action(DEF_COPPER));
    legal.set(select_action(DEF_SILVER));
    REQUIRE(selfplay_opening_preferred_action(state, 0U, 1U, 8, legal) == select_action(DEF_CURSE));

    legal = ActionMask{};
    legal.set(A_PASS);
    legal.set(select_action(DEF_ESTATE));
    legal.set(select_action(DEF_COPPER));
    REQUIRE(selfplay_opening_preferred_action(state, 0U, 1U, 8, legal) == select_action(DEF_ESTATE));

    // Copper becomes eligible only above three total treasures.
    add_owned_card(state, 0U, DEF_COPPER, 4U);
    legal = ActionMask{};
    legal.set(A_PASS);
    legal.set(select_action(DEF_COPPER));
    legal.set(select_action(DEF_SILVER));
    REQUIRE(selfplay_opening_preferred_action(state, 0U, 1U, 8, legal) == select_action(DEF_COPPER));

    clear_owned_cards(state, 0U);
    set_own_trash_decision(state, 0U, DEF_CHAPEL, 0U);
    add_owned_card(state, 0U, DEF_COPPER, 3U);
    legal = ActionMask{};
    legal.set(A_PASS);
    legal.set(select_action(DEF_COPPER));
    legal.set(select_action(DEF_SILVER));
    REQUIRE(selfplay_opening_preferred_action(state, 0U, 1U, 8, legal) == A_PASS);

    legal = ActionMask{};
    legal.set(A_PASS);
    legal.set(select_action(DEF_SILVER));
    REQUIRE(selfplay_opening_preferred_action(state, 0U, 1U, 8, legal) == A_PASS);
    state.decision.min_left = 1U;
    REQUIRE(selfplay_opening_preferred_action(state, 0U, 1U, 8, legal) == A_END);

    state.decision.min_left = 0U;
    state.turn_counter = 9U;
    REQUIRE(selfplay_opening_preferred_action(state, 0U, 1U, 8, legal) == A_END);

    // A same-named card in the target's play area does not make an opponent's
    // effect eligible for the guard.
    state.turn_counter = 0U;
    state.effect_stack[0].flags = FRAME_ATTACK;
    REQUIRE(selfplay_opening_preferred_action(state, 0U, 1U, 8, legal) == A_END);
}

TEST_CASE("v2 selfplay finished records have normalized policies and terminal values", "[v2][selfplay]") {
    const std::vector<SelfPlayRecord> records = run_until_finished(
        fixed_config(2U, 16U, 16U, 0x5E1F'0003ULL),
        1U);
    REQUIRE_FALSE(records.empty());

    const SelfPlayRecord& record = records.front();
    REQUIRE(record.moves > 0U);
    REQUIRE(record.observations.size() == static_cast<std::size_t>(record.moves) * OBS_SIZE);
    REQUIRE(record.policy_targets.size() == static_cast<std::size_t>(record.moves) * ACTION_SPACE_SIZE);
    REQUIRE(record.legal_mask_words.size() == static_cast<std::size_t>(record.moves) * ACTION_MASK_WORDS);
    REQUIRE(record.values.size() == record.moves);
    REQUIRE(record.margins.size() == record.moves);
    for (std::uint16_t move = 0; move < record.moves; ++move) {
        float sum = 0.0F;
        bool nonnegative = true;
        const float* policy = record.policy_targets.data() + (static_cast<std::size_t>(move) * ACTION_SPACE_SIZE);
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            nonnegative = nonnegative && policy[action] >= 0.0F;
            sum += policy[action];
        }
        REQUIRE(nonnegative);
        REQUIRE(sum == Catch::Approx(1.0F).margin(0.0001F));
        REQUIRE((record.values[move] == -1.0F || record.values[move] == 0.0F || record.values[move] == 1.0F));
    }
}

TEST_CASE("v2 selfplay turn cap truncates records and zeroes value targets", "[v2][selfplay]") {
    SelfPlayConfig config = fixed_config(1U, 2U, 8U, 0x5E1F'CA20ULL);
    config.selfplay_max_turns = 20U;
    config.auto_play_treasures = true;
    config.prune_treasure_plays = true;

    const std::vector<SelfPlayRecord> records = run_until_finished(config, 1U);
    REQUIRE(records.size() == 1U);

    const SelfPlayRecord& record = records.front();
    REQUIRE(record.turn_counter == 20U);
    REQUIRE(record.truncated);
    REQUIRE_FALSE(record.values.empty());
    for (const float value : record.values) {
        REQUIRE(value == 0.0F);
    }
}

TEST_CASE("v2 selfplay validates the training turn cap", "[v2][selfplay]") {
    SelfPlayConfig config = fixed_config(1U, 2U, 8U, 0x5E1F'CA21ULL);
    config.selfplay_max_turns = 0U;
    REQUIRE_NOTHROW(SelfPlayRunner(config));

    config.selfplay_max_turns = 10U;
    REQUIRE_THROWS_AS(SelfPlayRunner(config), std::invalid_argument);

    config.selfplay_max_turns = 201U;
    REQUIRE_THROWS_AS(SelfPlayRunner(config), std::invalid_argument);
}

TEST_CASE("v2 selfplay records final margins from each recorded seat perspective", "[v2][selfplay][margin]") {
    const std::vector<SelfPlayRecord> records = run_until_finished(
        fixed_config(2U, 8U, 16U, 0x5E1F'4D41'5247ULL),
        1U);
    REQUIRE_FALSE(records.empty());

    const SelfPlayRecord& record = records.front();
    bool saw_seat[MAX_PLAYERS]{};
    REQUIRE(record.margins.size() == record.moves);
    for (std::uint16_t move = 0U; move < record.moves; ++move) {
        const PlayerId player = record.players[move];
        REQUIRE(player < MAX_PLAYERS);
        const PlayerId opponent = static_cast<PlayerId>(player == 0U ? 1U : 0U);
        const int expected = std::clamp(
            static_cast<int>(record.scores[player]) - static_cast<int>(record.scores[opponent]),
            -127,
            127);
        REQUIRE(record.margins[move] == expected);
        saw_seat[player] = true;
    }
    REQUIRE(saw_seat[0U]);
    REQUIRE(saw_seat[1U]);
}

TEST_CASE("v2 selfplay records legal zero-visit root actions", "[v2][selfplay]") {
    SelfPlayConfig config = fixed_config(1U, 2U, 4U, 0x5E1F'0A11ULL);
    config.auto_play_treasures = true;
    config.prune_treasure_plays = true;
    // root_visit_policy uses its visit-weight form while this remains positive,
    // so a zero target here means the legal child had zero root visits.
    config.temp_moves = 255U;
    const std::vector<SelfPlayRecord> records = run_until_finished(config, 1U);

    bool found_legal_zero_visit = false;
    for (const SelfPlayRecord& record : records) {
        REQUIRE(record.legal_mask_words.size()
                == static_cast<std::size_t>(record.moves) * ACTION_MASK_WORDS);
        for (std::uint16_t move = 0U; move < record.moves; ++move) {
            std::uint32_t legal_count = 0U;
            std::uint32_t nonzero_policy_count = 0U;
            const float* policy = record.policy_targets.data()
                + (static_cast<std::size_t>(move) * ACTION_SPACE_SIZE);
            for (Action action = 0U; action < ACTION_SPACE_SIZE; ++action) {
                const bool legal = (record.legal_mask_words[
                    (static_cast<std::size_t>(move) * ACTION_MASK_WORDS) + (action >> 6U)]
                    & (std::uint64_t{1} << (action & 63U))) != 0U;
                legal_count += legal ? 1U : 0U;
                nonzero_policy_count += policy[action] > 0.0F ? 1U : 0U;
            }
            if (legal_count > nonzero_policy_count) {
                found_legal_zero_visit = true;
                break;
            }
        }
        if (found_legal_zero_visit) {
            break;
        }
    }
    REQUIRE(found_legal_zero_visit);
}

TEST_CASE("v2 selfplay MarginBlend terminal target tempers margin influence", "[v2][selfplay][value]") {
    constexpr float SCALE = 20.0F;
    constexpr float ALPHA = 0.6F;

    // v = sign(margin) * (alpha + (1 - alpha) *
    //     (0.5 + 0.5 * min(abs(margin), scale) / scale)); ties are zero.
    REQUIRE(selfplay_margin_blend_value(1.0F, SCALE, ALPHA) == Catch::Approx(0.81F));
    REQUIRE(selfplay_margin_blend_value(5.0F, SCALE, ALPHA) == Catch::Approx(0.85F));
    REQUIRE(selfplay_margin_blend_value(10.0F, SCALE, ALPHA) == Catch::Approx(0.9F));
    REQUIRE(selfplay_margin_blend_value(20.0F, SCALE, ALPHA) == Catch::Approx(1.0F));
    REQUIRE(selfplay_margin_blend_value(35.0F, SCALE, ALPHA) == Catch::Approx(1.0F));
    REQUIRE(selfplay_margin_blend_value(-1.0F, SCALE, ALPHA) == Catch::Approx(-0.81F));
    REQUIRE(selfplay_margin_blend_value(0.0F, SCALE, ALPHA) == 0.0F);

    // Endpoint alpha values exactly recover the established target behavior.
    for (const float margin : {-35.0F, -5.0F, 5.0F, 35.0F}) {
        const float sign = margin > 0.0F ? 1.0F : -1.0F;
        const float graded = std::min(std::abs(margin), SCALE) / SCALE;
        const float existing_margin = sign * (0.5F + 0.5F * graded);
        REQUIRE(selfplay_margin_blend_value(margin, SCALE, 0.0F) == existing_margin);
        REQUIRE(selfplay_margin_blend_value(margin, SCALE, 1.0F) == sign);
    }
}

TEST_CASE("v2 selfplay is deterministic with a fixed mock evaluator", "[v2][selfplay]") {
    const SelfPlayConfig config = fixed_config(4U, 16U, 16U, 0x5E1F'0004ULL);
    const std::vector<SelfPlayRecord> a = run_until_finished(config, 2U);
    const std::vector<SelfPlayRecord> b = run_until_finished(config, 2U);
    REQUIRE(a.size() == b.size());
    REQUIRE(a.size() >= 2U);

    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(a[i].seed == b[i].seed);
        REQUIRE(a[i].moves == b[i].moves);
        REQUIRE(a[i].kingdom_count == b[i].kingdom_count);
        REQUIRE(a[i].observations == b[i].observations);
        REQUIRE(a[i].policy_targets == b[i].policy_targets);
        REQUIRE(a[i].values == b[i].values);
    }
}

TEST_CASE("v2 selfplay disables forced-playout target changes by default", "[v2][selfplay][forced]") {
    SelfPlayConfig baseline = fixed_config(1U, 4U, 8U, 0x5E1F'F0CEULL);
    baseline.dirichlet_frac = 0.25F;
    baseline.auto_play_treasures = true;
    baseline.prune_treasure_plays = true;
    baseline.temp_moves = 0U;
    // Both configs take the historical raw root_visit_policy path. Altering
    // k while the flag is off must therefore be bit-for-bit inert.
    SelfPlayConfig disabled = baseline;
    disabled.forced_playouts = false;
    disabled.forced_playouts_k = 17.0F;

    require_same_records(
        run_until_finished(baseline, 1U),
        run_until_finished(disabled, 1U));
}

TEST_CASE("v2 per-decision determinized selfplay records are bitwise deterministic", "[v2][selfplay][determinize]") {
    SelfPlayConfig config = fixed_config(2U, 8U, 8U, 0x5E1F'D371ULL);
    config.auto_play_treasures = true;
    config.prune_treasure_plays = true;
    config.temp_moves = 0U;
    config.max_tree_nodes = 512U;
    config.determinize = SelfPlayDeterminizeMode::PerDecision;
    // Existing configs may request this combination. The runner must accept
    // it and disable reuse internally because every root is freshly sampled.
    config.tree_reuse = true;

    const std::vector<SelfPlayRecord> first = run_until_finished(config, 1U);
    const std::vector<SelfPlayRecord> second = run_until_finished(config, 1U);
    REQUIRE_FALSE(first.empty());
    require_same_records(first, second);
}

TEST_CASE("v2 templated selfplay is deterministic and disabled forcing is inert", "[v2][selfplay][opening]") {
    SelfPlayConfig templated = fixed_config(2U, 8U, 8U, 0x5E1F'0E01ULL);
    templated.auto_play_treasures = true;
    templated.prune_treasure_plays = true;
    templated.temp_moves = 0U;
    templated.opening_templates_enabled = true;
    templated.opening_lambda = 0.65F;
    templated.opening_turn_window = 8;
    templated.template_weights[0] = 0.25F;
    for (std::uint8_t template_id = 1U;
         template_id < SELFPLAY_OPENING_TEMPLATE_COUNT;
         ++template_id) {
        templated.template_weights[template_id] = 0.125F;
    }
    const std::vector<SelfPlayRecord> first = run_until_finished(templated, 1U);
    const std::vector<SelfPlayRecord> second = run_until_finished(templated, 1U);
    require_same_records(first, second);

    SelfPlayConfig baseline = fixed_config(2U, 8U, 8U, 0x5E1F'0FF0ULL);
    baseline.auto_play_treasures = true;
    baseline.prune_treasure_plays = true;
    baseline.temp_moves = 0U;
    SelfPlayConfig disabled_with_templates = baseline;
    disabled_with_templates.opening_templates_enabled = false;
    disabled_with_templates.opening_lambda = 1.0F;
    disabled_with_templates.opening_turn_window = 8;
    disabled_with_templates.template_weights[0] = 0.0F;
    disabled_with_templates.template_weights[1] = 1.0F;
    for (std::uint8_t template_id = 2U;
         template_id < SELFPLAY_OPENING_TEMPLATE_COUNT;
         ++template_id) {
        disabled_with_templates.template_weights[template_id] = 0.0F;
    }
    require_same_records(
        run_until_finished(baseline, 1U),
        run_until_finished(disabled_with_templates, 1U));
}

TEST_CASE("v2 lambda-one Chapel template buys Chapel in opening smoke", "[v2][selfplay][opening][integration]") {
    SelfPlayConfig config = fixed_config(2U, 8U, 8U, 0x5E1F'C180ULL);
    config.auto_play_treasures = true;
    config.prune_treasure_plays = true;
    config.temp_moves = 0U;
    config.max_tree_nodes = 512U;
    config.fixed_setup = opening_setup({DEF_CHAPEL});
    config.opening_templates_enabled = true;
    config.opening_lambda = 1.0F;
    config.opening_turn_window = 8;
    for (std::uint8_t template_id = 0U;
         template_id < SELFPLAY_OPENING_TEMPLATE_COUNT;
         ++template_id) {
        config.template_weights[template_id] = template_id == 1U ? 1.0F : 0.0F;
    }

    const std::vector<SelfPlayRecord> records = run_until_finished(config, 2U);
    REQUIRE(records.size() >= 2U);
    for (const SelfPlayRecord& record : records) {
        REQUIRE(record.seat_template_ids[0] == 1U);
        REQUIRE(record.seat_template_ids[1] == 1U);
        REQUIRE(record.opening_buy_counts[DEF_CHAPEL] > 0U);
    }
}

TEST_CASE("v2 selfplay random kingdom pools stay restricted and validate", "[v2][selfplay]") {
    SelfPlayConfig config = fixed_config(1U, 2U, 4U, 0x5E1F'0014ULL);
    config.kingdom_mode = SelfPlayKingdomMode::Random;
    constexpr DefId pool[] = {
        DEF_VILLAGE,
        DEF_SMITHY,
        DEF_LABORATORY,
        DEF_MARKET,
        DEF_FESTIVAL,
        DEF_CELLAR,
        DEF_CHAPEL,
        DEF_MOAT,
        DEF_COUNCIL_ROOM,
        DEF_THRONE_ROOM,
        DEF_HARBINGER,
        DEF_VASSAL,
    };
    config.kingdom_pool_count = static_cast<std::uint8_t>(sizeof(pool) / sizeof(pool[0]));
    for (std::uint8_t i = 0; i < config.kingdom_pool_count; ++i) {
        config.kingdom_pool[i] = pool[i];
    }

    const std::vector<SelfPlayRecord> records = run_until_finished(config, 1U);
    REQUIRE(records.size() == 1U);
    REQUIRE(records.front().kingdom_count == 10U);
    for (std::uint8_t i = 0; i < records.front().kingdom_count; ++i) {
        bool found = false;
        for (const DefId candidate : pool) {
            found = found || records.front().kingdom[i] == candidate;
        }
        REQUIRE(found);
    }

    config.kingdom_pool_count = 9U;
    REQUIRE_THROWS_AS(SelfPlayRunner(config), std::invalid_argument);
}

TEST_CASE("v2 selfplay Scaffold records only the NN seat deterministically", "[v2][selfplay][scripted]") {
    SelfPlayConfig config = fixed_config(1U, 4U, 8U, 0x5E1F'0006ULL);
    config.max_tree_nodes = 512U;
    config.scripted_bot = SelfPlayScriptedBotKind::Scaffold;
    config.scripted_nn_player = 1U;
    config.scaffold_sims = 8U;
    config.auto_play_treasures = true;
    config.prune_treasure_plays = true;
    SelfPlayRunner runner(config);

    std::vector<SelfPlayRecord> records;
    for (std::uint32_t guard = 0; guard < 30'000U; ++guard) {
        const std::uint32_t count = runner.collect_leaves(config.max_batch);
        if (count > 0U) {
            const PlayerId* players = runner.leaf_players();
            for (std::uint32_t i = 0; i < count; ++i) {
                REQUIRE(players[i] == config.scripted_nn_player);
            }
            provide_zero_eval(runner, count);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!runner.finished_games().empty()) {
            records = runner.take_finished_games();
            break;
        }
    }

    REQUIRE_FALSE(records.empty());
    REQUIRE(runner.games_completed() == 1U);
    REQUIRE(records.size() == 1U);
    const SelfPlayRecord& record = records.front();
    REQUIRE(record.scripted_nn_player == config.scripted_nn_player);
    REQUIRE_FALSE(record.players.empty());
    REQUIRE(record.observations.size() == static_cast<std::size_t>(record.moves) * OBS_SIZE);
    REQUIRE(record.policy_targets.size() == static_cast<std::size_t>(record.moves) * ACTION_SPACE_SIZE);
    for (const PlayerId player : record.players) {
        REQUIRE(player == config.scripted_nn_player);
    }
    const float expected = record.winner == NONE
        ? 0.0F
        : (record.winner == config.scripted_nn_player ? 1.0F : -1.0F);
    for (const float value : record.values) {
        REQUIRE(value == expected);
    }

    // The charted in-tree opponent must leave the seeded Scaffold trajectory
    // reproducible while drive_scripted still supplies its actual moves.
    const std::vector<SelfPlayRecord> repeat = run_until_finished(config, 1U);
    REQUIRE(repeat.size() == 1U);
    const SelfPlayRecord& repeated = repeat.front();
    REQUIRE(repeated.seed == record.seed);
    REQUIRE(repeated.winner == record.winner);
    REQUIRE(repeated.scripted_nn_player == record.scripted_nn_player);
    REQUIRE(repeated.kingdom_count == record.kingdom_count);
    REQUIRE(repeated.moves == record.moves);
    REQUIRE(repeated.observations == record.observations);
    REQUIRE(repeated.policy_targets == record.policy_targets);
    REQUIRE(repeated.players == record.players);
    REQUIRE(repeated.values == record.values);
}

TEST_CASE("v2 selfplay EngineV3 completes a scripted batch", "[v2][selfplay][scripted]") {
    constexpr std::uint32_t N = 8U;
    SelfPlayConfig config = fixed_config(4U, 2U, 16U, 0xE3B0'7001ULL);
    config.max_tree_nodes = 512U;
    config.scripted_bot = SelfPlayScriptedBotKind::EngineV3;
    config.scripted_nn_player = 1U;
    config.auto_play_treasures = true;
    config.prune_treasure_plays = true;

    const std::vector<SelfPlayRecord> records = run_until_finished(config, N);
    REQUIRE(records.size() >= N);
    for (const SelfPlayRecord& record : records) {
        REQUIRE(record.scripted_bot == SelfPlayScriptedBotKind::EngineV3);
        REQUIRE(record.scripted_nn_player == config.scripted_nn_player);
        REQUIRE_FALSE(record.players.empty());
        for (const PlayerId player : record.players) {
            REQUIRE(player == config.scripted_nn_player);
        }
    }
}

TEST_CASE("v2 selfplay Thinner completes a scripted batch", "[v2][selfplay][scripted]") {
    constexpr std::uint32_t N = 8U;
    SelfPlayConfig config = fixed_config(4U, 2U, 16U, 0x7A1E'7001ULL);
    config.max_tree_nodes = 512U;
    config.scripted_bot = SelfPlayScriptedBotKind::Thinner;
    config.scripted_nn_player = 1U;
    config.auto_play_treasures = true;
    config.prune_treasure_plays = true;

    const std::vector<SelfPlayRecord> records = run_until_finished(config, N);
    REQUIRE(records.size() >= N);
    for (const SelfPlayRecord& record : records) {
        REQUIRE(record.scripted_bot == SelfPlayScriptedBotKind::Thinner);
        REQUIRE(record.scripted_nn_player == config.scripted_nn_player);
        REQUIRE_FALSE(record.players.empty());
        for (const PlayerId player : record.players) {
            REQUIRE(player == config.scripted_nn_player);
        }
    }
}

TEST_CASE("v2 selfplay mock evaluator throughput smoke", "[v2][selfplay][throughput]") {
    constexpr std::uint32_t N = 64U;
    constexpr std::uint32_t SIMS = 64U;
    constexpr std::uint32_t MAX_BATCH = 256U;
    SelfPlayRunner runner(fixed_config(N, SIMS, MAX_BATCH, 0x5E1F'0005ULL));

    const auto start = std::chrono::steady_clock::now();
    const std::uint64_t leaves = drive_mock(runner, MAX_BATCH, 4096U);
    const auto end = std::chrono::steady_clock::now();
    const double seconds = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1'000'000'000.0;
    const double leaves_per_sec = seconds > 0.0 ? static_cast<double>(leaves) / seconds : 0.0;
    std::cout << "selfplay_cpp_leaves_per_sec=" << leaves_per_sec
              << " n=" << N << " sims=" << SIMS << " max_batch=" << MAX_BATCH << "\n";

    REQUIRE(leaves >= 4096U);
    REQUIRE(leaves_per_sec > 0.0);
}
