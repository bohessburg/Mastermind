#include "v2/mcts/eval_runner.h"

#include "v2/core/game.h"
#include "v2/core/score.h"
#include "v2/core/setup.h"
#include "v2/core/turns.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

namespace {

[[nodiscard]] Pile* find_pile(GameState& state, DefId def) {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        Pile& pile = state.piles[i];
        if (pile.mixed_len == 0U && state.slot_to_def[pile.base] == def) {
            return &pile;
        }
    }
    return nullptr;
}

void clear_player_cards(GameState& state, PlayerId player_id) {
    PlayerState& player = state.players[player_id];
    for (Slot slot = 0; slot < MAX_SLOTS; ++slot) {
        player.hand[slot] = 0U;
        player.exile[slot] = 0U;
        player.tavern[slot] = 0U;
        player.island_mat[slot] = 0U;
    }
    player.deck.size = 0U;
    player.discard.size = 0U;
    player.set_aside.size = 0U;
    player.in_play_size = 0U;
    player.pending_size = 0U;
}

void add_to_discard(GameState& state, PlayerId player_id, DefId def, std::uint8_t count) {
    const Slot slot = slot_of(state, def);
    PlayerState& player = state.players[player_id];
    for (std::uint8_t i = 0; i < count; ++i) {
        player.discard.cards[player.discard.size] = slot;
        ++player.discard.size;
    }
}

[[nodiscard]] GameState pile_clock_engine_chart_position(bool player_ahead) {
    Setup setup{};
    setup.kingdom_count = 1U;
    setup.kingdom[0] = DEF_VILLAGE;
    GameState state = Game::new_game(setup, 0xC10C'0002ULL);
    clear_player_cards(state, 0U);
    clear_player_cards(state, 1U);
    add_to_discard(state, player_ahead ? 0U : 1U, DEF_PROVINCE, 1U);
    add_to_discard(state, player_ahead ? 1U : 0U, DEF_ESTATE, 1U);

    Pile* copper = find_pile(state, DEF_COPPER);
    Pile* gold = find_pile(state, DEF_GOLD);
    Pile* village = find_pile(state, DEF_VILLAGE);
    REQUIRE(copper != nullptr);
    REQUIRE(gold != nullptr);
    REQUIRE(village != nullptr);
    copper->count = 0U;
    gold->count = 0U;
    village->count = 1U;

    state.phase = static_cast<std::uint8_t>(Phase::Buy);
    state.actions = 0U;
    state.buys = 1U;
    state.coins = 3;
    refresh_current_decision(state);
    return state;
}

[[nodiscard]] EvalRunnerConfig fixed_eval_config(
    std::uint32_t n_games,
    std::uint32_t sims,
    std::uint32_t max_batch,
    std::uint64_t seed) {
    EvalRunnerConfig config{};
    config.n_games = n_games;
    config.sims_per_move = sims;
    config.max_batch = max_batch;
    config.seed = seed;
    config.kingdom_mode = SelfPlayKingdomMode::Fixed;
    config.max_tree_nodes = 1024U;
    config.opponent = EvalScriptedBotKind::BigMoney;
    return config;
}

void provide_zero_eval(EvalRunner& runner, std::uint32_t count) {
    std::vector<float> values(count, 0.0F);
    std::vector<float> policies(static_cast<std::size_t>(count) * ACTION_SPACE_SIZE, 0.0F);
    runner.provide_evaluations(values.data(), policies.data(), count);
}

void drive_until_games(EvalRunner& runner, std::uint32_t max_batch, std::uint32_t target_games) {
    std::uint32_t guard = 0;
    while (runner.games_completed() < target_games && guard < 20000U) {
        const std::uint32_t count = runner.collect_leaves(max_batch);
        REQUIRE(count > 0U);
        provide_zero_eval(runner, count);
        ++guard;
    }
    REQUIRE(runner.games_completed() >= target_games);
}

} // namespace

TEST_CASE("v2 eval runner seat swapping alternates NN player", "[v2][eval_runner]") {
    EvalRunner runner(fixed_eval_config(4U, 4U, 8U, 0xE0A1'0001ULL));
    REQUIRE(runner.active_sequence(0U) == 0U);
    REQUIRE(runner.active_sequence(1U) == 1U);
    REQUIRE(runner.active_sequence(2U) == 2U);
    REQUIRE(runner.active_sequence(3U) == 3U);
    REQUIRE(runner.active_nn_player(0U) == 0U);
    REQUIRE(runner.active_nn_player(1U) == 1U);
    REQUIRE(runner.active_nn_player(2U) == 0U);
    REQUIRE(runner.active_nn_player(3U) == 1U);
}

TEST_CASE("v2 eval runner uses the selected observation-version stride", "[v2][eval_runner][encode]") {
    EvalRunnerConfig config = fixed_eval_config(2U, 4U, 4U, 0xE0A1'0101ULL);
    config.obs_version = ObsVersion::V2;
    EvalRunner runner(config);

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

TEST_CASE("v2 eval runner accepts the v3 observation stride", "[v2][eval_runner][encode]") {
    EvalRunnerConfig config = fixed_eval_config(2U, 4U, 4U, 0xE0A1'0102ULL);
    config.obs_version = ObsVersion::V3;
    EvalRunner runner(config);

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

TEST_CASE("v2 eval scripted BigMoney policy follows known phase choices", "[v2][eval_runner]") {
    GameState state = Game::new_game(Setup{}, 0xE0A1'0002ULL);
    Xoshiro256pp rng = Xoshiro256pp::seeded(0xE0A1'0002ULL);
    ActionMask legal{};

    int legal_count = Game::legal_actions(state, legal);
    REQUIRE(legal_count > 0);
    REQUIRE(eval_scripted_action(state, legal, legal_count, EvalScriptedBotKind::BigMoney, rng) == A_PASS);

    state.phase = static_cast<std::uint8_t>(Phase::Buy);
    state.buys = 1U;
    state.coins = 6;
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        state.players[0].hand[slot] = 0U;
    }
    state.decision.player = 0U;
    state.decision.kind = static_cast<std::uint8_t>(DecisionKind::PhaseBuy);
    state.decision.source = 0U;
    state.decision.min_left = 0U;
    state.decision.max_left = 0U;
    legal_count = Game::legal_actions(state, legal);
    REQUIRE(legal_count > 0);
    REQUIRE(eval_scripted_action(state, legal, legal_count, EvalScriptedBotKind::BigMoney, rng) == buy_action(DEF_GOLD));
}

TEST_CASE("v2 Engine chart avoids a third pile while behind", "[v2][eval_runner]") {
    GameState state = pile_clock_engine_chart_position(false);
    ActionMask legal{};
    const int legal_count = Game::legal_actions(state, legal);
    Xoshiro256pp rng = Xoshiro256pp::seeded(0xC10C'0002ULL);

    REQUIRE(legal_count > 0);
    REQUIRE(legal.test(buy_action(DEF_VILLAGE)));
    REQUIRE(legal.test(buy_action(DEF_ESTATE)));
    REQUIRE(score(state, 0U) < score(state, 1U));
    REQUIRE(eval_scripted_action(state, legal, legal_count, EvalScriptedBotKind::Engine, rng)
        == buy_action(DEF_ESTATE));
}

TEST_CASE("v2 Engine chart ends on a third pile while ahead", "[v2][eval_runner]") {
    GameState state = pile_clock_engine_chart_position(true);
    ActionMask legal{};
    const int legal_count = Game::legal_actions(state, legal);
    Xoshiro256pp rng = Xoshiro256pp::seeded(0xC10C'0003ULL);

    REQUIRE(legal_count > 0);
    REQUIRE(legal.test(buy_action(DEF_VILLAGE)));
    REQUIRE(score(state, 0U) > score(state, 1U));
    REQUIRE(eval_scripted_action(state, legal, legal_count, EvalScriptedBotKind::Engine, rng)
        == buy_action(DEF_VILLAGE));
}

TEST_CASE("v2 Scaffold MCTS config matches the rollout yardstick", "[v2][eval_runner][mcts]") {
    const MctsConfig config = make_scaffold_mcts_config(400U, 1.25F, 4096U, true);
    const MctsConfig custom_determinizations = make_scaffold_mcts_config(8U, 1.25F, 512U, false, 3U);

    REQUIRE(config.sims_per_move == 400U);
    REQUIRE(config.c_puct == 1.25F);
    REQUIRE(config.determinizations == 2U);
    REQUIRE(config.max_tree_nodes == 4096U);
    REQUIRE(config.rollout_step_cap == 1024U);
    REQUIRE(config.rollout_policy == MctsRolloutPolicy::EngineLike);
    REQUIRE(config.prune_treasure_plays);
    REQUIRE(config.prior_fn != nullptr);
    const GameState state = Game::new_game(Setup{}, 0x5CAFF01DULL);
    REQUIRE(config.prior_fn(state, 0U, A_PASS, config.prior_user) == 1.0F);
    REQUIRE(custom_determinizations.determinizations == 3U);
}

TEST_CASE("v2 eval runner mock evaluator completes games", "[v2][eval_runner]") {
    EvalRunner runner(fixed_eval_config(2U, 8U, 8U, 0xE0A1'0003ULL));
    drive_until_games(runner, 8U, 2U);

    const EvalRunnerResult result = runner.result();
    REQUIRE(result.games >= 2U);
    REQUIRE(result.games == result.nn_wins + result.scripted_wins + result.ties);
}

TEST_CASE("v2 eval runner completes games against Thinner", "[v2][eval_runner][thinner]") {
    EvalRunnerConfig config = fixed_eval_config(2U, 8U, 8U, 0x7A1E'0001ULL);
    config.opponent = EvalScriptedBotKind::Thinner;
    EvalRunner runner(config);
    drive_until_games(runner, config.max_batch, 2U);

    const EvalRunnerResult result = runner.result();
    REQUIRE(result.games >= 2U);
    REQUIRE(result.games == result.nn_wins + result.scripted_wins + result.ties);
}
