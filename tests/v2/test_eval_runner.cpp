#include "v2/mcts/eval_runner.h"

#include "v2/core/game.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

namespace {

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

TEST_CASE("v2 Scaffold MCTS config matches the rollout yardstick", "[v2][eval_runner][mcts]") {
    const MctsConfig config = make_scaffold_mcts_config(400U, 1.25F, 4096U, true);

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
}

TEST_CASE("v2 eval runner mock evaluator completes games", "[v2][eval_runner]") {
    EvalRunner runner(fixed_eval_config(2U, 8U, 8U, 0xE0A1'0003ULL));
    drive_until_games(runner, 8U, 2U);

    const EvalRunnerResult result = runner.result();
    REQUIRE(result.games >= 2U);
    REQUIRE(result.games == result.nn_wins + result.scripted_wins + result.ties);
}
