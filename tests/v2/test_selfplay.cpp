#include "v2/mcts/selfplay.h"

#include "v2/core/game.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
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

} // namespace

TEST_CASE("v2 selfplay virtual loss clears after evaluations", "[v2][selfplay]") {
    SelfPlayRunner runner(fixed_config(4U, 8U, 8U, 0x5E1F'0001ULL));

    const std::uint32_t count = runner.collect_leaves(8U);
    REQUIRE(count > 0U);
    REQUIRE(runner.total_virtual_loss() > 0.0F);

    provide_zero_eval(runner, count);
    REQUIRE(runner.total_virtual_loss() == Catch::Approx(0.0F));
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

TEST_CASE("v2 selfplay finished records have normalized policies and terminal values", "[v2][selfplay]") {
    const std::vector<SelfPlayRecord> records = run_until_finished(
        fixed_config(2U, 16U, 16U, 0x5E1F'0003ULL),
        1U);
    REQUIRE_FALSE(records.empty());

    const SelfPlayRecord& record = records.front();
    REQUIRE(record.moves > 0U);
    REQUIRE(record.observations.size() == static_cast<std::size_t>(record.moves) * OBS_SIZE);
    REQUIRE(record.policy_targets.size() == static_cast<std::size_t>(record.moves) * ACTION_SPACE_SIZE);
    REQUIRE(record.values.size() == record.moves);
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
