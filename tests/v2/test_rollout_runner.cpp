#include "v2/mcts/rollout_runner.h"

#include "v2/core/actions.h"
#include "v2/core/defs.h"
#include "v2/core/game.h"
#include "v2/core/setup.h"
#include "v2/mcts/tree.h"

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

[[nodiscard]] GameState rollout_root(std::uint64_t seed) {
    Setup setup{};
    setup.num_players = 2U;
    const DefId kingdom[] = {
        DEF_CELLAR, DEF_CHAPEL, DEF_VILLAGE, DEF_SMITHY, DEF_WORKSHOP,
        DEF_REMODEL, DEF_MINE, DEF_MERCHANT, DEF_MILITIA, DEF_WITCH,
    };
    setup.kingdom_count = 10U;
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        setup.kingdom[i] = kingdom[i];
    }
    GameState state = Game::new_game(setup, seed);
    // Advance to the first real choice: a buy decision with treasures played.
    for (int guard = 0; guard < 64; ++guard) {
        ActionMask legal{};
        const int count = Game::legal_actions(state, legal);
        const Action treasure = mcts_canonical_treasure_play(mcts_filter_treasure_plays(state, legal));
        if (treasure != A_PASS) {
            (void)Game::step(state, treasure);
            continue;
        }
        if (count >= 2) {
            break;
        }
        (void)Game::step(state, legal.nth_set(0U));
    }
    return state;
}

[[nodiscard]] RolloutConfig rollout_config(float temperature) {
    RolloutConfig config{};
    config.n_slots = 8U;
    config.auto_play_treasures = true;
    config.prune_treasure_plays = true;
    config.max_turns = 40U;
    config.temperature = temperature;
    return config;
}

[[nodiscard]] std::vector<Action> legal_list(const GameState& state) {
    ActionMask legal{};
    (void)Game::legal_actions(state, legal);
    std::vector<Action> out;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (legal.test(action)) {
            out.push_back(action);
        }
    }
    return out;
}

// Uniform logits: with temperature 1 this is a uniformly random legal policy.
[[nodiscard]] std::vector<RolloutResult> drive(RolloutRunner& runner) {
    std::vector<float> logits;
    std::uint32_t guard = 0U;
    for (;;) {
        const std::uint32_t count = runner.collect_leaves();
        if (count == 0U) {
            break;
        }
        REQUIRE(++guard < 1'000'000U);
        logits.assign(static_cast<std::size_t>(count) * ACTION_SPACE_SIZE, 0.0F);
        const bool* masks = runner.leaf_legal_masks();
        for (std::uint32_t leaf = 0; leaf < count; ++leaf) {
            int legal = 0;
            for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
                legal += masks[(static_cast<std::size_t>(leaf) * ACTION_SPACE_SIZE) + action] ? 1 : 0;
            }
            // Forced moves are played by the runner, never sent for evaluation.
            REQUIRE(legal >= 2);
        }
        runner.provide_policies(logits.data(), count);
    }
    REQUIRE(runner.queued_jobs() == 0U);
    REQUIRE(runner.active_rollouts() == 0U);
    std::vector<RolloutResult> results = runner.take_results();
    std::sort(results.begin(), results.end(), [](const RolloutResult& a, const RolloutResult& b) {
        return a.tag < b.tag;
    });
    return results;
}

[[nodiscard]] std::vector<RolloutResult> run_jobs(float temperature, bool determinize_world) {
    RolloutRunner runner(rollout_config(temperature));
    const GameState root = rollout_root(0x0220'77ULL);
    const std::uint32_t position = runner.add_position(root);
    const std::vector<Action> actions = legal_list(root);
    REQUIRE(actions.size() >= 2U);
    std::uint32_t tag = 0U;
    for (const Action action : {actions[0], actions[1]}) {
        for (std::uint64_t rollout = 0; rollout < 12U; ++rollout) {
            RolloutJob job{};
            job.position = position;
            job.action = action;
            job.seed = 0xABCD'0000ULL + rollout;
            job.tag = tag++;
            job.determinize = determinize_world;
            runner.add_job(job);
        }
    }
    return drive(runner);
}

} // namespace

TEST_CASE("v2 rollout runner finishes every job with a consistent result", "[v2][rollout]") {
    for (const bool determinize_world : {true, false}) {
        const std::vector<RolloutResult> results = run_jobs(1.0F, determinize_world);
        REQUIRE(results.size() == 24U);
        for (std::uint32_t i = 0; i < results.size(); ++i) {
            const RolloutResult& result = results[i];
            REQUIRE(result.tag == i);
            REQUIRE((result.outcome == 0.0F || result.outcome == 0.5F || result.outcome == 1.0F));
            // Outcome and margin are both from the root decision player's view.
            if (result.outcome == 1.0F) {
                REQUIRE(result.margin >= 0);
            }
            if (result.outcome == 0.0F) {
                REQUIRE(result.margin <= 0);
            }
            REQUIRE(result.turns <= 40U);
        }
    }
}

TEST_CASE("v2 rollout runner is deterministic for fixed seeds", "[v2][rollout]") {
    const std::vector<RolloutResult> a = run_jobs(1.0F, true);
    const std::vector<RolloutResult> b = run_jobs(1.0F, true);
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(a[i].tag == b[i].tag);
        REQUIRE(a[i].outcome == b[i].outcome);
        REQUIRE(a[i].margin == b[i].margin);
        REQUIRE(a[i].turns == b[i].turns);
    }
}

TEST_CASE("v2 rollout runner argmax playouts hit the turn cap as truncated games", "[v2][rollout]") {
    // Uniform logits at temperature 0 always pick the lowest legal action id
    // (pass), so nobody ever buys and the cap must end the game.
    const std::vector<RolloutResult> results = run_jobs(0.0F, true);
    REQUIRE(results.size() == 24U);
    for (const RolloutResult& result : results) {
        REQUIRE(result.truncated);
        REQUIRE(result.turns == 40U);
    }
}

TEST_CASE("v2 rollout runner rejects illegal jobs", "[v2][rollout]") {
    RolloutRunner runner(rollout_config(0.0F));
    const GameState root = rollout_root(0x0220'78ULL);
    const std::uint32_t position = runner.add_position(root);
    ActionMask legal{};
    (void)Game::legal_actions(root, legal);
    Action illegal = A_END;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (!legal.test(action)) {
            illegal = action;
            break;
        }
    }
    REQUIRE(illegal != A_END);
    RolloutJob job{};
    job.position = position;
    job.action = illegal;
    REQUIRE_THROWS_AS(runner.add_job(job), std::invalid_argument);
    job.position = position + 1U;
    job.action = A_PASS;
    REQUIRE_THROWS_AS(runner.add_job(job), std::invalid_argument);
    REQUIRE(runner.collect_leaves() == 0U);
}
