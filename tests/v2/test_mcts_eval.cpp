#include "v2/mcts/eval.h"

#include "v2/core/defs.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace {

[[nodiscard]] bool contains_def(const Setup& setup, DefId def) {
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        if (setup.kingdom[i] == def) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool has_duplicates(const Setup& setup) {
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        for (std::uint8_t j = static_cast<std::uint8_t>(i + 1U); j < setup.kingdom_count; ++j) {
            if (setup.kingdom[i] == setup.kingdom[j]) {
                return true;
            }
        }
    }
    return false;
}

} // namespace

TEST_CASE("v2 MCTS eval setups always use real kingdoms", "[v2][mcts][eval]") {
    const Setup fixed = fixed_mcts_eval_setup();
    REQUIRE(fixed.kingdom_count == 10U);
    REQUIRE(contains_def(fixed, DEF_VILLAGE));
    REQUIRE(contains_def(fixed, DEF_MOAT));
    REQUIRE_FALSE(has_duplicates(fixed));

    const Setup random = random_mcts_eval_setup(0xE0A1'0001ULL);
    REQUIRE(random.kingdom_count == 10U);
    REQUIRE_FALSE(has_duplicates(random));
    for (std::uint8_t i = 0; i < random.kingdom_count; ++i) {
        REQUIRE(random.kingdom[i] >= DEF_CELLAR);
        REQUIRE(random.kingdom[i] <= DEF_SENTRY);
    }
}

TEST_CASE("v2 MCTS phase gate beats EngineBot on random kingdoms", "[v2][mcts][eval][.][slow]") {
    MctsEvalOptions options{};
    options.sims = 1000U;
    options.games = 200U;
    options.opponent = MctsEvalOpponent::Engine;
    options.kingdoms = MctsEvalKingdoms::Random;
    options.seed = 0x600D'600DULL;
    options.determinizations = 8U;
    options.threads = 2U;

    const MctsEvalResult result = run_mcts_eval(options);
    INFO("wins=" << result.mcts_wins << " losses=" << result.opponent_wins
                 << " ties=" << result.ties << " truncated=" << result.truncated
                 << " sims/sec=" << result.mcts_sims_per_sec()
                 << " avg_move_ms=" << result.avg_move_ms());

    const std::uint32_t decisive = result.mcts_wins + result.opponent_wins;
    REQUIRE(result.games == options.games);
    REQUIRE(decisive > 0U);
    REQUIRE((result.mcts_wins * 2U) > decisive);
}
