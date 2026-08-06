#include "fuzz_harness.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("v2 seeded random-kingdom fuzz smoke stays invariant-clean", "[v2][fuzz]") {
    FuzzConfig config{};
    config.seed_start = 0xF022'2026ULL;
    config.seeds = 50;
    config.steps_budget = 250000;
    config.players = 0;

    const FuzzResult result = run_fuzz(config);
    INFO(result.violation.message);
    REQUIRE(result.violations == 0U);
    REQUIRE(result.games == config.seeds);
    REQUIRE(result.steps > 0U);
}
