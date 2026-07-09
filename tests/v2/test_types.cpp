#include "v2/core/types.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("v2 Cost fits_within is componentwise", "[v2][types]") {
    const Cost exact{3, 1, 2};
    REQUIRE(exact.fits_within(Cost{3, 1, 2}));
    REQUIRE(exact.fits_within(Cost{4, 1, 2}));
    REQUIRE(exact.fits_within(Cost{3, 2, 3}));

    REQUIRE_FALSE(Cost{4, 1, 2}.fits_within(exact));
    REQUIRE_FALSE(Cost{3, 2, 2}.fits_within(exact));
    REQUIRE_FALSE(Cost{3, 1, 3}.fits_within(exact));
}

TEST_CASE("v2 Cost equality is exact across coins potion and debt", "[v2][types]") {
    REQUIRE(Cost{5, 0, 0} == Cost{5, 0, 0});
    REQUIRE_FALSE(Cost{5, 0, 0} == Cost{5, 1, 0});
    REQUIRE_FALSE(Cost{5, 1, 0} == Cost{5, 0, 0});
    REQUIRE_FALSE(Cost{5, 0, 0} == Cost{5, 0, 1});
}

TEST_CASE("v2 Cost potion and debt budgets are independent", "[v2][types]") {
    REQUIRE(Cost{0, 1, 0}.fits_within(Cost{0, 1, 0}));
    REQUIRE_FALSE(Cost{0, 1, 0}.fits_within(Cost{20, 0, 20}));

    REQUIRE(Cost{0, 0, 5}.fits_within(Cost{0, 0, 5}));
    REQUIRE_FALSE(Cost{0, 0, 5}.fits_within(Cost{20, 1, 4}));
}
