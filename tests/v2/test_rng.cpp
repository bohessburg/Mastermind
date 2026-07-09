#include "v2/core/rng.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <type_traits>

TEST_CASE("v2 Xoshiro256pp is trivially copyable and 32 bytes", "[v2][rng]") {
    STATIC_REQUIRE(std::is_trivially_copyable_v<Xoshiro256pp>);
    STATIC_REQUIRE(sizeof(Xoshiro256pp) == 32U);
}

TEST_CASE("v2 Xoshiro256pp is deterministic for the same seed", "[v2][rng]") {
    Xoshiro256pp a = Xoshiro256pp::seeded(0x123456789ABCDEF0ULL);
    Xoshiro256pp b = Xoshiro256pp::seeded(0x123456789ABCDEF0ULL);

    for (int i = 0; i < 32; ++i) {
        REQUIRE(a.next() == b.next());
    }
}

TEST_CASE("v2 Xoshiro256pp bounded uniform stays in range", "[v2][rng]") {
    constexpr std::size_t kBound = 10;
    constexpr int kSamples = 100000;
    std::array<int, kBound> counts{};
    Xoshiro256pp rng = Xoshiro256pp::seeded(0xBADC0FFEE0DDF00DULL);

    for (int i = 0; i < kSamples; ++i) {
        const std::uint32_t value = rng.uniform(static_cast<std::uint32_t>(kBound));
        REQUIRE(value < kBound);
        ++counts[static_cast<std::size_t>(value)];
    }

    for (const int count : counts) {
        REQUIRE(count > 8500);
        REQUIRE(count < 11500);
    }
}
