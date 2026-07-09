#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>

struct Xoshiro256pp {
    std::uint64_t state[4];

    [[nodiscard]] static constexpr Xoshiro256pp seeded(std::uint64_t seed) noexcept {
        Xoshiro256pp rng{{0U, 0U, 0U, 0U}};
        rng.seed(seed);
        return rng;
    }

    constexpr void seed(std::uint64_t seed_value) noexcept {
        std::uint64_t x = seed_value;
        state[0] = splitmix64_next(x);
        state[1] = splitmix64_next(x);
        state[2] = splitmix64_next(x);
        state[3] = splitmix64_next(x);
    }

    [[nodiscard]] constexpr std::uint64_t next() noexcept {
        const std::uint64_t result = rotl(state[0] + state[3], 23) + state[0];
        const std::uint64_t t = state[1] << 17;

        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];

        state[2] ^= t;
        state[3] = rotl(state[3], 45);

        return result;
    }

    [[nodiscard]] std::uint32_t uniform(std::uint32_t bound) noexcept {
        assert(bound > 0U);
        if (bound == 0U) {
            return 0U;
        }

        const std::uint64_t range = bound;
        const std::uint64_t threshold = (std::uint64_t{0} - range) % range;

        for (;;) {
            const std::uint64_t value = next();
            if (value >= threshold) {
                return static_cast<std::uint32_t>(value % range);
            }
        }
    }

    [[nodiscard]] std::uint32_t bounded(std::uint32_t bound) noexcept {
        return uniform(bound);
    }

    [[nodiscard]] int uniform_int(int bound) noexcept {
        assert(bound > 0);
        if (bound <= 0) {
            return 0;
        }
        return static_cast<int>(uniform(static_cast<std::uint32_t>(bound)));
    }

private:
    [[nodiscard]] static constexpr std::uint64_t rotl(std::uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }

    [[nodiscard]] static constexpr std::uint64_t splitmix64_next(std::uint64_t& x) noexcept {
        x += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
};

static_assert(std::is_trivially_copyable_v<Xoshiro256pp>);
static_assert(sizeof(Xoshiro256pp) == 32U);
