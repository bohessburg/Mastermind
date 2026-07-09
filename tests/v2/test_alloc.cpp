#include "v2/drivers/bots.h"

#include "v2/core/determinize.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace {

bool g_count_allocations = false;
std::uint64_t g_allocations = 0;

struct AllocationScope {
    AllocationScope() {
        g_allocations = 0;
        g_count_allocations = true;
    }

    ~AllocationScope() {
        g_count_allocations = false;
    }

    [[nodiscard]] std::uint64_t count() const {
        return g_allocations;
    }
};

} // namespace

void* operator new(std::size_t size) {
    if (g_count_allocations) {
        ++g_allocations;
    }
    if (void* ptr = std::malloc(size)) {
        return ptr;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    if (g_count_allocations) {
        ++g_allocations;
    }
    if (void* ptr = std::malloc(size)) {
        return ptr;
    }
    throw std::bad_alloc();
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

TEST_CASE("v2 new_game and BigMoney loops allocate nothing", "[v2][alloc]") {
    std::uint64_t allocations = 0;
    int truncated = 0;
    {
        AllocationScope scope;
        for (std::uint64_t seed = 0; seed < 1000U; ++seed) {
            const GameResult result = run_game(
                Setup{},
                0xA110'C000ULL + seed,
                BotSpec{BotKind::BigMoney, seed},
                BotSpec{BotKind::BigMoney, seed + 100000U});
            if (result.truncated) {
                ++truncated;
            }

            GameState state = Game::new_game(Setup{}, 0xD373'C000ULL + seed);
            determinize(state, 0U, 0xD373'D000ULL + seed);
        }
        allocations = scope.count();
    }

    REQUIRE(truncated == 0);
    REQUIRE(allocations == 0U);
}
