#include "v2/core/game.h"
#include "v2/core/turns.h"
#include "v2/drivers/bots.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>

namespace {

constexpr int MAX_RECORDED_ACTIONS = 4096;

struct ActionLog {
    Action actions[MAX_RECORDED_ACTIONS]{};
    int count = 0;
};

[[nodiscard]] std::uint64_t hash_state(const GameState& state) {
    constexpr std::uint64_t kOffset = 14695981039346656037ULL;
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&state);
    std::uint64_t hash = kOffset;
    for (std::size_t i = 0; i < sizeof(GameState); ++i) {
        hash ^= bytes[i];
        hash *= kPrime;
    }
    return hash;
}

[[nodiscard]] bool same_bytes(const GameState& lhs, const GameState& rhs) {
    return std::memcmp(&lhs, &rhs, sizeof(GameState)) == 0;
}

[[nodiscard]] Action choose_for_player(
    GameState& state,
    RandomBot& bot0,
    RandomBot& bot1) {
    ActionMask legal{};
    const int count = Game::legal_actions(state, legal);
    REQUIRE(count > 0);
    const PlayerId player = Game::current_decision(state).player;
    return player == 0U
        ? bot0.choose_action(state, legal, count)
        : bot1.choose_action(state, legal, count);
}

[[nodiscard]] ActionLog record_random_game(std::uint64_t game_seed) {
    GameState state = Game::new_game(Setup{}, game_seed);
    RandomBot bot0{game_seed ^ 0xA5A5A5A5A5A5A5A5ULL};
    RandomBot bot1{game_seed ^ 0x5A5A5A5A5A5A5A5AULL};
    ActionLog log{};

    bool done = false;
    while (!done) {
        REQUIRE(log.count < MAX_RECORDED_ACTIONS);
        const Action action = choose_for_player(state, bot0, bot1);
        log.actions[log.count] = action;
        ++log.count;
        done = Game::step(state, action);
    }
    return log;
}

[[nodiscard]] GameState replay_actions(const ActionLog& log, std::uint64_t game_seed) {
    GameState state = Game::new_game(Setup{}, game_seed);
    bool done = false;
    for (int i = 0; i < log.count; ++i) {
        ActionMask legal{};
        const int legal_count = Game::legal_actions(state, legal);
        REQUIRE(legal_count > 0);
        REQUIRE(legal.test(log.actions[i]));
        done = Game::step(state, log.actions[i]);
    }
    REQUIRE(done);
    return state;
}

} // namespace

TEST_CASE("v2 same seed and action list replay to byte-identical final states", "[v2][determinism]") {
    constexpr std::uint64_t kSeed = 0xD37E'1201'5EED'0001ULL;
    const ActionLog log = record_random_game(kSeed);
    REQUIRE(log.count > 0);

    const GameState first = replay_actions(log, kSeed);
    const GameState second = replay_actions(log, kSeed);

    REQUIRE(hash_state(first) == hash_state(second));
    REQUIRE(same_bytes(first, second));
}

TEST_CASE("v2 clone-equivalence holds across random decisions", "[v2][determinism]") {
    GameState state = Game::new_game(Setup{}, 0xC10A'E000'0000'0020ULL);
    RandomBot bot0{0xC10A'E000'B070'0000ULL};
    RandomBot bot1{0xC10A'E000'B071'0000ULL};

    for (int i = 0; i < 20; ++i) {
        const Action action = choose_for_player(state, bot0, bot1);
        GameState clone = state;

        const bool done_a = Game::step(state, action);
        const bool done_b = Game::step(clone, action);

        REQUIRE(done_a == done_b);
        REQUIRE(hash_state(state) == hash_state(clone));
        REQUIRE(same_bytes(state, clone));
        REQUIRE_FALSE(done_a);
    }
}

TEST_CASE("v2 BigMoney games complete without truncation across seeds", "[v2][determinism]") {
    for (std::uint64_t seed = 0; seed < 50U; ++seed) {
        const GameResult result = run_game(
            Setup{},
            0xB16'0000ULL + seed,
            BotSpec{BotKind::BigMoney, seed},
            BotSpec{BotKind::BigMoney, seed + 1000U});

        REQUIRE_FALSE(result.truncated);
        REQUIRE(result.turns > 10U);
        REQUIRE(result.turns < MAX_TURNS);
        REQUIRE(result.scores[0] > 0);
        REQUIRE(result.scores[1] > 0);
    }
}
