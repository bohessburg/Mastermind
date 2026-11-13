#include "v2/core/defs.h"
#include "v2/core/game.h"
#include "v2/core/setup.h"
#include "v2/drivers/bots.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <sstream>
#include <string>

#ifndef V2_GOLDEN_DIR
#define V2_GOLDEN_DIR "tests/v2/golden"
#endif

namespace {

constexpr std::array<std::uint64_t, 5> kGoldenSeeds{
    0x601D'0001ULL,
    0x601D'0002ULL,
    0x601D'0003ULL,
    0x601D'0004ULL,
    0x601D'0005ULL,
};
constexpr std::size_t kMaxGoldenActions = 16384U;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

struct GoldenRecord {
    std::uint64_t seed = 0;
    std::uint64_t hash = 0;
    std::size_t action_count = 0;
    std::array<Action, kMaxGoldenActions> actions{};
};

[[nodiscard]] Setup golden_setup() noexcept {
    Setup setup{};
    setup.num_players = 2;
    setup.kingdom_count = 10;
    setup.kingdom[0] = DEF_SENTRY;
    setup.kingdom[1] = DEF_LIBRARY;
    setup.kingdom[2] = DEF_THRONE_ROOM;
    setup.kingdom[3] = DEF_BANDIT;
    setup.kingdom[4] = DEF_WITCH;
    setup.kingdom[5] = DEF_MOAT;
    setup.kingdom[6] = DEF_VILLAGE;
    setup.kingdom[7] = DEF_SMITHY;
    setup.kingdom[8] = DEF_MARKET;
    setup.kingdom[9] = DEF_REMODEL;
    return setup;
}

[[nodiscard]] std::uint64_t fnv1a_state(const GameState& state) noexcept {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&state);
    std::uint64_t hash = kFnvOffset;
    for (std::size_t i = 0; i < sizeof(GameState); ++i) {
        hash ^= bytes[i];
        hash *= kFnvPrime;
    }
    return hash;
}

[[nodiscard]] std::filesystem::path golden_path(std::uint64_t seed) {
    std::ostringstream name;
    name << "seed_" << std::hex << std::nouppercase << seed << ".txt";
    return std::filesystem::path(V2_GOLDEN_DIR) / name.str();
}

[[nodiscard]] std::uint64_t bot_seed(std::uint64_t seed, PlayerId player) noexcept {
    return seed ^ (0xB007'601D'0000'0000ULL + static_cast<std::uint64_t>(player));
}

[[nodiscard]] GoldenRecord record_random_game(std::uint64_t seed) {
    GoldenRecord record{};
    record.seed = seed;
    GameState state = Game::new_game(golden_setup(), seed);
    RandomBot bots[2] = {
        RandomBot(bot_seed(seed, 0U)),
        RandomBot(bot_seed(seed, 1U)),
    };

    bool done = state.phase == static_cast<std::uint8_t>(Phase::Over);
    ActionMask legal{};
    while (!done) {
        const int legal_count = Game::legal_actions(state, legal);
        REQUIRE(legal_count > 0);
        const PlayerId player = Game::current_decision(state).player;
        REQUIRE(player < 2U);
        REQUIRE(record.action_count < record.actions.size());
        const Action action = bots[player].choose_action(state, legal, legal_count);
        record.actions[record.action_count] = action;
        ++record.action_count;
        done = Game::step(state, action);
    }

    record.hash = fnv1a_state(state);
    return record;
}

void replay_record(const GoldenRecord& record) {
    GameState state = Game::new_game(golden_setup(), record.seed);
    bool done = state.phase == static_cast<std::uint8_t>(Phase::Over);
    ActionMask legal{};
    for (std::size_t i = 0; i < record.action_count; ++i) {
        const int legal_count = Game::legal_actions(state, legal);
        REQUIRE(legal_count > 0);
        REQUIRE(legal.test(record.actions[i]));
        done = Game::step(state, record.actions[i]);
    }
    REQUIRE(done);
    CHECK(fnv1a_state(state) == record.hash);
}

void write_record(const GoldenRecord& record) {
    std::filesystem::create_directories(V2_GOLDEN_DIR);
    std::ofstream out(golden_path(record.seed));
    REQUIRE(out.is_open());
    out << "seed 0x" << std::hex << record.seed << '\n';
    out << "hash 0x" << std::hex << record.hash << '\n';
    out << "actions " << std::dec << record.action_count << '\n';
    for (std::size_t i = 0; i < record.action_count; ++i) {
        out << record.actions[i] << '\n';
    }
}

[[nodiscard]] GoldenRecord read_record(std::uint64_t seed) {
    GoldenRecord record{};
    const std::filesystem::path path = golden_path(seed);
    std::ifstream in(path);
    INFO("golden file: " << path.string());
    REQUIRE(in.is_open());

    std::string label;
    in >> label;
    REQUIRE(label == "seed");
    in >> std::hex >> record.seed;
    REQUIRE(record.seed == seed);

    in >> label;
    REQUIRE(label == "hash");
    in >> std::hex >> record.hash;

    in >> label;
    REQUIRE(label == "actions");
    in >> std::dec >> record.action_count;
    REQUIRE(record.action_count <= record.actions.size());

    for (std::size_t i = 0; i < record.action_count; ++i) {
        std::uint32_t action = 0;
        in >> action;
        REQUIRE(action < ACTION_SPACE_SIZE);
        record.actions[i] = static_cast<Action>(action);
    }
    return record;
}

} // namespace

TEST_CASE("v2 golden random replays reproduce final state hashes", "[v2][golden]") {
    const bool regenerate = std::getenv("REGEN_GOLDEN") != nullptr;

    for (const std::uint64_t seed : kGoldenSeeds) {
        const GoldenRecord generated = record_random_game(seed);
        if (regenerate) {
            write_record(generated);
        }

        const GoldenRecord expected = regenerate ? generated : read_record(seed);
        CHECK(generated.seed == expected.seed);
        CHECK(generated.hash == expected.hash);
        REQUIRE(generated.action_count == expected.action_count);
        for (std::size_t i = 0; i < expected.action_count; ++i) {
            CHECK(generated.actions[i] == expected.actions[i]);
        }
        replay_record(expected);
    }
}
