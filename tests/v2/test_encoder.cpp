#include "v2/encode/encoder.h"

#include "v2/core/defs.h"
#include "v2/core/game.h"
#include "v2/drivers/bots.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace {

[[nodiscard]] Setup encoder_setup() noexcept {
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

[[nodiscard]] std::array<float, OBS_SIZE> encoded(const GameState& state, PlayerId player) noexcept {
    std::array<float, OBS_SIZE> out{};
    encode(state, player, out.data());
    return out;
}

void step_random(GameState& state, RandomBot& bot) {
    ActionMask legal{};
    const int legal_count = Game::legal_actions(state, legal);
    REQUIRE(legal_count > 0);
    const Action action = bot.choose_action(state, legal, legal_count);
    (void)Game::step(state, action);
}

} // namespace

TEST_CASE("v2 encoder clone output matches original at decision points", "[v2][encode]") {
    GameState state = Game::new_game(encoder_setup(), 0xE4C0'0001ULL);
    RandomBot bot(0xE4C0'1001ULL);

    for (int i = 0; i < 40; ++i) {
        const GameState clone = state;
        CHECK(encoded(clone, 0U) == encoded(state, 0U));
        CHECK(encoded(clone, 1U) == encoded(state, 1U));
        if (state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            break;
        }
        step_random(state, bot);
    }
}

TEST_CASE("v2 encoder is deterministic over identical seeded play", "[v2][encode]") {
    GameState a = Game::new_game(encoder_setup(), 0xE4C0'0002ULL);
    GameState b = Game::new_game(encoder_setup(), 0xE4C0'0002ULL);
    RandomBot bot_a(0xE4C0'2002ULL);
    RandomBot bot_b(0xE4C0'2002ULL);

    for (int i = 0; i < 40; ++i) {
        CHECK(encoded(a, 0U) == encoded(b, 0U));
        CHECK(encoded(a, 1U) == encoded(b, 1U));
        if (a.phase == static_cast<std::uint8_t>(Phase::Over)) {
            break;
        }
        step_random(a, bot_a);
        step_random(b, bot_b);
    }
}

TEST_CASE("v2 encoder excludes opponent hand composition", "[v2][encode]") {
    GameState a = Game::new_game(encoder_setup(), 0xE4C0'0003ULL);
    GameState b = a;

    const Slot copper = slot_of(a, DEF_COPPER);
    const Slot estate = slot_of(a, DEF_ESTATE);
    REQUIRE(copper != NONE);
    REQUIRE(estate != NONE);

    for (std::uint8_t slot = 0; slot < MAX_SLOTS; ++slot) {
        a.players[1].hand[slot] = 0;
        b.players[1].hand[slot] = 0;
    }
    a.players[1].hand[copper] = 5;
    b.players[1].hand[estate] = 5;

    CHECK(encoded(a, 0U) == encoded(b, 0U));
    CHECK(encoded(a, 1U) != encoded(b, 1U));
}

TEST_CASE("v2 encoder exposes version and expected shape", "[v2][encode]") {
    const GameState state = Game::new_game(Setup{}, 0xE4C0'0004ULL);
    const auto obs = encoded(state, 0U);
    CHECK(obs[OBS_META_OFFSET] == static_cast<float>(OBS_VERSION));
    CHECK(obs[OBS_META_OFFSET + 1U] == static_cast<float>(OBS_SIZE));
    CHECK(OBS_SIZE == 1141U);
}
