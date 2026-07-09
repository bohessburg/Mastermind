#include "v2/core/game.h"
#include "v2/core/setup.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace {

[[nodiscard]] const Pile* find_pile(const GameState& state, DefId def) {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        const Pile& pile = state.piles[i];
        if (pile.mixed_len == 0U && state.slot_to_def[pile.base] == def) {
            return &pile;
        }
        if (pile.mixed_len > 0U && state.slot_to_def[pile.mixed[pile.mixed_len - 1U]] == def) {
            return &pile;
        }
    }
    return nullptr;
}

[[nodiscard]] std::uint8_t zone_count(const OrderedZone& zone, Slot slot) {
    std::uint8_t count = 0;
    for (std::uint8_t i = 0; i < zone.size; ++i) {
        if (zone.cards[i] == slot) {
            ++count;
        }
    }
    return count;
}

[[nodiscard]] std::uint8_t total_owned(const PlayerState& player, Slot slot) {
    return static_cast<std::uint8_t>(player.hand[slot] + zone_count(player.deck, slot)
        + zone_count(player.discard, slot));
}

} // namespace

TEST_CASE("v2 setup initializes two-player basic piles and opening hands", "[v2][setup]") {
    Setup setup{};
    setup.num_players = 2;

    const GameState state = Game::new_game(setup, 1234U);

    REQUIRE(state.num_players == 2U);
    REQUIRE(state.num_piles == 7U);
    REQUIRE(find_pile(state, DEF_COPPER)->count == 46U);
    REQUIRE(find_pile(state, DEF_SILVER)->count == 40U);
    REQUIRE(find_pile(state, DEF_GOLD)->count == 30U);
    REQUIRE(find_pile(state, DEF_ESTATE)->count == 8U);
    REQUIRE(find_pile(state, DEF_DUCHY)->count == 8U);
    REQUIRE(find_pile(state, DEF_PROVINCE)->count == 8U);
    REQUIRE(find_pile(state, DEF_CURSE)->count == 10U);
    REQUIRE(find_pile(state, DEF_PLATINUM) == nullptr);
    REQUIRE(find_pile(state, DEF_COLONY) == nullptr);

    const Slot copper = slot_of(state, DEF_COPPER);
    const Slot estate = slot_of(state, DEF_ESTATE);
    for (PlayerId player = 0; player < state.num_players; ++player) {
        REQUIRE(state.players[player].deck.size == 5U);
        REQUIRE(total_owned(state.players[player], copper) == 7U);
        REQUIRE(total_owned(state.players[player], estate) == 3U);
        std::uint8_t hand_size = 0;
        for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
            hand_size = static_cast<std::uint8_t>(hand_size + state.players[player].hand[slot]);
        }
        REQUIRE(hand_size == 5U);
    }
}

TEST_CASE("v2 setup uses four-player and colony pile counts", "[v2][setup]") {
    Setup setup{};
    setup.num_players = 4;
    setup.use_colony_platinum = true;

    const GameState state = Game::new_game(setup, 5678U);

    REQUIRE(state.num_players == 4U);
    REQUIRE(state.num_piles == 9U);
    REQUIRE(find_pile(state, DEF_COPPER)->count == 32U);
    REQUIRE(find_pile(state, DEF_ESTATE)->count == 12U);
    REQUIRE(find_pile(state, DEF_DUCHY)->count == 12U);
    REQUIRE(find_pile(state, DEF_PROVINCE)->count == 12U);
    REQUIRE(find_pile(state, DEF_CURSE)->count == 30U);
    REQUIRE(find_pile(state, DEF_PLATINUM)->count == 12U);
    REQUIRE(find_pile(state, DEF_COLONY)->count == 12U);
}
