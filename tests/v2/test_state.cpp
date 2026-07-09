#include "v2/core/defs.h"
#include "v2/core/state.h"

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <string_view>
#include <type_traits>

TEST_CASE("v2 GameState remains a memcpy-safe value type", "[v2][state]") {
    STATIC_REQUIRE(std::is_trivially_copyable_v<GameState>);
    STATIC_REQUIRE(sizeof(GameState) <= 16384U);

    GameState state{};
    state.num_players = 2;
    state.num_slots = 3;
    state.slot_to_def[0] = DEF_COPPER;
    state.slot_to_def[1] = DEF_SILVER;
    state.slot_to_def[2] = DEF_ESTATE;
    state.players[0].hand[0] = 5;
    state.players[0].deck.size = 1;
    state.players[0].deck.cards[0] = 2;
    state.players[1].discard.size = 1;
    state.players[1].discard.cards[0] = 0;
    state.piles[0].base = 0;
    state.piles[0].count = 46;
    state.rng = Xoshiro256pp::seeded(42U);

    GameState clone{};
    std::memcpy(&clone, &state, sizeof(GameState));

    REQUIRE(std::memcmp(&clone, &state, sizeof(GameState)) == 0);
    REQUIRE(clone.num_players == state.num_players);
    REQUIRE(clone.players[0].hand[0] == state.players[0].hand[0]);
    REQUIRE(clone.rng.next() == state.rng.next());
}

TEST_CASE("v2 basic CardDef table has expected card data", "[v2][defs]") {
    REQUIRE(card_def_count() == BASIC_CARD_COUNT);
    REQUIRE(base_card_def_count() == BASIC_CARD_COUNT);
    REQUIRE(card_defs() == base_card_defs());

    const CardDef& copper = card_def(DEF_COPPER);
    REQUIRE(std::string_view(copper.name) == "Copper");
    REQUIRE(copper.cost == Cost{0, 0, 0});
    REQUIRE(copper.types == TYPE_TREASURE);
    REQUIRE(copper.vp == 0);
    REQUIRE(copper.coin_value == 1);

    const CardDef& silver = card_def(DEF_SILVER);
    REQUIRE(std::string_view(silver.name) == "Silver");
    REQUIRE(silver.cost == Cost{3, 0, 0});
    REQUIRE(silver.coin_value == 2);

    const CardDef& gold = card_def(DEF_GOLD);
    REQUIRE(std::string_view(gold.name) == "Gold");
    REQUIRE(gold.cost == Cost{6, 0, 0});
    REQUIRE(gold.coin_value == 3);

    const CardDef& platinum = card_def(DEF_PLATINUM);
    REQUIRE(std::string_view(platinum.name) == "Platinum");
    REQUIRE(platinum.cost == Cost{9, 0, 0});
    REQUIRE(platinum.coin_value == 5);

    const CardDef& potion = card_def(DEF_POTION);
    REQUIRE(std::string_view(potion.name) == "Potion");
    REQUIRE(potion.cost == Cost{4, 0, 0});
    REQUIRE(potion.types == TYPE_TREASURE);
    REQUIRE(potion.coin_value == 0);

    REQUIRE(card_def(DEF_ESTATE).cost == Cost{2, 0, 0});
    REQUIRE(card_def(DEF_ESTATE).vp == 1);
    REQUIRE(card_def(DEF_DUCHY).cost == Cost{5, 0, 0});
    REQUIRE(card_def(DEF_DUCHY).vp == 3);
    REQUIRE(card_def(DEF_PROVINCE).cost == Cost{8, 0, 0});
    REQUIRE(card_def(DEF_PROVINCE).vp == 6);
    REQUIRE(card_def(DEF_COLONY).cost == Cost{11, 0, 0});
    REQUIRE(card_def(DEF_COLONY).vp == 10);
    REQUIRE(card_def(DEF_CURSE).cost == Cost{0, 0, 0});
    REQUIRE(card_def(DEF_CURSE).types == TYPE_CURSE);
    REQUIRE(card_def(DEF_CURSE).vp == -1);
}
