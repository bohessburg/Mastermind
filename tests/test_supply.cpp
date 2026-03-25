#include <catch2/catch_test_macros.hpp>
#include "test_helpers.h"

TEST_CASE("Supply setup creates correct pile sizes", "[supply]") {
    TestGame game(2);
    game.setup_supply({});

    CHECK(game.state().get_supply().count("Copper") == 46);  // 60 - 14
    CHECK(game.state().get_supply().count("Silver") == 40);
    CHECK(game.state().get_supply().count("Gold") == 30);
    CHECK(game.state().get_supply().count("Estate") == 8);
    CHECK(game.state().get_supply().count("Province") == 8);
    CHECK(game.state().get_supply().count("Curse") == 10);
}

TEST_CASE("Supply gain_from removes top card", "[supply]") {
    TestGame game(2);
    game.setup_supply({});

    int count_before = game.state().get_supply().count("Silver");
    int card_id = game.state().get_supply().gain_from("Silver");
    CHECK(card_id >= 0);
    CHECK(game.state().get_supply().count("Silver") == count_before - 1);
}

TEST_CASE("Supply game over when Province empty", "[supply]") {
    TestGame game(2);
    game.setup_supply({});

    CHECK_FALSE(game.state().get_supply().is_game_over());
    while (game.state().get_supply().count("Province") > 0) {
        game.state().get_supply().gain_from("Province");
    }
    CHECK(game.state().get_supply().is_game_over());
}

TEST_CASE("Supply game over when 3 piles empty", "[supply]") {
    TestGame game(2);
    game.setup_supply({"Smithy", "Village", "Market"});

    // Empty Smithy, Village, Market
    while (game.state().get_supply().count("Smithy") > 0)
        game.state().get_supply().gain_from("Smithy");
    while (game.state().get_supply().count("Village") > 0)
        game.state().get_supply().gain_from("Village");
    CHECK_FALSE(game.state().get_supply().is_game_over());

    while (game.state().get_supply().count("Market") > 0)
        game.state().get_supply().gain_from("Market");
    CHECK(game.state().get_supply().is_game_over());
}
