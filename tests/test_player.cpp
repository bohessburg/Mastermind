#include <catch2/catch_test_macros.hpp>
#include "test_helpers.h"

TEST_CASE("Player draws cards from deck", "[player]") {
    TestGame game(1);
    game.set_deck(0, {"Copper", "Silver", "Gold"});

    game.state().get_player(0).draw_cards(2);
    auto hand = game.hand_names(0);
    REQUIRE(hand.size() == 2);
    // Gold is on top (last pushed), drawn first
    CHECK(hand[0] == "Gold");
    CHECK(hand[1] == "Silver");
    CHECK(game.state().get_player(0).deck_size() == 1);
}

TEST_CASE("Player shuffles discard when deck empty", "[player]") {
    TestGame game(1);
    game.set_discard(0, {"Copper", "Silver", "Gold"});
    game.set_seed(42);

    game.state().get_player(0).draw_cards(2);
    CHECK(game.state().get_player(0).hand_size() == 2);
    CHECK(game.state().get_player(0).deck_size() == 1);
    CHECK(game.state().get_player(0).discard_size() == 0);
}

TEST_CASE("Player peek_deck shows top cards", "[player]") {
    TestGame game(1);
    game.set_deck(0, {"Copper", "Silver", "Gold"});

    auto peeked = game.state().get_player(0).peek_deck(2);
    REQUIRE(peeked.size() == 2);
    CHECK(game.state().card_name(peeked[0]) == "Gold");   // top
    CHECK(game.state().card_name(peeked[1]) == "Silver");  // second
    CHECK(game.state().get_player(0).deck_size() == 3);    // unchanged
}

TEST_CASE("Player remove_deck_top", "[player]") {
    TestGame game(1);
    game.set_deck(0, {"Copper", "Silver"});

    int id = game.state().get_player(0).remove_deck_top();
    CHECK(game.state().card_name(id) == "Silver"); // top = last pushed
    CHECK(game.state().get_player(0).deck_size() == 1);
}

TEST_CASE("Player topdeck_from_hand", "[player]") {
    TestGame game(1);
    game.set_hand(0, {"Copper", "Estate"});
    game.set_deck(0, {"Silver"});

    game.state().get_player(0).topdeck_from_hand(1); // Estate
    CHECK(game.state().get_player(0).hand_size() == 1);
    CHECK(game.state().get_player(0).deck_size() == 2);
    auto peeked = game.state().get_player(0).peek_deck(1);
    CHECK(game.state().card_name(peeked[0]) == "Estate");
}

TEST_CASE("Player find_in_hand", "[player]") {
    TestGame game(1);
    game.set_hand(0, {"Copper", "Silver", "Gold"});
    const auto& hand = game.state().get_player(0).get_hand();

    CHECK(game.state().get_player(0).find_in_hand(hand[1]) == 1);
    CHECK(game.state().get_player(0).find_in_hand(9999) == -1);
}

TEST_CASE("Player all_cards counts everything", "[player]") {
    TestGame game(1);
    game.set_hand(0, {"Copper", "Silver"});
    game.set_deck(0, {"Gold"});
    game.set_discard(0, {"Estate"});

    auto all = game.state().get_player(0).all_cards();
    CHECK(all.size() == 4);
}

TEST_CASE("Player cleanup discards and draws", "[player]") {
    TestGame game(1);
    game.set_hand(0, {"Copper", "Silver"});
    game.set_deck(0, {"Gold", "Estate", "Duchy", "Province", "Copper",
                       "Silver", "Gold"});

    game.state().get_player(0).cleanup();
    CHECK(game.state().get_player(0).hand_size() == 5);
    CHECK(game.state().get_player(0).discard_size() == 2);
}
