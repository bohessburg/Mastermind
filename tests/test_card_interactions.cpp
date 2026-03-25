#include <catch2/catch_test_macros.hpp>
#include "test_helpers.h"

TEST_CASE("Throne Room + Village gives +2 Cards +4 Actions", "[interaction]") {
    TestGame game;
    game.set_hand(0, {"Throne Room", "Village", "Copper"});
    game.set_deck(0, {"Gold", "Silver", "Estate", "Duchy"});
    game.state().start_turn();

    auto decide = TestGame::scripted_decisions({{0}}); // choose Village
    game.play_card(0, "Throne Room", decide);

    // Village x2: +1 Card x2, +2 Actions x2
    CHECK(game.state().get_player(0).hand_size() == 3); // Copper + 2 drawn
    // play_card doesn't consume actions. TR doesn't consume the sub-card's action either.
    // Start: 1. Village x2 each gives +2 = 1 + 2 + 2 = 5
    CHECK(game.state().actions() == 5);
}

TEST_CASE("Throne Room + Militia attacks and gives +4 Coins", "[interaction]") {
    TestGame game;
    game.set_hand(0, {"Throne Room", "Militia", "Copper"});
    game.set_hand(1, {"Estate", "Estate", "Copper", "Silver", "Gold"});

    // Choose Militia (index 0 after TR leaves)
    // Then opponent discards twice (Militia runs twice)
    // First Militia: discard to 3 → discard 2 (indices 0,1)
    // Second Militia: already at 3, nothing to discard
    auto decide = TestGame::scripted_decisions({{0}, {0, 1}});
    game.play_card(0, "Throne Room", decide);

    CHECK(game.state().coins() == 4); // +2 + +2
    CHECK(game.state().get_player(1).hand_size() == 3);
}

TEST_CASE("Throne Room with no actions in hand does nothing", "[interaction]") {
    TestGame game;
    game.set_hand(0, {"Throne Room", "Copper", "Estate"});

    game.play_card(0, "Throne Room", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 2); // Copper, Estate
    CHECK(game.in_play_names(0).size() == 1); // just Throne Room
}
