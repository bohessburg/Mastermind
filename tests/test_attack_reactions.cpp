#include <catch2/catch_test_macros.hpp>
#include "test_helpers.h"

#include <algorithm>

TEST_CASE("Moat blocks Militia", "[reaction]") {
    TestGame game;
    game.set_hand(0, {"Militia"});
    game.set_hand(1, {"Moat", "Estate", "Estate", "Copper", "Silver"});

    // Reaction decision: reveal Moat (1=yes)
    auto decide = TestGame::scripted_decisions({{1}});
    game.play_card(0, "Militia", decide);

    CHECK(game.state().get_player(1).hand_size() == 5); // unaffected
    CHECK(game.state().coins() == 2); // attacker still gets +2
}

TEST_CASE("Moat decline to reveal", "[reaction]") {
    TestGame game;
    game.set_hand(0, {"Militia"});
    game.set_hand(1, {"Moat", "Estate", "Estate", "Copper", "Silver"});

    // Decline Moat (0=no), then discard 2 (indices 1,2 = Estates)
    auto decide = TestGame::scripted_decisions({{0}, {1, 2}});
    game.play_card(0, "Militia", decide);

    CHECK(game.state().get_player(1).hand_size() == 3);
}

TEST_CASE("Witch blocked by Moat", "[reaction]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Witch"});
    game.set_deck(0, {"Copper", "Silver"});
    game.set_hand(1, {"Moat"});

    auto decide = TestGame::scripted_decisions({{1}}); // reveal Moat
    game.play_card(0, "Witch", decide);

    CHECK(game.state().get_player(0).hand_size() == 2); // drew 2
    // Opponent NOT cursed
    auto p1_discard = game.discard_names(1);
    bool has_curse = std::find(p1_discard.begin(), p1_discard.end(), "Curse") != p1_discard.end();
    CHECK_FALSE(has_curse);
}

TEST_CASE("Militia with opponent who has 3 or fewer cards", "[attack]") {
    TestGame game;
    game.set_hand(0, {"Militia"});
    game.set_hand(1, {"Copper", "Silver", "Gold"});

    game.play_card(0, "Militia", TestGame::minimal_decisions());

    CHECK(game.state().get_player(1).hand_size() == 3); // no change
    CHECK(game.state().coins() == 2);
}

TEST_CASE("Bandit gains Gold and trashes opponent treasure", "[attack]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Bandit"});
    game.set_deck(1, {"Copper", "Silver"}); // Silver on top, Copper below

    game.play_card(0, "Bandit", TestGame::minimal_decisions());

    // Player 0 gained a Gold to discard
    auto p0_discard = game.discard_names(0);
    bool has_gold = std::find(p0_discard.begin(), p0_discard.end(), "Gold") != p0_discard.end();
    CHECK(has_gold);

    // Opponent: Silver trashed, Copper discarded
    CHECK(game.state().get_trash().size() == 1);
    CHECK(game.state().card_name(game.state().get_trash()[0]) == "Silver");
}

TEST_CASE("Bureaucrat gains Silver to deck, opponent topdecks Victory", "[attack]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Bureaucrat"});
    game.set_hand(1, {"Estate", "Copper", "Silver"});

    // Opponent has only 1 Victory card, auto-topdecks
    game.play_card(0, "Bureaucrat", TestGame::minimal_decisions());

    // Player 0: Silver on top of deck
    auto p0_deck = game.deck_names(0);
    CHECK(!p0_deck.empty());
    CHECK(p0_deck.back() == "Silver"); // top = back

    // Opponent: Estate topdecked
    auto p1_deck = game.deck_names(1);
    CHECK(!p1_deck.empty());
    CHECK(p1_deck.back() == "Estate");
    CHECK(game.state().get_player(1).hand_size() == 2);
}
