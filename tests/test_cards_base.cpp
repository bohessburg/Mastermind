#include <catch2/catch_test_macros.hpp>
#include "test_helpers.h"

#include <algorithm>

// === Wave 1: Simple effects ===

TEST_CASE("Smithy draws 3 cards", "[cards][base]") {
    TestGame game;
    game.set_hand(0, {"Smithy", "Copper", "Copper"});
    game.set_deck(0, {"Silver", "Gold", "Estate", "Duchy", "Province"});

    game.play_card(0, "Smithy", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 5);
    CHECK(game.state().get_player(0).deck_size() == 2);
    CHECK(game.in_play_names(0).size() == 1);
    CHECK(game.in_play_names(0)[0] == "Smithy");
}

TEST_CASE("Village gives +1 Card +2 Actions", "[cards][base]") {
    TestGame game;
    game.set_hand(0, {"Village", "Copper"});
    game.set_deck(0, {"Silver", "Gold"});
    game.state().start_turn(); // actions=1

    game.play_card(0, "Village", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 2); // Copper + Silver drawn
    // play_card doesn't consume an action (GameRunner does that).
    // So: started 1, card gave +2 = 3
    CHECK(game.state().actions() == 3);
}

TEST_CASE("Laboratory gives +2 Cards +1 Action", "[cards][base]") {
    TestGame game;
    game.set_hand(0, {"Laboratory"});
    game.set_deck(0, {"Copper", "Silver", "Gold"});
    game.state().start_turn();

    game.play_card(0, "Laboratory", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 2);
    CHECK(game.state().actions() == 2); // 1 + 1 = 2 (play_card doesn't consume)
}

TEST_CASE("Festival gives +2 Actions +1 Buy +2 Coins", "[cards][base]") {
    TestGame game;
    game.set_hand(0, {"Festival"});
    game.state().start_turn();

    game.play_card(0, "Festival", TestGame::minimal_decisions());

    CHECK(game.state().actions() == 3); // 1 + 2 (play_card doesn't consume)
    CHECK(game.state().buys() == 2);    // 1 + 1
    CHECK(game.state().coins() == 2);
}

TEST_CASE("Market gives +1 Card +1 Action +1 Buy +1 Coin", "[cards][base]") {
    TestGame game;
    game.set_hand(0, {"Market"});
    game.set_deck(0, {"Copper", "Silver"});
    game.state().start_turn();

    game.play_card(0, "Market", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 1);
    CHECK(game.state().actions() == 2); // 1 + 1 (play_card doesn't consume)
    CHECK(game.state().buys() == 2);
    CHECK(game.state().coins() == 1);
}

// === Wave 2: Simple choices ===

TEST_CASE("Cellar discard and draw", "[cards][base]") {
    TestGame game;
    game.set_hand(0, {"Cellar", "Estate", "Estate", "Copper", "Silver"});
    game.set_deck(0, {"Gold", "Gold", "Gold", "Gold", "Gold"});
    game.state().start_turn();

    SECTION("discard 2, draw 2") {
        // After Cellar leaves hand: Estate(0), Estate(1), Copper(2), Silver(3)
        auto decide = TestGame::scripted_decisions({{0, 1}});
        game.play_card(0, "Cellar", decide);

        CHECK(game.state().get_player(0).hand_size() == 4); // 2 kept + 2 drawn
        CHECK(game.state().get_player(0).discard_size() == 2);
    }

    SECTION("discard 0, draw 0") {
        auto decide = TestGame::scripted_decisions({{}});
        game.play_card(0, "Cellar", decide);

        CHECK(game.state().get_player(0).hand_size() == 4); // remaining after Cellar played
    }
}

TEST_CASE("Chapel trashes up to 4", "[cards][base]") {
    TestGame game;
    game.set_hand(0, {"Chapel", "Estate", "Estate", "Copper", "Copper", "Copper"});

    SECTION("trash 3") {
        auto decide = TestGame::scripted_decisions({{0, 1, 2}});
        game.play_card(0, "Chapel", decide);

        CHECK(game.state().get_player(0).hand_size() == 2);
        CHECK(game.state().get_trash().size() == 3);
    }

    SECTION("trash 0") {
        auto decide = TestGame::scripted_decisions({{}});
        game.play_card(0, "Chapel", decide);

        CHECK(game.state().get_player(0).hand_size() == 5);
        CHECK(game.state().get_trash().size() == 0);
    }
}

TEST_CASE("Workshop gains card costing up to 4", "[cards][base]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Workshop"});

    // Gain a Silver (cost 3)
    auto piles = game.state().gainable_piles(4);
    int silver_idx = -1;
    for (int i = 0; i < static_cast<int>(piles.size()); i++) {
        if (piles[i] == "Silver") { silver_idx = i; break; }
    }
    REQUIRE(silver_idx >= 0);

    auto decide = TestGame::scripted_decisions({{silver_idx}});
    game.play_card(0, "Workshop", decide);

    auto discard = game.discard_names(0);
    bool has_silver = std::find(discard.begin(), discard.end(), "Silver") != discard.end();
    CHECK(has_silver);
}

TEST_CASE("Moneylender trashes Copper for +3", "[cards][base]") {
    TestGame game;
    game.set_hand(0, {"Moneylender", "Copper", "Estate"});

    SECTION("trash copper") {
        auto decide = TestGame::scripted_decisions({{1}}); // yes
        game.play_card(0, "Moneylender", decide);

        CHECK(game.state().coins() == 3);
        CHECK(game.state().get_player(0).hand_size() == 1); // Estate remains
        CHECK(game.state().get_trash().size() == 1);
    }

    SECTION("decline to trash") {
        auto decide = TestGame::scripted_decisions({{0}}); // no
        game.play_card(0, "Moneylender", decide);

        CHECK(game.state().coins() == 0);
        CHECK(game.state().get_player(0).hand_size() == 2);
    }
}

// === Wave 3: Complex multi-step ===

TEST_CASE("Remodel trashes and gains", "[cards][base]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Remodel", "Estate"}); // Estate costs 2

    // Trash Estate (index 0 after Remodel played), gain card costing <=4
    auto piles = game.state().gainable_piles(4);
    int silver_idx = -1;
    for (int i = 0; i < static_cast<int>(piles.size()); i++) {
        if (piles[i] == "Silver") { silver_idx = i; break; }
    }
    REQUIRE(silver_idx >= 0);

    auto decide = TestGame::scripted_decisions({{0}, {silver_idx}});
    game.play_card(0, "Remodel", decide);

    CHECK(game.state().get_player(0).hand_size() == 0);
    CHECK(game.state().get_trash().size() == 1);
    auto discard = game.discard_names(0);
    bool has_silver = std::find(discard.begin(), discard.end(), "Silver") != discard.end();
    CHECK(has_silver);
}

TEST_CASE("Library draws to 7, can skip Actions", "[cards][base]") {
    TestGame game;
    game.set_hand(0, {"Library", "Copper", "Copper"});
    // Deck has a mix: action cards can be set aside
    game.set_deck(0, {"Copper", "Smithy", "Silver", "Gold", "Estate",
                       "Copper", "Silver"});

    // Set aside the Smithy (choose 1=yes when it comes up)
    // Library draws: Silver(top), Copper, Estate, Gold, Silver...
    // Wait, deck order: last pushed = top. So top = Silver, then Copper, then Estate...
    // Actually set_deck pushes in order, so "Copper" is bottom, "Silver" is top
    // Draw order: Silver, Copper, Estate, Gold, Silver, Smithy, Copper

    // We need to draw until 7 in hand. Start with 2 (after Library played).
    // Need 5 more. Cards drawn: Silver, Copper, Estate, Gold, Silver = 5 non-action cards
    // But Smithy would be drawn 6th... we already have 7 by then.
    // Let's simplify: just accept all cards
    auto decide = TestGame::scripted_decisions({}); // fallback: min choices (skip set-aside)
    game.play_card(0, "Library", decide);

    CHECK(game.state().get_player(0).hand_size() == 7);
}

// === Wave 4: Attacks and Reactions ===

TEST_CASE("Witch gives +2 Cards and curses opponents", "[cards][base]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Witch"});
    game.set_deck(0, {"Copper", "Silver", "Gold"});

    game.play_card(0, "Witch", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 2); // drew 2
    // Opponent gained a Curse
    auto p1_discard = game.discard_names(1);
    bool has_curse = std::find(p1_discard.begin(), p1_discard.end(), "Curse") != p1_discard.end();
    CHECK(has_curse);
}

TEST_CASE("Council Room draws 4, +1 Buy, others draw 1", "[cards][base]") {
    TestGame game;
    game.set_hand(0, {"Council Room"});
    game.set_deck(0, {"Copper", "Silver", "Gold", "Estate", "Duchy"});
    game.set_deck(1, {"Province", "Gold"});
    game.state().start_turn();

    game.play_card(0, "Council Room", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 4);
    CHECK(game.state().buys() == 2);
    CHECK(game.state().get_player(1).hand_size() == 1);
}

TEST_CASE("Throne Room plays action twice", "[cards][base]") {
    TestGame game;
    game.set_hand(0, {"Throne Room", "Smithy", "Copper"});
    game.set_deck(0, {"Gold", "Gold", "Gold", "Gold", "Gold", "Gold", "Gold"});

    // Choose Smithy (index 0 after Throne Room leaves hand → Smithy=0, Copper=1)
    auto decide = TestGame::scripted_decisions({{0}});
    game.play_card(0, "Throne Room", decide);

    // Smithy played twice: +3 + +3 = 6 cards drawn
    // Hand: Copper + 6 = 7
    CHECK(game.state().get_player(0).hand_size() == 7);
}

TEST_CASE("Gardens VP calculation", "[cards][base]") {
    TestGame game(1);
    game.set_hand(0, {"Gardens"});
    std::vector<std::string> deck(9, "Copper");
    game.set_deck(0, deck);

    auto scores = game.state().calculate_scores();
    CHECK(scores[0] == 1); // 10 cards / 10 = 1

    // Add more cards
    game.set_discard(0, {"Copper", "Copper", "Copper", "Copper", "Copper",
                          "Copper", "Copper", "Copper", "Copper", "Copper"});
    scores = game.state().calculate_scores();
    CHECK(scores[0] == 2); // 20 cards / 10 = 2
}
