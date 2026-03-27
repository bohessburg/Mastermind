#include <catch2/catch_test_macros.hpp>
#include "test_helpers.h"

#include <algorithm>

static bool contains(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

// ═══════════════════════════════════════════════════════════════════
// WORKER'S VILLAGE
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Worker's Village: +1 Card +2 Actions +1 Buy", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Worker's Village", "Copper"});
    game.set_deck(0, {"Silver", "Gold"});
    game.state().start_turn();

    game.play_card(0, "Worker's Village", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 2); // Copper + Silver drawn
    CHECK(game.state().actions() == 3); // 1 + 2
    CHECK(game.state().buys() == 2);    // 1 + 1
}

// ═══════════════════════════════════════════════════════════════════
// COURTYARD
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Courtyard: +3 Cards, topdeck one", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Courtyard", "Copper"});
    game.set_deck(0, {"Silver", "Gold", "Estate", "Duchy"});

    SECTION("topdeck a drawn card") {
        // After +3 Cards: hand = Copper, Estate, Gold, Silver (Courtyard in play)
        // Topdeck index 0 = Copper
        auto decide = TestGame::scripted_decisions({{0}});
        game.play_card(0, "Courtyard", decide);

        CHECK(game.state().get_player(0).hand_size() == 3); // 4 drawn - 1 topdecked + existing Copper - Courtyard = 1 + 3 - 1
        auto top = game.state().get_player(0).peek_deck(1);
        CHECK(game.state().card_name(top[0]) == "Copper");
    }

    SECTION("topdeck the last drawn card") {
        // Topdeck index 3 (Silver, the last card in hand after draw)
        auto decide = TestGame::scripted_decisions({{3}});
        game.play_card(0, "Courtyard", decide);

        CHECK(game.state().get_player(0).hand_size() == 3);
    }
}

TEST_CASE("Courtyard: empty deck draws partial, still topdecks", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Courtyard"});
    game.set_deck(0, {"Silver"});
    // Only 1 card to draw, no discard to shuffle

    auto decide = TestGame::scripted_decisions({{0}});
    game.play_card(0, "Courtyard", decide);

    CHECK(game.state().get_player(0).hand_size() == 0); // drew 1, topdecked 1
    CHECK(game.state().get_player(0).deck_size() == 1);
}

// ═══════════════════════════════════════════════════════════════════
// HAMLET
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Hamlet: +1 Card +1 Action, optional discards for +Action/+Buy", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Hamlet", "Estate", "Copper", "Silver"});
    game.set_deck(0, {"Gold", "Gold"});
    game.state().start_turn();

    SECTION("discard for both +Action and +Buy") {
        // First discard (for +Action): index 0 = Estate
        // After that discard, hand = Copper, Silver, Gold (drew 1)
        // Second discard (for +Buy): index 0 = Copper
        auto decide = TestGame::scripted_decisions({{0}, {0}});
        game.play_card(0, "Hamlet", decide);

        CHECK(game.state().actions() == 3); // 1 + 1 (base) + 1 (discard)
        CHECK(game.state().buys() == 2);    // 1 + 1 (discard)
        CHECK(game.state().get_player(0).hand_size() == 2); // started 3 after Hamlet played, +1 draw, -2 discards
        CHECK(game.state().get_player(0).discard_size() == 2);
    }

    SECTION("discard for +Action only") {
        auto decide = TestGame::scripted_decisions({{0}, {}}); // discard Estate, skip second
        game.play_card(0, "Hamlet", decide);

        CHECK(game.state().actions() == 3);
        CHECK(game.state().buys() == 1);
        CHECK(game.state().get_player(0).hand_size() == 3);
    }

    SECTION("discard for +Buy only (skip +Action)") {
        auto decide = TestGame::scripted_decisions({{}, {0}}); // skip action, discard for buy
        game.play_card(0, "Hamlet", decide);

        CHECK(game.state().actions() == 2); // 1 + 1 (base only, no action discard)
        CHECK(game.state().buys() == 2);    // 1 + 1 (buy discard)
        CHECK(game.state().get_player(0).hand_size() == 3); // 3 after Hamlet + 1 drawn - 1 discarded
        CHECK(game.state().get_player(0).discard_size() == 1);
    }

    SECTION("skip both discards") {
        auto decide = TestGame::scripted_decisions({{}, {}});
        game.play_card(0, "Hamlet", decide);

        CHECK(game.state().actions() == 2); // 1 + 1 (base only)
        CHECK(game.state().buys() == 1);    // no buy discard
        CHECK(game.state().get_player(0).hand_size() == 4); // 3 original + 1 drawn
        CHECK(game.state().get_player(0).discard_size() == 0);
    }

    SECTION("empty hand after Hamlet, no discard offered") {
        TestGame g2;
        g2.set_hand(0, {"Hamlet"});
        g2.set_deck(0, {"Gold"});
        g2.state().start_turn();

        // Draw 1 (Gold). Hand has 1 card. Can discard for +Action.
        // After that, hand is empty. Second discard not offered.
        auto decide = TestGame::scripted_decisions({{0}});
        g2.play_card(0, "Hamlet", decide);

        CHECK(g2.state().actions() == 3); // 1 + 1 + 1
        CHECK(g2.state().get_player(0).hand_size() == 0);
    }
}

// ═══════════════════════════════════════════════════════════════════
// LOOKOUT
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Lookout: +1 Action, look at 3, trash 1, discard 1, keep 1", "[cards][level2]") {
    TestGame game;
    game.state().start_turn();

    SECTION("normal 3-card lookout") {
        game.set_hand(0, {"Lookout"});
        // Deck top→bottom: Gold, Silver, Copper (Gold on top = last in vector)
        game.set_deck(0, {"Copper", "Silver", "Gold"});

        // peek_deck returns [Gold, Silver, Copper] (top first)
        // These are card_ids, not indices. The decide callback receives card_ids.
        // Trash: pick index 0 (first card_id in the options = Gold's id)
        // Discard: pick index 0 of remaining (Silver's id)
        // Copper goes back on deck
        auto decide = TestGame::scripted_decisions({{0}, {0}});
        game.play_card(0, "Lookout", decide);

        CHECK(game.state().actions() == 2); // 1 + 1
        CHECK(game.state().get_trash().size() == 1);
        CHECK(game.state().get_player(0).discard_size() == 1);
        CHECK(game.state().get_player(0).deck_size() == 1);
    }

    SECTION("only 1 card in deck — trash it, nothing else") {
        game.set_hand(0, {"Lookout"});
        game.set_deck(0, {"Estate"});

        auto decide = TestGame::scripted_decisions({{0}});
        game.play_card(0, "Lookout", decide);

        CHECK(game.state().get_trash().size() == 1);
        CHECK(game.state().get_player(0).deck_size() == 0);
        CHECK(game.state().get_player(0).discard_size() == 0);
    }

    SECTION("only 2 cards — trash 1, keep 1 (no discard step)") {
        game.set_hand(0, {"Lookout"});
        game.set_deck(0, {"Copper", "Estate"});

        // Trash one, the other goes back on deck (only 1 left, no discard choice needed when < 2 remain)
        auto decide = TestGame::scripted_decisions({{0}});
        game.play_card(0, "Lookout", decide);

        CHECK(game.state().get_trash().size() == 1);
        CHECK(game.state().get_player(0).deck_size() == 1);
    }

    SECTION("empty deck — nothing happens") {
        game.set_hand(0, {"Lookout"});

        game.play_card(0, "Lookout", TestGame::minimal_decisions());

        CHECK(game.state().actions() == 2);
        CHECK(game.state().get_trash().size() == 0);
    }
}

// ═══════════════════════════════════════════════════════════════════
// SWINDLER
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Swindler: +2 Coins, trash opponent's top card, give replacement", "[cards][level2]") {
    TestGame game;
    game.setup_supply({});

    SECTION("trash Copper, attacker picks replacement at cost 0") {
        game.set_hand(0, {"Swindler"});
        game.set_deck(1, {"Silver", "Copper"}); // Copper on top

        // Copper trashed (cost 0). Attacker picks from cost-0 piles.
        // Just pick first option — could be Copper or Curse.
        auto decide = TestGame::scripted_decisions({{0}});
        game.play_card(0, "Swindler", decide);

        CHECK(game.state().coins() == 2);
        CHECK(game.state().get_trash().size() == 1);
        // Opponent gained something costing 0
        CHECK(game.state().get_player(1).discard_size() == 1);
    }

    SECTION("opponent has empty deck — nothing happens") {
        game.set_hand(0, {"Swindler"});
        // Player 1 has no deck or discard

        game.play_card(0, "Swindler", TestGame::minimal_decisions());

        CHECK(game.state().coins() == 2);
        CHECK(game.state().get_trash().size() == 0);
    }

    SECTION("blocked by Moat") {
        game.set_hand(0, {"Swindler"});
        game.set_deck(1, {"Gold"});
        game.set_hand(1, {"Moat"});

        auto decide = TestGame::scripted_decisions({{1}}); // reveal Moat
        game.play_card(0, "Swindler", decide);

        CHECK(game.state().coins() == 2);
        CHECK(game.state().get_trash().size() == 0);
        CHECK(game.state().get_player(1).deck_size() == 1); // untouched
    }
}

// ═══════════════════════════════════════════════════════════════════
// SCHOLAR
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Scholar: discard hand, draw 7", "[cards][level2]") {
    TestGame game;

    SECTION("normal hand") {
        game.set_hand(0, {"Scholar", "Copper", "Estate", "Silver"});
        game.set_deck(0, {"Gold", "Gold", "Gold", "Gold", "Gold", "Gold", "Gold", "Gold"});

        game.play_card(0, "Scholar", TestGame::minimal_decisions());

        // Scholar in play. Hand was Copper, Estate, Silver → discarded.
        // Then draw 7.
        CHECK(game.state().get_player(0).hand_size() == 7);
        CHECK(game.state().get_player(0).discard_size() == 3); // the 3 discarded cards
    }

    SECTION("empty hand before Scholar") {
        game.set_hand(0, {"Scholar"});
        game.set_deck(0, {"Gold", "Gold", "Gold", "Gold", "Gold", "Gold", "Gold"});

        game.play_card(0, "Scholar", TestGame::minimal_decisions());

        // No hand to discard (Scholar already played), draw 7
        CHECK(game.state().get_player(0).hand_size() == 7);
        CHECK(game.state().get_player(0).discard_size() == 0);
    }

    SECTION("not enough cards to draw 7") {
        game.set_hand(0, {"Scholar", "Copper"});
        game.set_deck(0, {"Gold", "Silver"});
        // Copper discarded, then draw 7 but only 2 available + 1 from discard after shuffle = 3

        game.set_seed(42);
        game.play_card(0, "Scholar", TestGame::minimal_decisions());

        // Discarded Copper, shuffled it back with deck, drew as many as possible
        CHECK(game.state().get_player(0).hand_size() == 3); // Gold, Silver, Copper
    }
}

// ═══════════════════════════════════════════════════════════════════
// INTERACTIONS
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Throne Room + Scholar = discard hand twice, draw 14", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Throne Room", "Scholar", "Copper", "Estate"});
    game.set_deck(0, std::vector<std::string>(15, "Gold"));

    auto decide = TestGame::scripted_decisions({{0}}); // pick Scholar
    game.play_card(0, "Throne Room", decide);

    // First Scholar: discard Copper+Estate (2 cards), draw 7
    // Second Scholar: discard those 7, draw 7
    CHECK(game.state().get_player(0).hand_size() == 7);
}

TEST_CASE("Throne Room + Hamlet = double discard opportunities", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Throne Room", "Hamlet", "Estate", "Estate", "Copper", "Copper"});
    game.set_deck(0, {"Gold", "Gold", "Gold", "Gold"});
    game.state().start_turn();

    // Pick Hamlet. First play: draw 1, discard for +Action (idx 0), discard for +Buy (idx 0)
    // Second play: draw 1, discard for +Action (idx 0), skip +Buy
    auto decide = TestGame::scripted_decisions({{0}, {0}, {0}, {0}, {}});
    game.play_card(0, "Throne Room", decide);

    // Actions: 1 + 1+1 (first Hamlet base+discard) + 1+1 (second Hamlet base+discard) = 5
    CHECK(game.state().actions() == 5);
    // Buys: 1 + 1 (first Hamlet discard) = 2
    CHECK(game.state().buys() == 2);
}
