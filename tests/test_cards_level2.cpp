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

// ═══════════════════════════════════════════════════════════════════
// STOREROOM
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Storeroom: +1 Buy, discard-draw, discard-for-coins", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Storeroom", "Copper", "Estate", "Silver"});
    game.set_deck(0, {"Gold", "Gold", "Gold"});
    game.state().start_turn();

    SECTION("discard 2 for draw, then discard 1 for coins") {
        // First discard: indices 0,1 (Copper, Estate)
        // After draw 2: hand = Silver, Gold, Gold
        // Second discard: index 0 (Silver)
        auto decide = TestGame::scripted_decisions({{0, 1}, {0}});
        game.play_card(0, "Storeroom", decide);

        CHECK(game.state().buys() == 2); // 1 + 1
        CHECK(game.state().coins() == 1); // 1 coin from second discard
        CHECK(game.state().get_player(0).hand_size() == 2);
    }

    SECTION("skip both discards") {
        auto decide = TestGame::scripted_decisions({{}, {}});
        game.play_card(0, "Storeroom", decide);

        CHECK(game.state().buys() == 2);
        CHECK(game.state().coins() == 0);
        CHECK(game.state().get_player(0).hand_size() == 3);
    }
}

// ═══════════════════════════════════════════════════════════════════
// CONSPIRATOR
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Conspirator: +2 Coins, conditional +1 Card +1 Action", "[cards][level2]") {
    TestGame game;
    game.state().start_turn();

    SECTION("fewer than 3 actions played — no bonus") {
        game.set_hand(0, {"Conspirator"});
        game.set_deck(0, {"Gold"});

        game.play_card(0, "Conspirator", TestGame::minimal_decisions());

        CHECK(game.state().coins() == 2);
        CHECK(game.state().get_player(0).hand_size() == 0); // no draw
        CHECK(game.state().actions() == 1); // start_turn gives 1, Conspirator adds 0
    }

    SECTION("3+ actions played — bonus triggers") {
        game.set_hand(0, {"Village", "Village", "Conspirator"});
        game.set_deck(0, {"Gold", "Gold", "Gold", "Gold"});

        // Village 1: +2 Actions. Village 2: +2 Actions. Conspirator: 3rd action → bonus +1 Action
        game.play_card(0, "Village", TestGame::minimal_decisions());
        game.play_card(0, "Village", TestGame::minimal_decisions());
        game.play_card(0, "Conspirator", TestGame::minimal_decisions());

        CHECK(game.state().coins() == 2);
        CHECK(game.state().actions() == 6); // 1 + 2 + 2 + 1(bonus)
    }
}

// ═══════════════════════════════════════════════════════════════════
// MENAGERIE
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Menagerie: +1 Action, reveal hand, draw based on uniqueness", "[cards][level2]") {
    TestGame game;
    game.state().start_turn();

    SECTION("all unique names — +3 Cards") {
        game.set_hand(0, {"Menagerie", "Copper", "Silver", "Gold"});
        game.set_deck(0, {"Estate", "Duchy", "Province"});

        game.play_card(0, "Menagerie", TestGame::minimal_decisions());

        CHECK(game.state().actions() == 2); // 1 + 1
        CHECK(game.state().get_player(0).hand_size() == 6); // 3 original + 3 drawn
    }

    SECTION("duplicates — +1 Card") {
        game.set_hand(0, {"Menagerie", "Copper", "Copper", "Silver"});
        game.set_deck(0, {"Gold"});

        game.play_card(0, "Menagerie", TestGame::minimal_decisions());

        CHECK(game.state().actions() == 2); // 1 + 1
        CHECK(game.state().get_player(0).hand_size() == 4); // 3 original + 1 drawn
    }
}

// ═══════════════════════════════════════════════════════════════════
// OASIS
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Oasis: +1 Card +1 Action +1 Coin, discard 1", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Oasis", "Copper", "Estate"});
    game.set_deck(0, {"Gold"});
    game.state().start_turn();

    auto decide = TestGame::scripted_decisions({{0}}); // discard Copper
    game.play_card(0, "Oasis", decide);

    CHECK(game.state().actions() == 2); // 1 + 1
    CHECK(game.state().coins() == 1);
    CHECK(game.state().get_player(0).hand_size() == 2); // Estate + Gold drawn - Copper discarded
}

// ═══════════════════════════════════════════════════════════════════
// KING'S COURT
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("King's Court: play an action three times", "[cards][level2]") {
    TestGame game;
    game.state().start_turn();

    SECTION("KC + Smithy = draw 9") {
        game.set_hand(0, {"King's Court", "Smithy"});
        game.set_deck(0, std::vector<std::string>(10, "Copper"));

        auto decide = TestGame::scripted_decisions({{0}}); // pick Smithy
        game.play_card(0, "King's Court", decide);

        CHECK(game.state().get_player(0).hand_size() == 9);
    }

    SECTION("KC with no actions — does nothing") {
        game.set_hand(0, {"King's Court", "Copper"});
        game.play_card(0, "King's Court", TestGame::minimal_decisions());
        CHECK(game.state().get_player(0).hand_size() == 1); // just Copper
    }
}

// ═══════════════════════════════════════════════════════════════════
// COURIER
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Courier: +1 Coin, discard top, play Action/Treasure from discard", "[cards][level2]") {
    TestGame game;
    game.state().start_turn();

    SECTION("play a treasure from discard") {
        game.set_hand(0, {"Courier"});
        game.set_deck(0, {"Estate"}); // top card discarded
        game.set_discard(0, {"Silver"});

        auto decide = TestGame::scripted_decisions({{0}}); // play Silver from discard
        game.play_card(0, "Courier", decide);

        CHECK(game.state().coins() == 3); // 1 from Courier + 2 from Silver
        // Silver should be in play, not in discard
        CHECK(contains(game.in_play_names(0), "Silver"));
    }

    SECTION("victory card in discard is not offered") {
        game.set_hand(0, {"Courier"});
        game.set_deck(0, {"Copper"}); // discarded to discard pile
        game.set_discard(0, {"Estate"}); // not playable

        // After discarding Copper, discard has Estate + Copper. Only Copper is playable.
        auto decide = TestGame::scripted_decisions({{0}}); // play Copper
        game.play_card(0, "Courier", decide);

        CHECK(game.state().coins() == 2); // 1 + 1 from Copper
    }
}

// ═══════════════════════════════════════════════════════════════════
// SENTINEL
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Sentinel: look at top 5, trash up to 2, order rest", "[cards][level2]") {
    TestGame game;

    SECTION("trash 2, order remaining 3") {
        game.set_hand(0, {"Sentinel"});
        game.set_deck(0, {"Copper", "Estate", "Silver", "Gold", "Province"});
        // Top of deck: Province, Gold, Silver, Estate, Copper

        // Trash: pick first 2 card_ids offered (Province, Gold)
        // Order: pick first each time
        auto decide = TestGame::greedy_decisions();
        game.play_card(0, "Sentinel", decide);

        CHECK(game.state().get_trash().size() == 2);
        CHECK(game.state().get_player(0).deck_size() == 3);
    }

    SECTION("trash 0") {
        game.set_hand(0, {"Sentinel"});
        game.set_deck(0, {"Gold", "Gold", "Gold"});

        auto decide = TestGame::scripted_decisions({{}}); // trash nothing
        game.play_card(0, "Sentinel", decide);

        CHECK(game.state().get_trash().size() == 0);
        CHECK(game.state().get_player(0).deck_size() == 3);
    }
}

// ═══════════════════════════════════════════════════════════════════
// BAZAAR
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Bazaar: +1 Card +2 Actions +1 Coin", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Bazaar"});
    game.set_deck(0, {"Gold"});
    game.state().start_turn();

    game.play_card(0, "Bazaar", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 1); // drew 1
    CHECK(game.state().actions() == 3); // 1 + 2
    CHECK(game.state().coins() == 1);
}

// ═══════════════════════════════════════════════════════════════════
// JUNK DEALER
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Junk Dealer: +1 Card +1 Action +1 Coin, trash 1", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Junk Dealer", "Estate", "Copper"});
    game.set_deck(0, {"Gold"});
    game.state().start_turn();

    auto decide = TestGame::scripted_decisions({{0}}); // trash first card (Estate)
    game.play_card(0, "Junk Dealer", decide);

    CHECK(game.state().actions() == 2); // 1 + 1
    CHECK(game.state().coins() == 1);
    CHECK(game.state().get_trash().size() == 1);
    CHECK(game.state().get_player(0).hand_size() == 2); // Copper + Gold drawn
}

// ═══════════════════════════════════════════════════════════════════
// WAREHOUSE
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Warehouse: +3 Cards +1 Action, discard 3", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Warehouse", "Copper"});
    game.set_deck(0, {"Gold", "Gold", "Gold"});
    game.state().start_turn();

    auto decide = TestGame::scripted_decisions({{0, 1, 2}}); // discard first 3
    game.play_card(0, "Warehouse", decide);

    CHECK(game.state().actions() == 2); // 1 + 1
    CHECK(game.state().get_player(0).hand_size() == 1); // Copper + 3 drawn - 3 discarded
    CHECK(game.state().get_player(0).discard_size() == 3);
}

// ═══════════════════════════════════════════════════════════════════
// PILGRIM
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Pilgrim: +4 Cards, topdeck 1", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Pilgrim", "Estate"});
    game.set_deck(0, {"Gold", "Silver", "Copper", "Duchy"});

    auto decide = TestGame::scripted_decisions({{0}}); // topdeck first card
    game.play_card(0, "Pilgrim", decide);

    CHECK(game.state().get_player(0).hand_size() == 4); // Estate + 4 drawn - 1 topdecked
    CHECK(game.state().get_player(0).deck_size() == 1); // topdecked card
}

// ═══════════════════════════════════════════════════════════════════
// ALTAR
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Altar: trash from hand, gain costing up to 5", "[cards][level2]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Altar", "Copper"});

    auto decide = TestGame::scripted_decisions({{0}, {0}}); // trash Copper, gain first option
    game.play_card(0, "Altar", decide);

    CHECK(game.state().get_trash().size() == 1);
    CHECK(game.state().get_player(0).discard_size() == 1); // gained card
}

// ═══════════════════════════════════════════════════════════════════
// ARMORY
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Armory: gain to deck costing up to 4", "[cards][level2]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Armory"});

    auto decide = TestGame::scripted_decisions({{0}}); // gain first option
    game.play_card(0, "Armory", decide);

    CHECK(game.state().get_player(0).deck_size() >= 1); // gained to deck top
    CHECK(game.state().get_player(0).discard_size() == 0); // not to discard
}

// ═══════════════════════════════════════════════════════════════════
// EXPAND
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Expand: trash from hand, gain costing up to 3 more", "[cards][level2]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Expand", "Silver"}); // Silver costs 3

    auto decide = TestGame::scripted_decisions({{0}, {0}}); // trash Silver, gain first option <= 6
    game.play_card(0, "Expand", decide);

    CHECK(game.state().get_trash().size() == 1);
    CHECK(game.state().get_player(0).discard_size() == 1);
}

// ═══════════════════════════════════════════════════════════════════
// SALVAGER
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Salvager: +1 Buy, trash for coins", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Salvager", "Gold"}); // Gold costs 6
    game.state().start_turn();

    auto decide = TestGame::scripted_decisions({{0}}); // trash Gold
    game.play_card(0, "Salvager", decide);

    CHECK(game.state().buys() == 2); // 1 + 1
    CHECK(game.state().coins() == 6); // Gold's cost
    CHECK(game.state().get_trash().size() == 1);
}

// ═══════════════════════════════════════════════════════════════════
// UPGRADE
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Upgrade: +1 Card +1 Action, trash, gain exactly 1 more", "[cards][level2]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Upgrade", "Estate"}); // Estate costs 2
    game.set_deck(0, {"Gold"});
    game.state().start_turn();

    auto decide = TestGame::scripted_decisions({{0}, {0}}); // trash Estate, gain first cost-3 card
    game.play_card(0, "Upgrade", decide);

    CHECK(game.state().actions() == 2); // 1 + 1
    CHECK(game.state().get_trash().size() == 1);
    CHECK(game.state().get_player(0).discard_size() == 1);
}

// ═══════════════════════════════════════════════════════════════════
// SHANTY TOWN
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Shanty Town: +2 Actions, draw 2 if no Actions in hand", "[cards][level2]") {
    TestGame game;
    game.state().start_turn();

    SECTION("no actions in hand — draw 2") {
        game.set_hand(0, {"Shanty Town", "Copper", "Estate"});
        game.set_deck(0, {"Gold", "Silver"});

        game.play_card(0, "Shanty Town", TestGame::minimal_decisions());

        CHECK(game.state().actions() == 3); // 1 + 2
        CHECK(game.state().get_player(0).hand_size() == 4); // Copper, Estate + 2 drawn
    }

    SECTION("action in hand — no draw") {
        game.set_hand(0, {"Shanty Town", "Village", "Copper"});
        game.set_deck(0, {"Gold"});

        game.play_card(0, "Shanty Town", TestGame::minimal_decisions());

        CHECK(game.state().actions() == 3); // 1 + 2
        CHECK(game.state().get_player(0).hand_size() == 2); // Village, Copper — no draw
    }
}

// ═══════════════════════════════════════════════════════════════════
// MAGNATE
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Magnate: draw 1 per Treasure in hand", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Magnate", "Copper", "Silver", "Gold"});
    game.set_deck(0, {"Estate", "Estate", "Estate"});

    game.play_card(0, "Magnate", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 6); // 3 original + 3 drawn (3 treasures)
}

// ═══════════════════════════════════════════════════════════════════
// SEA CHART
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Sea Chart: +1 Card +1 Action, reveal top, copy in play to hand", "[cards][level2]") {
    TestGame game;
    game.state().start_turn();

    SECTION("copy in play — put in hand") {
        // Sea Chart is in play after playing it. Top card is another Sea Chart.
        game.set_hand(0, {"Sea Chart"});
        game.set_deck(0, {"Gold", "Sea Chart"}); // Sea Chart on top after draw
        // After +1 Card (draws Gold), top of deck is now Sea Chart
        // Sea Chart is in play → put it in hand

        game.play_card(0, "Sea Chart", TestGame::minimal_decisions());

        CHECK(game.state().actions() == 2); // 1 + 1
        CHECK(contains(game.hand_names(0), "Sea Chart")); // Sea Chart from deck in hand
    }

    SECTION("no copy in play — stays on deck") {
        game.set_hand(0, {"Sea Chart"});
        game.set_deck(0, {"Estate", "Gold"}); // Gold drawn, Estate revealed

        game.play_card(0, "Sea Chart", TestGame::minimal_decisions());

        CHECK(game.state().get_player(0).hand_size() == 1); // just Gold drawn
        CHECK(game.state().get_player(0).deck_size() == 1); // Estate stays
    }
}

// ═══════════════════════════════════════════════════════════════════
// CUTPURSE
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Cutpurse: +2 Coins, others discard Copper", "[cards][level2]") {
    TestGame game;
    game.state().start_turn();

    SECTION("opponent has Copper") {
        game.set_hand(0, {"Cutpurse"});
        game.set_hand(1, {"Copper", "Silver"});

        game.play_card(0, "Cutpurse", TestGame::minimal_decisions());

        CHECK(game.state().coins() == 2);
        CHECK(game.state().get_player(1).hand_size() == 1); // Silver remains
    }

    SECTION("opponent has no Copper") {
        game.set_hand(0, {"Cutpurse"});
        game.set_hand(1, {"Silver", "Gold"});

        game.play_card(0, "Cutpurse", TestGame::minimal_decisions());

        CHECK(game.state().coins() == 2);
        CHECK(game.state().get_player(1).hand_size() == 2); // no Copper to discard
    }
}

// ═══════════════════════════════════════════════════════════════════
// OLD WITCH
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Old Witch: +3 Cards, others gain Curse, may trash Curse from hand", "[cards][level2]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Old Witch"});
    game.set_deck(0, {"Gold", "Gold", "Gold"});
    game.set_hand(1, {"Curse", "Copper"}); // already has a Curse

    // Opponent gains Curse, then YES (1) to trash a Curse from hand
    auto decide = TestGame::scripted_decisions({{1}});
    game.play_card(0, "Old Witch", decide);

    CHECK(game.state().get_player(0).hand_size() == 3); // drew 3
    CHECK(game.state().get_trash().size() == 1); // opponent trashed a Curse
}

// ═══════════════════════════════════════════════════════════════════
// SOOTHSAYER
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Soothsayer: gain Gold, others gain Curse and draw if they did", "[cards][level2]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Soothsayer"});
    game.set_deck(1, {"Silver"});

    game.play_card(0, "Soothsayer", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).discard_size() == 1); // gained Gold
    // Opponent gained Curse AND drew a card
    CHECK(game.state().get_player(1).discard_size() == 1); // Curse
    CHECK(game.state().get_player(1).hand_size() == 1); // drew Silver
}

// ═══════════════════════════════════════════════════════════════════
// CARTOGRAPHER
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Cartographer: +1 Card +1 Action, look at 4, discard/order", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Cartographer"});
    game.set_deck(0, {"Copper", "Estate", "Silver", "Gold", "Province"});
    game.state().start_turn();
    // After +1 Card (draws Province), look at top 4: Gold, Silver, Estate, Copper

    // greedy_decisions discards all 4, leaving 0 to order
    auto decide = TestGame::greedy_decisions();
    game.play_card(0, "Cartographer", decide);

    CHECK(game.state().actions() == 2); // 1 + 1
    CHECK(game.state().get_player(0).discard_size() == 4);
    CHECK(game.state().get_player(0).deck_size() == 0);
}

// ═══════════════════════════════════════════════════════════════════
// WANDERING MINSTREL
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Wandering Minstrel: +1 Card +2 Actions, reveal 3, Actions back, rest discard", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Wandering Minstrel"});
    game.set_deck(0, {"Copper", "Village", "Smithy", "Gold"});
    game.state().start_turn();
    // Draws Gold. Reveals: Smithy, Village, Copper. Actions (Smithy, Village) go back, Copper discarded.

    game.play_card(0, "Wandering Minstrel", TestGame::minimal_decisions());

    CHECK(game.state().actions() == 3); // 1 + 2
    CHECK(game.state().get_player(0).deck_size() == 2); // Village, Smithy
    CHECK(game.state().get_player(0).discard_size() == 1); // Copper
}

// ═══════════════════════════════════════════════════════════════════
// SEER
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Seer: +1 Card +1 Action, cost 2-4 to hand, rest back", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Seer"});
    // Costs: Copper=0, Estate=2, Silver=3, Province=8
    game.set_deck(0, {"Copper", "Province", "Estate", "Silver", "Gold"});
    game.state().start_turn();
    // Draws Gold. Reveals top 3: Silver(3), Estate(2), Province(8)
    // Silver and Estate (cost 2-4) → hand. Province → back on deck.

    game.play_card(0, "Seer", TestGame::minimal_decisions());

    CHECK(game.state().actions() == 2); // 1 + 1
    CHECK(game.state().get_player(0).hand_size() == 3); // Gold + Silver + Estate
    CHECK(game.state().get_player(0).deck_size() == 2); // Province + Copper
}

// ═══════════════════════════════════════════════════════════════════
// PATROL
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Patrol: +3 Cards, reveal 4, Victory/Curse to hand, rest back", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Patrol"});
    // Province=Victory, Gold=Treasure, Estate=Victory, Curse=Curse, Silver, Copper, Duchy
    game.set_deck(0, {"Copper", "Silver", "Curse", "Estate", "Gold", "Province", "Duchy"});
    // After +3 Cards (Duchy, Province, Gold): reveals top 4: Estate, Curse, Silver, Copper
    // Victory+Curse (Estate, Curse) → hand. Silver, Copper → back on deck.

    game.play_card(0, "Patrol", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 5); // Duchy+Province+Gold (drawn) + Estate+Curse (revealed)
    CHECK(game.state().get_player(0).deck_size() == 2); // Silver, Copper back on deck
}

// ═══════════════════════════════════════════════════════════════════
// FORTUNE HUNTER
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Fortune Hunter: +2 Coins, play Treasure from top 3", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Fortune Hunter"});
    game.set_deck(0, {"Estate", "Silver", "Village"});
    game.state().start_turn();
    // Top 3: Village, Silver, Estate. Only Silver is Treasure.

    auto decide = TestGame::scripted_decisions({{0}}); // play the Silver
    game.play_card(0, "Fortune Hunter", decide);

    CHECK(game.state().coins() == 4); // 2 + 2 from Silver
    CHECK(game.state().get_player(0).deck_size() == 2); // Village, Estate back
    CHECK(contains(game.in_play_names(0), "Silver"));
}

// ═══════════════════════════════════════════════════════════════════
// CARNIVAL
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Carnival: reveal top 4, one of each name to hand", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Carnival"});
    game.set_deck(0, {"Copper", "Copper", "Silver", "Gold"});
    // Top 4: Gold, Silver, Copper, Copper. Unique names: Gold, Silver, Copper → hand. Second Copper → discard.

    game.play_card(0, "Carnival", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 3); // Gold, Silver, Copper
    CHECK(game.state().get_player(0).discard_size() == 1); // duplicate Copper
}

// ═══════════════════════════════════════════════════════════════════
// SAGE
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Sage: +1 Action, reveal until cost >= 3", "[cards][level2]") {
    TestGame game;
    game.state().start_turn();

    SECTION("first card costs 3+ — goes to hand immediately") {
        game.set_hand(0, {"Sage"});
        game.set_deck(0, {"Copper", "Silver"}); // Silver on top (cost 3)

        game.play_card(0, "Sage", TestGame::minimal_decisions());

        CHECK(game.state().actions() == 2); // 1 + 1
        CHECK(contains(game.hand_names(0), "Silver"));
        CHECK(game.state().get_player(0).discard_size() == 0);
    }

    SECTION("discards cheap cards before finding expensive one") {
        game.set_hand(0, {"Sage"});
        game.set_deck(0, {"Gold", "Estate", "Copper"}); // Copper(0) on top, Estate(2), Gold(6) at bottom
        // Reveals: Copper(0)→discard, Estate(2)→discard, Gold(6)→hand

        game.play_card(0, "Sage", TestGame::minimal_decisions());

        CHECK(contains(game.hand_names(0), "Gold"));
        CHECK(game.state().get_player(0).discard_size() == 2); // Copper, Estate
    }
}

// ═══════════════════════════════════════════════════════════════════
// HUNTING PARTY
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Hunting Party: +1 Card +1 Action, reveal until non-copy", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Hunting Party", "Copper", "Estate"});
    game.set_deck(0, {"Gold", "Estate", "Copper", "Silver"});
    game.state().start_turn();
    // Draws Silver. Hand: Copper, Estate, Silver. Reveals: Copper(copy), Estate(copy), Gold(not copy) → hand.

    game.play_card(0, "Hunting Party", TestGame::minimal_decisions());

    CHECK(game.state().actions() == 2); // 1 + 1
    CHECK(contains(game.hand_names(0), "Gold"));
    CHECK(game.state().get_player(0).discard_size() == 2); // Copper, Estate discarded
}

// ═══════════════════════════════════════════════════════════════════
// STABLES
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Stables: may discard Treasure for +3 Cards +1 Action", "[cards][level2]") {
    TestGame game;
    game.state().start_turn();

    SECTION("discard a Treasure") {
        game.set_hand(0, {"Stables", "Silver", "Estate"});
        game.set_deck(0, {"Gold", "Gold", "Gold"});

        auto decide = TestGame::scripted_decisions({{0}}); // discard Silver
        game.play_card(0, "Stables", decide);

        CHECK(game.state().actions() == 2); // 1 + 1 from Stables
        CHECK(game.state().get_player(0).hand_size() == 4); // Estate + 3 drawn
    }

    SECTION("choose not to discard") {
        game.set_hand(0, {"Stables", "Silver"});

        auto decide = TestGame::scripted_decisions({{}}); // skip
        game.play_card(0, "Stables", decide);

        CHECK(game.state().actions() == 1); // no +Action when skipping
        CHECK(game.state().get_player(0).hand_size() == 1); // Silver unchanged
    }
}

// ═══════════════════════════════════════════════════════════════════
// MOUNTAIN VILLAGE
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Mountain Village: +2 Actions, card from discard or draw", "[cards][level2]") {
    TestGame game;
    game.state().start_turn();

    SECTION("discard has cards — pick one to hand") {
        game.set_hand(0, {"Mountain Village"});
        game.set_discard(0, {"Gold", "Silver"});

        auto decide = TestGame::scripted_decisions({{0}}); // pick first card from discard
        game.play_card(0, "Mountain Village", decide);

        CHECK(game.state().actions() == 3); // 1 + 2
        CHECK(game.state().get_player(0).hand_size() == 1);
        CHECK(game.state().get_player(0).discard_size() == 1);
    }

    SECTION("empty discard — draw 1") {
        game.set_hand(0, {"Mountain Village"});
        game.set_deck(0, {"Gold"});

        game.play_card(0, "Mountain Village", TestGame::minimal_decisions());

        CHECK(game.state().actions() == 3); // 1 + 2
        CHECK(game.state().get_player(0).hand_size() == 1); // drew Gold
    }
}

// ═══════════════════════════════════════════════════════════════════
// SCAVENGER
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Scavenger: +2 Coins, may dump deck, topdeck from discard", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Scavenger"});
    game.set_deck(0, {"Gold", "Silver"});
    game.set_discard(0, {"Estate"});
    game.state().start_turn();

    // YES to dump deck, then pick Estate from discard to topdeck
    auto decide = TestGame::scripted_decisions({{1}, {0}});
    game.play_card(0, "Scavenger", decide);

    CHECK(game.state().coins() == 2);
    CHECK(game.state().get_player(0).discard_size() == 2); // Gold, Silver dumped
    CHECK(game.state().get_player(0).deck_size() == 1); // Estate topdecked
}

// ═══════════════════════════════════════════════════════════════════
// REMAKE
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Remake: do twice - trash, gain exactly 1 more", "[cards][level2]") {
    TestGame game;
    game.setup_supply({"Smithy"}); // Smithy costs 4, gives us a cost-4 target
    game.set_hand(0, {"Remake", "Estate", "Silver"}); // Estate(2), Silver(3)

    // Trash Estate(2), gain cost-3 card (Silver); Trash Silver(3), gain cost-4 card (Smithy)
    auto decide = TestGame::scripted_decisions({{0}, {0}, {0}, {0}});
    game.play_card(0, "Remake", decide);

    CHECK(game.state().get_trash().size() == 2);
    CHECK(game.state().get_player(0).discard_size() == 2); // 2 gained cards
}

// ═══════════════════════════════════════════════════════════════════
// DISMANTLE
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Dismantle: trash, if cost >= 1 gain cheaper + Gold", "[cards][level2]") {
    TestGame game;
    game.setup_supply({});

    SECTION("trash card costing >= 1") {
        game.set_hand(0, {"Dismantle", "Silver"}); // Silver costs 3

        auto decide = TestGame::scripted_decisions({{0}, {0}}); // trash Silver, gain first cheaper
        game.play_card(0, "Dismantle", decide);

        CHECK(game.state().get_trash().size() == 1);
        CHECK(game.state().get_player(0).discard_size() == 2); // cheaper card + Gold
    }

    SECTION("trash Copper (cost 0) — no bonus") {
        game.set_hand(0, {"Dismantle", "Copper"});

        auto decide = TestGame::scripted_decisions({{0}});
        game.play_card(0, "Dismantle", decide);

        CHECK(game.state().get_trash().size() == 1);
        CHECK(game.state().get_player(0).discard_size() == 0); // no gain
    }
}

// ═══════════════════════════════════════════════════════════════════
// TRADING POST
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Trading Post: trash 2 from hand, gain Silver to hand", "[cards][level2]") {
    TestGame game;
    game.setup_supply({});

    SECTION("trash 2 — gain Silver to hand") {
        game.set_hand(0, {"Trading Post", "Copper", "Estate"});

        auto decide = TestGame::scripted_decisions({{0, 1}});
        game.play_card(0, "Trading Post", decide);

        CHECK(game.state().get_trash().size() == 2);
        CHECK(contains(game.hand_names(0), "Silver"));
    }

    SECTION("only 1 card in hand — trash it, no Silver") {
        game.set_hand(0, {"Trading Post", "Copper"});

        auto decide = TestGame::scripted_decisions({{0}});
        game.play_card(0, "Trading Post", decide);

        CHECK(game.state().get_trash().size() == 1);
        CHECK(game.state().get_player(0).hand_size() == 0); // no Silver gained
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARQUIS
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Marquis: +1 Buy, draw per card in hand, discard to 10", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Marquis", "Copper", "Estate", "Silver", "Gold"});
    game.set_deck(0, std::vector<std::string>(20, "Copper"));
    game.state().start_turn();

    // Hand after Marquis played: 4 cards. Draw 4 more = 8 cards. No discard needed (< 10).
    game.play_card(0, "Marquis", TestGame::minimal_decisions());

    CHECK(game.state().buys() == 2);
    CHECK(game.state().get_player(0).hand_size() == 8);
}

// ═══════════════════════════════════════════════════════════════════
// TREASURE MAP
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Treasure Map: trash pair, gain 4 Golds to deck", "[cards][level2]") {
    TestGame game;
    game.setup_supply({});

    SECTION("two Treasure Maps — gain 4 Golds") {
        game.set_hand(0, {"Treasure Map", "Treasure Map", "Copper"});

        game.play_card(0, "Treasure Map", TestGame::minimal_decisions());

        CHECK(game.state().get_trash().size() == 2);
        CHECK(game.state().get_player(0).deck_size() == 4); // 4 Golds
        auto deck = game.deck_names(0);
        for (const auto& name : deck) CHECK(name == "Gold");
    }

    SECTION("only one Treasure Map — trashes self, no Golds") {
        game.set_hand(0, {"Treasure Map", "Copper"});

        game.play_card(0, "Treasure Map", TestGame::minimal_decisions());

        CHECK(game.state().get_trash().size() == 1);
        CHECK(game.state().get_player(0).deck_size() == 0);
    }
}

// ═══════════════════════════════════════════════════════════════════
// TRAGIC HERO
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Tragic Hero: +3 Cards +1 Buy, self-trash if 8+ in hand", "[cards][level2]") {
    TestGame game;
    game.setup_supply({});
    game.state().start_turn();

    SECTION("8+ cards — trash self, gain Treasure") {
        game.set_hand(0, {"Tragic Hero", "Copper", "Copper", "Copper", "Copper", "Copper"});
        game.set_deck(0, {"Gold", "Silver", "Estate"});
        // After +3 Cards: hand = 5 Coppers + 3 drawn = 8 cards → triggers

        auto decide = TestGame::scripted_decisions({{0}}); // gain first Treasure
        game.play_card(0, "Tragic Hero", decide);

        CHECK(game.state().buys() == 2);
        CHECK(game.state().get_trash().size() == 1); // Tragic Hero trashed
        CHECK(game.state().get_player(0).discard_size() == 1); // gained Treasure
    }

    SECTION("fewer than 8 cards — no self-trash") {
        game.set_hand(0, {"Tragic Hero", "Copper"});
        game.set_deck(0, {"Gold", "Silver", "Estate"});
        // After +3: hand = Copper + 3 = 4 cards → no trigger

        game.play_card(0, "Tragic Hero", TestGame::minimal_decisions());

        CHECK(game.state().get_trash().size() == 0);
        CHECK(contains(game.in_play_names(0), "Tragic Hero"));
    }
}

// ═══════════════════════════════════════════════════════════════════
// MINION
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Minion: choose +2 Coins or discard/draw/attack", "[cards][level2]") {
    TestGame game;
    game.state().start_turn();

    SECTION("choose +2 Coins") {
        game.set_hand(0, {"Minion"});

        auto decide = TestGame::scripted_decisions({{0}}); // choice 0 = +2 Coins
        game.play_card(0, "Minion", decide);

        CHECK(game.state().actions() == 2); // 1 + 1
        CHECK(game.state().coins() == 2);
    }

    SECTION("choose discard/draw mode") {
        game.set_hand(0, {"Minion", "Copper", "Estate"});
        game.set_deck(0, std::vector<std::string>(5, "Gold"));
        game.set_hand(1, {"Copper", "Copper", "Copper", "Copper", "Copper"}); // 5 cards
        game.set_deck(1, std::vector<std::string>(5, "Silver"));

        auto decide = TestGame::scripted_decisions({{1}}); // choice 1 = discard/draw
        game.play_card(0, "Minion", decide);

        CHECK(game.state().get_player(0).hand_size() == 4); // discarded hand, drew 4
        CHECK(game.state().get_player(1).hand_size() == 4); // had 5, discarded, drew 4
    }

    SECTION("opponent with < 5 cards unaffected") {
        game.set_hand(0, {"Minion"});
        game.set_deck(0, std::vector<std::string>(5, "Gold"));
        game.set_hand(1, {"Copper", "Silver"}); // only 2
        int hand_before = game.state().get_player(1).hand_size();

        auto decide = TestGame::scripted_decisions({{1}});
        game.play_card(0, "Minion", decide);

        CHECK(game.state().get_player(1).hand_size() == hand_before); // unchanged
    }
}

// ═══════════════════════════════════════════════════════════════════
// REPLACE
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Replace: trash, gain up to 2 more with conditional effects", "[cards][level2]") {
    TestGame game;
    game.setup_supply({});
    game.state().start_turn();

    SECTION("gain Action — topdecked") {
        game.set_hand(0, {"Replace", "Estate"}); // Estate costs 2, gain up to 4

        // Trash Estate(0), gain first cost<=4 option
        auto decide = TestGame::scripted_decisions({{0}, {0}});
        game.play_card(0, "Replace", decide);

        CHECK(game.state().get_trash().size() == 1);
        // Gained card should be on deck (Action or Treasure topdeck)
    }
}

// ═══════════════════════════════════════════════════════════════════
// MASQUERADE
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Masquerade: +2 Cards, pass cards, may trash", "[cards][level2]") {
    TestGame game;
    game.set_hand(0, {"Masquerade", "Copper", "Estate"});
    game.set_deck(0, {"Gold", "Silver"});
    game.set_hand(1, {"Silver", "Gold"});

    // +2 Cards → hand = Copper, Estate, Silver, Gold
    // P0 passes index 0 (Copper), P1 passes index 0 (Silver)
    // Then P0 may trash index 0
    auto decide = TestGame::scripted_decisions({{0}, {0}, {0}});
    game.play_card(0, "Masquerade", decide);

    // P0 received P1's Silver, P1 received P0's Copper
    // Then P0 trashed one card
    CHECK(game.state().get_trash().size() == 1);
}

// ═══════════════════════════════════════════════════════════════════
// INTERACTION TESTS
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Throne Room + Treasure Map: second play does nothing", "[cards][level2][interactions]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Throne Room", "Treasure Map", "Treasure Map"});

    auto decide = TestGame::scripted_decisions({{0}}); // pick Treasure Map
    game.play_card(0, "Throne Room", decide);

    // First play: trashes self + hand copy = 2 trashed, gain 4 Golds
    // Second play: Treasure Map already trashed from play, can't trash. No more TMs in hand either.
    CHECK(game.state().get_trash().size() == 2);
    CHECK(game.state().get_player(0).deck_size() == 4); // 4 Golds
}

TEST_CASE("Throne Room + Tragic Hero: second play still gains Treasure", "[cards][level2][interactions]") {
    TestGame game;
    game.setup_supply({});
    game.state().start_turn();
    game.set_hand(0, {"Throne Room", "Tragic Hero", "Copper", "Copper", "Copper", "Copper"});
    game.set_deck(0, std::vector<std::string>(10, "Copper"));

    // Pick Tragic Hero. First play: +3 cards (hand=4+3=7 Coppers), +1 Buy. Hand=7+drawn...
    // Actually hand after TR picks TH: Copper x4. TH plays: +3 cards = 7 cards in hand. 7 < 8 so no trigger on first.
    // Second play: +3 cards = 10 in hand. 10 >= 8 → trash self (already in play? No, TR replays effect).
    // Actually let me recalculate. After TR plays TH from hand, hand = Copper x4.
    // First TH effect: +3 cards → hand = 7. < 8, no trigger.
    // Second TH effect: +3 cards → hand = 10. >= 8, triggers. Trash TH from play, gain Treasure.
    // The gain happens even if TH was already trashed.

    auto decide = TestGame::scripted_decisions({{0}, {0}}); // pick TH, then gain first Treasure
    game.play_card(0, "Throne Room", decide);

    CHECK(game.state().get_trash().size() == 1); // TH trashed once
    CHECK(game.state().get_player(0).discard_size() >= 1); // gained at least one Treasure
}

TEST_CASE("King's Court + Conspirator: actions_played counting", "[cards][level2][interactions]") {
    TestGame game;
    game.state().start_turn();
    game.set_hand(0, {"King's Court", "Conspirator"});
    game.set_deck(0, std::vector<std::string>(10, "Copper"));

    auto decide = TestGame::scripted_decisions({{0}}); // pick Conspirator
    game.play_card(0, "King's Court", decide);

    // KC(1) → Conspirator play 1(2): +2 Coins, < 3 actions, no bonus
    // Conspirator play 2(3): +2 Coins, == 3 actions, bonus +1 Card +1 Action
    // Conspirator play 3(4): +2 Coins, > 3 actions, bonus +1 Card +1 Action
    CHECK(game.state().coins() == 6); // 3 × +2 Coins
    CHECK(game.state().get_player(0).hand_size() == 2); // 2 bonus draws
}

TEST_CASE("Swindler blocked by Moat", "[cards][level2][interactions]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Swindler"});
    game.set_deck(1, {"Gold"});
    game.set_hand(1, {"Moat"});

    auto decide = TestGame::scripted_decisions({{1}}); // reveal Moat
    game.play_card(0, "Swindler", decide);

    CHECK(game.state().coins() == 2);
    CHECK(game.state().get_trash().size() == 0);
    CHECK(game.state().get_player(1).deck_size() == 1);
}

TEST_CASE("Witch's Hut: both Actions discarded triggers Curse", "[cards][level2][interactions]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Witch's Hut", "Village", "Smithy", "Copper"});
    game.set_deck(0, {"Gold", "Gold", "Gold", "Gold"});
    game.state().start_turn();

    // +4 Cards. Discard indices 0, 1 (Village, Smithy — both Actions). Triggers Curse on opponent.
    auto decide = TestGame::scripted_decisions({{0, 1}});
    game.play_card(0, "Witch's Hut", decide);

    CHECK(game.state().get_player(1).discard_size() == 1); // gained Curse
}

TEST_CASE("Witch's Hut: non-Action discarded does not trigger", "[cards][level2][interactions]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Witch's Hut", "Village", "Copper", "Estate"});
    game.set_deck(0, {"Gold", "Gold", "Gold", "Gold"});
    game.state().start_turn();

    // Discard Village(Action) + Copper(Treasure) — not both Actions.
    auto decide = TestGame::scripted_decisions({{0, 1}});
    game.play_card(0, "Witch's Hut", decide);

    CHECK(game.state().get_player(1).discard_size() == 0); // no Curse
}
