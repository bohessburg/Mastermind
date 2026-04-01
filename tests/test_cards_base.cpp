#include <catch2/catch_test_macros.hpp>
#include "test_helpers.h"

#include <algorithm>
#include <set>

// Helper to check if a name appears in a name list
static bool contains(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

static int count_of(const std::vector<std::string>& v, const std::string& s) {
    return static_cast<int>(std::count(v.begin(), v.end(), s));
}

// ═══════════════════════════════════════════════════════════════════
// SMITHY
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Smithy: draws 3 cards", "[cards][smithy]") {
    TestGame game;
    game.set_hand(0, {"Smithy", "Copper", "Copper"});
    game.set_deck(0, {"Silver", "Gold", "Estate", "Duchy", "Province"});

    game.play_card(0, "Smithy", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 5);
    CHECK(game.state().get_player(0).deck_size() == 2);
    CHECK(game.in_play_names(0) == std::vector<std::string>{"Smithy"});
}

TEST_CASE("Smithy: draws with shuffle", "[cards][smithy]") {
    TestGame game;
    game.set_hand(0, {"Smithy"});
    game.set_deck(0, {"Gold"});
    game.set_discard(0, {"Silver", "Copper"});
    game.set_seed(42);

    game.play_card(0, "Smithy", TestGame::minimal_decisions());
    // 1 from deck + shuffle discard (2 cards) + draw 2 more = 3 total
    CHECK(game.state().get_player(0).hand_size() == 3);
}

TEST_CASE("Smithy: draws partial when not enough cards", "[cards][smithy]") {
    TestGame game;
    game.set_hand(0, {"Smithy"});
    game.set_deck(0, {"Gold"});
    // No discard to shuffle — can only draw 1

    game.play_card(0, "Smithy", TestGame::minimal_decisions());
    CHECK(game.state().get_player(0).hand_size() == 1);
}

// ═══════════════════════════════════════════════════════════════════
// VILLAGE
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Village: +1 Card +2 Actions", "[cards][village]") {
    TestGame game;
    game.set_hand(0, {"Village", "Copper"});
    game.set_deck(0, {"Silver", "Gold"});
    game.state().start_turn();

    game.play_card(0, "Village", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 2);
    CHECK(game.state().actions() == 3); // 1 + 2
}

// ═══════════════════════════════════════════════════════════════════
// LABORATORY
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Laboratory: +2 Cards +1 Action", "[cards][laboratory]") {
    TestGame game;
    game.set_hand(0, {"Laboratory"});
    game.set_deck(0, {"Copper", "Silver", "Gold"});
    game.state().start_turn();

    game.play_card(0, "Laboratory", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 2);
    CHECK(game.state().actions() == 2);
}

// ═══════════════════════════════════════════════════════════════════
// FESTIVAL
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Festival: +2 Actions +1 Buy +2 Coins", "[cards][festival]") {
    TestGame game;
    game.set_hand(0, {"Festival"});
    game.state().start_turn();

    game.play_card(0, "Festival", TestGame::minimal_decisions());

    CHECK(game.state().actions() == 3);
    CHECK(game.state().buys() == 2);
    CHECK(game.state().coins() == 2);
}

// ═══════════════════════════════════════════════════════════════════
// MARKET
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Market: +1 Card +1 Action +1 Buy +1 Coin", "[cards][market]") {
    TestGame game;
    game.set_hand(0, {"Market"});
    game.set_deck(0, {"Copper", "Silver"});
    game.state().start_turn();

    game.play_card(0, "Market", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 1);
    CHECK(game.state().actions() == 2);
    CHECK(game.state().buys() == 2);
    CHECK(game.state().coins() == 1);
}

// ═══════════════════════════════════════════════════════════════════
// CELLAR
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Cellar: discard N, draw N", "[cards][cellar]") {
    TestGame game;
    game.set_hand(0, {"Cellar", "Estate", "Estate", "Copper", "Silver"});
    game.set_deck(0, {"Gold", "Gold", "Gold", "Gold", "Gold"});
    game.state().start_turn();

    SECTION("discard 2, draw 2") {
        auto decide = TestGame::scripted_decisions({{0, 1}});
        game.play_card(0, "Cellar", decide);

        CHECK(game.state().get_player(0).hand_size() == 4);
        CHECK(game.state().get_player(0).discard_size() == 2);
        CHECK(game.state().actions() == 2); // +1 Action
    }

    SECTION("discard 0, draw 0") {
        auto decide = TestGame::scripted_decisions({{}});
        game.play_card(0, "Cellar", decide);

        CHECK(game.state().get_player(0).hand_size() == 4);
        CHECK(game.state().get_player(0).discard_size() == 0);
    }

    SECTION("discard all") {
        auto decide = TestGame::scripted_decisions({{0, 1, 2, 3}});
        game.play_card(0, "Cellar", decide);

        CHECK(game.state().get_player(0).hand_size() == 4); // drew 4
        CHECK(game.state().get_player(0).discard_size() == 4);
    }
}

// ═══════════════════════════════════════════════════════════════════
// CHAPEL
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Chapel: trash up to 4", "[cards][chapel]") {
    TestGame game;
    game.set_hand(0, {"Chapel", "Estate", "Estate", "Copper", "Copper", "Copper"});

    SECTION("trash 4") {
        auto decide = TestGame::scripted_decisions({{0, 1, 2, 3}});
        game.play_card(0, "Chapel", decide);
        CHECK(game.state().get_player(0).hand_size() == 1);
        CHECK(game.state().get_trash().size() == 4);
    }

    SECTION("trash 0") {
        auto decide = TestGame::scripted_decisions({{}});
        game.play_card(0, "Chapel", decide);
        CHECK(game.state().get_player(0).hand_size() == 5);
        CHECK(game.state().get_trash().size() == 0);
    }

    SECTION("empty hand does nothing") {
        TestGame g2;
        g2.set_hand(0, {"Chapel"});
        g2.play_card(0, "Chapel", TestGame::minimal_decisions());
        CHECK(g2.state().get_trash().size() == 0);
    }
}

// ═══════════════════════════════════════════════════════════════════
// WORKSHOP
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Workshop: gain card costing up to 4", "[cards][workshop]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Workshop"});

    auto piles = game.state().gainable_piles(4);
    int silver_pi = game.state().pile_silver();
    int silver_idx = -1;
    for (int i = 0; i < static_cast<int>(piles.size()); i++) {
        if (piles[i] == silver_pi) { silver_idx = i; break; }
    }
    REQUIRE(silver_idx >= 0);

    auto decide = TestGame::scripted_decisions({{silver_idx}});
    game.play_card(0, "Workshop", decide);

    CHECK(contains(game.discard_names(0), "Silver"));
}

// ═══════════════════════════════════════════════════════════════════
// MONEYLENDER
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Moneylender: trash Copper for +3 Coins", "[cards][moneylender]") {
    TestGame game;
    game.set_hand(0, {"Moneylender", "Copper", "Estate"});

    SECTION("accept") {
        auto decide = TestGame::scripted_decisions({{1}});
        game.play_card(0, "Moneylender", decide);
        CHECK(game.state().coins() == 3);
        CHECK(game.state().get_player(0).hand_size() == 1);
        CHECK(game.state().get_trash().size() == 1);
    }

    SECTION("decline") {
        auto decide = TestGame::scripted_decisions({{0}});
        game.play_card(0, "Moneylender", decide);
        CHECK(game.state().coins() == 0);
        CHECK(game.state().get_player(0).hand_size() == 2);
    }

    SECTION("no Copper in hand") {
        TestGame g2;
        g2.set_hand(0, {"Moneylender", "Silver", "Estate"});
        g2.play_card(0, "Moneylender", TestGame::minimal_decisions());
        CHECK(g2.state().coins() == 0);
        CHECK(g2.state().get_player(0).hand_size() == 2);
    }
}

// ═══════════════════════════════════════════════════════════════════
// POACHER
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Poacher: +1 Card +1 Action +1 Coin, discard per empty pile", "[cards][poacher]") {
    TestGame game;
    game.setup_supply({"Smithy", "Village"});
    game.set_hand(0, {"Poacher", "Copper", "Estate", "Silver"});
    game.set_deck(0, {"Gold", "Gold"});
    game.state().start_turn();

    SECTION("no empty piles, no discard") {
        game.play_card(0, "Poacher", TestGame::minimal_decisions());
        // +1 Card drawn, no discard needed
        CHECK(game.state().get_player(0).hand_size() == 4);
        CHECK(game.state().actions() == 2);
        CHECK(game.state().coins() == 1);
    }

    SECTION("1 empty pile, discard 1") {
        // Empty the Smithy pile
        while (game.state().get_supply().count("Smithy") > 0)
            game.state().get_supply().gain_from("Smithy");

        auto decide = TestGame::scripted_decisions({{0}}); // discard first card
        game.play_card(0, "Poacher", decide);
        CHECK(game.state().get_player(0).hand_size() == 3);
    }
}

// ═══════════════════════════════════════════════════════════════════
// VASSAL
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Vassal: +2 Coins, may play discarded Action", "[cards][vassal]") {
    TestGame game;

    SECTION("top card is Action, choose to play") {
        game.set_hand(0, {"Vassal"});
        game.set_deck(0, {"Gold", "Gold", "Gold", "Smithy"}); // Smithy on top
        // Vassal discards Smithy (top), we choose to play it → draws 3
        auto decide = TestGame::scripted_decisions({{1}}); // yes, play it
        game.play_card(0, "Vassal", decide);

        CHECK(game.state().coins() == 2);
        CHECK(game.state().get_player(0).hand_size() == 3); // Smithy drew 3
        CHECK(contains(game.in_play_names(0), "Smithy"));
    }

    SECTION("top card is Action, decline to play") {
        game.set_hand(0, {"Vassal"});
        game.set_deck(0, {"Gold", "Smithy"}); // Smithy on top

        auto decide = TestGame::scripted_decisions({{0}}); // no
        game.play_card(0, "Vassal", decide);

        CHECK(game.state().coins() == 2);
        CHECK(game.state().get_player(0).hand_size() == 0);
        CHECK(contains(game.discard_names(0), "Smithy"));
    }

    SECTION("top card is Treasure, no choice offered") {
        game.set_hand(0, {"Vassal"});
        game.set_deck(0, {"Silver"}); // Silver on top

        game.play_card(0, "Vassal", TestGame::minimal_decisions());

        CHECK(game.state().coins() == 2);
        CHECK(contains(game.discard_names(0), "Silver"));
    }

    SECTION("empty deck") {
        game.set_hand(0, {"Vassal"});
        game.play_card(0, "Vassal", TestGame::minimal_decisions());
        CHECK(game.state().coins() == 2);
    }
}

// ═══════════════════════════════════════════════════════════════════
// HARBINGER
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Harbinger: +1 Card +1 Action, topdeck from discard", "[cards][harbinger]") {
    TestGame game;
    game.set_hand(0, {"Harbinger"});
    game.set_deck(0, {"Copper", "Silver"});
    game.set_discard(0, {"Gold", "Estate"});
    game.state().start_turn();

    SECTION("topdeck Gold from discard") {
        auto decide = TestGame::scripted_decisions({{0}}); // Gold is at discard index 0
        game.play_card(0, "Harbinger", decide);

        CHECK(game.state().get_player(0).hand_size() == 1); // drew 1
        CHECK(game.state().actions() == 2);
        // Gold should be on top of deck
        auto top = game.state().get_player(0).peek_deck(1);
        CHECK(game.state().card_name(top[0]) == "Gold");
        CHECK(game.state().get_player(0).discard_size() == 1); // Estate remains
    }

    SECTION("skip topdecking") {
        auto decide = TestGame::scripted_decisions({{}}); // choose nothing
        game.play_card(0, "Harbinger", decide);

        CHECK(game.state().get_player(0).hand_size() == 1);
        CHECK(game.state().get_player(0).discard_size() == 2); // unchanged
    }

    SECTION("empty discard, no choice") {
        TestGame g2;
        g2.set_hand(0, {"Harbinger"});
        g2.set_deck(0, {"Copper"});
        g2.state().start_turn();
        g2.play_card(0, "Harbinger", TestGame::minimal_decisions());
        CHECK(g2.state().get_player(0).hand_size() == 1);
    }
}

// ═══════════════════════════════════════════════════════════════════
// REMODEL
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Remodel: trash a card, gain one costing up to 2 more", "[cards][remodel]") {
    TestGame game;
    game.setup_supply({});

    SECTION("trash Estate (2$), gain up to 4$") {
        game.set_hand(0, {"Remodel", "Estate"});
        auto piles = game.state().gainable_piles(4);
        int silver_pi = game.state().pile_silver();
        int silver_idx = -1;
        for (int i = 0; i < static_cast<int>(piles.size()); i++) {
            if (piles[i] == silver_pi) { silver_idx = i; break; }
        }
        REQUIRE(silver_idx >= 0);

        auto decide = TestGame::scripted_decisions({{0}, {silver_idx}});
        game.play_card(0, "Remodel", decide);

        CHECK(game.state().get_player(0).hand_size() == 0);
        CHECK(game.state().get_trash().size() == 1);
        CHECK(contains(game.discard_names(0), "Silver"));
    }

    SECTION("empty hand does nothing") {
        game.set_hand(0, {"Remodel"});
        game.play_card(0, "Remodel", TestGame::minimal_decisions());
        CHECK(game.state().get_trash().size() == 0);
    }
}

// ═══════════════════════════════════════════════════════════════════
// MINE
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Mine: trash Treasure, gain better Treasure to hand", "[cards][mine]") {
    TestGame game;
    game.setup_supply({});

    SECTION("upgrade Copper to Silver") {
        game.set_hand(0, {"Mine", "Copper"});
        auto piles = game.state().gainable_piles(3, CardType::Treasure);
        int silver_pi = game.state().pile_silver();
        int silver_idx = -1;
        for (int i = 0; i < static_cast<int>(piles.size()); i++) {
            if (piles[i] == silver_pi) { silver_idx = i; break; }
        }
        REQUIRE(silver_idx >= 0);

        auto decide = TestGame::scripted_decisions({{0}, {silver_idx}});
        game.play_card(0, "Mine", decide);

        CHECK(game.state().get_trash().size() == 1);
        CHECK(contains(game.hand_names(0), "Silver")); // gained to HAND
    }

    SECTION("upgrade Silver to Gold") {
        game.set_hand(0, {"Mine", "Silver"});
        auto piles = game.state().gainable_piles(6, CardType::Treasure);
        int gold_pi = game.state().pile_gold();
        int gold_idx = -1;
        for (int i = 0; i < static_cast<int>(piles.size()); i++) {
            if (piles[i] == gold_pi) { gold_idx = i; break; }
        }
        REQUIRE(gold_idx >= 0);

        auto decide = TestGame::scripted_decisions({{0}, {gold_idx}});
        game.play_card(0, "Mine", decide);

        CHECK(contains(game.hand_names(0), "Gold"));
    }

    SECTION("no Treasures in hand") {
        game.set_hand(0, {"Mine", "Estate"});
        game.play_card(0, "Mine", TestGame::minimal_decisions());
        CHECK(game.state().get_trash().size() == 0);
    }
}

// ═══════════════════════════════════════════════════════════════════
// ARTISAN
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Artisan: gain to hand costing <=5, topdeck a card", "[cards][artisan]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Artisan", "Copper", "Estate"});

    auto piles = game.state().gainable_piles(5);
    int duchy_pi = game.state().pile_duchy();
    int duchy_idx = -1;
    for (int i = 0; i < static_cast<int>(piles.size()); i++) {
        if (piles[i] == duchy_pi) { duchy_idx = i; break; }
    }
    REQUIRE(duchy_idx >= 0);

    // Gain Duchy to hand, then topdeck Estate (index 1 after gain)
    // After Artisan played: hand = Copper, Estate. After gain: Copper, Estate, Duchy
    // Topdeck index 1 = Estate
    auto decide = TestGame::scripted_decisions({{duchy_idx}, {1}});
    game.play_card(0, "Artisan", decide);

    CHECK(contains(game.hand_names(0), "Duchy"));
    CHECK(contains(game.hand_names(0), "Copper"));
    CHECK(!contains(game.hand_names(0), "Estate")); // topdecked
    auto top = game.state().get_player(0).peek_deck(1);
    CHECK(game.state().card_name(top[0]) == "Estate");
}

// ═══════════════════════════════════════════════════════════════════
// SENTRY
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Sentry: +1 Card +1 Action, look at top 2, trash/discard/keep", "[cards][sentry]") {
    TestGame game;
    game.state().start_turn();

    SECTION("trash both") {
        game.set_hand(0, {"Sentry"});
        game.set_deck(0, {"Estate", "Copper", "Gold"}); // top=Gold, Copper, Estate
        // Draws 1 (Gold to hand). Peeks top 2: Copper, Estate
        // Fate for Copper: 2=trash. Fate for Estate: 2=trash.
        auto decide = TestGame::scripted_decisions({{2}, {2}});
        game.play_card(0, "Sentry", decide);

        CHECK(game.state().get_player(0).hand_size() == 1); // just Gold
        CHECK(game.state().get_trash().size() == 2);
        CHECK(game.state().actions() == 2);
    }

    SECTION("discard one, keep one") {
        game.set_hand(0, {"Sentry"});
        game.set_deck(0, {"Gold", "Estate", "Silver"}); // top=Silver, Estate, Gold
        // Draws 1 (Silver to hand). Peeks top 2: Estate, Gold
        // Estate: 1=discard. Gold: 0=keep.
        auto decide = TestGame::scripted_decisions({{1}, {0}});
        game.play_card(0, "Sentry", decide);

        CHECK(game.state().get_player(0).hand_size() == 1); // Silver
        CHECK(game.state().get_player(0).discard_size() == 1); // Estate
        // Gold should be back on deck
        auto top = game.state().get_player(0).peek_deck(1);
        CHECK(game.state().card_name(top[0]) == "Gold");
    }
}

// ═══════════════════════════════════════════════════════════════════
// LIBRARY
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Library: draw to 7, can set aside Actions", "[cards][library]") {
    SECTION("draw to 7 with no actions in deck") {
        TestGame game;
        game.set_hand(0, {"Library", "Copper"});
        game.set_deck(0, {"Gold", "Silver", "Estate", "Copper", "Copper",
                           "Silver", "Gold"});

        game.play_card(0, "Library", TestGame::minimal_decisions());
        // Started with 1 (Copper after Library played). Draw 6 more to reach 7.
        CHECK(game.state().get_player(0).hand_size() == 7);
    }

    SECTION("skip Action cards") {
        TestGame game;
        game.set_hand(0, {"Library", "Copper"});
        // Deck: Silver(top), Smithy, Gold, Copper, Estate, Duchy, Province
        game.set_deck(0, {"Province", "Duchy", "Estate", "Copper", "Gold", "Smithy", "Silver"});

        // Draw: Silver (keep), Smithy (action → set aside? yes=1), Gold, Copper, Estate, Duchy = 7
        auto decide = TestGame::scripted_decisions({{1}}); // set aside Smithy
        game.play_card(0, "Library", decide);

        CHECK(game.state().get_player(0).hand_size() == 7);
        // Smithy should be in discard (set aside then discarded)
        CHECK(contains(game.discard_names(0), "Smithy"));
    }

    SECTION("already at 7, draw nothing") {
        TestGame game;
        game.set_hand(0, {"Library", "C1", "C2", "C3", "C4", "C5", "C6"});
        // 6 cards after Library played → needs 1 more
        // But we only have 6 cards named oddly. Let's fix:
        TestGame g2;
        g2.set_hand(0, {"Library", "Copper", "Copper", "Copper",
                         "Copper", "Copper", "Copper", "Copper"});
        g2.set_deck(0, {"Gold"});
        g2.play_card(0, "Library", TestGame::minimal_decisions());
        // Had 7 after Library played, so draws 0
        CHECK(g2.state().get_player(0).hand_size() == 7);
    }
}

// ═══════════════════════════════════════════════════════════════════
// BUREAUCRAT
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Bureaucrat: gain Silver to deck, opponent topdecks Victory", "[cards][bureaucrat]") {
    TestGame game;
    game.setup_supply({});

    SECTION("opponent has one Victory, auto-topdecks") {
        game.set_hand(0, {"Bureaucrat"});
        game.set_hand(1, {"Estate", "Copper", "Silver"});
        game.play_card(0, "Bureaucrat", TestGame::minimal_decisions());

        CHECK(game.deck_names(0).back() == "Silver");
        CHECK(game.deck_names(1).back() == "Estate");
        CHECK(game.state().get_player(1).hand_size() == 2);
    }

    SECTION("opponent has no Victory cards") {
        game.set_hand(0, {"Bureaucrat"});
        game.set_hand(1, {"Copper", "Silver", "Gold"});
        game.play_card(0, "Bureaucrat", TestGame::minimal_decisions());

        // Opponent hand unchanged (revealed, no Victory)
        CHECK(game.state().get_player(1).hand_size() == 3);
    }

    SECTION("opponent chooses which Victory to topdeck") {
        game.set_hand(0, {"Bureaucrat"});
        game.set_hand(1, {"Estate", "Duchy", "Copper"});
        // Two Victory cards: Estate(idx 0), Duchy(idx 1). Choose Duchy (idx 1).
        auto decide = TestGame::scripted_decisions({{1}});
        game.play_card(0, "Bureaucrat", decide);

        CHECK(game.deck_names(1).back() == "Duchy");
        CHECK(game.state().get_player(1).hand_size() == 2);
    }
}

// ═══════════════════════════════════════════════════════════════════
// MOAT
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Moat: +2 Cards as action", "[cards][moat]") {
    TestGame game;
    game.set_hand(0, {"Moat"});
    game.set_deck(0, {"Copper", "Silver", "Gold"});

    game.play_card(0, "Moat", TestGame::minimal_decisions());
    CHECK(game.state().get_player(0).hand_size() == 2);
}

// ═══════════════════════════════════════════════════════════════════
// MILITIA
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Militia: +2 Coins, opponent discards to 3", "[cards][militia]") {
    TestGame game;

    SECTION("opponent with 5 cards discards 2") {
        game.set_hand(0, {"Militia"});
        game.set_hand(1, {"Estate", "Estate", "Copper", "Silver", "Gold"});
        auto decide = TestGame::scripted_decisions({{0, 1}});
        game.play_card(0, "Militia", decide);

        CHECK(game.state().coins() == 2);
        CHECK(game.state().get_player(1).hand_size() == 3);
    }

    SECTION("opponent with 3 cards, no discard") {
        game.set_hand(0, {"Militia"});
        game.set_hand(1, {"Copper", "Silver", "Gold"});
        game.play_card(0, "Militia", TestGame::minimal_decisions());

        CHECK(game.state().get_player(1).hand_size() == 3);
        CHECK(game.state().coins() == 2);
    }

    SECTION("opponent with 0 cards, no discard") {
        game.set_hand(0, {"Militia"});
        // player 1 has empty hand (default)
        game.play_card(0, "Militia", TestGame::minimal_decisions());
        CHECK(game.state().get_player(1).hand_size() == 0);
        CHECK(game.state().coins() == 2);
    }
}

// ═══════════════════════════════════════════════════════════════════
// BANDIT
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Bandit: gain Gold, trash opponent non-Copper Treasure", "[cards][bandit]") {
    TestGame game;
    game.setup_supply({});

    SECTION("opponent reveals Silver and Copper") {
        game.set_hand(0, {"Bandit"});
        game.set_deck(1, {"Copper", "Silver"});
        game.play_card(0, "Bandit", TestGame::minimal_decisions());

        CHECK(contains(game.discard_names(0), "Gold"));
        CHECK(game.state().get_trash().size() == 1);
        CHECK(game.state().card_name(game.state().get_trash()[0]) == "Silver");
        CHECK(contains(game.discard_names(1), "Copper"));
    }

    SECTION("opponent reveals two Coppers, none trashed") {
        game.set_hand(0, {"Bandit"});
        game.set_deck(1, {"Copper", "Copper"});
        game.play_card(0, "Bandit", TestGame::minimal_decisions());

        CHECK(game.state().get_trash().size() == 0);
        CHECK(game.state().get_player(1).discard_size() == 2);
    }

    SECTION("opponent reveals no Treasures") {
        game.set_hand(0, {"Bandit"});
        game.set_deck(1, {"Estate", "Estate"});
        game.play_card(0, "Bandit", TestGame::minimal_decisions());

        CHECK(game.state().get_trash().size() == 0);
    }

    SECTION("opponent has empty deck") {
        game.set_hand(0, {"Bandit"});
        // Player 1 has nothing
        game.play_card(0, "Bandit", TestGame::minimal_decisions());
        CHECK(contains(game.discard_names(0), "Gold"));
    }
}

// ═══════════════════════════════════════════════════════════════════
// WITCH
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Witch: +2 Cards, each other player gains Curse", "[cards][witch]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Witch"});
    game.set_deck(0, {"Copper", "Silver", "Gold"});

    game.play_card(0, "Witch", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 2);
    CHECK(contains(game.discard_names(1), "Curse"));
}

TEST_CASE("Witch: no Curses left, opponent unaffected", "[cards][witch]") {
    TestGame game;
    game.setup_supply({});
    // Empty the Curse pile
    while (game.state().get_supply().count("Curse") > 0)
        game.state().get_supply().gain_from("Curse");

    game.set_hand(0, {"Witch"});
    game.set_deck(0, {"Copper", "Silver"});
    game.play_card(0, "Witch", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 2);
    CHECK(game.state().get_player(1).discard_size() == 0);
}

// ═══════════════════════════════════════════════════════════════════
// COUNCIL ROOM
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Council Room: +4 Cards +1 Buy, others draw 1", "[cards][councilroom]") {
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

// ═══════════════════════════════════════════════════════════════════
// THRONE ROOM
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Throne Room: plays action twice", "[cards][throneroom]") {
    SECTION("TR + Smithy = draw 6") {
        TestGame game;
        game.set_hand(0, {"Throne Room", "Smithy", "Copper"});
        game.set_deck(0, {"Gold", "Gold", "Gold", "Gold", "Gold", "Gold", "Gold"});

        auto decide = TestGame::scripted_decisions({{0}}); // pick Smithy
        game.play_card(0, "Throne Room", decide);

        CHECK(game.state().get_player(0).hand_size() == 7); // Copper + 6
    }

    SECTION("TR + Village = +2 Cards +4 Actions") {
        TestGame game;
        game.set_hand(0, {"Throne Room", "Village", "Copper"});
        game.set_deck(0, {"Gold", "Silver", "Estate", "Duchy"});
        game.state().start_turn();

        auto decide = TestGame::scripted_decisions({{0}});
        game.play_card(0, "Throne Room", decide);

        CHECK(game.state().get_player(0).hand_size() == 3);
        CHECK(game.state().actions() == 5); // 1 + 2 + 2
    }

    SECTION("TR + Festival = +4 Actions +2 Buys +4 Coins") {
        TestGame game;
        game.set_hand(0, {"Throne Room", "Festival"});
        game.state().start_turn();

        auto decide = TestGame::scripted_decisions({{0}});
        game.play_card(0, "Throne Room", decide);

        CHECK(game.state().actions() == 5); // 1 + 2 + 2
        CHECK(game.state().buys() == 3);     // 1 + 1 + 1
        CHECK(game.state().coins() == 4);
    }

    SECTION("TR + Militia = +4 Coins, opponent discards to 3") {
        TestGame game;
        game.set_hand(0, {"Throne Room", "Militia", "Copper"});
        game.set_hand(1, {"Estate", "Estate", "Copper", "Silver", "Gold"});

        // Pick Militia (index 0 after TR leaves hand: Militia=0, Copper=1)
        // First Militia: opponent discards 2
        // Second Militia: opponent already at 3, no discard
        auto decide = TestGame::scripted_decisions({{0}, {0, 1}});
        game.play_card(0, "Throne Room", decide);

        CHECK(game.state().coins() == 4);
        CHECK(game.state().get_player(1).hand_size() == 3);
    }

    SECTION("TR + Witch = +4 Cards, opponent gets 2 Curses") {
        TestGame game;
        game.setup_supply({});
        game.set_hand(0, {"Throne Room", "Witch"});
        game.set_deck(0, {"Copper", "Silver", "Gold", "Estate"});

        auto decide = TestGame::scripted_decisions({{0}});
        game.play_card(0, "Throne Room", decide);

        CHECK(game.state().get_player(0).hand_size() == 4);
        CHECK(count_of(game.discard_names(1), "Curse") == 2);
    }

    SECTION("TR with no actions in hand") {
        TestGame game;
        game.set_hand(0, {"Throne Room", "Copper", "Estate"});
        game.play_card(0, "Throne Room", TestGame::minimal_decisions());

        CHECK(game.state().get_player(0).hand_size() == 2);
        CHECK(game.in_play_names(0) == std::vector<std::string>{"Throne Room"});
    }

    SECTION("TR picks second action in hand (not first)") {
        // This tests the bug where chosen[0] was used as hand_idx directly
        TestGame game;
        game.set_hand(0, {"Throne Room", "Copper", "Smithy"});
        game.set_deck(0, {"Gold", "Gold", "Gold", "Gold", "Gold", "Gold"});

        // After TR leaves: Copper(0), Smithy(1). Actions: {Smithy at hand idx 1}.
        // action_indices = {1}. Agent picks index 0 into action_indices → hand_idx=1
        auto decide = TestGame::scripted_decisions({{0}});
        game.play_card(0, "Throne Room", decide);

        CHECK(game.state().get_player(0).hand_size() == 7); // Copper + 6 drawn
        CHECK(contains(game.in_play_names(0), "Smithy"));
    }

    SECTION("TR + Chapel trashes cards twice") {
        TestGame game;
        game.set_hand(0, {"Throne Room", "Chapel", "Estate", "Estate", "Copper", "Copper"});

        // First Chapel: trash indices 0,1 (Estate, Estate from remaining hand)
        // After TR+Chapel played, hand = Estate, Estate, Copper, Copper
        // First Chapel play: trash 0,1 → hand = Copper, Copper
        // Second Chapel play: trash 0,1 → hand = empty
        auto decide = TestGame::scripted_decisions({{0}, {0, 1}, {0, 1}});
        game.play_card(0, "Throne Room", decide);

        CHECK(game.state().get_player(0).hand_size() == 0);
        CHECK(game.state().get_trash().size() == 4);
    }
}

// ═══════════════════════════════════════════════════════════════════
// GARDENS
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Gardens: VP = total cards / 10", "[cards][gardens]") {
    TestGame game(1);

    SECTION("10 cards = 1 VP") {
        game.set_hand(0, {"Gardens"});
        game.set_deck(0, std::vector<std::string>(9, "Copper"));
        CHECK(game.state().calculate_scores()[0] == 1);
    }

    SECTION("20 cards = 2 VP") {
        game.set_hand(0, {"Gardens"});
        game.set_deck(0, std::vector<std::string>(9, "Copper"));
        game.set_discard(0, std::vector<std::string>(10, "Copper"));
        CHECK(game.state().calculate_scores()[0] == 2);
    }

    SECTION("9 cards = 0 VP") {
        game.set_hand(0, {"Gardens"});
        game.set_deck(0, std::vector<std::string>(8, "Copper"));
        CHECK(game.state().calculate_scores()[0] == 0);
    }

    SECTION("multiple Gardens") {
        game.set_hand(0, {"Gardens", "Gardens"});
        game.set_deck(0, std::vector<std::string>(8, "Copper"));
        // 10 cards, each Gardens = 1 VP, total = 2
        CHECK(game.state().calculate_scores()[0] == 2);
    }
}

// ═══════════════════════════════════════════════════════════════════
// MERCHANT
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Merchant: +1 Card +1 Action, Silver bonus tracked", "[cards][merchant]") {
    TestGame game;
    game.set_hand(0, {"Merchant"});
    game.set_deck(0, {"Copper", "Silver"});
    game.state().start_turn();

    game.play_card(0, "Merchant", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 1);
    CHECK(game.state().actions() == 2);
    CHECK(game.state().get_turn_flag(TurnFlag::MerchantCount) == 1);
}
