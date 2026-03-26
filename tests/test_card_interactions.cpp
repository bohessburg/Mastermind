#include <catch2/catch_test_macros.hpp>
#include "test_helpers.h"

#include <algorithm>

static bool contains(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

// ═══════════════════════════════════════════════════════════════════
// Throne Room interactions (the most complex card)
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("TR + Smithy = draw 6", "[interaction][throneroom]") {
    TestGame game;
    game.set_hand(0, {"Throne Room", "Smithy", "Copper"});
    game.set_deck(0, {"Gold", "Gold", "Gold", "Gold", "Gold", "Gold", "Gold"});

    auto decide = TestGame::scripted_decisions({{0}});
    game.play_card(0, "Throne Room", decide);

    CHECK(game.state().get_player(0).hand_size() == 7);
}

TEST_CASE("TR + Village = +2 Cards +4 Actions", "[interaction][throneroom]") {
    TestGame game;
    game.set_hand(0, {"Throne Room", "Village", "Copper"});
    game.set_deck(0, {"Gold", "Silver", "Estate", "Duchy"});
    game.state().start_turn();

    auto decide = TestGame::scripted_decisions({{0}});
    game.play_card(0, "Throne Room", decide);

    CHECK(game.state().get_player(0).hand_size() == 3);
    CHECK(game.state().actions() == 5);
}

TEST_CASE("TR + Militia = +4 Coins, opponent discards to 3", "[interaction][throneroom]") {
    TestGame game;
    game.set_hand(0, {"Throne Room", "Militia", "Copper"});
    game.set_hand(1, {"Estate", "Estate", "Copper", "Silver", "Gold"});

    auto decide = TestGame::scripted_decisions({{0}, {0, 1}});
    game.play_card(0, "Throne Room", decide);

    CHECK(game.state().coins() == 4);
    CHECK(game.state().get_player(1).hand_size() == 3);
}

TEST_CASE("TR with no actions does nothing", "[interaction][throneroom]") {
    TestGame game;
    game.set_hand(0, {"Throne Room", "Copper", "Estate"});

    game.play_card(0, "Throne Room", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 2);
    CHECK(game.in_play_names(0).size() == 1);
}

TEST_CASE("TR + Moat = draw 4", "[interaction][throneroom]") {
    TestGame game;
    game.set_hand(0, {"Throne Room", "Moat"});
    game.set_deck(0, {"Copper", "Silver", "Gold", "Estate"});

    auto decide = TestGame::scripted_decisions({{0}});
    game.play_card(0, "Throne Room", decide);

    CHECK(game.state().get_player(0).hand_size() == 4);
}

TEST_CASE("TR + Laboratory = +4 Cards +2 Actions", "[interaction][throneroom]") {
    TestGame game;
    game.set_hand(0, {"Throne Room", "Laboratory"});
    game.set_deck(0, {"Copper", "Silver", "Gold", "Estate"});
    game.state().start_turn();

    auto decide = TestGame::scripted_decisions({{0}});
    game.play_card(0, "Throne Room", decide);

    CHECK(game.state().get_player(0).hand_size() == 4);
    CHECK(game.state().actions() == 3); // 1 + 1 + 1
}

TEST_CASE("TR + Cellar = discard/draw twice", "[interaction][throneroom]") {
    TestGame game;
    game.set_hand(0, {"Throne Room", "Cellar", "Estate", "Estate", "Copper"});
    game.set_deck(0, {"Gold", "Gold", "Gold", "Gold", "Gold"});
    game.state().start_turn();

    // TR picks Cellar (idx 0 in action_indices)
    // First Cellar: discard Estates (indices 0,1 of remaining hand after TR+Cellar played)
    //   remaining: Estate, Estate, Copper → discard 0,1 → draw 2 → hand: Copper, Gold, Gold
    // Second Cellar: discard nothing
    auto decide = TestGame::scripted_decisions({{0}, {0, 1}, {}});
    game.play_card(0, "Throne Room", decide);

    // Started: 1 action. Cellar gives +1 Action twice = 1+1+1=3
    CHECK(game.state().actions() == 3);
}

TEST_CASE("TR + Moneylender = trash Copper twice (if two Coppers)", "[interaction][throneroom]") {
    TestGame game;
    game.set_hand(0, {"Throne Room", "Moneylender", "Copper", "Copper", "Estate"});

    // TR picks Moneylender. First play: yes trash Copper. Second play: yes trash other Copper.
    auto decide = TestGame::scripted_decisions({{0}, {1}, {1}});
    game.play_card(0, "Throne Room", decide);

    CHECK(game.state().coins() == 6); // 3 + 3
    CHECK(game.state().get_trash().size() == 2);
    CHECK(game.state().get_player(0).hand_size() == 1); // just Estate
}

TEST_CASE("TR + TR + two actions = play two actions twice each", "[interaction][throneroom]") {
    // Throne Room a Throne Room: you play TR, choose TR2.
    // TR2 plays twice. Each play of TR2 asks you to pick an action and plays it twice.
    // Net result: two actions each played twice = 4 total action plays.
    TestGame game;
    game.set_hand(0, {"Throne Room", "Throne Room", "Smithy", "Village"});
    game.set_deck(0, {"Gold", "Gold", "Gold", "Gold", "Gold", "Gold",
                       "Gold", "Gold", "Gold", "Gold", "Gold", "Gold"});
    game.state().start_turn();

    // Decision sequence:
    // 1. Outer TR: pick which action → TR2 (index 0 in action_indices: TR2=0, Smithy=1, Village=2)
    // 2. First play of inner TR2: pick action → Smithy (now hand has Smithy, Village after TR2 moved to play)
    //    action_indices for hand [Smithy, Village] → {0, 1}. Pick 0 = Smithy.
    //    Smithy plays twice: +3, +3 = 6 cards drawn.
    // 3. Second play of inner TR2: pick action → Village (only action left in hand after Smithy played)
    //    action_indices = {idx of Village}. Pick 0.
    //    Village plays twice: +1 Card +2 Actions, +1 Card +2 Actions
    auto decide = TestGame::scripted_decisions({{0}, {0}, {0}});
    game.play_card(0, "Throne Room", decide);

    // Cards drawn: Smithy x2 = 6, Village x2 = 2. Total = 8 new cards in hand.
    // Started with Village in hand (after TR, TR2, Smithy moved to play).
    // Wait — let me trace carefully:
    // Initial hand: [TR, TR2, Smithy, Village]
    // play_card removes TR → hand: [TR2, Smithy, Village]
    // TR's on_play: finds actions [TR2(0), Smithy(1), Village(2)]. Pick 0 → TR2.
    //   hand_idx = action_indices[0] = 0 (TR2). play_from_hand(0) → hand: [Smithy, Village]
    //   First play of TR2's on_play:
    //     finds actions in hand: Smithy(0), Village(1). Pick 0 → Smithy.
    //     hand_idx = action_indices[0] = 0. play_from_hand(0) → hand: [Village]
    //     Smithy on_play: draw 3 → hand: [Village, G, G, G]
    //     Smithy on_play again: draw 3 → hand: [Village, G, G, G, G, G, G]
    //   Second play of TR2's on_play:
    //     finds actions in hand: Village at some index. Pick 0.
    //     play_from_hand → hand: [G, G, G, G, G, G]
    //     Village on_play: draw 1 + 2 Actions → hand: [G, G, G, G, G, G, G]
    //     Village on_play again: draw 1 + 2 Actions → hand: [G, G, G, G, G, G, G, G]
    // Final hand: 8 cards (all Gold)
    CHECK(game.state().get_player(0).hand_size() == 8);

    // Actions: start 1, +2 +2 from Village x2 = 5
    CHECK(game.state().actions() == 5);

    // In play: TR, TR2, Smithy, Village
    CHECK(game.in_play_names(0).size() == 4);
}

TEST_CASE("TR + TR + TR + three actions", "[interaction][throneroom]") {
    // TR1 → TR2 (played twice). Each TR2 play picks another action and plays it twice.
    // First TR2 play → TR3 (played twice). Each TR3 play picks an action and plays it twice.
    //   First TR3 play → Smithy x2 = +6 cards
    //   Second TR3 play → Village x2 = +2 cards, +4 actions
    // Second TR2 play → Festival x2 = +4 actions, +2 buys, +4 coins
    //
    // Hand trace:
    // Start: [TR1, TR2, TR3, Smithy, Village, Festival]
    // play TR1 → hand: [TR2, TR3, Smithy, Village, Festival]
    // TR1 picks TR2 (idx 0) → play_from_hand → hand: [TR3, Smithy, Village, Festival]
    //   First play of TR2: picks TR3 (idx 0) → hand: [Smithy, Village, Festival]
    //     First play of TR3: picks Smithy (idx 0) → hand: [Village, Festival]
    //       Smithy x2: draw 6 → hand: [Village, Festival, G, G, G, G, G, G]
    //     Second play of TR3: picks Village (idx 0) → hand: [Festival, G, G, G, G, G, G]
    //       Village x2: draw 2, +4 actions → hand: [Festival, G, G, G, G, G, G, G, G]
    //   Second play of TR2: picks Festival (idx 0) → hand: [G, G, G, G, G, G, G, G]
    //     Festival x2: +4 actions, +2 buys, +4 coins
    // Final hand: 8 Golds
    TestGame game;
    game.set_hand(0, {"Throne Room", "Throne Room", "Throne Room",
                       "Smithy", "Village", "Festival"});
    game.set_deck(0, {"Gold", "Gold", "Gold", "Gold", "Gold", "Gold",
                       "Gold", "Gold", "Gold", "Gold", "Gold", "Gold"});
    game.state().start_turn();

    // Decisions: each TR picks index 0 from its action list
    auto decide = TestGame::scripted_decisions({{0}, {0}, {0}, {0}, {0}});
    game.play_card(0, "Throne Room", decide);

    CHECK(game.state().get_player(0).hand_size() == 8);
    // Actions: 1 + 2+2 (Village x2) + 2+2 (Festival x2) = 9
    CHECK(game.state().actions() == 9);
    // Buys: 1 + 1+1 (Festival x2) = 3
    CHECK(game.state().buys() == 3);
    // Coins: 2+2 (Festival x2) = 4
    CHECK(game.state().coins() == 4);
    // In play: TR1, TR2, TR3, Smithy, Village, Festival
    CHECK(game.in_play_names(0).size() == 6);
}

// ═══════════════════════════════════════════════════════════════════
// Moat blocking various attacks
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Moat blocks Witch", "[interaction][moat]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Witch"});
    game.set_deck(0, {"Copper", "Silver"});
    game.set_hand(1, {"Moat"});

    auto decide = TestGame::scripted_decisions({{1}}); // reveal Moat
    game.play_card(0, "Witch", decide);

    CHECK(!contains(game.discard_names(1), "Curse"));
}

TEST_CASE("Moat blocks Bandit", "[interaction][moat]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Bandit"});
    game.set_deck(1, {"Silver", "Gold"});
    game.set_hand(1, {"Moat"});

    auto decide = TestGame::scripted_decisions({{1}}); // reveal Moat
    game.play_card(0, "Bandit", decide);

    // Opponent's deck untouched
    CHECK(game.state().get_player(1).deck_size() == 2);
    CHECK(game.state().get_trash().size() == 0);
    // Attacker still gains Gold
    CHECK(contains(game.discard_names(0), "Gold"));
}

TEST_CASE("Moat blocks Bureaucrat", "[interaction][moat]") {
    TestGame game;
    game.setup_supply({});
    game.set_hand(0, {"Bureaucrat"});
    game.set_hand(1, {"Moat", "Estate"});

    auto decide = TestGame::scripted_decisions({{1}}); // reveal Moat
    game.play_card(0, "Bureaucrat", decide);

    // Opponent keeps their Estate in hand
    CHECK(game.state().get_player(1).hand_size() == 2);
}

// ═══════════════════════════════════════════════════════════════════
// Council Room is NOT an attack
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Council Room cannot be blocked by Moat", "[interaction]") {
    TestGame game;
    game.set_hand(0, {"Council Room"});
    game.set_deck(0, {"Copper", "Silver", "Gold", "Estate"});
    game.set_hand(1, {"Moat"});
    game.set_deck(1, {"Province"});
    game.state().start_turn();

    // No reaction window — Council Room is not an Attack
    game.play_card(0, "Council Room", TestGame::minimal_decisions());

    CHECK(game.state().get_player(0).hand_size() == 4);
    // Opponent drew 1 regardless of Moat
    CHECK(game.state().get_player(1).hand_size() == 2); // Moat + Province
}
