#include <catch2/catch_test_macros.hpp>
#include "test_helpers.h"

TEST_CASE("GameState initial turn state", "[gamestate]") {
    TestGame game(2);
    game.state().start_game();

    CHECK(game.state().current_phase() == Phase::ACTION);
    CHECK(game.state().actions() == 1);
    CHECK(game.state().buys() == 1);
    CHECK(game.state().coins() == 0);
    CHECK(game.state().current_player_id() == 0);
    CHECK(game.state().turn_number() == 1);
}

TEST_CASE("GameState phase transitions", "[gamestate]") {
    TestGame game(2);
    game.state().start_game();

    CHECK(game.state().current_phase() == Phase::ACTION);
    game.state().advance_phase();
    CHECK(game.state().current_phase() == Phase::TREASURE);
    game.state().advance_phase();
    CHECK(game.state().current_phase() == Phase::BUY);
}

TEST_CASE("GameState card ID system", "[gamestate]") {
    TestGame game(1);
    int id1 = game.state().create_card("Copper");
    int id2 = game.state().create_card("Silver");

    CHECK(id1 != id2);
    CHECK(game.state().card_name(id1) == "Copper");
    CHECK(game.state().card_name(id2) == "Silver");
    CHECK(game.state().card_def(id1)->is_treasure());
}

TEST_CASE("GameState scoring with static VP", "[gamestate]") {
    TestGame game(2);
    game.set_hand(0, {"Estate", "Estate", "Copper"});
    game.set_hand(1, {"Province", "Copper"});

    auto scores = game.state().calculate_scores();
    CHECK(scores[0] == 2);  // 2 Estates = 2 VP
    CHECK(scores[1] == 6);  // 1 Province = 6 VP
}

TEST_CASE("GameState scoring with dynamic VP (Gardens)", "[gamestate]") {
    TestGame game(1);
    game.set_hand(0, {"Gardens"});
    // 19 more cards = 20 total
    std::vector<std::string> deck(19, "Copper");
    game.set_deck(0, deck);

    auto scores = game.state().calculate_scores();
    CHECK(scores[0] == 2);  // 20 / 10 = 2
}

TEST_CASE("GameState gainable_piles filters by cost", "[gamestate]") {
    TestGame game(2);
    game.setup_supply({});

    auto piles = game.state().gainable_piles(3);
    // Should include Copper(0), Silver(3), Estate(2), Curse(0)
    // Should NOT include Gold(6), Duchy(5), Province(8)
    int silver_pi = game.state().pile_silver();
    int gold_pi = game.state().pile_gold();
    bool has_silver = false, has_gold = false;
    for (int p : piles) {
        if (p == silver_pi) has_silver = true;
        if (p == gold_pi) has_gold = true;
    }
    CHECK(has_silver);
    CHECK_FALSE(has_gold);
}

TEST_CASE("GameState turn flags", "[gamestate]") {
    TestGame game(1);
    CHECK(game.state().get_turn_flag(TurnFlag::MerchantCount) == 0);
    game.state().set_turn_flag(TurnFlag::MerchantCount, 3);
    CHECK(game.state().get_turn_flag(TurnFlag::MerchantCount) == 3);
    game.state().start_turn();
    CHECK(game.state().get_turn_flag(TurnFlag::MerchantCount) == 0);  // cleared on new turn
}
