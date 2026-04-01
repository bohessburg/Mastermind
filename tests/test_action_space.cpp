#include <catch2/catch_test_macros.hpp>
#include "test_helpers.h"
#include "engine/action_space.h"

TEST_CASE("Action decision lists playable actions", "[actionspace]") {
    TestGame game;
    game.set_hand(0, {"Smithy", "Village", "Copper", "Estate"});
    game.state().start_turn();

    auto dp = build_action_decision(game.state());
    CHECK(dp.player_id == 0);
    CHECK(dp.type == DecisionType::PLAY_ACTION);
    // 2 actions + 1 pass = 3 options
    CHECK(dp.num_options() == 3);
    CHECK(dp.options.back().is_pass);
}

TEST_CASE("Treasure decision includes play-all shortcut", "[actionspace]") {
    TestGame game;
    game.set_hand(0, {"Copper", "Silver", "Gold", "Estate"});

    auto dp = build_treasure_decision(game.state());
    CHECK(dp.options[0].is_play_all);
    CHECK(dp.options.back().is_pass);
    // Play all + 3 individual treasures + stop = 5
    CHECK(dp.num_options() == 5);
}

TEST_CASE("Buy decision filters by coins", "[actionspace]") {
    TestGame game;
    game.setup_supply({});
    game.state().add_coins(4);

    auto dp = build_buy_decision(game.state());
    // All piles costing <= 4 + End Buys
    for (int i = 0; i < dp.num_options() - 1; i++) {
        const Card* card = (dp.options[i].def_id >= 0) ? CardRegistry::get(dp.options[i].def_id) : nullptr;
        REQUIRE(card);
        CHECK(card->cost <= 4);
    }
    CHECK(dp.options.back().is_pass);
}

TEST_CASE("Action decision with no actions shows only pass", "[actionspace]") {
    TestGame game;
    game.set_hand(0, {"Copper", "Estate"});
    game.state().start_turn();

    auto dp = build_action_decision(game.state());
    CHECK(dp.num_options() == 1);
    CHECK(dp.options[0].is_pass);
}
