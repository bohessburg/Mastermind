#include "spec_harness.h"

#include "v2/core/defs.h"

#include <cstring>

namespace {

[[nodiscard]] bool same_bytes(const GameState& lhs, const GameState& rhs) {
    return std::memcmp(&lhs, &rhs, sizeof(GameState)) == 0;
}

} // namespace

CARD_SPEC("v2 Cellar discards selected cards and draws per chosen") {
    given().hand("Cellar", "Copper", "Estate", "Copper").deck("Silver", "Gold");

    play("Cellar");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.source == DEF_CELLAR);
    REQUIRE(state().decision.min_left == 0U);
    REQUIRE(state().decision.max_left == 3U);

    choose("Copper");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.min_left == 0U);
    REQUIRE(state().decision.max_left == 2U);

    choose("Estate");
    pass();

    expect().discard_has("Copper").discard_has("Estate").hand_has("Gold").hand_has("Silver").coins(0);
    expect_conservation();
}

CARD_SPEC("v2 Cellar can pass immediately with zero discards") {
    given().hand("Cellar", "Copper").deck("Silver");

    play("Cellar");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.min_left == 0U);

    pass();

    expect().hand_has("Copper").hand_lacks("Silver").coins(0);
    expect_conservation();
}

CARD_SPEC("v2 Chapel trashes up to four cards") {
    given().hand("Chapel", "Copper", "Copper", "Copper", "Estate", "Estate");

    play("Chapel");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.source == DEF_CHAPEL);
    REQUIRE(state().decision.min_left == 0U);
    REQUIRE(state().decision.max_left == 4U);

    choose("Copper");
    choose("Copper");
    choose("Copper");
    choose("Estate");

    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::PhaseBuy));
    expect().trash_count("Copper", 3).trash_count("Estate", 1).hand_has("Estate");
    expect_conservation();
}

CARD_SPEC("v2 Village draws one and adds two actions") {
    given().hand("Village").deck("Silver");

    play("Village");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(state().actions == 2U);
    expect().hand_has("Silver").coins(0);
    expect_conservation();
}

CARD_SPEC("v2 Smithy draws three cards") {
    given().hand("Smithy").deck("Copper", "Silver", "Gold");

    play("Smithy");

    expect().hand_has("Copper").hand_has("Silver").hand_has("Gold").coins(0);
    expect_conservation();
}

CARD_SPEC("v2 Workshop gains a card costing up to four to discard") {
    given().hand("Workshop");

    play("Workshop");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::ChooseGain));
    REQUIRE(state().decision.source == DEF_WORKSHOP);
    REQUIRE(state().decision.min_left == 1U);
    REQUIRE(state().decision.max_left == 1U);

    gain("Silver");

    expect().discard_has("Silver").hand_lacks("Silver");
    expect_conservation();
}

CARD_SPEC("v2 Remodel trashes one card and gains up to two more to discard") {
    given().hand("Remodel", "Estate");

    play("Remodel");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.min_left == 1U);
    REQUIRE(state().decision.max_left == 1U);
    choose("Estate");

    ActionMask legal{};
    (void)Game::legal_actions(state(), legal);
    REQUIRE(legal.test(select_action(DEF_SILVER)));
    REQUIRE_FALSE(legal.test(select_action(DEF_DUCHY)));

    gain("Silver");

    expect().trash_has("Estate").discard_has("Silver");
    expect_conservation();
}

CARD_SPEC("v2 Mine trashes a treasure and gains to hand") {
    given().hand("Mine", "Silver");

    play("Mine");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.source == DEF_MINE);
    REQUIRE(state().decision.min_left == 1U);
    REQUIRE(state().decision.max_left == 1U);
    choose("Silver");

    ActionMask legal{};
    (void)Game::legal_actions(state(), legal);
    REQUIRE(legal.test(select_action(DEF_GOLD)));
    REQUIRE_FALSE(legal.test(select_action(DEF_PROVINCE)));

    gain("Gold");

    expect().trash_has("Silver").hand_has("Gold").discard_lacks("Gold");
    expect_conservation();
}

CARD_SPEC("v2 clone-equivalence holds while suspended in Choose") {
    given().hand("Cellar", "Copper", "Estate").deck("Silver", "Gold");

    play("Cellar");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));

    GameState clone = state();
    REQUIRE_FALSE(Game::step(state(), select_action(DEF_COPPER)));
    REQUIRE_FALSE(Game::step(clone, select_action(DEF_COPPER)));
    REQUIRE(same_bytes(state(), clone));

    expect_conservation();
}

CARD_SPEC("v2 Remodel with no cards to trash auto-completes") {
    given().hand("Remodel");

    play("Remodel");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::PhaseBuy));
    expect().coins(0);
    expect_conservation();
}

CARD_SPEC("v2 Mine with no treasures auto-completes") {
    given().hand("Mine", "Estate");

    play("Mine");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::PhaseBuy));
    expect().hand_has("Estate").trash_count("Estate", 0).coins(0);
    expect_conservation();
}

CARD_SPEC("v2 mandatory Choose clamps min to available selections") {
    given().hand("ExactTwoTest", "Copper");

    play("ExactTwoTest");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.source == DEF_EXACT_TWO_TEST);
    REQUIRE(state().decision.min_left == 1U);
    REQUIRE(state().decision.max_left == 1U);

    choose("Copper");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::PhaseBuy));
    expect().trash_count("Copper", 1);
    expect_conservation();
}

CARD_SPEC("v2 ChooseGain auto-completes when no supply pile matches") {
    given().hand("Remodel", "Estate", "Gold");
    empty_supply_up_to(4);

    play("Remodel");
    choose("Estate");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::PhaseBuy));
    expect().hand_has("Gold").trash_count("Estate", 1).discard_lacks("Silver").discard_lacks("Copper");
    expect_conservation();
}

CARD_SPEC("v2 Repeat runs a program containing Choose the specified number of times") {
    given().hand("RepeatChooseTest", "Copper", "Estate");

    play("RepeatChooseTest");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.source == DEF_REPEAT_CHOOSE_TEST);

    choose("Copper");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().coins == 1);

    choose("Estate");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::PhaseBuy));
    expect().discard_has("Copper").discard_has("Estate").coins(2);
    expect_conservation();
}
