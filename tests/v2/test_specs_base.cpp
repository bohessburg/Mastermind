#include "spec_harness.h"

#include "v2/core/defs.h"

#include <cstring>

namespace {

[[nodiscard]] bool same_bytes(const GameState& lhs, const GameState& rhs) {
    return std::memcmp(&lhs, &rhs, sizeof(GameState)) == 0;
}

[[nodiscard]] std::uint8_t ordered_count(const OrderedZone& zone, Slot slot) {
    std::uint8_t total = 0;
    for (std::uint8_t i = 0; i < zone.size; ++i) {
        if (zone.cards[i] == slot) {
            ++total;
        }
    }
    return total;
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

CARD_SPEC("v2 Market draws and grants action buy and coin") {
    given().hand("Market").deck("Copper");

    play("Market");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().hand_has("Copper").actions(1).buys(2).coins(1);
    expect_conservation();
}

CARD_SPEC("v2 Festival grants actions buy and coins") {
    given().hand("Festival");

    play("Festival");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().actions(2).buys(2).coins(2);
    expect_conservation();
}

CARD_SPEC("v2 Laboratory draws two and grants an action") {
    given().hand("Laboratory").deck("Copper", "Silver");

    play("Laboratory");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().hand_has("Copper").hand_has("Silver").actions(1).coins(0);
    expect_conservation();
}

CARD_SPEC("v2 Gardens scores one VP per ten owned cards") {
    given().hand(
        "Gardens",
        "Copper", "Copper", "Copper", "Copper", "Copper",
        "Copper", "Copper", "Copper", "Copper", "Copper",
        "Copper", "Copper", "Copper", "Copper", "Copper",
        "Copper", "Copper", "Copper", "Copper");

    expect().score(0U, 2);
    expect_conservation();
}

CARD_SPEC("v2 Moneylender may trash Copper for three coins") {
    given().hand("Moneylender", "Copper", "Estate");

    play("Moneylender");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.source == DEF_MONEYLENDER);
    REQUIRE(state().decision.min_left == 0U);
    REQUIRE(state().decision.max_left == 1U);

    choose("Copper");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().trash_count("Copper", 1).hand_has("Estate").coins(3);
    expect_conservation();
}

CARD_SPEC("v2 Moneylender with no Copper resolves without coins") {
    given().hand("Moneylender", "Estate");

    play("Moneylender");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().hand_has("Estate").trash_count("Copper", 0).coins(0);
    expect_conservation();
}

CARD_SPEC("v2 Poacher with no empty piles discards nothing") {
    given().hand("Poacher", "Estate").deck("Copper");

    play("Poacher");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().hand_has("Copper").hand_has("Estate").actions(1).coins(1);
    expect_conservation();
}

CARD_SPEC("v2 Poacher discards one card for one empty pile") {
    given().hand("Poacher", "Copper", "Estate").deck("Silver");
    empty_supply("Curse");

    play("Poacher");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.source == DEF_POACHER);
    REQUIRE(state().decision.min_left == 1U);
    REQUIRE(state().decision.max_left == 1U);

    choose("Estate");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().discard_has("Estate").hand_has("Copper").hand_has("Silver").coins(1);
    expect_conservation();
}

CARD_SPEC("v2 Poacher discards two cards for two empty piles") {
    given().hand("Poacher", "Copper", "Estate").deck("Silver", "Gold");
    empty_supply("Curse");
    empty_supply("Copper");

    play("Poacher");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.min_left == 2U);
    REQUIRE(state().decision.max_left == 2U);

    choose("Copper");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.min_left == 1U);
    REQUIRE(state().decision.max_left == 1U);

    choose("Estate");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().discard_has("Copper").discard_has("Estate").hand_has("Gold").coins(1);
    expect_conservation();
}

CARD_SPEC("v2 Vassal discards a revealed non-Action") {
    given().hand("Vassal").deck("Estate");

    play("Vassal");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().discard_has("Estate").coins(2);
    expect_conservation();
}

CARD_SPEC("v2 Vassal may decline to play a revealed Action") {
    given().hand("Vassal").deck("Village");

    play("Vassal");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::ChooseOption));
    REQUIRE(state().decision.source == DEF_VASSAL);
    REQUIRE(state().decision.max_left == 2U);

    option(0);

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().discard_has("Village").coins(2);
    expect_conservation();
}

CARD_SPEC("v2 Vassal may play a revealed Action from discard") {
    given().hand("Vassal").deck("Silver", "Village");

    play("Vassal");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::ChooseOption));

    option(1);

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().hand_has("Silver").discard_lacks("Village").actions(2).coins(2);
    expect_conservation();
}

CARD_SPEC("v2 Harbinger with empty discard resolves immediately") {
    given().hand("Harbinger").deck("Copper");

    play("Harbinger");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().hand_has("Copper").actions(1).coins(0);
    expect_conservation();
}

CARD_SPEC("v2 Harbinger may topdeck from discard") {
    given().hand("Harbinger").deck("Silver").discard("Estate", "Copper");

    play("Harbinger");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.source == DEF_HARBINGER);
    REQUIRE(state().decision.min_left == 0U);
    REQUIRE(state().decision.max_left == 1U);

    choose("Copper");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().hand_has("Silver").deck_top("Copper").discard_has("Estate").coins(0);
    expect_conservation();
}

CARD_SPEC("v2 Throne Room with no Actions in hand auto-completes") {
    given().hand("Throne Room", "Copper");

    play("Throne Room");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().hand_has("Copper").coins(0);
    expect_conservation();
}

CARD_SPEC("v2 Throne Room plays Cellar twice") {
    given().hand("Throne Room", "Cellar", "Copper", "Estate").deck("Silver", "Gold");

    play("Throne Room");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.source == DEF_THRONE_ROOM);

    choose("Cellar");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.source == DEF_CELLAR);

    choose("Copper");
    pass();
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.source == DEF_CELLAR);

    choose("Estate");
    pass();

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().hand_has("Gold").hand_has("Silver").discard_has("Copper").discard_has("Estate").actions(2);
    expect_conservation();
}

CARD_SPEC("v2 Throne Room plays Remodel twice") {
    given().hand("Throne Room", "Remodel", "Estate", "Copper");

    play("Throne Room");
    choose("Remodel");

    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.source == DEF_REMODEL);
    choose("Estate");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::ChooseGain));
    gain("Silver");

    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.source == DEF_REMODEL);
    choose("Copper");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::ChooseGain));
    gain("Estate");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().trash_count("Estate", 1).trash_count("Copper", 1).discard_has("Silver").discard_has("Estate");
    expect_conservation();
}

CARD_SPEC("v2 Throne Room chained through Throne Room quadruple plays an Action") {
    given().hand("Throne Room", "Throne Room", "Village").deck("Copper", "Copper", "Copper", "Copper");

    play("Throne Room");
    choose("Throne Room");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.source == DEF_THRONE_ROOM);

    choose("Village");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().hand_has("Copper").actions(8).coins(0);
    expect_conservation();
}

CARD_SPEC("v2 Council Room draws for the player and each other player") {
    given()
        .hand("Council Room")
        .deck("Copper", "Silver", "Gold", "Estate")
        .player_deck(1U, "Copper");

    play("Council Room");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect()
        .hand_has("Copper")
        .hand_has("Silver")
        .hand_has("Gold")
        .hand_has("Estate")
        .player_hand_has(1U, "Copper")
        .buys(2);
    expect_conservation();
}

CARD_SPEC("v2 Artisan gains to hand and topdecks a card from hand") {
    given().hand("Artisan", "Copper");

    play("Artisan");
    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::ChooseGain));
    REQUIRE(state().decision.source == DEF_ARTISAN);
    gain("Duchy");

    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.source == DEF_ARTISAN);
    choose("Copper");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().hand_has("Duchy").deck_top("Copper");
    expect_conservation();
}

CARD_SPEC("v2 Bandit gains Gold and trashes a revealed non-Copper Treasure") {
    given().hand("Bandit").player_deck(1U, "Copper", "Silver");

    play("Bandit");

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    expect().discard_has("Gold").player_discard_has(1U, "Copper").trash_count("Silver", 1);
    expect_conservation();
}

CARD_SPEC("v2 Bandit does not re-reveal its first revealed Treasure after reshuffle") {
    given()
        .hand("Bandit")
        .player_deck(1U, "Silver")
        .player_discard(1U, "Copper", "Copper", "Copper");

    play("Bandit");

    const Slot copper = slot_of(state(), DEF_COPPER);
    const Slot silver = slot_of(state(), DEF_SILVER);
    REQUIRE(copper != NONE);
    REQUIRE(silver != NONE);
    const PlayerState& victim = state().players[1];

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(state().trash[silver] == 1U);
    REQUIRE(ordered_count(victim.deck, copper) == 2U);
    REQUIRE(ordered_count(victim.discard, copper) == 1U);
    REQUIRE(ordered_count(victim.deck, silver) == 0U);
    REQUIRE(ordered_count(victim.discard, silver) == 0U);
    expect().discard_has("Gold");
    expect_conservation();
}

CARD_SPEC("v2 Bandit keeps revealed cards set aside across a trash choice") {
    given()
        .hand("Bandit")
        .player_deck(1U, "Silver")
        .player_discard(1U, "Gold");

    play("Bandit");

    REQUIRE(state().decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state().decision.source == DEF_BANDIT);
    REQUIRE(state().decision.player == 1U);
    expect_conservation();

    ActionMask legal{};
    const int legal_count = Game::legal_actions(state(), legal);
    REQUIRE(legal_count == 2);
    REQUIRE(legal.test(select_action(DEF_SILVER)));
    REQUIRE(legal.test(select_action(DEF_GOLD)));

    choose("Gold");

    const Slot silver = slot_of(state(), DEF_SILVER);
    const Slot gold = slot_of(state(), DEF_GOLD);
    REQUIRE(silver != NONE);
    REQUIRE(gold != NONE);
    const PlayerState& victim = state().players[1];

    REQUIRE(state().phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(state().trash[gold] == 1U);
    REQUIRE(state().trash[silver] == 0U);
    REQUIRE(ordered_count(victim.discard, silver) == 1U);
    REQUIRE(ordered_count(victim.discard, gold) == 0U);
    expect().discard_has("Gold");
    expect_conservation();
}
