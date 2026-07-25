#include "v2/encode/encoder.h"

#include "v2/core/actions.h"
#include "v2/core/defs.h"
#include "v2/core/game.h"
#include "v2/core/turns.h"
#include "v2/drivers/bots.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

[[nodiscard]] Setup encoder_setup() noexcept {
    Setup setup{};
    setup.num_players = 2;
    setup.kingdom_count = 10;
    setup.kingdom[0] = DEF_SENTRY;
    setup.kingdom[1] = DEF_LIBRARY;
    setup.kingdom[2] = DEF_THRONE_ROOM;
    setup.kingdom[3] = DEF_BANDIT;
    setup.kingdom[4] = DEF_WITCH;
    setup.kingdom[5] = DEF_MOAT;
    setup.kingdom[6] = DEF_VILLAGE;
    setup.kingdom[7] = DEF_SMITHY;
    setup.kingdom[8] = DEF_MARKET;
    setup.kingdom[9] = DEF_REMODEL;
    return setup;
}

[[nodiscard]] std::array<float, OBS_SIZE> encoded(const GameState& state, PlayerId player) noexcept {
    std::array<float, OBS_SIZE> out{};
    encode(state, player, out.data());
    return out;
}

[[nodiscard]] std::array<float, OBS_SIZE_V2> encoded_v2(const GameState& state, PlayerId player) noexcept {
    std::array<float, OBS_SIZE_V2> out{};
    encode_v2(state, player, out.data());
    return out;
}

[[nodiscard]] std::array<float, OBS_SIZE_V3> encoded_v3(const GameState& state, PlayerId player) noexcept {
    std::array<float, OBS_SIZE_V3> out{};
    encode_v3(state, player, out.data());
    return out;
}

void clear_player_cards(GameState& state, PlayerId player_id) {
    PlayerState& player = state.players[player_id];
    for (std::uint8_t slot = 0; slot < MAX_SLOTS; ++slot) {
        player.hand[slot] = 0;
    }
    player.deck.size = 0;
    player.discard.size = 0;
    player.set_aside.size = 0;
    player.in_play_size = 0;
}

void advance_to_player_one_buy(GameState& state) {
    REQUIRE(Game::step(state, A_PASS) == false);
    REQUIRE(Game::step(state, A_PASS) == false);
    REQUIRE(current_player(state) == 1U);
    REQUIRE(state.phase == static_cast<std::uint8_t>(Phase::Buy));
}

void step_random(GameState& state, RandomBot& bot) {
    ActionMask legal{};
    const int legal_count = Game::legal_actions(state, legal);
    REQUIRE(legal_count > 0);
    const Action action = bot.choose_action(state, legal, legal_count);
    (void)Game::step(state, action);
}

} // namespace

TEST_CASE("v2 encoder clone output matches original at decision points", "[v2][encode]") {
    GameState state = Game::new_game(encoder_setup(), 0xE4C0'0001ULL);
    RandomBot bot(0xE4C0'1001ULL);

    for (int i = 0; i < 40; ++i) {
        const GameState clone = state;
        CHECK(encoded(clone, 0U) == encoded(state, 0U));
        CHECK(encoded(clone, 1U) == encoded(state, 1U));
        if (state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            break;
        }
        step_random(state, bot);
    }
}

TEST_CASE("v2 encoder is deterministic over identical seeded play", "[v2][encode]") {
    GameState a = Game::new_game(encoder_setup(), 0xE4C0'0002ULL);
    GameState b = Game::new_game(encoder_setup(), 0xE4C0'0002ULL);
    RandomBot bot_a(0xE4C0'2002ULL);
    RandomBot bot_b(0xE4C0'2002ULL);

    for (int i = 0; i < 40; ++i) {
        CHECK(encoded(a, 0U) == encoded(b, 0U));
        CHECK(encoded(a, 1U) == encoded(b, 1U));
        if (a.phase == static_cast<std::uint8_t>(Phase::Over)) {
            break;
        }
        step_random(a, bot_a);
        step_random(b, bot_b);
    }
}

TEST_CASE("v2 encoder excludes opponent hand composition", "[v2][encode]") {
    GameState a = Game::new_game(encoder_setup(), 0xE4C0'0003ULL);
    GameState b = a;

    const Slot copper = slot_of(a, DEF_COPPER);
    const Slot estate = slot_of(a, DEF_ESTATE);
    REQUIRE(copper != NONE);
    REQUIRE(estate != NONE);

    for (std::uint8_t slot = 0; slot < MAX_SLOTS; ++slot) {
        a.players[1].hand[slot] = 0;
        b.players[1].hand[slot] = 0;
    }
    a.players[1].hand[copper] = 5;
    b.players[1].hand[estate] = 5;

    CHECK(encoded(a, 0U) == encoded(b, 0U));
    CHECK(encoded(a, 1U) != encoded(b, 1U));
}

TEST_CASE("v2 encoder exposes version and expected shape", "[v2][encode]") {
    const GameState state = Game::new_game(Setup{}, 0xE4C0'0004ULL);
    const auto obs = encoded(state, 0U);
    const auto obs_v2 = encoded_v2(state, 0U);
    CHECK(obs[OBS_META_OFFSET] == static_cast<float>(OBS_VERSION));
    CHECK(obs[OBS_META_OFFSET + 1U] == static_cast<float>(OBS_SIZE));
    CHECK(OBS_SIZE == OBS_SIZE_V1);
    CHECK(OBS_SIZE_V1 == 1141U);
    CHECK(OBS_SIZE_V2 == 1717U);
    CHECK(OBS_SIZE_V3 == 1788U);
    CHECK(obs_size_for(ObsVersion::V1) == OBS_SIZE_V1);
    CHECK(obs_size_for(ObsVersion::V2) == OBS_SIZE_V2);
    CHECK(obs_size_for(ObsVersion::V3) == OBS_SIZE_V3);
    CHECK(obs_v2[OBS_V2_META_OFFSET] == static_cast<float>(ObsVersion::V2));
    CHECK(obs_v2[OBS_V2_META_OFFSET + 1U] == static_cast<float>(OBS_SIZE_V2));
}

TEST_CASE("v3 encoder appends trash and select semantics without changing the v2 prefix", "[v2][encode]") {
    Setup setup{};
    setup.kingdom_count = 1U;
    setup.kingdom[0] = DEF_CHAPEL;
    GameState state = Game::new_game(setup, 0xE4C0'5001ULL);
    advance_to_player_one_buy(state);
    state.phase = static_cast<std::uint8_t>(Phase::Action);
    state.actions = 1U;
    state.buys = 1U;
    state.coins = 0;
    refresh_current_decision(state);
    clear_player_cards(state, 1U);

    const Slot gold = slot_of(state, DEF_GOLD);
    const Slot chapel = slot_of(state, DEF_CHAPEL);
    REQUIRE(gold != NONE);
    REQUIRE(chapel != NONE);
    state.players[1].hand[gold] = 1U;
    state.players[1].hand[chapel] = 1U;

    REQUIRE(Game::step(state, play_action(DEF_CHAPEL)) == false);
    REQUIRE(state.decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state.decision.select_semantic == static_cast<std::uint8_t>(SelectSemantic::Trash));

    const auto v2 = encoded_v2(state, 0U);
    const auto v3 = encoded_v3(state, 0U);
    std::array<float, OBS_SIZE_V3> dispatched{};
    encode(state, 0U, dispatched.data(), ObsVersion::V3);

    CHECK(v3[OBS_V2_META_OFFSET] == static_cast<float>(ObsVersion::V3));
    CHECK(v3[OBS_V2_META_OFFSET + 1U] == static_cast<float>(OBS_SIZE_V3));
    CHECK(std::memcmp(v2.data() + 2U, v3.data() + 2U, (OBS_SIZE_V2 - 2U) * sizeof(float)) == 0);
    CHECK(std::memcmp(v3.data(), dispatched.data(), sizeof(v3)) == 0);
    for (std::uint8_t semantic = 0; semantic < OBS_SELECT_SEMANTIC_COUNT; ++semantic) {
        CHECK(v3[OBS_V3_SELECT_SEMANTIC_OFFSET + semantic]
              == (semantic == static_cast<std::uint8_t>(SelectSemantic::Trash) ? 1.0F : 0.0F));
    }

    REQUIRE(Game::step(state, select_action(DEF_GOLD)) == false);
    REQUIRE(state.trash[gold] == 1U);
    const auto after_trash = encoded_v3(state, 0U);
    for (std::uint8_t slot = 0; slot < MAX_SLOTS; ++slot) {
        CHECK(after_trash[OBS_V3_TRASH_OFFSET + slot] == static_cast<float>(state.trash[slot]));
    }

    Setup militia_setup{};
    militia_setup.kingdom_count = 1U;
    militia_setup.kingdom[0] = DEF_MILITIA;
    GameState militia = Game::new_game(militia_setup, 0xE4C0'5002ULL);
    advance_to_player_one_buy(militia);
    militia.phase = static_cast<std::uint8_t>(Phase::Action);
    militia.actions = 1U;
    militia.buys = 1U;
    militia.coins = 0;
    refresh_current_decision(militia);
    clear_player_cards(militia, 0U);
    clear_player_cards(militia, 1U);

    const Slot copper = slot_of(militia, DEF_COPPER);
    const Slot militia_slot = slot_of(militia, DEF_MILITIA);
    REQUIRE(copper != NONE);
    REQUIRE(militia_slot != NONE);
    militia.players[0].hand[copper] = 5U;
    militia.players[1].hand[militia_slot] = 1U;

    REQUIRE(Game::step(militia, play_action(DEF_MILITIA)) == false);
    REQUIRE(militia.decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(militia.decision.select_semantic == static_cast<std::uint8_t>(SelectSemantic::Keep));
    const auto militia_v3 = encoded_v3(militia, 0U);
    for (std::uint8_t semantic = 0; semantic < OBS_SELECT_SEMANTIC_COUNT; ++semantic) {
        CHECK(militia_v3[OBS_V3_SELECT_SEMANTIC_OFFSET + semantic]
              == (semantic == static_cast<std::uint8_t>(SelectSemantic::Keep) ? 1.0F : 0.0F));
    }
}

TEST_CASE("v2 encoder v1 dispatch is byte-identical to the retained v1 implementation", "[v2][encode]") {
    GameState state = Game::new_game(encoder_setup(), 0xE4C0'1001ULL);
    RandomBot bot(0xE4C0'2001ULL);

    for (int i = 0; i < 24; ++i) {
        std::array<float, OBS_SIZE_V1> direct{};
        std::array<float, OBS_SIZE_V1> dispatched{};
        std::array<float, OBS_SIZE_V1> legacy_default{};
        encode_v1(state, 0U, direct.data());
        encode(state, 0U, dispatched.data(), ObsVersion::V1);
        encode(state, 0U, legacy_default.data());
        CHECK(std::memcmp(direct.data(), dispatched.data(), sizeof(direct)) == 0);
        CHECK(std::memcmp(direct.data(), legacy_default.data(), sizeof(direct)) == 0);
        if (state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            break;
        }
        step_random(state, bot);
    }
}

TEST_CASE("v2 encoder exposes opponent public-memory compositions by Slot", "[v2][encode]") {
    GameState state = Game::new_game(encoder_setup(), 0xE4C0'3001ULL);
    PlayerState& opponent = state.players[1];
    clear_player_cards(state, 1U);

    const Slot copper = slot_of(state, DEF_COPPER);
    const Slot estate = slot_of(state, DEF_ESTATE);
    const Slot gold = slot_of(state, DEF_GOLD);
    const Slot village = slot_of(state, DEF_VILLAGE);
    REQUIRE(copper != NONE);
    REQUIRE(estate != NONE);
    REQUIRE(gold != NONE);
    REQUIRE(village != NONE);

    opponent.hand[copper] = 2U;
    opponent.hand[gold] = 1U;
    opponent.deck.cards[opponent.deck.size++] = estate;
    opponent.deck.cards[opponent.deck.size++] = gold;
    opponent.discard.cards[opponent.discard.size++] = copper;
    opponent.discard.cards[opponent.discard.size++] = estate;
    opponent.set_aside.cards[opponent.set_aside.size++] = estate;
    opponent.in_play[opponent.in_play_size++].slot = gold;
    opponent.in_play[opponent.in_play_size++].slot = village;

    const auto v1 = encoded(state, 0U);
    const auto v2 = encoded_v2(state, 0U);
    const std::size_t block = OBS_V2_OPPONENT_OFFSET;
    const std::size_t collection = block + OBS_V2_OPPONENT_COLLECTION_OFFSET;
    const std::size_t discard = block + OBS_V2_OPPONENT_DISCARD_OFFSET;
    const std::size_t set_aside = block + OBS_V2_OPPONENT_SET_ASIDE_OFFSET;

    CHECK(std::memcmp(
        v1.data() + OBS_OPPONENT_OFFSET,
        v2.data() + OBS_V2_OPPONENT_OFFSET,
        OBS_OPPONENT_BLOCK_SIZE_V1 * sizeof(float)) == 0);
    CHECK(v2[collection + copper] == 3.0F);
    CHECK(v2[collection + estate] == 3.0F);
    CHECK(v2[collection + gold] == 3.0F);
    CHECK(v2[collection + village] == 1.0F);
    CHECK(v2[discard + copper] == 1.0F);
    CHECK(v2[discard + estate] == 1.0F);
    CHECK(v2[discard + gold] == 0.0F);
    CHECK(v2[set_aside + estate] == 1.0F);
    CHECK(v2[set_aside + copper] == 0.0F);

    const std::size_t unused = OBS_V2_OPPONENT_OFFSET + OBS_OPPONENT_BLOCK_SIZE_V2;
    for (std::size_t index = 0; index < OBS_OPPONENT_BLOCK_SIZE_V2; ++index) {
        CHECK(v2[unused + index] == 0.0F);
    }
}

TEST_CASE("v2 opponent memory tracks public gains, trashing, and discards", "[v2][encode]") {
    {
        GameState state = Game::new_game(Setup{}, 0xE4C0'4001ULL);
        advance_to_player_one_buy(state);
        clear_player_cards(state, 1U);
        const Slot gold = slot_of(state, DEF_GOLD);
        const Slot copper = slot_of(state, DEF_COPPER);
        REQUIRE(gold != NONE);
        REQUIRE(copper != NONE);
        state.players[1].hand[gold] = 2U;
        for (std::uint8_t index = 0; index < 5U; ++index) {
            state.players[1].deck.cards[state.players[1].deck.size++] = copper;
        }

        const auto before = encoded_v2(state, 0U);
        const std::size_t collection = OBS_V2_OPPONENT_OFFSET + OBS_V2_OPPONENT_COLLECTION_OFFSET + gold;
        const std::size_t discard = OBS_V2_OPPONENT_OFFSET + OBS_V2_OPPONENT_DISCARD_OFFSET + gold;
        REQUIRE(Game::step(state, play_action(DEF_GOLD)) == false);
        REQUIRE(Game::step(state, play_action(DEF_GOLD)) == false);
        REQUIRE(Game::step(state, buy_action(DEF_GOLD)) == false);
        const auto after_gain = encoded_v2(state, 0U);
        CHECK(after_gain[collection] == before[collection] + 1.0F);
        CHECK(after_gain[discard] == 3.0F);
    }

    {
        Setup setup{};
        setup.kingdom_count = 1U;
        setup.kingdom[0] = DEF_CHAPEL;
        GameState state = Game::new_game(setup, 0xE4C0'4002ULL);
        advance_to_player_one_buy(state);
        state.phase = static_cast<std::uint8_t>(Phase::Action);
        state.actions = 1U;
        state.buys = 1U;
        state.coins = 0;
        refresh_current_decision(state);
        clear_player_cards(state, 1U);
        const Slot gold = slot_of(state, DEF_GOLD);
        const Slot chapel = slot_of(state, DEF_CHAPEL);
        REQUIRE(gold != NONE);
        REQUIRE(chapel != NONE);
        state.players[1].hand[gold] = 1U;
        state.players[1].hand[chapel] = 1U;

        const auto before = encoded_v2(state, 0U);
        const std::size_t collection = OBS_V2_OPPONENT_OFFSET + OBS_V2_OPPONENT_COLLECTION_OFFSET + gold;
        REQUIRE(Game::step(state, play_action(DEF_CHAPEL)) == false);
        REQUIRE(Game::step(state, select_action(DEF_GOLD)) == false);
        const auto after_trash = encoded_v2(state, 0U);
        CHECK(after_trash[collection] == before[collection] - 1.0F);
        CHECK(state.trash[gold] == 1U);
    }
}
