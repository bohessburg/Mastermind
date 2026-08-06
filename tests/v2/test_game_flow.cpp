#include "v2/core/actions.h"
#include "v2/core/game.h"
#include "v2/core/turns.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace {

[[nodiscard]] Pile* find_pile(GameState& state, DefId def) {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        Pile& pile = state.piles[i];
        if (pile.mixed_len == 0U && state.slot_to_def[pile.base] == def) {
            return &pile;
        }
    }
    return nullptr;
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

[[nodiscard]] std::uint8_t hand_size(const PlayerState& player, std::uint8_t slots) {
    std::uint8_t count = 0;
    for (std::uint8_t slot = 0; slot < slots; ++slot) {
        count = static_cast<std::uint8_t>(count + player.hand[slot]);
    }
    return count;
}

void give_hand(GameState& state, PlayerId player_id, DefId def, std::uint8_t count) {
    const Slot slot = slot_of(state, def);
    clear_player_cards(state, player_id);
    state.players[player_id].hand[slot] = count;
}

void set_deck(GameState& state, PlayerId player_id, DefId def, std::uint8_t count) {
    const Slot slot = slot_of(state, def);
    PlayerState& player = state.players[player_id];
    player.deck.size = 0;
    for (std::uint8_t i = 0; i < count; ++i) {
        player.deck.cards[player.deck.size] = slot;
        ++player.deck.size;
    }
}

} // namespace

TEST_CASE("v2 phase flow plays treasures buys and cleans up to next player", "[v2][flow]") {
    GameState state = Game::new_game(Setup{}, 42U);
    give_hand(state, 0U, DEF_GOLD, 3U);
    set_deck(state, 0U, DEF_COPPER, 5U);

    REQUIRE(Game::step(state, A_PASS) == false);
    REQUIRE(state.phase == static_cast<std::uint8_t>(Phase::Buy));

    REQUIRE(Game::step(state, play_action(DEF_GOLD)) == false);
    REQUIRE(Game::step(state, play_action(DEF_GOLD)) == false);
    REQUIRE(Game::step(state, play_action(DEF_GOLD)) == false);
    REQUIRE(state.coins == 9);
    REQUIRE(state.players[0].in_play_size == 3U);

    REQUIRE(Game::step(state, buy_action(DEF_PROVINCE)) == false);

    REQUIRE(current_player(state) == 1U);
    REQUIRE(state.phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(Game::current_decision(state).kind == static_cast<std::uint8_t>(DecisionKind::PhaseBuy));
    REQUIRE(state.players[0].in_play_size == 0U);
    REQUIRE(hand_size(state.players[0], state.num_slots) == 5U);
    REQUIRE(find_pile(state, DEF_PROVINCE)->count == 7U);
}

TEST_CASE("v2 game reaches over when Province pile empties", "[v2][flow]") {
    GameState state = Game::new_game(Setup{}, 77U);
    give_hand(state, 0U, DEF_GOLD, 3U);
    find_pile(state, DEF_PROVINCE)->count = 1U;

    REQUIRE(Game::step(state, A_PASS) == false);
    REQUIRE(Game::step(state, play_action(DEF_GOLD)) == false);
    REQUIRE(Game::step(state, play_action(DEF_GOLD)) == false);
    REQUIRE(Game::step(state, play_action(DEF_GOLD)) == false);
    REQUIRE(Game::step(state, buy_action(DEF_PROVINCE)) == true);

    REQUIRE(state.phase == static_cast<std::uint8_t>(Phase::Over));
    REQUIRE(state.truncated == 0U);
    REQUIRE(state.turn_counter == 1U);
}

TEST_CASE("v2 auto-advances forced pass decisions", "[v2][flow]") {
    GameState state = Game::new_game(Setup{}, 91U);
    give_hand(state, 0U, DEF_GOLD, 2U);

    REQUIRE(Game::step(state, A_PASS) == false);
    REQUIRE(Game::step(state, play_action(DEF_GOLD)) == false);
    REQUIRE(Game::step(state, play_action(DEF_GOLD)) == false);
    REQUIRE(Game::step(state, buy_action(DEF_GOLD)) == false);

    REQUIRE(current_player(state) == 1U);
    REQUIRE(state.phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(state.decision.player == 1U);
}

TEST_CASE("v2 turn cap sets truncated at end of turn", "[v2][flow]") {
    GameState state = Game::new_game(Setup{}, 222U);
    give_hand(state, 0U, DEF_COPPER, 0U);
    state.phase = static_cast<std::uint8_t>(Phase::Buy);
    state.turn_counter = static_cast<std::uint16_t>(MAX_TURNS - 1U);
    refresh_current_decision(state);

    REQUIRE(Game::step(state, A_PASS) == true);

    REQUIRE(state.phase == static_cast<std::uint8_t>(Phase::Over));
    REQUIRE(state.truncated == 1U);
    REQUIRE(state.turn_counter == MAX_TURNS);
}
