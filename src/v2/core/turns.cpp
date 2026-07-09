#include "v2/core/turns.h"

#include "v2/core/defs.h"
#include "v2/core/interp.h"

#include <cassert>
#include <cstdint>

namespace {

void append_discard(PlayerState& player, Slot slot) noexcept {
    assert(player.discard.size < MAX_DECK_CARDS);
    player.discard.cards[player.discard.size] = slot;
    ++player.discard.size;
}

void discard_hand(PlayerState& player) noexcept {
    for (std::uint8_t slot = 0; slot < MAX_SLOTS; ++slot) {
        const std::uint8_t count = player.hand[slot];
        for (std::uint8_t i = 0; i < count; ++i) {
            append_discard(player, slot);
        }
        player.hand[slot] = 0;
    }
}

void discard_in_play(PlayerState& player) noexcept {
    for (std::uint8_t i = 0; i < player.in_play_size; ++i) {
        append_discard(player, player.in_play[i].slot);
        player.in_play[i] = InPlayEntry{};
    }
    player.in_play_size = 0;
}

[[nodiscard]] bool pile_top_def(const GameState& state, const Pile& pile, DefId& out) noexcept {
    if (pile.mixed_len > 0U) {
        const Slot slot = pile.mixed[pile.mixed_len - 1U];
        out = state.slot_to_def[slot];
        return true;
    }
    if (pile.count > 0U) {
        out = state.slot_to_def[pile.base];
        return true;
    }
    return false;
}

[[nodiscard]] bool pile_for_def_empty(const GameState& state, DefId def) noexcept {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        DefId top = 0;
        if (pile_top_def(state, state.piles[i], top) && top == def) {
            return false;
        }
        if (state.piles[i].mixed_len == 0U && state.slot_to_def[state.piles[i].base] == def) {
            return state.piles[i].count == 0U;
        }
    }
    return false;
}

[[nodiscard]] bool has_pile_for_def(const GameState& state, DefId def) noexcept {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        if (state.piles[i].mixed_len == 0U && state.slot_to_def[state.piles[i].base] == def) {
            return true;
        }
        DefId top = 0;
        if (pile_top_def(state, state.piles[i], top) && top == def) {
            return true;
        }
    }
    return false;
}

} // namespace

void turn_queue_clear(TurnQueue& queue) noexcept {
    queue = TurnQueue{};
}

bool turn_queue_push(TurnQueue& queue, PlayerId player, TurnKind kind) noexcept {
    if (queue.size >= MAX_TURN_QUEUE) {
        return false;
    }
    const std::uint8_t index = static_cast<std::uint8_t>((queue.head + queue.size) % MAX_TURN_QUEUE);
    queue.entries[index] = TurnQueueEntry{player, kind};
    ++queue.size;
    return true;
}

bool turn_queue_pop(TurnQueue& queue, TurnQueueEntry& out) noexcept {
    if (queue.size == 0U) {
        return false;
    }
    out = queue.entries[queue.head];
    queue.entries[queue.head] = TurnQueueEntry{};
    queue.head = static_cast<std::uint8_t>((queue.head + 1U) % MAX_TURN_QUEUE);
    --queue.size;
    if (queue.size == 0U) {
        queue.head = 0;
    }
    return true;
}

TurnQueueEntry turn_queue_front(const TurnQueue& queue) noexcept {
    assert(queue.size > 0U);
    if (queue.size == 0U) {
        return TurnQueueEntry{};
    }
    return queue.entries[queue.head];
}

PlayerId current_player(const GameState& state) noexcept {
    return turn_queue_front(state.turn_queue).player;
}

void start_turn(GameState& state, PlayerId player) noexcept {
    state.phase = static_cast<std::uint8_t>(Phase::Action);
    state.actions = 1;
    state.buys = 1;
    state.coins = 0;
    state.potion_coins = 0;
    state.decision = PendingDecision{player, static_cast<std::uint8_t>(DecisionKind::PhaseAction), 0, 0, 0};
}

void cleanup_current_turn(GameState& state) noexcept {
    const PlayerId player_id = current_player(state);
    PlayerState& player = state.players[player_id];
    discard_in_play(player);
    discard_hand(player);
    draw_cards(state, player_id, 5U);

    state.actions = 0;
    state.buys = 0;
    state.coins = 0;
    state.potion_coins = 0;
    state.effect_depth = 0;
    state.decision = PendingDecision{};
}

bool is_game_over(const GameState& state) noexcept {
    if (pile_for_def_empty(state, DEF_PROVINCE)) {
        return true;
    }
    if (has_pile_for_def(state, DEF_COLONY) && pile_for_def_empty(state, DEF_COLONY)) {
        return true;
    }

    std::uint8_t empty_piles = 0;
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        if (state.piles[i].mixed_len == 0U && state.piles[i].count == 0U) {
            ++empty_piles;
        }
    }

    return empty_piles >= 3U;
}

bool advance_turn_machinery(GameState& state) noexcept {
    if (state.phase != static_cast<std::uint8_t>(Phase::Cleanup)) {
        return false;
    }

    const PlayerId completed_player = current_player(state);
    cleanup_current_turn(state);
    ++state.turn_counter;

    if (state.turn_counter >= MAX_TURNS) {
        state.truncated = 1U;
        state.phase = static_cast<std::uint8_t>(Phase::Over);
        return false;
    }
    if (is_game_over(state)) {
        state.phase = static_cast<std::uint8_t>(Phase::Over);
        return false;
    }

    TurnQueueEntry completed{};
    const bool popped = turn_queue_pop(state.turn_queue, completed);
    (void)popped;
    assert(popped);
    if (state.turn_queue.size == 0U) {
        const PlayerId next_player = static_cast<PlayerId>((completed_player + 1U) % state.num_players);
        const bool pushed = turn_queue_push(state.turn_queue, next_player, TurnKind::Normal);
        (void)pushed;
        assert(pushed);
    }
    start_turn(state, current_player(state));
    return true;
}
