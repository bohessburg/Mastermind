#include "v2/core/moves.h"

#include "v2/core/setup.h"
#include "v2/core/triggers.h"

#include <cassert>
#include <cstdint>

namespace {

[[nodiscard]] bool pile_has_cards(const Pile& pile) noexcept {
    return pile.mixed_len > 0U || pile.count > 0U;
}

[[nodiscard]] Slot pile_top_slot(const Pile& pile) noexcept {
    if (pile.mixed_len > 0U) {
        return pile.mixed[pile.mixed_len - 1U];
    }
    return pile.base;
}

[[nodiscard]] Pile* find_pile_with_top_slot(GameState& state, Slot slot) noexcept {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        Pile& pile = state.piles[i];
        if (pile_has_cards(pile) && pile_top_slot(pile) == slot) {
            return &pile;
        }
    }
    for (std::uint8_t i = 0; i < state.num_nonsupply; ++i) {
        Pile& pile = state.nonsupply[i];
        if (pile_has_cards(pile) && pile_top_slot(pile) == slot) {
            return &pile;
        }
    }
    return nullptr;
}

void decrement_pile(Pile& pile) noexcept {
    if (pile.mixed_len > 0U) {
        --pile.mixed_len;
    } else {
        assert(pile.count > 0U);
        --pile.count;
    }
}

void append_discard(PlayerState& player, Slot slot) noexcept {
    assert(player.discard.size < MAX_DECK_CARDS);
    player.discard.cards[player.discard.size] = slot;
    ++player.discard.size;
}

void append_topdeck(PlayerState& player, Slot slot) noexcept {
    assert(player.deck.size < MAX_DECK_CARDS);
    player.deck.cards[player.deck.size] = slot;
    ++player.deck.size;
}

[[nodiscard]] bool remove_from_hand(PlayerState& player, Slot slot) noexcept {
    if (player.hand[slot] == 0U) {
        return false;
    }
    --player.hand[slot];
    return true;
}

[[nodiscard]] bool remove_from_in_play(PlayerState& player, Slot slot) noexcept {
    for (std::uint8_t i = 0; i < player.in_play_size; ++i) {
        if (player.in_play[i].slot != slot) {
            continue;
        }
        for (std::uint8_t j = i; static_cast<std::uint8_t>(j + 1U) < player.in_play_size; ++j) {
            player.in_play[j] = player.in_play[j + 1U];
        }
        --player.in_play_size;
        player.in_play[player.in_play_size] = InPlayEntry{};
        return true;
    }
    return false;
}

[[nodiscard]] bool remove_from_zone(PlayerState& player, Slot slot, MoveZone from_zone) noexcept {
    if (from_zone == MoveZone::Hand) {
        return remove_from_hand(player, slot);
    }
    return remove_from_in_play(player, slot);
}

void gain_to_destination(PlayerState& player, Slot slot, GainDestination destination) noexcept {
    switch (destination) {
    case GainDestination::Discard:
        append_discard(player, slot);
        break;
    case GainDestination::Hand:
        ++player.hand[slot];
        break;
    case GainDestination::Topdeck:
        append_topdeck(player, slot);
        break;
    }
}

} // namespace

bool do_gain(GameState& state, PlayerId player, Slot slot, GainDestination destination) noexcept {
    assert(player < state.num_players);
    Pile* pile = find_pile_with_top_slot(state, slot);
    if (pile == nullptr) {
        return false;
    }

    const DefId def = state.slot_to_def[slot];
    emit(state, TriggerKind::WouldGain, TriggerPayload{player, def, slot, static_cast<std::uint8_t>(destination)});
    decrement_pile(*pile);
    gain_to_destination(state.players[player], slot, destination);
    emit(state, TriggerKind::OnGain, TriggerPayload{player, def, slot, static_cast<std::uint8_t>(destination)});
    return true;
}

bool do_trash(GameState& state, PlayerId player, Slot slot, MoveZone from_zone) noexcept {
    assert(player < state.num_players);
    if (!remove_from_zone(state.players[player], slot, from_zone)) {
        return false;
    }

    if (from_zone == MoveZone::InPlay) {
        mark_trigger_table_dirty(state);
    }

    ++state.trash[slot];
    emit(state, TriggerKind::OnTrash, TriggerPayload{player, state.slot_to_def[slot], slot, 0U});
    return true;
}

bool do_discard(GameState& state, PlayerId player, Slot slot, MoveZone from_zone) noexcept {
    assert(player < state.num_players);
    if (!remove_from_zone(state.players[player], slot, from_zone)) {
        return false;
    }

    if (from_zone == MoveZone::InPlay) {
        mark_trigger_table_dirty(state);
    }

    append_discard(state.players[player], slot);
    emit(state, TriggerKind::OnDiscard, TriggerPayload{player, state.slot_to_def[slot], slot, 0U});
    return true;
}
