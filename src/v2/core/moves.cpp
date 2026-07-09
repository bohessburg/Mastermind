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

[[nodiscard]] bool remove_from_deck(PlayerState& player, Slot slot) noexcept {
    if (player.deck.size == 0U || player.deck.cards[player.deck.size - 1U] != slot) {
        return false;
    }
    --player.deck.size;
    player.deck.cards[player.deck.size] = 0;
    return true;
}

[[nodiscard]] bool remove_from_discard(PlayerState& player, Slot slot) noexcept {
    for (std::uint8_t i = player.discard.size; i > 0U; --i) {
        const std::uint8_t index = static_cast<std::uint8_t>(i - 1U);
        if (player.discard.cards[index] != slot) {
            continue;
        }
        for (std::uint8_t j = index; static_cast<std::uint8_t>(j + 1U) < player.discard.size; ++j) {
            player.discard.cards[j] = player.discard.cards[j + 1U];
        }
        --player.discard.size;
        player.discard.cards[player.discard.size] = 0;
        return true;
    }
    return false;
}

[[nodiscard]] bool remove_from_ordered(OrderedZone& zone, Slot slot) noexcept {
    for (std::uint8_t i = zone.size; i > 0U; --i) {
        const std::uint8_t index = static_cast<std::uint8_t>(i - 1U);
        if (zone.cards[index] != slot) {
            continue;
        }
        for (std::uint8_t j = index; static_cast<std::uint8_t>(j + 1U) < zone.size; ++j) {
            zone.cards[j] = zone.cards[j + 1U];
        }
        --zone.size;
        zone.cards[zone.size] = 0;
        return true;
    }
    return false;
}

[[nodiscard]] bool remove_from_zone(PlayerState& player, Slot slot, MoveZone from_zone) noexcept {
    switch (from_zone) {
    case MoveZone::Hand:
        return remove_from_hand(player, slot);
    case MoveZone::InPlay:
        return remove_from_in_play(player, slot);
    case MoveZone::Deck:
        return remove_from_deck(player, slot);
    case MoveZone::Discard:
        return remove_from_discard(player, slot);
    case MoveZone::SetAside:
        return remove_from_ordered(player.set_aside, slot);
    case MoveZone::Revealed:
        return true;
    }
    return false;
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

[[nodiscard]] bool gain_from_pile(
    GameState& state,
    PlayerId player,
    Pile& pile,
    GainDestination destination) noexcept {
    if (!pile_has_cards(pile)) {
        return false;
    }

    const Slot slot = pile_top_slot(pile);
    const DefId def = state.slot_to_def[slot];
    const bool has_triggers = !trigger_table_clean_empty(state);
    if (has_triggers) {
        emit(state, TriggerKind::WouldGain, TriggerPayload{player, def, slot, static_cast<std::uint8_t>(destination)});
    }
    decrement_pile(pile);
    gain_to_destination(state.players[player], slot, destination);
    if (has_triggers) {
        emit(state, TriggerKind::OnGain, TriggerPayload{player, def, slot, static_cast<std::uint8_t>(destination)});
    }
    return true;
}

} // namespace

bool do_gain(GameState& state, PlayerId player, Slot slot, GainDestination destination) noexcept {
    assert(player < state.num_players);
    Pile* pile = find_pile_with_top_slot(state, slot);
    if (pile == nullptr) {
        return false;
    }

    return gain_from_pile(state, player, *pile, destination);
}

bool do_gain_from_supply_pile(
    GameState& state,
    PlayerId player,
    std::uint8_t pile_index,
    GainDestination destination) noexcept {
    assert(player < state.num_players);
    if (pile_index >= state.num_piles) {
        return false;
    }
    return gain_from_pile(state, player, state.piles[pile_index], destination);
}

bool do_trash(GameState& state, PlayerId player, Slot slot, MoveZone from_zone) noexcept {
    assert(player < state.num_players);
    if (!remove_from_zone(state.players[player], slot, from_zone)) {
        return false;
    }

    if (from_zone == MoveZone::InPlay && card_def(state.slot_to_def[slot]).trigger_mask != 0U) {
        mark_trigger_table_dirty(state);
    }

    ++state.trash[slot];
    if (!trigger_table_clean_empty(state)) {
        emit(state, TriggerKind::OnTrash, TriggerPayload{player, state.slot_to_def[slot], slot, 0U});
    }
    return true;
}

bool do_discard(GameState& state, PlayerId player, Slot slot, MoveZone from_zone) noexcept {
    assert(player < state.num_players);
    if (!remove_from_zone(state.players[player], slot, from_zone)) {
        return false;
    }

    if (from_zone == MoveZone::InPlay && card_def(state.slot_to_def[slot]).trigger_mask != 0U) {
        mark_trigger_table_dirty(state);
    }

    append_discard(state.players[player], slot);
    if (!trigger_table_clean_empty(state)) {
        emit(state, TriggerKind::OnDiscard, TriggerPayload{player, state.slot_to_def[slot], slot, 0U});
    }
    return true;
}

void do_discard_all_hand(GameState& state, PlayerId player) noexcept {
    assert(player < state.num_players);
    if (!trigger_table_clean_empty(state)) {
        for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
            const std::uint8_t count = state.players[player].hand[slot];
            for (std::uint8_t i = 0; i < count; ++i) {
                const bool discarded = do_discard(state, player, slot, MoveZone::Hand);
                (void)discarded;
                assert(discarded);
            }
        }
        return;
    }

    PlayerState& player_state = state.players[player];
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        const std::uint8_t count = player_state.hand[slot];
        player_state.hand[slot] = 0;
        for (std::uint8_t i = 0; i < count; ++i) {
            append_discard(player_state, slot);
        }
    }
}

void do_discard_all_in_play(GameState& state, PlayerId player) noexcept {
    assert(player < state.num_players);
    if (!trigger_table_clean_empty(state)) {
        while (state.players[player].in_play_size > 0U) {
            const Slot slot = state.players[player].in_play[0].slot;
            const bool discarded = do_discard(state, player, slot, MoveZone::InPlay);
            (void)discarded;
            assert(discarded);
        }
        return;
    }

    PlayerState& player_state = state.players[player];
    for (std::uint8_t i = 0; i < player_state.in_play_size; ++i) {
        append_discard(player_state, player_state.in_play[i].slot);
        player_state.in_play[i] = InPlayEntry{};
    }
    player_state.in_play_size = 0;
}
