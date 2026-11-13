#pragma once

#include "v2/core/defs.h"
#include "v2/core/state.h"
#include "v2/core/types.h"

#include <cstdint>

enum class MoveZone : std::uint8_t {
    Hand,
    InPlay,
    Deck,
    Discard,
    SetAside,
    Revealed,
};

[[nodiscard]] bool do_gain(
    GameState& state,
    PlayerId player,
    Slot slot,
    GainDestination destination) noexcept;
[[nodiscard]] bool do_gain_from_supply_pile(
    GameState& state,
    PlayerId player,
    std::uint8_t pile_index,
    GainDestination destination) noexcept;
[[nodiscard]] bool do_trash(GameState& state, PlayerId player, Slot slot, MoveZone from_zone) noexcept;
[[nodiscard]] bool do_discard(GameState& state, PlayerId player, Slot slot, MoveZone from_zone) noexcept;
void do_discard_all_hand(GameState& state, PlayerId player) noexcept;
void do_discard_all_in_play(GameState& state, PlayerId player) noexcept;
