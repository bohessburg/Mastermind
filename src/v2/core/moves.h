#pragma once

#include "v2/core/defs.h"
#include "v2/core/state.h"
#include "v2/core/types.h"

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
[[nodiscard]] bool do_trash(GameState& state, PlayerId player, Slot slot, MoveZone from_zone) noexcept;
[[nodiscard]] bool do_discard(GameState& state, PlayerId player, Slot slot, MoveZone from_zone) noexcept;
