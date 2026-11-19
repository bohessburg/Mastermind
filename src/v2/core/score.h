#pragma once

#include "v2/core/state.h"
#include "v2/core/types.h"

#include <cstdint>

[[nodiscard]] std::int16_t score(const GameState& state, PlayerId player) noexcept;

// Winner by VP. For a game that ended normally (Phase::Over, not truncated)
// equal VP is broken by fewer turns played; any remaining tie -> NONE.
// Truncated or in-progress states compare VP only.
[[nodiscard]] PlayerId game_winner(const GameState& state) noexcept;
