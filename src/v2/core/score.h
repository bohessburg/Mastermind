#pragma once

#include "v2/core/state.h"
#include "v2/core/types.h"

#include <cstdint>

[[nodiscard]] std::int16_t score(const GameState& state, PlayerId player) noexcept;
