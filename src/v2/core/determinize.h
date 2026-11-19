#pragma once

#include "v2/core/state.h"
#include "v2/core/types.h"

#include <cstdint>

void determinize(GameState& state, PlayerId perspective, std::uint64_t seed) noexcept;
