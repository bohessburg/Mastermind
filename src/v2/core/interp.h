#pragma once

#include "v2/core/actions.h"
#include "v2/core/defs.h"
#include "v2/core/state.h"
#include "v2/core/types.h"

#include <cstdint>

void draw_cards(GameState& state, PlayerId player, std::uint8_t count) noexcept;
bool push_effect(GameState& state, DefId source, PlayerId player) noexcept;
RunResult interp_run(GameState& state) noexcept;
void interp_resume(GameState& state, Action action) noexcept;
