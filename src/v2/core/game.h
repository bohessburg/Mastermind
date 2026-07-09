#pragma once

#include "v2/core/actions.h"
#include "v2/core/setup.h"
#include "v2/core/state.h"

#include <cstdint>

struct Game {
    static GameState new_game(const Setup& setup, std::uint64_t seed) noexcept;
    static PendingDecision current_decision(const GameState& state) noexcept;
    static int legal_actions(const GameState& state, ActionMask& out) noexcept;
    static bool step(GameState& state, Action action) noexcept;
};
