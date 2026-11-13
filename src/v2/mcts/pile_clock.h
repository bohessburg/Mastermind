#pragma once

#include "v2/core/actions.h"
#include "v2/core/state.h"

// Stack-only supply analysis used to guard scripted buying once two piles are
// empty. Callers can reuse the analysis to suppress rollout exploration.
struct PileClock {
    int empty_piles = 0;
    Action ending_buy = A_PASS;
    Action victory_buy = A_PASS;
    Action non_ending_victory_buy = A_PASS;
    DefId victory_def = NONE;
    DefId non_ending_victory_def = NONE;
    ActionMask ending_buys{};
};

using PileClockBaseBuyFn = Action (*)(const GameState&, const ActionMask&) noexcept;

[[nodiscard]] PileClock analyze_pile_clock(
    const GameState& state,
    const ActionMask& legal) noexcept;
[[nodiscard]] Action pile_clock_guarded_buy(
    const GameState& state,
    const ActionMask& legal,
    const PileClock& clock,
    PileClockBaseBuyFn base_buy) noexcept;
