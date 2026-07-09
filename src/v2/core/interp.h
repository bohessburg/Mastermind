#pragma once

#include "v2/core/actions.h"
#include "v2/core/defs.h"
#include "v2/core/state.h"
#include "v2/core/types.h"

#include <cstdint>

inline constexpr std::uint8_t FRAME_ABSOLUTE_PROGRAM = static_cast<std::uint8_t>(1U << 0);
inline constexpr std::uint8_t FRAME_ATTACK = static_cast<std::uint8_t>(1U << 1);
inline constexpr std::uint8_t FRAME_REACT_DONE = static_cast<std::uint8_t>(1U << 2);
inline constexpr std::uint8_t FRAME_ATTACK_IMMUNE = static_cast<std::uint8_t>(1U << 3);
inline constexpr std::uint8_t FRAME_TRIGGER_ORDER = static_cast<std::uint8_t>(1U << 4);

void draw_cards(GameState& state, PlayerId player, std::uint8_t count) noexcept;
bool push_effect(GameState& state, DefId source, PlayerId player) noexcept;
bool push_effect_span(
    GameState& state,
    DefId source,
    PlayerId player,
    EffectSpan span,
    std::uint8_t flags) noexcept;
bool push_trigger_order_frame(GameState& state, PlayerId player, DefId source, std::uint8_t count) noexcept;
[[nodiscard]] const Instr& current_effect_instr(const GameState& state) noexcept;
RunResult interp_run(GameState& state) noexcept;
void interp_resume(GameState& state, Action action) noexcept;
