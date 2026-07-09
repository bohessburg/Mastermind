#pragma once

#include "v2/core/defs.h"
#include "v2/core/state.h"

#include <bitset>
#include <cstdint>

using Action = std::uint16_t;

inline constexpr std::uint16_t ACTION_DEF_COUNT = BASIC_CARD_COUNT;
inline constexpr Action A_PASS = 0;
inline constexpr Action A_PLAY_BASE = 1;
inline constexpr Action A_WAY_BASE = static_cast<Action>(A_PLAY_BASE + ACTION_DEF_COUNT);
inline constexpr Action A_BUY_BASE = static_cast<Action>(A_WAY_BASE + (MAX_LANDSCAPES * ACTION_DEF_COUNT));
inline constexpr Action A_EVENT_BASE = static_cast<Action>(A_BUY_BASE + ACTION_DEF_COUNT);
inline constexpr Action A_SELECT_BASE = static_cast<Action>(A_EVENT_BASE + MAX_LANDSCAPES);
inline constexpr Action A_OPTION_BASE = static_cast<Action>(A_SELECT_BASE + ACTION_DEF_COUNT);
inline constexpr Action A_CALL_BASE = static_cast<Action>(A_OPTION_BASE + 16);
inline constexpr Action A_SPEND_BASE = static_cast<Action>(A_CALL_BASE + ACTION_DEF_COUNT);
inline constexpr Action A_END = static_cast<Action>(A_SPEND_BASE + 8);
inline constexpr std::uint16_t ACTION_SPACE_SIZE = A_END;

using ActionMask = std::bitset<ACTION_SPACE_SIZE>;

[[nodiscard]] constexpr Action play_action(DefId def) noexcept {
    return static_cast<Action>(A_PLAY_BASE + def);
}

[[nodiscard]] constexpr Action buy_action(DefId def) noexcept {
    return static_cast<Action>(A_BUY_BASE + def);
}

[[nodiscard]] bool action_is_pass(Action action) noexcept;
[[nodiscard]] bool action_is_play(Action action) noexcept;
[[nodiscard]] bool action_is_buy(Action action) noexcept;
[[nodiscard]] DefId action_def(Action action, Action base) noexcept;

[[nodiscard]] Cost effective_cost(const GameState& state, DefId def) noexcept;
[[nodiscard]] int legal_actions(const GameState& state, ActionMask& out) noexcept;
void apply_action(GameState& state, Action action) noexcept;
void refresh_current_decision(GameState& state) noexcept;
