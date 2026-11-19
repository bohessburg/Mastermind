#pragma once

#include "v2/core/defs.h"
#include "v2/core/state.h"

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

constexpr std::uint16_t ACTION_MASK_WORDS = static_cast<std::uint16_t>((ACTION_SPACE_SIZE + 63U) / 64U);

struct ActionMask {
    std::uint64_t words[ACTION_MASK_WORDS]{};

    void reset() noexcept {
        for (std::uint16_t i = 0; i < ACTION_MASK_WORDS; ++i) {
            words[i] = 0;
        }
    }

    void set(Action action) noexcept {
        words[action >> 6U] |= (std::uint64_t{1} << (action & 63U));
    }

    [[nodiscard]] bool test(Action action) const noexcept {
        return (words[action >> 6U] & (std::uint64_t{1} << (action & 63U))) != 0U;
    }

    [[nodiscard]] Action nth_set(std::uint32_t index) const noexcept {
        for (std::uint16_t word_index = 0; word_index < ACTION_MASK_WORDS; ++word_index) {
            std::uint64_t word = words[word_index];
            while (word != 0U) {
#if defined(__GNUC__) || defined(__clang__)
                const std::uint32_t bit = static_cast<std::uint32_t>(__builtin_ctzll(word));
#else
                std::uint32_t bit = 0;
                while (((word >> bit) & 1U) == 0U) {
                    ++bit;
                }
#endif
                if (index == 0U) {
                    return static_cast<Action>((word_index * 64U) + bit);
                }
                word &= (word - 1U);
                --index;
            }
        }
        return A_PASS;
    }
};

[[nodiscard]] constexpr Action play_action(DefId def) noexcept {
    return static_cast<Action>(A_PLAY_BASE + def);
}

[[nodiscard]] constexpr Action buy_action(DefId def) noexcept {
    return static_cast<Action>(A_BUY_BASE + def);
}

[[nodiscard]] constexpr Action select_action(DefId def) noexcept {
    return static_cast<Action>(A_SELECT_BASE + def);
}

[[nodiscard]] constexpr Action option_action(std::uint8_t option) noexcept {
    return static_cast<Action>(A_OPTION_BASE + option);
}

[[nodiscard]] bool action_is_pass(Action action) noexcept;
[[nodiscard]] bool action_is_play(Action action) noexcept;
[[nodiscard]] bool action_is_buy(Action action) noexcept;
[[nodiscard]] bool action_is_select(Action action) noexcept;
[[nodiscard]] bool action_is_option(Action action) noexcept;
[[nodiscard]] DefId action_def(Action action, Action base) noexcept;

[[nodiscard]] Cost effective_cost(const GameState& state, DefId def) noexcept;
[[nodiscard]] int legal_actions(const GameState& state, ActionMask& out) noexcept;
void apply_action(GameState& state, Action action) noexcept;
void refresh_current_decision(GameState& state) noexcept;
