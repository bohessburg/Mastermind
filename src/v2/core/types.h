#pragma once

#include <cstdint>

using DefId = std::uint16_t;
using Slot = std::uint8_t;
using PlayerId = std::uint8_t;

constexpr int MAX_PLAYERS = 4;
constexpr int MAX_SLOTS = 64;
constexpr int MAX_PILES = 48;
constexpr int MAX_NONSUPPLY = 8;
constexpr int MAX_DECK_CARDS = 160;
constexpr int MAX_EFFECT_DEPTH = 24;
constexpr int MAX_PENDING = 24;
constexpr int MAX_LANDSCAPES = 4;

constexpr std::uint8_t NO_LANDSCAPE = 0xFFU;
constexpr std::uint8_t NONE = 0xFFU;
constexpr int NUM_ARTIFACTS = 5; // Renaissance artifacts: Flag/Horn/Key/Lantern/Chest.

struct Cost {
    std::int8_t coins = 0;
    std::int8_t potion = 0;
    std::int16_t debt = 0;

    [[nodiscard]] constexpr bool fits_within(const Cost& budget) const noexcept {
        return coins <= budget.coins && potion <= budget.potion && debt <= budget.debt;
    }

    [[nodiscard]] constexpr bool operator==(const Cost&) const noexcept = default;
};
