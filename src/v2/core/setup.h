#pragma once

#include "v2/core/state.h"
#include "v2/core/types.h"

#include <cstdint>

// Basic piles a setup can add besides the kingdom: Copper, Silver, Gold,
// Estate, Duchy, Province, Curse, plus Platinum, Colony and Potion.
constexpr int MAX_BASIC_PILES = 10;
constexpr int MAX_KINGDOM_DEFS = MAX_PILES - MAX_BASIC_PILES;
static_assert(MAX_KINGDOM_DEFS >= 11, "room for 10 kingdom piles plus a bane");

struct Setup {
    PlayerId num_players = 2;
    std::uint8_t kingdom_count = 0;
    DefId kingdom[MAX_KINGDOM_DEFS]{};
    bool use_colony_platinum = false;
};

[[nodiscard]] GameState new_game(const Setup& setup, std::uint64_t seed) noexcept;
[[nodiscard]] bool has_slot(const GameState& state, DefId def) noexcept;
[[nodiscard]] Slot slot_of(const GameState& state, DefId def) noexcept;
