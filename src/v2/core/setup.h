#pragma once

#include "v2/core/state.h"
#include "v2/core/types.h"

#include <cstdint>

constexpr int MAX_KINGDOM_DEFS = 40;

struct Setup {
    PlayerId num_players = 2;
    std::uint8_t kingdom_count = 0;
    DefId kingdom[MAX_KINGDOM_DEFS]{};
    bool use_colony_platinum = false;
};

[[nodiscard]] GameState new_game(const Setup& setup, std::uint64_t seed) noexcept;
[[nodiscard]] bool has_slot(const GameState& state, DefId def) noexcept;
[[nodiscard]] Slot slot_of(const GameState& state, DefId def) noexcept;
