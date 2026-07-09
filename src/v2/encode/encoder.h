#pragma once

#include "v2/core/state.h"
#include "v2/core/types.h"

#include <cstddef>
#include <cstdint>

inline constexpr std::uint16_t OBS_VERSION = 1;

inline constexpr std::size_t OBS_META_SIZE = 4;
inline constexpr std::size_t OBS_OWN_ZONE_COUNT = 5;
inline constexpr std::size_t OBS_OWN_SIZE = OBS_OWN_ZONE_COUNT * MAX_SLOTS;
inline constexpr std::size_t OBS_OPPONENT_BLOCK_SIZE = 75;
inline constexpr std::size_t OBS_OPPONENT_SIZE = (MAX_PLAYERS - 1) * OBS_OPPONENT_BLOCK_SIZE;
inline constexpr std::size_t OBS_PILE_BLOCK_SIZE = 11;
inline constexpr std::size_t OBS_SUPPLY_SIZE = MAX_PILES * OBS_PILE_BLOCK_SIZE;
inline constexpr std::size_t OBS_LANDSCAPE_SIZE = 27;
inline constexpr std::size_t OBS_RESOURCE_SIZE = 12;
inline constexpr std::size_t OBS_PHASE_COUNT = 5;
inline constexpr std::size_t OBS_TURN_SIZE = OBS_PHASE_COUNT + 5;
inline constexpr std::size_t OBS_DECISION_KIND_COUNT = 10;
inline constexpr std::size_t OBS_DECISION_SIZE = OBS_DECISION_KIND_COUNT + 5;

inline constexpr std::size_t OBS_META_OFFSET = 0;
inline constexpr std::size_t OBS_OWN_OFFSET = OBS_META_OFFSET + OBS_META_SIZE;
inline constexpr std::size_t OBS_OPPONENT_OFFSET = OBS_OWN_OFFSET + OBS_OWN_SIZE;
inline constexpr std::size_t OBS_SUPPLY_OFFSET = OBS_OPPONENT_OFFSET + OBS_OPPONENT_SIZE;
inline constexpr std::size_t OBS_LANDSCAPE_OFFSET = OBS_SUPPLY_OFFSET + OBS_SUPPLY_SIZE;
inline constexpr std::size_t OBS_RESOURCE_OFFSET = OBS_LANDSCAPE_OFFSET + OBS_LANDSCAPE_SIZE;
inline constexpr std::size_t OBS_TURN_OFFSET = OBS_RESOURCE_OFFSET + OBS_RESOURCE_SIZE;
inline constexpr std::size_t OBS_DECISION_OFFSET = OBS_TURN_OFFSET + OBS_TURN_SIZE;
inline constexpr std::size_t OBS_SIZE = OBS_DECISION_OFFSET + OBS_DECISION_SIZE;

void encode(const GameState& state, PlayerId perspective, float* out) noexcept;
