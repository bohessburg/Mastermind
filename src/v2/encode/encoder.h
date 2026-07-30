#pragma once

#include "v2/core/state.h"
#include "v2/core/types.h"

#include <cstddef>
#include <cstdint>

enum class ObsVersion : std::uint16_t {
    V1 = 1,
    V2 = 2,
    V3 = 3,
};

// Encoder compatibility history: generation 1 predates the landscape/trait
// sentinel fix; generation 2 encodes absent landscapes and pile traits as 0.
inline constexpr std::uint16_t ENCODER_GENERATION = 2;

// Compatibility aliases for callers and checkpoints built before observation
// versioning. They deliberately continue to describe the v1/default layout.
inline constexpr std::uint16_t OBS_VERSION = static_cast<std::uint16_t>(ObsVersion::V1);

inline constexpr std::size_t OBS_META_SIZE = 4;
inline constexpr std::size_t OBS_OWN_ZONE_COUNT = 5;
inline constexpr std::size_t OBS_OWN_SIZE = OBS_OWN_ZONE_COUNT * MAX_SLOTS;
inline constexpr std::size_t OBS_OPPONENT_BLOCK_SIZE_V1 = 75;
inline constexpr std::size_t OBS_OPPONENT_SIZE_V1 = (MAX_PLAYERS - 1) * OBS_OPPONENT_BLOCK_SIZE_V1;
inline constexpr std::size_t OBS_PILE_BLOCK_SIZE = 11;
inline constexpr std::size_t OBS_PILE_COUNT_FIELD = 0;
inline constexpr std::size_t OBS_PILE_BASE_FIELD = 2;
inline constexpr std::size_t OBS_PILE_TRAIT_FIELD = 4;
inline constexpr std::size_t OBS_SUPPLY_SIZE = MAX_PILES * OBS_PILE_BLOCK_SIZE;
inline constexpr std::size_t OBS_LANDSCAPE_ID_SECTION_COUNT = 4;
inline constexpr std::size_t OBS_LANDSCAPE_ID_SIZE = OBS_LANDSCAPE_ID_SECTION_COUNT * MAX_LANDSCAPES;
inline constexpr std::size_t OBS_LANDSCAPE_PROJECT_BOUGHT_OFFSET = OBS_LANDSCAPE_ID_SIZE;
inline constexpr std::size_t OBS_LANDSCAPE_PROPHECY_OFFSET = OBS_LANDSCAPE_PROJECT_BOUGHT_OFFSET + MAX_LANDSCAPES;
inline constexpr std::size_t OBS_LANDSCAPE_SIZE = 27;
inline constexpr std::size_t OBS_RESOURCE_SIZE = 12;
inline constexpr std::size_t OBS_PHASE_COUNT = 5;
inline constexpr std::size_t OBS_TURN_SIZE = OBS_PHASE_COUNT + 5;
inline constexpr std::size_t OBS_DECISION_KIND_COUNT = 10;
inline constexpr std::size_t OBS_DECISION_SIZE = OBS_DECISION_KIND_COUNT + 5;

inline constexpr std::size_t OBS_META_OFFSET = 0;
inline constexpr std::size_t OBS_OWN_OFFSET = OBS_META_OFFSET + OBS_META_SIZE;
inline constexpr std::size_t OBS_OPPONENT_OFFSET = OBS_OWN_OFFSET + OBS_OWN_SIZE;
inline constexpr std::size_t OBS_SUPPLY_OFFSET = OBS_OPPONENT_OFFSET + OBS_OPPONENT_SIZE_V1;
inline constexpr std::size_t OBS_LANDSCAPE_OFFSET = OBS_SUPPLY_OFFSET + OBS_SUPPLY_SIZE;
inline constexpr std::size_t OBS_RESOURCE_OFFSET = OBS_LANDSCAPE_OFFSET + OBS_LANDSCAPE_SIZE;
inline constexpr std::size_t OBS_TURN_OFFSET = OBS_RESOURCE_OFFSET + OBS_RESOURCE_SIZE;
inline constexpr std::size_t OBS_DECISION_OFFSET = OBS_TURN_OFFSET + OBS_TURN_SIZE;
inline constexpr std::size_t OBS_SIZE_V1 = OBS_DECISION_OFFSET + OBS_DECISION_SIZE;

// v2 preserves the v1 prefix inside every opponent block, then appends the
// three perfect-memory public compositions indexed by Slot.
inline constexpr std::size_t OBS_V2_OPPONENT_COLLECTION_OFFSET = OBS_OPPONENT_BLOCK_SIZE_V1;
inline constexpr std::size_t OBS_V2_OPPONENT_DISCARD_OFFSET = OBS_V2_OPPONENT_COLLECTION_OFFSET + MAX_SLOTS;
inline constexpr std::size_t OBS_V2_OPPONENT_SET_ASIDE_OFFSET = OBS_V2_OPPONENT_DISCARD_OFFSET + MAX_SLOTS;
inline constexpr std::size_t OBS_OPPONENT_BLOCK_SIZE_V2 = OBS_V2_OPPONENT_SET_ASIDE_OFFSET + MAX_SLOTS;
inline constexpr std::size_t OBS_OPPONENT_SIZE_V2 = (MAX_PLAYERS - 1) * OBS_OPPONENT_BLOCK_SIZE_V2;

inline constexpr std::size_t OBS_V2_META_OFFSET = 0;
inline constexpr std::size_t OBS_V2_OWN_OFFSET = OBS_V2_META_OFFSET + OBS_META_SIZE;
inline constexpr std::size_t OBS_V2_OPPONENT_OFFSET = OBS_V2_OWN_OFFSET + OBS_OWN_SIZE;
inline constexpr std::size_t OBS_V2_SUPPLY_OFFSET = OBS_V2_OPPONENT_OFFSET + OBS_OPPONENT_SIZE_V2;
inline constexpr std::size_t OBS_V2_LANDSCAPE_OFFSET = OBS_V2_SUPPLY_OFFSET + OBS_SUPPLY_SIZE;
inline constexpr std::size_t OBS_V2_RESOURCE_OFFSET = OBS_V2_LANDSCAPE_OFFSET + OBS_LANDSCAPE_SIZE;
inline constexpr std::size_t OBS_V2_TURN_OFFSET = OBS_V2_RESOURCE_OFFSET + OBS_RESOURCE_SIZE;
inline constexpr std::size_t OBS_V2_DECISION_OFFSET = OBS_V2_TURN_OFFSET + OBS_TURN_SIZE;
inline constexpr std::size_t OBS_SIZE_V2 = OBS_V2_DECISION_OFFSET + OBS_DECISION_SIZE;

// v3 retains the entire v2 layout, then appends the global trash-pile
// composition indexed by Slot and decision selection semantics.
inline constexpr std::size_t OBS_V3_TRASH_OFFSET = OBS_SIZE_V2;
inline constexpr std::size_t OBS_TRASH_SIZE = MAX_SLOTS;
inline constexpr std::size_t OBS_SELECT_SEMANTIC_COUNT = 7;
inline constexpr std::size_t OBS_V3_SELECT_SEMANTIC_OFFSET = OBS_V3_TRASH_OFFSET + OBS_TRASH_SIZE;
inline constexpr std::size_t OBS_SIZE_V3 = OBS_V3_SELECT_SEMANTIC_OFFSET + OBS_SELECT_SEMANTIC_COUNT;

// Legacy source aliases. Keep these fixed to v1 so existing integrations that
// allocate OBS_SIZE retain their byte-identical default behavior.
inline constexpr std::size_t OBS_OPPONENT_BLOCK_SIZE = OBS_OPPONENT_BLOCK_SIZE_V1;
inline constexpr std::size_t OBS_OPPONENT_SIZE = OBS_OPPONENT_SIZE_V1;
inline constexpr std::size_t OBS_SIZE = OBS_SIZE_V1;

[[nodiscard]] constexpr bool is_valid_obs_version(ObsVersion version) noexcept {
    return version == ObsVersion::V1 || version == ObsVersion::V2 || version == ObsVersion::V3;
}

[[nodiscard]] constexpr std::size_t obs_size_for(ObsVersion version) noexcept {
    return version == ObsVersion::V3
        ? OBS_SIZE_V3
        : (version == ObsVersion::V2 ? OBS_SIZE_V2 : OBS_SIZE_V1);
}

// The v1 implementation intentionally retains the original encoder's exact
// logic and is kept indefinitely for existing checkpoints.
void encode_v1(const GameState& state, PlayerId perspective, float* out) noexcept;
void encode_v2(const GameState& state, PlayerId perspective, float* out) noexcept;
void encode_v3(const GameState& state, PlayerId perspective, float* out) noexcept;
void encode(
    const GameState& state,
    PlayerId perspective,
    float* out,
    ObsVersion version = ObsVersion::V1) noexcept;
