#pragma once

#include "v2/core/defs.h"
#include "v2/core/state.h"
#include "v2/core/types.h"

#include <cstdint>

enum class TriggerKind : std::uint8_t {
    OnGain,
    WouldGain,
    OnTrash,
    OnDiscard,
    OnShuffle,
    OnPlayAction,
    OnPlayTreasure,
    StartOfTurn,
    EndOfTurn,
    StartOfBuy,
    EndOfBuy,
    OnFirstPlay,
    OnCall,
    OnExile,
};

struct TriggerPayload {
    PlayerId player = 0;
    DefId def = 0;
    Slot slot = NONE;
    std::uint8_t destination = 0;
};

[[nodiscard]] constexpr std::uint32_t trigger_mask(TriggerKind kind) noexcept {
    return static_cast<std::uint32_t>(1ULL << static_cast<std::uint8_t>(kind));
}

void mark_trigger_table_dirty(GameState& state) noexcept;
void emit(GameState& state, TriggerKind kind, TriggerPayload payload) noexcept;
