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

// Replay hook for a driver that mirrors a game played elsewhere (the arena
// shadow).  While a rig is installed on the calling thread, each reshuffle of
// rig.player consumes the next prescription and brings its cards to the top
// of the new deck in draw order (first card drawn first).  The shuffle itself
// still runs first, so the state RNG advances exactly as it does unrigged.
// A prescription the reshuffled deck cannot supply sets `failed` and leaves
// the deck as shuffled.
struct ShuffleRig {
    static constexpr int MAX_SHUFFLES = 4;

    PlayerId player = 0;
    std::uint8_t count = 0;  // prescriptions installed
    std::uint8_t next = 0;   // prescriptions consumed
    std::uint8_t failed = 0;
    std::uint8_t sizes[MAX_SHUFFLES]{};
    DefId cards[MAX_SHUFFLES][MAX_DECK_CARDS]{};
};

// Thread-local; nullptr removes it.  The rig must outlive its installation.
void install_shuffle_rig(ShuffleRig* rig) noexcept;

void draw_cards(GameState& state, PlayerId player, std::uint8_t count) noexcept;
[[nodiscard]] Slot take_deck_top_slot(GameState& state, PlayerId player) noexcept;
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
