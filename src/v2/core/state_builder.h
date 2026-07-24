#pragma once

#include "v2/core/state.h"
#include "v2/core/types.h"

#include <cstdint>
#include <span>
#include <type_traits>

struct SnapshotCardCounts {
    std::uint16_t by_def[MAX_SLOTS]{};
};

struct SnapshotPlayer {
    SnapshotCardCounts hand{};
    std::uint16_t hand_count = 0;
    SnapshotCardCounts hand_deck{};
    std::uint16_t deck_count = 0;
    SnapshotCardCounts discard{};
    SnapshotCardCounts in_play{};
    SnapshotCardCounts set_aside{};
    std::uint16_t actions = 0;
    std::uint16_t buys = 0;
    std::int16_t coins = 0;
};

enum class SnapshotPhase : std::uint8_t {
    Action,
    Buy,
    Cleanup,
};

enum class SeededInterrupt : std::uint8_t {
    None,
    MoatReaction,
    MilitiaDiscard,
    BureaucratTopdeck,
    BanditTrash,
};

struct Snapshot {
    PlayerId num_players = 0;
    PlayerId our_player = 0;
    std::uint8_t supply_present[MAX_SLOTS]{};
    SnapshotCardCounts supply{};
    SnapshotPlayer players[MAX_PLAYERS]{};
    SnapshotCardCounts trash{};
    SnapshotCardCounts card_totals{};
    std::uint16_t turn_number = 0;
    SnapshotPhase phase = SnapshotPhase::Action;
    PlayerId current_player = 0;
    SeededInterrupt interrupt = SeededInterrupt::None;
    PlayerId attacker = 0;
    PlayerId defender = 0;
};

static_assert(std::is_trivially_copyable_v<Snapshot>);

[[nodiscard]] GameState build_game_from_snapshot(const Snapshot& snapshot);
void set_deck_order(GameState& state, PlayerId player, std::span<const DefId> defs);
void validate_game(const GameState& state);
