#pragma once

#include "v2/core/state.h"
#include "v2/core/types.h"

constexpr std::uint16_t MAX_TURNS = 200;

void turn_queue_clear(TurnQueue& queue) noexcept;
bool turn_queue_push(TurnQueue& queue, PlayerId player, TurnKind kind) noexcept;
bool turn_queue_pop(TurnQueue& queue, TurnQueueEntry& out) noexcept;
[[nodiscard]] TurnQueueEntry turn_queue_front(const TurnQueue& queue) noexcept;
[[nodiscard]] PlayerId current_player(const GameState& state) noexcept;

void start_turn(GameState& state, PlayerId player) noexcept;
void cleanup_current_turn(GameState& state) noexcept;
[[nodiscard]] bool is_game_over(const GameState& state) noexcept;
bool advance_turn_machinery(GameState& state) noexcept;
