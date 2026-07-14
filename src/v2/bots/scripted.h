#pragma once

#include "v2/core/actions.h"
#include "v2/core/game.h"
#include "v2/core/rng.h"
#include "v2/core/types.h"

#include <cstdint>

struct RandomBot {
    Xoshiro256pp rng{};

    explicit RandomBot(std::uint64_t seed = 0U) noexcept;
    [[nodiscard]] Action choose_action(
        const GameState& state,
        const ActionMask& legal,
        int legal_count) noexcept;
};

struct BigMoneyBot {
    [[nodiscard]] Action choose_action(
        const GameState& state,
        const ActionMask& legal,
        int legal_count) const noexcept;
};

struct HeuristicBot {
    [[nodiscard]] Action choose_action(
        const GameState& state,
        const ActionMask& legal,
        int legal_count) const noexcept;
};

struct EngineBot {
    std::uint8_t chapel_plays[MAX_PLAYERS]{};

    [[nodiscard]] Action choose_action(
        const GameState& state,
        const ActionMask& legal,
        int legal_count) noexcept;
};

struct EngineBotV3 {
    std::uint8_t chapel_plays[MAX_PLAYERS]{};
    bool greening_started[MAX_PLAYERS]{};
    std::uint32_t green_buy_turn[MAX_PLAYERS]{};
    bool green_vp_bought[MAX_PLAYERS]{};

    [[nodiscard]] Action choose_action(
        const GameState& state,
        const ActionMask& legal,
        int legal_count) noexcept;
};
