#pragma once

#include "v2/core/actions.h"
#include "v2/core/game.h"
#include "v2/core/rng.h"
#include "v2/core/setup.h"
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

enum class BotKind : std::uint8_t {
    Random,
    BigMoney,
};

struct BotSpec {
    BotKind kind = BotKind::BigMoney;
    std::uint64_t seed = 0U;
};

struct GameResult {
    PlayerId winner = NONE;
    std::int16_t scores[MAX_PLAYERS]{};
    std::uint16_t turns = 0;
    bool truncated = false;
};

[[nodiscard]] GameResult run_game(
    const Setup& setup,
    std::uint64_t seed,
    BotSpec bot0,
    BotSpec bot1) noexcept;
