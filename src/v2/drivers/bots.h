#pragma once

#include "v2/bots/scripted.h"
#include "v2/core/setup.h"
#include "v2/mcts/tree.h"

#include <cstdint>

struct MctsBot {
    MctsConfig config{};
    Mcts search;
    std::uint64_t searches = 0;
    std::uint64_t sims = 0;

    explicit MctsBot(const MctsConfig& cfg);
    [[nodiscard]] Action choose_action(
        const GameState& state,
        const ActionMask& legal,
        int legal_count) noexcept;
};

enum class BotKind : std::uint8_t {
    Random,
    BigMoney,
    Heuristic,
    Engine,
    EngineV3,
    Mcts,
};

struct BotSpec {
    BotKind kind = BotKind::BigMoney;
    std::uint64_t seed = 0U;
    MctsConfig mcts_config{};
};

struct GameResult {
    PlayerId winner = NONE;
    std::int16_t scores[MAX_PLAYERS]{};
    std::uint16_t turns = 0;
    bool truncated = false;
};

struct MatchupResult {
    std::uint16_t games = 0;
    std::uint16_t wins_a = 0;
    std::uint16_t wins_b = 0;
    std::uint16_t ties = 0;
    std::uint16_t truncated = 0;

    [[nodiscard]] double win_rate_a() const noexcept;
    [[nodiscard]] double win_rate_b() const noexcept;
};

[[nodiscard]] GameResult run_game(
    const Setup& setup,
    std::uint64_t seed,
    BotSpec bot0,
    BotSpec bot1) noexcept;

[[nodiscard]] MatchupResult eval_matchup(
    const Setup& setup,
    BotSpec bot_a,
    BotSpec bot_b,
    std::uint16_t n_games,
    std::uint64_t seed) noexcept;

[[nodiscard]] MatchupResult eval_matchup(
    BotSpec bot_a,
    BotSpec bot_b,
    std::uint16_t n_games,
    std::uint64_t seed) noexcept;
