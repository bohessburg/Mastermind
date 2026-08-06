#pragma once

#include "v2/core/setup.h"
#include "v2/drivers/bots.h"

#include <cstdint>

enum class MctsEvalOpponent : std::uint8_t {
    Engine,
    BigMoney,
    Heuristic,
    Random,
};

enum class MctsEvalKingdoms : std::uint8_t {
    Random,
    Fixed,
};

struct MctsEvalOptions {
    std::uint32_t sims = 1000;
    std::uint32_t games = 1000;
    MctsEvalOpponent opponent = MctsEvalOpponent::Engine;
    MctsEvalKingdoms kingdoms = MctsEvalKingdoms::Random;
    std::uint64_t seed = 0x6D435453ULL;
    std::uint8_t determinizations = 2;
    std::uint32_t threads = 0;
    MctsRolloutPolicy rollout_policy = MctsRolloutPolicy::EngineLike;
};

struct MctsEvalResult {
    std::uint32_t games = 0;
    std::uint32_t mcts_wins = 0;
    std::uint32_t opponent_wins = 0;
    std::uint32_t ties = 0;
    std::uint32_t truncated = 0;
    std::uint64_t mcts_searches = 0;
    std::uint64_t mcts_sims = 0;
    std::uint64_t mcts_search_ns = 0;
    double wall_seconds = 0.0;

    [[nodiscard]] double mcts_win_percent() const noexcept;
    [[nodiscard]] double opponent_win_percent() const noexcept;
    [[nodiscard]] double tie_percent() const noexcept;
    [[nodiscard]] double mcts_sims_per_sec() const noexcept;
    [[nodiscard]] double avg_move_ms() const noexcept;
};

[[nodiscard]] Setup fixed_mcts_eval_setup() noexcept;
[[nodiscard]] Setup random_mcts_eval_setup(std::uint64_t seed) noexcept;
[[nodiscard]] BotKind bot_kind_for(MctsEvalOpponent opponent) noexcept;
[[nodiscard]] MctsEvalResult run_mcts_eval(const MctsEvalOptions& options);
