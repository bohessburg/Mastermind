#include "v2/mcts/eval.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

namespace {

[[nodiscard]] bool parse_u32(const char* text, std::uint32_t& out) noexcept {
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    out = static_cast<std::uint32_t>(value);
    return true;
}

[[nodiscard]] bool parse_u64(const char* text, std::uint64_t& out) noexcept {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    out = static_cast<std::uint64_t>(value);
    return true;
}

[[nodiscard]] bool parse_opponent(const char* text, MctsEvalOpponent& out) noexcept {
    if (std::strcmp(text, "engine") == 0) {
        out = MctsEvalOpponent::Engine;
        return true;
    }
    if (std::strcmp(text, "bigmoney") == 0) {
        out = MctsEvalOpponent::BigMoney;
        return true;
    }
    if (std::strcmp(text, "heuristic") == 0) {
        out = MctsEvalOpponent::Heuristic;
        return true;
    }
    if (std::strcmp(text, "random") == 0) {
        out = MctsEvalOpponent::Random;
        return true;
    }
    return false;
}

[[nodiscard]] bool parse_kingdoms(const char* text, MctsEvalKingdoms& out) noexcept {
    if (std::strcmp(text, "random") == 0) {
        out = MctsEvalKingdoms::Random;
        return true;
    }
    if (std::strcmp(text, "fixed") == 0) {
        out = MctsEvalKingdoms::Fixed;
        return true;
    }
    return false;
}

[[nodiscard]] const char* opponent_name(MctsEvalOpponent opponent) noexcept {
    switch (opponent) {
    case MctsEvalOpponent::BigMoney:
        return "bigmoney";
    case MctsEvalOpponent::Heuristic:
        return "heuristic";
    case MctsEvalOpponent::Random:
        return "random";
    case MctsEvalOpponent::Engine:
    default:
        return "engine";
    }
}

[[nodiscard]] const char* kingdom_name(MctsEvalKingdoms kingdoms) noexcept {
    return kingdoms == MctsEvalKingdoms::Fixed ? "fixed" : "random";
}

void usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " [--sims N] [--games N] [--opponent engine|bigmoney|heuristic|random]"
              << " [--kingdoms random|fixed] [--seed N] [--determinizations K] [--threads T]\n";
}

[[nodiscard]] bool parse_args(int argc, char** argv, MctsEvalOptions& options) {
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) {
            usage(argv[0]);
            return false;
        }
        const char* arg = argv[i];
        const char* value = argv[++i];
        if (std::strcmp(arg, "--sims") == 0) {
            if (!parse_u32(value, options.sims)) {
                return false;
            }
        } else if (std::strcmp(arg, "--games") == 0) {
            if (!parse_u32(value, options.games)) {
                return false;
            }
        } else if (std::strcmp(arg, "--opponent") == 0) {
            if (!parse_opponent(value, options.opponent)) {
                return false;
            }
        } else if (std::strcmp(arg, "--kingdoms") == 0) {
            if (!parse_kingdoms(value, options.kingdoms)) {
                return false;
            }
        } else if (std::strcmp(arg, "--seed") == 0) {
            if (!parse_u64(value, options.seed)) {
                return false;
            }
        } else if (std::strcmp(arg, "--determinizations") == 0) {
            std::uint32_t parsed = 0;
            if (!parse_u32(value, parsed)) {
                return false;
            }
            options.determinizations = parsed > 255U ? 255U : static_cast<std::uint8_t>(parsed);
        } else if (std::strcmp(arg, "--threads") == 0) {
            if (!parse_u32(value, options.threads)) {
                return false;
            }
        } else {
            usage(argv[0]);
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    MctsEvalOptions options{};
    if (options.threads == 0U) {
        const std::uint32_t hardware = std::thread::hardware_concurrency();
        options.threads = hardware == 0U ? 1U : hardware;
    }
    if (!parse_args(argc, argv, options)) {
        return 2;
    }

    const MctsEvalResult result = run_mcts_eval(options);
    std::cout << "mcts_eval"
              << " opponent=" << opponent_name(options.opponent)
              << " kingdoms=" << kingdom_name(options.kingdoms)
              << " games=" << result.games
              << " wins=" << result.mcts_wins
              << " losses=" << result.opponent_wins
              << " ties=" << result.ties
              << " truncated=" << result.truncated
              << " win_pct=" << result.mcts_win_percent()
              << " loss_pct=" << result.opponent_win_percent()
              << " tie_pct=" << result.tie_percent()
              << " mcts_sims_per_sec=" << result.mcts_sims_per_sec()
              << " avg_move_ms=" << result.avg_move_ms()
              << " wall_seconds=" << result.wall_seconds
              << "\n";
    return 0;
}
