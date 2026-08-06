#include "v2/core/game.h"
#include "v2/drivers/bots.h"
#include "v2/mcts/tree.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int MAX_STREAM_ACTIONS = 4096;
constexpr int MAX_STEP_SAMPLES = 65536;
constexpr int STEP_REPEATS = 128;
constexpr int CLONE_ITERS = 100000;
constexpr int LEGAL_ITERS = 100000;
constexpr int RANDOM_GAMES = 1000;
constexpr int BM_GAMES = 1000;
constexpr int HEURISTIC_GAMES = 1000;
constexpr int ENGINE_GAMES = 1000;
constexpr int MCTS_SEARCHES = 20;
constexpr std::uint64_t STREAM_SEED = 0xBEE5'0001ULL;

struct ActionStream {
    Action actions[MAX_STREAM_ACTIONS]{};
    int count = 0;
};

struct Percentiles {
    std::uint64_t median = 0;
    std::uint64_t p99 = 0;
};

#if defined(_MSC_VER)
#define V2_NOINLINE __declspec(noinline)
#else
#define V2_NOINLINE __attribute__((noinline))
#endif

V2_NOINLINE void clone_state(GameState& dst, const GameState& src) {
    std::memcpy(&dst, &src, sizeof(GameState));
}

[[nodiscard]] std::uint64_t elapsed_ns(Clock::time_point start, Clock::time_point end) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

[[nodiscard]] Action choose_big_money(GameState& state, BigMoneyBot& bot) {
    ActionMask legal{};
    const int legal_count = Game::legal_actions(state, legal);
    return bot.choose_action(state, legal, legal_count);
}

[[nodiscard]] ActionStream record_big_money_stream() {
    ActionStream stream{};
    GameState state = Game::new_game(Setup{}, STREAM_SEED);
    BigMoneyBot bot{};
    bool done = false;
    while (!done && stream.count < MAX_STREAM_ACTIONS) {
        const Action action = choose_big_money(state, bot);
        stream.actions[stream.count] = action;
        ++stream.count;
        done = Game::step(state, action);
    }
    return stream;
}

[[nodiscard]] Percentiles percentiles(std::uint64_t* values, int count) {
    std::sort(values, values + count);
    const int median_index = count / 2;
    int p99_index = static_cast<int>((static_cast<std::int64_t>(count) * 99) / 100);
    if (p99_index >= count) {
        p99_index = count - 1;
    }
    return Percentiles{values[median_index], values[p99_index]};
}

[[nodiscard]] Percentiles measure_step_ns(const ActionStream& stream) {
    static std::uint64_t samples[MAX_STEP_SAMPLES]{};
    int sample_count = 0;

    for (int repeat = 0; repeat < STEP_REPEATS && sample_count < MAX_STEP_SAMPLES; ++repeat) {
        GameState state = Game::new_game(Setup{}, STREAM_SEED);
        for (int i = 0; i < stream.count && sample_count < MAX_STEP_SAMPLES; ++i) {
            const auto start = Clock::now();
            (void)Game::step(state, stream.actions[i]);
            const auto end = Clock::now();
            samples[sample_count] = elapsed_ns(start, end);
            ++sample_count;
        }
    }

    return percentiles(samples, sample_count);
}

[[nodiscard]] std::uint64_t measure_clone_ns() {
    const GameState state = Game::new_game(Setup{}, 0xC10E'0001ULL);
    GameState clones[2]{};
    std::uint64_t checksum = 0;
    const auto start = Clock::now();
    for (int i = 0; i < CLONE_ITERS; ++i) {
        GameState& clone = clones[i & 1];
        clone_state(clone, state);
        checksum += clone.rng.state[static_cast<std::uint8_t>(i & 3)];
    }
    const auto end = Clock::now();
    if (checksum == 0xFFFF'FFFF'FFFF'FFFFULL) {
        std::cout << "";
    }
    return elapsed_ns(start, end) / CLONE_ITERS;
}

[[nodiscard]] std::uint64_t measure_legal_actions_ns() {
    GameState state = Game::new_game(Setup{}, 0x1E6A'1001ULL);
    (void)Game::step(state, A_PASS);
    ActionMask mask{};
    std::uint64_t checksum = 0;

    const auto start = Clock::now();
    for (int i = 0; i < LEGAL_ITERS; ++i) {
        checksum += static_cast<std::uint64_t>(Game::legal_actions(state, mask));
    }
    const auto end = Clock::now();
    if (checksum == 0U) {
        std::cout << "";
    }
    return elapsed_ns(start, end) / LEGAL_ITERS;
}

[[nodiscard]] double measure_random_games_per_sec() {
    const auto start = Clock::now();
    for (std::uint64_t i = 0; i < RANDOM_GAMES; ++i) {
        (void)run_game(
            Setup{},
            0x2A9D'0000ULL + i,
            BotSpec{BotKind::Random, 0x2A9D'1000ULL + i},
            BotSpec{BotKind::Random, 0x2A9D'2000ULL + i});
    }
    const auto end = Clock::now();
    const double seconds = static_cast<double>(elapsed_ns(start, end)) / 1'000'000'000.0;
    return static_cast<double>(RANDOM_GAMES) / seconds;
}

[[nodiscard]] double measure_bm_games_per_sec() {
    const auto start = Clock::now();
    for (std::uint64_t i = 0; i < BM_GAMES; ++i) {
        (void)run_game(
            Setup{},
            0xB160'0000ULL + i,
            BotSpec{BotKind::BigMoney, i},
            BotSpec{BotKind::BigMoney, i + 1U});
    }
    const auto end = Clock::now();
    const double seconds = static_cast<double>(elapsed_ns(start, end)) / 1'000'000'000.0;
    return static_cast<double>(BM_GAMES) / seconds;
}

[[nodiscard]] double measure_heuristic_games_per_sec() {
    const auto start = Clock::now();
    for (std::uint64_t i = 0; i < HEURISTIC_GAMES; ++i) {
        (void)run_game(
            Setup{},
            0x4E00'0000ULL + i,
            BotSpec{BotKind::Heuristic, i},
            BotSpec{BotKind::Heuristic, i + 1U});
    }
    const auto end = Clock::now();
    const double seconds = static_cast<double>(elapsed_ns(start, end)) / 1'000'000'000.0;
    return static_cast<double>(HEURISTIC_GAMES) / seconds;
}

[[nodiscard]] double measure_engine_games_per_sec() {
    const auto start = Clock::now();
    for (std::uint64_t i = 0; i < ENGINE_GAMES; ++i) {
        (void)run_game(
            Setup{},
            0xE600'0000ULL + i,
            BotSpec{BotKind::Engine, i},
            BotSpec{BotKind::Engine, i + 1U});
    }
    const auto end = Clock::now();
    const double seconds = static_cast<double>(elapsed_ns(start, end)) / 1'000'000'000.0;
    return static_cast<double>(ENGINE_GAMES) / seconds;
}

[[nodiscard]] Setup fixed_mcts_setup() {
    Setup setup{};
    setup.num_players = 2U;
    constexpr DefId kFixed[] = {
        DEF_VILLAGE,
        DEF_SMITHY,
        DEF_MARKET,
        DEF_FESTIVAL,
        DEF_LABORATORY,
        DEF_CELLAR,
        DEF_CHAPEL,
        DEF_MILITIA,
        DEF_WITCH,
        DEF_MOAT,
    };
    setup.kingdom_count = static_cast<std::uint8_t>(sizeof(kFixed) / sizeof(kFixed[0]));
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        setup.kingdom[i] = kFixed[i];
    }
    return setup;
}

[[nodiscard]] double measure_mcts_sims_per_sec() {
    GameState state = Game::new_game(fixed_mcts_setup(), 0x4D43'0001ULL);
    (void)Game::step(state, A_PASS);
    MctsConfig config{};
    config.sims_per_move = 1000U;
    config.determinizations = 2U;
    config.max_tree_nodes = 8192U;
    config.rollout_seed = 0x4D43'0002ULL;
    Mcts mcts(config);

    std::uint64_t checksum = 0;
    const auto start = Clock::now();
    for (int i = 0; i < MCTS_SEARCHES; ++i) {
        checksum += mcts.choose(state, Game::current_decision(state).player);
    }
    const auto end = Clock::now();
    if (checksum == 0U) {
        std::cout << "";
    }
    const double seconds = static_cast<double>(elapsed_ns(start, end)) / 1'000'000'000.0;
    return static_cast<double>(MCTS_SEARCHES * 1000) / seconds;
}

} // namespace

int main() {
    const ActionStream stream = record_big_money_stream();
    const Percentiles step = measure_step_ns(stream);
    const std::uint64_t clone_ns = measure_clone_ns();
    const std::uint64_t legal_ns = measure_legal_actions_ns();
    const double random_games_sec = measure_random_games_per_sec();
    const double bm_games_sec = measure_bm_games_per_sec();
    const double heuristic_games_sec = measure_heuristic_games_per_sec();
    const double engine_games_sec = measure_engine_games_per_sec();
    const double mcts_sims_sec = measure_mcts_sims_per_sec();

    std::cout << "{\n"
              << "  \"step_median_ns\": " << step.median << ",\n"
              << "  \"step_p99_ns\": " << step.p99 << ",\n"
              << "  \"clone_ns\": " << clone_ns << ",\n"
              << "  \"legal_actions_ns\": " << legal_ns << ",\n"
              << "  \"random_games_per_sec\": " << random_games_sec << ",\n"
              << "  \"bm_games_per_sec\": " << bm_games_sec << ",\n"
              << "  \"heuristic_games_per_sec\": " << heuristic_games_sec << ",\n"
              << "  \"engine_games_per_sec\": " << engine_games_sec << ",\n"
              << "  \"mcts_sims_per_sec\": " << mcts_sims_sec << "\n"
              << "}\n";
    return 0;
}
