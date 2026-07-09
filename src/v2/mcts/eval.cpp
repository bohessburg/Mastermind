#include "v2/mcts/eval.h"

#include "v2/core/actions.h"
#include "v2/core/game.h"
#include "v2/core/rng.h"
#include "v2/core/score.h"
#include "v2/core/turns.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <optional>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr DefId kImplementedKingdoms[] = {
    DEF_CELLAR,
    DEF_CHAPEL,
    DEF_VILLAGE,
    DEF_SMITHY,
    DEF_WORKSHOP,
    DEF_REMODEL,
    DEF_MINE,
    DEF_MERCHANT,
    DEF_MILITIA,
    DEF_WITCH,
    DEF_MOAT,
    DEF_BUREAUCRAT,
    DEF_MARKET,
    DEF_FESTIVAL,
    DEF_LABORATORY,
    DEF_GARDENS,
    DEF_MONEYLENDER,
    DEF_POACHER,
    DEF_VASSAL,
    DEF_HARBINGER,
    DEF_THRONE_ROOM,
    DEF_COUNCIL_ROOM,
    DEF_ARTISAN,
    DEF_BANDIT,
    DEF_LIBRARY,
    DEF_SENTRY,
};

constexpr std::uint8_t kImplementedKingdomCount =
    static_cast<std::uint8_t>(sizeof(kImplementedKingdoms) / sizeof(kImplementedKingdoms[0]));

struct ThreadStats {
    std::uint32_t games = 0;
    std::uint32_t mcts_wins = 0;
    std::uint32_t opponent_wins = 0;
    std::uint32_t ties = 0;
    std::uint32_t truncated = 0;
    std::uint64_t mcts_searches = 0;
    std::uint64_t mcts_sims = 0;
    std::uint64_t mcts_search_ns = 0;
};

struct EvalBot {
    BotKind kind = BotKind::Engine;
    RandomBot random{};
    BigMoneyBot big_money{};
    HeuristicBot heuristic{};
    EngineBot engine{};
    std::optional<MctsBot> mcts{};

    EvalBot(BotKind bot_kind, std::uint64_t seed, const MctsConfig& mcts_config)
        : kind(bot_kind),
          random(seed),
          big_money(),
          heuristic(),
          engine(),
          mcts() {
        if (kind == BotKind::Mcts) {
            mcts.emplace(mcts_config);
        }
    }

    [[nodiscard]] Action choose(const GameState& state, const ActionMask& legal, int legal_count, ThreadStats& stats) {
        switch (kind) {
        case BotKind::Random:
            return random.choose_action(state, legal, legal_count);
        case BotKind::BigMoney:
            return big_money.choose_action(state, legal, legal_count);
        case BotKind::Heuristic:
            return heuristic.choose_action(state, legal, legal_count);
        case BotKind::Mcts: {
            if (!mcts.has_value()) {
                return legal_count > 0 ? legal.nth_set(0U) : A_PASS;
            }
            const std::uint64_t searches_before = mcts->searches;
            const std::uint64_t sims_before = mcts->sims;
            const auto start = Clock::now();
            const Action action = mcts->choose_action(state, legal, legal_count);
            const auto end = Clock::now();
            if (mcts->searches != searches_before) {
                stats.mcts_searches += mcts->searches - searches_before;
                stats.mcts_sims += mcts->sims - sims_before;
                stats.mcts_search_ns += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
            }
            return action;
        }
        case BotKind::Engine:
        default:
            return engine.choose_action(state, legal, legal_count);
        }
    }
};

[[nodiscard]] std::uint32_t default_threads() noexcept {
    const std::uint32_t hardware = std::thread::hardware_concurrency();
    return hardware == 0U ? 1U : hardware;
}

[[nodiscard]] MctsConfig config_for_game(const MctsEvalOptions& options, std::uint64_t seed) noexcept {
    MctsConfig config{};
    config.sims_per_move = options.sims;
    config.determinizations = options.determinizations == 0U ? 1U : options.determinizations;
    config.rollout_seed = seed;
    config.max_tree_nodes = options.sims >= 1000U ? 8192U : 4096U;
    return config;
}

[[nodiscard]] Setup setup_for_game(const MctsEvalOptions& options, std::uint32_t game_index) noexcept {
    if (options.kingdoms == MctsEvalKingdoms::Fixed) {
        return fixed_mcts_eval_setup();
    }
    const std::uint64_t pair_index = static_cast<std::uint64_t>(game_index / 2U);
    return random_mcts_eval_setup(options.seed ^ (0xD1B5'4A32'D192'ED03ULL * (pair_index + 1U)));
}

[[nodiscard]] PlayerId winner_for(const GameState& state) noexcept {
    const std::int16_t score0 = score(state, 0U);
    const std::int16_t score1 = score(state, 1U);
    if (score0 > score1) {
        return 0U;
    }
    if (score1 > score0) {
        return 1U;
    }
    return NONE;
}

void run_one_game(const MctsEvalOptions& options, std::uint32_t game_index, ThreadStats& stats) {
    const bool swapped = (game_index & 1U) != 0U;
    const PlayerId mcts_player = swapped ? 1U : 0U;
    const std::uint64_t game_seed =
        options.seed + (static_cast<std::uint64_t>(game_index) * 0x9E37'79B9'7F4A'7C15ULL);
    const Setup setup = setup_for_game(options, game_index);
    GameState state = Game::new_game(setup, game_seed);

    const MctsConfig mcts_config = config_for_game(options, game_seed ^ 0x4D43'5453'0000'0001ULL);
    EvalBot first(swapped ? bot_kind_for(options.opponent) : BotKind::Mcts, game_seed ^ 0xB007'0001ULL, mcts_config);
    EvalBot second(swapped ? BotKind::Mcts : bot_kind_for(options.opponent), game_seed ^ 0xB007'0002ULL, mcts_config);

    bool done = state.phase == static_cast<std::uint8_t>(Phase::Over);
    ActionMask legal{};
    while (!done) {
        const int legal_count = Game::legal_actions(state, legal);
        if (legal_count <= 0) {
            break;
        }
        const PlayerId player = Game::current_decision(state).player;
        EvalBot& bot = player == 0U ? first : second;
        const Action action = bot.choose(state, legal, legal_count, stats);
        if (!legal.test(action)) {
            break;
        }
        done = Game::step(state, action);
    }

    ++stats.games;
    if (state.truncated != 0U) {
        ++stats.truncated;
    }
    const PlayerId winner = winner_for(state);
    if (winner == NONE) {
        ++stats.ties;
    } else if (winner == mcts_player) {
        ++stats.mcts_wins;
    } else {
        ++stats.opponent_wins;
    }
}

void merge(ThreadStats& dst, const ThreadStats& src) noexcept {
    dst.games += src.games;
    dst.mcts_wins += src.mcts_wins;
    dst.opponent_wins += src.opponent_wins;
    dst.ties += src.ties;
    dst.truncated += src.truncated;
    dst.mcts_searches += src.mcts_searches;
    dst.mcts_sims += src.mcts_sims;
    dst.mcts_search_ns += src.mcts_search_ns;
}

} // namespace

double MctsEvalResult::mcts_win_percent() const noexcept {
    return games == 0U ? 0.0 : (100.0 * static_cast<double>(mcts_wins)) / static_cast<double>(games);
}

double MctsEvalResult::opponent_win_percent() const noexcept {
    return games == 0U ? 0.0 : (100.0 * static_cast<double>(opponent_wins)) / static_cast<double>(games);
}

double MctsEvalResult::tie_percent() const noexcept {
    return games == 0U ? 0.0 : (100.0 * static_cast<double>(ties)) / static_cast<double>(games);
}

double MctsEvalResult::mcts_sims_per_sec() const noexcept {
    if (mcts_search_ns == 0U) {
        return 0.0;
    }
    return (static_cast<double>(mcts_sims) * 1'000'000'000.0) / static_cast<double>(mcts_search_ns);
}

double MctsEvalResult::avg_move_ms() const noexcept {
    if (mcts_searches == 0U) {
        return 0.0;
    }
    return static_cast<double>(mcts_search_ns) / (1'000'000.0 * static_cast<double>(mcts_searches));
}

Setup fixed_mcts_eval_setup() noexcept {
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

Setup random_mcts_eval_setup(std::uint64_t seed) noexcept {
    Setup setup{};
    setup.num_players = 2U;
    setup.kingdom_count = 10U;
    DefId defs[kImplementedKingdomCount]{};
    for (std::uint8_t i = 0; i < kImplementedKingdomCount; ++i) {
        defs[i] = kImplementedKingdoms[i];
    }

    Xoshiro256pp rng = Xoshiro256pp::seeded(seed);
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        const std::uint32_t offset = rng.uniform(static_cast<std::uint32_t>(kImplementedKingdomCount - i));
        const std::uint8_t swap_index = static_cast<std::uint8_t>(i + offset);
        const DefId selected = defs[swap_index];
        defs[swap_index] = defs[i];
        defs[i] = selected;
        setup.kingdom[i] = selected;
    }
    return setup;
}

BotKind bot_kind_for(MctsEvalOpponent opponent) noexcept {
    switch (opponent) {
    case MctsEvalOpponent::Random:
        return BotKind::Random;
    case MctsEvalOpponent::BigMoney:
        return BotKind::BigMoney;
    case MctsEvalOpponent::Heuristic:
        return BotKind::Heuristic;
    case MctsEvalOpponent::Engine:
    default:
        return BotKind::Engine;
    }
}

MctsEvalResult run_mcts_eval(const MctsEvalOptions& options) {
    const auto start = Clock::now();
    const std::uint32_t thread_count = options.threads == 0U ? default_threads() : std::max(1U, options.threads);
    std::atomic<std::uint32_t> next_game{0U};
    std::vector<ThreadStats> thread_stats(thread_count);
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    for (std::uint32_t thread = 0; thread < thread_count; ++thread) {
        workers.emplace_back([&options, &next_game, &thread_stats, thread]() {
            while (true) {
                const std::uint32_t game = next_game.fetch_add(1U, std::memory_order_relaxed);
                if (game >= options.games) {
                    break;
                }
                run_one_game(options, game, thread_stats[thread]);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }

    ThreadStats total{};
    for (const ThreadStats& stats : thread_stats) {
        merge(total, stats);
    }

    const auto end = Clock::now();
    MctsEvalResult result{};
    result.games = total.games;
    result.mcts_wins = total.mcts_wins;
    result.opponent_wins = total.opponent_wins;
    result.ties = total.ties;
    result.truncated = total.truncated;
    result.mcts_searches = total.mcts_searches;
    result.mcts_sims = total.mcts_sims;
    result.mcts_search_ns = total.mcts_search_ns;
    result.wall_seconds = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1'000'000'000.0;
    return result;
}
