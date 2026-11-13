#include "v2/core/rng.h"
#include "v2/core/score.h"
#include "v2/core/setup.h"
#include "v2/drivers/bots.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::array<DefId, 26> kImplementedKingdoms = {
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

enum class KingdomMode : std::uint8_t {
    Random,
    Fixed,
};

struct Options {
    BotKind a = BotKind::EngineV3;
    BotKind b = BotKind::Engine;
    std::uint32_t games = 5000U;
    std::uint32_t threads = 0U;
    std::uint64_t seed = 42U;
    KingdomMode kingdom = KingdomMode::Random;
    bool json = false;
    bool custom_cards = false;
    Setup custom_setup{};
    bool trace = false;
};

struct CardStats {
    std::uint64_t boards = 0U;
    std::uint64_t games = 0U;
    std::uint64_t wins_a = 0U;
};

struct DuelStats {
    std::uint64_t games = 0U;
    std::uint64_t wins_a = 0U;
    std::uint64_t wins_b = 0U;
    std::uint64_t ties = 0U;
    std::uint64_t truncated = 0U;
    std::uint64_t total_turns = 0U;
    std::uint64_t a_p1_games = 0U;
    std::uint64_t a_p1_wins = 0U;
    std::uint64_t a_p2_games = 0U;
    std::uint64_t a_p2_wins = 0U;
    std::array<CardStats, kImplementedKingdoms.size()> cards{};
};

struct CardRow {
    std::size_t index = 0U;
    double win_percent = 0.0;
    double delta_percent = 0.0;
};

[[nodiscard]] const char* bot_name(BotKind kind) noexcept {
    switch (kind) {
    case BotKind::Random:
        return "random";
    case BotKind::BigMoney:
        return "bm";
    case BotKind::Heuristic:
        return "heuristic";
    case BotKind::Engine:
        return "engine";
    case BotKind::EngineV3:
        return "engine3";
    case BotKind::Thinner:
        return "thinner";
    case BotKind::Mcts:
    default:
        return "mcts";
    }
}

[[nodiscard]] bool parse_bot(const std::string& name, BotKind& out) noexcept {
    if (name == "random") {
        out = BotKind::Random;
        return true;
    }
    if (name == "bm") {
        out = BotKind::BigMoney;
        return true;
    }
    if (name == "heuristic") {
        out = BotKind::Heuristic;
        return true;
    }
    if (name == "engine") {
        out = BotKind::Engine;
        return true;
    }
    if (name == "engine3") {
        out = BotKind::EngineV3;
        return true;
    }
    if (name == "thinner") {
        out = BotKind::Thinner;
        return true;
    }
    return false;
}

[[nodiscard]] bool parse_u32(const char* text, std::uint32_t& out) noexcept {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 0);
    if (end == text || *end != '\0' || parsed > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    out = static_cast<std::uint32_t>(parsed);
    return true;
}

[[nodiscard]] bool parse_u64(const char* text, std::uint64_t& out) noexcept {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 0);
    if (end == text || *end != '\0') {
        return false;
    }
    out = static_cast<std::uint64_t>(parsed);
    return true;
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " [--a random|bm|heuristic|engine|engine3|thinner]"
              << " [--b random|bm|heuristic|engine|engine3|thinner]"
              << " [--games even-count] [--threads 0|count] [--seed value]"
              << " [--kingdom random|fixed] [--cards Name,Name,...] [--json]\n";
}

[[nodiscard]] bool parse_cards(const char* text, Setup& setup) noexcept {
    setup = Setup{};
    setup.num_players = 2U;
    std::string token;
    const std::string list = text;
    std::size_t start = 0U;
    while (start <= list.size()) {
        const std::size_t comma = list.find(',', start);
        token = list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            DefId found = NONE;
            for (const DefId def : kImplementedKingdoms) {
                const char* name = card_def(def).name;
                if (name != nullptr && token == name) {
                    found = def;
                    break;
                }
            }
            if (found == NONE || setup.kingdom_count >= MAX_KINGDOM_DEFS) {
                std::cerr << "Unknown or duplicate-capacity kingdom card: " << token << "\n";
                return false;
            }
            setup.kingdom[setup.kingdom_count] = found;
            setup.kingdom_count = static_cast<std::uint8_t>(setup.kingdom_count + 1U);
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1U;
    }
    return setup.kingdom_count > 0U;
}

[[nodiscard]] bool parse_options(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        }
        if (arg == "--json") {
            options.json = true;
            continue;
        }
        if (arg == "--trace") {
            options.trace = true;
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << arg << "\n";
            return false;
        }
        const char* value = argv[++i];
        if (arg == "--a") {
            if (!parse_bot(value, options.a)) {
                std::cerr << "Unknown bot: " << value << "\n";
                return false;
            }
        } else if (arg == "--b") {
            if (!parse_bot(value, options.b)) {
                std::cerr << "Unknown bot: " << value << "\n";
                return false;
            }
        } else if (arg == "--games") {
            if (!parse_u32(value, options.games) || options.games == 0U || (options.games & 1U) != 0U) {
                std::cerr << "--games must be a positive even count\n";
                return false;
            }
        } else if (arg == "--threads") {
            if (!parse_u32(value, options.threads)) {
                std::cerr << "Invalid thread count: " << value << "\n";
                return false;
            }
        } else if (arg == "--seed") {
            if (!parse_u64(value, options.seed)) {
                std::cerr << "Invalid seed: " << value << "\n";
                return false;
            }
        } else if (arg == "--cards") {
            if (!parse_cards(value, options.custom_setup)) {
                return false;
            }
            options.custom_cards = true;
        } else if (arg == "--kingdom") {
            const std::string mode = value;
            if (mode == "random") {
                options.kingdom = KingdomMode::Random;
            } else if (mode == "fixed") {
                options.kingdom = KingdomMode::Fixed;
            } else {
                std::cerr << "Unknown kingdom mode: " << value << "\n";
                return false;
            }
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return false;
        }
    }
    return true;
}

[[nodiscard]] Setup fixed_setup() noexcept {
    Setup setup{};
    setup.num_players = 2U;
    constexpr std::array<DefId, 10> kFixed = {
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
    setup.kingdom_count = static_cast<std::uint8_t>(kFixed.size());
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        setup.kingdom[i] = kFixed[i];
    }
    return setup;
}

[[nodiscard]] Setup random_setup(std::uint64_t seed) noexcept {
    Setup setup{};
    setup.num_players = 2U;
    setup.kingdom_count = 10U;
    std::array<DefId, kImplementedKingdoms.size()> defs = kImplementedKingdoms;
    Xoshiro256pp rng = Xoshiro256pp::seeded(seed);
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        const std::uint32_t offset = rng.uniform(static_cast<std::uint32_t>(defs.size() - i));
        const std::uint8_t swap_index = static_cast<std::uint8_t>(i + offset);
        const DefId selected = defs[swap_index];
        defs[swap_index] = defs[i];
        defs[i] = selected;
        setup.kingdom[i] = selected;
    }
    return setup;
}

void trace_game(const Options& options) {
    const Setup setup = options.custom_cards ? options.custom_setup : fixed_setup();
    GameState state = Game::new_game(setup, options.seed);
    RandomBot rng_a(1), rng_b(2);
    BigMoneyBot bm{};
    HeuristicBot heur{};
    EngineBot eng_a{}, eng_b{};
    EngineBotV3 v3_a{}, v3_b{};
    ThinnerBot thinner_a{}, thinner_b{};

    const auto choose = [&](BotKind kind, bool is_a, const ActionMask& legal, int n) -> Action {
        switch (kind) {
        case BotKind::Random: return (is_a ? rng_a : rng_b).choose_action(state, legal, n);
        case BotKind::BigMoney: return bm.choose_action(state, legal, n);
        case BotKind::Heuristic: return heur.choose_action(state, legal, n);
        case BotKind::Engine: return (is_a ? eng_a : eng_b).choose_action(state, legal, n);
        case BotKind::EngineV3: return (is_a ? v3_a : v3_b).choose_action(state, legal, n);
        case BotKind::Thinner: return (is_a ? thinner_a : thinner_b).choose_action(state, legal, n);
        default: return (is_a ? v3_a : v3_b).choose_action(state, legal, n);
        }
    };

    const auto is_treasure_play = [&](Action action) {
        const DefId def = action_def(action, A_PLAY_BASE);
        return def < card_def_count() && (card_def(def).types & TYPE_TREASURE) != 0U;
    };

    ActionMask legal{};
    bool done = state.phase == static_cast<std::uint8_t>(Phase::Over);
    while (!done) {
        const int n = Game::legal_actions(state, legal);
        if (n <= 0) {
            break;
        }
        const PlayerId p = Game::current_decision(state).player;
        const bool is_a = p == 0U;
        const Action action = choose(is_a ? options.a : options.b, is_a, legal, n);
        if (!legal.test(action)) {
            break;
        }
        const int my_turn = static_cast<int>(state.turn_counter / 2U) + 1;
        if (action >= A_BUY_BASE && action < A_EVENT_BASE) {
            const DefId def = action_def(action, A_BUY_BASE);
            std::cout << "T" << std::setw(2) << my_turn << " P" << int(p)
                      << " [" << bot_name(is_a ? options.a : options.b) << "]"
                      << " coins=" << int(state.coins)
                      << " buys=" << int(state.buys)
                      << " BUY " << card_def(def).name << "\n";
        } else if (action == A_PASS
            && static_cast<DecisionKind>(state.decision.kind) == DecisionKind::PhaseBuy
            && state.coins >= 4) {
            std::cout << "T" << std::setw(2) << my_turn << " P" << int(p)
                      << " [" << bot_name(is_a ? options.a : options.b) << "]"
                      << " coins=" << int(state.coins) << " PASS-BUY\n";
        } else if (action_is_play(action) && !is_treasure_play(action)) {
            const DefId def = action_def(action, A_PLAY_BASE);
            std::cout << "T" << std::setw(2) << my_turn << " P" << int(p)
                      << " play " << card_def(def).name << "\n";
        }
        done = Game::step(state, action);
    }
    int provinces = 0;
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        const Pile& pile = state.piles[i];
        const Slot top = pile.mixed_len > 0U ? pile.mixed[pile.mixed_len - 1U] : pile.base;
        if (top < state.num_slots && state.slot_to_def[top] == DEF_PROVINCE) {
            provinces = pile.mixed_len > 0U ? int(pile.mixed_len) : int(pile.count);
        }
    }
    std::cout << "END turns=" << state.turn_counter
              << " scores=" << score(state, 0) << "," << score(state, 1)
              << " provinces_left=" << provinces << "\n";
}

[[nodiscard]] Setup setup_for_pair(const Options& options, std::uint32_t pair) noexcept {
    if (options.custom_cards) {
        return options.custom_setup;
    }
    if (options.kingdom == KingdomMode::Fixed) {
        return fixed_setup();
    }
    const std::uint64_t kingdom_seed = options.seed
        ^ (0xD1B5'4A32'D192'ED03ULL * (static_cast<std::uint64_t>(pair) + 1U));
    return random_setup(kingdom_seed);
}

void account_game(DuelStats& stats, const GameResult& result, bool a_is_p1) noexcept {
    ++stats.games;
    stats.total_turns += result.turns;
    if (a_is_p1) {
        ++stats.a_p1_games;
    } else {
        ++stats.a_p2_games;
    }
    if (result.truncated) {
        ++stats.truncated;
    }
    if (result.winner == NONE) {
        ++stats.ties;
        return;
    }
    const bool a_won = a_is_p1 ? result.winner == 0U : result.winner == 1U;
    if (a_won) {
        ++stats.wins_a;
        if (a_is_p1) {
            ++stats.a_p1_wins;
        } else {
            ++stats.a_p2_wins;
        }
    } else {
        ++stats.wins_b;
    }
}

void run_pair(const Options& options, std::uint32_t pair, DuelStats& stats) noexcept {
    const Setup setup = setup_for_pair(options, pair);
    const std::uint64_t game_seed = options.seed
        + (0x9E37'79B9'7F4A'7C15ULL * (static_cast<std::uint64_t>(pair) + 1U));
    BotSpec bot_a{options.a, game_seed ^ 0xA11C'E001ULL};
    BotSpec bot_b{options.b, game_seed ^ 0xB007'B001ULL};

    const std::uint64_t wins_before = stats.wins_a;
    const GameResult first = run_game(setup, game_seed, bot_a, bot_b);
    account_game(stats, first, true);
    const GameResult second = run_game(setup, game_seed, bot_b, bot_a);
    account_game(stats, second, false);
    const std::uint64_t pair_wins_a = stats.wins_a - wins_before;

    for (std::size_t i = 0; i < kImplementedKingdoms.size(); ++i) {
        bool present = false;
        for (std::uint8_t card = 0; card < setup.kingdom_count; ++card) {
            if (setup.kingdom[card] == kImplementedKingdoms[i]) {
                present = true;
                break;
            }
        }
        if (present) {
            CardStats& card_stats = stats.cards[i];
            ++card_stats.boards;
            card_stats.games += 2U;
            card_stats.wins_a += pair_wins_a;
        }
    }
}

void merge(DuelStats& dst, const DuelStats& src) noexcept {
    dst.games += src.games;
    dst.wins_a += src.wins_a;
    dst.wins_b += src.wins_b;
    dst.ties += src.ties;
    dst.truncated += src.truncated;
    dst.total_turns += src.total_turns;
    dst.a_p1_games += src.a_p1_games;
    dst.a_p1_wins += src.a_p1_wins;
    dst.a_p2_games += src.a_p2_games;
    dst.a_p2_wins += src.a_p2_wins;
    for (std::size_t i = 0; i < dst.cards.size(); ++i) {
        dst.cards[i].boards += src.cards[i].boards;
        dst.cards[i].games += src.cards[i].games;
        dst.cards[i].wins_a += src.cards[i].wins_a;
    }
}

[[nodiscard]] double percent(std::uint64_t numerator, std::uint64_t denominator) noexcept {
    return denominator == 0U ? 0.0
        : (100.0 * static_cast<double>(numerator)) / static_cast<double>(denominator);
}

[[nodiscard]] std::array<CardRow, kImplementedKingdoms.size()> card_rows(const DuelStats& stats) {
    const double overall = percent(stats.wins_a, stats.games);
    std::array<CardRow, kImplementedKingdoms.size()> rows{};
    for (std::size_t i = 0; i < rows.size(); ++i) {
        rows[i].index = i;
        rows[i].win_percent = percent(stats.cards[i].wins_a, stats.cards[i].games);
        rows[i].delta_percent = rows[i].win_percent - overall;
    }
    std::sort(rows.begin(), rows.end(), [](const CardRow& left, const CardRow& right) {
        if (left.delta_percent != right.delta_percent) {
            return left.delta_percent < right.delta_percent;
        }
        return std::string(card_def(kImplementedKingdoms[left.index]).name)
            < std::string(card_def(kImplementedKingdoms[right.index]).name);
    });
    return rows;
}

void print_text(const Options& options, const DuelStats& stats, double elapsed_seconds, std::uint32_t threads) {
    const double overall = percent(stats.wins_a, stats.games);
    const double a_p1 = percent(stats.a_p1_wins, stats.a_p1_games);
    const double a_p2 = percent(stats.a_p2_wins, stats.a_p2_games);
    const double avg_turns = stats.games == 0U ? 0.0
        : static_cast<double>(stats.total_turns) / static_cast<double>(stats.games);
    const double games_per_second = elapsed_seconds <= 0.0 ? 0.0
        : static_cast<double>(stats.games) / elapsed_seconds;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "v2_botduel  a=" << bot_name(options.a) << "  b=" << bot_name(options.b)
              << "  kingdom=" << (options.kingdom == KingdomMode::Random ? "random" : "fixed")
              << "  threads=" << threads << "\n";
    std::cout << "games=" << stats.games << " wins_a=" << stats.wins_a << " wins_b=" << stats.wins_b
              << " ties=" << stats.ties << " truncated=" << stats.truncated << "\n";
    std::cout << "seat-adjusted A win%=" << overall << "  as-P1=" << a_p1 << "  as-P2=" << a_p2
              << "  avg_turns=" << avg_turns << "\n";
    std::cout << "elapsed=" << elapsed_seconds << "s  games/sec=" << games_per_second << "\n";
    std::cout << "\nper-card A win delta\n";
    std::cout << std::left << std::setw(16) << "card" << std::right << std::setw(9) << "boards"
              << std::setw(11) << "A win%" << std::setw(10) << "delta" << "\n";
    for (const CardRow& row : card_rows(stats)) {
        const CardStats& card = stats.cards[row.index];
        std::cout << std::left << std::setw(16) << card_def(kImplementedKingdoms[row.index]).name
                  << std::right << std::setw(9) << card.boards
                  << std::setw(11) << row.win_percent
                  << std::setw(10) << row.delta_percent << "\n";
    }
}

void print_json(const Options& options, const DuelStats& stats, double elapsed_seconds, std::uint32_t threads) {
    const double overall = percent(stats.wins_a, stats.games);
    const double a_p1 = percent(stats.a_p1_wins, stats.a_p1_games);
    const double a_p2 = percent(stats.a_p2_wins, stats.a_p2_games);
    const double avg_turns = stats.games == 0U ? 0.0
        : static_cast<double>(stats.total_turns) / static_cast<double>(stats.games);
    const double games_per_second = elapsed_seconds <= 0.0 ? 0.0
        : static_cast<double>(stats.games) / elapsed_seconds;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "{\n"
              << "  \"a\": \"" << bot_name(options.a) << "\",\n"
              << "  \"b\": \"" << bot_name(options.b) << "\",\n"
              << "  \"kingdom\": \"" << (options.kingdom == KingdomMode::Random ? "random" : "fixed") << "\",\n"
              << "  \"threads\": " << threads << ",\n"
              << "  \"games\": " << stats.games << ",\n"
              << "  \"wins_a\": " << stats.wins_a << ",\n"
              << "  \"wins_b\": " << stats.wins_b << ",\n"
              << "  \"ties\": " << stats.ties << ",\n"
              << "  \"truncated\": " << stats.truncated << ",\n"
              << "  \"seat_adjusted_win_percent\": " << overall << ",\n"
              << "  \"as_p1_win_percent\": " << a_p1 << ",\n"
              << "  \"as_p2_win_percent\": " << a_p2 << ",\n"
              << "  \"avg_turns\": " << avg_turns << ",\n"
              << "  \"elapsed_seconds\": " << elapsed_seconds << ",\n"
              << "  \"games_per_second\": " << games_per_second << ",\n"
              << "  \"cards\": [\n";
    const auto rows = card_rows(stats);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const CardRow& row = rows[i];
        const CardStats& card = stats.cards[row.index];
        std::cout << "    {\"card\": \"" << card_def(kImplementedKingdoms[row.index]).name
                  << "\", \"boards\": " << card.boards
                  << ", \"a_win_percent\": " << row.win_percent
                  << ", \"delta_percent\": " << row.delta_percent << "}";
        std::cout << (i + 1U == rows.size() ? "\n" : ",\n");
    }
    std::cout << "  ]\n}\n";
}

} // namespace

int main(int argc, char** argv) {
    Options options{};
    if (!parse_options(argc, argv, options)) {
        return 2;
    }

    if (options.trace) {
        trace_game(options);
        return 0;
    }

    const std::uint32_t pairs = options.games / 2U;
    std::uint32_t thread_count = options.threads;
    if (thread_count == 0U) {
        thread_count = std::thread::hardware_concurrency();
    }
    if (thread_count == 0U) {
        thread_count = 1U;
    }
    thread_count = std::min(thread_count, pairs);

    std::vector<DuelStats> per_thread(thread_count);
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    const auto start = Clock::now();
    for (std::uint32_t thread = 0; thread < thread_count; ++thread) {
        const std::uint32_t begin = static_cast<std::uint32_t>((static_cast<std::uint64_t>(pairs) * thread) / thread_count);
        const std::uint32_t end = static_cast<std::uint32_t>((static_cast<std::uint64_t>(pairs) * (thread + 1U)) / thread_count);
        workers.emplace_back([&options, &per_thread, thread, begin, end]() {
            for (std::uint32_t pair = begin; pair < end; ++pair) {
                run_pair(options, pair, per_thread[thread]);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    const auto end = Clock::now();

    DuelStats total{};
    for (const DuelStats& stats : per_thread) {
        merge(total, stats);
    }
    const double elapsed_seconds = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1'000'000'000.0;
    if (options.json) {
        print_json(options, total, elapsed_seconds, thread_count);
    } else {
        print_text(options, total, elapsed_seconds, thread_count);
    }
    return 0;
}
