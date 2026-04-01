#include "engine/game_runner.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

enum class AgentType { BETTER_RANDOM, BIG_MONEY, ENGINE };

static std::string agent_name(AgentType t) {
    switch (t) {
        case AgentType::BETTER_RANDOM: return "BetterRandom";
        case AgentType::BIG_MONEY:     return "BigMoney";
        case AgentType::ENGINE:        return "Engine";
    }
    return "???";
}

static std::unique_ptr<Agent> make_agent(AgentType t, uint64_t seed) {
    switch (t) {
        case AgentType::BETTER_RANDOM: return std::make_unique<BetterRandomAgent>(seed);
        case AgentType::BIG_MONEY:     return std::make_unique<BigMoneyAgent>();
        case AgentType::ENGINE:        return std::make_unique<EngineBot>();
    }
    return nullptr;
}

struct StratBucket {
    int games = 0, wins = 0, losses = 0, ties = 0;
};

struct BenchResult {
    std::string label;
    int num_games;
    int p1_wins, p2_wins, ties;
    double elapsed;
    double avg_turns;
    double avg_score_p1, avg_score_p2;
    int skipped;
    std::map<std::string, StratBucket> strat_breakdown;
};

static constexpr double BENCH_TIMEOUT_SEC = 10.0;

static const std::vector<std::string> ALL_KINGDOM = {
    // Level 1 (Base Set)
    "Cellar", "Chapel", "Moat", "Harbinger", "Merchant",
    "Vassal", "Village", "Workshop", "Bureaucrat", "Gardens",
    "Laboratory", "Market", "Militia", "Moneylender", "Poacher",
    "Remodel", "Smithy", "Throne Room", "Bandit", "Council Room",
    "Festival", "Library", "Mine", "Sentry", "Witch", "Artisan",
    // Level 2
    "Worker's Village", "Courtyard", "Hamlet", "Lookout", "Swindler", "Scholar"
};

static std::vector<std::string> random_kingdom(std::mt19937& rng) {
    auto shuffled = ALL_KINGDOM;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    return {shuffled.begin(), shuffled.begin() + 10};
}

static BenchResult run_bench(const std::string& label, int max_games,
                              AgentType t1, AgentType t2) {
    int p1_wins = 0, p2_wins = 0, ties = 0;
    long total_turns = 0;
    long total_score_p1 = 0, total_score_p2 = 0;
    int games_played = 0;
    int skipped = 0;
    std::mt19937 kingdom_rng(std::hash<std::string>{}(label));
    std::map<std::string, StratBucket> strat_breakdown;

    // Track strategy breakdown when Engine is one of the agents
    bool engine_is_p1 = (t1 == AgentType::ENGINE);
    bool engine_is_p2 = (t2 == AgentType::ENGINE);
    bool track_strat = (engine_is_p1 != engine_is_p2); // exactly one Engine

    auto start = std::chrono::high_resolution_clock::now();

    for (int g = 0; g < max_games; g++) {
        auto kingdom = random_kingdom(kingdom_rng);
        GameRunner runner(2, kingdom);
        auto a1 = make_agent(t1, g * 2);
        auto a2 = make_agent(t2, g * 2 + 1);
        std::vector<Agent*> agents = {a1.get(), a2.get()};

        std::string strat;
        if (track_strat) {
            strat = EngineBot::strategy_for(kingdom);
        }

        auto result = runner.run(agents);

        if (result.total_turns >= 79) {
            skipped++;
            continue;
        }

        if (result.winner == 0) p1_wins++;
        else if (result.winner == 1) p2_wins++;
        else ties++;
        total_turns += result.total_turns;
        total_score_p1 += result.scores[0];
        total_score_p2 += result.scores[1];
        games_played++;

        if (track_strat) {
            auto& b = strat_breakdown[strat];
            b.games++;
            bool engine_won = (engine_is_p1 && result.winner == 0) ||
                              (engine_is_p2 && result.winner == 1);
            bool engine_lost = (engine_is_p1 && result.winner == 1) ||
                               (engine_is_p2 && result.winner == 0);
            if (engine_won) b.wins++;
            else if (engine_lost) b.losses++;
            else b.ties++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    return {label, games_played, p1_wins, p2_wins, ties, elapsed,
            games_played ? static_cast<double>(total_turns) / games_played : 0,
            games_played ? static_cast<double>(total_score_p1) / games_played : 0,
            games_played ? static_cast<double>(total_score_p2) / games_played : 0,
            skipped, strat_breakdown};
}

static void print_result(const BenchResult& r) {
    int total = r.num_games + r.skipped;
    std::cout << "  " << r.label << "\n";
    std::cout << "    Total: " << total << " games ("
              << r.num_games << " completed, " << r.skipped << " timed out)\n";
    std::cout << "    " << r.elapsed << "s ("
              << static_cast<int>(total / r.elapsed) << " games/sec)\n";
    std::cout << "    P1 wins: " << r.p1_wins
              << "  P2 wins: " << r.p2_wins
              << "  Ties: " << r.ties << "\n";
    std::cout << "    Avg score: P1=" << r.avg_score_p1
              << "  P2=" << r.avg_score_p2 << "\n";
    std::cout << "    Avg turns: " << r.avg_turns << "\n";
    if (!r.strat_breakdown.empty()) {
        std::cout << "    Strategy breakdown (Engine win%):\n";
        for (auto& [strat, b] : r.strat_breakdown) {
            double pct = b.games > 0 ? 100.0 * b.wins / b.games : 0;
            std::cout << "      " << strat << ": "
                      << b.wins << "W/" << b.losses << "L/" << b.ties << "T"
                      << " (" << b.games << " games, " << pct << "%)\n";
        }
    }
    std::cout << "\n" << std::flush;
}

static void run_loss_analysis() {
    int n = 20000;
    std::mt19937 kingdom_rng(12345);
    std::map<std::string, int> loss_card_counts;
    std::map<std::string, int> win_card_counts;
    int total_losses = 0;
    int total_wins = 0;

    std::cout << "=== LOSS ANALYSIS: Engine vs BigMoney ===\n";
    std::cout << n << " games per seat...\n\n" << std::flush;

    for (int seat = 0; seat < 2; seat++) {
        for (int g = 0; g < n; g++) {
            auto kingdom = random_kingdom(kingdom_rng);
            GameRunner runner(2, kingdom);
            std::unique_ptr<Agent> engine, bm;
            engine = std::make_unique<EngineBot>();
            bm = std::make_unique<BigMoneyAgent>();

            std::vector<Agent*> agents;
            if (seat == 0) agents = {engine.get(), bm.get()};
            else agents = {bm.get(), engine.get()};

            auto result = runner.run(agents);
            if (result.total_turns >= 79) continue;

            bool engine_won = (seat == 0 && result.winner == 0) ||
                              (seat == 1 && result.winner == 1);
            bool engine_lost = (seat == 0 && result.winner == 1) ||
                               (seat == 1 && result.winner == 0);

            if (engine_lost) {
                total_losses++;
                for (const auto& card : kingdom)
                    loss_card_counts[card]++;
            }
            if (engine_won) {
                total_wins++;
                for (const auto& card : kingdom)
                    win_card_counts[card]++;
            }
        }
    }

    // Sort by loss count descending
    std::vector<std::pair<std::string, int>> sorted_losses(
        loss_card_counts.begin(), loss_card_counts.end());
    std::sort(sorted_losses.begin(), sorted_losses.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    printf("Total: %d wins, %d losses\n\n", total_wins, total_losses);
    printf("%-20s  Losses  Wins    Loss%%   Win%%    Delta\n", "Card");
    printf("%-20s  ------  ------  ------  ------  ------\n", "----");
    for (const auto& [card, losses] : sorted_losses) {
        int wins = win_card_counts.count(card) ? win_card_counts.at(card) : 0;
        double loss_pct = 100.0 * losses / total_losses;
        double win_pct = total_wins > 0 ? 100.0 * wins / total_wins : 0;
        double delta = loss_pct - win_pct;
        printf("%-20s  %5d   %5d   %5.1f%%  %5.1f%%  %+5.1f%%\n",
               card.c_str(), losses, wins, loss_pct, win_pct, delta);
    }
    printf("\nPositive Delta = card appears MORE in losses than wins (bad for Engine)\n");
    printf("Negative Delta = card appears MORE in wins than losses (good for Engine)\n");
}

static void run_sweep() {
    int n = 5000;
    std::vector<double> term_vals = {0.05, 0.08, 0.10, 0.12, 0.15, 0.20};
    std::vector<double> act_vals  = {0.15, 0.20, 0.25, 0.30, 0.35, 0.40};

    std::cout << "=== DENSITY LIMIT SWEEP ===\n";
    std::cout << "5000 games per seat per config\n\n";
    std::cout << "TermDens  ActDens   P1win%  P2win%  SeatAdj%\n";
    std::cout << "--------  --------  ------  ------  --------\n" << std::flush;

    struct SweepResult { double td, ad, p1pct, p2pct, seat; };
    std::vector<SweepResult> results;

    for (double td : term_vals) {
        for (double ad : act_vals) {
            EngineBot::terminal_density_limit = td;
            EngineBot::action_density_limit = ad;

            auto f1 = std::async(std::launch::async, run_bench,
                "EvBM", n, AgentType::ENGINE, AgentType::BIG_MONEY);
            auto f2 = std::async(std::launch::async, run_bench,
                "BMvE", n, AgentType::BIG_MONEY, AgentType::ENGINE);

            auto r1 = f1.get();
            auto r2 = f2.get();

            double p1pct = r1.num_games > 0 ?
                100.0 * r1.p1_wins / r1.num_games : 0;
            double p2pct = r2.num_games > 0 ?
                100.0 * r2.p2_wins / r2.num_games : 0;
            double seat = (p1pct + p2pct) / 2.0;

            results.push_back({td, ad, p1pct, p2pct, seat});

            printf("  %.2f      %.2f    %5.1f%%  %5.1f%%   %5.1f%%\n",
                   td, ad, p1pct, p2pct, seat);
            std::cout << std::flush;
        }
    }

    // Find and print best
    auto best = std::max_element(results.begin(), results.end(),
        [](const SweepResult& a, const SweepResult& b) { return a.seat < b.seat; });
    printf("\nBest: terminal=%.2f action=%.2f -> %.1f%% seat-adjusted\n",
           best->td, best->ad, best->seat);
}

int main(int argc, char* argv[]) {
    BaseCards::register_all();
    Level1Cards::register_all();

    if (argc > 1 && std::string(argv[1]) == "--sweep") {
        run_sweep();
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--losses") {
        run_loss_analysis();
        return 0;
    }

    int n = 5000;

    std::cout << "Random kingdoms (10 of 32 cards per game)\n";
    std::cout << n << " games per matchup, all in parallel\n\n" << std::flush;

    auto f1 = std::async(std::launch::async, run_bench, "Engine vs BigMoney", n, AgentType::ENGINE, AgentType::BIG_MONEY);
    auto f2 = std::async(std::launch::async, run_bench, "BigMoney vs Engine", n, AgentType::BIG_MONEY, AgentType::ENGINE);
    auto f3 = std::async(std::launch::async, run_bench, "Engine vs BetterRandom", n, AgentType::ENGINE, AgentType::BETTER_RANDOM);
    auto f4 = std::async(std::launch::async, run_bench, "BetterRandom vs Engine", n, AgentType::BETTER_RANDOM, AgentType::ENGINE);
    auto f5 = std::async(std::launch::async, run_bench, "BigMoney vs BetterRandom", n, AgentType::BIG_MONEY, AgentType::BETTER_RANDOM);
    auto f6 = std::async(std::launch::async, run_bench, "BetterRandom vs BigMoney", n, AgentType::BETTER_RANDOM, AgentType::BIG_MONEY);
    auto f7 = std::async(std::launch::async, run_bench, "Engine vs Engine", n, AgentType::ENGINE, AgentType::ENGINE);
    auto f8 = std::async(std::launch::async, run_bench, "BigMoney vs BigMoney", n, AgentType::BIG_MONEY, AgentType::BIG_MONEY);
    auto f9 = std::async(std::launch::async, run_bench, "BetterRandom vs BetterRandom", n, AgentType::BETTER_RANDOM, AgentType::BETTER_RANDOM);

    print_result(f1.get());
    print_result(f2.get());
    print_result(f3.get());
    print_result(f4.get());
    print_result(f5.get());
    print_result(f6.get());
    print_result(f7.get());
    print_result(f8.get());
    print_result(f9.get());

    return 0;
}
