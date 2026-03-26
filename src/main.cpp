#include "engine/game_runner.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
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

struct BenchResult {
    std::string label;
    int num_games;
    int p1_wins, p2_wins, ties;
    double elapsed;
    double avg_turns;
    double avg_score_p1, avg_score_p2;
    int skipped;
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

    auto start = std::chrono::high_resolution_clock::now();

    for (int g = 0; g < max_games; g++) {
        auto kingdom = random_kingdom(kingdom_rng);
        GameRunner runner(2, kingdom);
        auto a1 = make_agent(t1, g * 2);
        auto a2 = make_agent(t2, g * 2 + 1);
        std::vector<Agent*> agents = {a1.get(), a2.get()};

        auto result = runner.run(agents);

        if (result.total_turns >= 119) {
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
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    return {label, games_played, p1_wins, p2_wins, ties, elapsed,
            games_played ? static_cast<double>(total_turns) / games_played : 0,
            games_played ? static_cast<double>(total_score_p1) / games_played : 0,
            games_played ? static_cast<double>(total_score_p2) / games_played : 0,
            skipped};
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
    std::cout << "    Avg turns: " << r.avg_turns << "\n\n" << std::flush;
}

int main() {
    BaseCards::register_all();
    Level1Cards::register_all();
    Level2Cards::register_all();

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
