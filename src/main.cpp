#include "engine/game_runner.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

enum class AgentType { RANDOM, BIG_MONEY, HEURISTIC, ENGINE };

static std::string agent_name(AgentType t) {
    switch (t) {
        case AgentType::RANDOM:    return "Random";
        case AgentType::BIG_MONEY: return "BigMoney";
        case AgentType::HEURISTIC: return "Heuristic";
        case AgentType::ENGINE:    return "Engine";
    }
    return "???";
}

static std::unique_ptr<Agent> make_agent(AgentType t, uint64_t seed) {
    switch (t) {
        case AgentType::RANDOM:    return std::make_unique<RandomAgent>(seed);
        case AgentType::BIG_MONEY: return std::make_unique<BigMoneyAgent>();
        case AgentType::HEURISTIC: return std::make_unique<HeuristicAgent>();
        case AgentType::ENGINE:    return std::make_unique<EngineBot>();
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
};

static constexpr double BENCH_TIMEOUT_SEC = 10.0;

static BenchResult run_bench(const std::string& label, int max_games,
                              const std::vector<std::string>& kingdom,
                              AgentType t1, AgentType t2) {
    int p1_wins = 0, p2_wins = 0, ties = 0;
    long total_turns = 0;
    long total_score_p1 = 0, total_score_p2 = 0;
    int games_played = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int g = 0; g < max_games; g++) {
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(now - start).count() >= BENCH_TIMEOUT_SEC) break;

        GameRunner runner(2, kingdom);
        auto a1 = make_agent(t1, g * 2);
        auto a2 = make_agent(t2, g * 2 + 1);
        std::vector<Agent*> agents = {a1.get(), a2.get()};

        auto result = runner.run(agents);
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
            games_played ? static_cast<double>(total_score_p2) / games_played : 0};
}

static void print_result(const BenchResult& r) {
    std::cout << "\n  " << r.label << "\n";
    std::cout << "  " << r.num_games << " games in " << r.elapsed << "s ("
              << static_cast<int>(r.num_games / r.elapsed) << " games/sec)\n";
    std::cout << "  P1 wins: " << r.p1_wins
              << "  P2 wins: " << r.p2_wins
              << "  Ties: " << r.ties << "\n";
    std::cout << "  Avg score: P1=" << r.avg_score_p1
              << "  P2=" << r.avg_score_p2 << "\n";
    std::cout << "  Avg turns: " << r.avg_turns << "\n";
}

int main() {
    std::vector<std::string> kingdom = {
        "Smithy", "Village", "Market", "Laboratory", "Festival",
        "Cellar", "Chapel", "Workshop", "Moat", "Militia"
    };

    int n = 100000; // cap, actual count limited by 10s timeout per matchup

    std::cout << "Kingdom: ";
    for (const auto& k : kingdom) std::cout << k << " ";
    std::cout << "\n";
    std::cout << "(Each matchup runs for up to " << BENCH_TIMEOUT_SEC << "s)\n";

    print_result(run_bench("BigMoney vs BigMoney", n, kingdom,
                            AgentType::BIG_MONEY, AgentType::BIG_MONEY));

    print_result(run_bench("Heuristic vs BigMoney", n, kingdom,
                            AgentType::HEURISTIC, AgentType::BIG_MONEY));

    print_result(run_bench("BigMoney vs Heuristic", n, kingdom,
                            AgentType::BIG_MONEY, AgentType::HEURISTIC));

    print_result(run_bench("Engine vs BigMoney", n, kingdom,
                            AgentType::ENGINE, AgentType::BIG_MONEY));

    print_result(run_bench("BigMoney vs Engine", n, kingdom,
                            AgentType::BIG_MONEY, AgentType::ENGINE));

    print_result(run_bench("Engine vs Heuristic", n, kingdom,
                            AgentType::ENGINE, AgentType::HEURISTIC));

    print_result(run_bench("BigMoney vs Random", n, kingdom,
                            AgentType::BIG_MONEY, AgentType::RANDOM));

    return 0;
}
