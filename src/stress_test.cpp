#include "engine/game_runner.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>

static const std::vector<std::string> ALL_KINGDOM = {
    "Cellar", "Chapel", "Moat", "Harbinger", "Merchant",
    "Vassal", "Village", "Workshop", "Bureaucrat", "Gardens",
    "Laboratory", "Market", "Militia", "Moneylender", "Poacher",
    "Remodel", "Smithy", "Throne Room", "Bandit", "Council Room",
    "Festival", "Library", "Mine", "Sentry", "Witch", "Artisan"
};

struct ThreadStats {
    int completed = 0;
    int timed_out = 0;
    long total_turns = 0;
    long total_decisions = 0;
};

static void run_batch(int start_game, int num_games, ThreadStats& stats) {
    std::mt19937 kingdom_rng(42 + start_game);

    for (int g = 0; g < num_games; g++) {
        auto shuffled = ALL_KINGDOM;
        std::shuffle(shuffled.begin(), shuffled.end(), kingdom_rng);
        std::vector<std::string> kingdom(shuffled.begin(), shuffled.begin() + 10);

        GameRunner runner(2, kingdom);
        EngineBot a1, a2;
        std::vector<Agent*> agents = {&a1, &a2};
        auto result = runner.run(agents);

        if (result.total_turns >= 79) {
            stats.timed_out++;
        } else {
            stats.completed++;
            stats.total_turns += result.total_turns;
            stats.total_decisions += result.total_decisions;
        }
    }
}

int main() {
    BaseCards::register_all();
    BaseKingdom::register_all();

    constexpr int NUM_GAMES = 100000;
    int num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads < 1) num_threads = 4;

    std::cout << "Engine vs Engine: " << NUM_GAMES << " games on "
              << num_threads << " threads\n" << std::flush;

    int games_per_thread = NUM_GAMES / num_threads;
    int remainder = NUM_GAMES % num_threads;

    std::vector<std::thread> threads;
    std::vector<ThreadStats> stats(num_threads);

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < num_threads; t++) {
        int batch_size = games_per_thread + (t < remainder ? 1 : 0);
        int start_game = t * games_per_thread + std::min(t, remainder);
        threads.emplace_back(run_batch, start_game, batch_size, std::ref(stats[t]));
    }

    for (auto& t : threads) t.join();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    // Aggregate
    int completed = 0, timed_out = 0;
    long total_turns = 0, total_decisions = 0;
    for (const auto& s : stats) {
        completed += s.completed;
        timed_out += s.timed_out;
        total_turns += s.total_turns;
        total_decisions += s.total_decisions;
    }

    std::cout << "\n=== Results ===\n";
    std::cout << NUM_GAMES << " games in " << elapsed << "s ("
              << static_cast<int>(NUM_GAMES / elapsed) << " games/sec)\n";
    std::cout << "Completed: " << completed << "  Timed out: " << timed_out << "\n";
    if (completed > 0) {
        std::cout << "Avg turns: " << static_cast<double>(total_turns) / completed << "\n";
        std::cout << "Avg decisions: " << static_cast<double>(total_decisions) / completed << "\n";
    }

    return 0;
}
