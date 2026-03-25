#include "engine/game_runner.h"

#include <chrono>
#include <iostream>

int main() {
    std::vector<std::string> kingdom = {
        "Smithy", "Village", "Market", "Laboratory", "Festival",
        "Cellar", "Chapel", "Workshop", "Moat", "Militia"
    };

    int num_games = 1000;
    int p1_wins = 0, p2_wins = 0, ties = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int g = 0; g < num_games; g++) {
        GameRunner runner(2, kingdom);
        BigMoneyAgent agent1;
        BigMoneyAgent agent2;
        std::vector<Agent*> agents = {&agent1, &agent2};

        auto result = runner.run(agents);
        if (result.winner == 0) p1_wins++;
        else if (result.winner == 1) p2_wins++;
        else ties++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Ran " << num_games << " games in " << elapsed << "s ("
              << static_cast<int>(num_games / elapsed) << " games/sec)\n";
    std::cout << "P1 wins: " << p1_wins << ", P2 wins: " << p2_wins
              << ", Ties: " << ties << "\n";

    return 0;
}
