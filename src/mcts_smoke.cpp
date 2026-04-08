// Smoke test for MCTSBot. Plays a small number of games against
// BetterRandomAgent and BigMoneyAgent and prints win counts + timing.
//
// MCTS is slow (200 sims per top-level decision), so this is intentionally
// small — it exists to verify that MCTSBot runs to completion without
// crashing and that it isn't catastrophically worse than the baselines.

#include "engine/game_runner.h"
#include "engine/card_ids.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

namespace {

const std::vector<std::string> KINGDOM = {
    "Cellar", "Moat", "Village", "Workshop", "Smithy",
    "Market", "Mine", "Witch", "Council Room", "Festival"
};

struct Tally {
    int mcts_wins = 0;
    int opp_wins = 0;
    int ties = 0;
    int timeouts = 0;
    double seconds = 0.0;
};

template <typename MakeOpp>
Tally play_matchup(int num_games, const std::string& opp_name, MakeOpp make_opp) {
    Tally t;
    auto start = std::chrono::high_resolution_clock::now();

    for (int g = 0; g < num_games; g++) {
        // Alternate seats so position bias is averaged out.
        int mcts_seat = g % 2;
        MCTSBot mcts(/*num_simulations=*/200, /*c=*/1.41421356237,
                     /*seed=*/static_cast<uint64_t>(g) * 2 + 1);
        auto opp = make_opp(static_cast<uint64_t>(g) * 2);

        std::vector<Agent*> agents(2);
        agents[mcts_seat] = &mcts;
        agents[1 - mcts_seat] = opp.get();

        GameRunner runner(2, KINGDOM);
        auto result = runner.run(agents);

        if (result.total_turns >= 79) { t.timeouts++; continue; }
        if (result.winner == mcts_seat)        t.mcts_wins++;
        else if (result.winner == 1 - mcts_seat) t.opp_wins++;
        else                                     t.ties++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    t.seconds = std::chrono::duration<double>(end - start).count();

    std::cout << "MCTSBot vs " << opp_name << " (" << num_games << " games):\n"
              << "  MCTS wins:  " << t.mcts_wins << "\n"
              << "  Opp wins:   " << t.opp_wins << "\n"
              << "  Ties:       " << t.ties << "\n"
              << "  Timeouts:   " << t.timeouts << "\n"
              << "  Elapsed:    " << t.seconds << "s\n\n";
    return t;
}

} // namespace

int main() {
    BaseCards::register_all();
    Level1Cards::register_all();
    CardIds::init();

    std::cout << "MCTSBot smoke test (200 simulations per top-level decision)\n\n";

    play_matchup(10, "BetterRandomAgent",
                 [](uint64_t s) -> std::unique_ptr<Agent> {
                     return std::make_unique<BetterRandomAgent>(s);
                 });

    play_matchup(10, "BigMoneyAgent",
                 [](uint64_t /*s*/) -> std::unique_ptr<Agent> {
                     return std::make_unique<BigMoneyAgent>();
                 });

    return 0;
}
