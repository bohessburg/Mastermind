#include "v2/core/game.h"
#include "v2/drivers/bots.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace {

[[nodiscard]] int count_ordered(const OrderedZone& zone) {
    return zone.size;
}

[[nodiscard]] int count_zone(const std::uint8_t (&zone)[MAX_SLOTS]) {
    int total = 0;
    for (std::uint8_t slot = 0; slot < MAX_SLOTS; ++slot) {
        total += zone[slot];
    }
    return total;
}

[[nodiscard]] int count_player_cards(const PlayerState& player) {
    int total = 0;
    total += count_zone(player.hand);
    total += count_zone(player.exile);
    total += count_zone(player.tavern);
    total += count_zone(player.island_mat);
    total += count_ordered(player.deck);
    total += count_ordered(player.discard);
    total += player.in_play_size;
    return total;
}

[[nodiscard]] int count_pile_cards(const Pile& pile) {
    return pile.mixed_len > 0U ? pile.mixed_len : pile.count;
}

[[nodiscard]] int total_cards(const GameState& state) {
    int total = 0;
    for (PlayerId player = 0; player < state.num_players; ++player) {
        total += count_player_cards(state.players[player]);
    }
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        total += count_pile_cards(state.piles[i]);
    }
    for (std::uint8_t i = 0; i < state.num_nonsupply; ++i) {
        total += count_pile_cards(state.nonsupply[i]);
    }
    total += count_zone(state.trash);
    return total;
}

} // namespace

TEST_CASE("v2 card conservation holds through random-agent games", "[v2][conservation]") {
    for (std::uint64_t seed = 0; seed < 10U; ++seed) {
        GameState state = Game::new_game(Setup{}, 0xCA2D'0000ULL + seed);
        RandomBot bot0{0xA11C'0000ULL + seed};
        RandomBot bot1{0xA11C'1000ULL + seed};
        const int baseline = total_cards(state);

        bool done = false;
        int steps = 0;
        while (!done) {
            REQUIRE(total_cards(state) == baseline);

            ActionMask legal{};
            const int legal_count = Game::legal_actions(state, legal);
            REQUIRE(legal_count > 0);
            const PlayerId player = Game::current_decision(state).player;
            const Action action = player == 0U
                ? bot0.choose_action(state, legal, legal_count)
                : bot1.choose_action(state, legal, legal_count);

            done = Game::step(state, action);
            ++steps;
            REQUIRE(steps < 10000);
        }
        REQUIRE(total_cards(state) == baseline);
    }
}
