#include "v2/core/game.h"
#include "v2/drivers/bots.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace {

[[nodiscard]] Setup all_implemented_kingdom_setup() {
    Setup setup{};
    constexpr DefId kKingdom[] = {
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
    };
    setup.kingdom_count = static_cast<std::uint8_t>(sizeof(kKingdom) / sizeof(kKingdom[0]));
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        setup.kingdom[i] = kKingdom[i];
    }
    return setup;
}

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

[[nodiscard]] int count_bandit_revealed(const GameState& state) {
    int total = 0;
    for (std::uint8_t i = 0; i < state.effect_depth; ++i) {
        const EffectFrame& frame = state.effect_stack[i];
        if (frame.source != DEF_BANDIT) {
            continue;
        }
        const std::uint8_t revealed_count = frame.data[3] < 0 ? 0U : static_cast<std::uint8_t>(frame.data[3]);
        for (std::uint8_t j = 0; j < revealed_count && j < 2U; ++j) {
            if (frame.data[1 + j] >= 0) {
                ++total;
            }
        }
    }
    return total;
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
    total += count_bandit_revealed(state);
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

TEST_CASE("v2 card conservation holds with all implemented kingdom cards", "[v2][conservation][fuzz]") {
    const Setup setup = all_implemented_kingdom_setup();
    for (std::uint64_t seed = 0; seed < 25U; ++seed) {
        GameState state = Game::new_game(setup, 0xA11D'0000ULL + seed);
        RandomBot bot0{0xB007'0000ULL + seed};
        RandomBot bot1{0xB007'1000ULL + seed};
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
            REQUIRE(steps < 25000);
        }
        REQUIRE(total_cards(state) == baseline);
    }
}
