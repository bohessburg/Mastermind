#include "fuzz_harness.h"

#include "v2/core/defs.h"
#include "v2/core/determinize.h"
#include "v2/core/game.h"
#include "v2/core/setup.h"
#include "v2/drivers/bots.h"
#include "v2/encode/encoder.h"
#include "v2/mcts/selfplay.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstring>

namespace {

[[nodiscard]] Setup determinize_setup() noexcept {
    Setup setup{};
    setup.num_players = 2;
    setup.kingdom_count = 10;
    setup.kingdom[0] = DEF_VILLAGE;
    setup.kingdom[1] = DEF_SMITHY;
    setup.kingdom[2] = DEF_MARKET;
    setup.kingdom[3] = DEF_REMODEL;
    setup.kingdom[4] = DEF_MINE;
    setup.kingdom[5] = DEF_MOAT;
    setup.kingdom[6] = DEF_WITCH;
    setup.kingdom[7] = DEF_BANDIT;
    setup.kingdom[8] = DEF_LIBRARY;
    setup.kingdom[9] = DEF_SENTRY;
    return setup;
}

[[nodiscard]] std::array<float, OBS_SIZE> encoded(const GameState& state, PlayerId player) noexcept {
    std::array<float, OBS_SIZE> out{};
    encode(state, player, out.data());
    return out;
}

[[nodiscard]] std::array<float, OBS_SIZE_V2> encoded_v2(const GameState& state, PlayerId player) noexcept {
    std::array<float, OBS_SIZE_V2> out{};
    encode_v2(state, player, out.data());
    return out;
}

[[nodiscard]] std::array<float, OBS_SIZE_V3> encoded_v3(const GameState& state, PlayerId player) noexcept {
    std::array<float, OBS_SIZE_V3> out{};
    encode_v3(state, player, out.data());
    return out;
}

void step_random(GameState& state, RandomBot& bot) {
    ActionMask legal{};
    const int legal_count = Game::legal_actions(state, legal);
    REQUIRE(legal_count > 0);
    const PlayerId player = Game::current_decision(state).player;
    REQUIRE(player < state.num_players);
    (void)Game::step(state, bot.choose_action(state, legal, legal_count));
}

void seed_hidden_pool(GameState& state) {
    PlayerState& opponent = state.players[1];
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        opponent.hand[slot] = 0;
    }
    opponent.deck.size = 0;

    const Slot copper = slot_of(state, DEF_COPPER);
    const Slot estate = slot_of(state, DEF_ESTATE);
    const Slot silver = slot_of(state, DEF_SILVER);
    const Slot gold = slot_of(state, DEF_GOLD);
    const Slot village = slot_of(state, DEF_VILLAGE);
    REQUIRE(copper != NONE);
    REQUIRE(estate != NONE);
    REQUIRE(silver != NONE);
    REQUIRE(gold != NONE);
    REQUIRE(village != NONE);

    opponent.hand[copper] = 3;
    opponent.hand[estate] = 2;
    opponent.deck.cards[opponent.deck.size++] = silver;
    opponent.deck.cards[opponent.deck.size++] = gold;
    opponent.deck.cards[opponent.deck.size++] = copper;
    opponent.deck.cards[opponent.deck.size++] = estate;
    opponent.deck.cards[opponent.deck.size++] = village;
}

[[nodiscard]] bool hand_equal(
    const GameState& lhs,
    const GameState& rhs,
    PlayerId player) noexcept {
    return std::memcmp(lhs.players[player].hand, rhs.players[player].hand, sizeof(lhs.players[player].hand)) == 0;
}

[[nodiscard]] std::array<std::uint16_t, MAX_SLOTS> hidden_zone_multiset(
    const GameState& state,
    PlayerId player) noexcept {
    std::array<std::uint16_t, MAX_SLOTS> counts{};
    const PlayerState& owner = state.players[player];
    for (Slot slot = 0U; slot < state.num_slots; ++slot) {
        counts[slot] = owner.hand[slot];
    }
    for (std::uint8_t index = 0U; index < owner.deck.size; ++index) {
        const Slot slot = owner.deck.cards[index];
        if (slot < state.num_slots) {
            ++counts[slot];
        }
    }
    return counts;
}

[[nodiscard]] std::uint16_t hidden_hand_size(const GameState& state, PlayerId player) noexcept {
    std::uint16_t total = 0U;
    for (Slot slot = 0U; slot < state.num_slots; ++slot) {
        total = static_cast<std::uint16_t>(total + state.players[player].hand[slot]);
    }
    return total;
}

[[nodiscard]] bool hidden_zones_equal(
    const GameState& lhs,
    const GameState& rhs,
    PlayerId player) noexcept {
    const PlayerState& left = lhs.players[player];
    const PlayerState& right = rhs.players[player];
    return std::memcmp(left.hand, right.hand, sizeof(left.hand)) == 0
        && left.deck.size == right.deck.size
        && std::memcmp(left.deck.cards, right.deck.cards, left.deck.size * sizeof(Slot)) == 0;
}

void play_to_completion(GameState state, std::uint64_t seed) {
    const InvariantBaseline baseline = capture_baseline(state);
    RandomBot bots[MAX_PLAYERS] = {
        RandomBot(seed ^ 0xB070'0000ULL),
        RandomBot(seed ^ 0xB071'0000ULL),
        RandomBot(seed ^ 0xB072'0000ULL),
        RandomBot(seed ^ 0xB073'0000ULL),
    };

    for (int step = 0; step < 10000 && state.phase != static_cast<std::uint8_t>(Phase::Over); ++step) {
        const InvariantViolation violation = check_invariants(state, baseline);
        INFO(violation.message);
        REQUIRE(violation.ok);

        ActionMask legal{};
        const int legal_count = Game::legal_actions(state, legal);
        REQUIRE(legal_count > 0);
        const PlayerId player = Game::current_decision(state).player;
        REQUIRE(player < state.num_players);
        const Action action = bots[player].choose_action(state, legal, legal_count);
        (void)Game::step(state, action);
    }

    REQUIRE(state.phase == static_cast<std::uint8_t>(Phase::Over));
}

} // namespace

TEST_CASE("v2 determinize preserves public encoding and own hand", "[v2][determinize]") {
    GameState original = Game::new_game(determinize_setup(), 0xD373'0001ULL);
    seed_hidden_pool(original);
    const InvariantBaseline baseline = capture_baseline(original);

    GameState sampled = original;
    determinize(sampled, 0U, 0xD373'1001ULL);

    const InvariantViolation violation = check_invariants(sampled, baseline);
    INFO(violation.message);
    REQUIRE(violation.ok);
    CHECK(encoded(original, 0U) == encoded(sampled, 0U));
    CHECK(hand_equal(original, sampled, 0U));
    CHECK(original.players[1].deck.size == sampled.players[1].deck.size);
}

TEST_CASE("v2 determinize changes opponent hidden zones while preserving their multiset", "[v2][determinize]") {
    GameState original = Game::new_game(determinize_setup(), 0xD373'0002ULL);
    seed_hidden_pool(original);

    GameState sampled = original;
    bool saw_difference = false;
    for (std::uint64_t seed = 0; seed < 32U; ++seed) {
        sampled = original;
        determinize(sampled, 0U, 0xD373'2000ULL + seed);
        if (!hidden_zones_equal(original, sampled, 1U)) {
            saw_difference = true;
            break;
        }
    }

    REQUIRE(saw_difference);
    CHECK(original.players[1].deck.size == sampled.players[1].deck.size);
    CHECK(hidden_hand_size(original, 1U) == hidden_hand_size(sampled, 1U));
    CHECK(hidden_zone_multiset(original, 1U) == hidden_zone_multiset(sampled, 1U));
}

TEST_CASE("v2 determinize preserves v2 and v3 mover observations across mid-game states", "[v2][determinize]") {
    GameState state = Game::new_game(determinize_setup(), 0xD373'5001ULL);
    RandomBot bot(0xD373'6001ULL);
    int checked_states = 0;

    for (int step = 0; step < 28; ++step) {
        const PlayerId mover = Game::current_decision(state).player;
        REQUIRE(mover < state.num_players);
        const auto original_v2 = encoded_v2(state, mover);
        const auto original_v3 = encoded_v3(state, mover);
        for (std::uint64_t sample = 0; sample < 4U; ++sample) {
            GameState determinized = state;
            determinize(
                determinized,
                mover,
                0xD373'7000ULL + (static_cast<std::uint64_t>(step) * 4U) + sample);
            CHECK(encoded_v2(determinized, mover) == original_v2);
            CHECK(encoded_v3(determinized, mover) == original_v3);
        }
        ++checked_states;
        if (state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            break;
        }
        step_random(state, bot);
    }

    REQUIRE(checked_states >= 8);
}

TEST_CASE("v2 per-turn selfplay determinization seeds are stable within a turn", "[v2][determinize][selfplay]") {
    constexpr std::uint64_t GAME_SEED = 0xD373'8001ULL;
    constexpr PlayerId SEAT = 0U;
    constexpr std::uint16_t TURN = 11U;
    const std::uint64_t per_turn_first = selfplay_determinization_seed(
        GAME_SEED,
        SEAT,
        3U,
        TURN,
        SelfPlayDeterminizeMode::PerTurn);
    const std::uint64_t per_turn_repeat = selfplay_determinization_seed(
        GAME_SEED,
        SEAT,
        91U,
        TURN,
        SelfPlayDeterminizeMode::PerTurn);
    const std::uint64_t per_turn_next = selfplay_determinization_seed(
        GAME_SEED,
        SEAT,
        92U,
        static_cast<std::uint16_t>(TURN + 1U),
        SelfPlayDeterminizeMode::PerTurn);
    const std::uint64_t per_decision_next = selfplay_determinization_seed(
        GAME_SEED,
        SEAT,
        4U,
        TURN,
        SelfPlayDeterminizeMode::PerDecision);

    REQUIRE(per_turn_first == per_turn_repeat);
    REQUIRE(per_turn_first != per_turn_next);
    REQUIRE(per_turn_first != per_decision_next);

    GameState original = Game::new_game(determinize_setup(), GAME_SEED);
    seed_hidden_pool(original);
    GameState first = original;
    GameState repeat = original;
    GameState next_turn = original;
    determinize(first, SEAT, per_turn_first);
    determinize(repeat, SEAT, per_turn_repeat);
    determinize(next_turn, SEAT, per_turn_next);

    REQUIRE(std::memcmp(&first, &repeat, sizeof(GameState)) == 0);
    REQUIRE(hidden_zones_equal(first, repeat, 1U));
    REQUIRE_FALSE(hidden_zones_equal(first, next_turn, 1U));
}

TEST_CASE("v2 original and determinized states can finish", "[v2][determinize]") {
    GameState original = Game::new_game(determinize_setup(), 0xD373'0003ULL);
    seed_hidden_pool(original);

    GameState sampled = original;
    determinize(sampled, 0U, 0xD373'3003ULL);

    play_to_completion(original, 0xD373'4003ULL);
    play_to_completion(sampled, 0xD373'5003ULL);
}
