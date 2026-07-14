#include "v2/drivers/bots.h"
#include "v2/mcts/eval.h"

#include <catch2/catch_test_macros.hpp>

namespace {

[[nodiscard]] Setup all_kingdom_setup() noexcept {
    Setup setup{};
    setup.num_players = 2;
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
        DEF_LIBRARY,
        DEF_SENTRY,
    };
    setup.kingdom_count = static_cast<std::uint8_t>(sizeof(kKingdom) / sizeof(kKingdom[0]));
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        setup.kingdom[i] = kKingdom[i];
    }
    return setup;
}

void require_bot_picks_legal(const GameState& state, const ActionMask& legal, int legal_count) {
    RandomBot random{0xB075U};
    BigMoneyBot big_money{};
    HeuristicBot heuristic{};
    EngineBot engine{};
    EngineBotV3 engine_v3{};

    const Action random_action = random.choose_action(state, legal, legal_count);
    const Action big_money_action = big_money.choose_action(state, legal, legal_count);
    const Action heuristic_action = heuristic.choose_action(state, legal, legal_count);
    const Action engine_action = engine.choose_action(state, legal, legal_count);
    const Action engine_v3_action = engine_v3.choose_action(state, legal, legal_count);

    REQUIRE(legal.test(random_action));
    REQUIRE(legal.test(big_money_action));
    REQUIRE(legal.test(heuristic_action));
    REQUIRE(legal.test(engine_action));
    REQUIRE(legal.test(engine_v3_action));
}

void add_options(ActionMask& mask, std::uint8_t count) noexcept {
    for (std::uint8_t option = 0; option < count; ++option) {
        mask.set(option_action(option));
    }
}

[[nodiscard]] MatchupResult random_kingdom_matchup(
    BotSpec bot_a,
    BotSpec bot_b,
    std::uint16_t games,
    std::uint64_t seed) noexcept {
    MatchupResult result{};
    result.games = games;
    for (std::uint16_t game = 0; game < games; ++game) {
        const std::uint64_t pair = static_cast<std::uint64_t>(game / 2U);
        const Setup setup = random_mcts_eval_setup(seed ^ (0xD1B5'4A32'D192'ED03ULL * (pair + 1U)));
        const bool swapped = (game & 1U) != 0U;
        const std::uint64_t game_seed = seed + (0x9E37'79B9'7F4A'7C15ULL * (pair + 1U));
        const GameResult one = run_game(
            setup,
            game_seed,
            swapped ? bot_b : bot_a,
            swapped ? bot_a : bot_b);
        if (one.truncated) {
            ++result.truncated;
        }
        if (one.winner == NONE) {
            ++result.ties;
        } else {
            const bool a_won = swapped ? one.winner == 1U : one.winner == 0U;
            if (a_won) {
                ++result.wins_a;
            } else {
                ++result.wins_b;
            }
        }
    }
    return result;
}

} // namespace

TEST_CASE("v2 bot matchup eval is deterministic and seat-swapped", "[v2][bots]") {
    const MatchupResult first = eval_matchup(
        BotSpec{BotKind::Engine, 0x1000U},
        BotSpec{BotKind::BigMoney, 0x2000U},
        80,
        0x5151'0001U);
    const MatchupResult second = eval_matchup(
        BotSpec{BotKind::Engine, 0x1000U},
        BotSpec{BotKind::BigMoney, 0x2000U},
        80,
        0x5151'0001U);

    REQUIRE(first.games == 80U);
    REQUIRE(first.games == second.games);
    REQUIRE(first.wins_a == second.wins_a);
    REQUIRE(first.wins_b == second.wins_b);
    REQUIRE(first.ties == second.ties);
    REQUIRE(first.truncated == second.truncated);
}

TEST_CASE("v2 EngineBot beats RandomBot in eval smoke", "[v2][bots]") {
    const MatchupResult engine_random = eval_matchup(
        BotSpec{BotKind::Engine, 0x3000U},
        BotSpec{BotKind::Random, 0x4000U},
        200,
        0x5151'0002U);
    INFO("Engine wins=" << engine_random.wins_a
        << " Random wins=" << engine_random.wins_b
        << " ties=" << engine_random.ties
        << " truncated=" << engine_random.truncated);
    REQUIRE(engine_random.truncated == 0U);
    REQUIRE(engine_random.win_rate_a() > 0.90);
}

TEST_CASE("v2 EngineBotV3 beats RandomBot in eval smoke", "[v2][bots]") {
    const MatchupResult engine_random = eval_matchup(
        BotSpec{BotKind::EngineV3, 0x3001U},
        BotSpec{BotKind::Random, 0x4001U},
        200,
        0x5151'0004U);
    INFO("EngineV3 wins=" << engine_random.wins_a
        << " Random wins=" << engine_random.wins_b
        << " ties=" << engine_random.ties
        << " truncated=" << engine_random.truncated);
    REQUIRE(engine_random.truncated == 0U);
    REQUIRE(engine_random.win_rate_a() > 0.90);
}

TEST_CASE("v2 EngineBotV3 holds its gate vs EngineBot on random kingdoms", "[v2][bots]") {
    const MatchupResult result = random_kingdom_matchup(
        BotSpec{BotKind::EngineV3, 0x5100U},
        BotSpec{BotKind::Engine, 0x5200U},
        1000U,
        0x5151'0005U);
    INFO("EngineV3 wins=" << result.wins_a
        << " Engine wins=" << result.wins_b
        << " ties=" << result.ties
        << " truncated=" << result.truncated);
    REQUIRE(result.games == 1000U);
    REQUIRE(result.truncated == 0U);
    REQUIRE(result.wins_a + result.wins_b + result.ties == result.games);
    // Measured 2026-07-13: 50.65% seat-adjusted over 10k games (v3 ahead of
    // v2 in decided games). Gate well below to absorb seed variance while
    // catching real regressions toward the pre-tuning 31-45% range.
    REQUIRE(result.wins_a >= 440U);
    REQUIRE(result.wins_a + 40U >= result.wins_b);
}

TEST_CASE("v2 EngineBotV3 beats BigMoney on random kingdoms", "[v2][bots]") {
    const MatchupResult result = random_kingdom_matchup(
        BotSpec{BotKind::EngineV3, 0x5300U},
        BotSpec{BotKind::BigMoney, 0x5400U},
        1000U,
        0x5151'0006U);
    INFO("EngineV3 wins=" << result.wins_a
        << " BigMoney wins=" << result.wins_b
        << " ties=" << result.ties
        << " truncated=" << result.truncated);
    REQUIRE(result.truncated == 0U);
    // Measured 2026-07-13: 75.4% seat-adjusted over 10k games.
    REQUIRE(result.win_rate_a() > 0.70);
}

TEST_CASE("v2 bot policies return legal actions for every decision kind", "[v2][bots]") {
    GameState state = Game::new_game(all_kingdom_setup(), 0x5151'0003U);
    ActionMask legal{};

    state.decision = PendingDecision{0, static_cast<std::uint8_t>(DecisionKind::PhaseAction), 0, 0, 0};
    legal.reset();
    legal.set(A_PASS);
    legal.set(play_action(DEF_VILLAGE));
    require_bot_picks_legal(state, legal, 2);

    state.decision = PendingDecision{0, static_cast<std::uint8_t>(DecisionKind::PhaseBuy), 0, 0, 0};
    legal.reset();
    legal.set(A_PASS);
    legal.set(play_action(DEF_COPPER));
    legal.set(buy_action(DEF_SILVER));
    require_bot_picks_legal(state, legal, 3);

    state.decision = PendingDecision{0, static_cast<std::uint8_t>(DecisionKind::PhaseNight), 0, 0, 0};
    legal.reset();
    legal.set(A_PASS);
    require_bot_picks_legal(state, legal, 1);

    state.decision = PendingDecision{1, static_cast<std::uint8_t>(DecisionKind::Choose), DEF_MILITIA, 1, 1};
    legal.reset();
    legal.set(select_action(DEF_COPPER));
    legal.set(select_action(DEF_SILVER));
    require_bot_picks_legal(state, legal, 2);

    state.decision = PendingDecision{0, static_cast<std::uint8_t>(DecisionKind::ChooseGain), DEF_WORKSHOP, 0, 1};
    legal.reset();
    legal.set(A_PASS);
    legal.set(select_action(DEF_ESTATE));
    legal.set(select_action(DEF_SILVER));
    require_bot_picks_legal(state, legal, 3);

    state.effect_depth = 1;
    state.effect_stack[0] = EffectFrame{};
    state.effect_stack[0].source = DEF_SENTRY;
    state.effect_stack[0].player = 0;
    state.effect_stack[0].data[1] = slot_of(state, DEF_COPPER);
    state.effect_stack[0].data[3] = 1;
    state.effect_stack[0].data[4] = 0;
    state.decision = PendingDecision{0, static_cast<std::uint8_t>(DecisionKind::ChooseOption), DEF_SENTRY, 1, 3};
    legal.reset();
    add_options(legal, 3);
    require_bot_picks_legal(state, legal, 3);

    state.effect_depth = 0;
    state.players[0].set_aside = OrderedZone{};
    state.players[0].set_aside.cards[0] = slot_of(state, DEF_VASSAL);
    state.players[0].set_aside.cards[1] = slot_of(state, DEF_SILVER);
    state.players[0].set_aside.size = 2;
    state.decision = PendingDecision{0, static_cast<std::uint8_t>(DecisionKind::ChooseOrder), DEF_SENTRY, 1, 2};
    legal.reset();
    add_options(legal, 2);
    require_bot_picks_legal(state, legal, 2);

    state.decision = PendingDecision{1, static_cast<std::uint8_t>(DecisionKind::ReactWindow), DEF_MILITIA, 0, 1};
    legal.reset();
    legal.set(A_PASS);
    legal.set(select_action(DEF_MOAT));
    require_bot_picks_legal(state, legal, 2);

    state.decision = PendingDecision{0, static_cast<std::uint8_t>(DecisionKind::OrderTriggers), DEF_MERCHANT, 1, 2};
    legal.reset();
    add_options(legal, 2);
    require_bot_picks_legal(state, legal, 2);
}

TEST_CASE("v2 bots complete all-card kingdom games without stalls", "[v2][bots]") {
    const Setup setup = all_kingdom_setup();
    constexpr BotKind kBots[] = {
        BotKind::Random,
        BotKind::BigMoney,
        BotKind::Heuristic,
        BotKind::Engine,
        BotKind::EngineV3,
    };

    for (const BotKind kind : kBots) {
        for (std::uint64_t seed = 0; seed < 25U; ++seed) {
            const GameResult result = run_game(
                setup,
                0x5151'1000U + seed,
                BotSpec{kind, 0x5151'2000U + seed},
                BotSpec{BotKind::Engine, 0x5151'3000U + seed});
            INFO("kind=" << static_cast<int>(kind)
                << " seed=" << seed
                << " turns=" << result.turns
                << " scores=" << result.scores[0] << "," << result.scores[1]
                << " winner=" << static_cast<int>(result.winner));
            REQUIRE(result.truncated == false);
            REQUIRE(result.turns > 0U);
            const bool winner_valid = result.winner == NONE || result.winner < 2U;
            REQUIRE(winner_valid);
        }
    }
}
