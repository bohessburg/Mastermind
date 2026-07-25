#include "v2/drivers/bots.h"

#include "v2/core/score.h"

#include <cstdint>
#include <optional>

namespace {

[[nodiscard]] Action first_legal(const ActionMask& legal) noexcept {
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (legal.test(action)) {
            return action;
        }
    }
    return A_PASS;
}

[[nodiscard]] Action first_legal_play_treasure(const ActionMask& legal) noexcept {
    constexpr DefId TREASURES[] = {DEF_PLATINUM, DEF_GOLD, DEF_SILVER, DEF_COPPER, DEF_POTION};
    for (const DefId def : TREASURES) {
        const Action action = play_action(def);
        if (legal.test(action)) {
            return action;
        }
    }
    return A_PASS;
}

struct BotController {
    BotKind kind = BotKind::BigMoney;
    RandomBot random{};
    BigMoneyBot big_money{};
    HeuristicBot heuristic{};
    EngineBot engine{};
    EngineBotV3 engine_v3{};
    ThinnerBot thinner{};
    std::optional<MctsBot> mcts{};

    explicit BotController(BotSpec spec) noexcept
        : kind(spec.kind), random(spec.seed), big_money(), heuristic(), engine(), engine_v3(), thinner(), mcts() {
        if (kind == BotKind::Mcts) {
            MctsConfig config = spec.mcts_config;
            config.rollout_seed ^= (spec.seed * 0x9E37'79B9'7F4A'7C15ULL);
            mcts.emplace(config);
        }
    }

    [[nodiscard]] Action choose_action(
        const GameState& state,
        const ActionMask& legal,
        int legal_count) noexcept {
        switch (kind) {
        case BotKind::Mcts:
            return mcts.has_value()
                ? mcts->choose_action(state, legal, legal_count)
                : first_legal(legal);
        case BotKind::Random:
            return random.choose_action(state, legal, legal_count);
        case BotKind::Heuristic:
            return heuristic.choose_action(state, legal, legal_count);
        case BotKind::Engine:
            return engine.choose_action(state, legal, legal_count);
        case BotKind::EngineV3:
            return engine_v3.choose_action(state, legal, legal_count);
        case BotKind::Thinner:
            return thinner.choose_action(state, legal, legal_count);
        case BotKind::BigMoney:
        default:
            return big_money.choose_action(state, legal, legal_count);
        }
    }
};

[[nodiscard]] PlayerId winner_for(
    const GameState& state,
    const std::int16_t (&scores)[MAX_PLAYERS]) noexcept {
    PlayerId winner = 0;
    bool tied = false;
    for (PlayerId player = 1; player < state.num_players; ++player) {
        if (scores[player] > scores[winner]) {
            winner = player;
            tied = false;
        } else if (scores[player] == scores[winner]) {
            tied = true;
        }
    }
    return tied ? NONE : winner;
}

} // namespace

MctsBot::MctsBot(const MctsConfig& cfg)
    : config(cfg), search(cfg) {}

Action MctsBot::choose_action(
    const GameState& state,
    const ActionMask& legal,
    int legal_count) noexcept {
    if (legal_count <= 0) {
        return A_PASS;
    }
    if (legal_count == 1) {
        return legal.nth_set(0U);
    }

    const DecisionKind decision = static_cast<DecisionKind>(state.decision.kind);
    if (decision == DecisionKind::PhaseBuy) {
        const Action treasure = first_legal_play_treasure(legal);
        if (treasure != A_PASS) {
            return treasure;
        }
    }

    ++searches;
    sims += config.sims_per_move;
    return search.choose(state, state.decision.player);
}

double MatchupResult::win_rate_a() const noexcept {
    return games == 0U ? 0.0 : static_cast<double>(wins_a) / static_cast<double>(games);
}

double MatchupResult::win_rate_b() const noexcept {
    return games == 0U ? 0.0 : static_cast<double>(wins_b) / static_cast<double>(games);
}

GameResult run_game(
    const Setup& setup,
    std::uint64_t seed,
    BotSpec bot0,
    BotSpec bot1) noexcept {
    GameState state = Game::new_game(setup, seed);
    BotController first_bot(bot0);
    BotController other_bot(bot1);

    bool done = state.phase == static_cast<std::uint8_t>(Phase::Over);
    ActionMask legal{};
    while (!done) {
        const int legal_count = Game::legal_actions(state, legal);
        if (legal_count <= 0) {
            break;
        }

        const PlayerId player = Game::current_decision(state).player;
        BotController& bot = player == 0U ? first_bot : other_bot;
        const Action action = bot.choose_action(state, legal, legal_count);
        if (!legal.test(action)) {
            break;
        }
        done = Game::step(state, action);
    }

    GameResult result{};
    result.turns = state.turn_counter;
    result.truncated = state.truncated != 0U;
    for (PlayerId player = 0; player < state.num_players; ++player) {
        result.scores[player] = score(state, player);
    }
    result.winner = winner_for(state, result.scores);
    return result;
}

MatchupResult eval_matchup(
    const Setup& setup,
    BotSpec bot_a,
    BotSpec bot_b,
    std::uint16_t n_games,
    std::uint64_t seed) noexcept {
    MatchupResult result{};
    result.games = n_games;
    for (std::uint16_t game = 0; game < n_games; ++game) {
        const bool swapped = (game & 1U) != 0U;
        BotSpec first = swapped ? bot_b : bot_a;
        BotSpec second = swapped ? bot_a : bot_b;
        first.seed = static_cast<std::uint64_t>(first.seed + seed + (game * 17U));
        second.seed = static_cast<std::uint64_t>(second.seed + seed + (game * 31U) + 1U);
        const GameResult game_result = run_game(
            setup,
            seed + (static_cast<std::uint64_t>(game) * 0x9E37'79B9U),
            first,
            second);
        if (game_result.truncated) {
            ++result.truncated;
        }
        if (game_result.winner == NONE) {
            ++result.ties;
        } else {
            const bool winner_is_a = swapped ? game_result.winner == 1U : game_result.winner == 0U;
            if (winner_is_a) {
                ++result.wins_a;
            } else {
                ++result.wins_b;
            }
        }
    }
    return result;
}

MatchupResult eval_matchup(
    BotSpec bot_a,
    BotSpec bot_b,
    std::uint16_t n_games,
    std::uint64_t seed) noexcept {
    return eval_matchup(Setup{}, bot_a, bot_b, n_games, seed);
}
