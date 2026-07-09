#include "v2/drivers/bots.h"

#include "v2/core/score.h"

#include <cstdint>

namespace {

struct BotController {
    BotKind kind = BotKind::BigMoney;
    RandomBot random{};
    BigMoneyBot big_money{};

    explicit BotController(BotSpec spec) noexcept
        : kind(spec.kind), random(spec.seed), big_money() {}

    [[nodiscard]] Action choose_action(
        const GameState& state,
        const ActionMask& legal,
        int legal_count) noexcept {
        if (kind == BotKind::Random) {
            return random.choose_action(state, legal, legal_count);
        }
        return big_money.choose_action(state, legal, legal_count);
    }
};

[[nodiscard]] Action first_legal(const ActionMask& legal) noexcept {
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (legal.test(action)) {
            return action;
        }
    }
    return A_PASS;
}

[[nodiscard]] Action first_legal_play_treasure(const ActionMask& legal) noexcept {
    constexpr DefId kTreasures[] = {
        DEF_PLATINUM,
        DEF_GOLD,
        DEF_SILVER,
        DEF_COPPER,
        DEF_POTION,
    };

    for (const DefId def : kTreasures) {
        const Action action = play_action(def);
        if (legal.test(action)) {
            return action;
        }
    }
    return A_PASS;
}

[[nodiscard]] Action first_legal_big_money_buy(const ActionMask& legal) noexcept {
    constexpr DefId kBuys[] = {
        DEF_PROVINCE,
        DEF_GOLD,
        DEF_SILVER,
    };

    for (const DefId def : kBuys) {
        const Action action = buy_action(def);
        if (legal.test(action)) {
            return action;
        }
    }
    return A_PASS;
}

[[nodiscard]] PlayerId winner_for(const GameState& state, const std::int16_t (&scores)[MAX_PLAYERS]) noexcept {
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

RandomBot::RandomBot(std::uint64_t seed) noexcept
    : rng(Xoshiro256pp::seeded(seed)) {}

Action RandomBot::choose_action(
    const GameState& state,
    const ActionMask& legal,
    int legal_count) noexcept {
    (void)state;
    if (legal_count <= 0) {
        return A_PASS;
    }

    std::uint32_t index = rng.uniform(static_cast<std::uint32_t>(legal_count));
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (!legal.test(action)) {
            continue;
        }
        if (index == 0U) {
            return action;
        }
        --index;
    }
    return A_PASS;
}

Action BigMoneyBot::choose_action(
    const GameState& state,
    const ActionMask& legal,
    int legal_count) const noexcept {
    (void)state;
    if (legal_count <= 0) {
        return A_PASS;
    }

    const Action treasure = first_legal_play_treasure(legal);
    if (treasure != A_PASS) {
        return treasure;
    }

    const Action buy = first_legal_big_money_buy(legal);
    if (buy != A_PASS) {
        return buy;
    }

    if (legal.test(A_PASS)) {
        return A_PASS;
    }
    return first_legal(legal);
}

GameResult run_game(
    const Setup& setup,
    std::uint64_t seed,
    BotSpec bot0,
    BotSpec bot1) noexcept {
    GameState state = Game::new_game(setup, seed);
    BotController bots[MAX_PLAYERS] = {
        BotController(bot0),
        BotController(bot1),
        BotController(bot1),
        BotController(bot1),
    };

    bool done = state.phase == static_cast<std::uint8_t>(Phase::Over);
    while (!done) {
        ActionMask legal{};
        const int legal_count = Game::legal_actions(state, legal);
        if (legal_count <= 0) {
            break;
        }

        const PlayerId player = Game::current_decision(state).player;
        const Action action = bots[player].choose_action(state, legal, legal_count);
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
