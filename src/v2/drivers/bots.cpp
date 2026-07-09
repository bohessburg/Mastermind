#include "v2/drivers/bots.h"

#include "v2/core/score.h"

#include <cstdint>
#include <optional>

namespace {

enum class EngineStrategy : std::uint8_t {
    PureBigMoney,
    BigMoneyPlusAction,
};

struct CardYield {
    std::int8_t actions = 0;
    std::int8_t cards = 0;
    std::int8_t coins = 0;
};

struct DeckProfile {
    std::uint8_t counts[BASIC_CARD_COUNT]{};
    std::uint16_t deck_size = 0;
    std::uint16_t total_actions = 0;
    std::uint16_t villages = 0;
    std::uint16_t terminal_draw = 0;
    std::uint16_t cantrips = 0;
    std::uint16_t chapels = 0;
    std::uint16_t sentries = 0;
    std::uint16_t junk = 0;
    std::int16_t total_money = 0;
    std::int16_t total_plus_actions = 0;
    std::int16_t total_plus_cards = 0;
    std::int16_t total_plus_coins = 0;
};

struct KingdomAnalysis {
    bool has_trasher = false;
    bool has_chapel = false;
    bool has_terminal_draw = false;
    bool has_cantrip_draw = false;
    bool has_good_action = false;
    bool has_militia_or_bandit = false;
    DefId best_action = NONE;
};

[[nodiscard]] Action first_legal(const ActionMask& legal) noexcept;

struct BotController {
    BotKind kind = BotKind::BigMoney;
    RandomBot random{};
    BigMoneyBot big_money{};
    HeuristicBot heuristic{};
    EngineBot engine{};
    std::optional<MctsBot> mcts{};

    explicit BotController(BotSpec spec) noexcept
        : kind(spec.kind), random(spec.seed), big_money(), heuristic(), engine(), mcts() {
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
        case BotKind::BigMoney:
        default:
            return big_money.choose_action(state, legal, legal_count);
        }
    }
};

[[nodiscard]] bool is_action_def(DefId def) noexcept {
    return def < card_def_count() && (card_def(def).types & TYPE_ACTION) != 0U;
}

[[nodiscard]] bool is_treasure_def(DefId def) noexcept {
    return def < card_def_count() && (card_def(def).types & TYPE_TREASURE) != 0U;
}

[[nodiscard]] bool is_victory_def(DefId def) noexcept {
    return def < card_def_count() && (card_def(def).types & TYPE_VICTORY) != 0U;
}

[[nodiscard]] DefId def_for_slot(const GameState& state, Slot slot) noexcept {
    return slot < state.num_slots ? state.slot_to_def[slot] : DEF_COPPER;
}

[[nodiscard]] int pile_count(const Pile& pile) noexcept {
    return pile.mixed_len > 0U ? static_cast<int>(pile.mixed_len) : static_cast<int>(pile.count);
}

[[nodiscard]] DefId pile_top_def(const GameState& state, const Pile& pile) noexcept {
    const Slot slot = pile.mixed_len > 0U ? pile.mixed[pile.mixed_len - 1U] : pile.base;
    return def_for_slot(state, slot);
}

[[nodiscard]] int supply_count(const GameState& state, DefId def) noexcept {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        if (pile_top_def(state, state.piles[i]) == def) {
            return pile_count(state.piles[i]);
        }
    }
    return 0;
}

[[nodiscard]] bool kingdom_has_available(const GameState& state, DefId def) noexcept {
    return supply_count(state, def) > 0;
}

void add_count(DeckProfile& profile, DefId def, std::uint8_t amount) noexcept {
    if (def >= BASIC_CARD_COUNT || amount == 0U) {
        return;
    }
    const std::uint16_t next = static_cast<std::uint16_t>(profile.counts[def] + amount);
    profile.counts[def] = next > 255U ? 255U : static_cast<std::uint8_t>(next);
    profile.deck_size = static_cast<std::uint16_t>(profile.deck_size + amount);
}

void count_ordered_zone(const GameState& state, const OrderedZone& zone, DeckProfile& profile) noexcept {
    for (std::uint8_t i = 0; i < zone.size; ++i) {
        add_count(profile, def_for_slot(state, zone.cards[i]), 1U);
    }
}

[[nodiscard]] CardYield card_yield(DefId def) noexcept {
    switch (def) {
    case DEF_VILLAGE:
        return CardYield{2, 1, 0};
    case DEF_FESTIVAL:
        return CardYield{2, 0, 2};
    case DEF_LABORATORY:
        return CardYield{1, 2, 0};
    case DEF_MARKET:
        return CardYield{1, 1, 1};
    case DEF_SENTRY:
    case DEF_POACHER:
        return CardYield{1, 1, 1};
    case DEF_MERCHANT:
    case DEF_HARBINGER:
        return CardYield{1, 1, 0};
    case DEF_CELLAR:
        return CardYield{1, 0, 0};
    case DEF_SMITHY:
        return CardYield{0, 3, 0};
    case DEF_WITCH:
    case DEF_MOAT:
    case DEF_LIBRARY:
        return CardYield{0, 2, 0};
    case DEF_COUNCIL_ROOM:
        return CardYield{0, 4, 0};
    case DEF_MILITIA:
    case DEF_VASSAL:
        return CardYield{0, 0, 2};
    case DEF_MONEYLENDER:
        return CardYield{0, 0, 3};
    default:
        return CardYield{};
    }
}

[[nodiscard]] bool is_terminal_draw(DefId def) noexcept {
    return def == DEF_SMITHY || def == DEF_COUNCIL_ROOM || def == DEF_WITCH
        || def == DEF_MOAT || def == DEF_LIBRARY;
}

[[nodiscard]] bool is_cantrip(DefId def) noexcept {
    return def == DEF_LABORATORY || def == DEF_MARKET || def == DEF_SENTRY
        || def == DEF_MERCHANT || def == DEF_HARBINGER || def == DEF_POACHER
        || def == DEF_CELLAR;
}

[[nodiscard]] bool is_village(DefId def) noexcept {
    return def == DEF_VILLAGE || def == DEF_FESTIVAL;
}

[[nodiscard]] bool is_junk(DefId def) noexcept {
    return def == DEF_COPPER || def == DEF_ESTATE || def == DEF_CURSE;
}

[[nodiscard]] DeckProfile analyze_deck(const GameState& state, PlayerId player) noexcept {
    DeckProfile profile{};
    if (player >= state.num_players) {
        return profile;
    }

    const PlayerState& ps = state.players[player];
    for (Slot slot = 0; slot < state.num_slots; ++slot) {
        if (ps.hand[slot] != 0U) {
            add_count(profile, def_for_slot(state, slot), ps.hand[slot]);
        }
        if (ps.exile[slot] != 0U) {
            add_count(profile, def_for_slot(state, slot), ps.exile[slot]);
        }
        if (ps.tavern[slot] != 0U) {
            add_count(profile, def_for_slot(state, slot), ps.tavern[slot]);
        }
        if (ps.island_mat[slot] != 0U) {
            add_count(profile, def_for_slot(state, slot), ps.island_mat[slot]);
        }
    }
    count_ordered_zone(state, ps.deck, profile);
    count_ordered_zone(state, ps.discard, profile);
    count_ordered_zone(state, ps.set_aside, profile);
    for (std::uint8_t i = 0; i < ps.in_play_size; ++i) {
        add_count(profile, def_for_slot(state, ps.in_play[i].slot), 1U);
    }

    for (DefId def = 0; def < BASIC_CARD_COUNT; ++def) {
        const std::uint8_t count = profile.counts[def];
        if (count == 0U) {
            continue;
        }
        const CardDef& card = card_def(def);
        if ((card.types & TYPE_ACTION) != 0U) {
            profile.total_actions = static_cast<std::uint16_t>(profile.total_actions + count);
            const CardYield yield = card_yield(def);
            profile.total_plus_actions = static_cast<std::int16_t>(
                profile.total_plus_actions + (yield.actions * count));
            profile.total_plus_cards = static_cast<std::int16_t>(
                profile.total_plus_cards + (yield.cards * count));
            profile.total_plus_coins = static_cast<std::int16_t>(
                profile.total_plus_coins + (yield.coins * count));
        }
        if ((card.types & TYPE_TREASURE) != 0U) {
            profile.total_money = static_cast<std::int16_t>(
                profile.total_money + (card.coin_value * count));
        }
        if (is_village(def)) {
            profile.villages = static_cast<std::uint16_t>(profile.villages + count);
        }
        if (is_terminal_draw(def)) {
            profile.terminal_draw = static_cast<std::uint16_t>(profile.terminal_draw + count);
        }
        if (is_cantrip(def)) {
            profile.cantrips = static_cast<std::uint16_t>(profile.cantrips + count);
        }
        if (def == DEF_CHAPEL) {
            profile.chapels = static_cast<std::uint16_t>(profile.chapels + count);
        }
        if (def == DEF_SENTRY) {
            profile.sentries = static_cast<std::uint16_t>(profile.sentries + count);
        }
        if (is_junk(def)) {
            profile.junk = static_cast<std::uint16_t>(profile.junk + count);
        }
    }
    return profile;
}

[[nodiscard]] int action_priority(DefId def) noexcept {
    switch (def) {
    case DEF_THRONE_ROOM:
    case DEF_VILLAGE:
        return 0;
    case DEF_FESTIVAL:
        return 1;
    case DEF_LABORATORY:
        return 10;
    case DEF_MARKET:
        return 11;
    case DEF_SENTRY:
        return 12;
    case DEF_POACHER:
        return 13;
    case DEF_MERCHANT:
        return 14;
    case DEF_HARBINGER:
        return 15;
    case DEF_CELLAR:
        return 16;
    case DEF_WITCH:
        return 30;
    case DEF_COUNCIL_ROOM:
        return 32;
    case DEF_SMITHY:
        return 33;
    case DEF_MILITIA:
        return 35;
    case DEF_LIBRARY:
        return 37;
    case DEF_MINE:
        return 38;
    case DEF_REMODEL:
        return 39;
    case DEF_MONEYLENDER:
        return 40;
    case DEF_CHAPEL:
        return 41;
    case DEF_ARTISAN:
        return 42;
    case DEF_BANDIT:
        return 43;
    case DEF_BUREAUCRAT:
        return 44;
    case DEF_WORKSHOP:
        return 45;
    case DEF_MOAT:
        return 46;
    case DEF_VASSAL:
        return 47;
    default:
        return 50;
    }
}

[[nodiscard]] int buy_priority(DefId def) noexcept {
    switch (def) {
    case DEF_PROVINCE:
        return 0;
    case DEF_GOLD:
        return 10;
    case DEF_ARTISAN:
        return 11;
    case DEF_WITCH:
        return 20;
    case DEF_LABORATORY:
        return 22;
    case DEF_FESTIVAL:
        return 23;
    case DEF_MARKET:
        return 24;
    case DEF_SENTRY:
        return 25;
    case DEF_COUNCIL_ROOM:
        return 26;
    case DEF_MINE:
        return 27;
    case DEF_LIBRARY:
        return 28;
    case DEF_DUCHY:
        return 29;
    case DEF_MILITIA:
        return 30;
    case DEF_SMITHY:
        return 31;
    case DEF_THRONE_ROOM:
        return 33;
    case DEF_REMODEL:
        return 34;
    case DEF_MONEYLENDER:
        return 35;
    case DEF_POACHER:
        return 36;
    case DEF_BUREAUCRAT:
        return 37;
    case DEF_GARDENS:
        return 38;
    case DEF_SILVER:
        return 40;
    case DEF_VILLAGE:
        return 41;
    case DEF_MERCHANT:
        return 44;
    case DEF_WORKSHOP:
        return 45;
    case DEF_HARBINGER:
        return 46;
    case DEF_VASSAL:
        return 47;
    case DEF_CHAPEL:
        return 50;
    case DEF_MOAT:
        return 53;
    case DEF_CELLAR:
        return 54;
    case DEF_ESTATE:
        return 55;
    case DEF_COPPER:
        return 900;
    case DEF_CURSE:
        return 999;
    default:
        return 100;
    }
}

[[nodiscard]] int discard_priority(DefId def) noexcept {
    if (def == DEF_CURSE) {
        return 0;
    }
    if (def == DEF_ESTATE) {
        return 1;
    }
    if (def == DEF_DUCHY) {
        return 2;
    }
    if (def == DEF_PROVINCE || def == DEF_COLONY) {
        return 3;
    }
    if (def == DEF_COPPER) {
        return 10;
    }
    if (is_victory_def(def)) {
        return 5;
    }
    if (is_action_def(def)) {
        return 15;
    }
    if (is_treasure_def(def)) {
        return 20;
    }
    return 25;
}

[[nodiscard]] int keep_priority(DefId def) noexcept {
    if (is_treasure_def(def)) {
        return 100 + card_def(def).coin_value;
    }
    if (is_action_def(def)) {
        return 60 - action_priority(def);
    }
    if (def == DEF_PROVINCE || def == DEF_COLONY) {
        return 30;
    }
    if (def == DEF_DUCHY) {
        return 20;
    }
    if (def == DEF_ESTATE || def == DEF_CURSE) {
        return 0;
    }
    return 10;
}

[[nodiscard]] int trash_priority(DefId def) noexcept {
    if (def == DEF_CURSE) {
        return 0;
    }
    if (def == DEF_ESTATE) {
        return 1;
    }
    if (def == DEF_COPPER) {
        return 2;
    }
    return 50;
}

[[nodiscard]] Action first_legal(const ActionMask& legal) noexcept {
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (legal.test(action)) {
            return action;
        }
    }
    return A_PASS;
}

[[nodiscard]] Action first_legal_option(const ActionMask& legal) noexcept {
    for (Action action = A_OPTION_BASE; action < A_CALL_BASE; ++action) {
        if (legal.test(action)) {
            return action;
        }
    }
    return first_legal(legal);
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

[[nodiscard]] Action legal_buy(const ActionMask& legal, DefId def) noexcept {
    const Action action = buy_action(def);
    return legal.test(action) ? action : A_PASS;
}

[[nodiscard]] Action first_legal_reaction(const ActionMask& legal) noexcept {
    const Action moat = select_action(DEF_MOAT);
    if (legal.test(moat)) {
        return moat;
    }
    return legal.test(A_PASS) ? A_PASS : first_legal(legal);
}

[[nodiscard]] Action best_legal_play_action(const ActionMask& legal, bool skip_chapel) noexcept {
    Action best = A_PASS;
    int best_priority = 999;
    for (DefId def = 0; def < ACTION_DEF_COUNT; ++def) {
        if (skip_chapel && def == DEF_CHAPEL) {
            continue;
        }
        const Action action = play_action(def);
        if (!legal.test(action) || !is_action_def(def)) {
            continue;
        }
        const int priority = action_priority(def);
        if (best == A_PASS || priority < best_priority) {
            best = action;
            best_priority = priority;
        }
    }
    return best;
}

[[nodiscard]] Action choose_select_min_priority(
    const ActionMask& legal,
    int (*priority)(DefId),
    bool allow_pass) noexcept {
    Action best = A_PASS;
    int best_priority = 9999;
    for (DefId def = 0; def < ACTION_DEF_COUNT; ++def) {
        const Action action = select_action(def);
        if (!legal.test(action)) {
            continue;
        }
        const int value = priority(def);
        if (best == A_PASS || value < best_priority) {
            best = action;
            best_priority = value;
        }
    }
    if (best != A_PASS && (!allow_pass || best_priority < 50)) {
        return best;
    }
    return allow_pass && legal.test(A_PASS) ? A_PASS : (best != A_PASS ? best : first_legal(legal));
}

[[nodiscard]] Action choose_select_max_priority(
    const ActionMask& legal,
    int (*priority)(DefId),
    bool allow_pass) noexcept {
    Action best = A_PASS;
    int best_priority = -9999;
    for (DefId def = 0; def < ACTION_DEF_COUNT; ++def) {
        const Action action = select_action(def);
        if (!legal.test(action)) {
            continue;
        }
        const int value = priority(def);
        if (best == A_PASS || value > best_priority) {
            best = action;
            best_priority = value;
        }
    }
    if (best != A_PASS) {
        return best;
    }
    return allow_pass && legal.test(A_PASS) ? A_PASS : first_legal(legal);
}

[[nodiscard]] Action choose_gain_most_expensive(const ActionMask& legal) noexcept {
    Action best = A_PASS;
    int best_cost = -1;
    int best_priority = 9999;
    for (DefId def = 0; def < ACTION_DEF_COUNT; ++def) {
        const Action action = select_action(def);
        if (!legal.test(action)) {
            continue;
        }
        const CardDef& card = card_def(def);
        const int cost = static_cast<int>(card.cost.coins)
            + (static_cast<int>(card.cost.potion) * 10)
            + (static_cast<int>(card.cost.debt) / 2);
        const int priority = buy_priority(def);
        if (best == A_PASS || cost > best_cost || (cost == best_cost && priority < best_priority)) {
            best = action;
            best_cost = cost;
            best_priority = priority;
        }
    }
    return best != A_PASS ? best : (legal.test(A_PASS) ? A_PASS : first_legal(legal));
}

[[nodiscard]] Action choose_smart_trash(
    const GameState& state,
    const ActionMask& legal,
    PlayerId player,
    bool aggressive_copper) noexcept {
    const DeckProfile profile = analyze_deck(state, player);
    Action best = A_PASS;
    int best_priority = 9999;
    for (DefId def = 0; def < ACTION_DEF_COUNT; ++def) {
        const Action action = select_action(def);
        if (!legal.test(action)) {
            continue;
        }
        int priority = trash_priority(def);
        if (def == DEF_COPPER && !aggressive_copper && profile.total_money <= 3) {
            priority = 60;
        }
        if (best == A_PASS || priority < best_priority) {
            best = action;
            best_priority = priority;
        }
    }
    if (best != A_PASS && (best_priority < 50 || !legal.test(A_PASS))) {
        return best;
    }
    return legal.test(A_PASS) ? A_PASS : (best != A_PASS ? best : first_legal(legal));
}

[[nodiscard]] DefId current_sentry_def(const GameState& state) noexcept {
    if (state.effect_depth > 0U) {
        const EffectFrame& frame = state.effect_stack[state.effect_depth - 1U];
        if (frame.source == DEF_SENTRY && frame.data[3] > 0 && frame.data[4] < frame.data[3]) {
            const std::uint8_t index = static_cast<std::uint8_t>(frame.data[4]);
            const Slot slot = static_cast<Slot>(frame.data[1U + index]);
            return def_for_slot(state, slot);
        }
    }
    const PlayerId player = state.decision.player;
    if (player < state.num_players && state.players[player].set_aside.size > 0U) {
        return def_for_slot(state, state.players[player].set_aside.cards[0]);
    }
    return DEF_COPPER;
}

[[nodiscard]] int sentry_order_priority(DefId def) noexcept {
    if (def == DEF_VASSAL) {
        return 0;
    }
    if (def == DEF_SENTRY) {
        return 1;
    }
    if (is_action_def(def)) {
        return 2;
    }
    if (is_treasure_def(def)) {
        return 10;
    }
    return 20;
}

[[nodiscard]] Action choose_sentry_option(const GameState& state, const ActionMask& legal) noexcept {
    const DefId def = current_sentry_def(state);
    Action desired = option_action(2U);
    if (def == DEF_CURSE || def == DEF_ESTATE || def == DEF_COPPER) {
        desired = option_action(0U);
    } else if (def == DEF_DUCHY || def == DEF_PROVINCE || def == DEF_COLONY) {
        desired = option_action(1U);
    }
    if (legal.test(desired)) {
        return desired;
    }
    return first_legal_option(legal);
}

[[nodiscard]] Action choose_order_for_set_aside(const GameState& state, const ActionMask& legal) noexcept {
    const PlayerId player = state.decision.player;
    if (player >= state.num_players || state.players[player].set_aside.size < 2U) {
        return first_legal_option(legal);
    }
    const DefId first = def_for_slot(state, state.players[player].set_aside.cards[0]);
    const DefId second = def_for_slot(state, state.players[player].set_aside.cards[1]);
    const Action desired = sentry_order_priority(first) <= sentry_order_priority(second)
        ? option_action(0U)
        : option_action(1U);
    return legal.test(desired) ? desired : first_legal_option(legal);
}

[[nodiscard]] Action big_money_buy(const GameState& state, const ActionMask& legal) noexcept {
    const int provinces_left = supply_count(state, DEF_PROVINCE);
    const PlayerId player = state.decision.player;
    const DeckProfile profile = analyze_deck(state, player);
    const int coins = state.coins;

    if (coins >= 8) {
        if (profile.counts[DEF_GOLD] == 0U && profile.counts[DEF_SILVER] < 5U) {
            const Action gold = legal_buy(legal, DEF_GOLD);
            if (gold != A_PASS) {
                return gold;
            }
        }
        const Action province = legal_buy(legal, DEF_PROVINCE);
        if (province != A_PASS) {
            return province;
        }
    }
    if (coins >= 6) {
        if (provinces_left <= 4) {
            const Action duchy = legal_buy(legal, DEF_DUCHY);
            if (duchy != A_PASS) {
                return duchy;
            }
        }
        const Action gold = legal_buy(legal, DEF_GOLD);
        if (gold != A_PASS) {
            return gold;
        }
    }
    if (coins == 5) {
        if (provinces_left <= 5) {
            const Action duchy = legal_buy(legal, DEF_DUCHY);
            if (duchy != A_PASS) {
                return duchy;
            }
        }
        const Action silver = legal_buy(legal, DEF_SILVER);
        if (silver != A_PASS) {
            return silver;
        }
    }
    if (coins >= 3) {
        if (provinces_left <= 2) {
            const Action estate = legal_buy(legal, DEF_ESTATE);
            if (estate != A_PASS) {
                return estate;
            }
        }
        const Action silver = legal_buy(legal, DEF_SILVER);
        if (silver != A_PASS) {
            return silver;
        }
    }
    if (coins == 2 && provinces_left <= 3) {
        const Action estate = legal_buy(legal, DEF_ESTATE);
        if (estate != A_PASS) {
            return estate;
        }
    }
    return legal.test(A_PASS) ? A_PASS : first_legal(legal);
}

[[nodiscard]] Action heuristic_buy(const GameState& state, const ActionMask& legal) noexcept {
    const int provinces_left = supply_count(state, DEF_PROVINCE);
    const DeckProfile profile = analyze_deck(state, state.decision.player);
    Action best = A_PASS;
    int best_cost = -1;
    int best_priority = 999999;

    for (DefId def = 0; def < ACTION_DEF_COUNT; ++def) {
        const Action action = buy_action(def);
        if (!legal.test(action)) {
            continue;
        }
        if (def == DEF_CURSE || def == DEF_COPPER) {
            continue;
        }
        if (def == DEF_ESTATE && provinces_left > 2) {
            continue;
        }
        if (def == DEF_DUCHY && provinces_left > 5) {
            continue;
        }

        const CardDef& card = card_def(def);
        const int cost = card.cost.coins;
        int priority = buy_priority(def);
        if (is_action_def(def)) {
            const int copies = profile.counts[def];
            priority *= (1 + copies) * (1 + copies);
        }
        if (best == A_PASS || cost > best_cost || (cost == best_cost && priority < best_priority)) {
            best = action;
            best_cost = cost;
            best_priority = priority;
        }
    }
    return best != A_PASS ? best : (legal.test(A_PASS) ? A_PASS : first_legal(legal));
}

[[nodiscard]] KingdomAnalysis analyze_kingdom(const GameState& state) noexcept {
    KingdomAnalysis analysis{};
    constexpr DefId kBestActions[] = {
        DEF_WITCH,
        DEF_BANDIT,
        DEF_MILITIA,
        DEF_LABORATORY,
        DEF_SMITHY,
        DEF_COUNCIL_ROOM,
        DEF_FESTIVAL,
        DEF_MARKET,
        DEF_LIBRARY,
        DEF_MOAT,
        DEF_MONEYLENDER,
    };

    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        if (pile_count(state.piles[i]) <= 0) {
            continue;
        }
        const DefId def = pile_top_def(state, state.piles[i]);
        if (is_village(def)) {
            analysis.has_cantrip_draw = true;
        }
        if (def == DEF_CHAPEL || def == DEF_SENTRY) {
            analysis.has_trasher = true;
        }
        if (def == DEF_CHAPEL) {
            analysis.has_chapel = true;
        }
        if (is_terminal_draw(def)) {
            analysis.has_terminal_draw = true;
        }
        if (def == DEF_LABORATORY || def == DEF_MARKET || def == DEF_FESTIVAL) {
            analysis.has_cantrip_draw = true;
        }
        if (def == DEF_MILITIA || def == DEF_BANDIT) {
            analysis.has_militia_or_bandit = true;
        }
    }

    for (const DefId def : kBestActions) {
        if (kingdom_has_available(state, def)) {
            analysis.has_good_action = true;
            analysis.best_action = def;
            break;
        }
    }
    return analysis;
}

[[nodiscard]] EngineStrategy pick_strategy(const KingdomAnalysis& analysis) noexcept {
    return (analysis.has_good_action || analysis.has_terminal_draw || analysis.has_cantrip_draw)
        ? EngineStrategy::BigMoneyPlusAction
        : EngineStrategy::PureBigMoney;
}

[[nodiscard]] Action engine_green_buy(const GameState& state, const ActionMask& legal) noexcept {
    const int provinces_left = supply_count(state, DEF_PROVINCE);
    const int coins = state.coins;
    if (coins >= 8) {
        const Action province = legal_buy(legal, DEF_PROVINCE);
        if (province != A_PASS) {
            return province;
        }
    }
    if (coins >= 6) {
        if (provinces_left <= 4) {
            const Action duchy = legal_buy(legal, DEF_DUCHY);
            if (duchy != A_PASS) {
                return duchy;
            }
        }
        const Action gold = legal_buy(legal, DEF_GOLD);
        if (gold != A_PASS) {
            return gold;
        }
    }
    if (coins == 5) {
        if (provinces_left <= 5) {
            const Action duchy = legal_buy(legal, DEF_DUCHY);
            if (duchy != A_PASS) {
                return duchy;
            }
        }
        constexpr DefId kGreenActions[] = {DEF_LABORATORY, DEF_MARKET, DEF_FESTIVAL};
        for (const DefId def : kGreenActions) {
            const Action action = legal_buy(legal, def);
            if (action != A_PASS) {
                return action;
            }
        }
        const Action silver = legal_buy(legal, DEF_SILVER);
        if (silver != A_PASS) {
            return silver;
        }
    }
    if (coins >= 3) {
        if (provinces_left <= 2) {
            const Action estate = legal_buy(legal, DEF_ESTATE);
            if (estate != A_PASS) {
                return estate;
            }
        }
        const Action silver = legal_buy(legal, DEF_SILVER);
        if (silver != A_PASS) {
            return silver;
        }
    }
    if (coins == 2 && provinces_left <= 3) {
        const Action estate = legal_buy(legal, DEF_ESTATE);
        if (estate != A_PASS) {
            return estate;
        }
    }
    return A_PASS;
}

[[nodiscard]] Action engine_buy(const GameState& state, const ActionMask& legal) noexcept {
    const int provinces_left = supply_count(state, DEF_PROVINCE);
    if (state.coins >= 8) {
        const Action province = legal_buy(legal, DEF_PROVINCE);
        if (province != A_PASS) {
            return province;
        }
    }
    if (state.coins >= 5 && provinces_left <= 4) {
        const Action duchy = legal_buy(legal, DEF_DUCHY);
        if (duchy != A_PASS) {
            return duchy;
        }
    }
    if (state.coins >= 2 && provinces_left <= 2) {
        const Action estate = legal_buy(legal, DEF_ESTATE);
        if (estate != A_PASS) {
            return estate;
        }
    }

    const KingdomAnalysis kingdom = analyze_kingdom(state);
    if (pick_strategy(kingdom) == EngineStrategy::PureBigMoney) {
        return big_money_buy(state, legal);
    }

    const PlayerId player = state.decision.player;
    const DeckProfile profile = analyze_deck(state, player);
    const double deck_size = profile.deck_size > 0U ? static_cast<double>(profile.deck_size) : 1.0;
    const double money_density = static_cast<double>(profile.total_money + profile.total_plus_coins) / deck_size;
    const int my_turns = static_cast<int>(state.turn_counter / state.num_players);
    const bool is_second = player == 1U;
    bool greening = money_density >= (is_second ? 1.1 : 1.2);
    if (my_turns > 5) {
        greening = true;
    }
    if (greening) {
        const Action green = engine_green_buy(state, legal);
        if (green != A_PASS) {
            return green;
        }
    }

    const bool has_trasher = profile.chapels > 0U || profile.sentries > 0U;
    if (kingdom.has_trasher && !has_trasher && kingdom.has_militia_or_bandit) {
        if (my_turns <= 2 && (state.coins == 2 || state.coins == 3)) {
            const Action chapel = legal_buy(legal, DEF_CHAPEL);
            if (chapel != A_PASS) {
                return chapel;
            }
        }
        const Action sentry = legal_buy(legal, DEF_SENTRY);
        if (sentry != A_PASS) {
            return sentry;
        }
    }

    int terminal_count = 0;
    int action_count = 0;
    for (DefId def = 0; def < BASIC_CARD_COUNT; ++def) {
        if (profile.counts[def] == 0U || !is_action_def(def)) {
            continue;
        }
        if (def == DEF_CHAPEL || def == DEF_SENTRY || def == DEF_THRONE_ROOM) {
            continue;
        }
        action_count += profile.counts[def];
        if (card_yield(def).actions == 0) {
            terminal_count += profile.counts[def];
        }
    }
    const double terminal_density = static_cast<double>(terminal_count) / deck_size;
    const double action_density = static_cast<double>(action_count) / deck_size;

    if (kingdom.best_action != NONE && kingdom.best_action != DEF_SENTRY) {
        bool under_limit = false;
        if (kingdom.best_action == DEF_WITCH) {
            under_limit = profile.counts[DEF_WITCH] < 2U;
        } else if (card_yield(kingdom.best_action).actions == 0) {
            under_limit = terminal_density < 0.08;
        } else {
            under_limit = action_density < 0.35;
        }
        if (under_limit) {
            const Action best = legal_buy(legal, kingdom.best_action);
            if (best != A_PASS) {
                return best;
            }
        }
    }

    if (state.coins >= 5 && action_density < 0.35) {
        constexpr DefId kCantripBuys[] = {DEF_LABORATORY, DEF_MARKET};
        for (const DefId def : kCantripBuys) {
            const Action action = legal_buy(legal, def);
            if (action != A_PASS) {
                return action;
            }
        }
    }
    const Action gold = legal_buy(legal, DEF_GOLD);
    if (state.coins >= 6 && gold != A_PASS) {
        return gold;
    }
    const Action silver = legal_buy(legal, DEF_SILVER);
    if (state.coins >= 3 && silver != A_PASS) {
        return silver;
    }
    return legal.test(A_PASS) ? A_PASS : first_legal(legal);
}

[[nodiscard]] Action choose_generic_subdecision(
    const GameState& state,
    const ActionMask& legal,
    bool engine_style) noexcept {
    const DecisionKind decision = static_cast<DecisionKind>(state.decision.kind);
    const DefId source = static_cast<DefId>(state.decision.source);
    const bool pass_allowed = legal.test(A_PASS) && state.decision.min_left == 0U;
    const PlayerId player = state.decision.player;

    if (decision == DecisionKind::ReactWindow) {
        return first_legal_reaction(legal);
    }
    if (decision == DecisionKind::OrderTriggers) {
        return first_legal_option(legal);
    }
    if (decision == DecisionKind::ChooseOrder) {
        return choose_order_for_set_aside(state, legal);
    }
    if (decision == DecisionKind::ChooseOption) {
        if (source == DEF_SENTRY) {
            return choose_sentry_option(state, legal);
        }
        if (source == DEF_LIBRARY) {
            const Action set_aside = option_action(1U);
            return engine_style && legal.test(set_aside) ? set_aside : first_legal_option(legal);
        }
        if (source == DEF_VASSAL) {
            const Action play = option_action(1U);
            return legal.test(play) ? play : first_legal_option(legal);
        }
        return first_legal_option(legal);
    }
    if (decision == DecisionKind::ChooseGain) {
        return choose_gain_most_expensive(legal);
    }
    if (decision == DecisionKind::Choose) {
        if (source == DEF_MILITIA) {
            return choose_select_max_priority(legal, keep_priority, false);
        }
        if (source == DEF_CELLAR || source == DEF_POACHER) {
            return choose_select_min_priority(legal, discard_priority, pass_allowed);
        }
        if (source == DEF_CHAPEL || source == DEF_REMODEL || source == DEF_MINE
            || source == DEF_MONEYLENDER || source == DEF_BANDIT) {
            (void)engine_style;
            return choose_smart_trash(state, legal, player, false);
        }
        if (source == DEF_THRONE_ROOM) {
            return choose_select_min_priority(legal, action_priority, pass_allowed);
        }
        if (source == DEF_BUREAUCRAT || source == DEF_ARTISAN) {
            return choose_select_min_priority(legal, discard_priority, pass_allowed);
        }
        if (source == DEF_HARBINGER) {
            return choose_select_max_priority(legal, keep_priority, pass_allowed);
        }
    }

    if (legal.test(A_PASS) && state.decision.min_left == 0U) {
        return A_PASS;
    }
    return first_legal(legal);
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
    return legal.nth_set(index);
}

Action BigMoneyBot::choose_action(
    const GameState& state,
    const ActionMask& legal,
    int legal_count) const noexcept {
    if (legal_count <= 0) {
        return A_PASS;
    }

    const DecisionKind decision = static_cast<DecisionKind>(state.decision.kind);
    if (decision == DecisionKind::PhaseBuy) {
        const Action treasure = first_legal_play_treasure(legal);
        if (treasure != A_PASS) {
            return treasure;
        }
        return big_money_buy(state, legal);
    }
    if (decision == DecisionKind::PhaseAction || decision == DecisionKind::PhaseNight) {
        return legal.test(A_PASS) ? A_PASS : first_legal(legal);
    }
    if (decision == DecisionKind::ReactWindow) {
        return first_legal_reaction(legal);
    }
    if (decision == DecisionKind::Choose && state.decision.source == DEF_MILITIA) {
        return choose_select_max_priority(legal, keep_priority, false);
    }
    if (decision == DecisionKind::Choose && state.decision.source == DEF_BUREAUCRAT) {
        return choose_select_min_priority(legal, discard_priority, false);
    }
    if (legal.test(A_PASS) && state.decision.min_left == 0U) {
        return A_PASS;
    }
    return first_legal(legal);
}

Action HeuristicBot::choose_action(
    const GameState& state,
    const ActionMask& legal,
    int legal_count) const noexcept {
    if (legal_count <= 0) {
        return A_PASS;
    }

    const DecisionKind decision = static_cast<DecisionKind>(state.decision.kind);
    if (decision == DecisionKind::PhaseAction) {
        const Action action = best_legal_play_action(legal, false);
        return action != A_PASS ? action : (legal.test(A_PASS) ? A_PASS : first_legal(legal));
    }
    if (decision == DecisionKind::PhaseBuy) {
        const Action treasure = first_legal_play_treasure(legal);
        if (treasure != A_PASS) {
            return treasure;
        }
        return heuristic_buy(state, legal);
    }
    if (decision == DecisionKind::PhaseNight) {
        return legal.test(A_PASS) ? A_PASS : first_legal(legal);
    }
    return choose_generic_subdecision(state, legal, false);
}

Action EngineBot::choose_action(
    const GameState& state,
    const ActionMask& legal,
    int legal_count) noexcept {
    if (legal_count <= 0) {
        return A_PASS;
    }

    const DecisionKind decision = static_cast<DecisionKind>(state.decision.kind);
    if (decision == DecisionKind::PhaseAction) {
        const KingdomAnalysis kingdom = analyze_kingdom(state);
        if (pick_strategy(kingdom) == EngineStrategy::PureBigMoney) {
            return legal.test(A_PASS) ? A_PASS : first_legal(legal);
        }

        bool has_other_terminal = false;
        const PlayerId player = state.decision.player;
        if (player < MAX_PLAYERS && chapel_plays[player] >= 1U) {
            for (DefId def = 0; def < ACTION_DEF_COUNT; ++def) {
                const Action action = play_action(def);
                if (def != DEF_CHAPEL && legal.test(action) && card_yield(def).actions == 0) {
                    has_other_terminal = true;
                    break;
                }
            }
        }

        const Action action = best_legal_play_action(legal, has_other_terminal);
        if (action != A_PASS) {
            const DefId def = action_def(action, A_PLAY_BASE);
            if (def == DEF_CHAPEL && player < MAX_PLAYERS && chapel_plays[player] < 255U) {
                ++chapel_plays[player];
            }
            return action;
        }
        return legal.test(A_PASS) ? A_PASS : first_legal(legal);
    }
    if (decision == DecisionKind::PhaseBuy) {
        const Action treasure = first_legal_play_treasure(legal);
        if (treasure != A_PASS) {
            return treasure;
        }
        return engine_buy(state, legal);
    }
    if (decision == DecisionKind::PhaseNight) {
        return legal.test(A_PASS) ? A_PASS : first_legal(legal);
    }
    return choose_generic_subdecision(state, legal, true);
}

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
