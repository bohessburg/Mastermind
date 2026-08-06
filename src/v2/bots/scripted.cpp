#include "v2/bots/scripted.h"

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

enum class EngineV3Strategy : std::uint8_t {
    PureBigMoney,
    BigMoneyPlusAction,
    Engine,
    Rush,
    ThinEngine,
    FatHybrid,
};

struct EngineV3Kingdom {
    bool village_ok = false;
    bool village_card_ok = false;
    bool draw_terminal = false;
    bool lab_ok = false;
    bool chapel_ok = false;
    bool witch_ok = false;
    bool militia_ok = false;
    bool bandit_ok = false;
    bool moat_ok = false;
    bool market_ok = false;
    bool festival_ok = false;
    bool council_room_ok = false;
    bool plus_buy = false;
    bool sentry_ok = false;
    bool bmx_ok = false;
    bool gardens_ok = false;
    bool workshop_ok = false;
};

struct V3ActionCounts {
    int terminals = 0;
    int actions = 0;
};

[[nodiscard]] EngineV3Kingdom analyze_v3_kingdom(const GameState& state) noexcept {
    EngineV3Kingdom kingdom{};
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        const DefId def = pile_top_def(state, state.piles[i]);
        switch (def) {
        case DEF_VILLAGE:
            kingdom.village_card_ok = true;
            [[fallthrough]];
        case DEF_FESTIVAL:
            kingdom.village_ok = true;
            break;
        case DEF_SMITHY:
        case DEF_COUNCIL_ROOM:
        case DEF_WITCH:
        case DEF_MOAT:
        case DEF_LIBRARY:
            kingdom.draw_terminal = true;
            break;
        case DEF_LABORATORY:
            kingdom.lab_ok = true;
            break;
        case DEF_CHAPEL:
            kingdom.chapel_ok = true;
            break;
        case DEF_MILITIA:
            kingdom.militia_ok = true;
            break;
        case DEF_BANDIT:
            kingdom.bandit_ok = true;
            break;
        case DEF_SENTRY:
            kingdom.sentry_ok = true;
            break;
        case DEF_MARKET:
            kingdom.market_ok = true;
            break;
        case DEF_GARDENS:
            kingdom.gardens_ok = true;
            break;
        case DEF_WORKSHOP:
            kingdom.workshop_ok = true;
            break;
        default:
            break;
        }
        if (def == DEF_FESTIVAL) {
            kingdom.festival_ok = true;
        }
        if (def == DEF_COUNCIL_ROOM) {
            kingdom.council_room_ok = true;
        }
        if (def == DEF_WITCH || def == DEF_BANDIT || def == DEF_MILITIA
            || def == DEF_LABORATORY || def == DEF_SMITHY || def == DEF_COUNCIL_ROOM
            || def == DEF_FESTIVAL || def == DEF_MARKET || def == DEF_LIBRARY
            || def == DEF_MOAT || def == DEF_MONEYLENDER) {
            kingdom.bmx_ok = true;
        }
        if (def == DEF_WITCH) {
            kingdom.witch_ok = true;
        }
        if (def == DEF_MOAT) {
            kingdom.moat_ok = true;
        }
        if (def == DEF_MARKET || def == DEF_FESTIVAL || def == DEF_COUNCIL_ROOM) {
            kingdom.plus_buy = true;
        }
    }
    return kingdom;
}

[[nodiscard]] EngineV3Strategy pick_v3_strategy(const EngineV3Kingdom& kingdom) noexcept {
    if (kingdom.gardens_ok && kingdom.workshop_ok && !kingdom.bmx_ok) {
        return EngineV3Strategy::Rush;
    }
    if (kingdom.chapel_ok && kingdom.village_card_ok && kingdom.witch_ok
        && (kingdom.draw_terminal || kingdom.lab_ok)) {
        return EngineV3Strategy::Engine;
    }
    const bool real_trasher = kingdom.chapel_ok || kingdom.sentry_ok;
    const bool draw_ok = kingdom.draw_terminal || kingdom.lab_ok;
    const bool quiet_engine = !kingdom.witch_ok
        && !kingdom.militia_ok && !kingdom.bandit_ok;
    if (kingdom.militia_ok && kingdom.lab_ok && !kingdom.village_ok) {
        return EngineV3Strategy::BigMoneyPlusAction;
    }
    if (quiet_engine && kingdom.plus_buy
        && real_trasher && kingdom.village_ok && draw_ok) {
        return EngineV3Strategy::ThinEngine;
    }
    if (quiet_engine && kingdom.plus_buy && ((kingdom.village_ok && draw_ok)
        || (kingdom.lab_ok && !kingdom.village_ok)
        || (kingdom.sentry_ok && kingdom.draw_terminal && !kingdom.village_ok))) {
        return EngineV3Strategy::FatHybrid;
    }
    if (kingdom.bmx_ok) {
        return EngineV3Strategy::BigMoneyPlusAction;
    }
    return EngineV3Strategy::PureBigMoney;
}

[[nodiscard]] bool v3_is_engine_strategy(EngineV3Strategy strategy) noexcept {
    return strategy == EngineV3Strategy::Engine
        || strategy == EngineV3Strategy::ThinEngine
        || strategy == EngineV3Strategy::FatHybrid;
}

[[nodiscard]] int v3_my_turns(const GameState& state) noexcept {
    if (state.num_players == 0U) {
        return 0;
    }
    return static_cast<int>(state.turn_counter / state.num_players);
}

[[nodiscard]] int v3_empty_supply_piles(const GameState& state) noexcept {
    int empty = 0;
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        if (pile_count(state.piles[i]) == 0) {
            ++empty;
        }
    }
    return empty;
}

[[nodiscard]] PlayerId v3_opponent(const GameState& state, PlayerId player) noexcept {
    if (state.num_players < 2U) {
        return player;
    }
    return player == 0U ? 1U : 0U;
}

[[nodiscard]] bool v3_province_allowed(const GameState& state) noexcept {
    const int provinces_left = supply_count(state, DEF_PROVINCE);
    if (state.coins < 8 || provinces_left > 2) {
        return true;
    }

    const PlayerId player = state.decision.player;
    const int my_score = score(state, player);
    const int opponent_score = score(state, v3_opponent(state, player));
    if (provinces_left == 1) {
        // Taking the last province ends the game: take a win or a tie; when
        // losing anyway, keep the game alive and dance instead.
        return my_score + 6 > opponent_score;
    }
    // Penultimate province: racing beats dancing against opponents that
    // reliably reach $8, so gate nothing here.
    return true;
}

[[nodiscard]] bool v3_avoid_last_pile_buy(const GameState& state, DefId def) noexcept {
    if (v3_empty_supply_piles(state) != 2) {
        return false;
    }
    if (supply_count(state, def) > 1) {
        return false;
    }
    const PlayerId player = state.decision.player;
    const int my_after = score(state, player)
        + (def < card_def_count() ? static_cast<int>(card_def(def).vp) : 0);
    return my_after <= score(state, v3_opponent(state, player));
}

[[nodiscard]] Action v3_legal_buy(
    const GameState& state,
    const ActionMask& legal,
    DefId def) noexcept {
    if (def == DEF_PROVINCE && !v3_province_allowed(state)) {
        return A_PASS;
    }
    if (v3_avoid_last_pile_buy(state, def)) {
        return A_PASS;
    }
    return legal_buy(legal, def);
}

[[nodiscard]] int v3_vp_value(DefId def) noexcept {
    return def < card_def_count() ? static_cast<int>(card_def(def).vp) : 0;
}

[[nodiscard]] std::optional<Action> v3_endgame_buy(
    const GameState& state,
    const ActionMask& legal,
    EngineV3Strategy strategy,
    bool greening_started) noexcept {
    const PlayerId player = state.decision.player;
    const int my_score = score(state, player);
    const int opponent_score = score(state, v3_opponent(state, player));
    const int provinces_left = supply_count(state, DEF_PROVINCE);
    const int coins = state.coins;

    if (provinces_left <= 2 && coins >= 8 && v3_province_allowed(state)) {
        const Action province = legal_buy(legal, DEF_PROVINCE);
        if (province != A_PASS) {
            return province;
        }
    }

    if (v3_empty_supply_piles(state) == 2) {
        Action best = A_PASS;
        int best_vp = -999;
        int best_priority = 9999;
        for (std::uint8_t i = 0; i < state.num_piles; ++i) {
            const Pile& pile = state.piles[i];
            if (pile_count(pile) <= 0 || pile_count(pile) > static_cast<int>(state.buys)) {
                continue;
            }
            const DefId def = pile_top_def(state, pile);
            if (def == DEF_COPPER || def == DEF_CURSE || (def == DEF_PROVINCE && !v3_province_allowed(state))) {
                continue;
            }
            const int vp = v3_vp_value(def);
            if (my_score + vp <= opponent_score) {
                continue;
            }
            const Action action = legal_buy(legal, def);
            const int priority = buy_priority(def);
            if (action != A_PASS && (best == A_PASS || vp > best_vp
                || (vp == best_vp && priority < best_priority))) {
                best = action;
                best_vp = vp;
                best_priority = priority;
            }
        }
        if (best != A_PASS) {
            return best;
        }
    }

    const bool blocked_last_province = coins >= 8 && provinces_left == 1
        && !v3_province_allowed(state);
    const bool building_engine = v3_is_engine_strategy(strategy)
        && !greening_started && provinces_left > 3;
    const bool p2_engine = player == 1U && strategy == EngineV3Strategy::Engine;
    const int duchy_threshold = p2_engine ? 1 : 4;
    const int five_coin_duchy_threshold = p2_engine ? 2 : 5;
    if (!building_engine && coins >= 5 && (coins <= 7 || blocked_last_province)
        && (provinces_left <= duchy_threshold
            || (coins == 5 && provinces_left <= five_coin_duchy_threshold))) {
        const Action duchy = v3_legal_buy(state, legal, DEF_DUCHY);
        if (duchy != A_PASS) {
            return duchy;
        }
    }
    const bool early_estate = strategy != EngineV3Strategy::Engine
        && provinces_left <= 3;
    if (!building_engine && (provinces_left <= 2 || early_estate)
        && coins >= 2 && coins <= 4) {
        const Action estate = v3_legal_buy(state, legal, DEF_ESTATE);
        if (estate != A_PASS) {
            return estate;
        }
    }
    return std::nullopt;
}

[[nodiscard]] V3ActionCounts v3_action_counts(const DeckProfile& profile) noexcept {
    V3ActionCounts counts{};
    for (DefId def = 0; def < BASIC_CARD_COUNT; ++def) {
        if (profile.counts[def] == 0U || !is_action_def(def)) {
            continue;
        }
        if (def == DEF_CHAPEL || def == DEF_SENTRY || def == DEF_THRONE_ROOM) {
            continue;
        }
        counts.actions += profile.counts[def];
        if (card_yield(def).actions == 0) {
            counts.terminals += profile.counts[def];
        }
    }
    return counts;
}

[[nodiscard]] int v3_engine_terminal_count(const DeckProfile& profile) noexcept {
    int terminals = 0;
    for (DefId def = 0; def < BASIC_CARD_COUNT; ++def) {
        if (profile.counts[def] == 0U || !is_action_def(def)
            || def == DEF_CHAPEL || def == DEF_MOAT) {
            continue;
        }
        if (card_yield(def).actions == 0) {
            terminals += profile.counts[def];
        }
    }
    return terminals;
}

[[nodiscard]] int v3_money_capacity(const DeckProfile& profile) noexcept {
    // card_yield groups Sentry with Poacher for action/card flow, but Sentry
    // does not produce a coin. Correct that shared analysis value for v3.
    return static_cast<int>(profile.total_money)
        + static_cast<int>(profile.total_plus_coins)
        - static_cast<int>(profile.counts[DEF_SENTRY]);
}

[[nodiscard]] int v3_plus_buy_count(const DeckProfile& profile) noexcept {
    return static_cast<int>(profile.counts[DEF_MARKET])
        + static_cast<int>(profile.counts[DEF_FESTIVAL])
        + static_cast<int>(profile.counts[DEF_COUNCIL_ROOM]);
}

[[nodiscard]] int v3_draw_piece_count(const DeckProfile& profile) noexcept {
    return static_cast<int>(profile.terminal_draw)
        + static_cast<int>(profile.counts[DEF_LABORATORY]);
}

[[nodiscard]] bool v3_engine_capacity(const DeckProfile& profile) noexcept {
    const int deck_size = static_cast<int>(profile.deck_size);
    const int draw_target = deck_size > 5 ? deck_size - 5 : 0;
    return v3_money_capacity(profile) >= 8 && static_cast<int>(profile.total_plus_cards) >= draw_target;
}

[[nodiscard]] bool v3_engine_capacity_for(
    EngineV3Strategy strategy,
    const DeckProfile& profile) noexcept {
    if (strategy == EngineV3Strategy::FatHybrid) {
        const bool action_flow = profile.villages > 0U
            || profile.counts[DEF_LABORATORY] >= 2U;
        return v3_money_capacity(profile) >= 10
            && profile.counts[DEF_GOLD] >= 2U
            && v3_plus_buy_count(profile) > 0
            && v3_draw_piece_count(profile) >= 2
            && action_flow;
    }
    if (strategy != EngineV3Strategy::ThinEngine) {
        return v3_engine_capacity(profile);
    }
    const int deck_size = static_cast<int>(profile.deck_size);
    const int draw_target = deck_size > 6 ? deck_size - 6 : 0;
    return v3_money_capacity(profile) >= 8
        && static_cast<int>(profile.total_plus_cards) >= draw_target;
}

[[nodiscard]] bool v3_engine_greening(const GameState& state, const DeckProfile& profile) noexcept {
    return v3_engine_capacity(profile)
        || supply_count(state, DEF_PROVINCE) <= 5
        || v3_my_turns(state) >= 15;
}

[[nodiscard]] bool v3_engine_greening_for(
    const GameState& state,
    const DeckProfile& profile,
    EngineV3Strategy strategy) noexcept {
    if (strategy != EngineV3Strategy::ThinEngine
        && strategy != EngineV3Strategy::FatHybrid) {
        return v3_engine_greening(state, profile);
    }
    const int provinces_left = supply_count(state, DEF_PROVINCE);
    const int turn = v3_my_turns(state);
    const int seat_adjustment = state.decision.player == 1U ? 1 : 0;
    const EngineV3Kingdom kingdom = analyze_v3_kingdom(state);
    if (strategy == EngineV3Strategy::FatHybrid) {
        const bool plus_buy_ready = !kingdom.plus_buy || v3_plus_buy_count(profile) > 0;
        const bool action_flow = profile.villages > 0U
            || profile.counts[DEF_LABORATORY] >= 2U;
        const bool build_ready = v3_draw_piece_count(profile) >= 2
            && plus_buy_ready && action_flow;
        return v3_engine_capacity_for(strategy, profile)
            || provinces_left <= 5
            || (turn >= 12 - seat_adjustment && build_ready
                && v3_money_capacity(profile) >= 9
                && profile.counts[DEF_GOLD] >= 2U);
    }
    const int deck_size = static_cast<int>(profile.deck_size);
    const int draw_target = deck_size > 6 ? deck_size - 6 : 0;
    const bool owned_trasher = profile.chapels > 0U || profile.sentries > 0U;
    const bool chapel_build = kingdom.chapel_ok
        && v3_draw_piece_count(profile) >= 1;
    const bool sentry_build = !kingdom.chapel_ok
        && owned_trasher
        && v3_draw_piece_count(profile) >= 2
        && profile.villages >= 2U
        && (!kingdom.plus_buy || v3_plus_buy_count(profile) > 0);
    const bool build_ready = chapel_build || sentry_build;
    const int gold_target = kingdom.chapel_ok ? 0 : 2;
    const int money_target = kingdom.chapel_ok ? 8 : 12;
    const bool capacity_ready = build_ready
        && v3_money_capacity(profile) >= money_target
        && profile.counts[DEF_GOLD] >= gold_target
        && (kingdom.chapel_ok
            || static_cast<int>(profile.total_plus_cards) >= draw_target);
    return capacity_ready
        || provinces_left <= 5
        || (turn >= 11 - seat_adjustment && build_ready
            && v3_money_capacity(profile) >= money_target
            && profile.counts[DEF_GOLD] >= gold_target);
}

[[nodiscard]] Action v3_big_money_buy(const GameState& state, const ActionMask& legal) noexcept {
    const int provinces_left = supply_count(state, DEF_PROVINCE);
    const DeckProfile profile = analyze_deck(state, state.decision.player);
    const int coins = state.coins;

    if (coins >= 8) {
        if (profile.counts[DEF_GOLD] == 0U && profile.counts[DEF_SILVER] < 5U) {
            const Action gold = v3_legal_buy(state, legal, DEF_GOLD);
            if (gold != A_PASS) {
                return gold;
            }
        }
        const Action province = v3_legal_buy(state, legal, DEF_PROVINCE);
        if (province != A_PASS) {
            return province;
        }
    }
    if (coins >= 6) {
        if (provinces_left <= 4) {
            const Action duchy = v3_legal_buy(state, legal, DEF_DUCHY);
            if (duchy != A_PASS) {
                return duchy;
            }
        }
        const Action gold = v3_legal_buy(state, legal, DEF_GOLD);
        if (gold != A_PASS) {
            return gold;
        }
    }
    if (coins >= 5) {
        if (provinces_left <= 5) {
            const Action duchy = v3_legal_buy(state, legal, DEF_DUCHY);
            if (duchy != A_PASS) {
                return duchy;
            }
        }
        const Action silver = v3_legal_buy(state, legal, DEF_SILVER);
        if (silver != A_PASS) {
            return silver;
        }
    }
    if (coins >= 3) {
        if (provinces_left <= 2) {
            const Action estate = v3_legal_buy(state, legal, DEF_ESTATE);
            if (estate != A_PASS) {
                return estate;
            }
        }
        const Action silver = v3_legal_buy(state, legal, DEF_SILVER);
        if (silver != A_PASS) {
            return silver;
        }
    }
    if (coins == 2 && provinces_left <= 3) {
        const Action estate = v3_legal_buy(state, legal, DEF_ESTATE);
        if (estate != A_PASS) {
            return estate;
        }
    }
    return legal.test(A_PASS) ? A_PASS : first_legal(legal);
}

[[nodiscard]] DefId v3_best_bmx_action(const GameState& state) noexcept {
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
    for (const DefId def : kBestActions) {
        if (kingdom_has_available(state, def)) {
            return def;
        }
    }
    return NONE;
}

[[nodiscard]] Action v3_bmx_late_component_buy(
    const GameState& state,
    const ActionMask& legal,
    const EngineV3Kingdom& kingdom,
    const DeckProfile& profile,
    const DeckProfile& opponent) noexcept {
    const int turn = v3_my_turns(state);
    const int treasure_depth = static_cast<int>(profile.counts[DEF_SILVER])
        + static_cast<int>(profile.counts[DEF_GOLD]);
    if (turn < 5 || turn > 8 || treasure_depth < 2) {
        return A_PASS;
    }

    if (kingdom.moat_ok && opponent.counts[DEF_WITCH] > profile.counts[DEF_WITCH]
        && profile.counts[DEF_MOAT] == 0U && state.coins >= 2 && state.coins <= 3) {
        const Action moat = v3_legal_buy(state, legal, DEF_MOAT);
        if (moat != A_PASS) {
            return moat;
        }
    }
    if (kingdom.sentry_ok && profile.sentries == 0U
        && profile.counts[DEF_CURSE] >= 2U) {
        const Action sentry = v3_legal_buy(state, legal, DEF_SENTRY);
        if (sentry != A_PASS) {
            return sentry;
        }
    }
    const std::uint8_t witch_limit = kingdom.chapel_ok ? 1U : 2U;
    if (kingdom.witch_ok && profile.counts[DEF_WITCH] < witch_limit) {
        const Action witch = v3_legal_buy(state, legal, DEF_WITCH);
        if (witch != A_PASS) {
            return witch;
        }
    }
    return A_PASS;
}

[[nodiscard]] Action v3_bmx_green_buy(const GameState& state, const ActionMask& legal) noexcept {
    const int coins = state.coins;
    if (coins >= 8) {
        const Action province = v3_legal_buy(state, legal, DEF_PROVINCE);
        if (province != A_PASS) {
            return province;
        }
    }
    if (coins >= 6) {
        const Action gold = v3_legal_buy(state, legal, DEF_GOLD);
        if (gold != A_PASS) {
            return gold;
        }
    }
    if (coins >= 5) {
        constexpr DefId kGreenActions[] = {DEF_LABORATORY, DEF_MARKET, DEF_FESTIVAL};
        for (const DefId def : kGreenActions) {
            const Action action = v3_legal_buy(state, legal, def);
            if (action != A_PASS) {
                return action;
            }
        }
        const Action silver = v3_legal_buy(state, legal, DEF_SILVER);
        if (silver != A_PASS) {
            return silver;
        }
    }
    if (coins >= 3) {
        const Action silver = v3_legal_buy(state, legal, DEF_SILVER);
        if (silver != A_PASS) {
            return silver;
        }
    }
    return legal.test(A_PASS) ? A_PASS : first_legal(legal);
}

[[nodiscard]] Action v3_bmx_buy(
    const GameState& state,
    const ActionMask& legal,
    const EngineV3Kingdom& kingdom) noexcept {
    if (state.coins >= 8) {
        const Action province = v3_legal_buy(state, legal, DEF_PROVINCE);
        if (province != A_PASS) {
            return province;
        }
    }
    const DeckProfile profile = analyze_deck(state, state.decision.player);
    const DeckProfile opponent = analyze_deck(
        state, v3_opponent(state, state.decision.player));
    const double deck_size = profile.deck_size > 0U ? static_cast<double>(profile.deck_size) : 1.0;
    const double money_density = static_cast<double>(v3_money_capacity(profile)) / deck_size;
    const bool is_second = state.decision.player == 1U;
    const Action late_component = v3_bmx_late_component_buy(
        state, legal, kingdom, profile, opponent);
    if (late_component != A_PASS) {
        return late_component;
    }
    if (v3_my_turns(state) <= 9) {
        const DefId first_attack = v3_best_bmx_action(state);
        if (first_attack != NONE && first_attack != DEF_SENTRY
            && profile.counts[first_attack] == 0U) {
            const Action action = v3_legal_buy(state, legal, first_attack);
            if (action != A_PASS) {
                return action;
            }
        }
    }
    const V3ActionCounts action_counts = v3_action_counts(profile);
    const bool bandit_race = kingdom.bandit_ok && !kingdom.witch_ok;
    const bool lab_race = kingdom.lab_ok && !kingdom.witch_ok
        && !kingdom.militia_ok && !kingdom.bandit_ok;
    const int green_turn = lab_race ? 1 : (bandit_race ? 2 : 4);
    const bool greening = money_density >= (is_second ? 1.3 : 1.2)
        || v3_my_turns(state) > green_turn;
    if (greening) {
        const Action green = v3_bmx_green_buy(state, legal);
        if (green != A_PASS) {
            return green;
        }
    }

    const bool has_trasher = profile.chapels > 0U || profile.sentries > 0U;
    if ((kingdom.chapel_ok || kingdom.sentry_ok) && !has_trasher
        && (kingdom.witch_ok || kingdom.militia_ok || kingdom.bandit_ok)) {
        if (v3_my_turns(state) <= 2 && (state.coins == 2 || state.coins == 3)) {
            const Action chapel = v3_legal_buy(state, legal, DEF_CHAPEL);
            if (chapel != A_PASS) {
                return chapel;
            }
        }
        const Action sentry = v3_legal_buy(state, legal, DEF_SENTRY);
        if (sentry != A_PASS) {
            return sentry;
        }
    }

    if (kingdom.witch_ok && supply_count(state, DEF_WITCH) == 0
        && profile.counts[DEF_WITCH] == 0U && kingdom.moat_ok
        && state.coins >= 2 && state.coins <= 3) {
        const Action moat = v3_legal_buy(state, legal, DEF_MOAT);
        if (moat != A_PASS) {
            return moat;
        }
    }

    const double terminal_density = static_cast<double>(action_counts.terminals) / deck_size;
    const double action_density = static_cast<double>(action_counts.actions) / deck_size;
    const DefId best_def = v3_best_bmx_action(state);
    if (best_def != NONE && best_def != DEF_SENTRY) {
        bool under_limit = false;
        if (best_def == DEF_WITCH) {
            const std::uint8_t witch_limit = kingdom.chapel_ok ? 1U : 3U;
            under_limit = profile.counts[DEF_WITCH] < witch_limit;
        } else if (best_def == DEF_MILITIA) {
            const std::uint8_t militia_limit = is_second ? 4U : 3U;
            under_limit = profile.counts[DEF_MILITIA] < militia_limit;
        } else if (card_yield(best_def).actions == 0) {
            under_limit = terminal_density < 0.08;
        } else {
            under_limit = action_density < 0.35;
        }
        if (under_limit) {
            const Action best = v3_legal_buy(state, legal, best_def);
            if (best != A_PASS) {
                return best;
            }
        }
    }

    if (state.coins >= 5 && action_density < 0.35) {
        constexpr DefId kCantripBuys[] = {DEF_LABORATORY, DEF_MARKET};
        for (const DefId def : kCantripBuys) {
            const Action action = v3_legal_buy(state, legal, def);
            if (action != A_PASS) {
                return action;
            }
        }
    }
    if (state.coins >= 6) {
        const Action gold = v3_legal_buy(state, legal, DEF_GOLD);
        if (gold != A_PASS) {
            return gold;
        }
    }
    if (state.coins >= 3) {
        const Action silver = v3_legal_buy(state, legal, DEF_SILVER);
        if (silver != A_PASS) {
            return silver;
        }
    }
    return legal.test(A_PASS) ? A_PASS : first_legal(legal);
}

[[nodiscard]] Action v3_best_terminal_draw_buy(
    const GameState& state,
    const ActionMask& legal) noexcept {
    constexpr DefId kTerminalDraw[] = {
        DEF_COUNCIL_ROOM,
        DEF_SMITHY,
        DEF_LIBRARY,
        DEF_MOAT,
    };
    for (const DefId def : kTerminalDraw) {
        const Action action = v3_legal_buy(state, legal, def);
        if (action != A_PASS) {
            return action;
        }
    }
    return A_PASS;
}

[[nodiscard]] Action v3_engine_opening_buy(
    const GameState& state,
    const ActionMask& legal,
    const EngineV3Kingdom& kingdom,
    const DeckProfile& profile,
    EngineV3Strategy strategy) noexcept {
    const int coins = state.coins;
    if (coins >= 2 && coins <= 3) {
        if (strategy != EngineV3Strategy::Engine) {
            return coins == 3 ? v3_legal_buy(state, legal, DEF_SILVER) : A_PASS;
        }
        if (profile.chapels == 0U) {
            const Action chapel = v3_legal_buy(state, legal, DEF_CHAPEL);
            if (chapel != A_PASS) {
                return chapel;
            }
        }
        if (coins == 3) {
            const Action silver = v3_legal_buy(state, legal, DEF_SILVER);
            if (silver != A_PASS) {
                return silver;
            }
        }
        return A_PASS;
    }
    if (coins == 4) {
        if (!kingdom.witch_ok && !kingdom.lab_ok
            && kingdom.draw_terminal && profile.counts[DEF_SMITHY] == 0U) {
            const Action smithy = v3_legal_buy(state, legal, DEF_SMITHY);
            if (smithy != A_PASS) {
                return smithy;
            }
        }
        return v3_legal_buy(state, legal, DEF_SILVER);
    }
    if (coins == 5) {
        if (kingdom.witch_ok && profile.counts[DEF_WITCH] < 1U) {
            const Action witch = v3_legal_buy(state, legal, DEF_WITCH);
            if (witch != A_PASS) {
                return witch;
            }
        }
        if (kingdom.lab_ok) {
            const Action laboratory = v3_legal_buy(state, legal, DEF_LABORATORY);
            if (laboratory != A_PASS) {
                return laboratory;
            }
        }
        const Action terminal = v3_best_terminal_draw_buy(state, legal);
        if (terminal != A_PASS) {
            return terminal;
        }
        return v3_legal_buy(state, legal, DEF_SILVER);
    }
    return A_PASS;
}

[[nodiscard]] Action v3_village_buy(
    const GameState& state,
    const ActionMask& legal,
    int coins) noexcept {
    if (coins >= 5) {
        const Action festival = v3_legal_buy(state, legal, DEF_FESTIVAL);
        if (festival != A_PASS) {
            return festival;
        }
    }
    if (coins >= 3) {
        return v3_legal_buy(state, legal, DEF_VILLAGE);
    }
    return A_PASS;
}

[[nodiscard]] Action v3_payload_buy(
    const GameState& state,
    const ActionMask& legal,
    int coins) noexcept {
    if (coins < 5) {
        return A_PASS;
    }
    constexpr DefId kPayloads[] = {
        DEF_MARKET,
        DEF_FESTIVAL,
        DEF_COUNCIL_ROOM,
    };
    for (const DefId def : kPayloads) {
        const Action action = v3_legal_buy(state, legal, def);
        if (action != A_PASS) {
            return action;
        }
    }
    return A_PASS;
}

[[nodiscard]] Action v3_thin_component_buy(
    const GameState& state,
    const ActionMask& legal,
    const EngineV3Kingdom& kingdom,
    const DeckProfile& profile) noexcept {
    const int coins = state.coins;
    const int turn = v3_my_turns(state);
    const int draw_pieces = v3_draw_piece_count(profile);
    const int villages = static_cast<int>(profile.villages);
    const int component_target = profile.chapels > 0U ? 1 : 2;

    if (kingdom.chapel_ok && profile.chapels == 0U && turn <= 2
        && coins == 2) {
        const Action chapel = v3_legal_buy(state, legal, DEF_CHAPEL);
        if (chapel != A_PASS) {
            return chapel;
        }
    }
    if (kingdom.sentry_ok && profile.sentries < 2U && coins >= 5) {
        const Action sentry = v3_legal_buy(state, legal, DEF_SENTRY);
        if (sentry != A_PASS) {
            return sentry;
        }
    }
    if (kingdom.witch_ok && profile.counts[DEF_WITCH] < 2U && coins >= 5) {
        const Action witch = v3_legal_buy(state, legal, DEF_WITCH);
        if (witch != A_PASS) {
            return witch;
        }
    }

    // The no-Witch Chapel race is short: establish real draw, then let the
    // thinned money deck race.  Extra villages/payload are too slow unless
    // bought later with surplus buys.
    if (kingdom.chapel_ok) {
        if (draw_pieces < 1 && coins >= 4) {
            return v3_best_terminal_draw_buy(state, legal);
        }
        return A_PASS;
    }

    const int covered_draw = draw_pieces < component_target
        ? draw_pieces : component_target;
    if (covered_draw > 0 && villages < covered_draw) {
        const Action village = v3_village_buy(state, legal, coins);
        if (village != A_PASS) {
            return village;
        }
    }
    if (draw_pieces < component_target && coins >= 4) {
        const Action terminal = v3_best_terminal_draw_buy(state, legal);
        if (terminal != A_PASS) {
            return terminal;
        }
        if (coins >= 5 && kingdom.lab_ok) {
            const Action laboratory = v3_legal_buy(state, legal, DEF_LABORATORY);
            if (laboratory != A_PASS) {
                return laboratory;
            }
        }
    }
    if (draw_pieces >= component_target && villages < component_target) {
        const Action village = v3_village_buy(state, legal, coins);
        if (village != A_PASS) {
            return village;
        }
    }
    if (draw_pieces >= component_target && v3_plus_buy_count(profile) == 0) {
        const Action payload = v3_payload_buy(state, legal, coins);
        if (payload != A_PASS) {
            return payload;
        }
    }
    if (kingdom.militia_ok && profile.counts[DEF_MILITIA] == 0U
        && villages >= v3_engine_terminal_count(profile)
        && coins >= 4) {
        const Action militia = v3_legal_buy(state, legal, DEF_MILITIA);
        if (militia != A_PASS) {
            return militia;
        }
    }
    return A_PASS;
}

[[nodiscard]] Action v3_fat_component_buy(
    const GameState& state,
    const ActionMask& legal,
    const EngineV3Kingdom& kingdom,
    const DeckProfile& profile) noexcept {
    const int coins = state.coins;
    const int terminals = static_cast<int>(profile.terminal_draw);
    const int villages = static_cast<int>(profile.villages);
    const int labs = static_cast<int>(profile.counts[DEF_LABORATORY]);
    const int draw_pieces = v3_draw_piece_count(profile);
    const int plus_buys = v3_plus_buy_count(profile);

    if (kingdom.witch_ok && profile.counts[DEF_WITCH] < 2U && coins >= 5) {
        const Action witch = v3_legal_buy(state, legal, DEF_WITCH);
        if (witch != A_PASS) {
            return witch;
        }
    }
    if (kingdom.sentry_ok && profile.sentries < 2U && coins >= 5) {
        const Action sentry = v3_legal_buy(state, legal, DEF_SENTRY);
        if (sentry != A_PASS) {
            return sentry;
        }
    }
    if (kingdom.militia_ok && profile.counts[DEF_MILITIA] == 0U
        && (villages >= v3_engine_terminal_count(profile) || labs > 0)
        && coins >= 4) {
        const Action militia = v3_legal_buy(state, legal, DEF_MILITIA);
        if (militia != A_PASS) {
            return militia;
        }
    }

    const int terminal_cap = kingdom.lab_ok ? 1 : 2;
    if (terminals < terminal_cap && terminals < 3
        && coins >= 4) {
        const Action terminal = v3_best_terminal_draw_buy(state, legal);
        if (terminal != A_PASS) {
            return terminal;
        }
    }
    if (kingdom.lab_ok && labs < 4 && coins >= 5) {
        if (labs >= 2 && plus_buys == 0) {
            const Action payload = v3_payload_buy(state, legal, coins);
            if (payload != A_PASS) {
                return payload;
            }
        }
        if (labs < 2 || coins == 5) {
            const Action laboratory = v3_legal_buy(state, legal, DEF_LABORATORY);
            if (laboratory != A_PASS) {
                return laboratory;
            }
        }
    }
    const int village_target = kingdom.lab_ok ? 0 : 1;
    if (terminals >= 2 && villages < village_target) {
        const Action village = v3_village_buy(state, legal, coins);
        if (village != A_PASS) {
            return village;
        }
    }
    if (draw_pieces >= 2 && plus_buys < 1) {
        const Action payload = v3_payload_buy(state, legal, coins);
        if (payload != A_PASS) {
            return payload;
        }
    }
    return A_PASS;
}

[[nodiscard]] Action v3_engine_component_buy(
    const GameState& state,
    const ActionMask& legal,
    const EngineV3Kingdom& kingdom,
    const DeckProfile& profile,
    EngineV3Strategy strategy) noexcept {
    const int coins = state.coins;
    const int terminals = v3_engine_terminal_count(profile);
    const int villages = static_cast<int>(profile.villages);
    const int draw_cards = static_cast<int>(profile.total_plus_cards);
    const int deck_size = static_cast<int>(profile.deck_size);
    const int draw_target = deck_size > 5 ? deck_size - 5 : 0;

    if (kingdom.witch_ok && profile.counts[DEF_WITCH] < 1U && coins >= 5) {
        const Action witch = v3_legal_buy(state, legal, DEF_WITCH);
        if (witch != A_PASS) {
            return witch;
        }
    }

    if (strategy == EngineV3Strategy::ThinEngine) {
        return v3_thin_component_buy(state, legal, kingdom, profile);
    }
    if (strategy == EngineV3Strategy::FatHybrid) {
        return v3_fat_component_buy(state, legal, kingdom, profile);
    }

    if (villages < terminals && kingdom.village_ok) {
        if (coins >= 5) {
            const Action festival = v3_legal_buy(state, legal, DEF_FESTIVAL);
            if (festival != A_PASS) {
                return festival;
            }
        }
        if (coins >= 3) {
            const Action village = v3_legal_buy(state, legal, DEF_VILLAGE);
            if (village != A_PASS) {
                return village;
            }
        }
    }

    if (kingdom.lab_ok && profile.counts[DEF_LABORATORY] < 3U
        && coins >= 5 && draw_cards < draw_target) {
        const Action laboratory = v3_legal_buy(state, legal, DEF_LABORATORY);
        if (laboratory != A_PASS) {
            return laboratory;
        }
    }

    const bool terminal_draw_room = profile.terminal_draw < profile.villages + 1U
        && profile.terminal_draw < 3U;
    if (draw_cards < draw_target && terminal_draw_room) {
        if (coins >= 5) {
            const Action council_room = v3_legal_buy(state, legal, DEF_COUNCIL_ROOM);
            if (council_room != A_PASS) {
                return council_room;
            }
        }
        if (coins >= 4) {
            const Action smithy = v3_legal_buy(state, legal, DEF_SMITHY);
            if (smithy != A_PASS) {
                return smithy;
            }
        }
    }

    if (kingdom.militia_ok && profile.counts[DEF_MILITIA] == 0U
        && coins >= 4 && coins <= 5 && terminals <= villages + 1) {
        const Action militia = v3_legal_buy(state, legal, DEF_MILITIA);
        if (militia != A_PASS) {
            return militia;
        }
    }

    return A_PASS;
}

[[nodiscard]] Action v3_engine_green_buy(
    const GameState& state,
    const ActionMask& legal,
    const EngineV3Kingdom& kingdom,
    const DeckProfile& profile,
    EngineV3Strategy strategy) noexcept {
    const int provinces_left = supply_count(state, DEF_PROVINCE);
    const int coins = state.coins;
    if (coins >= 8) {
        const Action province = v3_legal_buy(state, legal, DEF_PROVINCE);
        if (province != A_PASS) {
            return province;
        }
    }
    if (state.buys > 1U) {
        if (coins >= 5 && provinces_left <= 6) {
            const Action duchy = v3_legal_buy(state, legal, DEF_DUCHY);
            if (duchy != A_PASS) {
                return duchy;
            }
        }
        if (coins >= 2 && (provinces_left <= 2 || v3_empty_supply_piles(state) >= 2)) {
            const Action estate = v3_legal_buy(state, legal, DEF_ESTATE);
            if (estate != A_PASS) {
                return estate;
            }
        }
        const Action component = v3_engine_component_buy(
            state, legal, kingdom, profile, strategy);
        if (component != A_PASS) {
            return component;
        }
    }
    if (coins >= 6) {
        const Action gold = v3_legal_buy(state, legal, DEF_GOLD);
        if (gold != A_PASS) {
            return gold;
        }
    }
    if (coins >= 3) {
        const Action silver = v3_legal_buy(state, legal, DEF_SILVER);
        if (silver != A_PASS) {
            return silver;
        }
    }
    return legal.test(A_PASS) ? A_PASS : first_legal(legal);
}

[[nodiscard]] Action v3_burst_green_buy(
    const GameState& state,
    const ActionMask& legal,
    bool green_vp_bought) noexcept {
    const int coins = state.coins;
    const int provinces_left = supply_count(state, DEF_PROVINCE);

    if (coins >= 8) {
        const Action province = v3_legal_buy(state, legal, DEF_PROVINCE);
        if (province != A_PASS) {
            return province;
        }
    }
    // A single Duchy in the middle of the Province race clogs the engine and
    // throws away its payload advantage.  Take Duchy early only as surplus
    // after another VP buy; otherwise keep growing the economy until the
    // Province pile is in its closing half.
    if (coins >= 5 && (green_vp_bought || provinces_left <= 4)) {
        const Action duchy = v3_legal_buy(state, legal, DEF_DUCHY);
        if (duchy != A_PASS) {
            return duchy;
        }
    }
    if (coins >= 2
        && (green_vp_bought || provinces_left <= 3)) {
        const Action estate = v3_legal_buy(state, legal, DEF_ESTATE);
        if (estate != A_PASS) {
            return estate;
        }
    }
    if (coins >= 6) {
        const Action gold = v3_legal_buy(state, legal, DEF_GOLD);
        if (gold != A_PASS) {
            return gold;
        }
    }
    if (coins >= 3) {
        const Action silver = v3_legal_buy(state, legal, DEF_SILVER);
        if (silver != A_PASS) {
            return silver;
        }
    }
    return legal.test(A_PASS) ? A_PASS : first_legal(legal);
}

[[nodiscard]] Action v3_engine_buy(
    const GameState& state,
    const ActionMask& legal,
    const EngineV3Kingdom& kingdom,
    EngineV3Strategy strategy,
    bool greening_started,
    bool green_vp_bought) noexcept {
    const DeckProfile profile = analyze_deck(state, state.decision.player);
    if (strategy == EngineV3Strategy::Engine && state.coins >= 8) {
        const Action province = v3_legal_buy(state, legal, DEF_PROVINCE);
        if (province != A_PASS) {
            return province;
        }
    }
    if (strategy == EngineV3Strategy::Engine && v3_my_turns(state) <= 2) {
        const Action opening = v3_engine_opening_buy(
            state, legal, kingdom, profile, strategy);
        if (opening != A_PASS) {
            return opening;
        }
    }
    if (greening_started) {
        if (strategy == EngineV3Strategy::Engine) {
            return v3_engine_green_buy(state, legal, kingdom, profile, strategy);
        }
        return v3_burst_green_buy(state, legal, green_vp_bought);
    }
    const Action component = v3_engine_component_buy(
        state, legal, kingdom, profile, strategy);
    if (component != A_PASS) {
        return component;
    }
    if (state.coins >= 6) {
        const Action gold = v3_legal_buy(state, legal, DEF_GOLD);
        if (gold != A_PASS) {
            return gold;
        }
    }
    const bool archetype_money = strategy == EngineV3Strategy::ThinEngine
        || strategy == EngineV3Strategy::FatHybrid;
    if (state.coins >= 3
        && (archetype_money || v3_my_turns(state) <= 4 || profile.total_money < 4)) {
        const Action silver = v3_legal_buy(state, legal, DEF_SILVER);
        if (silver != A_PASS) {
            return silver;
        }
    }
    return legal.test(A_PASS) ? A_PASS : first_legal(legal);
}

[[nodiscard]] Action v3_rush_buy(const GameState& state, const ActionMask& legal) noexcept {
    const DeckProfile profile = analyze_deck(state, state.decision.player);
    const int coins = state.coins;
    const int provinces_left = supply_count(state, DEF_PROVINCE);

    if (coins >= 8) {
        const Action province = v3_legal_buy(state, legal, DEF_PROVINCE);
        if (province != A_PASS) {
            return province;
        }
    }
    if (v3_my_turns(state) <= 2 && coins >= 3 && coins <= 4
        && profile.counts[DEF_WORKSHOP] == 0U) {
        const Action workshop = v3_legal_buy(state, legal, DEF_WORKSHOP);
        if (workshop != A_PASS) {
            return workshop;
        }
    }
    if (coins >= 6) {
        if (provinces_left <= 4) {
            const Action duchy = v3_legal_buy(state, legal, DEF_DUCHY);
            if (duchy != A_PASS) {
                return duchy;
            }
        }
        const Action gold = v3_legal_buy(state, legal, DEF_GOLD);
        if (gold != A_PASS) {
            return gold;
        }
    }
    if (coins >= 3) {
        const Action silver = v3_legal_buy(state, legal, DEF_SILVER);
        if (silver != A_PASS) {
            return silver;
        }
    }
    if (coins >= 2 && provinces_left <= 2) {
        const Action estate = v3_legal_buy(state, legal, DEF_ESTATE);
        if (estate != A_PASS) {
            return estate;
        }
    }
    return legal.test(A_PASS) ? A_PASS : first_legal(legal);
}

[[nodiscard]] Action v3_rush_gain(const ActionMask& legal) noexcept {
    constexpr DefId kGains[] = {DEF_GARDENS, DEF_SILVER, DEF_ESTATE};
    for (const DefId def : kGains) {
        const Action action = select_action(def);
        if (legal.test(action)) {
            return action;
        }
    }
    return choose_gain_most_expensive(legal);
}

[[nodiscard]] Action v3_buy(
    const GameState& state,
    const ActionMask& legal,
    bool& greening_started,
    bool green_vp_bought) noexcept {
    const EngineV3Kingdom kingdom = analyze_v3_kingdom(state);
    const EngineV3Strategy strategy = pick_v3_strategy(kingdom);
    const DeckProfile profile = analyze_deck(state, state.decision.player);
    if (v3_is_engine_strategy(strategy)
        && v3_engine_greening_for(state, profile, strategy)) {
        greening_started = true;
    }
    if (const std::optional<Action> endgame = v3_endgame_buy(
            state, legal, strategy, greening_started); endgame.has_value()) {
        return *endgame;
    }
    switch (strategy) {
    case EngineV3Strategy::Rush:
        return v3_rush_buy(state, legal);
    case EngineV3Strategy::Engine:
    case EngineV3Strategy::ThinEngine:
    case EngineV3Strategy::FatHybrid:
        return v3_engine_buy(
            state, legal, kingdom, strategy, greening_started, green_vp_bought);
    case EngineV3Strategy::BigMoneyPlusAction:
        return v3_bmx_buy(state, legal, kingdom);
    case EngineV3Strategy::PureBigMoney:
    default:
        return v3_big_money_buy(state, legal);
    }
}

[[nodiscard]] int v3_action_priority(DefId def) noexcept {
    switch (def) {
    case DEF_VILLAGE:
        return 1;
    case DEF_FESTIVAL:
        return 2;
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
    case DEF_THRONE_ROOM:
        return 20;
    case DEF_WITCH:
        return 30;
    case DEF_COUNCIL_ROOM:
        return 31;
    case DEF_SMITHY:
        return 32;
    case DEF_MILITIA:
        return 33;
    case DEF_LIBRARY:
        return 34;
    case DEF_MOAT:
        return 35;
    case DEF_CHAPEL:
        return 36;
    case DEF_MINE:
        return 38;
    case DEF_REMODEL:
        return 39;
    case DEF_MONEYLENDER:
        return 40;
    case DEF_ARTISAN:
        return 42;
    case DEF_BANDIT:
        return 43;
    case DEF_BUREAUCRAT:
        return 44;
    case DEF_WORKSHOP:
        return 45;
    case DEF_VASSAL:
        return 47;
    default:
        return 50;
    }
}

[[nodiscard]] bool v3_hand_contains(
    const GameState& state,
    PlayerId player,
    DefId def) noexcept {
    if (player >= state.num_players) {
        return false;
    }
    for (Slot slot = 0; slot < state.num_slots; ++slot) {
        if (def_for_slot(state, slot) == def && state.players[player].hand[slot] > 0U) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool v3_chapel_can_trash_from_hand(
    const GameState& state,
    const DeckProfile& profile,
    PlayerId player,
    EngineV3Strategy strategy) noexcept {
    if (v3_hand_contains(state, player, DEF_CURSE)) {
        return true;
    }
    if (supply_count(state, DEF_PROVINCE) > 4 && v3_hand_contains(state, player, DEF_ESTATE)) {
        return true;
    }
    if (!v3_hand_contains(state, player, DEF_COPPER)) {
        return false;
    }
    // Copper thinning keeps a deck-money floor: aggressive floors starve
    // the build (all-copper trashing measured 25+ points worse in the
    // no-Witch race); the floor below keeps $5-hand purchasing power.
    if (strategy == EngineV3Strategy::ThinEngine) {
        return v3_money_capacity(profile) - 1 >= 6;
    }
    if (v3_engine_capacity_for(strategy, profile)) {
        return true;
    }
    return v3_money_capacity(profile) - 1 >= 3;
}

[[nodiscard]] Action v3_best_play_without_chapel(const ActionMask& legal) noexcept {
    Action best = A_PASS;
    int best_priority = 999;
    for (DefId def = 0; def < ACTION_DEF_COUNT; ++def) {
        if (def == DEF_CHAPEL || !is_action_def(def)) {
            continue;
        }
        const Action action = play_action(def);
        if (!legal.test(action)) {
            continue;
        }
        const int priority = v3_action_priority(def);
        if (best == A_PASS || priority < best_priority) {
            best = action;
            best_priority = priority;
        }
    }
    return best;
}

[[nodiscard]] Action v3_engine_play_action(
    const GameState& state,
    const ActionMask& legal,
    EngineV3Strategy strategy) noexcept {
    const PlayerId player = state.decision.player;
    const DeckProfile profile = analyze_deck(state, player);
    const bool curse_in_hand = v3_hand_contains(state, player, DEF_CURSE);
    const bool chapel_playable = legal.test(play_action(DEF_CHAPEL));
    const bool chapel_useful = chapel_playable
        && v3_chapel_can_trash_from_hand(state, profile, player, strategy)
        && (!v3_engine_greening_for(state, profile, strategy) || curse_in_hand);
    const Action best = v3_best_play_without_chapel(legal);
    if (best != A_PASS) {
        const DefId best_def = action_def(best, A_PLAY_BASE);
        if (v3_action_priority(best_def) < v3_action_priority(DEF_WITCH)) {
            return best;
        }
        if (chapel_useful && (profile.junk >= 3U || curse_in_hand)) {
            return play_action(DEF_CHAPEL);
        }
        return best;
    }
    if (chapel_useful) {
        return play_action(DEF_CHAPEL);
    }
    return legal.test(A_PASS) ? A_PASS : first_legal(legal);
}

[[nodiscard]] Action v3_sentry_option(const GameState& state, const ActionMask& legal) noexcept {
    const DefId def = current_sentry_def(state);
    const PlayerId player = state.decision.player;
    const DeckProfile profile = analyze_deck(state, player);
    const EngineV3Strategy strategy = pick_v3_strategy(analyze_v3_kingdom(state));
    const int provinces_left = supply_count(state, DEF_PROVINCE);

    Action desired = option_action(2U);
    if (def == DEF_CURSE) {
        desired = option_action(0U);
    } else if (def == DEF_ESTATE) {
        // Same guard as Chapel: estates become keepable VP late.
        desired = provinces_left > 4 ? option_action(0U) : option_action(1U);
    } else if (def == DEF_COPPER) {
        // Respect the per-strategy copper economy floor Chapel uses; keep
        // (topdeck) the copper when the deck still needs its money.
        const int copper_floor = strategy == EngineV3Strategy::ThinEngine
            || strategy == EngineV3Strategy::FatHybrid ? 6 : 3;
        const bool drawing_deck = v3_engine_capacity_for(strategy, profile);
        desired = (drawing_deck || v3_money_capacity(profile) - 1 >= copper_floor)
            ? option_action(0U)
            : option_action(2U);
    } else if (def == DEF_DUCHY || def == DEF_PROVINCE || def == DEF_COLONY) {
        desired = option_action(1U);
    }
    if (legal.test(desired)) {
        return desired;
    }
    return first_legal_option(legal);
}

[[nodiscard]] Action v3_chapel_trash(
    const GameState& state,
    const ActionMask& legal,
    PlayerId player) noexcept {
    const DeckProfile profile = analyze_deck(state, player);
    const EngineV3Strategy strategy = pick_v3_strategy(analyze_v3_kingdom(state));
    const bool greening = v3_engine_greening_for(state, profile, strategy);
    const bool capacity = v3_engine_capacity_for(strategy, profile);
    const int provinces_left = supply_count(state, DEF_PROVINCE);
    const bool pass_allowed = legal.test(A_PASS) && state.decision.min_left == 0U;

    const Action curse = select_action(DEF_CURSE);
    if (legal.test(curse)) {
        return curse;
    }
    const Action estate = select_action(DEF_ESTATE);
    if (provinces_left > 4 && legal.test(estate)) {
        return estate;
    }
    const Action copper = select_action(DEF_COPPER);
    const int copper_floor = strategy == EngineV3Strategy::ThinEngine ? 6 : 3;
    if (legal.test(copper)
        && (capacity
            || (!greening && v3_money_capacity(profile) - 1 >= copper_floor))) {
        return copper;
    }

    if (!pass_allowed) {
        return choose_select_min_priority(legal, trash_priority, false);
    }
    return A_PASS;
}

[[nodiscard]] bool thinner_kingdom_has(const GameState& state, DefId def) noexcept {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        if (pile_top_def(state, state.piles[i]) == def) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool thinner_supply_near_empty(const GameState& state) noexcept {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        if (pile_count(state.piles[i]) <= 2) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool thinner_has_attack_card(const GameState& state) noexcept {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        const DefId def = pile_top_def(state, state.piles[i]);
        if (def < card_def_count() && (card_def(def).types & TYPE_ATTACK) != 0U) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool thinner_hand_contains(
    const GameState& state,
    PlayerId player,
    DefId def) noexcept {
    if (player >= state.num_players) {
        return false;
    }
    for (Slot slot = 0; slot < state.num_slots; ++slot) {
        if (def_for_slot(state, slot) == def && state.players[player].hand[slot] > 0U) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] int thinner_total_treasures(const DeckProfile& profile) noexcept {
    int total = 0;
    for (DefId def = 0; def < BASIC_CARD_COUNT; ++def) {
        if (is_treasure_def(def)) {
            total += static_cast<int>(profile.counts[def]);
        }
    }
    return total;
}

[[nodiscard]] bool thinner_trash_eligible(
    const GameState& state,
    const DeckProfile& profile,
    DefId def) noexcept {
    if (def == DEF_CURSE) {
        return true;
    }
    if (def == DEF_ESTATE) {
        return supply_count(state, DEF_PROVINCE) > 4;
    }
    if (def == DEF_COPPER) {
        const int silver_and_gold = static_cast<int>(profile.counts[DEF_SILVER])
            + static_cast<int>(profile.counts[DEF_GOLD]);
        return thinner_total_treasures(profile) > 3 && silver_and_gold >= 2;
    }
    return false;
}

[[nodiscard]] bool thinner_has_eligible_junk_in_hand(
    const GameState& state,
    const DeckProfile& profile,
    PlayerId player) noexcept {
    constexpr DefId kJunk[] = {DEF_CURSE, DEF_ESTATE, DEF_COPPER};
    for (const DefId def : kJunk) {
        if (thinner_hand_contains(state, player, def)
            && thinner_trash_eligible(state, profile, def)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] Action thinner_chapel_trash(
    const GameState& state,
    const ActionMask& legal,
    PlayerId player) noexcept {
    const DeckProfile profile = analyze_deck(state, player);
    constexpr DefId kJunk[] = {DEF_CURSE, DEF_ESTATE, DEF_COPPER};
    for (const DefId def : kJunk) {
        const Action action = select_action(def);
        if (legal.test(action) && thinner_trash_eligible(state, profile, def)) {
            return action;
        }
    }
    return legal.test(A_PASS) && state.decision.min_left == 0U
        ? A_PASS : first_legal(legal);
}

[[nodiscard]] Action thinner_sentry_option(
    const GameState& state,
    const ActionMask& legal) noexcept {
    const DeckProfile profile = analyze_deck(state, state.decision.player);
    const DefId def = current_sentry_def(state);
    const Action desired = thinner_trash_eligible(state, profile, def)
        ? option_action(0U) : option_action(2U);
    return legal.test(desired) ? desired : first_legal_option(legal);
}

[[nodiscard]] bool thinner_has_trasher_buyable(
    const ActionMask& legal) noexcept {
    return legal.test(buy_action(DEF_CHAPEL))
        || legal.test(buy_action(DEF_SENTRY))
        || legal.test(buy_action(DEF_MONEYLENDER));
}

[[nodiscard]] bool thinner_engine_viable(
    const GameState& state,
    const EngineV3Kingdom& kingdom) noexcept {
    const bool cantrip_economy = kingdom.market_ok || kingdom.festival_ok
        || thinner_kingdom_has(state, DEF_MERCHANT);
    return (kingdom.lab_ok && (kingdom.village_ok || kingdom.witch_ok || cantrip_economy))
        || (kingdom.village_ok && kingdom.draw_terminal);
}

[[nodiscard]] bool thinner_engine_online(const DeckProfile& profile) noexcept {
    const V3ActionCounts action_counts = v3_action_counts(profile);
    return profile.counts[DEF_LABORATORY] >= 2U
        || (action_counts.actions >= 3
            && (profile.villages > 0U || profile.counts[DEF_LABORATORY] > 0U)
            && profile.total_plus_cards >= 2);
}

[[nodiscard]] Action thinner_engine_component_buy(
    const GameState& state,
    const ActionMask& legal,
    const EngineV3Kingdom& kingdom,
    const DeckProfile& profile) noexcept {
    if (!thinner_engine_viable(state, kingdom)) {
        return A_PASS;
    }

    const int coins = state.coins;
    const int terminals = v3_engine_terminal_count(profile);
    const int labs = static_cast<int>(profile.counts[DEF_LABORATORY]);

    const bool economy_bootstrapped = profile.counts[DEF_SILVER] >= 2U;

    // A genuinely thin deck can profitably chain several Laboratories.  Keep
    // adding non-terminal draw before its first payload Gold; this is the
    // capability the sentinel is meant to probe rather than a Chapel-money
    // splash with one draw card.
    if (kingdom.lab_ok && labs < 4 && coins >= 5) {
        const Action laboratory = v3_legal_buy(state, legal, DEF_LABORATORY);
        if (laboratory != A_PASS) {
            return laboratory;
        }
    }

    // After a real draw core is in place, turn the next $6-$7 hand into
    // payload instead of endlessly cycling cards that cannot buy Provinces.
    if (profile.counts[DEF_GOLD] == 0U && (labs >= 3 || terminals >= 2) && coins >= 6) {
        return A_PASS;
    }

    if (!economy_bootstrapped) {
        // A Village/terminal build can take its first terminal at $4, but do
        // not spend the early $3-$4 hands on Merchant or extra villages.
        if (!kingdom.lab_ok && terminals == 0 && coins >= 4) {
            return v3_best_terminal_draw_buy(state, legal);
        }
        return A_PASS;
    }

    // Reuse EngineV3's balanced Village/terminal rules after the thin deck
    // has opening money. Witch remains governed by Thinner's two-copy splash
    // above, so this helper's one-copy Witch branch is intentionally skipped.
    const Action balanced = v3_engine_component_buy(
        state, legal, kingdom, profile, EngineV3Strategy::Engine);
    if (balanced != A_PASS && action_def(balanced, A_BUY_BASE) != DEF_WITCH) {
        return balanced;
    }

    // +action cantrips add economy without making the deck terminal-heavy.
    if (coins >= 5 && profile.counts[DEF_MARKET] < 1U) {
        const Action market = v3_legal_buy(state, legal, DEF_MARKET);
        if (market != A_PASS) {
            return market;
        }
    }
    if (coins >= 5 && profile.counts[DEF_FESTIVAL] < 1U) {
        const Action festival = v3_legal_buy(state, legal, DEF_FESTIVAL);
        if (festival != A_PASS) {
            return festival;
        }
    }
    if (coins >= 3 && profile.counts[DEF_MERCHANT] < 2U) {
        const Action merchant = v3_legal_buy(state, legal, DEF_MERCHANT);
        if (merchant != A_PASS) {
            return merchant;
        }
    }

    return A_PASS;
}

[[nodiscard]] Action thinner_buy(
    const GameState& state,
    const ActionMask& legal,
    EngineBotV3& engine_buy_policy) noexcept {
    const PlayerId player = state.decision.player;
    const DeckProfile profile = analyze_deck(state, player);
    const int coins = state.coins;
    const int provinces_left = supply_count(state, DEF_PROVINCE);
    const EngineV3Kingdom kingdom = analyze_v3_kingdom(state);
    const bool chapel_in_kingdom = thinner_kingdom_has(state, DEF_CHAPEL);
    const bool fallback_trasher = profile.counts[DEF_SENTRY] > 0U
        || profile.counts[DEF_MONEYLENDER] > 0U;
    const int my_turns = state.num_players == 0U
        ? 0 : static_cast<int>(state.turn_counter / state.num_players);
    const bool engine_viable = thinner_engine_viable(state, kingdom);
    const bool engine_building = engine_viable
        && (profile.counts[DEF_CHAPEL] > 0U || fallback_trasher);

    if (coins >= 8) {
        // A Chapel engine gets a short, bounded chance to establish a real
        // action core instead of immediately reverting to money green.
        if (engine_building && profile.total_actions < 3U && my_turns < 10) {
            const Action component = thinner_engine_component_buy(state, legal, kingdom, profile);
            if (component != A_PASS) {
                return component;
            }
        }
        const Action province = legal_buy(legal, DEF_PROVINCE);
        if (province != A_PASS) {
            return province;
        }
    }

    if (my_turns < 4 && chapel_in_kingdom && profile.counts[DEF_CHAPEL] == 0U
        && coins >= 2 && coins <= 4) {
        const Action chapel = legal_buy(legal, DEF_CHAPEL);
        if (chapel != A_PASS) {
            return chapel;
        }
    }

    if (!chapel_in_kingdom && !fallback_trasher) {
        if (coins >= 5) {
            const Action sentry = legal_buy(legal, DEF_SENTRY);
            if (sentry != A_PASS) {
                return sentry;
            }
        }
        if (coins == 4) {
            const Action moneylender = legal_buy(legal, DEF_MONEYLENDER);
            if (moneylender != A_PASS) {
                return moneylender;
            }
        }
    }

    if (coins >= 5 && (provinces_left <= 4 || thinner_supply_near_empty(state))) {
        const Action duchy = legal_buy(legal, DEF_DUCHY);
        if (duchy != A_PASS) {
            return duchy;
        }
    }
    if (coins >= 2 && provinces_left <= 2) {
        const Action estate = legal_buy(legal, DEF_ESTATE);
        if (estate != A_PASS) {
            return estate;
        }
    }

    if (coins == 5 && thinner_kingdom_has(state, DEF_WITCH)
        && profile.counts[DEF_WITCH] < 2U) {
        const Action witch = legal_buy(legal, DEF_WITCH);
        if (witch != A_PASS) {
            return witch;
        }
    }

    // Keep the specialised Chapel opening, trashing, and greening overlays,
    // then let EngineV3 select a balanced draw/action core.  Its kingdom
    // selection avoids constructing an engine from weak pieces, while the
    // filters retain Thinner's smaller Silver and Witch caps.
    if (engine_viable) {
        const Action engine_choice = engine_buy_policy.choose_action(
            state, legal, 1);
        if (engine_choice != A_PASS
            && engine_choice >= A_BUY_BASE && engine_choice < A_EVENT_BASE) {
            const DefId def = action_def(engine_choice, A_BUY_BASE);
            const bool capped_witch = def == DEF_WITCH && profile.counts[DEF_WITCH] >= 2U;
            const bool excess_silver = def == DEF_SILVER
                && (profile.counts[DEF_SILVER] >= 3U || thinner_engine_online(profile));
            const bool early_estate = def == DEF_ESTATE && provinces_left > 2;
            const bool early_duchy = def == DEF_DUCHY
                && !(provinces_left <= 4 || thinner_supply_near_empty(state));
            if (!capped_witch && !excess_silver && !early_estate && !early_duchy) {
                return engine_choice;
            }
        }
    }

    // Silver is only opening fuel for an engine build.  In dead kingdoms the
    // old money fallback retains its reliable Silver/Gold behavior.
    if (engine_building && coins >= 3 && coins <= 4 && profile.counts[DEF_SILVER] < 2U) {
        const Action silver = legal_buy(legal, DEF_SILVER);
        if (silver != A_PASS) {
            return silver;
        }
    }
    if (engine_viable) {
        const Action component = thinner_engine_component_buy(state, legal, kingdom, profile);
        if (component != A_PASS) {
            return component;
        }
    }
    if (coins >= 6 && coins <= 7) {
        const Action gold = legal_buy(legal, DEF_GOLD);
        if (gold != A_PASS) {
            return gold;
        }
    }
    if (coins >= 3 && coins <= 5) {
        if (!engine_viable || (profile.counts[DEF_SILVER] < 3U && !thinner_engine_online(profile))) {
            const Action silver = legal_buy(legal, DEF_SILVER);
            if (silver != A_PASS) {
                return silver;
            }
        }
    }
    if (coins == 2 && !thinner_has_trasher_buyable(legal)
        && thinner_has_attack_card(state)) {
        const Action moat = legal_buy(legal, DEF_MOAT);
        if (moat != A_PASS) {
            return moat;
        }
    }
    return legal.test(A_PASS) ? A_PASS : first_legal(legal);
}

[[nodiscard]] Action thinner_engine_play_action(
    const GameState& state,
    const ActionMask& legal,
    const DeckProfile& profile,
    PlayerId player) noexcept {
    const bool chapel_useful = legal.test(play_action(DEF_CHAPEL))
        && thinner_has_eligible_junk_in_hand(state, profile, player);
    const Action best = v3_best_play_without_chapel(legal);
    if (best != A_PASS) {
        const DefId def = action_def(best, A_PLAY_BASE);
        const bool moneylender_ready = def != DEF_MONEYLENDER
            || (!thinner_kingdom_has(state, DEF_CHAPEL)
                && thinner_hand_contains(state, player, DEF_COPPER)
                && thinner_trash_eligible(state, profile, DEF_COPPER));
        if (moneylender_ready && v3_action_priority(def) < v3_action_priority(DEF_CHAPEL)) {
            return best;
        }
    }
    if (chapel_useful) {
        return play_action(DEF_CHAPEL);
    }
    if (best != A_PASS) {
        const DefId def = action_def(best, A_PLAY_BASE);
        if (def != DEF_MONEYLENDER
            || (!thinner_kingdom_has(state, DEF_CHAPEL)
                && thinner_hand_contains(state, player, DEF_COPPER)
                && thinner_trash_eligible(state, profile, DEF_COPPER))) {
            return best;
        }
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

Action EngineBotV3::choose_action(
    const GameState& state,
    const ActionMask& legal,
    int legal_count) noexcept {
    if (legal_count <= 0) {
        return A_PASS;
    }

    const DecisionKind decision = static_cast<DecisionKind>(state.decision.kind);
    if (decision == DecisionKind::PhaseAction) {
        const EngineV3Strategy strategy = pick_v3_strategy(analyze_v3_kingdom(state));
        if (strategy == EngineV3Strategy::PureBigMoney) {
            return legal.test(A_PASS) ? A_PASS : first_legal(legal);
        }

        if (strategy == EngineV3Strategy::Rush) {
            const Action workshop = play_action(DEF_WORKSHOP);
            if (legal.test(workshop)) {
                return workshop;
            }
            return legal.test(A_PASS) ? A_PASS : first_legal(legal);
        }

        Action action = A_PASS;
        const EngineV3Kingdom kingdom = analyze_v3_kingdom(state);
        const DeckProfile profile = analyze_deck(state, state.decision.player);
        if (v3_is_engine_strategy(strategy)
            || (strategy == EngineV3Strategy::BigMoneyPlusAction
                && kingdom.bandit_ok && !kingdom.militia_ok && !kingdom.witch_ok
                && profile.chapels > 0U)) {
            action = v3_engine_play_action(state, legal, strategy);
        } else {
            action = best_legal_play_action(legal, false);
            if (action == A_PASS) {
                action = legal.test(A_PASS) ? A_PASS : first_legal(legal);
            }
        }
        if (action != A_PASS && action_is_play(action)
            && action_def(action, A_PLAY_BASE) == DEF_CHAPEL
            && state.decision.player < MAX_PLAYERS
            && chapel_plays[state.decision.player] < 255U) {
            ++chapel_plays[state.decision.player];
        }
        return action;
    }
    if (decision == DecisionKind::PhaseBuy) {
        const Action treasure = first_legal_play_treasure(legal);
        if (treasure != A_PASS) {
            return treasure;
        }
        const PlayerId player = state.decision.player;
        bool fallback_greening = false;
        bool fallback_vp_bought = false;
        bool& started = player < MAX_PLAYERS
            ? greening_started[player] : fallback_greening;
        bool& vp_bought = player < MAX_PLAYERS
            ? green_vp_bought[player] : fallback_vp_bought;
        if (player < MAX_PLAYERS
            && green_buy_turn[player] != static_cast<std::uint32_t>(state.turn_counter)) {
            green_buy_turn[player] = static_cast<std::uint32_t>(state.turn_counter);
            vp_bought = false;
        }
        const Action action = v3_buy(state, legal, started, vp_bought);
        if (started && action >= A_BUY_BASE && action < A_EVENT_BASE) {
            const DefId def = action_def(action, A_BUY_BASE);
            if (is_victory_def(def)) {
                vp_bought = true;
            }
        }
        return action;
    }
    if (decision == DecisionKind::PhaseNight) {
        return legal.test(A_PASS) ? A_PASS : first_legal(legal);
    }
    if (decision == DecisionKind::Choose && state.decision.source == DEF_CHAPEL) {
        return v3_chapel_trash(state, legal, state.decision.player);
    }
    if (decision == DecisionKind::ChooseOption && state.decision.source == DEF_SENTRY) {
        return v3_sentry_option(state, legal);
    }
    if (decision == DecisionKind::ChooseGain && state.decision.source == DEF_WORKSHOP
        && pick_v3_strategy(analyze_v3_kingdom(state)) == EngineV3Strategy::Rush) {
        return v3_rush_gain(legal);
    }
    return choose_generic_subdecision(state, legal, true);
}

Action ThinnerBot::choose_action(
    const GameState& state,
    const ActionMask& legal,
    int legal_count) const noexcept {
    if (legal_count <= 0) {
        return A_PASS;
    }

    const DecisionKind decision = static_cast<DecisionKind>(state.decision.kind);
    const PlayerId player = state.decision.player;
    if (decision == DecisionKind::PhaseAction) {
        const DeckProfile profile = analyze_deck(state, player);
        return thinner_engine_play_action(state, legal, profile, player);
    }
    if (decision == DecisionKind::PhaseBuy) {
        const Action treasure = first_legal_play_treasure(legal);
        if (treasure != A_PASS) {
            return treasure;
        }
        return thinner_buy(state, legal, engine_buy_policy);
    }
    if (decision == DecisionKind::PhaseNight) {
        return legal.test(A_PASS) ? A_PASS : first_legal(legal);
    }
    if (decision == DecisionKind::ReactWindow) {
        const Action moat = select_action(DEF_MOAT);
        return legal.test(moat) ? moat : (legal.test(A_PASS) ? A_PASS : first_legal(legal));
    }
    if (decision == DecisionKind::Choose && state.decision.source == DEF_CHAPEL) {
        return thinner_chapel_trash(state, legal, player);
    }
    if (decision == DecisionKind::Choose && state.decision.source == DEF_MONEYLENDER) {
        const DeckProfile profile = analyze_deck(state, player);
        const Action copper = select_action(DEF_COPPER);
        if (legal.test(copper) && thinner_trash_eligible(state, profile, DEF_COPPER)) {
            return copper;
        }
        return legal.test(A_PASS) ? A_PASS : first_legal(legal);
    }
    if (decision == DecisionKind::ChooseOption && state.decision.source == DEF_SENTRY) {
        return thinner_sentry_option(state, legal);
    }
    return choose_generic_subdecision(state, legal, true);
}
