#include "v2/mcts/tree.h"

#include "v2/mcts/pile_clock.h"

#include "v2/core/determinize.h"
#include "v2/core/game.h"
#include "v2/core/score.h"
#include "v2/core/turns.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace {

[[nodiscard]] bool terminal_state(const GameState& state) noexcept {
    return state.phase == static_cast<std::uint8_t>(Phase::Over);
}

[[nodiscard]] PlayerId other_player(PlayerId player) noexcept {
    return static_cast<PlayerId>(player == 0U ? 1U : 0U);
}

[[nodiscard]] std::uint8_t safe_determinizations(const MctsConfig& config) noexcept {
    return config.determinizations == 0U ? 1U : config.determinizations;
}

[[nodiscard]] std::uint32_t safe_capacity(const MctsConfig& config) noexcept {
    return config.max_tree_nodes == 0U ? 1U : config.max_tree_nodes;
}

[[nodiscard]] float uniform_float(Xoshiro256pp& rng) noexcept {
    const double value = static_cast<double>(rng.next() >> 11U) * 0x1.0p-53;
    return static_cast<float>(value <= 0.0 ? 0x1.0p-24 : value);
}

[[nodiscard]] double normal01(Xoshiro256pp& rng) noexcept {
    const double u1 = static_cast<double>(uniform_float(rng));
    const double u2 = static_cast<double>(uniform_float(rng));
    constexpr double kPi = 3.141592653589793238462643383279502884;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
}

[[nodiscard]] double gamma_sample(double alpha, Xoshiro256pp& rng) noexcept {
    if (alpha <= 0.0) {
        return 0.0;
    }
    if (alpha < 1.0) {
        const double boosted = gamma_sample(alpha + 1.0, rng);
        return boosted * std::pow(static_cast<double>(uniform_float(rng)), 1.0 / alpha);
    }

    const double d = alpha - (1.0 / 3.0);
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (int attempt = 0; attempt < 32; ++attempt) {
        const double x = normal01(rng);
        const double v_base = 1.0 + (c * x);
        if (v_base <= 0.0) {
            continue;
        }
        const double v = v_base * v_base * v_base;
        const double u = static_cast<double>(uniform_float(rng));
        if (u < 1.0 - (0.0331 * x * x * x * x)) {
            return d * v;
        }
        if (std::log(u) < (0.5 * x * x) + (d * (1.0 - v + std::log(v)))) {
            return d * v;
        }
    }
    return alpha;
}

[[nodiscard]] float positive_prior(
    const MctsConfig& config,
    const GameState& state,
    PlayerId player,
    Action action) noexcept {
    if (config.prior_fn == nullptr) {
        return 1.0F;
    }
    const float prior = config.prior_fn(state, player, action, config.prior_user);
    return prior > 0.0F ? prior : 0.0F;
}

[[nodiscard]] DefId def_for_slot(const GameState& state, Slot slot) noexcept {
    return slot < state.num_slots ? state.slot_to_def[slot] : DEF_COPPER;
}

[[nodiscard]] bool is_action_def(DefId def) noexcept {
    return def < card_def_count() && (card_def(def).types & TYPE_ACTION) != 0U;
}

[[nodiscard]] bool is_treasure_def(DefId def) noexcept {
    return def < card_def_count() && (card_def(def).types & TYPE_TREASURE) != 0U;
}

[[nodiscard]] int action_mask_count(const ActionMask& legal) noexcept {
    int count = 0;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        count += legal.test(action) ? 1 : 0;
    }
    return count;
}

[[nodiscard]] PlayerId player_to_move(const GameState& state) noexcept {
    if (state.decision.player < state.num_players) {
        return state.decision.player;
    }
    return current_player(state);
}

[[nodiscard]] bool is_victory_def(DefId def) noexcept {
    return def < card_def_count() && (card_def(def).types & TYPE_VICTORY) != 0U;
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

struct CardYield {
    std::int8_t actions = 0;
    std::int8_t cards = 0;
    std::int8_t coins = 0;
};

struct RolloutDeckProfile {
    std::uint8_t counts[BASIC_CARD_COUNT]{};
    std::uint16_t deck_size = 0;
    std::uint16_t chapels = 0;
    std::uint16_t sentries = 0;
    std::int16_t total_money = 0;
    std::int16_t total_plus_coins = 0;
};

struct KingdomAnalysis {
    bool has_trasher = false;
    bool has_militia_or_bandit = false;
    bool has_action_value = false;
    DefId best_action = NONE;
};

[[nodiscard]] bool kingdom_has_available(const GameState& state, DefId def) noexcept {
    return supply_count(state, def) > 0;
}

void add_count(RolloutDeckProfile& profile, DefId def, std::uint8_t amount) noexcept {
    if (def >= BASIC_CARD_COUNT || amount == 0U) {
        return;
    }
    const std::uint16_t next = static_cast<std::uint16_t>(profile.counts[def] + amount);
    profile.counts[def] = next > 255U ? 255U : static_cast<std::uint8_t>(next);
    profile.deck_size = static_cast<std::uint16_t>(profile.deck_size + amount);
}

void count_ordered_zone(const GameState& state, const OrderedZone& zone, RolloutDeckProfile& profile) noexcept {
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

[[nodiscard]] RolloutDeckProfile analyze_deck(const GameState& state, PlayerId player) noexcept {
    RolloutDeckProfile profile{};
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
        if ((card.types & TYPE_TREASURE) != 0U) {
            profile.total_money = static_cast<std::int16_t>(
                profile.total_money + (card.coin_value * count));
        }
        if ((card.types & TYPE_ACTION) != 0U) {
            const CardYield yield = card_yield(def);
            profile.total_plus_coins = static_cast<std::int16_t>(
                profile.total_plus_coins + (yield.coins * count));
        }
        if (def == DEF_CHAPEL) {
            profile.chapels = static_cast<std::uint16_t>(profile.chapels + count);
        }
        if (def == DEF_SENTRY) {
            profile.sentries = static_cast<std::uint16_t>(profile.sentries + count);
        }
    }
    return profile;
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
        return 31;
    case DEF_SMITHY:
        return 32;
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

[[nodiscard]] Action best_legal_play_action(const ActionMask& legal) noexcept {
    Action best = A_PASS;
    int best_priority = 999;
    for (DefId def = 0; def < ACTION_DEF_COUNT; ++def) {
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

[[nodiscard]] int gain_priority(DefId def) noexcept {
    switch (def) {
    case DEF_PROVINCE:
        return 0;
    case DEF_GOLD:
        return 10;
    case DEF_DUCHY:
        return 20;
    case DEF_SILVER:
        return 30;
    case DEF_ESTATE:
        return 40;
    case DEF_COPPER:
    case DEF_CURSE:
        return 900;
    default:
        return 100;
    }
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
        const int priority = gain_priority(def);
        if (best == A_PASS || cost > best_cost || (cost == best_cost && priority < best_priority)) {
            best = action;
            best_cost = cost;
            best_priority = priority;
        }
    }
    return best != A_PASS ? best : (legal.test(A_PASS) ? A_PASS : first_legal(legal));
}

[[nodiscard]] Action big_money_buy(const GameState& state, const ActionMask& legal) noexcept {
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
        if (def == DEF_CHAPEL || def == DEF_SENTRY) {
            analysis.has_trasher = true;
        }
        if (def == DEF_MILITIA || def == DEF_BANDIT) {
            analysis.has_militia_or_bandit = true;
        }
        if (is_action_def(def)) {
            analysis.has_action_value = true;
        }
    }

    for (const DefId def : kBestActions) {
        if (kingdom_has_available(state, def)) {
            analysis.best_action = def;
            break;
        }
    }
    return analysis;
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

[[nodiscard]] Action engine_like_buy(const GameState& state, const ActionMask& legal) noexcept {
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
    if (!kingdom.has_action_value) {
        return big_money_buy(state, legal);
    }

    const PlayerId player = state.decision.player;
    const RolloutDeckProfile profile = analyze_deck(state, player);
    const double deck_size = profile.deck_size > 0U ? static_cast<double>(profile.deck_size) : 1.0;
    const double money_density = static_cast<double>(profile.total_money + profile.total_plus_coins) / deck_size;
    const int my_turns = state.num_players == 0U ? 0 : static_cast<int>(state.turn_counter / state.num_players);
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
        constexpr DefId kCantripBuys[] = {DEF_LABORATORY, DEF_MARKET, DEF_FESTIVAL};
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

[[nodiscard]] Action sentry_option(const GameState& state, const ActionMask& legal) noexcept {
    DefId def = DEF_COPPER;
    if (state.effect_depth > 0U) {
        const EffectFrame& frame = state.effect_stack[state.effect_depth - 1U];
        if (frame.source == DEF_SENTRY && frame.data[3] > 0 && frame.data[4] < frame.data[3]) {
            const std::uint8_t index = static_cast<std::uint8_t>(frame.data[4]);
            const Slot slot = static_cast<Slot>(frame.data[1U + index]);
            def = def_for_slot(state, slot);
        }
    }

    Action desired = option_action(2U);
    if (def == DEF_CURSE || def == DEF_ESTATE || def == DEF_COPPER) {
        desired = option_action(0U);
    } else if (def == DEF_DUCHY || def == DEF_PROVINCE || def == DEF_COLONY) {
        desired = option_action(1U);
    }
    return legal.test(desired) ? desired : first_legal_option(legal);
}

[[nodiscard]] Action rollout_random_action(
    const ActionMask& legal,
    int legal_count,
    Xoshiro256pp& rng) noexcept {
    if (legal_count <= 0) {
        return A_PASS;
    }
    return legal.nth_set(rng.uniform(static_cast<std::uint32_t>(legal_count)));
}

[[nodiscard]] bool epsilon_explore(float epsilon, Xoshiro256pp& rng) noexcept {
    if (epsilon <= 0.0F) {
        return false;
    }
    if (epsilon > 1.0F) {
        epsilon = 1.0F;
    }
    const std::uint32_t threshold = static_cast<std::uint32_t>(epsilon * 10000.0F);
    return rng.uniform(10000U) < threshold;
}

[[nodiscard]] Action rollout_policy_action(
    const GameState& state,
    const ActionMask& legal,
    int legal_count,
    Xoshiro256pp& rng,
    float epsilon,
    MctsRolloutPolicy policy) noexcept {
    if (legal_count <= 0) {
        return A_PASS;
    }
    if (legal_count == 1) {
        return legal.nth_set(0U);
    }

    // Local scripted rollout policy. This intentionally duplicates only
    // compact driver bot choices so MCTS keeps no dependency on drivers.
    const DecisionKind decision = static_cast<DecisionKind>(state.decision.kind);
    PileClock pile_clock{};
    bool has_pile_clock = false;
    if (decision == DecisionKind::PhaseBuy && policy == MctsRolloutPolicy::EngineLike) {
        pile_clock = analyze_pile_clock(state, legal);
        has_pile_clock = true;
        if (pile_clock.empty_piles >= 2) {
            // Keep playing forced treasure value, but never let epsilon turn a
            // live third-pile threat into an arbitrary random purchase.
            const Action treasure = first_legal_play_treasure(legal);
            if (treasure != A_PASS) {
                return treasure;
            }
            return pile_clock_guarded_buy(state, legal, pile_clock, engine_like_buy);
        }
    }
    if (epsilon_explore(epsilon, rng)) {
        return rollout_random_action(legal, legal_count, rng);
    }

    if (decision == DecisionKind::PhaseBuy) {
        const Action treasure = first_legal_play_treasure(legal);
        if (treasure != A_PASS) {
            return treasure;
        }
        if (policy == MctsRolloutPolicy::EngineLike) {
            return has_pile_clock
                ? pile_clock_guarded_buy(state, legal, pile_clock, engine_like_buy)
                : mcts_engine_like_rollout_buy(state, legal);
        }
        return big_money_buy(state, legal);
    }
    if (decision == DecisionKind::PhaseAction) {
        const Action action = best_legal_play_action(legal);
        if (action != A_PASS) {
            return action;
        }
        return legal.test(A_PASS) ? A_PASS : first_legal(legal);
    }
    if (decision == DecisionKind::PhaseNight) {
        return legal.test(A_PASS) ? A_PASS : first_legal(legal);
    }
    if (decision == DecisionKind::ReactWindow) {
        const Action moat = select_action(DEF_MOAT);
        return legal.test(moat) ? moat : (legal.test(A_PASS) ? A_PASS : first_legal(legal));
    }
    if (decision == DecisionKind::OrderTriggers || decision == DecisionKind::ChooseOrder) {
        return first_legal_option(legal);
    }
    if (decision == DecisionKind::ChooseOption) {
        if (state.decision.source == DEF_SENTRY) {
            return sentry_option(state, legal);
        }
        if (state.decision.source == DEF_LIBRARY) {
            const Action set_aside = option_action(1U);
            if (policy == MctsRolloutPolicy::EngineLike && legal.test(set_aside)) {
                return set_aside;
            }
        }
        if (state.decision.source == DEF_VASSAL) {
            const Action play = option_action(1U);
            if (legal.test(play)) {
                return play;
            }
        }
        const Action decline = option_action(0U);
        return legal.test(decline) ? decline : first_legal_option(legal);
    }
    if (decision == DecisionKind::ChooseGain) {
        return choose_gain_most_expensive(legal);
    }
    if (decision == DecisionKind::Choose) {
        const bool pass_allowed = legal.test(A_PASS) && state.decision.min_left == 0U;
        const DefId source = static_cast<DefId>(state.decision.source);
        if (source == DEF_MILITIA) {
            return choose_select_max_priority(legal, keep_priority, false);
        }
        if (source == DEF_CELLAR || source == DEF_POACHER) {
            return choose_select_min_priority(legal, discard_priority, pass_allowed);
        }
        if (source == DEF_CHAPEL || source == DEF_REMODEL || source == DEF_MINE
            || source == DEF_MONEYLENDER || source == DEF_BANDIT) {
            return choose_select_min_priority(legal, trash_priority, pass_allowed);
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
        return pass_allowed ? A_PASS : first_legal(legal);
    }

    return legal.test(A_PASS) && state.decision.min_left == 0U ? A_PASS : first_legal(legal);
}

} // namespace

std::uint64_t mcts_state_hash(const GameState& state) noexcept {
    constexpr std::uint64_t kOffset = 14695981039346656037ULL;
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&state);
    std::uint64_t hash = kOffset;
    for (std::size_t i = 0; i < sizeof(GameState); ++i) {
        hash ^= bytes[i];
        hash *= kPrime;
    }
    return hash;
}

ActionMask mcts_top_k_actions(
    const ActionMask& legal,
    int legal_count,
    const float* priors,
    std::uint8_t top_k,
    Xoshiro256pp& rng) noexcept {
    if (top_k == 0U) {
        return legal;
    }

    const int actual_count = action_mask_count(legal);
    if (actual_count <= 0 || actual_count != legal_count || priors == nullptr) {
        // A malformed mask/count/prior input must never empty an expansion.
        return legal;
    }
    const int retained_count = std::max(2, static_cast<int>(top_k));
    if (actual_count <= retained_count) {
        return legal;
    }

    ActionMask retained{};
    const int ranked_count = retained_count - 1;
    for (int selected = 0; selected < ranked_count; ++selected) {
        Action best = A_PASS;
        float best_prior = -1.0F;
        bool found = false;
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            if (!legal.test(action) || retained.test(action)) {
                continue;
            }
            if (!std::isfinite(priors[action])) {
                // Priors are probabilities; a non-finite score makes their
                // ordering unreliable, so preserve the unfiltered legal set.
                return legal;
            }
            const float prior = priors[action] > 0.0F ? priors[action] : 0.0F;
            if (!found || prior > best_prior
                || (prior == best_prior && action < best)) {
                best = action;
                best_prior = prior;
                found = true;
            }
        }
        if (!found) {
            // NaN/Inf or another inconsistency: retain the established full
            // legal set rather than making a legal move unselectable.
            return legal;
        }
        retained.set(best);
    }
    const int excluded_count = actual_count - ranked_count;
    if (excluded_count <= 0) {
        return legal;
    }
    const int wildcard_index = static_cast<int>(rng.uniform(
        static_cast<std::uint32_t>(excluded_count)));
    Action wildcard = A_PASS;
    bool wildcard_found = false;
    int seen = 0;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (!legal.test(action) || retained.test(action)) {
            continue;
        }
        if (seen == wildcard_index) {
            wildcard = action;
            wildcard_found = true;
            break;
        }
        ++seen;
    }
    if (!wildcard_found || !legal.test(wildcard) || retained.test(wildcard)) {
        return legal;
    }
    retained.set(wildcard);
    return action_mask_count(retained) == retained_count ? retained : legal;
}

void mcts_add_dirichlet_noise(
    float* priors,
    const ActionMask& actions,
    int action_count,
    float alpha,
    float frac,
    Xoshiro256pp& rng) noexcept {
    if (priors == nullptr || frac <= 0.0F || action_count <= 1) {
        return;
    }

    float noise[ACTION_SPACE_SIZE]{};
    double noise_sum = 0.0;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (!actions.test(action)) {
            continue;
        }
        const double sample = gamma_sample(static_cast<double>(alpha), rng);
        noise[action] = static_cast<float>(sample);
        noise_sum += sample;
    }
    if (noise_sum <= 0.0) {
        return;
    }
    if (frac > 1.0F) {
        frac = 1.0F;
    }
    const float keep = 1.0F - frac;
    const float noise_scale = static_cast<float>(frac / noise_sum);
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (actions.test(action)) {
            priors[action] = (priors[action] * keep) + (noise[action] * noise_scale);
        }
    }
}

Action mcts_engine_like_rollout_buy(
    const GameState& state,
    const ActionMask& legal) noexcept {
    return pile_clock_guarded_buy(state, legal, analyze_pile_clock(state, legal), engine_like_buy);
}

ActionMask mcts_filter_treasure_plays(
    const GameState& state,
    const ActionMask& legal) noexcept {
    bool has_legal_treasure = false;
    for (DefId def = 0; def < ACTION_DEF_COUNT; ++def) {
        if (is_treasure_def(def) && legal.test(play_action(def))) {
            has_legal_treasure = true;
            break;
        }
    }
    if (!has_legal_treasure) {
        return legal;
    }

    /*
     * The engine's legal mask adds treasure plays and buys only in Phase::Buy;
     * Phase::Action contains A_PASS and action-card plays. Therefore a legal
     * treasure play identifies the buy-phase coexistence this filter targets.
     */
    DefId highest_in_play_treasure = 0;
    bool has_in_play_treasure = false;
    const PlayerId player = player_to_move(state);
    if (player < state.num_players) {
        const PlayerState& player_state = state.players[player];
        for (std::uint8_t i = 0; i < player_state.in_play_size; ++i) {
            const Slot slot = player_state.in_play[i].slot;
            if (slot >= state.num_slots) {
                continue;
            }
            const DefId def = state.slot_to_def[slot];
            if (is_treasure_def(def)
                && (!has_in_play_treasure || def > highest_in_play_treasure)) {
                highest_in_play_treasure = def;
                has_in_play_treasure = true;
            }
        }
    }

    ActionMask filtered{};
    for (DefId def = 0; def < ACTION_DEF_COUNT; ++def) {
        const Action action = play_action(def);
        if (!is_treasure_def(def) || !legal.test(action)) {
            continue;
        }
        if (!has_in_play_treasure || def >= highest_in_play_treasure) {
            filtered.set(action);
        }
    }

    // Defensive fallback: legal treasure plays should always leave a move.
    return action_mask_count(filtered) > 0 ? filtered : legal;
}

Action mcts_canonical_treasure_play(const ActionMask& legal) noexcept {
    for (DefId def = 0; def < ACTION_DEF_COUNT; ++def) {
        const Action action = play_action(def);
        if (is_treasure_def(def) && legal.test(action)) {
            return action;
        }
    }
    return A_PASS;
}

Mcts::Mcts(const MctsConfig& config)
    : config_(config),
      nodes_(new MctsNode[safe_capacity(config)]),
      states_(new GameState[safe_capacity(config)]),
      remap_(new std::uint32_t[safe_capacity(config)]),
      capacity_(safe_capacity(config)) {
    if (config_.tree_reuse && safe_determinizations(config_) != 1U) {
        // Each root-sampled tree encodes a different hidden-information
        // world, so a retained child cannot safely represent their aggregate.
        throw std::invalid_argument("MctsConfig.tree_reuse requires determinizations == 1");
    }
}

Action Mcts::choose(const GameState& root, PlayerId perspective) noexcept {
    ActionMask root_legal{};
    int root_count = Game::legal_actions(root, root_legal);
    if (config_.prune_treasure_plays) {
        root_legal = mcts_filter_treasure_plays(root, root_legal);
        root_count = action_mask_count(root_legal);
    }
    if (root_count <= 0) {
        return A_PASS;
    }

    float action_value[ACTION_SPACE_SIZE]{};
    std::uint32_t action_visits[ACTION_SPACE_SIZE]{};
    std::uint32_t action_seen[ACTION_SPACE_SIZE]{};

    const std::uint8_t dets = safe_determinizations(config_);
    const std::uint32_t total_sims = config_.sims_per_move == 0U ? 1U : config_.sims_per_move;
    const std::uint32_t base_sims = total_sims / dets;
    const std::uint32_t extra_sims = total_sims % dets;

    /*
     * Root sampling: each determinization gets its own tree, then root child
     * statistics are averaged. This keeps hidden-information worlds internally
     * consistent and avoids mixing incompatible private card orders below root.
     */
    for (std::uint8_t det = 0; det < dets; ++det) {
        GameState sampled = root;
        determinize(sampled, perspective, config_.rollout_seed + (0x9E37'79B9U * static_cast<std::uint64_t>(det + 1U)));
        reset(sampled, perspective);

        Xoshiro256pp rng = Xoshiro256pp::seeded(
            config_.rollout_seed ^ (0xD1B5'4A32'D192'ED03ULL * static_cast<std::uint64_t>(det + 1U)));
        const std::uint32_t sims = base_sims + (det < extra_sims ? 1U : 0U);
        run_simulations(sims == 0U ? 1U : sims, rng);

        const MctsNode& root_node = nodes_[0];
        for (std::uint32_t child_index = root_node.first_child; child_index != MCTS_NULL;
             child_index = nodes_[child_index].next_sibling) {
            const MctsNode& child = nodes_[child_index];
            const Action action = child.action_from_parent;
            if (action >= ACTION_SPACE_SIZE) {
                continue;
            }
            const std::uint32_t visits = child.visits;
            const float value = child_value_for_parent(root_node, child);
            action_value[action] += value * static_cast<float>(visits);
            action_visits[action] += visits;
            action_seen[action] += 1U;
        }
    }

    Action best = root_legal.nth_set(0U);
    std::uint32_t best_visits = 0;
    float best_value = -2.0F;
    for (int i = 0; i < root_count; ++i) {
        const Action action = root_legal.nth_set(static_cast<std::uint32_t>(i));
        const std::uint32_t visits = action_visits[action];
        const float value = visits == 0U
            ? -2.0F
            : action_value[action] / static_cast<float>(visits);
        if (visits > best_visits || (visits == best_visits && value > best_value)) {
            best = action;
            best_visits = visits;
            best_value = value;
        } else if (visits == 0U && best_visits == 0U && action_seen[action] > action_seen[best]) {
            best = action;
        }
    }
    return best;
}

void Mcts::reset(const GameState& root, PlayerId perspective) noexcept {
    clear_retained_root();
    node_count_ = 0;
    exhausted_ = false;
    root_perspective_ = perspective;
    expansion_rng_ = Xoshiro256pp::seeded(
        config_.rollout_seed ^ mcts_state_hash(root) ^ 0xE7A1'0F5E'BADC'0DEULL);
    const std::uint32_t root_index = allocate_node();
    assert(root_index == 0U);
    (void)root_index;
    states_[0] = root;
    MctsNode& root_node = nodes_[0];
    root_node.state_index = 0U;
    root_node.player = terminal_state(root) ? perspective : current_player(root);
    root_node.terminal = terminal_state(root);
}

bool Mcts::retain_root_child(Action action, std::uint64_t post_action_hash) noexcept {
    clear_retained_root();
    if (!config_.tree_reuse || safe_determinizations(config_) != 1U || node_count_ == 0U) {
        return false;
    }

    for (std::uint32_t child_index = nodes_[0].first_child; child_index != MCTS_NULL;
         child_index = nodes_[child_index].next_sibling) {
        const MctsNode& child = nodes_[child_index];
        if (child.action_from_parent != action) {
            continue;
        }
        // Game::step() in the runner and expansion must agree before a child
        // can become a candidate. This also protects unusual scripted paths.
        if (mcts_state_hash(states_[child.state_index]) != post_action_hash) {
            return false;
        }
        retained_root_ = child_index;
        retained_root_state_hash_ = post_action_hash;
        return true;
    }
    return false;
}

bool Mcts::adopt_retained_root(const GameState& root, PlayerId perspective) noexcept {
    const std::uint32_t retained_root = retained_root_;
    const std::uint64_t retained_hash = retained_root_state_hash_;
    clear_retained_root();

    if (safe_determinizations(config_) != 1U || retained_root == MCTS_NULL
        || retained_root >= node_count_) {
        return false;
    }
    // Re-rooting occurs only between batched searches on the main thread.
    // A parked external leaf would otherwise retain a path into the old tree.
    assert(total_virtual_loss() == 0.0F);
    if (mcts_state_hash(root) != retained_hash
        || mcts_state_hash(states_[nodes_[retained_root].state_index]) != retained_hash) {
        return false;
    }
    return compact_subtree_to_root(retained_root, perspective);
}

void Mcts::clear_retained_root() noexcept {
    retained_root_ = MCTS_NULL;
    retained_root_state_hash_ = 0U;
}

void Mcts::set_root_priors(const float* priors) noexcept {
    if (priors == nullptr || node_count_ == 0U) {
        return;
    }

    float sum = 0.0F;
    std::uint32_t count = 0U;
    for (std::uint32_t child_index = nodes_[0].first_child; child_index != MCTS_NULL;
         child_index = nodes_[child_index].next_sibling) {
        MctsNode& child = nodes_[child_index];
        const float prior = priors[child.action_from_parent];
        child.prior = std::isfinite(prior) && prior > 0.0F ? prior : 0.0F;
        sum += child.prior;
        ++count;
    }
    if (count == 0U) {
        return;
    }
    if (sum <= 0.0F) {
        const float uniform = 1.0F / static_cast<float>(count);
        for (std::uint32_t child_index = nodes_[0].first_child; child_index != MCTS_NULL;
             child_index = nodes_[child_index].next_sibling) {
            nodes_[child_index].prior = uniform;
        }
        return;
    }
    const float inv_sum = 1.0F / sum;
    for (std::uint32_t child_index = nodes_[0].first_child; child_index != MCTS_NULL;
         child_index = nodes_[child_index].next_sibling) {
        nodes_[child_index].prior *= inv_sum;
    }
}

void Mcts::run_simulations(std::uint32_t simulations, Xoshiro256pp& rng) noexcept {
    if (node_count_ == 0U) {
        return;
    }

    for (std::uint32_t sim = 0; sim < simulations; ++sim) {
        std::uint32_t path[MCTS_MAX_PATH]{};
        std::uint8_t depth = 0;
        std::uint32_t node_index = 0U;
        path[depth++] = node_index;

        while (nodes_[node_index].expanded && nodes_[node_index].first_child != MCTS_NULL
               && !nodes_[node_index].terminal) {
            node_index = select_child(node_index);
            path[depth++] = node_index;
            if (depth >= MCTS_MAX_PATH) {
                break;
            }
        }

        MctsNode& leaf = nodes_[node_index];
        if (!leaf.terminal && !leaf.expanded) {
            const bool expanded = expand(node_index, rng);
            if (expanded && leaf.first_child != MCTS_NULL) {
                node_index = select_child(node_index);
                if (depth < MCTS_MAX_PATH) {
                    path[depth++] = node_index;
                }
            }
        }

        GameState terminal = states_[nodes_[node_index].state_index];
        if (!terminal_state(terminal)) {
            rollout(terminal, rng);
        }
        backpropagate(path, depth, terminal);
    }
}

void Mcts::add_virtual_loss(std::uint32_t node, float amount) noexcept {
    if (node >= node_count_) {
        return;
    }
    nodes_[node].virtual_loss += amount;
}

void Mcts::revert_virtual_loss(std::uint32_t node, float amount) noexcept {
    if (node >= node_count_) {
        return;
    }
    nodes_[node].virtual_loss -= amount;
    if (nodes_[node].virtual_loss < 0.0F) {
        nodes_[node].virtual_loss = 0.0F;
    }
}

const MctsNode& Mcts::node(std::uint32_t index) const noexcept {
    assert(index < node_count_);
    return nodes_[index];
}

std::uint32_t Mcts::node_count() const noexcept {
    return node_count_;
}

bool Mcts::exhausted() const noexcept {
    return exhausted_;
}

Action Mcts::best_root_action() const noexcept {
    if (node_count_ == 0U) {
        return A_PASS;
    }

    const MctsNode& root = nodes_[0];
    Action best = A_PASS;
    std::uint32_t best_visits = 0;
    float best_value = -2.0F;
    for (std::uint32_t child_index = root.first_child; child_index != MCTS_NULL;
         child_index = nodes_[child_index].next_sibling) {
        const MctsNode& child = nodes_[child_index];
        const float value = child_value_for_parent(root, child);
        if (child.visits > best_visits || (child.visits == best_visits && value > best_value)) {
            best = child.action_from_parent;
            best_visits = child.visits;
            best_value = value;
        }
    }
    return best;
}

std::uint32_t Mcts::root_visits_for(Action action) const noexcept {
    if (node_count_ == 0U) {
        return 0U;
    }
    for (std::uint32_t child_index = nodes_[0].first_child; child_index != MCTS_NULL;
         child_index = nodes_[child_index].next_sibling) {
        if (nodes_[child_index].action_from_parent == action) {
            return nodes_[child_index].visits;
        }
    }
    return 0U;
}

float Mcts::root_value_for(Action action) const noexcept {
    if (node_count_ == 0U) {
        return 0.0F;
    }
    const MctsNode& root = nodes_[0];
    for (std::uint32_t child_index = root.first_child; child_index != MCTS_NULL;
         child_index = nodes_[child_index].next_sibling) {
        const MctsNode& child = nodes_[child_index];
        if (child.action_from_parent == action) {
            return child_value_for_parent(root, child);
        }
    }
    return 0.0F;
}

const GameState& Mcts::state_for(std::uint32_t state_index) const noexcept {
    assert(state_index < capacity_);
    return states_[state_index < capacity_ ? state_index : 0U];
}

bool Mcts::collect_external_leaf(MctsPendingLeaf& leaf) noexcept {
    leaf = MctsPendingLeaf{};
    if (node_count_ == 0U) {
        return false;
    }

    std::uint32_t node_index = 0U;
    std::uint32_t path[MCTS_MAX_PATH]{};
    std::uint8_t depth = 0;
    path[depth++] = node_index;

    while (nodes_[node_index].expanded && nodes_[node_index].first_child != MCTS_NULL
           && !nodes_[node_index].terminal) {
        node_index = select_child(node_index);
        if (depth < MCTS_MAX_PATH) {
            path[depth++] = node_index;
        } else {
            break;
        }
    }

    MctsNode& node = nodes_[node_index];
    const GameState& state = states_[node.state_index];
    if (node.terminal || terminal_state(state)) {
        node.terminal = true;
        backpropagate(path, depth, state);
        return false;
    }

    if (node.expanded && node.first_child == MCTS_NULL) {
        node.terminal = true;
        backpropagate(path, depth, state);
        return false;
    }

    ActionMask legal{};
    int legal_count = Game::legal_actions(state, legal);
    if (config_.prune_treasure_plays) {
        legal = mcts_filter_treasure_plays(state, legal);
        legal_count = action_mask_count(legal);
    }
    if (legal_count <= 0) {
        node.terminal = true;
        backpropagate(path, depth, state);
        return false;
    }

    leaf.node = node_index;
    leaf.state_index = node.state_index;
    for (std::uint8_t i = 0; i < depth; ++i) {
        leaf.path[i] = path[i];
    }
    leaf.depth = depth;
    leaf.player = node.player;
    leaf.legal = legal;
    leaf.legal_count = legal_count;
    leaf.valid = true;
    apply_virtual_loss(path, depth, 1.0F);
    return true;
}

void Mcts::provide_external_evaluation(
    const MctsPendingLeaf& leaf,
    float value,
    const float* priors) noexcept {
    provide_external_evaluation(leaf, value, priors, expansion_rng_);
}

void Mcts::provide_external_evaluation(
    const MctsPendingLeaf& leaf,
    float value,
    const float* priors,
    Xoshiro256pp& rng) noexcept {
    if (!leaf.valid || leaf.node >= node_count_) {
        return;
    }
    if (value > 1.0F) {
        value = 1.0F;
    } else if (value < -1.0F) {
        value = -1.0F;
    }

    MctsNode& node = nodes_[leaf.node];
    if (!node.expanded && !node.terminal) {
        (void)expand_with_priors(leaf.node, priors, rng);
    }
    apply_virtual_loss(leaf.path, leaf.depth, -1.0F);
    backpropagate_value(leaf.path, leaf.depth, leaf.player, value);
}

void Mcts::root_visit_policy(float* out, float temperature) const noexcept {
    if (out == nullptr) {
        return;
    }
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        out[action] = 0.0F;
    }
    if (node_count_ == 0U) {
        out[A_PASS] = 1.0F;
        return;
    }

    const MctsNode& root = nodes_[0];
    if (root.first_child == MCTS_NULL) {
        out[A_PASS] = 1.0F;
        return;
    }
    if (temperature <= 0.0F) {
        out[best_root_action()] = 1.0F;
        return;
    }

    const double inv_temp = 1.0 / static_cast<double>(temperature);
    double sum = 0.0;
    for (std::uint32_t child_index = root.first_child; child_index != MCTS_NULL;
         child_index = nodes_[child_index].next_sibling) {
        const MctsNode& child = nodes_[child_index];
        const double weight = child.visits == 0U
            ? 0.0
            : std::pow(static_cast<double>(child.visits), inv_temp);
        out[child.action_from_parent] = static_cast<float>(weight);
        sum += weight;
    }
    if (sum <= 0.0) {
        std::uint32_t count = 0;
        for (std::uint32_t child_index = root.first_child; child_index != MCTS_NULL;
             child_index = nodes_[child_index].next_sibling) {
            ++count;
        }
        const float uniform = count == 0U ? 1.0F : 1.0F / static_cast<float>(count);
        for (std::uint32_t child_index = root.first_child; child_index != MCTS_NULL;
             child_index = nodes_[child_index].next_sibling) {
            out[nodes_[child_index].action_from_parent] = uniform;
        }
        return;
    }
    const float scale = static_cast<float>(1.0 / sum);
    for (std::uint32_t child_index = root.first_child; child_index != MCTS_NULL;
         child_index = nodes_[child_index].next_sibling) {
        out[nodes_[child_index].action_from_parent] *= scale;
    }
}

Action Mcts::sample_root_action(float temperature, Xoshiro256pp& rng) const noexcept {
    float policy[ACTION_SPACE_SIZE]{};
    root_visit_policy(policy, temperature);
    float sample = static_cast<float>(rng.uniform(1'000'000U)) / 1'000'000.0F;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        sample -= policy[action];
        if (sample <= 0.0F) {
            return action;
        }
    }
    return best_root_action();
}

float Mcts::total_virtual_loss() const noexcept {
    float total = 0.0F;
    for (std::uint32_t i = 0; i < node_count_; ++i) {
        total += nodes_[i].virtual_loss;
    }
    return total;
}

std::uint32_t Mcts::allocate_node() noexcept {
    if (node_count_ >= capacity_) {
        assert(false && "MCTS node slab exhausted");
        exhausted_ = true;
        return MCTS_NULL;
    }
    const std::uint32_t index = node_count_;
    nodes_[index] = MctsNode{};
    states_[index] = GameState{};
    ++node_count_;
    return index;
}

bool Mcts::compact_subtree_to_root(
    std::uint32_t retained_root,
    PlayerId perspective) noexcept {
    if (retained_root == MCTS_NULL || retained_root >= node_count_) {
        return false;
    }

    const std::uint32_t old_count = node_count_;
    for (std::uint32_t index = 0U; index < old_count; ++index) {
        remap_[index] = MCTS_NULL;
    }

    // Node indices are allocated parent-before-child. Marking descendants in
    // ascending slab order therefore needs no traversal stack or allocation.
    remap_[retained_root] = 0U;
    std::uint32_t retained_count = 1U;
    for (std::uint32_t source = retained_root + 1U; source < old_count; ++source) {
        const std::uint32_t parent = nodes_[source].parent;
        if (parent < old_count && remap_[parent] != MCTS_NULL) {
            remap_[source] = retained_count;
            ++retained_count;
        }
    }

    // Every destination is below its source because the retained root is a
    // child (never index zero). Ascending copies are consequently safe in the
    // same fixed slabs; no subtree node or GameState is leaked or overwritten
    // before it has been copied.
    for (std::uint32_t source = retained_root; source < old_count; ++source) {
        const std::uint32_t destination = remap_[source];
        if (destination == MCTS_NULL) {
            continue;
        }
        nodes_[destination] = nodes_[source];
        states_[destination] = states_[source];
    }

    const auto mapped = [this, old_count](std::uint32_t index) noexcept {
        return index < old_count ? remap_[index] : MCTS_NULL;
    };
    for (std::uint32_t source = retained_root; source < old_count; ++source) {
        const std::uint32_t destination = remap_[source];
        if (destination == MCTS_NULL) {
            continue;
        }
        MctsNode& node = nodes_[destination];
        node.parent = source == retained_root ? MCTS_NULL : mapped(node.parent);
        node.first_child = mapped(node.first_child);
        node.next_sibling = source == retained_root ? MCTS_NULL : mapped(node.next_sibling);
        node.state_index = destination;
    }

    nodes_[0].action_from_parent = A_PASS;
    // This child was expanded under non-root rules. Re-evaluate it as the
    // next search root so its existing subtree is retained but any missing
    // legal root children are restored at full width.
    if (!nodes_[0].terminal) {
        nodes_[0].expanded = false;
    }
    node_count_ = retained_count;
    root_perspective_ = perspective;
    exhausted_ = false;
    return true;
}

bool Mcts::expand(std::uint32_t node_index, Xoshiro256pp& rng) noexcept {
    if (node_index >= node_count_) {
        return false;
    }

    MctsNode& node = nodes_[node_index];
    if (node.terminal) {
        node.expanded = true;
        return true;
    }

    ActionMask legal{};
    int legal_count = Game::legal_actions(states_[node.state_index], legal);
    if (config_.prune_treasure_plays) {
        legal = mcts_filter_treasure_plays(states_[node.state_index], legal);
        legal_count = action_mask_count(legal);
    }
    if (legal_count <= 0) {
        node.terminal = true;
        node.expanded = true;
        return true;
    }

    const bool use_heuristic_prior = config_.prior_fn == nullptr
        && (config_.rollout_policy == MctsRolloutPolicy::Heuristic
            || config_.rollout_policy == MctsRolloutPolicy::EngineLike)
        && legal_count > 1;
    Action heuristic_prior_action = A_PASS;
    if (use_heuristic_prior) {
        Xoshiro256pp prior_rng = Xoshiro256pp::seeded(0xC0FF'EE01ULL);
        heuristic_prior_action = rollout_policy_action(
            states_[node.state_index],
            legal,
            legal_count,
            prior_rng,
            0.0F,
            config_.rollout_policy);
    }

    float raw_priors[ACTION_SPACE_SIZE]{};
    for (int i = 0; i < legal_count; ++i) {
        const Action action = legal.nth_set(static_cast<std::uint32_t>(i));
        if (use_heuristic_prior) {
            raw_priors[action] = action == heuristic_prior_action
                ? 0.80F
                : 0.20F / static_cast<float>(legal_count - 1);
        } else {
            raw_priors[action] = positive_prior(
                config_, states_[node.state_index], node.player, action);
        }
    }

    // Policy targets and Dirichlet exploration live at the root, so it keeps
    // its full legal width. Interior pruning keeps a random wildcard child so
    // a low-prior move remains searchably reachable.
    ActionMask retained = node_index == 0U
        ? legal
        : mcts_top_k_actions(
            legal,
            legal_count,
            raw_priors,
            config_.expand_top_k,
            rng);
    int retained_count = action_mask_count(retained);
    if (retained_count <= 0) {
        retained = legal;
        retained_count = legal_count;
    }

    float prior_sum = 0.0F;
    for (int i = 0; i < retained_count; ++i) {
        const Action action = retained.nth_set(static_cast<std::uint32_t>(i));
        prior_sum += raw_priors[action];
    }
    if (prior_sum <= 0.0F) {
        prior_sum = static_cast<float>(retained_count);
    }

    ActionMask existing{};
    std::uint32_t previous_child = MCTS_NULL;
    for (std::uint32_t child_index = node.first_child; child_index != MCTS_NULL;
         child_index = nodes_[child_index].next_sibling) {
        const MctsNode& child = nodes_[child_index];
        existing.set(child.action_from_parent);
        previous_child = child_index;
    }
    for (int i = 0; i < retained_count; ++i) {
        const Action action = retained.nth_set(static_cast<std::uint32_t>(i));
        const float prior = raw_priors[action];
        if (existing.test(action)) {
            for (std::uint32_t child_index = node.first_child; child_index != MCTS_NULL;
                 child_index = nodes_[child_index].next_sibling) {
                MctsNode& child = nodes_[child_index];
                if (child.action_from_parent == action) {
                    child.prior = prior > 0.0F
                        ? prior / prior_sum
                        : 1.0F / static_cast<float>(retained_count);
                    break;
                }
            }
            continue;
        }
        const std::uint32_t child_index = allocate_node();
        if (child_index == MCTS_NULL) {
            break;
        }

        GameState child_state = states_[node.state_index];
        const bool done = Game::step(child_state, action);
        states_[child_index] = child_state;

        MctsNode& child = nodes_[child_index];
        child.parent = node_index;
        child.state_index = child_index;
        child.action_from_parent = action;
        child.terminal = done || terminal_state(child_state);
        child.player = next_player_for_child(child_state, node.player);
        child.prior = prior > 0.0F
            ? prior / prior_sum
            : 1.0F / static_cast<float>(retained_count);

        if (previous_child == MCTS_NULL) {
            node.first_child = child_index;
        } else {
            nodes_[previous_child].next_sibling = child_index;
        }
        previous_child = child_index;
    }

    node.expanded = true;
    return node.first_child != MCTS_NULL;
}

bool Mcts::expand_with_priors(
    std::uint32_t node_index,
    const float* priors,
    Xoshiro256pp& rng) noexcept {
    if (node_index >= node_count_) {
        return false;
    }

    MctsNode& node = nodes_[node_index];
    if (node.terminal) {
        node.expanded = true;
        return true;
    }

    ActionMask legal{};
    int legal_count = Game::legal_actions(states_[node.state_index], legal);
    if (config_.prune_treasure_plays) {
        legal = mcts_filter_treasure_plays(states_[node.state_index], legal);
        legal_count = action_mask_count(legal);
    }
    if (legal_count <= 0) {
        node.terminal = true;
        node.expanded = true;
        return true;
    }

    float raw_priors[ACTION_SPACE_SIZE]{};
    bool priors_consistent = true;
    for (int i = 0; i < legal_count; ++i) {
        const Action action = legal.nth_set(static_cast<std::uint32_t>(i));
        if (priors == nullptr) {
            raw_priors[action] = 1.0F;
        } else {
            if (!std::isfinite(priors[action])) {
                priors_consistent = false;
                continue;
            }
            if (priors[action] > 0.0F) {
                raw_priors[action] = priors[action];
            }
        }
    }

    ActionMask retained = node_index == 0U || !priors_consistent
        ? legal
        : mcts_top_k_actions(
            legal,
            legal_count,
            raw_priors,
            config_.expand_top_k,
            rng);
    int retained_count = action_mask_count(retained);
    if (retained_count <= 0) {
        retained = legal;
        retained_count = legal_count;
    }

    float prior_sum = 0.0F;
    for (int i = 0; i < retained_count; ++i) {
        prior_sum += raw_priors[retained.nth_set(static_cast<std::uint32_t>(i))];
    }
    const bool use_uniform = prior_sum <= 0.0F;
    if (use_uniform) {
        prior_sum = static_cast<float>(retained_count);
    }

    ActionMask existing{};
    std::uint32_t previous_child = MCTS_NULL;
    for (std::uint32_t child_index = node.first_child; child_index != MCTS_NULL;
         child_index = nodes_[child_index].next_sibling) {
        const MctsNode& child = nodes_[child_index];
        existing.set(child.action_from_parent);
        previous_child = child_index;
    }
    for (int i = 0; i < retained_count; ++i) {
        const Action action = retained.nth_set(static_cast<std::uint32_t>(i));
        const float raw_prior = use_uniform ? 1.0F : raw_priors[action];
        if (existing.test(action)) {
            for (std::uint32_t child_index = node.first_child; child_index != MCTS_NULL;
                 child_index = nodes_[child_index].next_sibling) {
                MctsNode& child = nodes_[child_index];
                if (child.action_from_parent == action) {
                    child.prior = raw_prior / prior_sum;
                    break;
                }
            }
            continue;
        }
        const std::uint32_t child_index = allocate_node();
        if (child_index == MCTS_NULL) {
            break;
        }

        GameState child_state = states_[node.state_index];
        const bool done = Game::step(child_state, action);
        states_[child_index] = child_state;

        MctsNode& child = nodes_[child_index];
        child.parent = node_index;
        child.state_index = child_index;
        child.action_from_parent = action;
        child.terminal = done || terminal_state(child_state);
        child.player = next_player_for_child(child_state, node.player);
        child.prior = raw_prior / prior_sum;

        if (previous_child == MCTS_NULL) {
            node.first_child = child_index;
        } else {
            nodes_[previous_child].next_sibling = child_index;
        }
        previous_child = child_index;
    }

    node.expanded = true;
    return node.first_child != MCTS_NULL;
}

std::uint32_t Mcts::select_child(std::uint32_t node_index) const noexcept {
    const MctsNode& node = nodes_[node_index];
    std::uint32_t best_child = node.first_child;
    float best_score = -1.0e30F;
    const float parent_visits = static_cast<float>(node.visits + 1U);
    const float exploration_base = std::sqrt(parent_visits);

    for (std::uint32_t child_index = node.first_child; child_index != MCTS_NULL;
         child_index = nodes_[child_index].next_sibling) {
        const MctsNode& child = nodes_[child_index];
        const float q = child_value_for_parent(node, child);
        const float denom = 1.0F + static_cast<float>(child.visits) + child.virtual_loss;
        const float u = config_.c_puct * child.prior * exploration_base / denom;
        const float score_value = q + u - child.virtual_loss;
        if (score_value > best_score) {
            best_score = score_value;
            best_child = child_index;
        }
    }
    return best_child;
}

void Mcts::rollout(GameState& state, Xoshiro256pp& rng) const noexcept {
    std::uint16_t guard = 0;
    while (!terminal_state(state) && guard < config_.rollout_step_cap) {
        ActionMask legal{};
        int legal_count = Game::legal_actions(state, legal);
        if (config_.prune_treasure_plays) {
            legal = mcts_filter_treasure_plays(state, legal);
            legal_count = action_mask_count(legal);
        }
        if (legal_count <= 0) {
            break;
        }
        const Action action = config_.rollout_policy == MctsRolloutPolicy::Random
                || config_.rollout_policy == MctsRolloutPolicy::External
            ? rollout_random_action(legal, legal_count, rng)
            : rollout_policy_action(
                state,
                legal,
                legal_count,
                rng,
                config_.rollout_epsilon,
                config_.rollout_policy);
        (void)Game::step(state, action);
        ++guard;
    }
}

void Mcts::backpropagate(const std::uint32_t* path, std::uint8_t depth, const GameState& terminal) noexcept {
    // A rollout may stop at its step cap with a non-terminal state.  In that
    // case mcts_terminal_value intentionally evaluates the current point
    // margin from each node's perspective, just as it does at game end.
    for (std::uint8_t i = 0; i < depth; ++i) {
        MctsNode& path_node = nodes_[path[i]];
        ++path_node.visits;
        path_node.value_sum += terminal_value_for(path_node.player, terminal);
    }
}

void Mcts::backpropagate_value(
    const std::uint32_t* path,
    std::uint8_t depth,
    PlayerId value_player,
    float value) noexcept {
    for (std::uint8_t i = 0; i < depth; ++i) {
        MctsNode& path_node = nodes_[path[i]];
        ++path_node.visits;
        path_node.value_sum += path_node.player == value_player ? value : -value;
    }
}

void Mcts::apply_virtual_loss(const std::uint32_t* path, std::uint8_t depth, float amount) noexcept {
    for (std::uint8_t i = 0; i < depth; ++i) {
        MctsNode& path_node = nodes_[path[i]];
        path_node.virtual_loss += amount;
        if (path_node.virtual_loss < 0.0F) {
            path_node.virtual_loss = 0.0F;
        }
    }
}

float Mcts::terminal_value_for(PlayerId player, const GameState& terminal) const noexcept {
    return mcts_terminal_value(terminal, player);
}

float Mcts::child_value_for_parent(const MctsNode& parent, const MctsNode& child) const noexcept {
    if (child.visits == 0U) {
        return 0.0F;
    }
    float value = child.value_sum / static_cast<float>(child.visits);
    if (parent.player != child.player) {
        value = -value;
    }
    return value;
}

PlayerId Mcts::next_player_for_child(const GameState& child_state, PlayerId parent_player) const noexcept {
    if (terminal_state(child_state)) {
        return other_player(parent_player);
    }
    const PlayerId player = current_player(child_state);
    return player < child_state.num_players ? player : root_perspective_;
}

float mcts_terminal_value(const GameState& state, PlayerId player) noexcept {
    if (state.num_players != 2U || player >= state.num_players) {
        return 0.0F;
    }
    const PlayerId opponent = other_player(player);
    const std::int16_t player_score = score(state, player);
    const std::int16_t opponent_score = score(state, opponent);
    if (player_score > opponent_score) {
        return 1.0F;
    }
    if (player_score < opponent_score) {
        return -1.0F;
    }
    return 0.0F;
}

Action mcts_choose(const GameState& state, PlayerId perspective, const MctsConfig& config) noexcept {
    Mcts search(config);
    return search.choose(state, perspective);
}
