#include "v2/mcts/eval_runner.h"

#include "v2/bots/scripted.h"
#include "v2/mcts/pile_clock.h"

#include "v2/core/game.h"
#include "v2/core/score.h"
#include "v2/core/turns.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace {

constexpr DefId IMPLEMENTED_KINGDOMS[] = {
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

constexpr std::uint8_t IMPLEMENTED_KINGDOM_COUNT =
    static_cast<std::uint8_t>(sizeof(IMPLEMENTED_KINGDOMS) / sizeof(IMPLEMENTED_KINGDOMS[0]));

struct CardYield {
    std::int8_t actions = 0;
    std::int8_t cards = 0;
    std::int8_t coins = 0;
};

struct DeckProfile {
    std::uint8_t counts[BASIC_CARD_COUNT]{};
    std::uint16_t deck_size = 0;
    std::uint16_t terminal_actions = 0;
    std::uint16_t action_cards = 0;
    std::int16_t total_money = 0;
    std::int16_t total_plus_coins = 0;
};

[[nodiscard]] Setup default_fixed_setup() noexcept {
    Setup setup{};
    setup.num_players = 2U;
    constexpr DefId FIXED[] = {
        DEF_VILLAGE,
        DEF_SMITHY,
        DEF_MARKET,
        DEF_FESTIVAL,
        DEF_LABORATORY,
        DEF_CELLAR,
        DEF_CHAPEL,
        DEF_MILITIA,
        DEF_WITCH,
        DEF_MOAT,
    };
    setup.kingdom_count = static_cast<std::uint8_t>(sizeof(FIXED) / sizeof(FIXED[0]));
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        setup.kingdom[i] = FIXED[i];
    }
    return setup;
}

[[nodiscard]] PlayerId decision_player(const GameState& state) noexcept {
    if (state.decision.player < state.num_players) {
        return state.decision.player;
    }
    return current_player(state);
}

[[nodiscard]] PlayerId winner_for(const GameState& state) noexcept {
    const std::int16_t score0 = score(state, 0U);
    const std::int16_t score1 = score(state, 1U);
    if (score0 > score1) {
        return 0U;
    }
    if (score1 > score0) {
        return 1U;
    }
    return NONE;
}

[[nodiscard]] float uniform_scaffold_prior(
    const GameState&,
    PlayerId,
    Action,
    void*) noexcept {
    return 1.0F;
}

[[nodiscard]] std::uint64_t game_seed(const EvalRunnerConfig& config, std::uint64_t sequence) noexcept {
    return config.seed + (sequence * 0x9E37'79B9'7F4A'7C15ULL);
}

[[nodiscard]] std::size_t checked_obs_size(ObsVersion version) {
    if (!is_valid_obs_version(version)) {
        throw std::invalid_argument("EvalRunnerConfig.obs_version must be V1 or V2");
    }
    return obs_size_for(version);
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
        if ((card.types & TYPE_TREASURE) != 0U) {
            profile.total_money = static_cast<std::int16_t>(profile.total_money + (card.coin_value * count));
        }
        if ((card.types & TYPE_ACTION) != 0U) {
            ++profile.action_cards;
            const CardYield yield = card_yield(def);
            profile.total_plus_coins = static_cast<std::int16_t>(
                profile.total_plus_coins + (yield.coins * count));
            if (yield.actions == 0) {
                profile.terminal_actions = static_cast<std::uint16_t>(profile.terminal_actions + count);
            }
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
    constexpr DefId TREASURES[] = {DEF_PLATINUM, DEF_GOLD, DEF_SILVER, DEF_COPPER, DEF_POTION};
    for (const DefId def : TREASURES) {
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
    case DEF_HARBINGER:
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
    case DEF_REMODEL:
    case DEF_MONEYLENDER:
    case DEF_CHAPEL:
        return 42;
    case DEF_ARTISAN:
    case DEF_BANDIT:
    case DEF_BUREAUCRAT:
    case DEF_WORKSHOP:
    case DEF_MOAT:
    case DEF_VASSAL:
        return 48;
    default:
        return 60;
    }
}

[[nodiscard]] Action best_play_action(const ActionMask& legal) noexcept {
    Action best = A_PASS;
    int best_priority = 999;
    for (DefId def = 0; def < BASIC_CARD_COUNT; ++def) {
        if (!is_action_def(def)) {
            continue;
        }
        const Action action = play_action(def);
        if (!legal.test(action)) {
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
    if (is_victory_def(def)) {
        return 5;
    }
    if (def == DEF_COPPER) {
        return 10;
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
    return 0;
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

[[nodiscard]] int buy_priority(DefId def) noexcept {
    switch (def) {
    case DEF_PROVINCE:
        return 0;
    case DEF_GOLD:
        return 10;
    case DEF_WITCH:
    case DEF_LABORATORY:
    case DEF_FESTIVAL:
    case DEF_MARKET:
    case DEF_SENTRY:
        return 25;
    case DEF_DUCHY:
        return 30;
    case DEF_MILITIA:
    case DEF_SMITHY:
    case DEF_THRONE_ROOM:
        return 35;
    case DEF_SILVER:
        return 40;
    case DEF_VILLAGE:
    case DEF_MERCHANT:
    case DEF_CHAPEL:
        return 50;
    case DEF_ESTATE:
        return 60;
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
    for (DefId def = 0; def < BASIC_CARD_COUNT; ++def) {
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
    for (DefId def = 0; def < BASIC_CARD_COUNT; ++def) {
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
    for (DefId def = 0; def < BASIC_CARD_COUNT; ++def) {
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

[[nodiscard]] Action big_money_buy(const GameState& state, const ActionMask& legal) noexcept {
    const int provinces_left = supply_count(state, DEF_PROVINCE);
    if (state.coins >= 8) {
        const Action province = legal_buy(legal, DEF_PROVINCE);
        if (province != A_PASS) {
            return province;
        }
    }
    if (state.coins >= 6) {
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
    if (state.coins == 5 && provinces_left <= 5) {
        const Action duchy = legal_buy(legal, DEF_DUCHY);
        if (duchy != A_PASS) {
            return duchy;
        }
    }
    if (state.coins >= 3) {
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
    if (state.coins == 2 && provinces_left <= 3) {
        const Action estate = legal_buy(legal, DEF_ESTATE);
        if (estate != A_PASS) {
            return estate;
        }
    }
    return legal.test(A_PASS) ? A_PASS : first_legal(legal);
}

[[nodiscard]] Action heuristic_buy(const GameState& state, const ActionMask& legal, bool engine_style) noexcept {
    const int provinces_left = supply_count(state, DEF_PROVINCE);
    if (state.coins >= 8) {
        const Action province = legal_buy(legal, DEF_PROVINCE);
        if (province != A_PASS) {
            return province;
        }
    }
    if (state.coins >= 6 && provinces_left <= 4) {
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

    if (!engine_style) {
        return big_money_buy(state, legal);
    }

    const DeckProfile profile = analyze_deck(state, state.decision.player);
    const double deck_size = profile.deck_size > 0U ? static_cast<double>(profile.deck_size) : 1.0;
    const double action_density = static_cast<double>(profile.action_cards) / deck_size;
    if (state.coins >= 5 && action_density < 0.35) {
        constexpr DefId ACTION_BUYS[] = {
            DEF_WITCH,
            DEF_LABORATORY,
            DEF_FESTIVAL,
            DEF_MARKET,
            DEF_SENTRY,
            DEF_MILITIA,
            DEF_SMITHY,
        };
        for (const DefId def : ACTION_BUYS) {
            const Action action = legal_buy(legal, def);
            if (action != A_PASS) {
                return action;
            }
        }
    }
    return big_money_buy(state, legal);
}

[[nodiscard]] Action engine_chart_buy(const GameState& state, const ActionMask& legal) noexcept {
    return heuristic_buy(state, legal, true);
}

[[nodiscard]] DefId current_sentry_def(const GameState& state) noexcept {
    if (state.effect_depth > 0U) {
        const EffectFrame& frame = state.effect_stack[state.effect_depth - 1U];
        if (frame.source == DEF_SENTRY && frame.data[3] > 0 && frame.data[4] < frame.data[3]) {
            const std::uint8_t index = static_cast<std::uint8_t>(frame.data[4]);
            return def_for_slot(state, static_cast<Slot>(frame.data[1U + index]));
        }
    }
    return DEF_COPPER;
}

[[nodiscard]] Action sentry_option(const GameState& state, const ActionMask& legal) noexcept {
    const DefId def = current_sentry_def(state);
    Action desired = option_action(2U);
    if (def == DEF_CURSE || def == DEF_ESTATE || def == DEF_COPPER) {
        desired = option_action(0U);
    } else if (def == DEF_DUCHY || def == DEF_PROVINCE || def == DEF_COLONY) {
        desired = option_action(1U);
    }
    return legal.test(desired) ? desired : first_legal_option(legal);
}

[[nodiscard]] Action random_action(const ActionMask& legal, int legal_count, Xoshiro256pp& rng) noexcept {
    if (legal_count <= 0) {
        return A_PASS;
    }
    return legal.nth_set(rng.uniform(static_cast<std::uint32_t>(legal_count)));
}

} // namespace

struct EvalRunner::GameSlot {
    GameState state{};
    Setup setup{};
    Mcts mcts;
    std::optional<Mcts> scripted_mcts{};
    Xoshiro256pp rng{};
    EngineBot engine_v2{};
    EngineBotV3 engine_v3{};
    std::uint64_t seed = 0;
    std::uint64_t sequence = 0;
    std::uint32_t sims_started = 0;
    std::uint32_t sims_completed = 0;
    std::uint32_t pending = 0;
    PlayerId nn_player = 0;
    bool search_active = false;
    bool active = false;

    GameSlot() : mcts(MctsConfig{}) {}
};

struct EvalRunner::PendingLeaf {
    MctsPendingLeaf leaf{};
    std::uint32_t game = 0;
};

Action eval_scripted_action(
    const GameState& state,
    const ActionMask& legal,
    int legal_count,
    EvalScriptedBotKind kind,
    Xoshiro256pp& rng) noexcept {
    if (legal_count <= 0) {
        return A_PASS;
    }
    if (kind == EvalScriptedBotKind::Random) {
        return random_action(legal, legal_count, rng);
    }

    const DecisionKind decision = static_cast<DecisionKind>(state.decision.kind);
    if (decision == DecisionKind::PhaseBuy) {
        const Action treasure = first_legal_play_treasure(legal);
        if (treasure != A_PASS) {
            return treasure;
        }
        if (kind == EvalScriptedBotKind::Engine) {
            return pile_clock_guarded_buy(
                state,
                legal,
                analyze_pile_clock(state, legal),
                engine_chart_buy);
        }
        return big_money_buy(state, legal);
    }
    if (decision == DecisionKind::PhaseAction) {
        if (kind == EvalScriptedBotKind::BigMoney) {
            return legal.test(A_PASS) ? A_PASS : first_legal(legal);
        }
        const Action action = best_play_action(legal);
        return action != A_PASS ? action : (legal.test(A_PASS) ? A_PASS : first_legal(legal));
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
            if (kind == EvalScriptedBotKind::Engine && legal.test(set_aside)) {
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
        if (source == DEF_CELLAR || source == DEF_POACHER || source == DEF_BUREAUCRAT
            || source == DEF_ARTISAN) {
            return choose_select_min_priority(legal, discard_priority, pass_allowed);
        }
        if (source == DEF_CHAPEL || source == DEF_REMODEL || source == DEF_MINE
            || source == DEF_MONEYLENDER || source == DEF_BANDIT) {
            return choose_select_min_priority(legal, trash_priority, pass_allowed);
        }
        if (source == DEF_THRONE_ROOM) {
            return choose_select_min_priority(legal, action_priority, pass_allowed);
        }
        if (source == DEF_HARBINGER) {
            return choose_select_max_priority(legal, keep_priority, pass_allowed);
        }
        return pass_allowed ? A_PASS : first_legal(legal);
    }
    if (legal.test(A_PASS) && state.decision.min_left == 0U) {
        return A_PASS;
    }
    return first_legal(legal);
}

MctsConfig make_scaffold_mcts_config(
    std::uint32_t sims_per_move,
    float c_puct,
    std::uint32_t max_tree_nodes,
    bool prune_treasure_plays,
    std::uint8_t determinizations,
    MctsCPuctSchedule c_puct_schedule,
    float c_puct_init,
    float c_puct_base) noexcept {
    MctsConfig config{};
    config.sims_per_move = sims_per_move;
    config.c_puct = c_puct;
    config.c_puct_schedule = c_puct_schedule;
    config.c_puct_init = c_puct_init;
    config.c_puct_base = c_puct_base;
    config.determinizations = determinizations;
    config.max_tree_nodes = max_tree_nodes;
    config.rollout_step_cap = 1024U;
    config.rollout_policy = MctsRolloutPolicy::EngineLike;
    // EngineLike rollouts normally install a heuristic expansion prior when
    // prior_fn is null. Supply this explicit unit callback for uniform priors.
    config.prior_fn = uniform_scaffold_prior;
    config.prior_user = nullptr;
    config.prune_treasure_plays = prune_treasure_plays;
    return config;
}

std::uint64_t scaffold_rollout_seed(std::uint64_t game_seed) noexcept {
    return game_seed ^ 0x5CAFF01D'0000'0001ULL;
}

[[nodiscard]] Action eval_scaffold_mcts_action(
    Mcts& search,
    const GameState& state,
    const ActionMask& legal,
    int legal_count) noexcept {
    if (legal_count <= 0) {
        return A_PASS;
    }
    if (legal_count == 1) {
        return legal.nth_set(0U);
    }
    // Preserve MctsBot's Phase-6 treasure fast path; this is a forced,
    // non-strategic subdecision and leaves search budget for meaningful ones.
    if (static_cast<DecisionKind>(state.decision.kind) == DecisionKind::PhaseBuy) {
        const Action treasure = first_legal_play_treasure(legal);
        if (treasure != A_PASS) {
            return treasure;
        }
    }
    return search.choose(state, decision_player(state));
}

EvalRunner::EvalRunner(const EvalRunnerConfig& config)
    : config_(config),
      obs_size_(checked_obs_size(config.obs_version)),
      mcts_config_(),
      games_(new GameSlot[std::max(1U, config.n_games)]),
      pending_(new PendingLeaf[std::max(1U, config.max_batch)]),
      leaf_obs_(new float[static_cast<std::size_t>(std::max(1U, config.max_batch)) * obs_size_]),
      leaf_masks_(new bool[static_cast<std::size_t>(std::max(1U, config.max_batch)) * ACTION_SPACE_SIZE]),
      normalized_policy_(new float[static_cast<std::size_t>(std::max(1U, config.max_batch)) * ACTION_SPACE_SIZE]) {
    if (config_.n_games == 0U) {
        throw std::invalid_argument("EvalRunnerConfig.n_games must be positive");
    }
    if (config_.max_batch == 0U) {
        throw std::invalid_argument("EvalRunnerConfig.max_batch must be positive");
    }
    if (config_.sims_per_move == 0U) {
        throw std::invalid_argument("EvalRunnerConfig.sims_per_move must be positive");
    }
    if (!mcts_is_valid_c_puct_schedule(config_.c_puct_schedule)) {
        throw std::invalid_argument("EvalRunnerConfig.c_puct_schedule is invalid");
    }
    if (!(config_.c_puct_init > 0.0F) || !std::isfinite(config_.c_puct_init)) {
        throw std::invalid_argument("EvalRunnerConfig.c_puct_init must be finite and positive");
    }
    if (!(config_.c_puct_base > 0.0F) || !std::isfinite(config_.c_puct_base)) {
        throw std::invalid_argument("EvalRunnerConfig.c_puct_base must be finite and positive");
    }
    if (config_.fixed_setup.num_players == 0U) {
        config_.fixed_setup = default_fixed_setup();
    }
    config_.fixed_setup.num_players = 2U;

    mcts_config_.sims_per_move = config_.sims_per_move;
    mcts_config_.c_puct = config_.c_puct;
    mcts_config_.c_puct_schedule = config_.c_puct_schedule;
    mcts_config_.c_puct_init = config_.c_puct_init;
    mcts_config_.c_puct_base = config_.c_puct_base;
    mcts_config_.determinizations = 1U;
    mcts_config_.max_tree_nodes = config_.max_tree_nodes == 0U ? 4096U : config_.max_tree_nodes;
    mcts_config_.rollout_policy = MctsRolloutPolicy::External;
    mcts_config_.prune_treasure_plays = config_.prune_treasure_plays;

    scaffold_mcts_config_ = make_scaffold_mcts_config(
        config_.sims_per_move,
        config_.c_puct,
        mcts_config_.max_tree_nodes,
        config_.prune_treasure_plays,
        2U,
        config_.c_puct_schedule,
        config_.c_puct_init,
        config_.c_puct_base);

    for (std::uint32_t i = 0; i < config_.n_games; ++i) {
        games_[i].mcts = Mcts(mcts_config_);
        reset_game(i);
    }
}

EvalRunner::~EvalRunner() = default;

std::uint32_t EvalRunner::collect_leaves(std::uint32_t max_batch) noexcept {
    if (pending_count_ != 0U) {
        return pending_count_;
    }
    const std::uint32_t limit = std::min(max_batch == 0U ? config_.max_batch : max_batch, config_.max_batch);
    pending_count_ = 0;
    std::uint32_t idle = 0;
    while (pending_count_ < limit && idle < config_.n_games) {
        const std::uint32_t index = next_collect_game_;
        next_collect_game_ = (next_collect_game_ + 1U) % config_.n_games;
        GameSlot& game = games_[index];

        if (!game.active) {
            ++idle;
            continue;
        }
        if (game.pending != 0U) {
            ++idle;
            continue;
        }
        drive_scripted(game);
        if (game.pending != 0U) {
            ++idle;
            continue;
        }
        if (decision_player(game.state) != game.nn_player) {
            ++idle;
            continue;
        }
        if (!game.search_active) {
            auto_play_treasures(game);
        }
        if (!game.active) {
            ++idle;
            continue;
        }
        if (decision_player(game.state) != game.nn_player) {
            ++idle;
            continue;
        }
        if (!game.search_active) {
            start_search(game);
        }
        if (game.sims_started >= config_.sims_per_move) {
            maybe_finish_move(game);
            ++idle;
            continue;
        }

        MctsPendingLeaf leaf{};
        const bool need_eval = game.mcts.collect_external_leaf(leaf);
        ++game.sims_started;
        if (!need_eval) {
            ++game.sims_completed;
            maybe_finish_move(game);
            idle = 0;
            continue;
        }
        if (leaf.player != game.nn_player) {
            if (resolve_scripted_tree_leaf(game, leaf)) {
                ++game.sims_completed;
            }
            maybe_finish_move(game);
            idle = 0;
            continue;
        }

        PendingLeaf& pending = pending_[pending_count_];
        pending.leaf = leaf;
        pending.game = index;
        encode(
            game.mcts.state_for(leaf.state_index),
            leaf.player,
            leaf_obs_.get() + (pending_count_ * obs_size_),
            config_.obs_version);
        bool* mask = leaf_masks_.get() + (pending_count_ * ACTION_SPACE_SIZE);
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            mask[action] = leaf.legal.test(action);
        }
        ++game.pending;
        ++pending_count_;
        idle = 0;
    }
    return pending_count_;
}

void EvalRunner::provide_evaluations(const float* values, const float* policies, std::uint32_t count) noexcept {
    const std::uint32_t n = std::min(count, pending_count_);
    for (std::uint32_t i = 0; i < n; ++i) {
        PendingLeaf& pending = pending_[i];
        GameSlot& game = games_[pending.game];
        float* normalized = normalized_policy_.get() + (i * ACTION_SPACE_SIZE);
        normalize_policy(
            policies == nullptr ? nullptr : policies + (i * ACTION_SPACE_SIZE),
            pending.leaf.legal,
            pending.leaf.legal_count,
            normalized);
        const float value = values == nullptr ? 0.0F : values[i];
        game.mcts.provide_external_evaluation(pending.leaf, value, normalized, game.rng);
        if (game.pending > 0U) {
            --game.pending;
        }
        ++game.sims_completed;
        maybe_finish_move(game);
    }
    pending_count_ = 0;
}

const float* EvalRunner::leaf_observations() const noexcept {
    return leaf_obs_.get();
}

const bool* EvalRunner::leaf_legal_masks() const noexcept {
    return leaf_masks_.get();
}

std::uint32_t EvalRunner::leaf_count() const noexcept {
    return pending_count_;
}

std::size_t EvalRunner::observation_size() const noexcept {
    return obs_size_;
}

EvalRunnerResult EvalRunner::result() const noexcept {
    return result_;
}

std::uint64_t EvalRunner::games_completed() const noexcept {
    return result_.games;
}

float EvalRunner::total_virtual_loss() const noexcept {
    float total = 0.0F;
    for (std::uint32_t i = 0; i < config_.n_games; ++i) {
        total += games_[i].mcts.total_virtual_loss();
    }
    return total;
}

PlayerId EvalRunner::active_nn_player(std::uint32_t index) const noexcept {
    return index < config_.n_games && games_[index].active ? games_[index].nn_player : NONE;
}

std::uint64_t EvalRunner::active_sequence(std::uint32_t index) const noexcept {
    return index < config_.n_games ? games_[index].sequence : 0U;
}

Action EvalRunner::last_scripted_action() const noexcept {
    return last_scripted_action_;
}

std::vector<GameState> EvalRunner::take_finished_games() {
    std::vector<GameState> out = std::move(finished_games_);
    finished_games_.clear();
    return out;
}

void EvalRunner::reset_game(std::uint32_t index) noexcept {
    GameSlot& game = games_[index];
    game.engine_v2 = EngineBot{};
    game.engine_v3 = EngineBotV3{};
    if (config_.target_games != 0U && next_sequence_ >= config_.target_games) {
        game.state = GameState{};
        game.setup = Setup{};
        game.seed = 0;
        game.sequence = next_sequence_;
        game.sims_started = 0;
        game.sims_completed = 0;
        game.pending = 0;
        game.nn_player = NONE;
        game.search_active = false;
        game.active = false;
        game.scripted_mcts.reset();
        return;
    }
    game.sequence = next_sequence_++;
    game.setup = setup_for(game.sequence);
    game.seed = game_seed(config_, game.sequence);
    game.state = Game::new_game(game.setup, game.seed);
    game.rng = Xoshiro256pp::seeded(game.seed ^ 0xE0A1'600D'0000'0001ULL);
    if (config_.opponent == EvalScriptedBotKind::Mcts) {
        MctsConfig scaffold_config = scaffold_mcts_config_;
        scaffold_config.rollout_seed = scaffold_rollout_seed(game.seed);
        game.scripted_mcts.emplace(scaffold_config);
    } else {
        game.scripted_mcts.reset();
    }
    game.nn_player = static_cast<PlayerId>((game.sequence & 1ULL) == 0ULL ? 0U : 1U);
    game.sims_started = 0;
    game.sims_completed = 0;
    game.pending = 0;
    game.search_active = false;
    game.active = true;
}

void EvalRunner::start_search(GameSlot& game) noexcept {
    game.mcts.reset(game.state, decision_player(game.state));
    game.sims_started = 0;
    game.sims_completed = 0;
    game.pending = 0;
    game.search_active = true;
}

void EvalRunner::auto_play_treasures(GameSlot& game) noexcept {
    if (!config_.auto_play_treasures) {
        return;
    }

    std::uint16_t guard = 0;
    while (game.state.phase != static_cast<std::uint8_t>(Phase::Over) && guard < 512U) {
        ActionMask legal{};
        (void)Game::legal_actions(game.state, legal);
        const Action action = mcts_canonical_treasure_play(
            mcts_filter_treasure_plays(game.state, legal));
        if (action == A_PASS) {
            return;
        }
        const bool done = Game::step(game.state, action);
        ++guard;
        if (done || game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            finish_game(game);
            return;
        }
    }
}

void EvalRunner::drive_scripted(GameSlot& game) noexcept {
    std::uint16_t guard = 0;
    while (game.state.phase != static_cast<std::uint8_t>(Phase::Over)
           && decision_player(game.state) != game.nn_player
           && guard < 512U) {
        ActionMask legal{};
        const int legal_count = Game::legal_actions(game.state, legal);
        if (legal_count <= 0) {
            break;
        }
        Action action = A_PASS;
        if (config_.opponent == EvalScriptedBotKind::Mcts) {
            action = eval_scaffold_mcts_action(*game.scripted_mcts, game.state, legal, legal_count);
        } else if (config_.opponent == EvalScriptedBotKind::EngineV2) {
            action = game.engine_v2.choose_action(game.state, legal, legal_count);
        } else if (config_.opponent == EvalScriptedBotKind::EngineV3) {
            action = game.engine_v3.choose_action(game.state, legal, legal_count);
        } else {
            action = eval_scripted_action(game.state, legal, legal_count, config_.opponent, game.rng);
        }
        if (!legal.test(action)) {
            action = first_legal(legal);
        }
        last_scripted_action_ = action;
        const bool done = Game::step(game.state, action);
        ++guard;
        if (done || game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            finish_game(game);
            break;
        }
    }
}

void EvalRunner::maybe_finish_move(GameSlot& game) noexcept {
    if (!game.search_active || game.pending != 0U || game.sims_completed < config_.sims_per_move) {
        return;
    }

    Action action = game.mcts.best_root_action();
    ActionMask legal{};
    const int legal_count = Game::legal_actions(game.state, legal);
    if (legal_count <= 0 || !legal.test(action)) {
        action = legal_count > 0 ? legal.nth_set(0U) : A_PASS;
    }
    const bool done = Game::step(game.state, action);
    game.search_active = false;
    if (done || game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
        finish_game(game);
    }
}

void EvalRunner::finish_game(GameSlot& game) noexcept {
    if (config_.retain_finished_games) {
        finished_games_.push_back(game.state);
    }
    ++result_.games;
    if (game.state.truncated != 0U) {
        ++result_.truncated;
    }
    const PlayerId winner = winner_for(game.state);
    if (winner == NONE) {
        ++result_.ties;
    } else if (winner == game.nn_player) {
        ++result_.nn_wins;
    } else {
        ++result_.scripted_wins;
    }
    reset_game(static_cast<std::uint32_t>(&game - games_.get()));
}

Setup EvalRunner::setup_for(std::uint64_t sequence) const noexcept {
    if (config_.kingdom_mode == SelfPlayKingdomMode::Fixed) {
        Setup setup = config_.fixed_setup;
        setup.num_players = 2U;
        return setup;
    }

    Setup setup{};
    setup.num_players = 2U;
    setup.kingdom_count = 10U;
    DefId defs[IMPLEMENTED_KINGDOM_COUNT]{};
    for (std::uint8_t i = 0; i < IMPLEMENTED_KINGDOM_COUNT; ++i) {
        defs[i] = IMPLEMENTED_KINGDOMS[i];
    }
    const std::uint64_t pair = sequence / 2ULL;
    Xoshiro256pp rng = Xoshiro256pp::seeded(
        config_.seed ^ (0xBADC'0FFE'1234'5678ULL * (pair + 1ULL)));
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        const std::uint32_t offset = rng.uniform(static_cast<std::uint32_t>(IMPLEMENTED_KINGDOM_COUNT - i));
        const std::uint8_t swap_index = static_cast<std::uint8_t>(i + offset);
        const DefId selected = defs[swap_index];
        defs[swap_index] = defs[i];
        defs[i] = selected;
        setup.kingdom[i] = selected;
    }
    return setup;
}

void EvalRunner::normalize_policy(
    const float* logits,
    const ActionMask& legal,
    int legal_count,
    float* out) const noexcept {
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        out[action] = 0.0F;
    }
    if (legal_count <= 0) {
        out[A_PASS] = 1.0F;
        return;
    }
    float max_logit = -3.4e38F;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (legal.test(action)) {
            const float logit = logits == nullptr ? 0.0F : logits[action];
            if (logit > max_logit) {
                max_logit = logit;
            }
        }
    }
    double sum = 0.0;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (!legal.test(action)) {
            continue;
        }
        const float logit = logits == nullptr ? 0.0F : logits[action];
        const double value = std::exp(static_cast<double>(logit - max_logit));
        out[action] = static_cast<float>(value);
        sum += value;
    }
    if (sum <= 0.0) {
        const float uniform = 1.0F / static_cast<float>(legal_count);
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            out[action] = legal.test(action) ? uniform : 0.0F;
        }
        return;
    }
    const float inv_sum = static_cast<float>(1.0 / sum);
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        out[action] *= inv_sum;
    }
}

bool EvalRunner::resolve_scripted_tree_leaf(GameSlot& game, const MctsPendingLeaf& leaf) noexcept {
    const GameState& leaf_state = game.mcts.state_for(leaf.state_index);
    Action action = A_PASS;
    if (config_.opponent == EvalScriptedBotKind::Mcts) {
        action = eval_scaffold_mcts_action(*game.scripted_mcts, leaf_state, leaf.legal, leaf.legal_count);
    } else if (config_.opponent == EvalScriptedBotKind::EngineV2) {
        action = game.engine_v2.choose_action(leaf_state, leaf.legal, leaf.legal_count);
    } else if (config_.opponent == EvalScriptedBotKind::EngineV3) {
        action = game.engine_v3.choose_action(leaf_state, leaf.legal, leaf.legal_count);
    } else {
        action = eval_scripted_action(leaf_state, leaf.legal, leaf.legal_count, config_.opponent, game.rng);
    }
    if (!leaf.legal.test(action)) {
        action = leaf.legal_count > 0 ? leaf.legal.nth_set(0U) : A_PASS;
    }
    float priors[ACTION_SPACE_SIZE]{};
    priors[action] = 1.0F;
    game.mcts.provide_external_evaluation(leaf, 0.0F, priors, game.rng);
    return true;
}
