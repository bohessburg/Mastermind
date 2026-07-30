#include "v2/mcts/selfplay.h"

#include "v2/bots/scripted.h"
#include "v2/core/actions.h"
#include "v2/core/determinize.h"
#include "v2/core/game.h"
#include "v2/core/interp.h"
#include "v2/core/score.h"
#include "v2/core/turns.h"
#include "v2/mcts/eval_runner.h"
#include "v2/mcts/pile_clock.h"

#include <algorithm>
#include <cassert>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

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

constexpr std::uint64_t SELFPLAY_DETERMINIZATION_STREAM = 0x4454'4552'4D49'4E45ULL;
constexpr std::uint64_t SPLITMIX_GOLDEN_RATIO = 0x9E37'79B9'7F4A'7C15ULL;
constexpr std::uint64_t SELFPLAY_DETERMINIZATION_SEAT_MIX = 0xD1B5'4A32'D192'ED03ULL;
constexpr std::uint64_t SELFPLAY_DETERMINIZATION_INDEX_MIX = 0x94D0'49BB'1331'11EBULL;

[[nodiscard]] constexpr bool valid_determinize_mode(SelfPlayDeterminizeMode mode) noexcept {
    return mode == SelfPlayDeterminizeMode::Off
        || mode == SelfPlayDeterminizeMode::PerDecision
        || mode == SelfPlayDeterminizeMode::PerTurn;
}

[[nodiscard]] constexpr bool valid_temp_mode(SelfPlayTempMode mode) noexcept {
    return mode == SelfPlayTempMode::Legacy
        || mode == SelfPlayTempMode::PerSeatBuy;
}

[[nodiscard]] constexpr std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += SPLITMIX_GOLDEN_RATIO;
    value = (value ^ (value >> 30U)) * 0xBF58'476D'1CE4'E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D0'49BB'1331'11EBULL;
    return value ^ (value >> 31U);
}

[[nodiscard]] bool implemented_kingdom(DefId def) noexcept {
    for (const DefId implemented : IMPLEMENTED_KINGDOMS) {
        if (implemented == def) {
            return true;
        }
    }
    return false;
}

enum class ScriptedSlotStatus : std::uint8_t {
    Idle,
    ScriptedPending,
    ScriptedRunning,
    ScriptedReady,
};

[[nodiscard]] PlayerId decision_player(const GameState& state) noexcept {
    if (state.decision.player < state.num_players) {
        return state.decision.player;
    }
    return current_player(state);
}

[[nodiscard]] int pile_count(const Pile& pile) noexcept {
    return pile.mixed_len > 0U ? static_cast<int>(pile.mixed_len) : static_cast<int>(pile.count);
}

[[nodiscard]] DefId pile_top_def(const GameState& state, const Pile& pile) noexcept {
    const Slot slot = pile.mixed_len > 0U ? pile.mixed[pile.mixed_len - 1U] : pile.base;
    return slot < state.num_slots ? state.slot_to_def[slot] : DEF_COPPER;
}

[[nodiscard]] bool scaffold_endgame_armed(
    const GameState& state,
    const ActionMask& legal) noexcept {
    // analyze_pile_clock is the shared, stack-only supply analysis used by
    // the Engine rollout chart. Its empty-pile count is independent of which
    // buys happen to be legal at this decision.
    if (analyze_pile_clock(state, legal).empty_piles > 0) {
        return true;
    }
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        const Pile& pile = state.piles[i];
        if (pile_top_def(state, pile) == DEF_PROVINCE) {
            return pile_count(pile) <= 4;
        }
    }
    return false;
}

[[nodiscard]] bool scripted_mode(SelfPlayScriptedBotKind kind) noexcept {
    return kind != SelfPlayScriptedBotKind::None;
}

[[nodiscard]] bool scripted_mode(const SelfPlayConfig& config) noexcept {
    return scripted_mode(config.scripted_bot);
}

enum class OpeningPreferenceCondition : std::uint8_t {
    Always,
    VillageToTerminals,
    VillageToSmithies,
};

struct OpeningPreference {
    DefId def = DEF_COPPER;
    std::uint8_t cap = 0U; // Zero means unlimited.
    OpeningPreferenceCondition condition = OpeningPreferenceCondition::Always;
};

struct OpeningPriceBand {
    std::int16_t min_coins = 0;
    std::int16_t max_coins = 0;
    const OpeningPreference* preferences = nullptr;
    std::uint8_t preference_count = 0U;
};

struct OpeningTemplateBands {
    const OpeningPriceBand* bands = nullptr;
    std::uint8_t band_count = 0U;
};

constexpr DefId OPENING_TERMINALS[] = {
    DEF_SMITHY,
    DEF_MILITIA,
    DEF_WITCH,
    DEF_BANDIT,
    DEF_COUNCIL_ROOM,
    DEF_LIBRARY,
    DEF_MONEYLENDER,
};

constexpr DefId OPENING_TELEMETRY_DEFS[SELFPLAY_OPENING_TELEMETRY_CARD_COUNT] = {
    DEF_CHAPEL,
    DEF_SENTRY,
    DEF_MONEYLENDER,
    DEF_VILLAGE,
};

constexpr OpeningPreference T1_2_TO_3[] = {
    {DEF_CHAPEL, 1U, OpeningPreferenceCondition::Always},
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T1_4_TO_5[] = {
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T1_6_TO_7[] = {
    {DEF_GOLD, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPriceBand T1_BANDS[] = {
    {2, 3, T1_2_TO_3, static_cast<std::uint8_t>(std::size(T1_2_TO_3))},
    {4, 5, T1_4_TO_5, static_cast<std::uint8_t>(std::size(T1_4_TO_5))},
    {6, 7, T1_6_TO_7, static_cast<std::uint8_t>(std::size(T1_6_TO_7))},
};

constexpr OpeningPreference T2_3[] = {
    {DEF_VILLAGE, 4U, OpeningPreferenceCondition::VillageToTerminals},
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T2_4[] = {
    {DEF_SMITHY, 3U, OpeningPreferenceCondition::VillageToSmithies},
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T2_5_TO_6[] = {
    {DEF_MARKET, 0U, OpeningPreferenceCondition::Always},
    {DEF_FESTIVAL, 0U, OpeningPreferenceCondition::Always},
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T2_7[] = {
    {DEF_GOLD, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPriceBand T2_BANDS[] = {
    {3, 3, T2_3, static_cast<std::uint8_t>(std::size(T2_3))},
    {4, 4, T2_4, static_cast<std::uint8_t>(std::size(T2_4))},
    {5, 6, T2_5_TO_6, static_cast<std::uint8_t>(std::size(T2_5_TO_6))},
    {7, 7, T2_7, static_cast<std::uint8_t>(std::size(T2_7))},
};

constexpr OpeningPreference T3_3[] = {
    {DEF_MERCHANT, 2U, OpeningPreferenceCondition::Always},
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T3_4[] = {
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T3_5_TO_6[] = {
    {DEF_LABORATORY, 5U, OpeningPreferenceCondition::Always},
    {DEF_MARKET, 0U, OpeningPreferenceCondition::Always},
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T3_7[] = {
    {DEF_GOLD, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPriceBand T3_BANDS[] = {
    {3, 3, T3_3, static_cast<std::uint8_t>(std::size(T3_3))},
    {4, 4, T3_4, static_cast<std::uint8_t>(std::size(T3_4))},
    {5, 6, T3_5_TO_6, static_cast<std::uint8_t>(std::size(T3_5_TO_6))},
    {7, 7, T3_7, static_cast<std::uint8_t>(std::size(T3_7))},
};

constexpr OpeningPreference T4_3[] = {
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T4_4[] = {
    {DEF_MONEYLENDER, 1U, OpeningPreferenceCondition::Always},
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T4_5_TO_6[] = {
    {DEF_SENTRY, 2U, OpeningPreferenceCondition::Always},
    {DEF_LABORATORY, 0U, OpeningPreferenceCondition::Always},
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T4_7[] = {
    {DEF_GOLD, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPriceBand T4_BANDS[] = {
    {3, 3, T4_3, static_cast<std::uint8_t>(std::size(T4_3))},
    {4, 4, T4_4, static_cast<std::uint8_t>(std::size(T4_4))},
    {5, 6, T4_5_TO_6, static_cast<std::uint8_t>(std::size(T4_5_TO_6))},
    {7, 7, T4_7, static_cast<std::uint8_t>(std::size(T4_7))},
};

constexpr OpeningPreference T5_3[] = {
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T5_4[] = {
    {DEF_MILITIA, 1U, OpeningPreferenceCondition::Always},
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T5_5_TO_6[] = {
    {DEF_WITCH, 2U, OpeningPreferenceCondition::Always},
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T5_7[] = {
    {DEF_GOLD, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPriceBand T5_BANDS[] = {
    {3, 3, T5_3, static_cast<std::uint8_t>(std::size(T5_3))},
    {4, 4, T5_4, static_cast<std::uint8_t>(std::size(T5_4))},
    {5, 6, T5_5_TO_6, static_cast<std::uint8_t>(std::size(T5_5_TO_6))},
    {7, 7, T5_7, static_cast<std::uint8_t>(std::size(T5_7))},
};

constexpr OpeningPreference T6_3[] = {
    {DEF_WORKSHOP, 3U, OpeningPreferenceCondition::Always},
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T6_4[] = {
    {DEF_GARDENS, 0U, OpeningPreferenceCondition::Always},
    {DEF_WORKSHOP, 3U, OpeningPreferenceCondition::Always},
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T6_5_TO_6[] = {
    {DEF_GARDENS, 0U, OpeningPreferenceCondition::Always},
    {DEF_SILVER, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPreference T6_7[] = {
    {DEF_GOLD, 0U, OpeningPreferenceCondition::Always},
};
constexpr OpeningPriceBand T6_BANDS[] = {
    {3, 3, T6_3, static_cast<std::uint8_t>(std::size(T6_3))},
    {4, 4, T6_4, static_cast<std::uint8_t>(std::size(T6_4))},
    {5, 6, T6_5_TO_6, static_cast<std::uint8_t>(std::size(T6_5_TO_6))},
    {7, 7, T6_7, static_cast<std::uint8_t>(std::size(T6_7))},
};

constexpr OpeningTemplateBands OPENING_TEMPLATE_BANDS[SELFPLAY_OPENING_TEMPLATE_COUNT] = {
    {},
    {T1_BANDS, static_cast<std::uint8_t>(std::size(T1_BANDS))},
    {T2_BANDS, static_cast<std::uint8_t>(std::size(T2_BANDS))},
    {T3_BANDS, static_cast<std::uint8_t>(std::size(T3_BANDS))},
    {T4_BANDS, static_cast<std::uint8_t>(std::size(T4_BANDS))},
    {T5_BANDS, static_cast<std::uint8_t>(std::size(T5_BANDS))},
    {T6_BANDS, static_cast<std::uint8_t>(std::size(T6_BANDS))},
};

[[nodiscard]] std::uint16_t count_owned_def(
    const GameState& state,
    PlayerId player_id,
    DefId wanted) noexcept {
    if (player_id >= state.num_players) {
        return 0U;
    }
    const PlayerState& player = state.players[player_id];
    std::uint16_t total = 0U;
    const auto count_slot = [&state, wanted, &total](Slot slot, std::uint8_t count) noexcept {
        if (slot < state.num_slots && state.slot_to_def[slot] == wanted) {
            total = static_cast<std::uint16_t>(total + count);
        }
    };
    for (Slot slot = 0U; slot < state.num_slots; ++slot) {
        count_slot(slot, player.hand[slot]);
        count_slot(slot, player.exile[slot]);
        count_slot(slot, player.tavern[slot]);
        count_slot(slot, player.island_mat[slot]);
    }
    const auto count_ordered = [&count_slot](const OrderedZone& zone) noexcept {
        for (std::uint8_t index = 0U; index < zone.size; ++index) {
            count_slot(zone.cards[index], 1U);
        }
    };
    count_ordered(player.deck);
    count_ordered(player.discard);
    count_ordered(player.set_aside);
    for (std::uint8_t index = 0U; index < player.in_play_size; ++index) {
        count_slot(player.in_play[index].slot, 1U);
    }
    return total;
}

[[nodiscard]] std::uint16_t count_owned_treasures(
    const GameState& state,
    PlayerId player_id) noexcept {
    if (player_id >= state.num_players) {
        return 0U;
    }
    const PlayerState& player = state.players[player_id];
    std::uint16_t total = 0U;
    const auto count_slot = [&state, &total](Slot slot, std::uint8_t count) noexcept {
        if (slot < state.num_slots
            && (card_def(state.slot_to_def[slot]).types & TYPE_TREASURE) != 0U) {
            total = static_cast<std::uint16_t>(total + count);
        }
    };
    for (Slot slot = 0U; slot < state.num_slots; ++slot) {
        count_slot(slot, player.hand[slot]);
        count_slot(slot, player.exile[slot]);
        count_slot(slot, player.tavern[slot]);
        count_slot(slot, player.island_mat[slot]);
    }
    const auto count_ordered = [&count_slot](const OrderedZone& zone) noexcept {
        for (std::uint8_t index = 0U; index < zone.size; ++index) {
            count_slot(zone.cards[index], 1U);
        }
    };
    count_ordered(player.deck);
    count_ordered(player.discard);
    count_ordered(player.set_aside);
    for (std::uint8_t index = 0U; index < player.in_play_size; ++index) {
        count_slot(player.in_play[index].slot, 1U);
    }
    return total;
}

[[nodiscard]] std::uint16_t terminal_actions_owned(
    const GameState& state,
    PlayerId player_id) noexcept {
    std::uint16_t total = 0U;
    for (const DefId def : OPENING_TERMINALS) {
        total = static_cast<std::uint16_t>(total + count_owned_def(state, player_id, def));
    }
    return total;
}

[[nodiscard]] bool opening_condition_matches(
    const GameState& state,
    PlayerId player,
    OpeningPreferenceCondition condition) noexcept {
    const std::uint16_t villages = count_owned_def(state, player, DEF_VILLAGE);
    switch (condition) {
    case OpeningPreferenceCondition::Always:
        return true;
    case OpeningPreferenceCondition::VillageToTerminals:
        return villages <= terminal_actions_owned(state, player);
    case OpeningPreferenceCondition::VillageToSmithies:
        return villages > count_owned_def(state, player, DEF_SMITHY);
    }
    return false;
}

[[nodiscard]] bool source_is_own_played_card(
    const GameState& state,
    PlayerId player) noexcept {
    if (player >= state.num_players
        || state.decision.source >= card_def_count()
        || state.effect_depth == 0U) {
        return false;
    }
    const EffectFrame& active_frame = state.effect_stack[state.effect_depth - 1U];
    if (active_frame.source != state.decision.source
        || active_frame.player != player
        || (active_frame.flags & FRAME_ATTACK) != 0U) {
        return false;
    }
    const PlayerState& owner = state.players[player];
    for (std::uint8_t index = 0U; index < owner.in_play_size; ++index) {
        const Slot slot = owner.in_play[index].slot;
        if (slot < state.num_slots && state.slot_to_def[slot] == state.decision.source) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] Action opening_buy_preference(
    const GameState& state,
    PlayerId player,
    std::uint8_t template_id,
    const ActionMask& legal) noexcept {
    if (template_id == SELFPLAY_UNCONSTRAINED_TEMPLATE
        || template_id >= SELFPLAY_OPENING_TEMPLATE_COUNT) {
        return A_END;
    }
    if (state.coins >= 8) {
        const Action province = buy_action(DEF_PROVINCE);
        return legal.test(province) ? province : A_END;
    }

    const OpeningTemplateBands& template_bands = OPENING_TEMPLATE_BANDS[template_id];
    for (std::uint8_t band_index = 0U; band_index < template_bands.band_count; ++band_index) {
        const OpeningPriceBand& band = template_bands.bands[band_index];
        if (state.coins < band.min_coins || state.coins > band.max_coins) {
            continue;
        }
        for (std::uint8_t preference_index = 0U;
             preference_index < band.preference_count;
             ++preference_index) {
            const OpeningPreference& preference = band.preferences[preference_index];
            if (preference.cap != 0U
                && count_owned_def(state, player, preference.def) >= preference.cap) {
                continue;
            }
            if (!opening_condition_matches(state, player, preference.condition)) {
                continue;
            }
            const Action action = buy_action(preference.def);
            if (legal.test(action)) {
                return action;
            }
        }
        return A_END;
    }
    return A_END;
}

[[nodiscard]] Action opening_trash_preference(
    const GameState& state,
    PlayerId player,
    const ActionMask& legal) noexcept {
    if (state.decision.select_semantic != static_cast<std::uint8_t>(SelectSemantic::Trash)
        || !source_is_own_played_card(state, player)) {
        return A_END;
    }
    constexpr DefId JUNK[] = {DEF_CURSE, DEF_ESTATE};
    for (const DefId def : JUNK) {
        const Action action = select_action(def);
        if (legal.test(action)) {
            return action;
        }
    }
    const Action copper = select_action(DEF_COPPER);
    if (count_owned_treasures(state, player) > 3U && legal.test(copper)) {
        return copper;
    }
    // Optional selections should end rather than touching Silver, Gold, or
    // actions. This also protects the Copper floor when Copper is the only
    // remaining selectable card.
    if (state.decision.min_left == 0U && legal.test(A_PASS)) {
        return A_PASS;
    }
    return A_END;
}

[[nodiscard]] std::uint8_t draw_opening_template(
    const SelfPlayConfig& config,
    Xoshiro256pp& rng) noexcept {
    double total = 0.0;
    for (const float weight : config.template_weights) {
        total += static_cast<double>(weight);
    }
    const double unit = static_cast<double>(rng.next() >> 11U)
        * (1.0 / 9007199254740992.0);
    const double target = unit * total;
    double cumulative = 0.0;
    for (std::uint8_t template_id = 0U;
         template_id < SELFPLAY_OPENING_TEMPLATE_COUNT;
         ++template_id) {
        cumulative += static_cast<double>(config.template_weights[template_id]);
        if (target < cumulative) {
            return template_id;
        }
    }
    return SELFPLAY_OPENING_TEMPLATE_COUNT - 1U;
}

[[nodiscard]] int opening_telemetry_index(DefId def) noexcept {
    for (std::uint8_t index = 0U;
         index < SELFPLAY_OPENING_TELEMETRY_CARD_COUNT;
         ++index) {
        if (OPENING_TELEMETRY_DEFS[index] == def) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

[[nodiscard]] std::size_t checked_obs_size(ObsVersion version) {
    if (!is_valid_obs_version(version)) {
        throw std::invalid_argument("SelfPlayConfig.obs_version must be V1, V2, or V3");
    }
    return obs_size_for(version);
}

[[nodiscard]] EvalScriptedBotKind eval_scripted_kind(SelfPlayScriptedBotKind kind) noexcept {
    switch (kind) {
    case SelfPlayScriptedBotKind::BigMoney:
        return EvalScriptedBotKind::BigMoney;
    case SelfPlayScriptedBotKind::Engine:
        return EvalScriptedBotKind::Engine;
    case SelfPlayScriptedBotKind::EngineV3:
        return EvalScriptedBotKind::EngineV3;
    case SelfPlayScriptedBotKind::Thinner:
        return EvalScriptedBotKind::Thinner;
    case SelfPlayScriptedBotKind::Random:
        return EvalScriptedBotKind::Random;
    case SelfPlayScriptedBotKind::Scaffold:
    case SelfPlayScriptedBotKind::None:
        break;
    }
    return EvalScriptedBotKind::BigMoney;
}

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

[[nodiscard]] std::uint64_t game_seed(
    const SelfPlayConfig& config,
    std::uint64_t index,
    std::uint64_t generation) noexcept {
    return config.seed
        + (generation * 0x9E37'79B9'7F4A'7C15ULL)
        + (static_cast<std::uint64_t>(index) * 0xD1B5'4A32'D192'ED03ULL);
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

} // namespace

std::uint64_t selfplay_determinization_seed(
    std::uint64_t game_seed,
    PlayerId seat,
    std::uint16_t move_index,
    std::uint16_t turn_counter,
    SelfPlayDeterminizeMode mode) noexcept {
    const std::uint64_t index = mode == SelfPlayDeterminizeMode::PerTurn
        ? static_cast<std::uint64_t>(turn_counter)
        : static_cast<std::uint64_t>(move_index);
    std::uint64_t mixed = game_seed ^ SELFPLAY_DETERMINIZATION_STREAM;
    mixed += SELFPLAY_DETERMINIZATION_SEAT_MIX * (static_cast<std::uint64_t>(seat) + 1U);
    mixed ^= SELFPLAY_DETERMINIZATION_INDEX_MIX * (index + 1U);
    return splitmix64(mixed);
}

float selfplay_temperature_for(
    const SelfPlayConfig& config,
    DecisionKind decision_kind,
    std::uint16_t move_index,
    std::uint16_t seat_decision_count,
    std::uint16_t seat_turn_number) noexcept {
    if (config.temp_mode == SelfPlayTempMode::Legacy) {
        return move_index < config.temp_moves ? 1.0F : config.temp_final;
    }

    if (decision_kind == DecisionKind::PhaseBuy) {
        return seat_turn_number <= config.temp_buy_turns ? 1.0F : config.temp_final;
    }
    if (decision_kind == DecisionKind::PhaseAction) {
        return seat_decision_count < config.temp_action_plies ? 1.0F : config.temp_final;
    }
    return seat_decision_count < config.temp_effect_plies ? 1.0F : config.temp_final;
}

std::uint16_t selfplay_seat_turn_number(std::uint16_t turn_counter) noexcept {
    return static_cast<std::uint16_t>((turn_counter / 2U) + 1U);
}

struct SelfPlayRunner::GameSlot {
    GameState state{};
    Setup setup{};
    Mcts mcts;
    Xoshiro256pp rng{};
    std::vector<float> observations;
    std::vector<float> policy_targets;
    std::vector<Action> sampled_actions;
    std::vector<std::uint64_t> legal_mask_words;
    std::vector<PlayerId> players;
    SelfPlaySlotConfig slot{};
    std::uint64_t seed = 0;
    std::uint64_t generation = 0;
    // Counts only simulations started for this decision. On a successfully
    // adopted tree, sims_target is a visit-target-derived new-sim quota.
    std::uint32_t sims_target = 0;
    std::uint32_t sims_started = 0;
    std::uint32_t sims_completed = 0;
    std::uint32_t pending = 0;
    std::uint16_t move_index = 0;
    std::uint16_t seat_decision_counts[MAX_PLAYERS]{};
    std::uint8_t seat_template_ids[MAX_PLAYERS]{};
    std::uint16_t opening_buy_counts[ACTION_DEF_COUNT]{};
    std::uint16_t unconstrained_buy_counts[SELFPLAY_OPENING_TELEMETRY_CARD_COUNT]{};
    PlayerId nn_player = NONE;
    bool search_active = false;
    // Manifest slots represent one prescribed game for a generation.  Once
    // complete they stay parked so a fast slot cannot begin an unassigned
    // follow-up game while another slot is still finishing.
    bool retired = false;
    // GameSlot is runner bookkeeping (it already owns Mcts and vectors), so
    // the atomic leaves the POD GameState/frame types untouched.
    std::atomic<ScriptedSlotStatus> scripted_status{ScriptedSlotStatus::Idle};

    GameSlot() : mcts(MctsConfig{}) {}
};

struct SelfPlayRunner::PendingLeaf {
    MctsPendingLeaf leaf{};
    std::uint32_t game = 0;
    std::uint32_t model_id = 0;
    bool root = false;
};

struct SelfPlayRunner::ScriptedPool {
    struct ScriptedBotSlot {
        // EngineV3 keeps arrays indexed by PlayerId. One fixed-size instance
        // per game slot prevents state from crossing game boundaries without
        // allocating on the decision path.
        EngineBotV3 engine_v3{};
        ThinnerBot thinner{};
    };

    ScriptedPool(
        SelfPlayRunner& runner,
        std::uint32_t slot_count,
        std::uint8_t thread_count,
        const MctsConfig& scratch_config)
        : runner_(runner),
          bots_(new ScriptedBotSlot[slot_count]),
          queue_(new std::uint32_t[slot_count]),
          queue_capacity_(slot_count) {
        scratches_.reserve(thread_count);
        for (std::uint32_t i = 0; i < thread_count; ++i) {
            scratches_.emplace_back(scratch_config);
        }

        workers_.reserve(thread_count);
        try {
            for (std::uint32_t i = 0; i < thread_count; ++i) {
                workers_.emplace_back([this, i]() noexcept { worker_loop(i); });
            }
        } catch (...) {
            stop();
            throw;
        }
    }

    ~ScriptedPool() {
        stop();
    }

    [[nodiscard]] bool enqueue(std::uint32_t slot) noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_size_ >= queue_capacity_ || stopping_) {
                return false;
            }
            queue_[queue_tail_] = slot;
            queue_tail_ = (queue_tail_ + 1U) % queue_capacity_;
            ++queue_size_;
        }
        ready_.notify_one();
        return true;
    }

    void reset_slot(std::uint32_t slot) noexcept {
        if (slot < queue_capacity_) {
            bots_[slot] = ScriptedBotSlot{};
        }
    }

    [[nodiscard]] Action choose_engine_v3(
        std::uint32_t slot,
        const GameState& state,
        const ActionMask& legal,
        int legal_count) noexcept {
        return slot < queue_capacity_
            ? bots_[slot].engine_v3.choose_action(state, legal, legal_count)
            : A_PASS;
    }

    [[nodiscard]] Action choose_thinner(
        std::uint32_t slot,
        const GameState& state,
        const ActionMask& legal,
        int legal_count) noexcept {
        return slot < queue_capacity_
            ? bots_[slot].thinner.choose_action(state, legal, legal_count)
            : A_PASS;
    }

    [[nodiscard]] bool has_workers() const noexcept {
        return !workers_.empty();
    }

    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    void worker_loop(std::uint32_t worker_index) noexcept {
        while (true) {
            std::uint32_t slot = 0U;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this]() { return stopping_ || queue_size_ != 0U; });
                if (queue_size_ == 0U) {
                    return;
                }
                slot = queue_[queue_head_];
                queue_head_ = (queue_head_ + 1U) % queue_capacity_;
                --queue_size_;
            }
            runner_.run_scripted_job(slot, scratches_[worker_index]);
        }
    }

    SelfPlayRunner& runner_;
    std::unique_ptr<ScriptedBotSlot[]> bots_;
    std::vector<Mcts> scratches_;
    std::vector<std::thread> workers_;
    std::unique_ptr<std::uint32_t[]> queue_;
    std::uint32_t queue_capacity_ = 0U;
    std::uint32_t queue_head_ = 0U;
    std::uint32_t queue_tail_ = 0U;
    std::uint32_t queue_size_ = 0U;
    std::mutex mutex_;
    std::condition_variable ready_;
    bool stopping_ = false;
};

bool is_selfplay_implemented_kingdom(DefId def) noexcept {
    return implemented_kingdom(def);
}

Action selfplay_opening_preferred_action(
    const GameState& state,
    PlayerId player,
    std::uint8_t template_id,
    std::int32_t opening_turn_window,
    const ActionMask& legal) noexcept {
    if (template_id == SELFPLAY_UNCONSTRAINED_TEMPLATE
        || template_id >= SELFPLAY_OPENING_TEMPLATE_COUNT
        || opening_turn_window < 0
        || static_cast<std::int32_t>(state.turn_counter) > opening_turn_window
        || player >= state.num_players) {
        return A_END;
    }
    if (state.phase == static_cast<std::uint8_t>(Phase::Buy)
        && state.decision.kind == static_cast<std::uint8_t>(DecisionKind::PhaseBuy)) {
        return opening_buy_preference(state, player, template_id, legal);
    }
    return opening_trash_preference(state, player, legal);
}

void selfplay_mix_opening_prior(
    float* priors,
    const ActionMask& legal,
    int legal_count,
    Action preferred_action,
    float lambda) noexcept {
    if (priors == nullptr
        || legal_count <= 0
        || preferred_action >= ACTION_SPACE_SIZE
        || !legal.test(preferred_action)
        || !(lambda > 0.0F)
        || !std::isfinite(lambda)) {
        return;
    }
    const float clamped_lambda = std::min(lambda, 1.0F);
    const float retained = 1.0F - clamped_lambda;
    double sum = 0.0;
    for (Action action = 0U; action < ACTION_SPACE_SIZE; ++action) {
        if (!legal.test(action)) {
            priors[action] = 0.0F;
            continue;
        }
        const float prior = priors[action] > 0.0F && std::isfinite(priors[action])
            ? priors[action]
            : 0.0F;
        const float mixed = retained * prior
            + (action == preferred_action ? clamped_lambda : 0.0F);
        priors[action] = mixed;
        sum += static_cast<double>(mixed);
    }
    if (sum <= 0.0) {
        return;
    }
    const float inv_sum = static_cast<float>(1.0 / sum);
    for (Action action = 0U; action < ACTION_SPACE_SIZE; ++action) {
        if (legal.test(action)) {
            priors[action] *= inv_sum;
        }
    }
}

SelfPlayRunner::SelfPlayRunner(const SelfPlayConfig& config)
    : config_(config),
      slot_count_(config.slot_manifest.empty()
          ? config.n_games
          : static_cast<std::uint32_t>(config.slot_manifest.size())),
      manifest_mode_(!config.slot_manifest.empty()),
      obs_size_(checked_obs_size(config.obs_version)),
      mcts_config_(),
      games_(new GameSlot[std::max(1U, slot_count_)]),
      pending_(new PendingLeaf[std::max(1U, config.max_batch)]),
      leaf_obs_(new float[static_cast<std::size_t>(std::max(1U, config.max_batch)) * obs_size_]),
      leaf_masks_(new bool[static_cast<std::size_t>(std::max(1U, config.max_batch)) * ACTION_SPACE_SIZE]),
      leaf_players_(new PlayerId[std::max(1U, config.max_batch)]),
      leaf_model_ids_(new std::uint32_t[std::max(1U, config.max_batch)]),
      leaf_game_indices_(new std::uint64_t[std::max(1U, config.max_batch)]),
      normalized_policy_(new float[static_cast<std::size_t>(std::max(1U, config.max_batch)) * ACTION_SPACE_SIZE]) {
    if (slot_count_ == 0U) {
        throw std::invalid_argument(
            manifest_mode_ ? "SelfPlayConfig.slot_manifest must not be empty"
                           : "SelfPlayConfig.n_games must be positive");
    }
    // Keep internal legacy helpers that still refer to n_games aligned with
    // the real slot count.  Callers retain the original object by value.
    config_.n_games = slot_count_;
    if (config_.max_batch == 0U) {
        throw std::invalid_argument("SelfPlayConfig.max_batch must be positive");
    }
    if (config_.sims_per_move == 0U) {
        throw std::invalid_argument("SelfPlayConfig.sims_per_move must be positive");
    }
    if (!mcts_is_valid_c_puct_schedule(config_.c_puct_schedule)) {
        throw std::invalid_argument("SelfPlayConfig.c_puct_schedule is invalid");
    }
    if (!(config_.c_puct_init > 0.0F) || !std::isfinite(config_.c_puct_init)) {
        throw std::invalid_argument("SelfPlayConfig.c_puct_init must be finite and positive");
    }
    if (!(config_.c_puct_base > 0.0F) || !std::isfinite(config_.c_puct_base)) {
        throw std::invalid_argument("SelfPlayConfig.c_puct_base must be finite and positive");
    }
    if (!(config_.forced_playouts_k > 0.0F) || !std::isfinite(config_.forced_playouts_k)) {
        throw std::invalid_argument("SelfPlayConfig.forced_playouts_k must be finite and positive");
    }
    if (!valid_determinize_mode(config_.determinize)) {
        throw std::invalid_argument("SelfPlayConfig.determinize is invalid");
    }
    if (!valid_temp_mode(config_.temp_mode)) {
        throw std::invalid_argument("SelfPlayConfig.temp_mode is invalid");
    }
    if (!(config_.temp_final >= 0.0F) || !std::isfinite(config_.temp_final)) {
        throw std::invalid_argument("SelfPlayConfig.temp_final must be finite and non-negative");
    }
    if (config_.determinize != SelfPlayDeterminizeMode::Off && config_.tree_reuse) {
        // A retained subtree belongs to the previous sampled hidden world.
        // Keep legacy configs usable, but make the unsafe request visible.
        std::fputs(
            "SelfPlayRunner: disabling tree_reuse because selfplay.determinize is enabled\n",
            stderr);
        config_.tree_reuse = false;
    }
    if (config_.kingdom_pool_count > MAX_SELFPLAY_KINGDOM_POOL) {
        throw std::invalid_argument("SelfPlayConfig.kingdom_pool has too many cards");
    }
    if (config_.kingdom_pool_count > 0U && config_.kingdom_pool_count < 10U) {
        throw std::invalid_argument("SelfPlayConfig.kingdom_pool must contain at least 10 cards");
    }
    for (std::uint8_t i = 0; i < config_.kingdom_pool_count; ++i) {
        const DefId def = config_.kingdom_pool[i];
        if (!implemented_kingdom(def)) {
            throw std::invalid_argument("SelfPlayConfig.kingdom_pool contains an unimplemented kingdom card");
        }
        for (std::uint8_t previous = 0; previous < i; ++previous) {
            if (config_.kingdom_pool[previous] == def) {
                throw std::invalid_argument("SelfPlayConfig.kingdom_pool contains duplicate cards");
            }
        }
    }
    if (!(config_.margin_scale > 0.0F) || !std::isfinite(config_.margin_scale)) {
        throw std::invalid_argument("SelfPlayConfig.margin_scale must be finite and positive");
    }
    if (!(config_.margin_blend_alpha >= 0.0F && config_.margin_blend_alpha <= 1.0F)
        || !std::isfinite(config_.margin_blend_alpha)) {
        throw std::invalid_argument("SelfPlayConfig.margin_blend_alpha must be finite and between zero and one");
    }
    if (config_.opening_templates_enabled) {
        if (!(config_.opening_lambda >= 0.0F && config_.opening_lambda <= 1.0F)
            || !std::isfinite(config_.opening_lambda)) {
            throw std::invalid_argument(
                "SelfPlayConfig.opening_lambda must be finite and between zero and one");
        }
        if (config_.opening_turn_window < 0) {
            throw std::invalid_argument("SelfPlayConfig.opening_turn_window must be non-negative");
        }
        double template_weight_sum = 0.0;
        for (const float weight : config_.template_weights) {
            if (!(weight >= 0.0F) || !std::isfinite(weight)) {
                throw std::invalid_argument(
                    "SelfPlayConfig.template_weights must be finite and non-negative");
            }
            template_weight_sum += static_cast<double>(weight);
        }
        if (!(template_weight_sum > 0.0) || !std::isfinite(template_weight_sum)) {
            throw std::invalid_argument("SelfPlayConfig.template_weights must have positive sum");
        }
    }
    if (config_.scripted_bot == SelfPlayScriptedBotKind::Scaffold && config_.scaffold_sims == 0U) {
        throw std::invalid_argument("SelfPlayConfig.scaffold_sims must be positive for Scaffold");
    }
    if (scripted_mode(config_) && config_.scripted_nn_player >= 2U) {
        throw std::invalid_argument("SelfPlayConfig.scripted_nn_player must be zero or one");
    }
    if (config_.fixed_setup.num_players == 0U) {
        config_.fixed_setup = default_fixed_setup();
    }
    config_.fixed_setup.num_players = 2U;

    for (std::size_t slot_index = 0; slot_index < config_.slot_manifest.size(); ++slot_index) {
        const SelfPlaySlotConfig& slot = config_.slot_manifest[slot_index];
        if (slot.kingdom_pool_count > MAX_SELFPLAY_KINGDOM_POOL) {
            throw std::invalid_argument("SelfPlaySlotConfig.kingdom_pool has too many cards");
        }
        if (slot.kingdom_pool_count > 0U && slot.kingdom_pool_count < 10U) {
            throw std::invalid_argument("SelfPlaySlotConfig.kingdom_pool must contain at least 10 cards");
        }
        for (std::uint8_t i = 0; i < slot.kingdom_pool_count; ++i) {
            const DefId def = slot.kingdom_pool[i];
            if (!implemented_kingdom(def)) {
                throw std::invalid_argument("SelfPlaySlotConfig.kingdom_pool contains an unimplemented kingdom card");
            }
            for (std::uint8_t previous = 0; previous < i; ++previous) {
                if (slot.kingdom_pool[previous] == def) {
                    throw std::invalid_argument("SelfPlaySlotConfig.kingdom_pool contains duplicate cards");
                }
            }
        }
        if (scripted_mode(slot.scripted_bot) && slot.scripted_nn_player >= 2U) {
            throw std::invalid_argument("SelfPlaySlotConfig.scripted_nn_player must be zero or one");
        }
        for (std::size_t previous = 0; previous < slot_index; ++previous) {
            if (config_.slot_manifest[previous].game_index == slot.game_index) {
                throw std::invalid_argument("SelfPlayConfig.slot_manifest contains duplicate game_index values");
            }
        }
    }

    mcts_config_.sims_per_move = config_.sims_per_move;
    mcts_config_.c_puct = config_.c_puct;
    mcts_config_.c_puct_schedule = config_.c_puct_schedule;
    mcts_config_.c_puct_init = config_.c_puct_init;
    mcts_config_.c_puct_base = config_.c_puct_base;
    mcts_config_.determinizations = 1U;
    mcts_config_.max_tree_nodes = config_.max_tree_nodes == 0U ? 4096U : config_.max_tree_nodes;
    mcts_config_.rollout_policy = MctsRolloutPolicy::External;
    mcts_config_.prune_treasure_plays = config_.prune_treasure_plays;
    mcts_config_.expand_top_k = config_.expand_top_k;
    mcts_config_.tree_reuse = config_.tree_reuse;
    mcts_config_.min_new_sims = config_.min_new_sims;
    mcts_config_.forced_playouts = config_.forced_playouts;
    mcts_config_.forced_playouts_k = config_.forced_playouts_k;
    if (mcts_config_.tree_reuse && mcts_config_.determinizations != 1U) {
        // A reused subtree belongs to one root-sampled hidden-information
        // world; K>1 trees aggregate different worlds below the root.
        throw std::invalid_argument("SelfPlayConfig.tree_reuse requires determinizations == 1");
    }
    const auto manifest_uses = [this](SelfPlayScriptedBotKind kind) {
        return std::any_of(
            config_.slot_manifest.begin(),
            config_.slot_manifest.end(),
            [kind](const SelfPlaySlotConfig& slot) {
                return slot.scripted_bot == kind;
            });
    };
    const bool uses_scaffold = manifest_mode_
        ? manifest_uses(SelfPlayScriptedBotKind::Scaffold)
        : config_.scripted_bot == SelfPlayScriptedBotKind::Scaffold;
    const bool uses_engine_v3 = manifest_mode_
        ? manifest_uses(SelfPlayScriptedBotKind::EngineV3)
        : config_.scripted_bot == SelfPlayScriptedBotKind::EngineV3;
    const bool uses_thinner = manifest_mode_
        ? manifest_uses(SelfPlayScriptedBotKind::Thinner)
        : config_.scripted_bot == SelfPlayScriptedBotKind::Thinner;
    if (uses_scaffold && config_.scaffold_sims == 0U) {
        throw std::invalid_argument("SelfPlayConfig.scaffold_sims must be positive for Scaffold");
    }
    if (config_.tree_reuse && uses_scaffold
        && config_.scaffold_determinizations > 1U) {
        // Keep the self-play/scaffold configuration unambiguous: subtree
        // reuse is valid only for a single determinized world, never a
        // root-level aggregate of K hidden-information samples.
        throw std::invalid_argument(
            "SelfPlayConfig.tree_reuse requires scaffold_determinizations <= 1");
    }

    if (uses_scaffold) {
        scaffold_mcts_config_ = make_scaffold_mcts_config(
            config_.scaffold_sims,
            config_.c_puct,
            mcts_config_.max_tree_nodes,
            config_.prune_treasure_plays,
            config_.scaffold_determinizations,
            config_.c_puct_schedule,
            config_.c_puct_init,
            config_.c_puct_base);
        if (config_.scripted_threads == 0U) {
            scaffold_mcts_.emplace(scaffold_mcts_config_);
        }
    }
    // EngineV3 needs the pool's fixed per-slot bot storage even when no
    // Scaffold worker is enabled. Passing zero workers keeps that state-only
    // pool allocation-free during moves and avoids spawning idle threads.
    if ((uses_scaffold && config_.scripted_threads != 0U) || uses_engine_v3 || uses_thinner) {
        scripted_pool_ = std::make_unique<ScriptedPool>(
            *this,
            slot_count_,
            uses_scaffold ? config_.scripted_threads : 0U,
            scaffold_mcts_config_);
    }

    finished_.reserve(slot_count_);
    for (std::uint32_t i = 0; i < slot_count_; ++i) {
        GameSlot& game = games_[i];
        if (manifest_mode_) {
            game.slot = config_.slot_manifest[i];
        } else {
            game.slot.game_index = i;
            game.slot.seat0_model_id = 0U;
            game.slot.seat1_model_id = 0U;
            game.slot.kingdom_mode = config_.kingdom_mode;
            game.slot.kingdom_pool_count = config_.kingdom_pool_count;
            for (std::uint8_t pool_index = 0; pool_index < config_.kingdom_pool_count; ++pool_index) {
                game.slot.kingdom_pool[pool_index] = config_.kingdom_pool[pool_index];
            }
            game.slot.sims_override = 0U;
            game.slot.scripted_bot = config_.scripted_bot;
            game.slot.scripted_nn_player = config_.scripted_nn_player;
        }
        game.mcts = Mcts(mcts_config_);
        game.mcts.set_sims_per_move(
            game.slot.sims_override == 0U ? config_.sims_per_move : game.slot.sims_override);
        game.observations.reserve(static_cast<std::size_t>(config_.max_recorded_moves) * obs_size_);
        game.policy_targets.reserve(
            static_cast<std::size_t>(config_.max_recorded_moves) * ACTION_SPACE_SIZE);
        game.sampled_actions.reserve(config_.max_recorded_moves);
        game.legal_mask_words.reserve(
            static_cast<std::size_t>(config_.max_recorded_moves) * ACTION_MASK_WORDS);
        game.players.reserve(config_.max_recorded_moves);
        reset_game(i);
    }
}

SelfPlayRunner::~SelfPlayRunner() {
    if (scripted_pool_) {
        scripted_pool_->stop();
    }
}

std::uint32_t SelfPlayRunner::collect_leaves(std::uint32_t max_batch) noexcept {
    if (pending_count_ != 0U) {
        return pending_count_;
    }
    // Ready jobs never touch shared runner counters. Reap them on the main
    // thread in slot order before building the next NN batch, which makes the
    // finished-record order less timing-dependent without coupling searches.
    drain_scripted_ready();
    const std::uint32_t limit = std::min(
        max_batch == 0U ? config_.max_batch : max_batch,
        config_.max_batch);
    pending_count_ = 0;
    std::uint32_t idle = 0;
    while (pending_count_ < limit && idle < slot_count_) {
        const std::uint32_t index = next_collect_game_;
        next_collect_game_ = (next_collect_game_ + 1U) % slot_count_;
        GameSlot& game = games_[index];

        if (game.retired) {
            ++idle;
            continue;
        }
        if (game.pending != 0U) {
            ++idle;
            continue;
        }
        if (offload_scripted_slot(index)) {
            ++idle;
            continue;
        }
        if (game.retired) {
            ++idle;
            continue;
        }
        drive_scripted(game);
        if (game.retired) {
            ++idle;
            continue;
        }
        if (game.pending != 0U) {
            ++idle;
            continue;
        }
        if (scripted_mode(game.slot.scripted_bot) && decision_player(game.state) != game.nn_player) {
            ++idle;
            continue;
        }
        maybe_finish_move(game);
        if (game.retired) {
            ++idle;
            continue;
        }
        if (game.pending != 0U) {
            ++idle;
            continue;
        }
        if (scripted_mode(game.slot.scripted_bot) && decision_player(game.state) != game.nn_player) {
            ++idle;
            continue;
        }
        if (!game.search_active) {
            auto_play_treasures(game);
        }
        if (game.retired) {
            ++idle;
            continue;
        }
        if (game.pending != 0U) {
            ++idle;
            continue;
        }
        if (scripted_mode(game.slot.scripted_bot) && decision_player(game.state) != game.nn_player) {
            ++idle;
            continue;
        }
        if (!game.search_active) {
            start_search(game);
        }
        if (game.sims_started >= game.sims_target) {
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
        if (scripted_mode(game.slot.scripted_bot) && leaf.player != game.nn_player) {
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
        pending.model_id = leaf.player == 0U ? game.slot.seat0_model_id : game.slot.seat1_model_id;
        pending.root = leaf.node == 0U;
        leaf_players_[pending_count_] = leaf.player;
        leaf_model_ids_[pending_count_] = pending.model_id;
        leaf_game_indices_[pending_count_] = game.slot.game_index;
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
    if (pending_count_ == 0U && scripted_pool_) {
        // A caller may immediately poll again while every slot is in a
        // scripted job. Give those CPU workers a scheduling opportunity
        // without delaying a non-empty NN batch.
        std::this_thread::yield();
    }
    return pending_count_;
}

void SelfPlayRunner::provide_evaluations(const float* values, const float* policies, std::uint32_t count) noexcept {
    const std::uint32_t n = std::min(count, pending_count_);
    for (std::uint32_t i = 0; i < n; ++i) {
        PendingLeaf& pending = pending_[i];
        GameSlot& game = games_[pending.game];
        float* normalized = normalized_policy_.get() + (i * ACTION_SPACE_SIZE);
        Action opening_preference = A_END;
        if (config_.opening_templates_enabled && pending.root
            && pending.leaf.player < MAX_PLAYERS) {
            const GameState& root_state = game.mcts.state_for(pending.leaf.state_index);
            opening_preference = selfplay_opening_preferred_action(
                root_state,
                pending.leaf.player,
                game.seat_template_ids[pending.leaf.player],
                config_.opening_turn_window,
                pending.leaf.legal);
        }
        normalize_policy(
            policies == nullptr ? nullptr : policies + (i * ACTION_SPACE_SIZE),
            pending.leaf.legal,
            pending.leaf.legal_count,
            pending.root,
            opening_preference,
            normalized,
            game.rng);
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

const float* SelfPlayRunner::leaf_observations() const noexcept {
    return leaf_obs_.get();
}

const bool* SelfPlayRunner::leaf_legal_masks() const noexcept {
    return leaf_masks_.get();
}

const PlayerId* SelfPlayRunner::leaf_players() const noexcept {
    return leaf_players_.get();
}

const std::uint32_t* SelfPlayRunner::leaf_model_ids() const noexcept {
    return leaf_model_ids_.get();
}

const std::uint64_t* SelfPlayRunner::leaf_game_indices() const noexcept {
    return leaf_game_indices_.get();
}

std::uint32_t SelfPlayRunner::leaf_count() const noexcept {
    return pending_count_;
}

std::size_t SelfPlayRunner::observation_size() const noexcept {
    return obs_size_;
}

std::uint64_t SelfPlayRunner::games_completed() const noexcept {
    return completed_;
}

float SelfPlayRunner::total_virtual_loss() const noexcept {
    float total = 0.0F;
    for (std::uint32_t i = 0; i < slot_count_; ++i) {
        total += games_[i].mcts.total_virtual_loss();
    }
    return total;
}

SelfPlaySearchStats SelfPlayRunner::search_stats(std::uint32_t index) const noexcept {
    SelfPlaySearchStats stats{};
    if (index >= slot_count_) {
        return stats;
    }

    const GameSlot& game = games_[index];
    stats.sims_target = game.sims_target;
    stats.sims_started = game.sims_started;
    stats.sims_completed = game.sims_completed;
    stats.root_visits = game.mcts.node_count() == 0U ? 0U : game.mcts.node(0U).visits;
    stats.search_active = game.search_active;
    return stats;
}

const std::vector<SelfPlayRecord>& SelfPlayRunner::finished_games() const noexcept {
    return finished_;
}

std::vector<SelfPlayRecord> SelfPlayRunner::take_finished_games() {
    std::vector<SelfPlayRecord> out = std::move(finished_);
    finished_.clear();
    finished_.reserve(slot_count_);
    return out;
}

void SelfPlayRunner::reset_game(std::uint32_t index) noexcept {
    GameSlot& game = games_[index];
    game.scripted_status.store(ScriptedSlotStatus::Idle, std::memory_order_relaxed);
    if (scripted_pool_) {
        scripted_pool_->reset_slot(index);
    }
    game.mcts.clear_retained_root();
    game.setup = setup_for(game);
    game.seed = game_seed(config_, game.slot.game_index, game.generation);
    game.state = Game::new_game(game.setup, game.seed);
    game.rng = Xoshiro256pp::seeded(game.seed ^ 0x53E1'F019'0000'0001ULL);
    std::memset(game.seat_template_ids, 0, sizeof(game.seat_template_ids));
    std::memset(game.opening_buy_counts, 0, sizeof(game.opening_buy_counts));
    std::memset(game.unconstrained_buy_counts, 0, sizeof(game.unconstrained_buy_counts));
    if (config_.opening_templates_enabled) {
        // Keep template sampling independent of the MCTS/noise RNG stream:
        // templates alter priors by design, but their assignment should not
        // also shift unrelated random samples.
        Xoshiro256pp template_rng = Xoshiro256pp::seeded(
            game.seed ^ 0x4F50'454E'494E'4731ULL);
        for (PlayerId player = 0U; player < game.state.num_players; ++player) {
            game.seat_template_ids[player] = draw_opening_template(config_, template_rng);
        }
    }
    game.observations.clear();
    game.policy_targets.clear();
    game.sampled_actions.clear();
    game.legal_mask_words.clear();
    game.players.clear();
    game.sims_target = 0;
    game.sims_started = 0;
    game.sims_completed = 0;
    game.pending = 0;
    game.move_index = 0;
    std::memset(game.seat_decision_counts, 0, sizeof(game.seat_decision_counts));
    game.nn_player = scripted_mode(game.slot.scripted_bot) ? game.slot.scripted_nn_player : NONE;
    game.search_active = false;
    game.retired = false;
    game.mcts.set_sims_per_move(
        game.slot.sims_override == 0U ? config_.sims_per_move : game.slot.sims_override);
}

void SelfPlayRunner::start_search(GameSlot& game) noexcept {
    assert(game.pending == 0U);
    const PlayerId player = decision_player(game.state);
    bool reused = false;
    if (config_.determinize != SelfPlayDeterminizeMode::Off) {
        // Search only in a freshly sampled world.  The live state remains the
        // source of records and receives the selected root action below.
        GameState sampled = game.state;
        determinize(
            sampled,
            player,
            selfplay_determinization_seed(
                game.seed,
                player,
                game.move_index,
                game.state.turn_counter,
                config_.determinize));
        // Tree reuse was disabled at construction: a retained child would
        // encode a prior sampled world and cannot be hash-checked against the
        // real state's hidden zones.
        game.mcts.reset(sampled, player);
    } else {
        reused = config_.tree_reuse
            && game.mcts.adopt_retained_root(game.state, player);
        if (reused) {
            reapply_root_noise(game);
        } else {
            game.mcts.reset(game.state, player);
        }
    }
    // Forced root selection is meaningful only for the same training roots
    // that received Dirichlet exploration. Eval and duel roots leave this off.
    game.mcts.set_root_dirichlet_noise_active(config_.dirichlet_frac > 0.0F);
    // Keep the counters below strictly about newly started work so parked
    // leaves and batch completion remain balanced. Only a hash-gated adopted
    // root turns sims_per_move into a total root-visit target.
    game.sims_target = game.mcts.new_simulation_target(reused);
    game.sims_started = 0;
    game.sims_completed = 0;
    game.pending = 0;
    game.search_active = true;
}

void SelfPlayRunner::reapply_root_noise(GameSlot& game) noexcept {
    if (game.mcts.node_count() == 0U) {
        return;
    }
    const bool apply_noise = config_.dirichlet_frac > 0.0F;
    const PlayerId player = decision_player(game.state);
    const bool apply_template = config_.opening_templates_enabled && player < MAX_PLAYERS;
    if (!apply_noise && !apply_template) {
        return;
    }

    float priors[ACTION_SPACE_SIZE]{};
    ActionMask retained{};
    float sum = 0.0F;
    int count = 0;
    const MctsNode& root = game.mcts.node(0U);
    // A reused interior node is deliberately marked unexpanded so a fresh
    // root evaluation restores every legal child before Dirichlet noise.
    if (!root.expanded) {
        return;
    }
    for (std::uint32_t child_index = root.first_child; child_index != MCTS_NULL;
         child_index = game.mcts.node(child_index).next_sibling) {
        const MctsNode& child = game.mcts.node(child_index);
        if (child.action_from_parent >= ACTION_SPACE_SIZE) {
            continue;
        }
        retained.set(child.action_from_parent);
        priors[child.action_from_parent] = child.prior > 0.0F ? child.prior : 0.0F;
        sum += priors[child.action_from_parent];
        ++count;
    }
    if (count <= 1) {
        return;
    }
    if (sum <= 0.0F) {
        const float uniform = 1.0F / static_cast<float>(count);
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            if (retained.test(action)) {
                priors[action] = uniform;
            }
        }
    } else {
        const float inv_sum = 1.0F / sum;
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            if (retained.test(action)) {
                priors[action] *= inv_sum;
            }
        }
    }
    if (apply_template) {
        const Action preference = selfplay_opening_preferred_action(
            game.state,
            player,
            game.seat_template_ids[player],
            config_.opening_turn_window,
            retained);
        selfplay_mix_opening_prior(
            priors,
            retained,
            count,
            preference,
            config_.opening_lambda);
    }
    if (apply_noise) {
        mcts_add_dirichlet_noise(
            priors,
            retained,
            count,
            config_.dirichlet_alpha,
            config_.dirichlet_frac,
            game.rng);
    }
    game.mcts.set_root_priors(priors);
}

void SelfPlayRunner::auto_play_treasures(GameSlot& game) noexcept {
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
        game.mcts.clear_retained_root();
        ++guard;
        if (done || game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            finish_game(game);
            return;
        }
    }
}

void SelfPlayRunner::drive_scripted(GameSlot& game) noexcept {
    if (!scripted_mode(game.slot.scripted_bot)) {
        return;
    }

    if (game.slot.scripted_bot == SelfPlayScriptedBotKind::Scaffold) {
        // Async Scaffold jobs own their slots through scripted_pool_. The
        // synchronous reference path keeps one runner-local scratch tree.
        if (!scaffold_mcts_.has_value()) {
            return;
        }
        drive_scaffold(game, *scaffold_mcts_);
        if (game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            finish_game(game);
        }
        return;
    }

    std::uint16_t guard = 0;
    while (game.state.phase != static_cast<std::uint8_t>(Phase::Over)
           && decision_player(game.state) != game.nn_player
           && guard < 512U) {
        ActionMask legal{};
        const int legal_count = Game::legal_actions(game.state, legal);
        if (legal_count <= 0) {
            break;
        }
        Action action = choose_scripted_action(game, game.state, legal, legal_count);
        if (!legal.test(action)) {
            action = legal.nth_set(0U);
        }
        const bool done = Game::step(game.state, action);
        game.mcts.clear_retained_root();
        ++guard;
        if (done || game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            finish_game(game);
            break;
        }
    }
}

Action SelfPlayRunner::choose_scripted_action(
    GameSlot& game,
    const GameState& state,
    const ActionMask& legal,
    int legal_count) noexcept {
    // Match EvalRunner's EngineV3 lifecycle: the bot owns state for both
    // player indices in this one ScriptedPool slot and is reset by
    // reset_game().
    // Older scripted kinds retain the established eval-chart implementation,
    // including Engine's pile-clock guarded buy behavior.
    if (game.slot.scripted_bot == SelfPlayScriptedBotKind::EngineV3) {
        assert(scripted_pool_);
        if (!scripted_pool_) {
            return A_PASS;
        }
        const auto slot = static_cast<std::uint32_t>(&game - games_.get());
        return scripted_pool_->choose_engine_v3(slot, state, legal, legal_count);
    }
    if (game.slot.scripted_bot == SelfPlayScriptedBotKind::Thinner) {
        assert(scripted_pool_);
        if (!scripted_pool_) {
            return A_PASS;
        }
        const auto slot = static_cast<std::uint32_t>(&game - games_.get());
        return scripted_pool_->choose_thinner(slot, state, legal, legal_count);
    }
    return eval_scripted_action(
        state,
        legal,
        legal_count,
        eval_scripted_kind(game.slot.scripted_bot),
        game.rng);
}

void SelfPlayRunner::drive_scaffold(GameSlot& game, Mcts& scratch) noexcept {
    std::uint16_t guard = 0;
    while (game.state.phase != static_cast<std::uint8_t>(Phase::Over)
           && decision_player(game.state) != game.nn_player
           && guard < 512U) {
        ActionMask legal{};
        const int legal_count = Game::legal_actions(game.state, legal);
        if (legal_count <= 0) {
            break;
        }

        // Re-seed every search from the game seed exactly as the serialized
        // path did. The scratch's worker identity therefore cannot influence
        // a game's trajectory.
        scratch.set_rollout_seed(scaffold_rollout_seed(game.seed));
        scratch.set_sims_per_move(scaffold_sims_for(game.state, legal));
        Action action = eval_scaffold_mcts_action(scratch, game.state, legal, legal_count);
        if (!legal.test(action)) {
            action = legal.nth_set(0U);
        }
        const bool done = Game::step(game.state, action);
        game.mcts.clear_retained_root();
        ++guard;
        if (done || game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            break;
        }
    }
}

bool SelfPlayRunner::offload_scripted_slot(std::uint32_t index) noexcept {
    if (!scripted_pool_ || !scripted_pool_->has_workers()
        || games_[index].slot.scripted_bot != SelfPlayScriptedBotKind::Scaffold) {
        return false;
    }

    GameSlot& game = games_[index];
    const ScriptedSlotStatus status = game.scripted_status.load(std::memory_order_acquire);
    if (status == ScriptedSlotStatus::ScriptedPending
        || status == ScriptedSlotStatus::ScriptedRunning) {
        return true;
    }
    if (status == ScriptedSlotStatus::ScriptedReady) {
        // The acquire load above pairs with the worker's Ready store, so the
        // main thread now owns all of the job's GameSlot writes.
        game.scripted_status.store(ScriptedSlotStatus::Idle, std::memory_order_release);
        if (game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            finish_game(game);
        }
        return false;
    }

    if (game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
        finish_game(game);
        return false;
    }
    if (decision_player(game.state) == game.nn_player) {
        return false;
    }

    game.scripted_status.store(ScriptedSlotStatus::ScriptedPending, std::memory_order_release);
    if (!scripted_pool_->enqueue(index)) {
        // This should be unreachable: every slot can be queued at most once,
        // and the ring has one entry per slot. Keep the slot retryable rather
        // than letting a failed enqueue strand it in Pending.
        game.scripted_status.store(ScriptedSlotStatus::Idle, std::memory_order_release);
        return false;
    }
    return true;
}

void SelfPlayRunner::drain_scripted_ready() noexcept {
    if (!scripted_pool_) {
        return;
    }
    for (std::uint32_t index = 0; index < slot_count_; ++index) {
        GameSlot& game = games_[index];
        if (game.scripted_status.load(std::memory_order_acquire) != ScriptedSlotStatus::ScriptedReady) {
            continue;
        }
        game.scripted_status.store(ScriptedSlotStatus::Idle, std::memory_order_release);
        if (game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            finish_game(game);
        }
    }
}

void SelfPlayRunner::run_scripted_job(std::uint32_t index, Mcts& scratch) noexcept {
    if (index >= slot_count_) {
        return;
    }
    GameSlot& game = games_[index];
    game.scripted_status.store(ScriptedSlotStatus::ScriptedRunning, std::memory_order_release);
    // This is deliberately limited to the slot and the worker's scratch.
    // finish_game(), completed_, and finished_ remain main-thread-only.
    drive_scaffold(game, scratch);
    game.scripted_status.store(ScriptedSlotStatus::ScriptedReady, std::memory_order_release);
}

std::uint32_t SelfPlayRunner::scaffold_sims_for(
    const GameState& state,
    const ActionMask& legal) const noexcept {
    if (config_.scaffold_sims_opening == 0U || scaffold_endgame_armed(state, legal)) {
        return config_.scaffold_sims;
    }
    return config_.scaffold_sims_opening;
}

bool SelfPlayRunner::game_has_pending(std::uint32_t index) const noexcept {
    return index < slot_count_ && games_[index].pending != 0U;
}

void selfplay_pruned_root_visit_policy(
    const Mcts& search,
    float forced_playouts_k,
    float* out) noexcept {
    if (out == nullptr) {
        return;
    }
    // Preserve Mcts's established zero-visit and degenerate-root fallbacks.
    search.root_visit_policy(out, 1.0F);
    if (search.node_count() == 0U) {
        return;
    }

    const MctsNode& root = search.node(0U);
    if (root.first_child == MCTS_NULL) {
        return;
    }
    const Action best_action = search.best_root_action();
    const std::uint32_t total_visits = root.visits;
    double retained_sum = 0.0;
    for (std::uint32_t child_index = root.first_child; child_index != MCTS_NULL;
         child_index = search.node(child_index).next_sibling) {
        const MctsNode& child = search.node(child_index);
        std::uint32_t retained = child.visits;
        if (child.action_from_parent != best_action) {
            // Recompute rather than incrementally track quotas. Because root
            // visits only grow, this final quota can slightly exceed what was
            // actually forced during search; that upper-bound approximation is
            // simpler and matches the spirit of KataGo's published rule.
            const std::uint32_t forced = mcts_forced_playout_visits(
                child.prior,
                total_visits,
                forced_playouts_k);
            retained = child.visits > forced ? child.visits - forced : 0U;
        }
        retained_sum += static_cast<double>(retained);
    }
    if (retained_sum <= 0.0) {
        // The raw-policy fallback above is already normalized when no child
        // has visit mass left after pruning.
        return;
    }

    const float scale = static_cast<float>(1.0 / retained_sum);
    for (std::uint32_t child_index = root.first_child; child_index != MCTS_NULL;
         child_index = search.node(child_index).next_sibling) {
        const MctsNode& child = search.node(child_index);
        std::uint32_t retained = child.visits;
        if (child.action_from_parent != best_action) {
            const std::uint32_t forced = mcts_forced_playout_visits(
                child.prior,
                total_visits,
                forced_playouts_k);
            retained = child.visits > forced ? child.visits - forced : 0U;
        }
        out[child.action_from_parent] = static_cast<float>(retained) * scale;
    }
}

void SelfPlayRunner::maybe_finish_move(GameSlot& game) noexcept {
    if (!game.search_active || game.pending != 0U || game.sims_completed < game.sims_target) {
        return;
    }

    float policy[ACTION_SPACE_SIZE]{};
    // Re-rooting preserves child visit counts. Policy targets intentionally
    // use those full inherited-plus-new counts as legitimate search evidence.
    if (config_.forced_playouts && config_.dirichlet_frac > 0.0F) {
        selfplay_pruned_root_visit_policy(game.mcts, config_.forced_playouts_k, policy);
    } else {
        // Keep the historical target call path byte-identical when disabled.
        game.mcts.root_visit_policy(policy, 1.0F);
    }
    ActionMask legal{};
    const int legal_count = Game::legal_actions(game.state, legal);
    record_decision(game, policy, legal);

    const PlayerId player = decision_player(game.state);
    const std::uint16_t seat_decision_count = player < MAX_PLAYERS
        ? game.seat_decision_counts[player]
        : 0U;
    // turn_counter is incremented after a player cleans up. With the fixed
    // two-player turn queue, counters 0/1 are the two players' first turns,
    // 2/3 their second, and so on. Buy decisions always belong to that live
    // turn, so this is the acting seat's one-based turn number.
    const std::uint16_t seat_turn_number = selfplay_seat_turn_number(game.state.turn_counter);
    const float temperature = selfplay_temperature_for(
        config_,
        static_cast<DecisionKind>(game.state.decision.kind),
        game.move_index,
        seat_decision_count,
        seat_turn_number);
    Action action = game.mcts.sample_root_action(temperature, game.rng);
    if (legal_count <= 0 || !legal.test(action)) {
        action = legal_count > 0 ? legal.nth_set(0U) : A_PASS;
    }

    if (action_is_buy(action)) {
        const DefId def = action_def(action, A_BUY_BASE);
        if (config_.opening_templates_enabled
            && static_cast<std::int32_t>(game.state.turn_counter) <= config_.opening_turn_window
            && def < ACTION_DEF_COUNT) {
            ++game.opening_buy_counts[def];
        }
        if (config_.opening_templates_enabled
            && player < MAX_PLAYERS
            && game.seat_template_ids[player] == SELFPLAY_UNCONSTRAINED_TEMPLATE) {
            const int telemetry_index = opening_telemetry_index(def);
            if (telemetry_index >= 0) {
                ++game.unconstrained_buy_counts[static_cast<std::size_t>(telemetry_index)];
            }
        }
    }

    game.sampled_actions.push_back(action);
    const bool done = Game::step(game.state, action);
    if (player < MAX_PLAYERS
        && game.seat_decision_counts[player] < std::numeric_limits<std::uint16_t>::max()) {
        ++game.seat_decision_counts[player];
    }
    if (config_.tree_reuse && !done
        && game.state.phase != static_cast<std::uint8_t>(Phase::Over)) {
        // Store the runner's post-step hash with the selected child. The next
        // search adopts it only when this exact state still survives all
        // intervening engine/scripted work.
        (void)game.mcts.retain_root_child(action, mcts_state_hash(game.state));
    } else {
        game.mcts.clear_retained_root();
    }
    ++game.move_index;
    game.search_active = false;
    if (done || game.state.phase == static_cast<std::uint8_t>(Phase::Over)) {
        finish_game(game);
    }
}

void SelfPlayRunner::record_decision(
    GameSlot& game,
    const float* policy,
    const ActionMask& legal) noexcept {
    const std::uint16_t recorded = static_cast<std::uint16_t>(game.players.size());
    if (recorded >= config_.max_recorded_moves) {
        return;
    }
    const PlayerId player = decision_player(game.state);
    if (scripted_mode(game.slot.scripted_bot) && player != game.nn_player) {
        return;
    }
    const std::size_t obs_offset = game.observations.size();
    game.observations.resize(obs_offset + obs_size_);
    // This is intentionally the live, true state rather than the sampled
    // search root. Determinize preserves this player's public observation;
    // training records must never expose a sampled opponent arrangement.
    encode(game.state, player, game.observations.data() + obs_offset, config_.obs_version);

    const std::size_t policy_offset = game.policy_targets.size();
    game.policy_targets.resize(policy_offset + ACTION_SPACE_SIZE);
    std::memcpy(
        game.policy_targets.data() + policy_offset,
        policy,
        sizeof(float) * ACTION_SPACE_SIZE);
    game.legal_mask_words.insert(
        game.legal_mask_words.end(),
        legal.words,
        legal.words + ACTION_MASK_WORDS);
    game.players.push_back(player);
}

void SelfPlayRunner::finish_game(GameSlot& game) noexcept {
    if (game.retired) {
        return;
    }
    SelfPlayRecord record{};
    record.observations = game.observations;
    record.policy_targets = game.policy_targets;
    record.sampled_actions = game.sampled_actions;
    record.legal_mask_words = game.legal_mask_words;
    record.players = game.players;
    record.moves = static_cast<std::uint16_t>(game.players.size());
    record.values.resize(record.moves);
    record.margins.resize(record.moves);
    for (std::uint16_t i = 0; i < record.moves; ++i) {
        const int terminal_margin = terminal_margin_for(game.state, game.players[i]);
        // Int16 is deliberate for replay density. Dominion score differences
        // encountered in normal games are much smaller; retain a defined
        // representation for pathological terminal states as well.
        record.margins[i] = static_cast<std::int16_t>(std::clamp(terminal_margin, -127, 127));
        record.values[i] = terminal_value_for(game.state, game.players[i], terminal_margin);
    }
    record.seed = game.seed;
    record.winner = winner_for(game.state);
    for (PlayerId player = 0U; player < game.state.num_players; ++player) {
        record.scores[player] = score(game.state, player);
    }
    record.scripted_nn_player = game.nn_player;
    record.game_index = game.slot.game_index;
    record.seat0_model_id = game.slot.seat0_model_id;
    record.seat1_model_id = game.slot.seat1_model_id;
    record.scripted_bot = game.slot.scripted_bot;
    record.sims_override = game.slot.sims_override;
    std::memcpy(
        record.seat_template_ids,
        game.seat_template_ids,
        sizeof(record.seat_template_ids));
    std::memcpy(
        record.opening_buy_counts,
        game.opening_buy_counts,
        sizeof(record.opening_buy_counts));
    std::memcpy(
        record.unconstrained_buy_counts,
        game.unconstrained_buy_counts,
        sizeof(record.unconstrained_buy_counts));
    for (Slot slot = 0U; slot < game.state.num_slots; ++slot) {
        record.cards_trashed = static_cast<std::uint16_t>(
            record.cards_trashed + game.state.trash[slot]);
    }
    record.kingdom_count = game.setup.kingdom_count;
    for (std::uint8_t i = 0; i < game.setup.kingdom_count; ++i) {
        record.kingdom[i] = game.setup.kingdom[i];
    }
    finished_.push_back(std::move(record));

    ++completed_;
    if (manifest_mode_) {
        game.retired = true;
        game.search_active = false;
        return;
    }
    ++game.generation;
    reset_game(static_cast<std::uint32_t>(&game - games_.get()));
}

bool SelfPlayRunner::resolve_scripted_tree_leaf(GameSlot& game, const MctsPendingLeaf& leaf) noexcept {
    const GameState& leaf_state = game.mcts.state_for(leaf.state_index);
    Action action = A_PASS;
    if (game.slot.scripted_bot == SelfPlayScriptedBotKind::Scaffold) {
        // Actual moves keep full Scaffold fidelity in drive_scripted; charting
        // its tree leaves as Engine avoids batch stalls (2026-07-11: 36+ min
        // for ~10 Scaffold games in 1,024, versus a ~5 min baseline).
        action = eval_scripted_action(
            leaf_state,
            leaf.legal,
            leaf.legal_count,
            EvalScriptedBotKind::Engine,
            game.rng);
    } else {
        action = choose_scripted_action(game, leaf_state, leaf.legal, leaf.legal_count);
    }
    if (!leaf.legal.test(action)) {
        action = leaf.legal_count > 0 ? leaf.legal.nth_set(0U) : A_PASS;
    }
    float priors[ACTION_SPACE_SIZE]{};
    priors[action] = 1.0F;
    game.mcts.provide_external_evaluation(leaf, 0.0F, priors, game.rng);
    return true;
}

Setup SelfPlayRunner::setup_for(const GameSlot& game) const noexcept {
    if (game.slot.kingdom_mode == SelfPlayKingdomMode::Fixed) {
        Setup setup = config_.fixed_setup;
        setup.num_players = 2U;
        return setup;
    }

    Setup setup{};
    setup.num_players = 2U;
    setup.kingdom_count = 10U;
    const DefId* kingdom_pool = IMPLEMENTED_KINGDOMS;
    std::uint8_t kingdom_pool_count = IMPLEMENTED_KINGDOM_COUNT;
    if (game.slot.kingdom_pool_count > 0U) {
        kingdom_pool = game.slot.kingdom_pool;
        kingdom_pool_count = game.slot.kingdom_pool_count;
    }
    DefId defs[MAX_SELFPLAY_KINGDOM_POOL]{};
    for (std::uint8_t i = 0; i < kingdom_pool_count; ++i) {
        defs[i] = kingdom_pool[i];
    }
    Xoshiro256pp rng = Xoshiro256pp::seeded(
        config_.seed
        ^ ((game.slot.game_index + 1U) * 0xBADC'0FFE'1234'5678ULL)
        ^ (game.generation * 0x9E37'79B9'7F4A'7C15ULL));
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        const std::uint32_t offset = rng.uniform(static_cast<std::uint32_t>(kingdom_pool_count - i));
        const std::uint8_t swap_index = static_cast<std::uint8_t>(i + offset);
        const DefId selected = defs[swap_index];
        defs[swap_index] = defs[i];
        defs[i] = selected;
        setup.kingdom[i] = selected;
    }
    return setup;
}

int SelfPlayRunner::terminal_margin_for(const GameState& state, PlayerId player) const noexcept {
    const PlayerId opponent = static_cast<PlayerId>(player == 0U ? 1U : 0U);
    return static_cast<int>(score(state, player)) - static_cast<int>(score(state, opponent));
}

float SelfPlayRunner::terminal_value_for(
    const GameState& state,
    PlayerId player,
    int terminal_margin) const noexcept {
    const PlayerId winner = winner_for(state);
    if (config_.value_target == SelfPlayValueTarget::Margin) {
        // Truncated games have no training outcome. Keep record.winner based
        // on the final board for counters and gates, but do not turn that
        // partial score into a value target.
        if (state.truncated != 0U || winner == NONE) {
            return 0.0F;
        }
        const float margin = static_cast<float>(terminal_margin);
        if (margin == 0.0F) {
            return 0.0F;
        }
        // Sign-preserving hybrid: every win is worth at least +0.5 so narrow
        // wins (a legal, often correct outcome — e.g. a well-executed pile
        // race) never train as near-ties; margin adds gradient WITHIN the
        // win/loss categories so crushes teach more than squeakers.
        const float scale = config_.margin_scale;
        const float sign = margin > 0.0F ? 1.0F : -1.0F;
        const float graded = std::clamp(std::abs(margin), 0.0F, scale) / scale;
        return sign * (0.5F + 0.5F * graded);
    }
    if (config_.value_target == SelfPlayValueTarget::MarginBlend) {
        // Truncated games have no training outcome. Keep record.winner based
        // on the final board for counters and gates, but do not turn that
        // partial score into a value target.
        if (state.truncated != 0U || winner == NONE) {
            return 0.0F;
        }
        const float margin = static_cast<float>(terminal_margin);
        return selfplay_margin_blend_value(
            margin,
            config_.margin_scale,
            config_.margin_blend_alpha);
    }
    if (winner == NONE) {
        return 0.0F;
    }
    return winner == player ? 1.0F : -1.0F;
}

float selfplay_margin_blend_value(
    float margin,
    float margin_scale,
    float margin_blend_alpha) noexcept {
    if (margin == 0.0F) {
        return 0.0F;
    }
    const float sign = margin > 0.0F ? 1.0F : -1.0F;
    const float graded = std::clamp(std::abs(margin), 0.0F, margin_scale) / margin_scale;
    const float margin_value = 0.5F + 0.5F * graded;
    // Preserve the endpoint behavior exactly: alpha=0 is the established
    // margin formula, while alpha=1 is the established win/loss target.
    if (margin_blend_alpha == 0.0F) {
        return sign * margin_value;
    }
    if (margin_blend_alpha == 1.0F) {
        return sign;
    }
    return sign * (
        margin_blend_alpha
        + (1.0F - margin_blend_alpha) * margin_value);
}

void SelfPlayRunner::normalize_policy(
    const float* logits,
    const ActionMask& legal,
    int legal_count,
    bool add_root_noise,
    Action opening_preference,
    float* out,
    Xoshiro256pp& rng) noexcept {
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
    } else {
        const float inv_sum = static_cast<float>(1.0 / sum);
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            out[action] *= inv_sum;
        }
    }

    // Template mixing occurs only for the root leaf supplied above, after
    // NN softmax and before root Dirichlet exploration. Interior leaves pass
    // A_END and therefore retain the historical policy path exactly.
    selfplay_mix_opening_prior(
        out,
        legal,
        legal_count,
        opening_preference,
        config_.opening_lambda);

    if (!add_root_noise || config_.dirichlet_frac <= 0.0F || legal_count <= 1) {
        return;
    }

    // Roots are always full-width. Noise must cover the same full legal set
    // so policy targets retain the original Dirichlet exploration guarantee.
    mcts_add_dirichlet_noise(
        out,
        legal,
        legal_count,
        config_.dirichlet_alpha,
        config_.dirichlet_frac,
        rng);
}
