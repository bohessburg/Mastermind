#include "v2/core/state_builder.h"

#include "v2/core/actions.h"
#include "v2/core/defs.h"
#include "v2/core/interp.h"
#include "v2/core/setup.h"
#include "v2/core/turns.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

constexpr int DATA_ACTIVE_PC = 5;
constexpr int DATA_ACTIVE_COUNT = 6;
constexpr std::int16_t ACTIVE_PC_NONE = -1;

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("snapshot: " + message);
}

[[noreturn]] void corrupt(const std::string& message) {
    throw std::logic_error("game validation failed: " + message);
}

[[nodiscard]] std::uint32_t count_cards(const SnapshotCardCounts& counts) noexcept {
    std::uint32_t total = 0;
    for (std::uint8_t def = 0; def < MAX_SLOTS; ++def) {
        total += counts.by_def[def];
    }
    return total;
}

void check_known_defs(const SnapshotCardCounts& counts, const char* zone) {
    for (std::uint8_t def = static_cast<std::uint8_t>(card_def_count());
         def < MAX_SLOTS;
         ++def) {
        if (counts.by_def[def] != 0U) {
            invalid(std::string(zone) + " contains unknown def " + std::to_string(def));
        }
    }
}

[[nodiscard]] Slot add_slot(GameState& state, DefId def) {
    const Slot existing = slot_of(state, def);
    if (existing != NONE) {
        return existing;
    }
    if (def >= card_def_count()) {
        invalid("unknown card def " + std::to_string(def));
    }
    if (state.num_slots >= MAX_SLOTS) {
        invalid("too many distinct card definitions");
    }
    const Slot slot = state.num_slots;
    state.slot_to_def[slot] = def;
    ++state.num_slots;
    return slot;
}

void append_ordered(
    GameState& state,
    OrderedZone& zone,
    DefId def,
    std::uint16_t count,
    const char* name) {
    if (count == 0U) {
        return;
    }
    const Slot slot = add_slot(state, def);
    for (std::uint16_t i = 0; i < count; ++i) {
        if (zone.size >= MAX_DECK_CARDS) {
            invalid(std::string(name) + " exceeds MAX_DECK_CARDS");
        }
        zone.cards[zone.size] = slot;
        ++zone.size;
    }
}

void fill_ordered(
    GameState& state,
    OrderedZone& zone,
    const SnapshotCardCounts& counts,
    const char* name) {
    if (count_cards(counts) > MAX_DECK_CARDS) {
        invalid(std::string(name) + " exceeds MAX_DECK_CARDS");
    }
    for (DefId def = 0; def < card_def_count(); ++def) {
        append_ordered(state, zone, def, counts.by_def[def], name);
    }
}

void fill_hand(
    GameState& state,
    PlayerState& player,
    const SnapshotCardCounts& counts,
    const char* name) {
    if (count_cards(counts) > MAX_DECK_CARDS) {
        invalid(std::string(name) + " exceeds MAX_DECK_CARDS");
    }
    for (DefId def = 0; def < card_def_count(); ++def) {
        const std::uint16_t count = counts.by_def[def];
        if (count > 255U) {
            invalid(std::string(name) + " has a per-card count above 255");
        }
        if (count != 0U) {
            const Slot slot = add_slot(state, def);
            player.hand[slot] = static_cast<std::uint8_t>(count);
        }
    }
}

void fill_in_play(
    GameState& state,
    PlayerState& player,
    const SnapshotCardCounts& counts,
    const char* name) {
    if (count_cards(counts) > MAX_IN_PLAY) {
        invalid(std::string(name) + " exceeds MAX_IN_PLAY");
    }
    for (DefId def = 0; def < card_def_count(); ++def) {
        const Slot slot = counts.by_def[def] == 0U ? NONE : add_slot(state, def);
        for (std::uint16_t i = 0; i < counts.by_def[def]; ++i) {
            player.in_play[player.in_play_size] = InPlayEntry{slot, slot, 0U};
            ++player.in_play_size;
        }
    }
}

void validate_player_snapshot(const Snapshot& snapshot, PlayerId player_id) {
    const SnapshotPlayer& player = snapshot.players[player_id];
    const std::string prefix = "player " + std::to_string(player_id) + " ";
    check_known_defs(player.hand, (prefix + "hand").c_str());
    check_known_defs(player.hand_deck, (prefix + "hand+deck").c_str());
    check_known_defs(player.discard, (prefix + "discard").c_str());
    check_known_defs(player.in_play, (prefix + "in-play").c_str());
    check_known_defs(player.set_aside, (prefix + "set-aside").c_str());

    const std::uint32_t hidden_total = count_cards(player.hand_deck);
    if (hidden_total != static_cast<std::uint32_t>(player.hand_count) + player.deck_count) {
        invalid(prefix + "hand+deck composition has " + std::to_string(hidden_total)
            + " cards but counts require "
            + std::to_string(static_cast<std::uint32_t>(player.hand_count) + player.deck_count));
    }
    if (player.hand_count > MAX_DECK_CARDS || player.deck_count > MAX_DECK_CARDS) {
        invalid(prefix + "hand or deck count exceeds MAX_DECK_CARDS");
    }
    if (player_id == snapshot.our_player) {
        if (count_cards(player.hand) != player.hand_count) {
            invalid(prefix + "exact hand does not match hand_count");
        }
        for (DefId def = 0; def < card_def_count(); ++def) {
            if (player.hand.by_def[def] > player.hand_deck.by_def[def]) {
                invalid(prefix + "exact hand is not a subset of hand+deck");
            }
        }
    }
    if (player.actions > 255U || player.buys > 255U) {
        invalid(prefix + "resources exceed engine range");
    }
}

void build_player(GameState& state, const Snapshot& snapshot, PlayerId player_id) {
    const SnapshotPlayer& source = snapshot.players[player_id];
    PlayerState& target = state.players[player_id];
    const std::string prefix = "player " + std::to_string(player_id);

    SnapshotCardCounts deck{};
    if (player_id == snapshot.our_player) {
        fill_hand(state, target, source.hand, (prefix + " hand").c_str());
        for (DefId def = 0; def < card_def_count(); ++def) {
            deck.by_def[def] = static_cast<std::uint16_t>(
                source.hand_deck.by_def[def] - source.hand.by_def[def]);
        }
    } else {
        std::uint16_t remaining_hand = source.hand_count;
        for (DefId def = 0; def < card_def_count(); ++def) {
            const std::uint16_t available = source.hand_deck.by_def[def];
            const std::uint16_t dealt = std::min(available, remaining_hand);
            if (dealt != 0U) {
                const Slot slot = add_slot(state, def);
                target.hand[slot] = static_cast<std::uint8_t>(dealt);
                remaining_hand = static_cast<std::uint16_t>(remaining_hand - dealt);
            }
            deck.by_def[def] = static_cast<std::uint16_t>(available - dealt);
        }
        if (remaining_hand != 0U) {
            invalid(prefix + " could not deal the requested hidden hand");
        }
    }

    fill_ordered(state, target.deck, deck, (prefix + " deck").c_str());
    fill_ordered(state, target.discard, source.discard, (prefix + " discard").c_str());
    fill_in_play(state, target, source.in_play, (prefix + " in-play").c_str());
    fill_ordered(state, target.set_aside, source.set_aside, (prefix + " set-aside").c_str());
}

[[nodiscard]] std::uint8_t attack_program_counter(DefId source) {
    if (source >= card_def_count()) {
        invalid("interrupt source is unknown");
    }
    const EffectSpan span = card_def(source).on_play;
    for (std::uint16_t offset = span.offset;
         offset < static_cast<std::uint16_t>(span.offset + span.len);
         ++offset) {
        const Instr& instr = effect_instr(offset);
        if (instr.op == Op::Attack) {
            return instr.a;
        }
    }
    invalid(std::string(card_def(source).name) + " has no attack program");
}

[[nodiscard]] DefId infer_attack_source(const GameState& state, PlayerId attacker) {
    const PlayerState& player = state.players[attacker];
    for (std::uint8_t i = player.in_play_size; i > 0U; --i) {
        const DefId def = state.slot_to_def[player.in_play[i - 1U].slot];
        if ((card_def(def).types & TYPE_ATTACK) != 0U) {
            return def;
        }
    }
    invalid("moat reaction has no attacking card in the attacker's in-play zone");
}

[[nodiscard]] bool in_play_has_def(
    const GameState& state,
    PlayerId player_id,
    DefId def) noexcept {
    const PlayerState& player = state.players[player_id];
    for (std::uint8_t i = 0; i < player.in_play_size; ++i) {
        if (state.slot_to_def[player.in_play[i].slot] == def) {
            return true;
        }
    }
    return false;
}

void seed_interrupt(GameState& state, const Snapshot& snapshot) {
    if (snapshot.interrupt == SeededInterrupt::None) {
        refresh_current_decision(state);
        return;
    }
    if (snapshot.attacker >= state.num_players || snapshot.defender >= state.num_players) {
        invalid("interrupt attacker or defender seat is out of range");
    }
    if (snapshot.attacker == snapshot.defender) {
        invalid("interrupt attacker and defender must differ");
    }

    DefId source = 0;
    switch (snapshot.interrupt) {
    case SeededInterrupt::MoatReaction:
        source = infer_attack_source(state, snapshot.attacker);
        break;
    case SeededInterrupt::MilitiaDiscard:
        source = DEF_MILITIA;
        break;
    case SeededInterrupt::BureaucratTopdeck:
        source = DEF_BUREAUCRAT;
        break;
    case SeededInterrupt::BanditTrash:
        source = DEF_BANDIT;
        break;
    case SeededInterrupt::None:
        return;
    }
    if (!in_play_has_def(state, snapshot.attacker, source)) {
        invalid(std::string(card_def(source).name)
            + " interrupt source is not in the attacker's in-play zone");
    }

    EffectFrame frame{};
    frame.source = source;
    frame.pc = attack_program_counter(source);
    frame.player = snapshot.defender;
    frame.flags = FRAME_ABSOLUTE_PROGRAM | FRAME_ATTACK;
    frame.data[DATA_ACTIVE_PC] = ACTIVE_PC_NONE;

    DecisionKind kind = DecisionKind::Choose;
    std::uint8_t minimum = 1U;
    std::uint8_t maximum = 1U;
    if (snapshot.interrupt == SeededInterrupt::MoatReaction) {
        kind = DecisionKind::ReactWindow;
        minimum = 0U;
        const Slot moat = slot_of(state, DEF_MOAT);
        if (moat == NONE || state.players[snapshot.defender].hand[moat] == 0U) {
            invalid("moat reaction defender has no Moat in hand");
        }
    } else {
        frame.flags = static_cast<std::uint8_t>(frame.flags | FRAME_REACT_DONE);
    }

    if (snapshot.interrupt == SeededInterrupt::MilitiaDiscard) {
        std::uint16_t hand = 0;
        for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
            hand = static_cast<std::uint16_t>(
                hand + state.players[snapshot.defender].hand[slot]);
        }
        if (hand <= 3U) {
            invalid("militia discard requires a defender hand larger than 3");
        }
        minimum = 3U;
        maximum = 3U;
    } else if (snapshot.interrupt == SeededInterrupt::BureaucratTopdeck) {
        bool victory = false;
        for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
            if (state.players[snapshot.defender].hand[slot] != 0U
                && (card_def(state.slot_to_def[slot]).types & TYPE_VICTORY) != 0U) {
                victory = true;
            }
        }
        if (!victory) {
            invalid("bureaucrat topdeck requires a Victory card in the defender's hand");
        }
    } else if (snapshot.interrupt == SeededInterrupt::BanditTrash) {
        PlayerState& defender = state.players[snapshot.defender];
        if (defender.set_aside.size != 2U) {
            invalid("bandit trash requires exactly two revealed set-aside cards");
        }
        const Slot first = defender.set_aside.cards[0];
        const Slot second = defender.set_aside.cards[1];
        const DefId first_def = state.slot_to_def[first];
        const DefId second_def = state.slot_to_def[second];
        const auto trashable = [](DefId def) {
            return def != DEF_COPPER && (card_def(def).types & TYPE_TREASURE) != 0U;
        };
        if (!trashable(first_def) || !trashable(second_def) || first_def == second_def) {
            invalid("bandit trash requires two distinct non-Copper Treasures");
        }
        frame.data[1] = first;
        frame.data[2] = second;
        frame.data[3] = 2;
        frame.data[4] = 2;
        frame.data[DATA_ACTIVE_PC] = frame.pc;
        frame.data[DATA_ACTIVE_COUNT] = 0;
        defender.set_aside = OrderedZone{};
    }

    state.effect_stack[0] = frame;
    state.effect_depth = 1U;
    state.decision = PendingDecision{
        snapshot.defender,
        static_cast<std::uint8_t>(kind),
        source,
        minimum,
        maximum,
    };
}

void count_slot(std::uint32_t (&counts)[MAX_SLOTS], Slot slot, std::uint16_t amount = 1U) {
    counts[slot] += amount;
}

void count_ordered(
    const GameState& state,
    const OrderedZone& zone,
    const std::string& name,
    std::uint32_t (&counts)[MAX_SLOTS]) {
    if (zone.size > MAX_DECK_CARDS) {
        corrupt(name + " exceeds MAX_DECK_CARDS");
    }
    for (std::uint8_t i = 0; i < zone.size; ++i) {
        if (zone.cards[i] >= state.num_slots) {
            corrupt(name + " contains an invalid slot");
        }
        count_slot(counts, zone.cards[i]);
    }
}

void count_player(
    const GameState& state,
    PlayerId player_id,
    std::uint32_t (&counts)[MAX_SLOTS]) {
    const PlayerState& player = state.players[player_id];
    for (std::uint8_t slot = 0; slot < MAX_SLOTS; ++slot) {
        if (slot >= state.num_slots
            && (player.hand[slot] != 0U || player.exile[slot] != 0U
                || player.tavern[slot] != 0U || player.island_mat[slot] != 0U)) {
            corrupt("player " + std::to_string(player_id)
                + " has count data outside active slots");
        }
        counts[slot] += player.hand[slot];
        counts[slot] += player.exile[slot];
        counts[slot] += player.tavern[slot];
        counts[slot] += player.island_mat[slot];
    }
    const std::string prefix = "player " + std::to_string(player_id) + " ";
    count_ordered(state, player.deck, prefix + "deck", counts);
    count_ordered(state, player.discard, prefix + "discard", counts);
    count_ordered(state, player.set_aside, prefix + "set-aside", counts);
    if (player.in_play_size > MAX_IN_PLAY) {
        corrupt(prefix + "in-play exceeds MAX_IN_PLAY");
    }
    for (std::uint8_t i = 0; i < player.in_play_size; ++i) {
        const InPlayEntry& entry = player.in_play[i];
        if (entry.slot >= state.num_slots || entry.behaves_as >= state.num_slots) {
            corrupt(prefix + "in-play contains an invalid slot");
        }
        count_slot(counts, entry.slot);
    }
    if (player.pending_size > MAX_PENDING) {
        corrupt(prefix + "pending exceeds MAX_PENDING");
    }
}

void count_pile(
    const GameState& state,
    const Pile& pile,
    const std::string& name,
    std::uint32_t (&counts)[MAX_SLOTS]) {
    if (pile.base >= state.num_slots) {
        corrupt(name + " has an invalid base slot");
    }
    if (pile.mixed_len > 12U) {
        corrupt(name + " mixed length exceeds capacity");
    }
    if (pile.mixed_len == 0U) {
        count_slot(counts, pile.base, pile.count);
        return;
    }
    for (std::uint8_t i = 0; i < pile.mixed_len; ++i) {
        if (pile.mixed[i] >= state.num_slots) {
            corrupt(name + " has an invalid mixed slot");
        }
        count_slot(counts, pile.mixed[i]);
    }
}

void collect_card_counts(
    const GameState& state,
    std::uint32_t (&counts)[MAX_SLOTS]) {
    if (state.num_players < 2U || state.num_players > MAX_PLAYERS) {
        corrupt("num_players is out of range");
    }
    if (state.num_slots == 0U || state.num_slots > MAX_SLOTS) {
        corrupt("num_slots is out of range");
    }
    if (state.num_piles > MAX_PILES || state.num_nonsupply > MAX_NONSUPPLY) {
        corrupt("pile count exceeds capacity");
    }
    if (state.effect_depth > MAX_EFFECT_DEPTH) {
        corrupt("effect stack exceeds capacity");
    }

    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        const DefId def = state.slot_to_def[slot];
        if (def >= card_def_count()) {
            corrupt("slot maps to an unknown definition");
        }
        for (std::uint8_t previous = 0; previous < slot; ++previous) {
            if (state.slot_to_def[previous] == def) {
                corrupt("multiple slots map to the same definition");
            }
        }
    }
    for (PlayerId player = 0; player < state.num_players; ++player) {
        count_player(state, player, counts);
    }
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        count_pile(state, state.piles[i], "supply pile " + std::to_string(i), counts);
    }
    for (std::uint8_t i = 0; i < state.num_nonsupply; ++i) {
        count_pile(
            state,
            state.nonsupply[i],
            "nonsupply pile " + std::to_string(i),
            counts);
    }
    for (std::uint8_t slot = 0; slot < MAX_SLOTS; ++slot) {
        if (slot >= state.num_slots && state.trash[slot] != 0U) {
            corrupt("trash has count data outside active slots");
        }
        counts[slot] += state.trash[slot];
    }

    for (std::uint8_t depth = 0; depth < state.effect_depth; ++depth) {
        const EffectFrame& frame = state.effect_stack[depth];
        if (frame.source >= card_def_count() || frame.player >= state.num_players) {
            corrupt("effect frame has an invalid source or player");
        }
        if (frame.source != DEF_BANDIT) {
            continue;
        }
        if (frame.data[3] < 0 || frame.data[3] > 2) {
            corrupt("Bandit frame has an invalid revealed count");
        }
        for (std::int16_t i = 0; i < frame.data[3]; ++i) {
            const std::int16_t slot = frame.data[1 + i];
            if (slot < 0 || slot >= state.num_slots) {
                corrupt("Bandit frame has an invalid revealed slot");
            }
            count_slot(counts, static_cast<Slot>(slot));
        }
    }
}

[[nodiscard]] bool has_supply_pile(
    const GameState& state,
    DefId def) noexcept {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        const Pile& pile = state.piles[i];
        if (state.slot_to_def[pile.base] == def) {
            return true;
        }
        for (std::uint8_t mixed = 0; mixed < pile.mixed_len; ++mixed) {
            if (state.slot_to_def[pile.mixed[mixed]] == def) {
                return true;
            }
        }
    }
    return false;
}

[[nodiscard]] std::uint16_t canonical_card_total(
    const GameState& state,
    DefId def) noexcept {
    if (!has_supply_pile(state, def)) {
        return 0U;
    }
    const std::uint16_t victory_supply =
        state.num_players == 2U ? 8U : 12U;
    switch (def) {
    case DEF_COPPER:
        return 60U;
    case DEF_SILVER:
        return 40U;
    case DEF_GOLD:
        return 30U;
    case DEF_PLATINUM:
        return 12U;
    case DEF_POTION:
        return 16U;
    case DEF_ESTATE:
        return static_cast<std::uint16_t>(
            victory_supply + (3U * state.num_players));
    case DEF_DUCHY:
    case DEF_PROVINCE:
    case DEF_COLONY:
        return victory_supply;
    case DEF_CURSE:
        return static_cast<std::uint16_t>((state.num_players - 1U) * 10U);
    default:
        return (card_def(def).types & TYPE_VICTORY) != 0U
            ? victory_supply
            : 10U;
    }
}

} // namespace

GameState build_game_from_snapshot(const Snapshot& snapshot) {
    if (snapshot.num_players < 2U || snapshot.num_players > MAX_PLAYERS) {
        invalid("num_players must be between 2 and MAX_PLAYERS");
    }
    if (snapshot.our_player >= snapshot.num_players
        || snapshot.current_player >= snapshot.num_players) {
        invalid("our_player or current_player seat is out of range");
    }
    check_known_defs(snapshot.supply, "supply");
    check_known_defs(snapshot.trash, "trash");
    check_known_defs(snapshot.card_totals, "card totals");
    for (std::uint8_t def = static_cast<std::uint8_t>(card_def_count());
         def < MAX_SLOTS;
         ++def) {
        if (snapshot.supply_present[def] != 0U) {
            invalid("supply contains an unknown definition");
        }
    }
    constexpr DefId required_supply[] = {
        DEF_COPPER,
        DEF_SILVER,
        DEF_GOLD,
        DEF_ESTATE,
        DEF_DUCHY,
        DEF_PROVINCE,
        DEF_CURSE,
    };
    for (const DefId def : required_supply) {
        if (snapshot.supply_present[def] == 0U) {
            invalid(std::string("required supply pile is missing: ")
                + card_def(def).name);
        }
    }
    for (PlayerId player = 0; player < snapshot.num_players; ++player) {
        validate_player_snapshot(snapshot, player);
    }

    GameState state{};
    std::memset(&state, 0, sizeof(state));
    state.num_players = snapshot.num_players;
    state.rng = Xoshiro256pp::seeded(0xA2E1'5A7E'B01D'0001ULL);
    state.trigger_table.dirty = 0U;
    for (int i = 0; i < NUM_ARTIFACTS; ++i) {
        state.artifact_holder[i] = NONE;
    }

    for (DefId def = 0; def < card_def_count(); ++def) {
        if (snapshot.supply_present[def] == 0U) {
            if (snapshot.supply.by_def[def] != 0U) {
                invalid("supply count is nonzero for absent def " + std::to_string(def));
            }
            continue;
        }
        if (snapshot.supply.by_def[def] > 255U) {
            invalid("supply count exceeds engine range");
        }
        if (state.num_piles >= MAX_PILES) {
            invalid("too many supply piles");
        }
        Pile& pile = state.piles[state.num_piles];
        pile.base = add_slot(state, def);
        pile.count = static_cast<std::uint8_t>(snapshot.supply.by_def[def]);
        ++state.num_piles;
    }

    for (PlayerId player = 0; player < snapshot.num_players; ++player) {
        build_player(state, snapshot, player);
    }
    for (DefId def = 0; def < card_def_count(); ++def) {
        const std::uint16_t count = snapshot.trash.by_def[def];
        if (count > 255U) {
            invalid("trash per-card count exceeds engine range");
        }
        if (count != 0U) {
            const Slot slot = add_slot(state, def);
            state.trash[slot] = static_cast<std::uint8_t>(count);
        }
    }
    for (DefId def = 0; def < card_def_count(); ++def) {
        if (snapshot.card_totals.by_def[def] != 0U) {
            (void)add_slot(state, def);
        }
    }

    switch (snapshot.phase) {
    case SnapshotPhase::Action:
        state.phase = static_cast<std::uint8_t>(Phase::Action);
        break;
    case SnapshotPhase::Buy:
        state.phase = static_cast<std::uint8_t>(Phase::Buy);
        break;
    case SnapshotPhase::Cleanup:
        state.phase = static_cast<std::uint8_t>(Phase::Cleanup);
        break;
    default:
        invalid("phase is unknown");
    }
    const SnapshotPlayer& resources = snapshot.players[snapshot.current_player];
    state.actions = static_cast<std::uint8_t>(resources.actions);
    state.buys = static_cast<std::uint8_t>(resources.buys);
    state.coins = resources.coins;
    state.turn_counter = snapshot.turn_number;
    turn_queue_clear(state.turn_queue);
    if (!turn_queue_push(state.turn_queue, snapshot.current_player, TurnKind::Normal)) {
        invalid("could not initialize turn queue");
    }

    for (PlayerId player = 0; player < state.num_players; ++player) {
        for (std::uint8_t i = 0; i < state.players[player].in_play_size; ++i) {
            const Slot slot = state.players[player].in_play[i].slot;
            if (card_def(state.slot_to_def[slot]).trigger_mask != 0U) {
                state.trigger_table.dirty = 1U;
            }
        }
    }
    seed_interrupt(state, snapshot);

    for (DefId def = 0; def < card_def_count(); ++def) {
        const std::uint16_t expected = canonical_card_total(state, def);
        if (snapshot.card_totals.by_def[def] != expected) {
            invalid(std::string(card_def(def).name)
                + " total does not match the base-set game total: expected "
                + std::to_string(expected) + ", got "
                + std::to_string(snapshot.card_totals.by_def[def]));
        }
    }
    validate_game(state);
    return state;
}

void set_deck_order(
    GameState& state,
    PlayerId player,
    std::span<const DefId> defs) {
    if (player >= state.num_players) {
        throw std::invalid_argument("deck order player is out of range");
    }
    OrderedZone& deck = state.players[player].deck;
    if (defs.size() != deck.size) {
        throw std::invalid_argument(
            "deck order length does not match current deck count");
    }

    std::uint16_t current[MAX_SLOTS]{};
    std::uint16_t requested[MAX_SLOTS]{};
    for (std::uint8_t i = 0; i < deck.size; ++i) {
        if (deck.cards[i] >= state.num_slots) {
            throw std::logic_error("current deck contains an invalid slot");
        }
        ++current[deck.cards[i]];
    }
    for (const DefId def : defs) {
        if (def >= card_def_count()) {
            throw std::invalid_argument("deck order contains an unknown def");
        }
        const Slot slot = slot_of(state, def);
        if (slot == NONE) {
            throw std::invalid_argument("deck order contains a def absent from the game");
        }
        ++requested[slot];
    }
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        if (current[slot] != requested[slot]) {
            throw std::invalid_argument(
                "deck order must be a permutation of the current deck");
        }
    }

    // Public API order is draw order: the first definition is the next card
    // drawn. OrderedZone stores its top at size - 1, so install in reverse.
    for (std::size_t i = 0; i < defs.size(); ++i) {
        deck.cards[defs.size() - 1U - i] = slot_of(state, defs[i]);
    }
    validate_game(state);
}

void validate_game(const GameState& state) {
    if (state.phase > static_cast<std::uint8_t>(Phase::Over)) {
        corrupt("phase is invalid");
    }
    if (state.turn_queue.size == 0U || state.turn_queue.size > MAX_TURN_QUEUE) {
        corrupt("turn queue size is invalid");
    }
    for (std::uint8_t i = 0; i < state.turn_queue.size; ++i) {
        const std::uint8_t index =
            static_cast<std::uint8_t>((state.turn_queue.head + i) % MAX_TURN_QUEUE);
        if (state.turn_queue.entries[index].player >= state.num_players) {
            corrupt("turn queue contains an invalid player");
        }
    }
    if (state.decision.kind > static_cast<std::uint8_t>(DecisionKind::OrderTriggers)) {
        corrupt("decision kind is invalid");
    }
    if (state.decision.kind != static_cast<std::uint8_t>(DecisionKind::None)
        && state.decision.player >= state.num_players) {
        corrupt("decision player is invalid");
    }

    std::uint32_t counts[MAX_SLOTS]{};
    collect_card_counts(state, counts);
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        const DefId def = state.slot_to_def[slot];
        const std::uint16_t expected = canonical_card_total(state, def);
        if (counts[slot] != expected) {
            corrupt(std::string(card_def(def).name) + " conservation mismatch: expected "
                + std::to_string(expected) + ", found "
                + std::to_string(counts[slot]));
        }
    }
}
