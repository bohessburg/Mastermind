#include "v2/core/interp.h"

#include "v2/core/moves.h"
#include "v2/core/setup.h"
#include "v2/core/triggers.h"

#include <cassert>
#include <cstdint>

namespace {

constexpr std::int16_t ACTIVE_PC_NONE = -1;

// EffectFrame::data[] layout:
//   [0] shared result: previous op selected count
//   [1] shared result: previous op selected def / option
//   [2] shared result: previous op selected cost coins
//   [3] shared result: previous op selected cost potion
//   [4] shared result: previous op selected cost debt
//   [5] choice/order owner: active op pc, or ACTIVE_PC_NONE
//   [6] choice/order owner: selections made in the active op
//   [7] control owner: Repeat completed-iteration count
//   BanditAttack active-op overlay:
//     [1], [2] revealed set-aside slots, [3] revealed count, [4] trashable count
constexpr int DATA_LAST_COUNT = 0;
constexpr int DATA_LAST_DEF = 1;
constexpr int DATA_LAST_COST_COINS = 2;
constexpr int DATA_LAST_COST_POTION = 3;
constexpr int DATA_LAST_COST_DEBT = 4;
constexpr int DATA_ACTIVE_PC = 5;
constexpr int DATA_ACTIVE_COUNT = 6;
constexpr int DATA_REPEAT_COUNT = 7;
constexpr int DATA_BANDIT_REVEALED_0 = DATA_LAST_DEF;
constexpr int DATA_BANDIT_REVEALED_1 = DATA_LAST_COST_COINS;
constexpr int DATA_BANDIT_REVEALED_COUNT = DATA_LAST_COST_POTION;
constexpr int DATA_BANDIT_TRASHABLE_COUNT = DATA_LAST_COST_DEBT;
constexpr std::int16_t DATA_NO_SLOT = -1;
constexpr std::uint8_t CHOOSE_MAX_ALL = 0xFFU;

void shuffle_zone(OrderedZone& zone, Xoshiro256pp& rng) noexcept {
    for (int i = static_cast<int>(zone.size); i > 1; --i) {
        const int last = i - 1;
        const std::uint32_t swap_index = rng.uniform(static_cast<std::uint32_t>(i));
        const Slot tmp = zone.cards[last];
        zone.cards[last] = zone.cards[swap_index];
        zone.cards[swap_index] = tmp;
    }
}

void reshuffle_discard_into_deck(GameState& state, PlayerId player_id) noexcept {
    PlayerState& player = state.players[player_id];
    if (player.discard.size == 0U) {
        return;
    }

    for (std::uint8_t i = 0; i < player.discard.size; ++i) {
        player.deck.cards[i] = player.discard.cards[i];
    }
    player.deck.size = player.discard.size;
    player.discard.size = 0;
    shuffle_zone(player.deck, state.rng);
    if (!trigger_table_clean_empty(state)) {
        emit(state, TriggerKind::OnShuffle, TriggerPayload{player_id, 0, NONE, 0U});
    }
}

[[nodiscard]] bool prepare_deck_top(GameState& state, PlayerId player_id) noexcept {
    PlayerState& player = state.players[player_id];
    if (player.deck.size == 0U) {
        reshuffle_discard_into_deck(state, player_id);
    }
    return player.deck.size > 0U;
}

[[nodiscard]] Slot deck_top_slot(GameState& state, PlayerId player_id) noexcept {
    if (!prepare_deck_top(state, player_id)) {
        return NONE;
    }
    const PlayerState& player = state.players[player_id];
    return player.deck.cards[player.deck.size - 1U];
}

[[nodiscard]] Slot take_deck_top_slot_impl(GameState& state, PlayerId player_id) noexcept {
    if (!prepare_deck_top(state, player_id)) {
        return NONE;
    }

    PlayerState& player = state.players[player_id];
    --player.deck.size;
    const Slot slot = player.deck.cards[player.deck.size];
    player.deck.cards[player.deck.size] = 0;
    return slot;
}

void pop_frame(GameState& state) noexcept {
    assert(state.effect_depth > 0U);
    state.effect_stack[state.effect_depth - 1U] = EffectFrame{};
    --state.effect_depth;
}

void complete_frame(GameState& state, EffectFrame& frame) noexcept {
    if (frame.repeats_left > 1U && (frame.flags & FRAME_ABSOLUTE_PROGRAM) == 0U) {
        --frame.repeats_left;
        frame.pc = 0;
        frame.data[DATA_ACTIVE_PC] = ACTIVE_PC_NONE;
        frame.data[DATA_ACTIVE_COUNT] = 0;
        state.decision = PendingDecision{};
        return;
    }
    pop_frame(state);
}

void add_uint8(std::uint8_t& value, std::int16_t delta) noexcept {
    int next = static_cast<int>(value) + static_cast<int>(delta);
    if (next < 0) {
        next = 0;
    }
    if (next > 255) {
        next = 255;
    }
    value = static_cast<std::uint8_t>(next);
}

[[nodiscard]] std::int8_t budget_component(std::int16_t value) noexcept {
    if (value < 0) {
        return 0;
    }
    if (value > 127) {
        return 127;
    }
    return static_cast<std::int8_t>(value);
}

[[nodiscard]] bool def_matches(DefId def, const Filter& filter) noexcept {
    if (filter.exact_def != ANY_DEF && def != filter.exact_def) {
        return false;
    }
    if (filter.exclude_def != ANY_DEF && def == filter.exclude_def) {
        return false;
    }
    return true;
}

[[nodiscard]] bool type_matches(DefId def, std::uint16_t type_mask) noexcept {
    return type_mask == 0U || (card_def(def).types & type_mask) != 0U;
}

[[nodiscard]] Cost limit_from_filter(const GameState& state, const EffectFrame& frame, const Filter& filter) noexcept {
    if (filter.cost_kind == CostLimitKind::Fixed) {
        return filter.max_cost;
    }
    if (filter.cost_kind == CostLimitKind::LastChosenPlus) {
        return Cost{
            budget_component(static_cast<std::int16_t>(frame.data[DATA_LAST_COST_COINS] + filter.coin_delta)),
            budget_component(frame.data[DATA_LAST_COST_POTION]),
            frame.data[DATA_LAST_COST_DEBT],
        };
    }
    (void)state;
    return Cost{127, 127, 32767};
}

[[nodiscard]] bool cost_matches(const GameState& state, const EffectFrame& frame, const Filter& filter, DefId def) noexcept {
    if (filter.cost_kind == CostLimitKind::None) {
        return true;
    }
    return card_def(def).cost.fits_within(limit_from_filter(state, frame, filter));
}

[[nodiscard]] bool hand_matches_filter(
    const GameState& state,
    const EffectFrame& frame,
    const Filter& filter,
    Slot slot) noexcept {
    const DefId def = state.slot_to_def[slot];
    return filter.zone == ZoneSelector::Hand
        && def_matches(def, filter)
        && type_matches(def, filter.type_mask)
        && cost_matches(state, frame, filter, def);
}

[[nodiscard]] bool discard_matches_filter(
    const GameState& state,
    const EffectFrame& frame,
    const Filter& filter,
    Slot slot) noexcept {
    const DefId def = state.slot_to_def[slot];
    return filter.zone == ZoneSelector::Discard
        && def_matches(def, filter)
        && type_matches(def, filter.type_mask)
        && cost_matches(state, frame, filter, def);
}

[[nodiscard]] bool pile_has_cards(const Pile& pile) noexcept {
    return pile.mixed_len > 0U || pile.count > 0U;
}

[[nodiscard]] Slot pile_top_slot(const Pile& pile) noexcept {
    if (pile.mixed_len > 0U) {
        return pile.mixed[pile.mixed_len - 1U];
    }
    return pile.base;
}

[[nodiscard]] DefId pile_top_def(const GameState& state, const Pile& pile) noexcept {
    return state.slot_to_def[pile_top_slot(pile)];
}

[[nodiscard]] bool supply_matches_filter(
    const GameState& state,
    const EffectFrame& frame,
    const Filter& filter,
    const Pile& pile) noexcept {
    if (filter.zone != ZoneSelector::Supply || !pile_has_cards(pile)) {
        return false;
    }
    const DefId def = pile_top_def(state, pile);
    return def_matches(def, filter) && type_matches(def, filter.type_mask) && cost_matches(state, frame, filter, def);
}

[[nodiscard]] Pile* find_matching_supply_pile(
    GameState& state,
    const EffectFrame& frame,
    const Filter& filter,
    DefId def) noexcept {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        Pile& pile = state.piles[i];
        if (supply_matches_filter(state, frame, filter, pile) && pile_top_def(state, pile) == def) {
            return &pile;
        }
    }
    return nullptr;
}

[[nodiscard]] std::uint8_t count_matching_hand(
    const GameState& state,
    const EffectFrame& frame,
    const Filter& filter) noexcept {
    const PlayerState& player = state.players[frame.player];
    std::uint8_t total = 0;
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        if (player.hand[slot] > 0U && hand_matches_filter(state, frame, filter, slot)) {
            total = static_cast<std::uint8_t>(total + player.hand[slot]);
        }
    }
    return total;
}

[[nodiscard]] std::uint8_t count_matching_supply(
    const GameState& state,
    const EffectFrame& frame,
    const Filter& filter) noexcept {
    std::uint8_t total = 0;
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        if (supply_matches_filter(state, frame, filter, state.piles[i])) {
            ++total;
        }
    }
    return total;
}

[[nodiscard]] std::uint8_t count_matching_discard(
    const GameState& state,
    const EffectFrame& frame,
    const Filter& filter) noexcept {
    const PlayerState& player = state.players[frame.player];
    std::uint8_t total = 0;
    for (std::uint8_t i = 0; i < player.discard.size; ++i) {
        if (discard_matches_filter(state, frame, filter, player.discard.cards[i])) {
            ++total;
        }
    }
    return total;
}

[[nodiscard]] bool discard_contains_slot(const PlayerState& player, Slot slot) noexcept {
    for (std::uint8_t i = 0; i < player.discard.size; ++i) {
        if (player.discard.cards[i] == slot) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool begin_active_op(EffectFrame& frame) noexcept {
    if (frame.data[DATA_ACTIVE_PC] != frame.pc) {
        frame.data[DATA_ACTIVE_PC] = frame.pc;
        frame.data[DATA_ACTIVE_COUNT] = 0;
        return true;
    }
    return false;
}

[[nodiscard]] std::uint8_t active_count(const EffectFrame& frame) noexcept {
    return static_cast<std::uint8_t>(frame.data[DATA_ACTIVE_COUNT]);
}

void finish_active_op(GameState& state, EffectFrame& frame) noexcept {
    frame.data[DATA_LAST_COUNT] = frame.data[DATA_ACTIVE_COUNT];
    frame.data[DATA_ACTIVE_PC] = ACTIVE_PC_NONE;
    frame.data[DATA_ACTIVE_COUNT] = 0;
    ++frame.pc;
    state.decision = PendingDecision{};
}

[[nodiscard]] std::uint8_t available_for_choice(
    const GameState& state,
    const EffectFrame& frame,
    const Filter& filter) noexcept {
    if (filter.zone == ZoneSelector::Supply) {
        return count_matching_supply(state, frame, filter);
    }
    if (filter.zone == ZoneSelector::Discard) {
        return count_matching_discard(state, frame, filter);
    }
    return count_matching_hand(state, frame, filter);
}

[[nodiscard]] std::uint8_t requested_remaining(const EffectFrame& frame, const Instr& instr) noexcept {
    if (instr.c == CHOOSE_MAX_ALL) {
        return 0xFFU;
    }
    const std::uint8_t chosen = active_count(frame);
    if (chosen >= instr.c) {
        return 0;
    }
    return static_cast<std::uint8_t>(instr.c - chosen);
}

[[nodiscard]] std::uint8_t min_remaining(const EffectFrame& frame, const Instr& instr) noexcept {
    const std::uint8_t chosen = active_count(frame);
    if (chosen >= instr.b) {
        return 0;
    }
    return static_cast<std::uint8_t>(instr.b - chosen);
}

[[nodiscard]] std::uint8_t min_u8(std::uint8_t lhs, std::uint8_t rhs) noexcept {
    return lhs < rhs ? lhs : rhs;
}

[[nodiscard]] bool suspend_choice(GameState& state, EffectFrame& frame, const Instr& instr, DecisionKind kind) noexcept {
    (void)begin_active_op(frame);
    const Filter& filter = filter_def(instr.a);
    const std::uint8_t available = available_for_choice(state, frame, filter);
    const std::uint8_t max_left = min_u8(requested_remaining(frame, instr), available);
    if (max_left == 0U) {
        finish_active_op(state, frame);
        return false;
    }

    const std::uint8_t min_left = min_u8(min_remaining(frame, instr), max_left);
    state.decision = PendingDecision{
        frame.player,
        static_cast<std::uint8_t>(kind),
        frame.source,
        min_left,
        max_left,
    };
    return true;
}

void suspend_option(GameState& state, EffectFrame& frame, const Instr& instr, DecisionKind kind) noexcept {
    (void)begin_active_op(frame);
    state.decision = PendingDecision{
        frame.player,
        static_cast<std::uint8_t>(kind),
        frame.source,
        1,
        instr.a,
    };
}

void suspend_order(GameState& state, EffectFrame& frame, const Instr& instr) noexcept {
    (void)begin_active_op(frame);
    const std::uint8_t chosen = active_count(frame);
    const std::uint8_t total = instr.a;
    const std::uint8_t remaining = chosen >= total ? 0U : static_cast<std::uint8_t>(total - chosen);
    state.decision = PendingDecision{
        frame.player,
        static_cast<std::uint8_t>(DecisionKind::ChooseOrder),
        frame.source,
        static_cast<std::uint8_t>(remaining == 0U ? 0U : 1U),
        remaining,
    };
}

void append_topdeck(PlayerState& player, Slot slot) noexcept {
    assert(player.deck.size < MAX_DECK_CARDS);
    player.deck.cards[player.deck.size] = slot;
    ++player.deck.size;
}

[[nodiscard]] bool topdeck_from_discard(GameState& state, PlayerId player_id, Slot slot) noexcept {
    PlayerState& player = state.players[player_id];
    for (std::uint8_t i = player.discard.size; i > 0U; --i) {
        const std::uint8_t index = static_cast<std::uint8_t>(i - 1U);
        if (player.discard.cards[index] != slot) {
            continue;
        }
        for (std::uint8_t j = index; static_cast<std::uint8_t>(j + 1U) < player.discard.size; ++j) {
            player.discard.cards[j] = player.discard.cards[j + 1U];
        }
        --player.discard.size;
        player.discard.cards[player.discard.size] = 0;
        append_topdeck(player, slot);
        return true;
    }
    return false;
}

void add_to_in_play(GameState& state, PlayerId player_id, Slot slot) noexcept {
    PlayerState& player = state.players[player_id];
    assert(player.in_play_size < MAX_IN_PLAY);
    player.in_play[player.in_play_size] = InPlayEntry{slot, slot, 0U};
    ++player.in_play_size;
    mark_trigger_table_dirty(state);
}

void play_slot_from_hand(GameState& state, PlayerId player_id, Slot slot) noexcept {
    PlayerState& player = state.players[player_id];
    assert(player.hand[slot] > 0U);
    --player.hand[slot];
    add_to_in_play(state, player_id, slot);
}

[[nodiscard]] bool play_slot_from_discard(GameState& state, PlayerId player_id, Slot slot) noexcept {
    PlayerState& player = state.players[player_id];
    for (std::uint8_t i = player.discard.size; i > 0U; --i) {
        const std::uint8_t index = static_cast<std::uint8_t>(i - 1U);
        if (player.discard.cards[index] != slot) {
            continue;
        }
        for (std::uint8_t j = index; static_cast<std::uint8_t>(j + 1U) < player.discard.size; ++j) {
            player.discard.cards[j] = player.discard.cards[j + 1U];
        }
        --player.discard.size;
        player.discard.cards[player.discard.size] = 0;
        add_to_in_play(state, player_id, slot);
        return true;
    }
    return false;
}

void record_last_choice(GameState& state, EffectFrame& frame, DefId def) noexcept {
    const Cost cost = card_def(def).cost;
    frame.data[DATA_LAST_DEF] = static_cast<std::int16_t>(def);
    frame.data[DATA_LAST_COST_COINS] = cost.coins;
    frame.data[DATA_LAST_COST_POTION] = cost.potion;
    frame.data[DATA_LAST_COST_DEBT] = cost.debt;
    (void)state;
}

void apply_then(GameState& state, EffectFrame& frame, const Instr& instr, Then then_kind, DefId def) noexcept {
    const Slot slot = slot_of(state, def);
    assert(slot != NONE);

    switch (then_kind) {
    case Then::Discard:
        assert(state.players[frame.player].hand[slot] > 0U);
        (void)do_discard(state, frame.player, slot, MoveZone::Hand);
        break;
    case Then::Trash:
        assert(state.players[frame.player].hand[slot] > 0U);
        (void)do_trash(state, frame.player, slot, MoveZone::Hand);
        break;
    case Then::Topdeck:
        if (filter_def(instr.a).zone == ZoneSelector::Discard) {
            (void)topdeck_from_discard(state, frame.player, slot);
        } else {
            PlayerState& player = state.players[frame.player];
            assert(player.hand[slot] > 0U);
            --player.hand[slot];
            append_topdeck(player, slot);
        }
        break;
    case Then::Play:
        assert(state.players[frame.player].hand[slot] > 0U);
        play_slot_from_hand(state, frame.player, slot);
        if (!trigger_table_clean_empty(state)) {
            emit(state, TriggerKind::OnPlayAction, TriggerPayload{frame.player, def, slot, 0U});
        }
        break;
    case Then::PutInHand:
    case Then::Keep:
        break;
    case Then::Reveal:
    case Then::SetAside:
    case Then::Exile:
        assert(false);
        break;
    }

    record_last_choice(state, frame, def);
    ++frame.data[DATA_ACTIVE_COUNT];
}

void apply_choose_gain(GameState& state, EffectFrame& frame, const Instr& instr, DefId def) noexcept {
    const Filter& filter = filter_def(instr.a);
    Pile* pile = find_matching_supply_pile(state, frame, filter, def);
    assert(pile != nullptr);
    if (pile == nullptr) {
        return;
    }

    const Slot slot = pile_top_slot(*pile);
    const bool gained = do_gain(state, frame.player, slot, static_cast<GainDestination>(instr.arg));
    (void)gained;
    assert(gained);
    record_last_choice(state, frame, def);
    ++frame.data[DATA_ACTIVE_COUNT];
}

void execute_multiplied_instr(GameState& state, EffectFrame& frame, const Instr& instr, std::int16_t multiplier) noexcept {
    if (multiplier <= 0) {
        return;
    }

    const std::int16_t total = static_cast<std::int16_t>(instr.arg * multiplier);
    switch (instr.op) {
    case Op::PlusCards:
        if (total > 0) {
            draw_cards(state, frame.player, static_cast<std::uint8_t>(total));
        }
        break;
    case Op::PlusActions:
        add_uint8(state.actions, total);
        break;
    case Op::PlusBuys:
        add_uint8(state.buys, total);
        break;
    case Op::PlusCoins:
        state.coins = static_cast<std::int16_t>(state.coins + total);
        break;
    default:
        assert(false);
        break;
    }
}

[[nodiscard]] bool predicate_matches(const GameState& state, const EffectFrame& frame, const Instr& instr) noexcept {
    const PredicateId predicate = static_cast<PredicateId>(instr.a);
    switch (predicate) {
    case PredicateId::AlwaysFalse:
        return false;
    case PredicateId::AlwaysTrue:
        return true;
    case PredicateId::ChosenAny:
        return frame.data[DATA_LAST_COUNT] > 0;
    case PredicateId::LastOptionEqualsArg:
        return frame.data[DATA_LAST_DEF] == instr.arg;
    case PredicateId::CoinsAtLeastArg:
        return state.coins >= instr.arg;
    case PredicateId::LastChosenIsAction:
        return frame.data[DATA_LAST_DEF] >= 0
            && (card_def(static_cast<DefId>(frame.data[DATA_LAST_DEF])).types & TYPE_ACTION) != 0U;
    }
    return false;
}

[[nodiscard]] std::uint8_t hand_size(const PlayerState& player, std::uint8_t num_slots) noexcept {
    std::uint8_t total = 0;
    for (std::uint8_t slot = 0; slot < num_slots; ++slot) {
        total = static_cast<std::uint8_t>(total + player.hand[slot]);
    }
    return total;
}

[[nodiscard]] bool has_reaction_in_hand(const GameState& state, PlayerId player_id) noexcept {
    const PlayerState& player = state.players[player_id];
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        if (player.hand[slot] == 0U) {
            continue;
        }
        if ((card_def(state.slot_to_def[slot]).types & TYPE_REACTION) != 0U) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool attack_prologue(GameState& state, EffectFrame& frame) noexcept {
    if ((frame.flags & FRAME_REACT_DONE) == 0U) {
        if (has_reaction_in_hand(state, frame.player)) {
            state.decision = PendingDecision{
                frame.player,
                static_cast<std::uint8_t>(DecisionKind::ReactWindow),
                frame.source,
                0,
                1,
            };
            return true;
        }
        frame.flags = static_cast<std::uint8_t>(frame.flags | FRAME_REACT_DONE);
    }

    if ((frame.flags & FRAME_ATTACK_IMMUNE) != 0U) {
        pop_frame(state);
        return true;
    }
    return false;
}

[[nodiscard]] bool suspend_discard_down_to(GameState& state, EffectFrame& frame, const Instr& instr) noexcept {
    (void)begin_active_op(frame);
    const std::uint8_t target = static_cast<std::uint8_t>(instr.arg);
    const std::uint8_t current = hand_size(state.players[frame.player], state.num_slots);
    if (current <= target) {
        finish_active_op(state, frame);
        return false;
    }

    if (active_count(frame) >= target) {
        PlayerState& player = state.players[frame.player];
        for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
            const DefId def = state.slot_to_def[slot];
            std::uint8_t keep = 0;
            for (std::uint8_t i = 0; i < target; ++i) {
                if (frame.data[DATA_LAST_DEF + i] == static_cast<std::int16_t>(def)) {
                    ++keep;
                }
            }
            while (player.hand[slot] > keep) {
                const bool discarded = do_discard(state, frame.player, slot, MoveZone::Hand);
                (void)discarded;
                assert(discarded);
            }
        }
        finish_active_op(state, frame);
        return false;
    }

    const std::uint8_t keep_count = static_cast<std::uint8_t>(target - active_count(frame));
    state.decision = PendingDecision{
        frame.player,
        static_cast<std::uint8_t>(DecisionKind::Choose),
        frame.source,
        keep_count,
        keep_count,
    };
    return true;
}

void push_attack_frames(GameState& state, EffectFrame& frame, const Instr& instr) noexcept {
    const PlayerId attacker = frame.player;
    for (std::uint8_t offset = static_cast<std::uint8_t>(state.num_players - 1U); offset > 0U; --offset) {
        const PlayerId target = static_cast<PlayerId>((attacker + offset) % state.num_players);
        assert(state.effect_depth < MAX_EFFECT_DEPTH);
        if (state.effect_depth >= MAX_EFFECT_DEPTH) {
            return;
        }
        EffectFrame attack{};
        attack.source = frame.source;
        attack.pc = instr.a;
        attack.player = target;
        attack.flags = static_cast<std::uint8_t>(FRAME_ABSOLUTE_PROGRAM | FRAME_ATTACK);
        attack.data[DATA_ACTIVE_PC] = ACTIVE_PC_NONE;
        state.effect_stack[state.effect_depth] = attack;
        ++state.effect_depth;
    }
}

void gain_specific(GameState& state, EffectFrame& frame, const Instr& instr) noexcept {
    const DefId def = static_cast<DefId>(instr.arg);
    const Slot slot = slot_of(state, def);
    if (slot != NONE) {
        (void)do_gain(state, frame.player, slot, static_cast<GainDestination>(instr.a));
    }
    ++frame.pc;
}

void gain_curse(GameState& state, EffectFrame& frame, const Instr& instr) noexcept {
    const Slot curse = slot_of(state, DEF_CURSE);
    if (curse != NONE) {
        (void)do_gain(state, frame.player, curse, static_cast<GainDestination>(instr.a));
    }
    ++frame.pc;
}

[[nodiscard]] std::uint8_t empty_supply_piles(const GameState& state) noexcept {
    std::uint8_t empty = 0;
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        if (state.piles[i].mixed_len == 0U && state.piles[i].count == 0U) {
            ++empty;
        }
    }
    return empty;
}

[[nodiscard]] bool suspend_discard_per_empty_supply(
    GameState& state,
    EffectFrame& frame,
    const Instr& instr) noexcept {
    (void)begin_active_op(frame);
    const std::uint8_t needed = empty_supply_piles(state);
    const std::uint8_t chosen = active_count(frame);
    if (chosen >= needed) {
        finish_active_op(state, frame);
        return false;
    }

    const std::uint8_t available = count_matching_hand(state, frame, filter_def(instr.a));
    const std::uint8_t remaining = static_cast<std::uint8_t>(needed - chosen);
    const std::uint8_t picks = min_u8(remaining, available);
    if (picks == 0U) {
        finish_active_op(state, frame);
        return false;
    }

    state.decision = PendingDecision{
        frame.player,
        static_cast<std::uint8_t>(DecisionKind::Choose),
        frame.source,
        picks,
        picks,
    };
    return true;
}

void discard_deck_top(GameState& state, EffectFrame& frame) noexcept {
    const Slot slot = deck_top_slot(state, frame.player);
    if (slot == NONE) {
        frame.data[DATA_LAST_COUNT] = 0;
        frame.data[DATA_LAST_DEF] = -1;
        frame.data[DATA_LAST_COST_DEBT] = -1;
        ++frame.pc;
        return;
    }

    const DefId def = state.slot_to_def[slot];
    const bool discarded = do_discard(state, frame.player, slot, MoveZone::Deck);
    (void)discarded;
    assert(discarded);
    record_last_choice(state, frame, def);
    frame.data[DATA_LAST_COST_DEBT] = static_cast<std::int16_t>(def);
    frame.data[DATA_LAST_COUNT] = 1;
    ++frame.pc;
}

void play_last_from_discard(GameState& state, EffectFrame& frame) noexcept {
    const DefId def = static_cast<DefId>(frame.data[DATA_LAST_COST_DEBT]);
    const Slot slot = slot_of(state, def);
    if (slot != NONE && play_slot_from_discard(state, frame.player, slot)) {
        if (!trigger_table_clean_empty(state)) {
            emit(state, TriggerKind::OnPlayAction, TriggerPayload{frame.player, def, slot, 0U});
        }
        const bool pushed = push_effect(state, def, frame.player);
        (void)pushed;
        assert(pushed);
    }
    ++frame.pc;
}

void play_chosen_repeated(GameState& state, EffectFrame& frame, const Instr& instr) noexcept {
    if (frame.data[DATA_LAST_COUNT] == 0) {
        ++frame.pc;
        return;
    }

    const DefId def = static_cast<DefId>(frame.data[DATA_LAST_DEF]);
    const std::uint8_t repeats = static_cast<std::uint8_t>(
        static_cast<std::uint8_t>(instr.arg) * (frame.repeats_left == 0U ? 1U : frame.repeats_left));
    const std::uint8_t old_depth = state.effect_depth;
    const bool pushed = push_effect(state, def, frame.player);
    (void)pushed;
    assert(pushed);
    if (pushed && state.effect_depth > old_depth) {
        state.effect_stack[state.effect_depth - 1U].repeats_left = repeats;
    }
    frame.repeats_left = 1U;
    ++frame.pc;
}

void push_each_other_frames(GameState& state, EffectFrame& frame, const Instr& instr) noexcept {
    const PlayerId source = frame.player;
    for (std::uint8_t offset = static_cast<std::uint8_t>(state.num_players - 1U); offset > 0U; --offset) {
        const PlayerId target = static_cast<PlayerId>((source + offset) % state.num_players);
        assert(state.effect_depth < MAX_EFFECT_DEPTH);
        if (state.effect_depth >= MAX_EFFECT_DEPTH) {
            return;
        }
        EffectFrame each{};
        each.source = frame.source;
        each.pc = instr.a;
        each.player = target;
        each.flags = FRAME_ABSOLUTE_PROGRAM;
        each.repeats_left = 1U;
        each.data[DATA_ACTIVE_PC] = ACTIVE_PC_NONE;
        state.effect_stack[state.effect_depth] = each;
        ++state.effect_depth;
    }
}

[[nodiscard]] Slot bandit_revealed_slot(const EffectFrame& frame, std::uint8_t index) noexcept {
    const std::int16_t value = index == 0U
        ? frame.data[DATA_BANDIT_REVEALED_0]
        : frame.data[DATA_BANDIT_REVEALED_1];
    return value < 0 ? NONE : static_cast<Slot>(value);
}

void set_bandit_revealed_slot(EffectFrame& frame, std::uint8_t index, Slot slot) noexcept {
    if (index == 0U) {
        frame.data[DATA_BANDIT_REVEALED_0] = static_cast<std::int16_t>(slot);
    } else {
        frame.data[DATA_BANDIT_REVEALED_1] = static_cast<std::int16_t>(slot);
    }
}

[[nodiscard]] bool bandit_slot_is_trashable(const GameState& state, Slot slot) noexcept {
    if (slot == NONE) {
        return false;
    }
    const DefId def = state.slot_to_def[slot];
    return def != DEF_COPPER && (card_def(def).types & TYPE_TREASURE) != 0U;
}

[[nodiscard]] std::uint8_t bandit_distinct_trashable_count(const GameState& state, const EffectFrame& frame) noexcept {
    const std::uint8_t revealed_count = static_cast<std::uint8_t>(frame.data[DATA_BANDIT_REVEALED_COUNT]);
    std::uint8_t total = 0;
    DefId seen[2]{NONE, NONE};
    for (std::uint8_t i = 0; i < revealed_count; ++i) {
        const Slot slot = bandit_revealed_slot(frame, i);
        if (!bandit_slot_is_trashable(state, slot)) {
            continue;
        }
        const DefId def = state.slot_to_def[slot];
        bool duplicate = false;
        for (std::uint8_t j = 0; j < total; ++j) {
            if (seen[j] == def) {
                duplicate = true;
            }
        }
        if (!duplicate) {
            seen[total] = def;
            ++total;
        }
    }
    return total;
}

[[nodiscard]] Slot first_bandit_trashable_slot(const GameState& state, const EffectFrame& frame) noexcept {
    const std::uint8_t revealed_count = static_cast<std::uint8_t>(frame.data[DATA_BANDIT_REVEALED_COUNT]);
    for (std::uint8_t i = 0; i < revealed_count; ++i) {
        const Slot slot = bandit_revealed_slot(frame, i);
        if (bandit_slot_is_trashable(state, slot)) {
            return slot;
        }
    }
    return NONE;
}

[[nodiscard]] Slot first_bandit_trashable_slot_matching(
    const GameState& state,
    const EffectFrame& frame,
    DefId def) noexcept {
    const std::uint8_t revealed_count = static_cast<std::uint8_t>(frame.data[DATA_BANDIT_REVEALED_COUNT]);
    for (std::uint8_t i = 0; i < revealed_count; ++i) {
        const Slot slot = bandit_revealed_slot(frame, i);
        if (bandit_slot_is_trashable(state, slot) && state.slot_to_def[slot] == def) {
            return slot;
        }
    }
    return NONE;
}

void resolve_bandit_revealed(GameState& state, EffectFrame& frame, Slot trash_slot) noexcept {
    bool trashed = false;
    const std::uint8_t revealed_count = static_cast<std::uint8_t>(frame.data[DATA_BANDIT_REVEALED_COUNT]);
    for (std::uint8_t i = 0; i < revealed_count; ++i) {
        const Slot slot = bandit_revealed_slot(frame, i);
        if (slot == NONE) {
            continue;
        }
        if (!trashed && slot == trash_slot) {
            const bool moved = do_trash(state, frame.player, slot, MoveZone::Revealed);
            (void)moved;
            assert(moved);
            trashed = true;
        } else {
            const bool moved = do_discard(state, frame.player, slot, MoveZone::Revealed);
            (void)moved;
            assert(moved);
        }
    }
    frame.data[DATA_BANDIT_REVEALED_0] = DATA_NO_SLOT;
    frame.data[DATA_BANDIT_REVEALED_1] = DATA_NO_SLOT;
    frame.data[DATA_BANDIT_REVEALED_COUNT] = 0;
    frame.data[DATA_BANDIT_TRASHABLE_COUNT] = 0;
}

[[nodiscard]] bool suspend_bandit_attack(GameState& state, EffectFrame& frame) noexcept {
    const bool fresh = begin_active_op(frame);
    if (fresh) {
        frame.data[DATA_BANDIT_REVEALED_0] = DATA_NO_SLOT;
        frame.data[DATA_BANDIT_REVEALED_1] = DATA_NO_SLOT;
        frame.data[DATA_BANDIT_REVEALED_COUNT] = 0;
        frame.data[DATA_BANDIT_TRASHABLE_COUNT] = 0;
        for (std::uint8_t i = 0; i < 2U; ++i) {
            const Slot slot = take_deck_top_slot_impl(state, frame.player);
            if (slot == NONE) {
                continue;
            }
            const std::uint8_t revealed_count = static_cast<std::uint8_t>(frame.data[DATA_BANDIT_REVEALED_COUNT]);
            set_bandit_revealed_slot(frame, revealed_count, slot);
            frame.data[DATA_BANDIT_REVEALED_COUNT] = static_cast<std::int16_t>(revealed_count + 1U);
            if (bandit_slot_is_trashable(state, slot)) {
                ++frame.data[DATA_BANDIT_TRASHABLE_COUNT];
            }
        }
    }

    if (frame.data[DATA_BANDIT_TRASHABLE_COUNT] <= 0) {
        resolve_bandit_revealed(state, frame, NONE);
        finish_active_op(state, frame);
        return false;
    }
    if (bandit_distinct_trashable_count(state, frame) <= 1U) {
        resolve_bandit_revealed(state, frame, first_bandit_trashable_slot(state, frame));
        finish_active_op(state, frame);
        return false;
    }

    state.decision = PendingDecision{
        frame.player,
        static_cast<std::uint8_t>(DecisionKind::Choose),
        frame.source,
        1,
        1,
    };
    return true;
}

void resume_bandit_attack(GameState& state, EffectFrame& frame, Action action) noexcept {
    assert(action_is_select(action));
    const DefId def = action_def(action, A_SELECT_BASE);
    const Slot slot = first_bandit_trashable_slot_matching(state, frame, def);
    assert(slot != NONE);
    resolve_bandit_revealed(state, frame, slot);
    finish_active_op(state, frame);
}

void resume_discard_down_to(GameState& state, EffectFrame& frame, Action action) noexcept {
    assert(action_is_select(action));
    const DefId def = action_def(action, A_SELECT_BASE);
    const Slot slot = slot_of(state, def);
    assert(slot != NONE);
    if (slot == NONE || state.players[frame.player].hand[slot] == 0U) {
        assert(false);
        return;
    }
    const std::uint8_t chosen = active_count(frame);
    if (chosen < 3U) {
        frame.data[DATA_LAST_DEF + chosen] = static_cast<std::int16_t>(def);
    }
    frame.data[DATA_ACTIVE_COUNT] = static_cast<std::int16_t>(frame.data[DATA_ACTIVE_COUNT] + 1);
    state.decision = PendingDecision{};
}

void resume_choose(GameState& state, EffectFrame& frame, const Instr& instr, Action action) noexcept {
    (void)begin_active_op(frame);
    if (instr.op == Op::DiscardDownTo) {
        resume_discard_down_to(state, frame, action);
        return;
    }
    if (instr.op == Op::BanditAttack) {
        resume_bandit_attack(state, frame, action);
        return;
    }

    const Filter& filter = filter_def(instr.a);
    if (action_is_pass(action)) {
        assert(state.decision.min_left == 0U);
        finish_active_op(state, frame);
        return;
    }

    assert(action_is_select(action));
    const DefId def = action_def(action, A_SELECT_BASE);
    const Slot slot = slot_of(state, def);
    assert(slot != NONE);
    const bool valid_hand = filter.zone == ZoneSelector::Hand
        && slot != NONE
        && state.players[frame.player].hand[slot] > 0U
        && hand_matches_filter(state, frame, filter, slot);
    const bool valid_discard = filter.zone == ZoneSelector::Discard
        && slot != NONE
        && discard_contains_slot(state.players[frame.player], slot)
        && discard_matches_filter(state, frame, filter, slot);
    if (!valid_hand && !valid_discard) {
        assert(false);
        return;
    }
    apply_then(state, frame, instr, static_cast<Then>(instr.arg), def);
    state.decision = PendingDecision{};
}

void resume_choose_gain(GameState& state, EffectFrame& frame, const Instr& instr, Action action) noexcept {
    (void)begin_active_op(frame);
    if (action_is_pass(action)) {
        assert(state.decision.min_left == 0U);
        finish_active_op(state, frame);
        return;
    }

    assert(action_is_select(action));
    const DefId def = action_def(action, A_SELECT_BASE);
    apply_choose_gain(state, frame, instr, def);
    state.decision = PendingDecision{};
}

void resume_option(GameState& state, EffectFrame& frame, Action action) noexcept {
    assert(action_is_option(action));
    const DefId option = action_def(action, A_OPTION_BASE);
    frame.data[DATA_LAST_DEF] = static_cast<std::int16_t>(option);
    frame.data[DATA_ACTIVE_COUNT] = 1;
    finish_active_op(state, frame);
}

void resume_order(GameState& state, EffectFrame& frame, const Instr& instr, Action action) noexcept {
    assert(action_is_option(action));
    const std::uint8_t option = static_cast<std::uint8_t>(action_def(action, A_OPTION_BASE));
    assert(option < state.decision.max_left);
    const std::uint8_t chosen = active_count(frame);
    if (chosen < 4U) {
        frame.data[DATA_LAST_DEF + chosen] = option;
    }
    ++frame.data[DATA_ACTIVE_COUNT];
    if (active_count(frame) >= instr.a) {
        finish_active_op(state, frame);
    }
}

void resume_react_window(GameState& state, EffectFrame& frame, Action action) noexcept {
    if (!action_is_pass(action)) {
        assert(action_is_select(action));
        const DefId def = action_def(action, A_SELECT_BASE);
        const Slot slot = slot_of(state, def);
        assert(slot != NONE);
        if (slot != NONE
            && state.players[frame.player].hand[slot] > 0U
            && def == DEF_MOAT) {
            frame.flags = static_cast<std::uint8_t>(frame.flags | FRAME_ATTACK_IMMUNE);
        }
    }

    frame.flags = static_cast<std::uint8_t>(frame.flags | FRAME_REACT_DONE);
    state.decision = PendingDecision{};
}

[[nodiscard]] std::uint8_t trigger_order_count(const EffectFrame& frame) noexcept {
    if (frame.data[DATA_LAST_COUNT] <= 0) {
        return 0;
    }
    return static_cast<std::uint8_t>(frame.data[DATA_LAST_COUNT]);
}

void resume_trigger_order(GameState& state, Action action) noexcept {
    assert(action_is_option(action));
    assert(state.effect_depth > 0U);
    if (state.effect_depth == 0U) {
        return;
    }

    const std::uint8_t order_index = static_cast<std::uint8_t>(state.effect_depth - 1U);
    const EffectFrame order = state.effect_stack[order_index];
    const std::uint8_t count = trigger_order_count(order);
    const std::uint8_t option = static_cast<std::uint8_t>(action_def(action, A_OPTION_BASE));
    assert(count > 1U);
    assert(option < count);
    if (count <= 1U || option >= count) {
        state.decision = PendingDecision{};
        return;
    }

    // A_OPTION(k) maps to the kth unresolved trigger frame from bottom to top,
    // among the contiguous frames immediately below this order frame.
    const std::uint8_t first_index = static_cast<std::uint8_t>(order_index - count);
    const std::uint8_t selected_index = static_cast<std::uint8_t>(first_index + option);
    const EffectFrame selected = state.effect_stack[selected_index];

    for (std::uint8_t i = selected_index; static_cast<std::uint8_t>(i + 1U) < order_index; ++i) {
        state.effect_stack[i] = state.effect_stack[static_cast<std::uint8_t>(i + 1U)];
    }

    EffectFrame next_order = order;
    next_order.data[DATA_LAST_COUNT] = static_cast<std::int16_t>(count - 1U);
    state.effect_stack[static_cast<std::uint8_t>(order_index - 1U)] = next_order;
    state.effect_stack[order_index] = selected;
    state.decision = PendingDecision{};
}

} // namespace

Slot take_deck_top_slot(GameState& state, PlayerId player) noexcept {
    return take_deck_top_slot_impl(state, player);
}

void draw_cards(GameState& state, PlayerId player, std::uint8_t count) noexcept {
    PlayerState& player_state = state.players[player];
    std::uint8_t remaining = count;
    while (remaining > 0U) {
        if (player_state.deck.size == 0U) {
            reshuffle_discard_into_deck(state, player);
        }
        if (player_state.deck.size == 0U) {
            return;
        }
        const std::uint8_t take = player_state.deck.size < remaining ? player_state.deck.size : remaining;
        for (std::uint8_t i = 0; i < take; ++i) {
            --player_state.deck.size;
            const Slot card = player_state.deck.cards[player_state.deck.size];
            assert(card < MAX_SLOTS);
            ++player_state.hand[card];
        }
        remaining = static_cast<std::uint8_t>(remaining - take);
    }
}

bool push_effect(GameState& state, DefId source, PlayerId player) noexcept {
    assert(source < card_def_count());
    if (source >= card_def_count()) {
        return false;
    }
    const CardDef& def = card_def(source);
    if (def.on_play.len == 0U && def.custom == nullptr) {
        return true;
    }
    assert(state.effect_depth < MAX_EFFECT_DEPTH);
    if (state.effect_depth >= MAX_EFFECT_DEPTH) {
        return false;
    }

    EffectFrame frame{};
    frame.source = source;
    frame.player = player;
    frame.data[DATA_ACTIVE_PC] = ACTIVE_PC_NONE;
    state.effect_stack[state.effect_depth] = frame;
    ++state.effect_depth;
    return true;
}

bool push_effect_span(
    GameState& state,
    DefId source,
    PlayerId player,
    EffectSpan span,
    std::uint8_t flags) noexcept {
    assert(source < card_def_count());
    if (source >= card_def_count() || span.len == 0U) {
        return source < card_def_count();
    }
    assert(state.effect_depth < MAX_EFFECT_DEPTH);
    if (state.effect_depth >= MAX_EFFECT_DEPTH) {
        return false;
    }

    EffectFrame frame{};
    frame.source = source;
    frame.pc = static_cast<std::uint8_t>(span.offset);
    frame.player = player;
    frame.flags = static_cast<std::uint8_t>(flags | FRAME_ABSOLUTE_PROGRAM);
    frame.data[DATA_ACTIVE_PC] = ACTIVE_PC_NONE;
    state.effect_stack[state.effect_depth] = frame;
    ++state.effect_depth;
    return true;
}

bool push_trigger_order_frame(GameState& state, PlayerId player, DefId source, std::uint8_t count) noexcept {
    assert(state.effect_depth < MAX_EFFECT_DEPTH);
    if (state.effect_depth >= MAX_EFFECT_DEPTH) {
        return false;
    }

    EffectFrame frame{};
    frame.source = source;
    frame.player = player;
    frame.flags = FRAME_TRIGGER_ORDER;
    frame.data[DATA_LAST_COUNT] = count;
    state.effect_stack[state.effect_depth] = frame;
    ++state.effect_depth;
    return true;
}

const Instr& current_effect_instr(const GameState& state) noexcept {
    assert(state.effect_depth > 0U);
    const EffectFrame& frame = state.effect_stack[state.effect_depth - 1U];
    if ((frame.flags & FRAME_ABSOLUTE_PROGRAM) != 0U) {
        assert(frame.pc < effect_instr_count());
        return effect_instr(frame.pc);
    }

    const CardDef& def = card_def(frame.source);
    assert(frame.pc < def.on_play.len);
    return effect_instr(static_cast<std::uint16_t>(def.on_play.offset + frame.pc));
}

RunResult interp_run(GameState& state) noexcept {
    while (state.effect_depth > 0U) {
        EffectFrame& frame = state.effect_stack[state.effect_depth - 1U];
        if ((frame.flags & FRAME_TRIGGER_ORDER) != 0U) {
            const std::uint8_t count = trigger_order_count(frame);
            if (count <= 1U) {
                pop_frame(state);
                continue;
            }
            state.decision = PendingDecision{
                frame.player,
                static_cast<std::uint8_t>(DecisionKind::OrderTriggers),
                frame.source,
                1,
                count,
            };
            return RunResult::NeedDecision;
        }

        if ((frame.flags & FRAME_ATTACK) != 0U && attack_prologue(state, frame)) {
            if (state.decision.kind != static_cast<std::uint8_t>(DecisionKind::None)) {
                return RunResult::NeedDecision;
            }
            continue;
        }

        assert(frame.source < card_def_count());
        const CardDef& def = card_def(frame.source);
        const bool absolute = (frame.flags & FRAME_ABSOLUTE_PROGRAM) != 0U;

        if (!absolute && def.custom != nullptr) {
            const RunResult result = def.custom(state, frame);
            if (result == RunResult::NeedDecision) {
                return result;
            }
            if (result == RunResult::FrameDone) {
                complete_frame(state, frame);
            }
            continue;
        }

        if (!absolute && frame.pc >= def.on_play.len) {
            complete_frame(state, frame);
            continue;
        }

        const std::uint16_t instr_offset = absolute
            ? frame.pc
            : static_cast<std::uint16_t>(def.on_play.offset + frame.pc);
        const Instr& instr = current_effect_instr(state);

        switch (instr.op) {
        case Op::PlusCards:
            if (instr.arg > 0) {
                draw_cards(state, frame.player, static_cast<std::uint8_t>(instr.arg));
            }
            ++frame.pc;
            break;
        case Op::PlusActions:
            add_uint8(state.actions, instr.arg);
            ++frame.pc;
            break;
        case Op::PlusBuys:
            add_uint8(state.buys, instr.arg);
            ++frame.pc;
            break;
        case Op::PlusCoins:
            state.coins = static_cast<std::int16_t>(state.coins + instr.arg);
            ++frame.pc;
            break;
        case Op::End:
            complete_frame(state, frame);
            break;
        case Op::Choose:
            if (suspend_choice(state, frame, instr, DecisionKind::Choose)) {
                return RunResult::NeedDecision;
            }
            break;
        case Op::ChooseGain:
            if (suspend_choice(state, frame, instr, DecisionKind::ChooseGain)) {
                return RunResult::NeedDecision;
            }
            break;
        case Op::ChooseOption:
            suspend_option(state, frame, instr, DecisionKind::ChooseOption);
            return RunResult::NeedDecision;
        case Op::ChooseOrder:
            suspend_order(state, frame, instr);
            return RunResult::NeedDecision;
        case Op::Attack:
            push_attack_frames(state, frame, instr);
            ++frame.pc;
            break;
        case Op::EachOtherPlayer:
            push_each_other_frames(state, frame, instr);
            ++frame.pc;
            break;
        case Op::DiscardDownTo:
            if (suspend_discard_down_to(state, frame, instr)) {
                return RunResult::NeedDecision;
            }
            break;
        case Op::DiscardPerEmptySupply:
            if (suspend_discard_per_empty_supply(state, frame, instr)) {
                return RunResult::NeedDecision;
            }
            break;
        case Op::DiscardDeckTop:
            discard_deck_top(state, frame);
            break;
        case Op::PlayLastFromDiscard:
            play_last_from_discard(state, frame);
            break;
        case Op::PlayChosenRepeated:
            play_chosen_repeated(state, frame, instr);
            break;
        case Op::TrashSelf: {
            const Slot slot = slot_of(state, frame.source);
            if (slot != NONE) {
                (void)do_trash(state, frame.player, slot, MoveZone::InPlay);
            }
            ++frame.pc;
            break;
        }
        case Op::GainSpecific:
            gain_specific(state, frame, instr);
            break;
        case Op::GainCurse:
            gain_curse(state, frame, instr);
            break;
        case Op::BanditAttack:
            if (suspend_bandit_attack(state, frame)) {
                return RunResult::NeedDecision;
            }
            break;
        case Op::PerChosen: {
            const std::uint16_t next_offset = static_cast<std::uint16_t>(instr_offset + 1U);
            assert(next_offset < effect_instr_count());
            execute_multiplied_instr(state, frame, effect_instr(next_offset), frame.data[DATA_LAST_COUNT]);
            frame.pc = static_cast<std::uint8_t>(frame.pc + 2U);
            break;
        }
        case Op::IfElse:
            frame.pc = static_cast<std::uint8_t>(
                frame.pc + (predicate_matches(state, frame, instr) ? instr.b : instr.c));
            break;
        case Op::Repeat:
            if (frame.data[DATA_REPEAT_COUNT] == 0) {
                frame.data[DATA_REPEAT_COUNT] = 1;
            }
            if (frame.data[DATA_REPEAT_COUNT] < instr.b) {
                ++frame.data[DATA_REPEAT_COUNT];
                frame.pc = instr.a;
            } else {
                frame.data[DATA_REPEAT_COUNT] = 0;
                ++frame.pc;
            }
            break;
        default:
            assert(false);
            pop_frame(state);
            break;
        }
    }

    return RunResult::FrameDone;
}

void interp_resume(GameState& state, Action action) noexcept {
    assert(state.effect_depth > 0U);
    if (state.effect_depth == 0U) {
        return;
    }

    EffectFrame& frame = state.effect_stack[state.effect_depth - 1U];
    const DecisionKind kind = static_cast<DecisionKind>(state.decision.kind);
    const bool custom_frame = (frame.flags & FRAME_ABSOLUTE_PROGRAM) == 0U
        && frame.source < card_def_count()
        && card_def(frame.source).custom != nullptr;
    if (custom_frame && (kind == DecisionKind::ChooseOption || kind == DecisionKind::ChooseOrder)) {
        assert(action_is_option(action));
        frame.data[0] = static_cast<std::int16_t>(action_def(action, A_OPTION_BASE));
        ++frame.pc;
        state.decision = PendingDecision{};
        return;
    }

    switch (kind) {
    case DecisionKind::Choose: {
        const Instr& instr = current_effect_instr(state);
        resume_choose(state, frame, instr, action);
        break;
    }
    case DecisionKind::ChooseGain: {
        const Instr& instr = current_effect_instr(state);
        resume_choose_gain(state, frame, instr, action);
        break;
    }
    case DecisionKind::ChooseOption:
        resume_option(state, frame, action);
        break;
    case DecisionKind::ChooseOrder: {
        const Instr& instr = current_effect_instr(state);
        resume_order(state, frame, instr, action);
        break;
    }
    case DecisionKind::ReactWindow:
        resume_react_window(state, frame, action);
        break;
    case DecisionKind::OrderTriggers:
        resume_trigger_order(state, action);
        break;
    default:
        assert(false);
        break;
    }

    (void)interp_run(state);
}
