#include "v2/core/actions.h"

#include "v2/core/interp.h"
#include "v2/core/moves.h"
#include "v2/core/setup.h"
#include "v2/core/triggers.h"
#include "v2/core/turns.h"

#include <cassert>
#include <cstdint>

namespace {

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

[[nodiscard]] Pile* find_buy_pile(GameState& state, DefId def) noexcept {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        Pile& pile = state.piles[i];
        if (pile_has_cards(pile) && pile_top_def(state, pile) == def) {
            return &pile;
        }
    }
    return nullptr;
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

[[nodiscard]] Cost current_budget(const GameState& state) noexcept {
    return Cost{
        budget_component(state.coins),
        budget_component(static_cast<std::int16_t>(state.potion_coins)),
        0,
    };
}

void add_action(ActionMask& out, int& count, Action action) noexcept {
    if (!out.test(action)) {
        out.set(action);
        ++count;
    }
}

void add_buy_actions(const GameState& state, ActionMask& out, int& count) noexcept {
    if (state.buys == 0U || state.players[current_player(state)].debt != 0U) {
        return;
    }

    const Cost budget = current_budget(state);
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        const Pile& pile = state.piles[i];
        if (!pile_has_cards(pile)) {
            continue;
        }
        const DefId def = pile_top_def(state, pile);
        if (effective_cost(state, def).fits_within(budget)) {
            add_action(out, count, buy_action(def));
        }
    }
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
        const std::int16_t coins = static_cast<std::int16_t>(frame.data[2] + filter.coin_delta);
        return Cost{
            budget_component(coins),
            budget_component(frame.data[3]),
            frame.data[4],
        };
    }
    (void)state;
    return Cost{127, 127, 32767};
}

[[nodiscard]] bool cost_matches(const GameState& state, const EffectFrame& frame, const Filter& filter, DefId def) noexcept {
    if (filter.cost_kind == CostLimitKind::None) {
        return true;
    }
    return effective_cost(state, def).fits_within(limit_from_filter(state, frame, filter));
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

[[nodiscard]] const Instr& current_decision_instr(const GameState& state) noexcept {
    return current_effect_instr(state);
}

[[nodiscard]] bool hand_selectable_for_decision(
    const GameState& state,
    const EffectFrame& frame,
    const Instr& instr,
    Slot slot) noexcept {
    if (instr.op == Op::DiscardDownTo) {
        const DefId def = state.slot_to_def[slot];
        std::uint8_t chosen = 0;
        const std::uint8_t active = frame.data[6] < 0 ? 0U : static_cast<std::uint8_t>(frame.data[6]);
        for (std::uint8_t i = 0; i < active && i < 3U; ++i) {
            if (frame.data[1 + i] == static_cast<std::int16_t>(def)) {
                ++chosen;
            }
        }
        return state.players[state.decision.player].hand[slot] > chosen;
    }
    if (instr.op == Op::BanditAttack) {
        const DefId def = state.slot_to_def[slot];
        if (def == DEF_COPPER || (card_def(def).types & TYPE_TREASURE) == 0U) {
            return false;
        }
        const std::uint8_t revealed_count = frame.data[3] < 0 ? 0U : static_cast<std::uint8_t>(frame.data[3]);
        for (std::uint8_t i = 0; i < revealed_count && i < 2U; ++i) {
            const std::int16_t revealed = frame.data[1 + i];
            if (revealed >= 0 && state.slot_to_def[static_cast<Slot>(revealed)] == def) {
                return true;
            }
        }
        return false;
    }

    const Filter& filter = filter_def(instr.a);
    if (filter.zone == ZoneSelector::Discard) {
        const PlayerState& player = state.players[frame.player];
        for (std::uint8_t i = 0; i < player.discard.size; ++i) {
            if (player.discard.cards[i] == slot && discard_matches_filter(state, frame, filter, slot)) {
                return true;
            }
        }
        return false;
    }
    return state.players[frame.player].hand[slot] > 0U && hand_matches_filter(state, frame, filter, slot);
}

void add_choose_actions(const GameState& state, ActionMask& out, int& count) noexcept {
    const EffectFrame& frame = state.effect_stack[state.effect_depth - 1U];
    const Instr& instr = current_decision_instr(state);

    if (state.decision.min_left == 0U) {
        add_action(out, count, A_PASS);
    }
    if (state.decision.max_left == 0U) {
        return;
    }

    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        if (hand_selectable_for_decision(state, frame, instr, slot)) {
            add_action(out, count, select_action(state.slot_to_def[slot]));
        }
    }
}

void add_choose_gain_actions(const GameState& state, ActionMask& out, int& count) noexcept {
    const EffectFrame& frame = state.effect_stack[state.effect_depth - 1U];
    const Instr& instr = current_decision_instr(state);
    const Filter& filter = filter_def(instr.a);

    if (state.decision.min_left == 0U) {
        add_action(out, count, A_PASS);
    }
    if (state.decision.max_left == 0U) {
        return;
    }

    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        if (supply_matches_filter(state, frame, filter, state.piles[i])) {
            add_action(out, count, select_action(pile_top_def(state, state.piles[i])));
        }
    }
}

void add_option_actions(const GameState& state, ActionMask& out, int& count) noexcept {
    const std::uint8_t options = state.decision.max_left;
    for (std::uint8_t i = 0; i < options && i < 16U; ++i) {
        add_action(out, count, option_action(i));
    }
}

void add_react_actions(const GameState& state, ActionMask& out, int& count) noexcept {
    add_action(out, count, A_PASS);
    const PlayerState& player = state.players[state.decision.player];
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        if (player.hand[slot] == 0U) {
            continue;
        }
        const DefId def = state.slot_to_def[slot];
        if ((card_def(def).types & TYPE_REACTION) != 0U) {
            add_action(out, count, select_action(def));
        }
    }
}

void add_action_card_actions(const GameState& state, ActionMask& out, int& count) noexcept {
    if (state.actions == 0U) {
        return;
    }

    const PlayerState& player = state.players[current_player(state)];
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        if (player.hand[slot] == 0U) {
            continue;
        }
        const DefId def = state.slot_to_def[slot];
        if ((card_def(def).types & TYPE_ACTION) != 0U) {
            add_action(out, count, play_action(def));
        }
    }
}

void add_treasure_actions(const GameState& state, ActionMask& out, int& count) noexcept {
    const PlayerState& player = state.players[current_player(state)];
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        if (player.hand[slot] == 0U) {
            continue;
        }
        const DefId def = state.slot_to_def[slot];
        if ((card_def(def).types & TYPE_TREASURE) != 0U) {
            add_action(out, count, play_action(def));
        }
    }
}

void spend_cost(GameState& state, const Cost& cost) noexcept {
    state.coins = static_cast<std::int16_t>(state.coins - cost.coins);
    state.potion_coins = static_cast<std::uint8_t>(state.potion_coins - cost.potion);
    state.players[current_player(state)].debt = static_cast<std::uint8_t>(
        state.players[current_player(state)].debt + cost.debt);
}

void play_card_to_in_play(GameState& state, DefId def) noexcept {
    const Slot slot = slot_of(state, def);
    assert(slot != NONE);

    PlayerState& player = state.players[current_player(state)];
    assert(player.hand[slot] > 0U);
    --player.hand[slot];

    assert(player.in_play_size < MAX_IN_PLAY);
    player.in_play[player.in_play_size] = InPlayEntry{slot, slot, 0};
    ++player.in_play_size;
    mark_trigger_table_dirty(state);
}

[[nodiscard]] std::uint8_t in_play_count(const GameState& state, PlayerId player_id, DefId def) noexcept {
    const PlayerState& player = state.players[player_id];
    std::uint8_t total = 0;
    for (std::uint8_t i = 0; i < player.in_play_size; ++i) {
        if (state.slot_to_def[player.in_play[i].slot] == def) {
            ++total;
        }
    }
    return total;
}

void play_treasure(GameState& state, DefId def) noexcept {
    const PlayerId player = current_player(state);
    play_card_to_in_play(state, def);
    if (def == DEF_POTION) {
        ++state.potion_coins;
    } else {
        state.coins = static_cast<std::int16_t>(state.coins + card_def(def).coin_value);
    }
    const bool pushed = push_effect(state, def, current_player(state));
    (void)pushed;
    assert(pushed);
    emit(state, TriggerKind::OnPlayTreasure, TriggerPayload{player, def, slot_of(state, def), 0U});
    if (def == DEF_SILVER && in_play_count(state, player, DEF_SILVER) == 1U) {
        emit(state, TriggerKind::OnFirstPlay, TriggerPayload{player, def, slot_of(state, def), 0U});
    }
}

void play_action_card(GameState& state, DefId def) noexcept {
    assert(state.actions > 0U);
    const PlayerId player = current_player(state);
    --state.actions;
    play_card_to_in_play(state, def);
    const bool pushed = push_effect(state, def, player);
    (void)pushed;
    assert(pushed);
    emit(state, TriggerKind::OnPlayAction, TriggerPayload{player, def, slot_of(state, def), 0U});
}

void buy_card(GameState& state, DefId def) noexcept {
    Pile* pile = find_buy_pile(state, def);
    assert(pile != nullptr);
    if (pile == nullptr) {
        return;
    }

    const Slot gained_slot = pile_top_slot(*pile);
    const Cost cost = effective_cost(state, def);
    assert(cost.fits_within(current_budget(state)));
    spend_cost(state, cost);
    --state.buys;
    const bool gained = do_gain(state, current_player(state), gained_slot, GainDestination::Discard);
    (void)gained;
    assert(gained);
}

} // namespace

bool action_is_pass(Action action) noexcept {
    return action == A_PASS;
}

bool action_is_play(Action action) noexcept {
    return action >= A_PLAY_BASE && action < A_WAY_BASE;
}

bool action_is_buy(Action action) noexcept {
    return action >= A_BUY_BASE && action < A_EVENT_BASE;
}

bool action_is_select(Action action) noexcept {
    return action >= A_SELECT_BASE && action < A_OPTION_BASE;
}

bool action_is_option(Action action) noexcept {
    return action >= A_OPTION_BASE && action < A_CALL_BASE;
}

DefId action_def(Action action, Action base) noexcept {
    return static_cast<DefId>(action - base);
}

Cost effective_cost(const GameState& state, DefId def) noexcept {
    (void)state;
    return card_def(def).cost;
}

int legal_actions(const GameState& state, ActionMask& out) noexcept {
    out.reset();
    int count = 0;

    const Phase phase = static_cast<Phase>(state.phase);
    if (phase == Phase::Over || state.turn_queue.size == 0U) {
        return 0;
    }

    const DecisionKind decision_kind = static_cast<DecisionKind>(state.decision.kind);
    if (decision_kind == DecisionKind::Choose) {
        add_choose_actions(state, out, count);
        return count;
    }
    if (decision_kind == DecisionKind::ChooseGain) {
        add_choose_gain_actions(state, out, count);
        return count;
    }
    if (decision_kind == DecisionKind::ChooseOption || decision_kind == DecisionKind::ChooseOrder) {
        add_option_actions(state, out, count);
        return count;
    }
    if (decision_kind == DecisionKind::ReactWindow) {
        add_react_actions(state, out, count);
        return count;
    }
    if (decision_kind == DecisionKind::OrderTriggers) {
        add_option_actions(state, out, count);
        return count;
    }

    switch (phase) {
    case Phase::Action:
        add_action(out, count, A_PASS);
        add_action_card_actions(state, out, count);
        break;
    case Phase::Buy:
        add_action(out, count, A_PASS);
        add_treasure_actions(state, out, count);
        add_buy_actions(state, out, count);
        break;
    case Phase::Night:
        add_action(out, count, A_PASS);
        break;
    case Phase::Cleanup:
        add_action(out, count, A_PASS);
        break;
    case Phase::Over:
        break;
    }

    return count;
}

void apply_action(GameState& state, Action action) noexcept {
    const DecisionKind decision_kind = static_cast<DecisionKind>(state.decision.kind);
    if (decision_kind == DecisionKind::Choose
        || decision_kind == DecisionKind::ChooseGain
        || decision_kind == DecisionKind::ChooseOption
        || decision_kind == DecisionKind::ChooseOrder
        || decision_kind == DecisionKind::ReactWindow
        || decision_kind == DecisionKind::OrderTriggers) {
        interp_resume(state, action);
        return;
    }

    const Phase phase = static_cast<Phase>(state.phase);

    if (action_is_pass(action)) {
        if (phase == Phase::Action) {
            state.phase = static_cast<std::uint8_t>(Phase::Buy);
        } else if (phase == Phase::Buy || phase == Phase::Night) {
            state.phase = static_cast<std::uint8_t>(Phase::Cleanup);
        }
        refresh_current_decision(state);
        return;
    }

    if (phase == Phase::Buy && action_is_play(action)) {
        const DefId def = action_def(action, A_PLAY_BASE);
        play_treasure(state, def);
        refresh_current_decision(state);
        return;
    }

    if (phase == Phase::Action && action_is_play(action)) {
        const DefId def = action_def(action, A_PLAY_BASE);
        play_action_card(state, def);
        refresh_current_decision(state);
        return;
    }

    if (phase == Phase::Buy && action_is_buy(action)) {
        const DefId def = action_def(action, A_BUY_BASE);
        buy_card(state, def);
        refresh_current_decision(state);
        return;
    }

    assert(false);
}

void refresh_current_decision(GameState& state) noexcept {
    if (state.phase == static_cast<std::uint8_t>(Phase::Over)
        || state.phase == static_cast<std::uint8_t>(Phase::Cleanup)
        || state.turn_queue.size == 0U) {
        state.decision = PendingDecision{};
        return;
    }

    DecisionKind kind = DecisionKind::None;
    if (state.phase == static_cast<std::uint8_t>(Phase::Action)) {
        kind = DecisionKind::PhaseAction;
    } else if (state.phase == static_cast<std::uint8_t>(Phase::Buy)) {
        kind = DecisionKind::PhaseBuy;
    } else if (state.phase == static_cast<std::uint8_t>(Phase::Night)) {
        kind = DecisionKind::PhaseNight;
    }

    state.decision = PendingDecision{
        current_player(state),
        static_cast<std::uint8_t>(kind),
        0,
        0,
        0,
    };
}
