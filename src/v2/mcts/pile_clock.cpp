#include "v2/mcts/pile_clock.h"

#include "v2/core/score.h"
#include "v2/core/turns.h"

namespace {

[[nodiscard]] PlayerId player_to_move(const GameState& state) noexcept {
    if (state.decision.player < state.num_players) {
        return state.decision.player;
    }
    return current_player(state);
}

[[nodiscard]] PlayerId other_player(PlayerId player) noexcept {
    return static_cast<PlayerId>(player == 0U ? 1U : 0U);
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

[[nodiscard]] bool is_victory_def(DefId def) noexcept {
    return def < card_def_count() && (card_def(def).types & TYPE_VICTORY) != 0U;
}

[[nodiscard]] bool better_victory_buy(DefId candidate, DefId current) noexcept {
    if (current == NONE) {
        return true;
    }
    const CardDef& candidate_card = card_def(candidate);
    const CardDef& current_card = card_def(current);
    if (candidate_card.vp != current_card.vp) {
        return candidate_card.vp > current_card.vp;
    }
    return candidate_card.cost.coins > current_card.cost.coins;
}

void consider_victory_buy(
    DefId def,
    Action action,
    Action& best_action,
    DefId& best_def) noexcept {
    if (better_victory_buy(def, best_def)) {
        best_action = action;
        best_def = def;
    }
}

[[nodiscard]] bool action_mask_any(const ActionMask& legal) noexcept {
    for (std::uint16_t i = 0; i < ACTION_MASK_WORDS; ++i) {
        if (legal.words[i] != 0U) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] ActionMask without_ending_buys(
    const ActionMask& legal,
    const ActionMask& ending_buys) noexcept {
    ActionMask safe = legal;
    for (std::uint16_t i = 0; i < ACTION_MASK_WORDS; ++i) {
        safe.words[i] &= ~ending_buys.words[i];
    }
    return safe;
}

} // namespace

PileClock analyze_pile_clock(const GameState& state, const ActionMask& legal) noexcept {
    PileClock clock{};
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        const Pile& pile = state.piles[i];
        const int count = pile_count(pile);
        if (count == 0) {
            ++clock.empty_piles;
            continue;
        }

        const DefId def = pile_top_def(state, pile);
        const Action action = buy_action(def);
        if (!legal.test(action)) {
            continue;
        }

        const bool ends_game = count == 1;
        if (ends_game) {
            clock.ending_buys.set(action);
            if (clock.ending_buy == A_PASS) {
                clock.ending_buy = action;
            }
        }
        if (!is_victory_def(def)) {
            continue;
        }
        consider_victory_buy(def, action, clock.victory_buy, clock.victory_def);
        if (!ends_game) {
            consider_victory_buy(
                def,
                action,
                clock.non_ending_victory_buy,
                clock.non_ending_victory_def);
        }
    }
    return clock;
}

Action pile_clock_guarded_buy(
    const GameState& state,
    const ActionMask& legal,
    const PileClock& clock,
    PileClockBaseBuyFn base_buy) noexcept {
    if (clock.empty_piles < 2 || state.num_players != 2U) {
        return base_buy(state, legal);
    }

    const PlayerId player = player_to_move(state);
    if (player >= state.num_players) {
        return base_buy(state, legal);
    }
    const std::int16_t my_score = score(state, player);
    const std::int16_t opponent_score = score(state, other_player(player));
    if (my_score > opponent_score) {
        // With two piles empty, a one-card pile ends the game immediately.
        // If that is unavailable, bank the best legal VP card instead.
        if (clock.ending_buy != A_PASS) {
            return clock.ending_buy;
        }
        if (clock.victory_buy != A_PASS) {
            return clock.victory_buy;
        }
        return base_buy(state, legal);
    }
    if (my_score < opponent_score) {
        // Do not hand a lead to the opponent by closing a third pile. Prefer
        // VP that keeps the game live, then run the normal chart with ending
        // buys removed.
        if (clock.non_ending_victory_buy != A_PASS) {
            return clock.non_ending_victory_buy;
        }
        const ActionMask safe_legal = without_ending_buys(legal, clock.ending_buys);
        if (action_mask_any(safe_legal)) {
            return base_buy(state, safe_legal);
        }
    }
    return base_buy(state, legal);
}
