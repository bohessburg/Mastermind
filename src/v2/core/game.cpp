#include "v2/core/game.h"

#include "v2/core/defs.h"
#include "v2/core/interp.h"
#include "v2/core/turns.h"

#include <cassert>

namespace {

constexpr bool kAutoAdvanceForcedPass = true;

[[nodiscard]] bool pile_has_cards(const Pile& pile) noexcept {
    return pile.mixed_len > 0U || pile.count > 0U;
}

[[nodiscard]] Slot pile_top_slot(const Pile& pile) noexcept {
    if (pile.mixed_len > 0U) {
        return pile.mixed[pile.mixed_len - 1U];
    }
    return pile.base;
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

[[nodiscard]] bool has_action_card_in_hand(const GameState& state, PlayerId player_id) noexcept {
    if (state.actions == 0U) {
        return false;
    }
    const PlayerState& player = state.players[player_id];
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        if (player.hand[slot] != 0U && (card_def(state.slot_to_def[slot]).types & TYPE_ACTION) != 0U) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool has_treasure_in_hand(const GameState& state, PlayerId player_id) noexcept {
    const PlayerState& player = state.players[player_id];
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        if (player.hand[slot] != 0U && (card_def(state.slot_to_def[slot]).types & TYPE_TREASURE) != 0U) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool has_affordable_buy(const GameState& state, PlayerId player_id) noexcept {
    if (state.buys == 0U || state.players[player_id].debt != 0U) {
        return false;
    }
    const Cost budget = current_budget(state);
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        const Pile& pile = state.piles[i];
        if (!pile_has_cards(pile)) {
            continue;
        }
        const DefId def = state.slot_to_def[pile_top_slot(pile)];
        if (effective_cost(state, def).fits_within(budget)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool current_phase_forced_pass(const GameState& state) noexcept {
    if (state.turn_queue.size == 0U) {
        return false;
    }
    const PlayerId player = current_player(state);
    const Phase phase = static_cast<Phase>(state.phase);
    switch (phase) {
    case Phase::Action:
        return !has_action_card_in_hand(state, player);
    case Phase::Buy:
        return !has_treasure_in_hand(state, player) && !has_affordable_buy(state, player);
    case Phase::Night:
        return true;
    case Phase::Cleanup:
    case Phase::Over:
        return false;
    }
    return false;
}

[[nodiscard]] bool drive_until_decision(GameState& state) noexcept {
    for (;;) {
        if (state.effect_depth > 0U) {
            const RunResult result = interp_run(state);
            if (result == RunResult::NeedDecision) {
                return false;
            }
        }

        while (advance_turn_machinery(state)) {
            if (state.effect_depth > 0U) {
                const RunResult turn_result = interp_run(state);
                if (turn_result == RunResult::NeedDecision) {
                    return false;
                }
            }
        }

        if (state.phase == static_cast<std::uint8_t>(Phase::Over)) {
            refresh_current_decision(state);
            return true;
        }

        refresh_current_decision(state);
        if (!kAutoAdvanceForcedPass) {
            return false;
        }

        if (current_phase_forced_pass(state)) {
            apply_action(state, A_PASS);
            continue;
        }
        return false;
    }
}

} // namespace

GameState Game::new_game(const Setup& setup, std::uint64_t seed) noexcept {
    return ::new_game(setup, seed);
}

PendingDecision Game::current_decision(const GameState& state) noexcept {
    return state.decision;
}

int Game::legal_actions(const GameState& state, ActionMask& out) noexcept {
    return ::legal_actions(state, out);
}

bool Game::step(GameState& state, Action action) noexcept {
#ifndef NDEBUG
    ActionMask legal{};
    (void)::legal_actions(state, legal);
    assert(action < ACTION_SPACE_SIZE && legal.test(action));
#endif

    apply_action(state, action);
    return drive_until_decision(state);
}
