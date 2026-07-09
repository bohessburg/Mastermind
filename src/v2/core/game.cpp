#include "v2/core/game.h"

#include "v2/core/interp.h"
#include "v2/core/turns.h"

#include <cassert>

namespace {

constexpr bool kAutoAdvanceForcedPass = true;

[[nodiscard]] bool drive_until_decision(GameState& state) noexcept {
    for (;;) {
        const RunResult result = interp_run(state);
        if (result == RunResult::NeedDecision) {
            return false;
        }

        while (advance_turn_machinery(state)) {
            const RunResult turn_result = interp_run(state);
            if (turn_result == RunResult::NeedDecision) {
                return false;
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

        ActionMask mask{};
        const int count = legal_actions(state, mask);
        if (count == 1 && mask.test(A_PASS)) {
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
