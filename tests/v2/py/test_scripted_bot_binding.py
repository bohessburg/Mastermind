"""Standalone checks for the persistent native scripted-bot binding."""

from __future__ import annotations

import dominion_v2_py as dz


KINGDOM = [
    "Sentry",
    "Library",
    "Throne Room",
    "Bandit",
    "Witch",
    "Moat",
    "Village",
    "Smithy",
    "Market",
    "Remodel",
]
STEP_CAP = 10_000


def _play(seed: int) -> list[int]:
    game = dz.new_game(dz.Setup(players=2, kingdom=KINGDOM), seed)
    bots = [
        dz.ScriptedBot("engine3"),
        dz.ScriptedBot(dz.EvalScriptedBotKind.EngineV3),
    ]
    actions: list[int] = []

    for _ in range(STEP_CAP):
        if game.game_over():
            return actions
        player = int(game.current_decision()["player"])
        legal = game.legal_mask()
        action = int(bots[player].choose(game))
        assert 0 <= action < dz.ACTION_SPACE_SIZE
        assert bool(legal[action]), f"bot chose illegal action {action}"
        actions.append(action)
        game.step(action)

    raise AssertionError(f"game did not complete within {STEP_CAP} steps")


def test_engine3_completes_deterministically_with_legal_actions() -> None:
    first = _play(0xE3B07)
    second = _play(0xE3B07)
    assert first
    assert first == second


def main() -> None:
    test_engine3_completes_deterministically_with_legal_actions()
    print("test_scripted_bot_binding: PASS")


if __name__ == "__main__":
    main()
