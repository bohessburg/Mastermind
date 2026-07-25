from __future__ import annotations

import asyncio

from src.v2.arena.actuate.clicks import MockActuator
from src.v2.arena.fsm.game import run_game_loop
from src.v2.arena.main import (
    _recover_rejected_resumed_game,
    _start_or_resume_game,
)
from src.v2.arena.protocol.events import FullState, GameEnd, GameStart


class _UnusedProvider:
    async def plan(self, **_: object) -> None:
        raise AssertionError("synthetic startup stream contains no decision")


class _StartupLobby:
    def __init__(
        self,
        *,
        has_running_game: bool,
        modal_results: tuple[bool, ...] = (),
    ) -> None:
        self.has_running_game = has_running_game
        self.modal_results = list(modal_results)
        self.queue_calls = 0
        self.recovery_calls = 0
        self.startup_modal_checks = 0

    async def resolve_startup_blocking_modal(self) -> bool:
        self.startup_modal_checks += 1
        return self.modal_results.pop(0) if self.modal_results else False

    async def resume_running_game_if_present(self) -> bool:
        return self.has_running_game

    async def queue_next_game(self) -> None:
        self.queue_calls += 1

    async def recover_resumed_game_and_queue_next(self) -> None:
        self.recovery_calls += 1


def _game_start() -> GameStart:
    return GameStart(
        game_id=991,
        kingdom=(),
        players=("bot", "opponent"),
        player_ids=(1, 2),
        our_seat=0,
    )


def _full_state(*, game_id: int = 991) -> FullState:
    return FullState(
        game_id=game_id,
        replacement=False,
        card_counts=(),
        zones=(),
        counters=(),
    )


def test_startup_running_game_buffers_full_state_then_invokes_game_loop() -> None:
    async def run() -> None:
        lobby = _StartupLobby(has_running_game=True)
        events = asyncio.Queue()
        start = _game_start()
        full_state = _full_state()
        await events.put(start)
        await events.put(full_state)
        await events.put(GameEnd(game_id=start.game_id, reason="completed"))

        resumed, game_events = await _start_or_resume_game(
            lobby,
            events,
            resume_full_state_timeout_seconds=1.0,
        )

        assert resumed
        assert game_events is not None
        results = await run_game_loop(
            game_events,
            actuator=MockActuator(replay=True),
            decision_provider=_UnusedProvider(),
            max_games=1,
        )
        assert len(results) == 1
        assert results[0].completed
        assert lobby.queue_calls == 0
        assert lobby.recovery_calls == 0
        assert lobby.startup_modal_checks == 1

    asyncio.run(run())


def test_startup_rechecks_for_a_running_game_after_a_blocking_modal() -> None:
    async def run() -> None:
        lobby = _StartupLobby(
            has_running_game=True,
            modal_results=(True,),
        )
        events = asyncio.Queue()
        start = _game_start()
        await events.put(start)
        await events.put(_full_state())
        await events.put(GameEnd(game_id=start.game_id, reason="completed"))

        resumed, game_events = await _start_or_resume_game(
            lobby,
            events,
            resume_full_state_timeout_seconds=1.0,
        )

        assert resumed
        assert game_events is not None
        assert lobby.queue_calls == 0
        assert lobby.startup_modal_checks == 2

    asyncio.run(run())


def test_resume_without_full_state_recovers_to_the_search_cycle() -> None:
    async def run() -> None:
        lobby = _StartupLobby(has_running_game=True)
        events = asyncio.Queue()

        resumed, game_events = await _start_or_resume_game(
            lobby,
            events,
            resume_full_state_timeout_seconds=0.01,
        )

        assert not resumed
        assert game_events is None
        assert lobby.queue_calls == 0
        assert lobby.recovery_calls == 1
        assert lobby.startup_modal_checks == 1

    asyncio.run(run())


def test_rejected_resumed_full_state_recovers_to_the_search_cycle() -> None:
    async def run() -> None:
        start = _game_start()
        result = (
            await run_game_loop(
                (start, _full_state(game_id=start.game_id + 1)),
                actuator=MockActuator(replay=True),
                decision_provider=_UnusedProvider(),
                max_games=1,
            )
        )[0]
        lobby = _StartupLobby(has_running_game=True)

        assert await _recover_rejected_resumed_game(lobby, result)
        assert result.divergence_report is not None
        assert result.divergence_report.error_type == "TrackerError"
        assert lobby.recovery_calls == 1

    asyncio.run(run())


def test_startup_homepage_keeps_the_existing_search_path() -> None:
    async def run() -> None:
        lobby = _StartupLobby(has_running_game=False)
        events = asyncio.Queue()

        resumed, game_events = await _start_or_resume_game(
            lobby,
            events,
            resume_full_state_timeout_seconds=1.0,
        )

        assert not resumed
        assert game_events is None
        assert lobby.queue_calls == 1
        assert lobby.recovery_calls == 0
        assert lobby.startup_modal_checks == 1

    asyncio.run(run())
