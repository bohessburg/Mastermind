"""Supervised entrypoint for one live dominion.games arena session."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path
from typing import Any

from .actuate.clicks import MockActuator, PlaywrightActuator
from .archive import GameArchive
from .bot.policy import NNCheckpointError, load_policy
from .config import ArenaConfig
from .fsm.game import (
    NNMCTSDecisionProvider,
    RecordedDecisionProvider,
    run_game_loop,
)
from .fsm.lobby import LobbyError, LobbyFSM
from .protocol.events import GameEvent
from .protocol.live import RawFrameQueue, events_from_queue
from .protocol.recording import events_from_recording


REFERENCE_RECORDING = Path(
    "arena-recordings/20260724T142103.096991Z/frames.jsonl"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the supervised arena bot")
    parser.add_argument("--config", type=Path, default=Path("configs/arena.json"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="replay the reference capture through the async live pump",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    config = ArenaConfig.load(args.config)
    policy = _load_policy_or_raise(config)
    provider = NNMCTSDecisionProvider(
        policy,
        sims=config.sims,
        determinizations=config.determinizations,
        wall_clock_cap=config.wall_clock_cap_seconds,
    )
    if args.dry_run:
        return await _run_dry(config)
    return await _run_live(config, provider)


def _load_policy_or_raise(config: ArenaConfig) -> Any:
    checkpoint = config.checkpoint_path
    if not checkpoint.is_file():
        raise RuntimeError(
            f"arena checkpoint is missing: {checkpoint}. "
            "Set checkpoint_path in the arena config to an available checkpoint."
        )
    try:
        return load_policy(checkpoint, obs_version=config.obs_version)
    except NNCheckpointError as error:
        raise RuntimeError(
            f"arena checkpoint could not be loaded from {checkpoint}: {error}"
        ) from error


async def _run_dry(config: ArenaConfig) -> int:
    if not REFERENCE_RECORDING.is_file():
        raise RuntimeError(
            f"reference arena recording is missing: {REFERENCE_RECORDING}"
        )
    reference_events = events_from_recording(REFERENCE_RECORDING)
    queue: RawFrameQueue = asyncio.Queue()
    producer = asyncio.create_task(
        _enqueue_recording(REFERENCE_RECORDING, queue),
        name="arena-dry-run-record-producer",
    )
    actuator = MockActuator(replay=True)
    try:
        results = await run_game_loop(
            events_from_queue(queue),
            actuator=actuator,
            decision_provider=RecordedDecisionProvider(reference_events),
            # The recorded answers validate the live plumbing; applying the
            # operator-facing polite delay here would turn dry-run into a
            # multi-minute test without exercising additional behavior.
            think_time_min_seconds=0.0,
            think_time_max_seconds=0.0,
            auto_deny_undo=config.undo.auto_deny,
        )
    finally:
        await producer

    diverged = any(result.divergence_aborted for result in results)
    completed = sum(result.completed for result in results)
    print(
        f"dry run: {completed}/{len(results)} games completed; "
        f"{len(actuator.gestures)} replay gestures",
        flush=True,
    )
    if diverged:
        print("dry run divergence detected", file=sys.stderr, flush=True)
        return 1
    if completed != 3:
        print("dry run did not reach all three reference games", file=sys.stderr)
        return 1
    return 0


async def _enqueue_recording(path: Path, queue: RawFrameQueue) -> None:
    """Feed fixture JSONL through the same queue used by the browser binding."""
    try:
        with path.open(encoding="utf-8") as frames_file:
            for line_number, line in enumerate(frames_file, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"{path}:{line_number}: invalid raw frame JSON"
                    ) from error
                if not isinstance(record, dict):
                    raise RuntimeError(
                        f"{path}:{line_number}: raw frame record must be an object"
                    )
                await queue.put(record)
    finally:
        await queue.put(None)


async def _run_live(
    config: ArenaConfig,
    provider: NNMCTSDecisionProvider,
) -> int:
    # Keep this import out of the dry path: --dry-run must need no browser.
    from .browser.session import ArenaSession

    session = ArenaSession()
    event_pump: asyncio.Task[None] | None = None
    modal_monitor: asyncio.Task[None] | None = None
    try:
        await session.start()
        assert session.run_dir is not None
        assert session.page is not None
        print(f"live archive: {session.run_dir}", flush=True)
        if not (config.arena_user and config.arena_pass):
            print(
                "ARENA_USER/ARENA_PASS are not both set; log in through the "
                "persistent browser profile before the homepage timeout.",
                flush=True,
            )

        async def screenshot(frame_index: int, _snapshot: object) -> Path | None:
            destination = session.run_dir / f"divergence-{frame_index}.png"
            try:
                return await session.screenshot(destination)
            except Exception as error:
                print(
                    f"ERROR: could not capture divergence screenshot: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                return None

        async def resign() -> None:
            await _request_resign(session.page)

        def archive_factory(game_id: int | None) -> GameArchive:
            # GameArchive refreshes this mirror at finish, so the completed
            # game directory itself remains a replayable JSONL fixture.
            return GameArchive(
                session.run_dir,
                game_id=game_id,
                source_frames=session.frames_path,
            )

        events: asyncio.Queue[GameEvent | None] = asyncio.Queue()
        event_pump = asyncio.create_task(
            _pump_session_events(session.events(), events),
            name="arena-live-event-pump",
        )
        lobby = LobbyFSM(
            session.page,
            config.lobby,
            snapshot_dom=session.snapshot_dom,
        )
        actuator = PlaywrightActuator(
            session.page,
            snapshot_dom=session.snapshot_dom,
        )
        modal_monitor = asyncio.create_task(
            _monitor_modals(actuator),
            name="arena-modal-monitor",
        )
        await lobby.queue_next_game()

        completed_games = 0
        while True:
            try:
                results = await asyncio.wait_for(
                    run_game_loop(
                        _events_for_one_game(events),
                        actuator=actuator,
                        decision_provider=provider,
                        think_time_min_seconds=config.think_time_min_seconds,
                        think_time_max_seconds=config.think_time_max_seconds,
                        screenshot_hook=screenshot,
                        resign_hook=resign,
                        archive_factory=archive_factory,
                        max_games=1,
                        auto_deny_undo=config.undo.auto_deny,
                    ),
                    timeout=config.lobby.in_game_timeout_seconds,
                )
            except TimeoutError as error:
                raise LobbyError(
                    "LOBBY ERROR [in_game]: timed out after "
                    f"{config.lobby.in_game_timeout_seconds:.1f}s waiting for "
                    "GameEnd. Browser left open; operator: inspect the visible "
                    "game and either correct it or press Ctrl-C."
                ) from error

            if len(results) != 1:
                raise RuntimeError(
                    "live event feed ended before the next game produced GameEnd"
                )
            result = results[0]
            if result.divergence_aborted:
                print(
                    "DIVERGENCE ABORT: the browser remains open for inspection. "
                    "Press Ctrl-C to close it cleanly.",
                    file=sys.stderr,
                    flush=True,
                )
                await asyncio.Event().wait()
                return 1

            completed_games += 1
            lobby.game_ended()
            at_limit = lobby.reached_game_limit(completed_games)
            await lobby.leave_after_game(requeue=not at_limit)
            if at_limit:
                print(
                    f"arena session completed {completed_games} game(s)",
                    flush=True,
                )
                return 0
    except LobbyError as error:
        print(str(error), file=sys.stderr, flush=True)
        await asyncio.Event().wait()
        return 1
    finally:
        if modal_monitor is not None:
            modal_monitor.cancel()
            await asyncio.gather(modal_monitor, return_exceptions=True)
        if event_pump is not None:
            event_pump.cancel()
            await asyncio.gather(event_pump, return_exceptions=True)
        await session.stop()


async def _monitor_modals(
    actuator: PlaywrightActuator,
    *,
    poll_seconds: float = 0.25,
) -> None:
    """Continuously archive novel modal markup without blocking game play."""
    while True:
        await actuator.inspect_unknown_modals()
        await asyncio.sleep(poll_seconds)


async def _pump_session_events(
    source: AsyncIterable[GameEvent],
    destination: asyncio.Queue[GameEvent | None],
) -> None:
    """Keep one stateful protocol parser alive across all lobby/game cycles."""
    try:
        async for event in source:
            await destination.put(event)
    finally:
        await destination.put(None)


async def _events_for_one_game(
    source: asyncio.Queue[GameEvent | None],
) -> AsyncIterator[GameEvent]:
    """Yield queued normalized events until a one-game loop returns."""
    while True:
        event = await source.get()
        if event is None:
            return
        yield event


async def _request_resign(page: Any) -> None:
    """Open the verified resign control, leaving any unseen confirmation manual."""
    resign_tab = page.locator("metagame-buttons .game-tab").filter(has_text="Resign")
    if await resign_tab.count() == 0:
        print(
            "ERROR: resign was not requested: recorded resign tab is absent. "
            "No unrecorded confirmation selector will be guessed.",
            file=sys.stderr,
            flush=True,
        )
        return
    try:
        await resign_tab.first.click()
        print(
            "Resign request opened using the recorded UI control; confirm it "
            "manually if the client presents a dialog.",
            file=sys.stderr,
            flush=True,
        )
    except Exception as error:
        print(f"ERROR: resign was not requested: {error}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, ValueError) as error:
        print(f"arena failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
