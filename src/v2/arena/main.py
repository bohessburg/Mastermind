"""Supervised entrypoint for one live dominion.games arena session."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import tempfile
import sys
from collections.abc import AsyncIterable, AsyncIterator
from datetime import datetime, timezone
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
EXIT_CLEAN = 0
EXIT_STALL = 3
EXIT_DIVERGENCE = 4
EXIT_LOBBY = 5
EXIT_UNEXPECTED = 1

LOGGER = logging.getLogger(__name__)


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
    with tempfile.TemporaryDirectory(prefix="arena-dry-run-") as temporary:
        run_dir = Path(temporary)
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
                archive_factory=lambda game_id: GameArchive(
                    run_dir,
                    game_id=game_id,
                    source_frames=REFERENCE_RECORDING,
                    ledger_path=run_dir / "record.jsonl",
                ),
                auto_deny_undo=config.undo.auto_deny,
                stall_watchdog_seconds=config.stall_watchdog_seconds,
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
            write_exit_summary(
                run_dir,
                exit_code=EXIT_DIVERGENCE,
                exit_reason="dry-run-divergence",
                game_id=results[-1].game_id if results else None,
                archive_dir=results[-1].archive_dir if results else None,
            )
            return EXIT_DIVERGENCE
        if completed != 3:
            print("dry run did not reach all three reference games", file=sys.stderr)
            write_exit_summary(
                run_dir,
                exit_code=EXIT_UNEXPECTED,
                exit_reason="dry-run-incomplete",
                game_id=results[-1].game_id if results else None,
                archive_dir=results[-1].archive_dir if results else None,
            )
            return EXIT_UNEXPECTED
        return EXIT_CLEAN


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
    current_game_id: int | None = None
    current_archive_dir: Path | None = None
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

        async def stall_screenshot(
            frame_index: int,
            _snapshot: object,
        ) -> Path | None:
            destination = session.run_dir / f"stall-{frame_index}.png"
            try:
                return await session.screenshot(destination)
            except Exception as error:
                print(
                    f"ERROR: could not capture stall screenshot: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                return None

        async def stall_dom(frame_index: int, _snapshot: object) -> Path | None:
            try:
                return await session.snapshot_dom(f"stall-{frame_index}")
            except Exception as error:
                print(
                    f"ERROR: could not capture stall DOM snapshot: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                return None

        async def resign() -> None:
            await _request_resign(session.page)

        def archive_factory(game_id: int | None) -> GameArchive:
            nonlocal current_game_id, current_archive_dir
            # GameArchive refreshes this mirror at finish, so the completed
            # game directory itself remains a replayable JSONL fixture.
            archive = GameArchive(
                session.run_dir,
                game_id=game_id,
                source_frames=session.frames_path,
                ledger_path=session.run_dir.parent / "record.jsonl",
            )
            current_game_id = game_id
            current_archive_dir = archive.path
            return archive

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
        wins = losses = ties = unknown = 0
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
                        stall_watchdog_seconds=config.stall_watchdog_seconds,
                        stall_screenshot_hook=stall_screenshot,
                        stall_dom_hook=stall_dom,
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
            abort_exit_code = _exit_code_for_game_result(result)
            if abort_exit_code == EXIT_STALL:
                print(
                    "STALL WATCHDOG ABORT: diagnostics archived; closing the "
                    "browser for a supervisor restart.",
                    file=sys.stderr,
                    flush=True,
                )
                write_exit_summary(
                    session.run_dir,
                    exit_code=EXIT_STALL,
                    exit_reason="stall-watchdog",
                    game_id=result.game_id,
                    archive_dir=result.archive_dir,
                )
                return EXIT_STALL
            if abort_exit_code == EXIT_DIVERGENCE:
                print(
                    "DIVERGENCE/ACTUATION ABORT: diagnostics archived; closing "
                    "the browser for a supervisor restart.",
                    file=sys.stderr,
                    flush=True,
                )
                write_exit_summary(
                    session.run_dir,
                    exit_code=EXIT_DIVERGENCE,
                    exit_reason="divergence-or-actuation",
                    game_id=result.game_id,
                    archive_dir=result.archive_dir,
                )
                return EXIT_DIVERGENCE

            completed_games += 1
            if result.outcome == "win":
                wins += 1
            elif result.outcome == "loss":
                losses += 1
            elif result.outcome == "tie":
                ties += 1
            else:
                unknown += 1
                print(
                    f"ERROR: game {result.game_id} has an unknown final result",
                    file=sys.stderr,
                    flush=True,
                )
            suffix = f", unknown={unknown}" if unknown else ""
            print(
                f"arena record: {wins}-{losses}-{ties} W-L-T{suffix}",
                flush=True,
            )
            lobby.game_ended()
            at_limit = lobby.reached_game_limit(completed_games)
            await lobby.leave_after_game(requeue=not at_limit)
            if at_limit:
                print(
                    f"arena session completed {completed_games} game(s)",
                    flush=True,
                )
                return EXIT_CLEAN
    except LobbyError as error:
        print(str(error), file=sys.stderr, flush=True)
        if session.run_dir is not None:
            write_exit_summary(
                session.run_dir,
                exit_code=EXIT_LOBBY,
                exit_reason="lobby-error",
                game_id=current_game_id,
                archive_dir=current_archive_dir or session.run_dir,
            )
        return EXIT_LOBBY
    except Exception as error:
        LOGGER.exception("unexpected arena crash")
        print(f"arena crashed unexpectedly: {error}", file=sys.stderr, flush=True)
        if session.run_dir is not None:
            write_exit_summary(
                session.run_dir,
                exit_code=EXIT_UNEXPECTED,
                exit_reason=f"unexpected-crash: {type(error).__name__}",
                game_id=current_game_id,
                archive_dir=current_archive_dir or session.run_dir,
            )
        return EXIT_UNEXPECTED
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
        return EXIT_CLEAN
    except Exception as error:
        print(f"arena failed: {error}", file=sys.stderr)
        run_dir = _create_failure_run_dir()
        write_exit_summary(
            run_dir,
            exit_code=EXIT_UNEXPECTED,
            exit_reason=f"unexpected-crash: {type(error).__name__}",
            game_id=None,
            archive_dir=run_dir,
        )
        return EXIT_UNEXPECTED


def write_exit_summary(
    run_dir: Path | str,
    *,
    exit_code: int,
    exit_reason: str,
    game_id: int | None,
    archive_dir: Path | str | None,
) -> Path:
    """Atomically replace the one-line supervisor diagnostic for this run."""
    destination = Path(run_dir) / "exit.json"
    payload = {
        "exit_code": exit_code,
        "exit_reason": exit_reason,
        "game_id": game_id,
        "archive_dir": None if archive_dir is None else str(archive_dir),
    }
    temporary = destination.with_suffix(".json.tmp")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except OSError as error:
        LOGGER.error("could not write arena exit summary %s: %s", destination, error)
    return destination


def _exit_code_for_game_result(result: object) -> int:
    if bool(getattr(result, "stall_aborted", False)):
        return EXIT_STALL
    if bool(getattr(result, "divergence_aborted", False)):
        return EXIT_DIVERGENCE
    return EXIT_CLEAN


def _create_failure_run_dir() -> Path:
    root = Path("exports/arena")
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = root / f"{timestamp}-startup-failure"
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{timestamp}-startup-failure-{suffix}"
        suffix += 1
    run_dir.mkdir()
    return run_dir


if __name__ == "__main__":
    raise SystemExit(main())
