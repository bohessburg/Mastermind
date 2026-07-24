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
from .protocol.events import GameEvent, GameStart
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
    try:
        await session.start()
        assert session.run_dir is not None
        assert session.page is not None
        print(f"live archive: {session.run_dir}", flush=True)
        print(
            "Manual lobby: use the headful browser to log in (if needed) and "
            "start or queue one base-set game. Lobby automation is not enabled; "
            "the bot attaches when the game's full-state arrives.",
            flush=True,
        )
        if not (config.arena_user and config.arena_pass):
            print(
                "ARENA_USER/ARENA_PASS are not both set; log in through the "
                "persistent browser profile.",
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

        async def chat(message: str) -> None:
            await _send_chat(session.page, message)

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

        results = await run_game_loop(
            _announce_games(session.events(), chat, config.chat_announcement),
            actuator=PlaywrightActuator(session.page),
            decision_provider=provider,
            think_time_min_seconds=config.think_time_min_seconds,
            think_time_max_seconds=config.think_time_max_seconds,
            screenshot_hook=screenshot,
            chat_hook=chat,
            resign_hook=resign,
            archive_factory=archive_factory,
        )
        if any(result.divergence_aborted for result in results):
            print(
                "DIVERGENCE ABORT: the browser remains open for inspection. "
                "Press Ctrl-C to close it cleanly.",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.Event().wait()
        return 1 if any(result.divergence_aborted for result in results) else 0
    finally:
        await session.stop()


async def _announce_games(
    events: AsyncIterable[GameEvent],
    chat: Any,
    announcement: str,
) -> AsyncIterator[GameEvent]:
    """Send the configured disclosure once per game before yielding its start."""
    announced: set[int] = set()
    async for event in events:
        if isinstance(event, GameStart) and event.game_id not in announced:
            announced.add(event.game_id)
            await chat(announcement)
        yield event


async def _send_chat(page: Any, message: str) -> None:
    """Use the recorded game's ``#game-chat-input`` form when available."""
    input_box = page.locator("#game-chat-input")
    if await input_box.count() == 0:
        print(
            "ERROR: chat was not sent: #game-chat-input was absent from the "
            "live DOM. No unrecorded selector will be guessed.",
            file=sys.stderr,
            flush=True,
        )
        return
    try:
        await input_box.first.fill(message)
        # The saved DOM wraps this input in form[ng-submit="$ctrl.sendChat()"],
        # so Enter submits without inventing an unrecorded send-button selector.
        await input_box.first.press("Enter")
    except Exception as error:
        print(f"ERROR: chat was not sent: {error}", file=sys.stderr, flush=True)


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
