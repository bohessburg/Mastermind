from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.v2.arena.actuate.clicks import MockActuator
from src.v2.arena.fsm.game import RecordedDecisionProvider, run_game_loop
from src.v2.arena.protocol.events import GameEvent
from src.v2.arena.protocol.live import RawFrameQueue, events_from_queue
from src.v2.arena.protocol.recording import parse_recording


RECORDING = Path(
    "arena-recordings/20260724T142103.096991Z/frames.jsonl"
)


async def _enqueue_recording(path: Path, queue: RawFrameQueue) -> None:
    try:
        with path.open(encoding="utf-8") as frames_file:
            for line in frames_file:
                await queue.put(json.loads(line))
    finally:
        await queue.put(None)


async def _collect_live_events(path: Path) -> tuple[GameEvent, ...]:
    queue: RawFrameQueue = asyncio.Queue()
    producer = asyncio.create_task(_enqueue_recording(path, queue))
    try:
        return tuple([event async for event in events_from_queue(queue)])
    finally:
        await producer


async def _async_events(events: tuple[GameEvent, ...]):
    for event in events:
        yield event


def test_async_pump_matches_recording_event_stream_exactly() -> None:
    assert RECORDING.is_file(), f"missing required arena fixture: {RECORDING}"

    expected = parse_recording(RECORDING).events
    observed = asyncio.run(_collect_live_events(RECORDING))

    assert observed == expected


def test_async_game_loop_replays_all_three_games() -> None:
    assert RECORDING.is_file(), f"missing required arena fixture: {RECORDING}"
    events = parse_recording(RECORDING).events
    actuator = MockActuator(replay=True)

    results = asyncio.run(
        run_game_loop(
            _async_events(events),
            actuator=actuator,
            decision_provider=RecordedDecisionProvider(events),
        )
    )

    assert len(results) == 3
    assert all(result.completed for result in results)
    assert not any(result.divergence_aborted for result in results)
    assert [result.game_id for result in results] == [
        181347875,
        181347888,
        181348150,
    ]
    assert sum(result.decisions for result in results) == 929
    assert sum(result.validations for result in results) >= 929
    assert sum(result.rigged_steps for result in results) == 259
    assert len(actuator.gestures) == 929
    assert not actuator.stopped
