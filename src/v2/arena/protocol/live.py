"""Async conversion of captured browser records into normalized events."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any

from .events import GameEvent, Reconnect, SessionStart
from .parser import ArenaParser
from .recording import decode_record_binary


RawFrameRecord = Mapping[str, Any]
RawFrameQueue = asyncio.Queue[dict[str, Any] | None]


async def records_from_queue(queue: RawFrameQueue) -> AsyncIterator[dict[str, Any]]:
    """Yield records from a session queue until its ``None`` close marker."""
    while True:
        record = await queue.get()
        if record is None:
            return
        yield record


async def events_from_queue(
    queue: RawFrameQueue,
    *,
    socket: int = 3,
) -> AsyncIterator[GameEvent]:
    """Parse the in-process records captured by :class:`ArenaSession`."""
    async for event in events_from_raw_records(records_from_queue(queue), socket=socket):
        yield event


async def events_from_raw_records(
    records: AsyncIterable[RawFrameRecord],
    *,
    socket: int = 3,
) -> AsyncIterator[GameEvent]:
    """Turn recorder-format raw records into an async game-event stream.

    This is the streaming equivalent of ``parse_recording``.  In particular,
    parser state survives a new game-channel socket and a :class:`Reconnect`
    records the final sequence of the preceding connection, exactly as the
    offline fixture loader does.
    """
    parser = ArenaParser()
    session_index = -1
    session_active = False
    previous_sequence: int | None = None

    async def start_session(record: RawFrameRecord) -> AsyncIterator[GameEvent]:
        nonlocal session_index, session_active
        session_index += 1
        session_active = True
        timestamp = _optional_int(record.get("ts"))
        yield SessionStart(
            session_index=session_index,
            socket=socket,
            url=str(record.get("url", "")),
            timestamp_ms=timestamp,
        )
        if session_index:
            yield Reconnect(
                session_index=session_index,
                previous_sequence=previous_sequence,
                timestamp_ms=timestamp,
            )

    async for record in records:
        if record.get("sock") != socket:
            continue
        kind = record.get("kind")
        if kind == "open":
            async for event in start_session(record):
                yield event
            continue
        if kind != "binary":
            continue
        if not session_active:
            # ``recording.py`` accepts a binary frame before an observed open
            # and creates a session from that record; retain that recovery
            # behavior for an attach already in progress.
            async for event in start_session(record):
                yield event
        frame = decode_record_binary(dict(record))
        if frame is None:
            continue
        for event in parser.parse_frame(frame):
            yield event
        if frame.sequence is not None:
            previous_sequence = frame.sequence


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None
