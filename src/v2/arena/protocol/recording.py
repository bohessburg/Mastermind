"""Load recorder JSONL and parse the game WebSocket in one call."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .events import GameEvent, Reconnect, SessionStart, UnknownFrame
from .frames import DecodedFrame, ProtocolError, decode_frame
from .parser import ArenaParser, ParseStats


@dataclass(frozen=True)
class RecordingSession:
    index: int
    socket: int
    url: str
    opened_at_ms: int | None
    frames: tuple[DecodedFrame, ...]


@dataclass(frozen=True)
class RecordingParseResult:
    path: Path
    sessions: tuple[RecordingSession, ...]
    events: tuple[GameEvent, ...]
    stats: ParseStats

    @property
    def unknown_msg_types(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            sorted(
                {
                    (event.direction, event.msg_type)
                    for event in self.events
                    if isinstance(event, UnknownFrame)
                }
            )
        )


# Evidence: the first outbound frame of every socket in the exports/arena
# archives (2026-07-24 through 2026-11-11). The client's two server probes
# send msg 45 and close. A login sends msg 24: socket 3 after a page load, a
# later socket after an in-page reconnect. Being moved to the other game
# server between games sends msg 46 on a fresh socket with no probes (8
# archived cases, e.g. socket 4 of 20261111T084201.458865Z, where the driver
# kept listening to closed socket 3 and was resigned from a game it never saw).
GAME_CHANNEL_OPENING_MSG_TYPES = frozenset({24, 46})


class GameChannelSelector:
    """Follow the game channel as the client moves it to a new WebSocket.

    ``socket`` starts as the page-load game socket and is trusted as is. Any
    other socket becomes the game channel when, after its observed open, its
    first outbound frame is a login or a game-server move. ``select`` returns
    the records to treat as game-channel records, in order.
    """

    def __init__(self, socket: int = 3) -> None:
        self.socket = socket
        self._default_socket = socket
        self._opened: dict[object, Mapping[str, Any]] = {}

    def select(self, record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        sock = record.get("sock")
        kind = record.get("kind")
        if kind == "open" and sock == self.socket and sock != self._default_socket:
            # The hook numbers sockets from 1 again after a page reload, so a
            # reused id is a new socket that has to prove itself again.
            self.socket = self._default_socket
        if sock == self.socket:
            return (record,)
        if kind == "open":
            self._opened[sock] = record
            return ()
        if kind != "binary":
            self._opened.pop(sock, None)
            return ()
        if record.get("dir") != "out" or sock not in self._opened:
            return ()
        opened = self._opened.pop(sock)
        try:
            frame = decode_record_binary(dict(record))
        except ProtocolError:
            return ()
        if frame is None or frame.msg_type not in GAME_CHANNEL_OPENING_MSG_TYPES:
            return ()
        self.socket = sock
        self._opened.clear()
        return (opened, record)


def newest_recording(root: Path | str = "arena-recordings") -> Path | None:
    """Return the lexically newest recording JSONL, if one exists."""
    candidates = sorted(Path(root).glob("*/frames.jsonl"))
    return candidates[-1] if candidates else None


def load_recording_sessions(
    path: Path | str,
    *,
    socket: int = 3,
) -> tuple[RecordingSession, ...]:
    """Split a recorder file on game-channel open records."""
    path = Path(path)
    selector = GameChannelSelector(socket)
    sessions: list[RecordingSession] = []
    current_frames: list[DecodedFrame] | None = None
    current_url = ""
    current_socket = socket
    opened_at: int | None = None

    def finish_session() -> None:
        nonlocal current_frames
        if current_frames is None:
            return
        sessions.append(
            RecordingSession(
                index=len(sessions),
                socket=current_socket,
                url=current_url,
                opened_at_ms=opened_at,
                frames=tuple(current_frames),
            )
        )
        current_frames = None

    with path.open(encoding="utf-8") as frames_file:
        for line_number, line in enumerate(frames_file, start=1):
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as error:
                raise ProtocolError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            for record in selector.select(record):
                kind = record.get("kind")
                if kind == "open":
                    finish_session()
                    current_frames = []
                    current_url = str(record.get("url", ""))
                    current_socket = selector.socket
                    opened_at = _optional_int(record.get("ts"))
                    continue
                if kind != "binary":
                    continue
                if current_frames is None:
                    current_frames = []
                    current_url = str(record.get("url", ""))
                    current_socket = selector.socket
                    opened_at = _optional_int(record.get("ts"))

                frame = decode_record_binary(
                    dict(record),
                    source=f"{path}:{line_number}",
                )
                if frame is not None:
                    current_frames.append(frame)

    finish_session()
    return tuple(sessions)


def parse_recording(
    path: Path | str,
    *,
    socket: int = 3,
) -> RecordingParseResult:
    """Parse all game-channel sessions while preserving reconnect state."""
    path = Path(path)
    sessions = load_recording_sessions(path, socket=socket)
    parser = ArenaParser()
    events: list[GameEvent] = []
    previous_sequence: int | None = None

    for session in sessions:
        events.append(
            SessionStart(
                session_index=session.index,
                socket=session.socket,
                url=session.url,
                timestamp_ms=session.opened_at_ms,
            )
        )
        if session.index:
            events.append(
                Reconnect(
                    session_index=session.index,
                    previous_sequence=previous_sequence,
                    timestamp_ms=session.opened_at_ms,
                )
            )
        for frame in session.frames:
            events.extend(parser.parse_frame(frame))
            if frame.sequence is not None:
                previous_sequence = frame.sequence

    return RecordingParseResult(
        path=path,
        sessions=sessions,
        events=tuple(events),
        stats=parser.stats,
    )


def events_from_recording(
    path: Path | str,
    *,
    socket: int = 3,
) -> tuple[GameEvent, ...]:
    """Convenience recording-to-event-stream API."""
    return parse_recording(path, socket=socket).events


def decode_record_binary(
    record: dict[str, Any],
    *,
    source: str = "raw frame record",
) -> DecodedFrame | None:
    """Decode one recorder-format binary record through the shared envelope.

    The live browser pump deliberately calls this helper too: JSONL fixtures
    and in-memory Playwright records must have identical base64 and latin-1
    handling before they reach :func:`decode_frame`.
    """
    raw_data = record.get("data", "")
    if record.get("b64"):
        try:
            raw = base64.b64decode(raw_data, validate=True)
        except (ValueError, TypeError) as error:
            raise ProtocolError(f"{source}: invalid base64 binary frame") from error
    elif isinstance(raw_data, str):
        raw = raw_data.encode("latin-1")
    else:
        raise ProtocolError(f"{source}: binary data must be a string")
    return decode_frame(
        raw,
        str(record.get("dir", "")),
        timestamp_ms=_optional_int(record.get("ts")),
    )


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None
