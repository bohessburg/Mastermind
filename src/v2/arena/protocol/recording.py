"""Load recorder JSONL and parse the game WebSocket in one call."""

from __future__ import annotations

import base64
import json
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
    sessions: list[RecordingSession] = []
    current_frames: list[DecodedFrame] | None = None
    current_url = ""
    opened_at: int | None = None

    def finish_session() -> None:
        nonlocal current_frames
        if current_frames is None:
            return
        sessions.append(
            RecordingSession(
                index=len(sessions),
                socket=socket,
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
            if record.get("sock") != socket:
                continue
            kind = record.get("kind")
            if kind == "open":
                finish_session()
                current_frames = []
                current_url = str(record.get("url", ""))
                opened_at = _optional_int(record.get("ts"))
                continue
            if kind != "binary":
                continue
            if current_frames is None:
                current_frames = []
                current_url = str(record.get("url", ""))
                opened_at = _optional_int(record.get("ts"))

            raw_data = record.get("data", "")
            if record.get("b64"):
                try:
                    raw = base64.b64decode(raw_data, validate=True)
                except (ValueError, TypeError) as error:
                    raise ProtocolError(
                        f"{path}:{line_number}: invalid base64 binary frame"
                    ) from error
            elif isinstance(raw_data, str):
                raw = raw_data.encode("latin-1")
            else:
                raise ProtocolError(
                    f"{path}:{line_number}: binary data must be a string"
                )
            frame = decode_frame(
                raw,
                str(record.get("dir", "")),
                timestamp_ms=_optional_int(record.get("ts")),
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


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None
