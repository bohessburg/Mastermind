"""Durable, inspectable records for supervised arena games."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from .protocol.events import GameEvent


@dataclass(frozen=True, kw_only=True)
class DecisionRecord:
    """One bot/client decision written to ``decisions.jsonl``."""

    frame_index: int
    question_index: int
    question_id: str
    engine_actions: tuple[int, ...]
    gesture_actions: tuple[int, ...]
    answers: tuple[int, ...]
    offered: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class ResultSummary:
    """End-of-game status independent of the site's score payload."""

    game_id: int | None
    completed: bool
    divergence_aborted: bool
    decisions: int
    reason: str


class GameArchive:
    """Write one self-contained game directory using only JSON/JSONL."""

    def __init__(
        self,
        root: Path | str,
        *,
        game_id: int | None,
        source_frames: Path | str | None = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        suffix = "unknown" if game_id is None else str(game_id)
        self.path = Path(root) / f"{timestamp}-game-{suffix}"
        self.path.mkdir(parents=True, exist_ok=False)
        self._events = (self.path / "events.jsonl").open("w", encoding="utf-8")
        self._decisions = (self.path / "decisions.jsonl").open(
            "w", encoding="utf-8"
        )
        if source_frames is not None:
            shutil.copyfile(source_frames, self.path / "frames.jsonl")

    def append_event(self, frame_index: int, event: GameEvent) -> None:
        self._events.write(
            json.dumps(
                {
                    "frame_index": frame_index,
                    "event_type": type(event).__name__,
                    "event": _jsonable(event),
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        self._events.flush()

    def append_decision(self, record: DecisionRecord) -> None:
        self._decisions.write(
            json.dumps(_jsonable(record), separators=(",", ":"), sort_keys=True)
            + "\n"
        )
        self._decisions.flush()

    def finish(
        self,
        summary: ResultSummary,
        *,
        divergence_report: Mapping[str, object] | None = None,
    ) -> None:
        (self.path / "result.json").write_text(
            json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if divergence_report is not None:
            (self.path / "divergence.json").write_text(
                json.dumps(
                    _jsonable(dict(divergence_report)),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        self.close()

    def close(self) -> None:
        self._events.close()
        self._decisions.close()

    def __enter__(self) -> GameArchive:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def copy_frame_records(
    destination: Path | str,
    records: Iterable[str],
) -> None:
    """Write already captured raw frame records without decoding them again."""
    with Path(destination).open("w", encoding="utf-8") as output:
        for record in records:
            output.write(record.rstrip("\n") + "\n")


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    return value
