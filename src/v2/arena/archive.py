"""Durable, inspectable records for supervised arena games."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .protocol.events import GameEvent


LOGGER = logging.getLogger(__name__)


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
    """End-of-game status, including the best-effort decoded standings."""

    game_id: int | None
    completed: bool
    divergence_aborted: bool
    decisions: int
    reason: str
    stall_aborted: bool = False
    our_seat: int | None = None
    opponent: str | None = None
    outcome: str = "unknown"
    scores: tuple[int, ...] = ()
    placings: tuple[int, ...] = ()
    winner_seat: int | None = None
    tie: bool = False


class GameArchive:
    """Write one self-contained game directory using only JSON/JSONL."""

    def __init__(
        self,
        root: Path | str,
        *,
        game_id: int | None,
        source_frames: Path | str | None = None,
        ledger_path: Path | str | None = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        suffix = "unknown" if game_id is None else str(game_id)
        self.path = Path(root) / f"{timestamp}-game-{suffix}"
        self.path.mkdir(parents=True, exist_ok=False)
        self._events = (self.path / "events.jsonl").open("w", encoding="utf-8")
        self._decisions = (self.path / "decisions.jsonl").open(
            "w", encoding="utf-8"
        )
        self._source_frames = Path(source_frames) if source_frames is not None else None
        self._ledger_path = (
            Path(ledger_path) if ledger_path is not None else None
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
        stall_report: Mapping[str, object] | None = None,
    ) -> None:
        self._snapshot_source_frames()
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
        if stall_report is not None:
            (self.path / "stall.json").write_text(
                json.dumps(
                    _jsonable(dict(stall_report)),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        if summary.completed:
            self._append_ledger(summary)
        self.close()

    def close(self) -> None:
        self._events.close()
        self._decisions.close()

    def _snapshot_source_frames(self) -> None:
        """Refresh a live session's frame mirror before finalizing a game."""
        if self._source_frames is None or not self._source_frames.is_file():
            return
        shutil.copyfile(self._source_frames, self.path / "frames.jsonl")

    def _append_ledger(self, summary: ResultSummary) -> None:
        """Best-effort append; bookkeeping must never abort a played game."""
        if self._ledger_path is None:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "game_id": summary.game_id,
            "opponent": summary.opponent,
            "our_seat": summary.our_seat,
            "result": summary.outcome,
            "scores": summary.scores,
            "run_dir": self.path.parent,
        }
        try:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self._ledger_path.open("a", encoding="utf-8") as ledger:
                ledger.write(
                    json.dumps(
                        _jsonable(record),
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
        except Exception as error:
            LOGGER.error(
                "could not append arena win/loss ledger %s for game %s: %s",
                self._ledger_path,
                summary.game_id,
                error,
            )

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


def serialize_frame_record(record: Mapping[str, Any]) -> str:
    """Return one recorder-compatible raw frame JSONL line."""
    return json.dumps(dict(record), separators=(",", ":")) + "\n"


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
