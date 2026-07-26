"""Small, crash-safe progress snapshots for long-running training jobs."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable


PROGRESS_PHASES = frozenset({"selfplay", "train", "eval", "gate", "checkpoint"})
PROGRESS_REQUIRED_KEYS = frozenset(
    {
        "generation",
        "phase",
        "phase_started_unix",
        "games_done",
        "games_total",
        "positions_done",
        "recent_games_per_hour",
        "updated_unix",
    }
)


def validate_progress_schema(payload: object) -> bool:
    """Return whether *payload* is a complete, externally consumable snapshot."""
    if not isinstance(payload, dict) or not PROGRESS_REQUIRED_KEYS.issubset(payload):
        return False
    if not isinstance(payload["generation"], int) or isinstance(payload["generation"], bool):
        return False
    if payload["phase"] not in PROGRESS_PHASES:
        return False
    for key in ("games_done", "games_total", "positions_done"):
        if not isinstance(payload[key], int) or isinstance(payload[key], bool):
            return False
    for key in ("phase_started_unix", "recent_games_per_hour", "updated_unix"):
        if not isinstance(payload[key], float):
            return False
    return "server_evals_per_sec" not in payload or isinstance(payload["server_evals_per_sec"], float)


def write_progress_file(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON snapshot without exposing partial JSON.

    The temporary file lives beside the destination so ``os.rename`` remains
    an atomic same-filesystem operation.  Only the trainer process calls this
    function; workers report through their already-existing result channel.
    """
    if not validate_progress_schema(payload):
        raise ValueError("invalid training progress schema")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class TrainingProgress:
    """Trainer-owned state with periodic atomic snapshots and heartbeats."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        write_interval_s: float = 25.0,
    ) -> None:
        if write_interval_s <= 0.0:
            raise ValueError("write_interval_s must be positive")
        self.path = Path(checkpoint_dir) / "progress.json"
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        self._write_interval_s = float(write_interval_s)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._heartbeat: Callable[[dict[str, Any]], None] | None = None
        self._last_written_monotonic: float | None = None
        self._last_console_monotonic: float | None = None
        self._state: dict[str, Any] | None = None

    def transition(
        self,
        generation: int,
        phase: str,
        *,
        games_total: int,
        games_done: int = 0,
        positions_done: int = 0,
        recent_games_per_hour: float = 0.0,
        server_mode: bool = False,
        server_evals_per_sec: float | None = None,
    ) -> None:
        if phase not in PROGRESS_PHASES:
            raise ValueError(f"unknown progress phase {phase!r}")
        now = float(self._clock())
        state: dict[str, Any] = {
            "generation": int(generation),
            "phase": phase,
            "phase_started_unix": now,
            "games_done": int(games_done),
            "games_total": int(games_total),
            "positions_done": int(positions_done),
            "recent_games_per_hour": float(recent_games_per_hour),
        }
        if server_mode:
            state["server_evals_per_sec"] = float(server_evals_per_sec or 0.0)
        with self._lock:
            self._state = state
            self._write_locked(now)
            self._last_console_monotonic = (
                self._monotonic_clock() if phase == "selfplay" else None
            )

    def update(
        self,
        *,
        games_done: int | None = None,
        positions_done: int | None = None,
        server_evals_per_sec: float | None = None,
        force: bool = False,
    ) -> None:
        now = float(self._clock())
        with self._lock:
            if self._state is None:
                return
            if games_done is not None:
                self._state["games_done"] = int(games_done)
            if positions_done is not None:
                self._state["positions_done"] = int(positions_done)
            if server_evals_per_sec is not None and "server_evals_per_sec" in self._state:
                self._state["server_evals_per_sec"] = float(server_evals_per_sec)
            elapsed = max(0.0, now - float(self._state["phase_started_unix"]))
            self._state["recent_games_per_hour"] = (
                3600.0 * float(self._state["games_done"]) / elapsed if elapsed > 0.0 else 0.0
            )
            if force or self._write_due_locked():
                self._write_locked(now)

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._state is None else dict(self._state, updated_unix=float(self._clock()))

    def start(self, heartbeat: Callable[[dict[str, Any]], None] | None = None) -> None:
        """Start the trainer-only periodic writer once the run is initialized."""
        if self._thread is not None:
            raise RuntimeError("training progress writer is already running")
        self._heartbeat = heartbeat
        self._thread = threading.Thread(target=self._run, name="training-progress", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._write_interval_s + 1.0)
            self._thread = None

    def _write_due_locked(self) -> bool:
        if self._last_written_monotonic is None:
            return True
        return self._monotonic_clock() - self._last_written_monotonic >= self._write_interval_s

    def _write_locked(self, now: float) -> None:
        assert self._state is not None
        payload = dict(self._state)
        payload["updated_unix"] = float(now)
        write_progress_file(self.path, payload)
        self._last_written_monotonic = self._monotonic_clock()

    def _run(self) -> None:
        # Five-second scheduling keeps the 25-second file cadence and the
        # independent 60-second console cadence close to their targets while
        # remaining entirely outside worker/server hot paths.
        while not self._stop.wait(min(5.0, self._write_interval_s)):
            callback: Callable[[dict[str, Any]], None] | None = None
            snapshot: dict[str, Any] | None = None
            now = float(self._clock())
            with self._lock:
                if self._state is None:
                    continue
                if self._write_due_locked():
                    self._write_locked(now)
                if self._state["phase"] == "selfplay":
                    monotonic_now = self._monotonic_clock()
                    if (
                        self._last_console_monotonic is not None
                        and monotonic_now - self._last_console_monotonic >= 60.0
                    ):
                        self._last_console_monotonic = monotonic_now
                        callback = self._heartbeat
                        snapshot = dict(self._state, updated_unix=now)
            if callback is not None and snapshot is not None:
                callback(snapshot)
