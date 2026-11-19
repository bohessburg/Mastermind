from __future__ import annotations

import json
import threading
from pathlib import Path

from . import progress


class _FakeClock:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


def test_progress_writer_is_atomic_and_keeps_a_valid_schema_for_readers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clock = _FakeClock(1_700_000_000.0)
    tracker = progress.TrainingProgress(
        tmp_path,
        clock=clock,
        monotonic_clock=clock,
        write_interval_s=25.0,
    )
    renames: list[tuple[Path, Path]] = []
    real_rename = progress.os.rename

    def checked_rename(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        renames.append((source_path, destination_path))
        assert source_path.parent == destination_path.parent
        assert source_path != destination_path
        real_rename(source_path, destination_path)

    monkeypatch.setattr(progress.os, "rename", checked_rename)
    tracker.transition(12, "selfplay", games_total=1024, server_mode=True)
    path = tmp_path / "progress.json"
    reader_errors: list[BaseException] = []
    finished = threading.Event()

    def reader() -> None:
        while not finished.is_set():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                assert progress.validate_progress_schema(payload)
            except BaseException as exc:  # pragma: no cover - asserted below
                reader_errors.append(exc)
                return

    thread = threading.Thread(target=reader)
    thread.start()
    for games in range(1, 101):
        clock.advance(0.25)
        tracker.update(
            games_done=games,
            positions_done=games * 17,
            server_evals_per_sec=61_000.0,
            force=True,
        )
    finished.set()
    thread.join(timeout=2.0)

    assert not reader_errors
    assert renames
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "generation": 12,
        "phase": "selfplay",
        "phase_started_unix": 1_700_000_000.0,
        "games_done": 100,
        "games_total": 1024,
        "positions_done": 1700,
        "recent_games_per_hour": 14_400.0,
        "server_evals_per_sec": 61_000.0,
        "updated_unix": 1_700_000_025.0,
    }


def test_progress_phase_transitions_use_the_injectable_clock(tmp_path: Path) -> None:
    clock = _FakeClock(100.0)
    tracker = progress.TrainingProgress(tmp_path, clock=clock, monotonic_clock=clock)
    tracker.transition(4, "selfplay", games_total=16)
    clock.advance(3.0)
    tracker.update(games_done=5, positions_done=80, force=True)
    clock.advance(2.0)
    tracker.transition(
        4,
        "train",
        games_total=16,
        games_done=5,
        positions_done=80,
        recent_games_per_hour=6000.0,
    )

    payload = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress.validate_progress_schema(payload)
    assert payload["phase"] == "train"
    assert payload["phase_started_unix"] == 105.0
    assert payload["updated_unix"] == 105.0
    assert payload["games_done"] == 5
    assert payload["positions_done"] == 80
    assert payload["recent_games_per_hour"] == 6000.0
    assert "server_evals_per_sec" not in payload
