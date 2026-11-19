"""Generation cut-off: training starts at a fraction of games, late games carry over."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from .config import SelfPlayConfig, validate_selfplay_max_turns_config
from .test_train_smoke import tiny_config
from .train import build_objects
from .workers import ParallelSelfPlayPool


def _pool_config(tmp_path: Path, cutoff: float):
    cfg = tiny_config(tmp_path, seed=7171, generations=2)
    cfg.parallel_workers = 2
    cfg.worker_device = "cpu"
    cfg.model.hidden_sizes = [16]
    cfg.selfplay.n_games = 4
    cfg.selfplay.games_per_generation = 8
    cfg.selfplay.sims_per_move = 2
    cfg.selfplay.max_batch = 8
    cfg.selfplay.max_recorded_moves = 64
    cfg.selfplay.max_tree_nodes = 256
    cfg.selfplay.generation_cutoff_fraction = cutoff
    cfg.replay.capacity = 1 << 16
    return cfg


def test_cutoff_returns_early_and_carries_late_games(tmp_path: Path) -> None:
    cfg = _pool_config(tmp_path, 0.5)
    model, _, replay = build_objects(cfg, torch.device("cpu"))
    pool = ParallelSelfPlayPool(cfg)
    try:
        first = pool.generate(model, replay, generation=1)
        # A worker reports done only after all of its records, so the cut-off
        # always fires before the generation is complete.
        assert 4 <= first.stats.games < 8
        assert pool._carried is not None and pool._carried.generation == 1
        late = 8 - first.stats.games

        second = pool.generate(model, replay, generation=2)
        # The second result holds generation 1's late games plus its own.
        assert second.stats.games >= late + 4
        assert pool._carried is not None and pool._carried.generation == 2

        tail = pool._finish_carried(replay)
        assert tail is not None and pool._carried is None
    finally:
        pool.close()

    # Every game of both generations arrives exactly once.
    assert first.stats.games + second.stats.games + tail.stats.games == 16
    total_positions = first.stats.positions + second.stats.positions + tail.stats.positions
    assert len(replay) == total_positions


def test_default_waits_for_every_game(tmp_path: Path) -> None:
    cfg = _pool_config(tmp_path, 1.0)
    model, _, replay = build_objects(cfg, torch.device("cpu"))
    pool = ParallelSelfPlayPool(cfg)
    try:
        result = pool.generate(model, replay, generation=1)
        assert pool._carried is None
    finally:
        pool.close()
    assert result.stats.games == 8
    assert len(replay) == result.stats.positions


@pytest.mark.parametrize("value", [0.4, 1.5, True, "0.9"])
def test_cutoff_fraction_validation(value) -> None:
    config = SelfPlayConfig()
    assert config.generation_cutoff_fraction == 1.0
    config.generation_cutoff_fraction = value
    with pytest.raises(ValueError, match="generation_cutoff_fraction"):
        validate_selfplay_max_turns_config(config)


def test_training_with_cutoff_and_scripted_segments_runs_end_to_end(tmp_path: Path) -> None:
    from .train import run_training

    cfg = _pool_config(tmp_path, 0.75)
    cfg.generations = 3
    cfg.optim.batch_size = 8
    cfg.optim.train_steps_per_generation = 1
    cfg.scripted_opponents = {"bigmoney": 0.25, "engine": 0.25}
    result = run_training(cfg)
    rows = result["metrics"]
    assert [row["generation"] for row in rows] == [1, 2, 3]
    # Generation 1 reports only its early games; later rows add the previous
    # generation's late ones, and the final generation's late games are dropped.
    assert 6 <= rows[0]["games"] < 8
    assert sum(row["games"] for row in rows) <= 24
    assert all(row["games"] >= 6 for row in rows)


def test_pause_file_holds_the_next_generation(tmp_path: Path, monkeypatch, capsys) -> None:
    import threading
    import time as real_time

    from . import train as train_module

    flag = tmp_path / "PAUSE"
    flag.write_text("")
    original_sleep = real_time.sleep
    monkeypatch.setattr(train_module.time, "sleep", lambda _s: original_sleep(0.01))
    threading.Timer(0.2, flag.unlink).start()
    started = real_time.perf_counter()
    train_module.wait_while_paused(tmp_path, 2)
    assert real_time.perf_counter() - started >= 0.2
    out = capsys.readouterr().out
    assert "gen 2 paused" in out and "gen 2 resumed" in out
    # No file, no wait.
    train_module.wait_while_paused(tmp_path, 3)
    assert "gen 3" not in capsys.readouterr().out
