"""Standalone EngineV3 scripted-training pipeline checks.

Run with:
    PYTHONPATH=build ./.venv/bin/python tests/v2/py/test_engine3_selfplay.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dominion_v2_py as dz

from src.v2.train.config import TrainConfig, load_config
from src.v2.train.gating import effective_scripted_fractions, plan_training_selfplay_segments
from src.v2.train.train import run_training


def test_engine3_schedule_plans_native_enginev3_segments() -> None:
    with tempfile.TemporaryDirectory(prefix="dominion_engine3_config_") as directory:
        path = Path(directory) / "engine3.json"
        path.write_text(
            json.dumps(
                {"scripted_opponent_schedule": {"engine3": [[1, 0.5]]}},
            )
        )
        config = load_config(path)

    fractions = effective_scripted_fractions(
        config.scripted_opponent_schedule,
        config.scripted_opponents,
        1,
    )
    assert fractions == {"engine3": 0.5}
    segments = plan_training_selfplay_segments(
        total_games=8,
        league_fraction=0.0,
        league_pool_size=0,
        scripted_opponents=fractions,
        seed=0xE3B0,
        sims_per_move=32,
    )
    engine3_segments = [segment for segment in segments if segment.scripted_kind == "engine3"]
    assert sum(segment.n_games for segment in engine3_segments) == 4
    assert {segment.nn_player for segment in engine3_segments} == {0, 1}

    # The string-to-binding leg is intentionally asserted here too, rather
    # than only trusting the planner's Python string metadata.
    from src.v2.train.selfplay import make_runner_config

    runner_config = make_runner_config(config.selfplay, 0xE3B0, scripted_kind="engine3")
    assert runner_config.scripted_bot == dz.SelfPlayScriptedBotKind.EngineV3


def test_config_rejects_unknown_scripted_kind() -> None:
    with tempfile.TemporaryDirectory(prefix="dominion_engine3_invalid_config_") as directory:
        path = Path(directory) / "invalid.json"
        path.write_text(
            json.dumps(
                {"scripted_opponent_schedule": {"not-a-scripted-bot": [[1, 0.5]]}},
            )
        )
        try:
            load_config(path)
        except ValueError as exc:
            assert "unknown scripted opponent" in str(exc)
        else:
            raise AssertionError("unknown scripted opponent schedule kind was accepted")


def test_engine3_worker_smoke_reports_per_kind_metrics() -> None:
    """Run eight tiny-search games through spawned training workers."""
    with tempfile.TemporaryDirectory(prefix="dominion_engine3_worker_") as directory:
        root = Path(directory)
        config = TrainConfig()
        config.seed = 0xE3B00001
        config.generations = 1
        config.device = "cpu"
        config.parallel_workers = 2
        config.worker_device = "cpu"
        config.checkpoint_dir = str(root / "checkpoints")
        config.metrics_csv = str(root / "checkpoints" / "metrics.csv")
        config.model.hidden_sizes = [8]
        config.selfplay.n_games = 4
        config.selfplay.games_per_generation = 8
        config.selfplay.sims_per_move = 32
        config.selfplay.max_batch = 16
        config.selfplay.dirichlet_frac = 0.0
        config.selfplay.kingdom_mode = "fixed"
        config.selfplay.max_recorded_moves = 128
        config.selfplay.max_tree_nodes = 512
        config.selfplay.auto_play_treasures = True
        config.selfplay.prune_treasure_plays = True
        config.optim.train_steps_per_generation = 0
        config.replay.capacity = 1024
        config.eval.eval_sentinels = []
        config.scripted_opponent_schedule = {"engine3": [[1, 0.5]]}

        result = run_training(config)

        row = result["metrics"][0]
        assert row["games"] == 8
        assert row["scripted_games"] == 4
        assert row["scripted_games_engine3"] == 4
        assert 0 <= row["scripted_wins_engine3"] <= 4
        with open(config.metrics_csv, newline="") as handle:
            csv_row = next(csv.DictReader(handle))
        assert int(csv_row["scripted_games_engine3"]) == 4
        assert 0 <= int(csv_row["scripted_wins_engine3"]) <= 4
    print(
        "engine3_worker_smoke "
        f"scripted_games_engine3={row['scripted_games_engine3']} "
        f"scripted_wins_engine3={row['scripted_wins_engine3']}"
    )


def main() -> None:
    test_engine3_schedule_plans_native_enginev3_segments()
    test_config_rejects_unknown_scripted_kind()
    test_engine3_worker_smoke_reports_per_kind_metrics()
    print("test_engine3_selfplay: PASS")


if __name__ == "__main__":
    main()
