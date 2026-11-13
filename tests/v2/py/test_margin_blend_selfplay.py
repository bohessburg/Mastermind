"""Standalone MarginBlend config and multiprocess self-play checks.

Run with:
    PYTHONPATH=build ./.venv/bin/python tests/v2/py/test_margin_blend_selfplay.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dominion_v2_py as dz

from src.v2.train.config import TrainConfig, load_config, save_config
from src.v2.train.model import build_model
from src.v2.train.observation import obs_size_for_config
from src.v2.train.replay import ReplayBuffer
from src.v2.train.selfplay import make_runner_config
from src.v2.train.workers import ParallelSelfPlayPool


def test_margin_blend_config_round_trip_and_validation() -> None:
    with tempfile.TemporaryDirectory(prefix="dominion_margin_blend_config_") as directory:
        path = Path(directory) / "margin_blend.json"
        config = TrainConfig()
        config.selfplay.value_target = "margin_blend"
        config.selfplay.margin_blend_alpha = 0.6
        save_config(config, path)

        loaded = load_config(path)
        assert loaded.selfplay.value_target == "margin_blend"
        assert loaded.selfplay.margin_blend_alpha == 0.6
        runner_config = make_runner_config(loaded.selfplay, 0xB1E0D)
        assert runner_config.value_target == dz.SelfPlayValueTarget.MarginBlend
        assert abs(runner_config.margin_blend_alpha - 0.6) < 1.0e-6

        path.write_text(json.dumps({"selfplay": {"margin_blend_alpha": 1.5}}))
        try:
            load_config(path)
        except ValueError as exc:
            assert "margin_blend_alpha" in str(exc)
        else:
            raise AssertionError("margin_blend_alpha=1.5 was accepted")

        path.write_text(json.dumps({"selfplay": {"value_target": "rank"}}))
        try:
            load_config(path)
        except ValueError as exc:
            assert "unknown value target" in str(exc)
        else:
            raise AssertionError("unknown value_target was accepted")

    # As with margin_scale, this setting remains valid and unused for
    # win/loss data so shared config templates do not need conditional fields.
    outcome_config = TrainConfig().selfplay
    outcome_config.margin_blend_alpha = 0.2
    outcome_runner = make_runner_config(outcome_config, 0xB1E0D)
    assert outcome_runner.value_target == dz.SelfPlayValueTarget.Outcome
    assert abs(outcome_runner.margin_blend_alpha - 0.2) < 1.0e-6


def test_margin_blend_worker_smoke_targets_are_tempered() -> None:
    """Collect four games through spawned workers and inspect packed replay values."""
    config = TrainConfig()
    config.seed = 0xB1E0D001
    config.parallel_workers = 2
    config.worker_device = "cpu"
    config.model.hidden_sizes = [8]
    config.selfplay.n_games = 2
    config.selfplay.games_per_generation = 4
    config.selfplay.sims_per_move = 8
    config.selfplay.max_batch = 8
    config.selfplay.dirichlet_frac = 0.0
    config.selfplay.kingdom_mode = "fixed"
    config.selfplay.fixed_kingdom = ["Village"]
    config.selfplay.max_recorded_moves = 128
    config.selfplay.max_tree_nodes = 256
    config.selfplay.auto_play_treasures = True
    config.selfplay.prune_treasure_plays = True
    config.selfplay.value_target = "margin_blend"
    config.selfplay.margin_scale = 20.0
    config.selfplay.margin_blend_alpha = 0.6

    obs_size = obs_size_for_config(config)
    model = build_model(config.model, obs_size, dz.ACTION_SPACE_SIZE)
    replay = ReplayBuffer(capacity=4096, obs_size=obs_size, action_size=dz.ACTION_SPACE_SIZE, seed=config.seed)
    pool = ParallelSelfPlayPool(config)
    try:
        result = pool.generate(model, replay, generation=1)
    finally:
        pool.close()

    values = replay.value[: replay.size]
    assert result.stats.games == 4
    assert values.size > 0
    tempered = np.logical_or(
        values == 0.0,
        np.logical_or(
            np.logical_and(values >= 0.8, values <= 1.0),
            np.logical_and(values >= -1.0, values <= -0.8),
        ),
    )
    assert bool(np.all(tempered)), values
    print(f"margin_blend_worker_smoke games={result.stats.games} positions={values.size}")


def main() -> None:
    test_margin_blend_config_round_trip_and_validation()
    test_margin_blend_worker_smoke_targets_are_tempered()
    print("test_margin_blend_selfplay: PASS")


if __name__ == "__main__":
    main()
