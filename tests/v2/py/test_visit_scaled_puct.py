"""Standalone visit-scaled PUCT config and self-play worker checks.

Run with:
    PYTHONPATH=build ./.venv/bin/python tests/v2/py/test_visit_scaled_puct.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dominion_v2_py as dz

from src.v2.train.config import TrainConfig, load_config, save_config
from src.v2.train.evaluate import make_eval_runner_config
from src.v2.train.model import build_model
from src.v2.train.observation import obs_size_for_config
from src.v2.train.replay import ReplayBuffer
from src.v2.train.selfplay import make_runner_config
from src.v2.train.workers import ParallelSelfPlayPool


def test_visit_scaled_config_round_trip_and_validation() -> None:
    with tempfile.TemporaryDirectory(prefix="dominion_visit_scaled_config_") as directory:
        path = Path(directory) / "visit_scaled.json"
        config = TrainConfig()
        config.selfplay.c_puct_schedule = "visit_scaled"
        config.selfplay.c_puct_init = 1.5
        config.selfplay.c_puct_base = 500.0
        save_config(config, path)

        loaded = load_config(path)
        assert loaded.selfplay.c_puct_schedule == "visit_scaled"
        assert loaded.selfplay.c_puct_init == 1.5
        assert loaded.selfplay.c_puct_base == 500.0

        selfplay_config = make_runner_config(loaded.selfplay, 0xC0FFEE)
        assert selfplay_config.c_puct_schedule == "visit_scaled"
        assert selfplay_config.c_puct_init == 1.5
        assert selfplay_config.c_puct_base == 500.0

        eval_config = make_eval_runner_config(
            opponent="random",
            games=1,
            sims=2,
            kingdoms="fixed",
            seed=0xC0FFEE,
            n_games=1,
            max_batch=4,
            c_puct=loaded.selfplay.c_puct,
            c_puct_schedule=loaded.selfplay.c_puct_schedule,
            c_puct_init=loaded.selfplay.c_puct_init,
            c_puct_base=loaded.selfplay.c_puct_base,
            fixed_kingdom=loaded.selfplay.fixed_kingdom,
            max_tree_nodes=64,
        )
        assert eval_config.c_puct_schedule == "visit_scaled"
        assert eval_config.c_puct_init == 1.5
        assert eval_config.c_puct_base == 500.0

        invalid = [
            {"c_puct_schedule": "adaptive"},
            {"c_puct_init": 0.0},
            {"c_puct_init": -1.0},
            {"c_puct_init": math.inf},
            {"c_puct_init": math.nan},
            {"c_puct_base": 0.0},
            {"c_puct_base": -1.0},
            {"c_puct_base": math.inf},
            {"c_puct_base": math.nan},
        ]
        for fields in invalid:
            path.write_text(json.dumps({"selfplay": fields}))
            try:
                load_config(path)
            except ValueError as exc:
                assert next(iter(fields)) in str(exc)
            else:
                raise AssertionError(f"invalid PUCT config was accepted: {fields!r}")

    # Native string parsing is also strict for both exposed runner configs.
    for native_config in (dz.SelfPlayConfig(), dz.EvalRunnerConfig()):
        try:
            native_config.c_puct_schedule = "adaptive"
        except ValueError as exc:
            assert "c_puct_schedule" in str(exc)
        else:
            raise AssertionError("native runner accepted an unknown c_puct_schedule")


def test_visit_scaled_four_game_worker_smoke() -> None:
    config = TrainConfig()
    config.seed = 0xC0FFEE01
    config.parallel_workers = 2
    config.worker_device = "cpu"
    config.model.hidden_sizes = [8]
    config.selfplay.n_games = 2
    config.selfplay.games_per_generation = 4
    config.selfplay.sims_per_move = 4
    config.selfplay.max_batch = 8
    config.selfplay.dirichlet_frac = 0.0
    config.selfplay.kingdom_mode = "fixed"
    config.selfplay.fixed_kingdom = ["Village"]
    config.selfplay.max_recorded_moves = 64
    config.selfplay.max_tree_nodes = 256
    config.selfplay.auto_play_treasures = True
    config.selfplay.prune_treasure_plays = True
    config.selfplay.c_puct_schedule = "visit_scaled"
    config.selfplay.c_puct_init = 1.25
    config.selfplay.c_puct_base = 500.0

    obs_size = obs_size_for_config(config)
    model = build_model(config.model, obs_size, dz.ACTION_SPACE_SIZE)
    replay = ReplayBuffer(
        capacity=4096,
        obs_size=obs_size,
        action_size=dz.ACTION_SPACE_SIZE,
        seed=config.seed,
    )
    pool = ParallelSelfPlayPool(config)
    try:
        result = pool.generate(model, replay, generation=1)
    finally:
        pool.close()

    assert result.stats.games == 4
    assert replay.size > 0
    print(f"visit_scaled_worker_smoke games={result.stats.games} positions={replay.size}")


def main() -> None:
    test_visit_scaled_config_round_trip_and_validation()
    test_visit_scaled_four_game_worker_smoke()
    print("test_visit_scaled_puct: PASS")


if __name__ == "__main__":
    main()
