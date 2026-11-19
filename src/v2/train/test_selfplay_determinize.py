"""Determinized training self-play config and runner checks.

Run with:
    PYTHONPATH=build ./.venv/bin/python src/v2/train/test_selfplay_determinize.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dominion_v2_py as dz

from src.v2.train.config import SelfPlayConfig, TrainConfig, load_config, save_config
from src.v2.train.model import build_model
from src.v2.train.observation import obs_size_for_config
from src.v2.train.selfplay import make_runner_config, play_routed_games
from src.v2.train.train import build_objects, load_checkpoint, load_full_checkpoint, save_checkpoint


def _assert_config_round_trip(root: Path) -> None:
    json_path = root / "determinize.json"
    json_path.write_text(json.dumps({"selfplay": {"determinize": "per_turn"}}), encoding="utf-8")
    config = load_config(json_path)
    assert config.selfplay.determinize == "per_turn"

    native = make_runner_config(config.selfplay, 0xD371)
    assert native.determinize == dz.SelfPlayDeterminizeMode.PerTurn
    assert dz.SelfPlayConfig().determinize == dz.SelfPlayDeterminizeMode.Off

    saved_config = root / "saved_config.json"
    save_config(config, saved_config)
    assert load_config(saved_config).selfplay.determinize == "per_turn"

    config.device = "cpu"
    config.model.hidden_sizes = [8]
    config.checkpoint_dir = str(root / "checkpoints")
    config.metrics_csv = str(root / "checkpoints" / "metrics.csv")
    model, optimizer, replay = build_objects(config, torch.device("cpu"))
    checkpoint = save_checkpoint(config, 1, model, optimizer, replay)

    payload = load_full_checkpoint(checkpoint, "cpu")
    assert payload["config"]["selfplay"]["determinize"] == "per_turn"
    _, generation, _restored_model, _restored_optimizer, _restored_replay = load_checkpoint(
        checkpoint,
        torch.device("cpu"),
    )
    assert generation == 1
    assert load_config(Path(config.checkpoint_dir) / "config.json").selfplay.determinize == "per_turn"


def test_determinize_config_round_trip_native_and_checkpoint(tmp_path: Path) -> None:
    _assert_config_round_trip(tmp_path)


def test_determinize_config_rejects_unknown_mode(tmp_path: Path) -> None:
    path = tmp_path / "invalid_determinize.json"
    path.write_text(json.dumps({"selfplay": {"determinize": "sometimes"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="determinize mode"):
        load_config(path)


def _tiny_selfplay_config(mode: str) -> SelfPlayConfig:
    return SelfPlayConfig(
        n_games=1,
        games_per_generation=1,
        sims_per_move=2,
        max_batch=8,
        dirichlet_frac=0.0,
        temp_moves=0,
        kingdom_mode="fixed",
        fixed_kingdom=["Village"],
        max_recorded_moves=128,
        max_tree_nodes=256,
        auto_play_treasures=True,
        prune_treasure_plays=True,
        # Ensure legacy configs that set reuse remain usable under honest
        # root sampling; the native runner emits one note and disables it.
        tree_reuse=mode != "off",
        determinize=mode,
    )


def _assert_tiny_run(mode: str) -> None:
    training = TrainConfig()
    training.model.hidden_sizes = [8]
    training.selfplay = _tiny_selfplay_config(mode)
    torch.manual_seed(0xD371)
    model = build_model(training.model, obs_size_for_config(training), dz.ACTION_SPACE_SIZE)

    stats, records = play_routed_games(
        (model, model),
        training.selfplay,
        seed=0xD371_0000 + ("off", "per_decision", "per_turn").index(mode),
        device=torch.device("cpu"),
        target_games=1,
    )

    assert stats.games == 1
    assert stats.positions > 0
    assert stats.leaves > 0
    assert stats.wall_time > 0.0
    assert len(records) == 1
    record = records[0]
    assert record["observations"].shape[0] > 0
    assert record["observations"].shape[0] == record["policy_targets"].shape[0]
    assert record["observations"].shape[0] == record["values"].shape[0]
    assert record["observations"].shape[0] == record["margins"].shape[0]
    assert record["margins"].dtype == np.int16
    assert np.isfinite(record["observations"]).all()
    assert np.isfinite(record["policy_targets"]).all()
    np.testing.assert_allclose(record["policy_targets"].sum(axis=1), 1.0, rtol=0.0, atol=1.0e-5)


@pytest.mark.parametrize("mode", ["off", "per_decision", "per_turn"])
def test_tiny_selfplay_completes_in_each_determinize_mode(mode: str) -> None:
    _assert_tiny_run(mode)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dominion_selfplay_determinize_") as directory:
        root = Path(directory)
        _assert_config_round_trip(root)
        for mode in ("off", "per_decision", "per_turn"):
            _assert_tiny_run(mode)
    print("test_selfplay_determinize: PASS")


if __name__ == "__main__":
    main()
