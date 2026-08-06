"""KataGo-style forced-playout configuration and self-play smoke coverage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import dominion_v2_py as dz

from .config import SelfPlayConfig, TrainConfig, load_config, save_config
from .model import build_model
from .observation import obs_size_for_config
from .selfplay import make_runner_config, play_routed_games
from .train import build_objects, load_full_checkpoint, save_checkpoint


def test_forced_playouts_config_round_trip_native_and_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "forced_playouts.json"
    source.write_text(
        json.dumps({"selfplay": {"forced_playouts": True, "forced_playouts_k": 3.5}}),
        encoding="utf-8",
    )
    config = load_config(source)
    assert config.selfplay.forced_playouts is True
    assert config.selfplay.forced_playouts_k == pytest.approx(3.5)

    native = make_runner_config(config.selfplay, 0xF0CE)
    assert native.forced_playouts is True
    assert native.forced_playouts_k == pytest.approx(3.5)
    assert dz.SelfPlayConfig().forced_playouts is False
    assert dz.SelfPlayConfig().forced_playouts_k == pytest.approx(2.0)

    saved = tmp_path / "saved.json"
    save_config(config, saved)
    loaded = load_config(saved)
    assert loaded.selfplay.forced_playouts is True
    assert loaded.selfplay.forced_playouts_k == pytest.approx(3.5)

    config.device = "cpu"
    config.model.hidden_sizes = [8]
    config.checkpoint_dir = str(tmp_path / "checkpoints")
    config.metrics_csv = str(tmp_path / "checkpoints" / "metrics.csv")
    model, optimizer, replay = build_objects(config, torch.device("cpu"))
    checkpoint = save_checkpoint(config, 1, model, optimizer, replay)
    payload = load_full_checkpoint(checkpoint, "cpu")
    assert payload["config"]["selfplay"]["forced_playouts"] is True
    assert payload["config"]["selfplay"]["forced_playouts_k"] == pytest.approx(3.5)
    restored = load_config(Path(config.checkpoint_dir) / "config.json")
    assert restored.selfplay.forced_playouts is True
    assert restored.selfplay.forced_playouts_k == pytest.approx(3.5)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), True, "2.0"])
def test_forced_playouts_k_must_be_positive(value: object, tmp_path: Path) -> None:
    path = tmp_path / "invalid_forced_playouts.json"
    path.write_text(
        json.dumps({"selfplay": {"forced_playouts_k": value}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forced_playouts_k"):
        load_config(path)


def test_forced_playouts_selfplay_records_normalized_policy_targets() -> None:
    selfplay = SelfPlayConfig(
        n_games=1,
        games_per_generation=1,
        sims_per_move=12,
        max_batch=16,
        dirichlet_frac=0.25,
        forced_playouts=True,
        forced_playouts_k=2.0,
        temp_moves=0,
        kingdom_mode="fixed",
        fixed_kingdom=["Village"],
        max_recorded_moves=128,
        max_tree_nodes=512,
        auto_play_treasures=True,
        prune_treasure_plays=True,
    )
    training = TrainConfig()
    training.model.hidden_sizes = [8]
    training.selfplay = selfplay
    torch.manual_seed(0xF0CE)
    model = build_model(training.model, obs_size_for_config(training), dz.ACTION_SPACE_SIZE)

    stats, records = play_routed_games(
        (model, model),
        selfplay,
        seed=0xF0CE0001,
        device=torch.device("cpu"),
        target_games=1,
    )

    assert stats.games == 1
    assert stats.positions > 0
    assert len(records) == 1
    policy = records[0]["policy_targets"]
    assert policy.shape[0] > 0
    assert np.isfinite(policy).all()
    assert np.all(policy >= 0.0)
    np.testing.assert_allclose(policy.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-5)
