"""Standalone persistence checks for config-selectable v2 model architectures."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

# This file is intentionally runnable as
# PYTHONPATH=build ./.venv/bin/python tests/v2/py/test_model_factory.py
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(name="root")
def root_fixture(tmp_path: Path) -> Path:
    return tmp_path

import dominion_v2_py as dz
from src.v2.train.card_transformer import CardTokenNet
from src.v2.train.config import TrainConfig
from src.v2.train.gating import SelfPlaySegment, load_best_checkpoint
from src.v2.train.inference_server import serialize_cpu_state_dict
from src.v2.train.model import DominionNet
from src.v2.train.train import build_objects, load_checkpoint, load_full_checkpoint, run_training, save_checkpoint
from src.v2.train.workers import ParallelSelfPlayPool


def _config(root: Path, arch: str) -> TrainConfig:
    config = TrainConfig()
    config.device = "cpu"
    config.checkpoint_dir = str(root / arch)
    config.metrics_csv = str(root / arch / "metrics.csv")
    config.selfplay.obs_version = 2
    config.model.arch = arch
    config.model.hidden_sizes = [8]
    config.model.input_scale = 16.0
    config.model.d_model = 16
    config.model.n_layers = 1
    config.model.n_heads = 4
    config.model.ffn_multiplier = 2
    config.model.dropout = 0.0
    return config


def test_factory_round_trip(root: Path) -> None:
    fixed_obs = torch.zeros((2, dz.OBS_SIZE_V2), dtype=torch.float32)
    full_mask = torch.ones((2, dz.ACTION_SPACE_SIZE), dtype=torch.bool)
    for arch, expected_class in (("mlp", DominionNet), ("card_transformer", CardTokenNet)):
        torch.manual_seed(20260712)
        config = _config(root, arch)
        model, optimizer, replay = build_objects(config, torch.device("cpu"))
        model.eval()
        expected_logits, expected_values = model.evaluate(fixed_obs, full_mask)

        checkpoint = save_checkpoint(config, 3, model, optimizer, replay)
        payload = load_full_checkpoint(checkpoint, "cpu")
        assert payload["config"]["model"]["arch"] == arch
        _, generation, restored, _, _ = load_checkpoint(checkpoint, torch.device("cpu"))
        assert generation == 3
        assert isinstance(restored, expected_class)
        restored.eval()
        actual_logits, actual_values = restored.evaluate(fixed_obs, full_mask)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_values, expected_values, rtol=0.0, atol=0.0)


def test_legacy_archless_checkpoint_loads_mlp(root: Path) -> None:
    config = _config(root, "mlp")
    model, optimizer, replay = build_objects(config, torch.device("cpu"))
    checkpoint = save_checkpoint(config, 1, model, optimizer, replay)
    payload = load_full_checkpoint(checkpoint, "cpu")
    del payload["config"]["model"]["arch"]
    legacy_path = checkpoint.with_name("legacy_no_arch.pt")
    torch.save(payload, legacy_path)

    _, _, restored, _, _ = load_checkpoint(legacy_path, torch.device("cpu"))
    assert isinstance(restored, DominionNet)
    logits, values = restored.evaluate(
        torch.zeros((1, dz.OBS_SIZE_V2), dtype=torch.float32),
        torch.ones((1, dz.ACTION_SPACE_SIZE), dtype=torch.bool),
    )
    assert logits.shape == (1, dz.ACTION_SPACE_SIZE)
    assert values.shape == (1,)


def test_card_transformer_obs_version_guard(root: Path) -> None:
    config = _config(root, "card_transformer")
    config.selfplay.obs_version = 1
    try:
        run_training(config)
    except ValueError as exc:
        assert "card_transformer" in str(exc)
        assert "obs_version" in str(exc)
    else:  # pragma: no cover - assertion path makes the standalone test clear
        raise AssertionError("card_transformer with obs_version=1 should be rejected")


def test_mixed_architecture_league_worker_table(root: Path) -> None:
    mlp_config = _config(root, "mlp")
    mlp_model, mlp_optimizer, mlp_replay = build_objects(mlp_config, torch.device("cpu"))
    mlp_checkpoint = save_checkpoint(mlp_config, 1, mlp_model, mlp_optimizer, mlp_replay)

    config = _config(root, "card_transformer")
    config.parallel_workers = 1
    config.worker_device = "cpu"
    config.selfplay.n_games = 1
    config.selfplay.games_per_generation = 1
    config.selfplay.sims_per_move = 1
    config.selfplay.max_batch = 4
    config.selfplay.max_recorded_moves = 64
    config.selfplay.max_tree_nodes = 256
    config.replay.capacity = 128
    active, _, replay = build_objects(config, torch.device("cpu"))
    opponent, _ = load_best_checkpoint(config, torch.device("cpu"), mlp_checkpoint)
    assert isinstance(active, CardTokenNet)
    assert isinstance(opponent, DominionNet)

    pool = ParallelSelfPlayPool(config)
    try:
        result = pool.generate(
            active,
            replay,
            generation=1,
            segments=[SelfPlaySegment(1, 0, 1)],
            model_state_payloads=[
                (serialize_cpu_state_dict(active), active._dominion_model_config),
                (serialize_cpu_state_dict(opponent), opponent._dominion_model_config),
            ],
        )
    finally:
        pool.close()
    assert result.stats.games == 1
    assert result.league_games == 1


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dominion_model_factory_") as directory:
        root = Path(directory)
        test_factory_round_trip(root)
        test_legacy_archless_checkpoint_loads_mlp(root)
        test_card_transformer_obs_version_guard(root)
        test_mixed_architecture_league_worker_table(root)
    print("test_model_factory: PASS")


if __name__ == "__main__":
    main()
