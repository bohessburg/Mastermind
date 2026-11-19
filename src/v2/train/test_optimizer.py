"""Optimizer-selection, checkpoint-resume, and AdamW smoke checks.

Run with:
    PYTHONPATH=build ./.venv/bin/python src/v2/train/test_optimizer.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.train.config import TrainConfig, load_config, save_config
from src.v2.train.train import build_objects, load_checkpoint, load_full_checkpoint, run_training, save_checkpoint


def _tiny_config(root: Path, *, generations: int = 2) -> TrainConfig:
    config = TrainConfig()
    config.device = "cpu"
    config.generations = generations
    config.checkpoint_dir = str(root / "checkpoints")
    config.metrics_csv = str(root / "checkpoints" / "metrics.csv")
    config.model.hidden_sizes = [8]
    config.selfplay.n_games = 1
    config.selfplay.games_per_generation = 1
    config.selfplay.sims_per_move = 2
    config.selfplay.max_batch = 8
    config.selfplay.dirichlet_frac = 0.0
    config.selfplay.temp_moves = 0
    config.selfplay.kingdom_mode = "fixed"
    config.selfplay.fixed_kingdom = ["Village"]
    config.selfplay.max_recorded_moves = 64
    config.selfplay.max_tree_nodes = 256
    config.selfplay.auto_play_treasures = True
    config.selfplay.prune_treasure_plays = True
    config.optim.batch_size = 8
    config.optim.train_steps_per_generation = 1
    config.replay.capacity = 256
    return config


@pytest.mark.parametrize("optimizer_kind", ("adamw", "adam"))
def test_optimizer_config_round_trip(tmp_path: Path, optimizer_kind: str) -> None:
    config = _tiny_config(tmp_path)
    config.optim.optimizer = optimizer_kind

    path = tmp_path / f"{optimizer_kind}.json"
    save_config(config, path)

    assert load_config(path).optim.optimizer == optimizer_kind


@pytest.mark.parametrize(
    ("optimizer_kind", "optimizer_class"),
    (("adamw", torch.optim.AdamW), ("adam", torch.optim.Adam)),
)
def test_build_objects_selects_configured_optimizer(
    tmp_path: Path,
    optimizer_kind: str,
    optimizer_class: type[torch.optim.Optimizer],
) -> None:
    config = _tiny_config(tmp_path)
    config.optim.optimizer = optimizer_kind

    _model, optimizer, _replay = build_objects(config, torch.device("cpu"))

    assert isinstance(optimizer, optimizer_class)


def _assert_adamw_two_step_smoke(root: Path) -> None:
    config = _tiny_config(root)
    assert config.optim.optimizer == "adamw"
    _model, optimizer, _replay = build_objects(config, torch.device("cpu"))
    assert isinstance(optimizer, torch.optim.AdamW)

    result = run_training(config)

    assert [row["generation"] for row in result["metrics"]] == [1, 2]
    payload = load_full_checkpoint(Path(config.checkpoint_dir) / "gen_0002.pt", "cpu")
    assert payload["config"]["optim"]["optimizer"] == "adamw"


def test_adamw_two_step_training_smoke(tmp_path: Path) -> None:
    _assert_adamw_two_step_smoke(tmp_path)


def _assert_checkpoint_resume_optimizer_guard(root: Path) -> None:
    source = _tiny_config(root / "source", generations=1)
    source.optim.optimizer = "adam"
    model, optimizer, replay = build_objects(source, torch.device("cpu"))
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    checkpoint = save_checkpoint(source, 1, model, optimizer, replay)

    payload = load_full_checkpoint(checkpoint, "cpu")
    assert payload["config"]["optim"]["optimizer"] == "adam"
    _config, generation, _model, restored_optimizer, _replay = load_checkpoint(checkpoint, torch.device("cpu"))
    assert generation == 1
    assert isinstance(restored_optimizer, torch.optim.Adam)
    assert restored_optimizer.state_dict()["state"]

    # Historical checkpoints have no optimizer selector but were all Adam.
    del payload["config"]["optim"]["optimizer"]
    legacy_checkpoint = checkpoint.with_name("legacy_without_optimizer_field.pt")
    torch.save(payload, legacy_checkpoint)
    _legacy_config, _generation, _legacy_model, legacy_optimizer, _legacy_replay = load_checkpoint(
        legacy_checkpoint,
        torch.device("cpu"),
    )
    assert isinstance(legacy_optimizer, torch.optim.Adam)

    conflicting_resume = _tiny_config(root / "resume")
    assert conflicting_resume.optim.optimizer == "adamw"
    with pytest.raises(ValueError, match="resume optimizer mismatch.*checkpoint config records 'adam'.*override requests 'adamw'"):
        run_training(conflicting_resume, resume=str(checkpoint))


def test_checkpoint_resume_preserves_optimizer_and_rejects_conflicting_override(tmp_path: Path) -> None:
    _assert_checkpoint_resume_optimizer_guard(tmp_path)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dominion_optimizer_") as directory:
        root = Path(directory)
        for optimizer_kind in ("adamw", "adam"):
            config = _tiny_config(root / optimizer_kind)
            config.optim.optimizer = optimizer_kind
            path = root / f"{optimizer_kind}.json"
            save_config(config, path)
            assert load_config(path).optim.optimizer == optimizer_kind
            _model, optimizer, _replay = build_objects(config, torch.device("cpu"))
            expected_class = torch.optim.AdamW if optimizer_kind == "adamw" else torch.optim.Adam
            assert isinstance(optimizer, expected_class)
        _assert_adamw_two_step_smoke(root / "adamw_smoke")
        _assert_checkpoint_resume_optimizer_guard(root / "resume_guard")
    print("test_optimizer: PASS")


if __name__ == "__main__":
    main()
