from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.v2.train import honest_eval
    from src.v2.train.config import TrainConfig
    from src.v2.train.train import build_objects, save_checkpoint
else:
    from . import honest_eval
    from .config import TrainConfig
    from .train import build_objects, save_checkpoint


def _tiny_checkpoint(tmp_path: Path) -> Path:
    """Create a fast, standard obs-v2 checkpoint without training a generation."""
    config = TrainConfig()
    config.device = "cpu"
    config.checkpoint_dir = str(tmp_path / "checkpoints")
    config.metrics_csv = str(tmp_path / "checkpoints" / "metrics.csv")
    config.model.hidden_sizes = [8]
    config.selfplay.obs_version = 2
    config.selfplay.auto_play_treasures = True
    config.selfplay.prune_treasure_plays = True
    model, optimizer, replay = build_objects(config, torch.device("cpu"))
    return save_checkpoint(config, 1, model, optimizer, replay)


def test_honest_eval_bigmoney_smoke_writes_complete_summary(tmp_path: Path) -> None:
    checkpoint = _tiny_checkpoint(tmp_path)
    out = tmp_path / "honest-eval.json"

    assert honest_eval.main(
        [
            "--a",
            str(checkpoint),
            "--opponent",
            "bigmoney",
            "--games",
            "2",
            "--sims",
            "8",
            "--determinizations",
            "2",
            "--workers",
            "1",
            "--device",
            "cpu",
            "--seed",
            "20260728",
            "--out",
            str(out),
        ]
    ) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["config"]["games"] == 2
    assert set(payload["overall"]) >= {"games", "wins_a", "losses_a", "ties", "errors"}
    assert payload["overall"]["wins_a"] + payload["overall"]["losses_a"] + payload["overall"]["ties"] + payload["overall"]["errors"] == 2
    assert payload["overall"]["errors"] == 0
    assert set(payload["per_seat_half"]) == {"a_seat_0", "a_seat_1"}
    assert set(payload["end_conditions"]) == {"province", "piles", "truncated"}
    assert sum(payload["end_conditions"].values()) == 2
    assert len(payload["games"]) == 2


if __name__ == "__main__":  # pragma: no cover - standalone test invocation
    raise SystemExit(pytest.main([__file__, "-q"]))
