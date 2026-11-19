from __future__ import annotations

import json
from pathlib import Path

import pytest

from .config import TrainConfig, load_config, validate_gate_config
from .gating import GateStats
from .test_gating import _small_gated_config
from .test_train_smoke import read_metrics
from .train import run_training, stop_marker_path, stop_rule_reason


def _reject_every_gate(monkeypatch) -> None:
    from . import train as train_module

    monkeypatch.setattr(train_module, "run_gate_match", lambda *_args, **_kwargs: GateStats(0, 2, 0))


def test_stop_rule_reason_counts_generations_since_the_best() -> None:
    cfg = TrainConfig()
    cfg.stop_after_rejections = 3
    assert stop_rule_reason(cfg, 5, 5) is None
    assert stop_rule_reason(cfg, 7, 5) is None
    reason = stop_rule_reason(cfg, 8, 5)
    assert reason is not None and "3 generations in a row" in reason and "generation 5" in reason
    assert stop_rule_reason(cfg, 8, None) is None
    cfg.stop_after_rejections = 0
    assert stop_rule_reason(cfg, 500, 0) is None


def test_stop_after_rejections_must_be_a_non_negative_integer(tmp_path: Path) -> None:
    for bad in (-1, 1.5, True, "3"):
        cfg = _small_gated_config(tmp_path, generations=1)
        cfg.stop_after_rejections = bad
        with pytest.raises(ValueError, match="stop_after_rejections"):
            validate_gate_config(cfg)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"stop_after_rejections": 10}))
    assert load_config(path).stop_after_rejections == 10
    assert TrainConfig().stop_after_rejections == 0


def test_run_stops_cleanly_after_the_configured_rejection_streak(monkeypatch, tmp_path: Path) -> None:
    cfg = _small_gated_config(tmp_path, generations=6)
    cfg.stop_after_rejections = 2
    _reject_every_gate(monkeypatch)

    result = run_training(cfg)

    assert result["stopped"] is True
    assert [row["gate_result"] for row in result["metrics"]] == ["rejected", "rejected"]
    assert len(read_metrics(Path(cfg.metrics_csv))) == 2
    marker = stop_marker_path(cfg.checkpoint_dir)
    payload = json.loads(marker.read_text())
    assert payload["generation"] == 2
    assert payload["best_generation"] == 0
    assert payload["stop_after_rejections"] == 2
    assert "2 generations in a row" in payload["reason"]
    # The last candidate is still checkpointed, so it can be inspected.
    assert (Path(cfg.checkpoint_dir) / "gen_0002.pt").exists()


def test_stopped_run_refuses_to_resume_until_the_marker_is_removed(monkeypatch, tmp_path: Path) -> None:
    cfg = _small_gated_config(tmp_path, generations=6)
    cfg.stop_after_rejections = 2
    _reject_every_gate(monkeypatch)
    run_training(cfg)
    last = Path(cfg.checkpoint_dir) / "gen_0002.pt"
    marker = stop_marker_path(cfg.checkpoint_dir)

    again = run_training(_small_gated_config(tmp_path, generations=6), resume=str(last))
    assert again["stopped"] is True
    assert again["metrics"] == []
    assert len(read_metrics(Path(cfg.metrics_csv))) == 2

    # Without the marker the resumed state still meets the rule, so the run
    # stops again before playing a game and puts the marker back.
    marker.unlink()
    resume_cfg = _small_gated_config(tmp_path, generations=6)
    resume_cfg.stop_after_rejections = 2
    again = run_training(resume_cfg, resume=str(last))
    assert again["stopped"] is True
    assert again["metrics"] == []
    assert marker.exists()

    # Raising the limit (and clearing the marker) lets it continue.
    marker.unlink()
    resume_cfg = _small_gated_config(tmp_path, generations=3)
    resume_cfg.stop_after_rejections = 5
    again = run_training(resume_cfg, resume=str(last))
    assert again["stopped"] is False
    assert [row["generation"] for row in again["metrics"]] == [3]
    assert not marker.exists()


def test_an_acceptance_resets_the_streak(monkeypatch, tmp_path: Path) -> None:
    from . import train as train_module

    cfg = _small_gated_config(tmp_path, generations=5)
    cfg.stop_after_rejections = 2
    outcomes = iter([GateStats(0, 2, 0), GateStats(2, 0, 0), GateStats(0, 2, 0), GateStats(0, 2, 0)])
    monkeypatch.setattr(train_module, "run_gate_match", lambda *_args, **_kwargs: next(outcomes))

    result = run_training(cfg)

    assert [row["gate_result"] for row in result["metrics"]] == ["rejected", "accepted", "rejected", "rejected"]
    assert json.loads(stop_marker_path(cfg.checkpoint_dir).read_text())["best_generation"] == 2


def test_stop_rule_is_off_by_default(monkeypatch, tmp_path: Path) -> None:
    cfg = _small_gated_config(tmp_path, generations=3)
    _reject_every_gate(monkeypatch)

    result = run_training(cfg)

    assert result["stopped"] is False
    assert len(result["metrics"]) == 3
    assert not stop_marker_path(cfg.checkpoint_dir).exists()
