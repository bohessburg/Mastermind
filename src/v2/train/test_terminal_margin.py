from __future__ import annotations

from pathlib import Path

import pytest
import torch

import dominion_v2_py as dz

from . import evaluate
from .config import SelfPlayConfig, terminal_margin_settings
from .evaluate import make_eval_runner_config
from .gating import third_party_eval_kwargs
from .test_train_smoke import tiny_config
from .train import build_objects, save_checkpoint


def _selfplay(target: str, *, scaled: bool, scale: float = 20.0, alpha: float = 0.6) -> SelfPlayConfig:
    cfg = SelfPlayConfig()
    cfg.value_target = target
    cfg.scaled_terminal_values = scaled
    cfg.margin_scale = scale
    cfg.margin_blend_alpha = alpha
    return cfg


def test_terminal_margin_settings_mirror_the_selfplay_search() -> None:
    assert terminal_margin_settings(None) == (0.0, 0.0)
    assert terminal_margin_settings(_selfplay("margin", scaled=True)) == (20.0, 0.0)
    assert terminal_margin_settings(_selfplay("margin", scaled=True, scale=30.0)) == (30.0, 0.0)
    assert terminal_margin_settings(_selfplay("margin_blend", scaled=True, alpha=0.25)) == (20.0, 0.25)
    # Search stays on +-1 whenever self-play search did.
    assert terminal_margin_settings(_selfplay("margin", scaled=False)) == (0.0, 0.0)
    assert terminal_margin_settings(_selfplay("outcome", scaled=True)) == (0.0, 0.0)
    assert terminal_margin_settings(_selfplay("winloss", scaled=True)) == (0.0, 0.0)


def test_terminal_margin_settings_read_a_checkpoint_dict() -> None:
    stored = {"value_target": "Margin", "scaled_terminal_values": True, "margin_scale": 20.0}
    assert terminal_margin_settings(stored) == (20.0, 0.0)
    assert terminal_margin_settings({"value_target": "margin_blend", "scaled_terminal_values": True}) == (20.0, 0.6)
    # Checkpoints from before the flag existed were searched with +-1.
    assert terminal_margin_settings({"value_target": "margin"}) == (0.0, 0.0)
    assert terminal_margin_settings({}) == (0.0, 0.0)


def _runner_config(**extra: float):
    return make_eval_runner_config(
        opponent="random",
        games=1,
        sims=4,
        kingdoms="fixed",
        seed=1,
        n_games=1,
        max_batch=4,
        c_puct=1.25,
        fixed_kingdom=[],
        max_tree_nodes=256,
        **extra,
    )


def test_eval_runner_config_carries_the_terminal_scale() -> None:
    native = dz.EvalRunnerConfig()
    assert native.terminal_margin_scale == 0.0
    assert native.terminal_margin_alpha == 0.0
    native = _runner_config(terminal_margin_scale=20.0, terminal_margin_alpha=0.25)
    assert native.terminal_margin_scale == 20.0
    assert native.terminal_margin_alpha == 0.25
    dz.EvalRunner(native)


@pytest.mark.parametrize(
    ("scale", "alpha"),
    [(-1.0, 0.0), (float("nan"), 0.0), (float("inf"), 0.0), (20.0, -0.1), (20.0, 1.5), (20.0, float("nan"))],
)
def test_eval_runner_rejects_invalid_terminal_settings(scale: float, alpha: float) -> None:
    native = _runner_config()
    native.terminal_margin_scale = scale
    native.terminal_margin_alpha = alpha
    with pytest.raises(ValueError, match="terminal margin"):
        dz.EvalRunner(native)


def _margin_checkpoint(tmp_path: Path, *, target: str = "margin", scaled: bool = True) -> Path:
    cfg = tiny_config(tmp_path, generations=1)
    cfg.selfplay.value_target = target
    cfg.selfplay.scaled_terminal_values = scaled
    cfg.selfplay.margin_scale = 20.0
    model, optimizer, replay = build_objects(cfg, torch.device("cpu"))
    return save_checkpoint(cfg, 1, model, optimizer, replay)


def test_checkpoint_eval_searches_on_the_scale_the_network_was_trained_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture_eval(_model: object, **kwargs: object) -> object:
        captured.clear()
        captured.update(kwargs)
        return None

    monkeypatch.setattr(evaluate, "evaluate_model", capture_eval)

    checkpoint = _margin_checkpoint(tmp_path)
    evaluate.evaluate_checkpoint(checkpoint, device_name="cpu")
    assert captured["terminal_margin_scale"] == 20.0
    assert captured["terminal_margin_alpha"] == 0.0

    evaluate.evaluate_checkpoint(checkpoint, device_name="cpu", scaled_terminals=False)
    assert captured["terminal_margin_scale"] == 0.0
    assert captured["terminal_margin_alpha"] == 0.0

    outcome = _margin_checkpoint(tmp_path / "outcome", target="outcome")
    evaluate.evaluate_checkpoint(outcome, device_name="cpu")
    assert captured["terminal_margin_scale"] == 0.0


def test_margin_checkpoint_plays_a_real_eval_with_scaled_terminals(tmp_path: Path) -> None:
    checkpoint = _margin_checkpoint(tmp_path)
    stats = evaluate.evaluate_checkpoint(
        checkpoint,
        opponent="random",
        games=2,
        sims=4,
        kingdoms="fixed",
        seed=11,
        device_name="cpu",
        n_games=2,
        max_batch=8,
        honest=True,
    )
    assert stats.games == 2
    assert stats.games == stats.wins + stats.losses + stats.ties


def test_third_party_gate_match_uses_the_trained_terminal_scale(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, generations=1)
    cfg.gate_third_party_opponent = "engine3"
    cfg.selfplay.value_target = "margin"
    cfg.selfplay.scaled_terminal_values = True
    cfg.selfplay.margin_scale = 20.0
    kwargs = third_party_eval_kwargs(cfg)
    assert kwargs["terminal_margin_scale"] == 20.0
    assert kwargs["terminal_margin_alpha"] == 0.0
    cfg.selfplay.scaled_terminal_values = False
    assert third_party_eval_kwargs(cfg)["terminal_margin_scale"] == 0.0
