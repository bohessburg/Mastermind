"""The paired gate can evaluate through the Apple host backends (MLX)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import dominion_v2_py as dz

from .apple_backend import evaluate_host_rows
from .config import validate_gate_config
from .gating import gate_host_backends, run_paired_gate
from .selfplay import route_leaf_evaluations
from .test_gating import _paired_gate_config
from .train import build_objects


class _FakeBackend:
    def __init__(self, value: float) -> None:
        self.value = value
        self.shapes: list[tuple[int, int]] = []

    def padded_rows(self, count: int, capacity: int) -> int:
        return min(capacity, ((count + 63) // 64) * 64)

    def evaluate(self, obs: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.shapes.append((int(obs.shape[0]), int(masks.shape[0])))
        rows = obs.shape[0]
        values = np.full((rows,), self.value, dtype=np.float32)
        # Real rows carry their row sum so the test can check none were shuffled.
        values[: rows] += obs.sum(axis=1)
        return np.zeros((rows, masks.shape[1]), dtype=np.float32), values


def test_host_rows_are_padded_to_the_bucket_and_trimmed_back() -> None:
    backend = _FakeBackend(0.0)
    obs = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    masks = np.ones((5, 4), dtype=bool)
    logits, values = evaluate_host_rows(backend, obs, masks)
    assert backend.shapes == [(64, 64)]
    assert logits.shape == (5, 4) and values.shape == (5,)
    np.testing.assert_allclose(values, obs.sum(axis=1))


def test_routing_uses_the_host_backend_only_for_its_model(tmp_path: Path) -> None:
    cfg = _paired_gate_config(tmp_path)
    seat0, _, _ = build_objects(cfg, torch.device("cpu"))
    seat1, _, _ = build_objects(cfg, torch.device("cpu"))
    obs = np.zeros((6, dz.OBS_SIZE), dtype=np.float32)
    masks = np.ones((6, dz.ACTION_SPACE_SIZE), dtype=np.bool_)
    players = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.uint8)
    reference_logits, reference_values = route_leaf_evaluations(
        (seat0, seat1), obs, masks, players, torch.device("cpu")
    )
    backend = _FakeBackend(7.0)
    logits, values = route_leaf_evaluations(
        (seat0, seat1), obs, masks, players, torch.device("cpu"), host_backends={id(seat1): backend}
    )
    seat1_rows = players == 1
    np.testing.assert_allclose(values[seat1_rows], 7.0)
    np.testing.assert_array_equal(values[~seat1_rows], reference_values[~seat1_rows])
    np.testing.assert_array_equal(logits[~seat1_rows], reference_logits[~seat1_rows])
    assert backend.shapes == [(64, 64)]


def test_gate_backend_defaults_to_pytorch_and_validates(tmp_path: Path) -> None:
    cfg = _paired_gate_config(tmp_path)
    model, _, _ = build_objects(cfg, torch.device("cpu"))
    assert gate_host_backends((model, model), cfg, torch.device("cpu")) is None
    cfg.gate_backend = "mlx"
    with pytest.raises(ValueError, match="mps"):
        gate_host_backends((model, model), cfg, torch.device("cpu"))
    cfg.gate_backend = "tpu"
    with pytest.raises(ValueError, match="gate_backend"):
        validate_gate_config(cfg)


def _use_card_transformer(cfg) -> None:
    cfg.model.arch = "card_transformer"
    cfg.model.obs_version = 3
    cfg.selfplay.obs_version = 3
    cfg.model.d_model, cfg.model.n_layers, cfg.model.n_heads = 48, 2, 4


def _mps_mlx_or_skip() -> torch.device:
    pytest.importorskip("mlx.core")
    if not torch.backends.mps.is_available():
        pytest.skip("needs an Apple GPU")
    return torch.device("mps")


def test_mlx_gate_routing_matches_pytorch(tmp_path: Path) -> None:
    device = _mps_mlx_or_skip()
    cfg = _paired_gate_config(tmp_path)
    _use_card_transformer(cfg)
    seat0, _, _ = build_objects(cfg, device)
    seat1, _, _ = build_objects(cfg, device)
    rng = np.random.default_rng(5)
    obs_rows, mask_rows = [], []
    game = dz.new_game(dz.Setup(players=2), 77)
    while len(obs_rows) < 40:
        legal = np.asarray(game.legal_mask(), dtype=bool)
        actions = np.flatnonzero(legal)
        if actions.size == 0:
            break
        player = int(game.current_decision()["player"])
        obs_rows.append(np.asarray(game.encode(player, 3), dtype=np.float32).copy())
        mask_rows.append(legal.copy())
        game.step(int(rng.choice(actions)))
    obs, masks = np.stack(obs_rows), np.stack(mask_rows)
    players = (np.arange(obs.shape[0]) % 2).astype(np.uint8)
    reference = route_leaf_evaluations((seat0, seat1), obs, masks, players, device)
    cfg.gate_backend = "mlx"
    backends = gate_host_backends((seat0, seat1), cfg, device)
    served = route_leaf_evaluations((seat0, seat1), obs, masks, players, device, host_backends=backends)
    legal = masks
    np.testing.assert_allclose(served[1], reference[1], atol=1e-4)
    np.testing.assert_allclose(served[0][legal], reference[0][legal], atol=1e-4)


def test_mlx_fp16_paired_gate_runs(tmp_path: Path) -> None:
    device = _mps_mlx_or_skip()
    cfg = _paired_gate_config(tmp_path)
    _use_card_transformer(cfg)
    cfg.gate_backend = "mlx"
    cfg.gate_fp16 = True
    candidate, _, _ = build_objects(cfg, device)
    best, _, _ = build_objects(cfg, device)
    best.load_state_dict(candidate.state_dict())
    outcome = run_paired_gate(candidate, best, cfg, generation=1, device=device)
    assert outcome.stats.wins + outcome.stats.losses + outcome.stats.ties == cfg.gate_games
