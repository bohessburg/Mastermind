"""Regression coverage for opt-in terminal-margin distribution training."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.train.card_transformer import (
    ACTION_SPACE_SIZE,
    OBS_SIZE_V2,
    CardTokenNet,
    margin_bucket_ids,
)
from src.v2.train.human_data import HumanBatch
from src.v2.train.model import DominionNet, build_model
from src.v2.train.replay import ReplayBuffer, load_replay_state, save_replay_state
from src.v2.train.selfplay import (
    SIL_PRIORITY_EPSILON,
    compute_sil_priorities,
    refresh_sil_priorities,
)
from src.v2.train.train import train_step


def _toy_replay(seed: int, *, rows: int = 8, obs_size: int = 5, action_size: int = 3) -> ReplayBuffer:
    replay = ReplayBuffer(capacity=16, obs_size=obs_size, action_size=action_size, seed=seed)
    obs = np.arange(rows * obs_size, dtype=np.float32).reshape(rows, obs_size) / 10.0
    policy = np.zeros((rows, action_size), dtype=np.float32)
    policy[np.arange(rows), np.arange(rows) % action_size] = 1.0
    legal = np.ones((rows, action_size), dtype=np.bool_)
    replay.add(
        obs,
        policy,
        np.linspace(-1.0, 1.0, rows, dtype=np.float32),
        legal,
        np.asarray([-23, -20, -1, 0, 1, 20, 27, 4][:rows], dtype=np.int16),
    )
    return replay


def test_margin_bucket_edges_use_documented_lower_edge_bins() -> None:
    labels = margin_bucket_ids(torch.tensor([-20, -1, 0, 1, 20]), 21)
    assert labels.tolist() == [0, 9, 10, 10, 20]


def test_replay_margin_roundtrip_and_legacy_absence_warning(tmp_path: Path) -> None:
    source = _toy_replay(701)
    current = save_replay_state(source, tmp_path / "current.npz")
    restored = ReplayBuffer(capacity=16, obs_size=5, action_size=3, seed=702)
    load_replay_state(restored, current)
    np.testing.assert_array_equal(restored.margin[: len(source)], source.margin[: len(source)])

    legacy = tmp_path / "legacy_without_margin.npz"
    with np.load(current, allow_pickle=False) as archive:
        np.savez_compressed(
            legacy,
            metadata=archive["metadata"],
            obs=archive["obs"],
            policy=archive["policy"],
            value=archive["value"],
            legal_mask=archive["legal_mask"],
        )
    legacy_restored = ReplayBuffer(capacity=16, obs_size=5, action_size=3, seed=703)
    with pytest.warns(RuntimeWarning, match="auxiliary margin training needs fresh data"):
        load_replay_state(legacy_restored, legacy)
    np.testing.assert_array_equal(legacy_restored.margin[: len(source)], np.zeros(len(source), dtype=np.int16))


def test_aux_train_step_is_finite_and_aux_off_keeps_the_legacy_path_bit_exact() -> None:
    rows = 8
    aux_replay = ReplayBuffer(capacity=16, obs_size=OBS_SIZE_V2, action_size=ACTION_SPACE_SIZE, seed=704)
    obs = np.zeros((rows, OBS_SIZE_V2), dtype=np.float32)
    obs[:, 0] = 2.0
    obs[:, 1] = float(OBS_SIZE_V2)
    policy = np.zeros((rows, ACTION_SPACE_SIZE), dtype=np.float32)
    policy[:, 0] = 1.0
    legal = np.zeros((rows, ACTION_SPACE_SIZE), dtype=np.bool_)
    legal[:, 0] = True
    aux_replay.add(
        obs,
        policy,
        np.linspace(-1.0, 1.0, rows, dtype=np.float32),
        legal,
        np.asarray([-20, -3, -1, 0, 1, 4, 19, 20], dtype=np.int16),
    )
    torch.manual_seed(705)
    aux_model = CardTokenNet(
        OBS_SIZE_V2,
        ACTION_SPACE_SIZE,
        d_model=16,
        n_layers=1,
        n_heads=4,
        ffn_multiplier=1,
        aux_margin_buckets=21,
    )
    aux_result = train_step(
        aux_model,
        torch.optim.Adam(aux_model.parameters(), lr=1.0e-3),
        aux_replay,
        batch_size=4,
        device=torch.device("cpu"),
        aux_margin_weight=0.3,
    )
    assert math.isfinite(aux_result["loss"])
    assert math.isfinite(aux_result["aux_margin_loss"])

    torch.manual_seed(706)
    first = DominionNet(5, 3, hidden_sizes=[8])
    second = DominionNet(5, 3, hidden_sizes=[8])
    second.load_state_dict(first.state_dict())
    first_optimizer = torch.optim.Adam(first.parameters(), lr=1.0e-3)
    second_optimizer = torch.optim.Adam(second.parameters(), lr=1.0e-3)
    expected = train_step(first, first_optimizer, _toy_replay(707), 4, torch.device("cpu"))
    actual = train_step(
        second,
        second_optimizer,
        _toy_replay(707),
        4,
        torch.device("cpu"),
        aux_margin_weight=0.0,
    )
    assert actual == expected
    for name, tensor in first.state_dict().items():
        torch.testing.assert_close(second.state_dict()[name], tensor, rtol=0.0, atol=0.0)


def test_aux_train_step_uses_raw_human_anchor_margins() -> None:
    rows = 4
    replay = ReplayBuffer(capacity=8, obs_size=OBS_SIZE_V2, action_size=ACTION_SPACE_SIZE, seed=709)
    obs = np.zeros((rows, OBS_SIZE_V2), dtype=np.float32)
    obs[:, 0] = 2.0
    obs[:, 1] = float(OBS_SIZE_V2)
    policy = np.zeros((rows, ACTION_SPACE_SIZE), dtype=np.float32)
    policy[:, 0] = 1.0
    legal = np.zeros((rows, ACTION_SPACE_SIZE), dtype=np.bool_)
    legal[:, 0] = True
    replay.add(obs, policy, np.zeros(rows, dtype=np.float32), legal, np.arange(rows, dtype=np.int16))
    human = HumanBatch(
        obs=obs[:2],
        action=np.zeros(2, dtype=np.int64),
        legal=legal[:2],
        value=np.zeros(2, dtype=np.float32),
        margin=np.asarray([-3, 4], dtype=np.int16),
    )
    torch.manual_seed(709)
    model = CardTokenNet(
        OBS_SIZE_V2,
        ACTION_SPACE_SIZE,
        d_model=16,
        n_layers=1,
        n_heads=4,
        ffn_multiplier=1,
        aux_margin_buckets=21,
    )
    result = train_step(
        model,
        torch.optim.Adam(model.parameters(), lr=1.0e-3),
        replay,
        batch_size=2,
        device=torch.device("cpu"),
        human_batches=iter((human,)),
        anchor_weight=0.25,
        aux_margin_weight=0.3,
    )
    assert math.isfinite(result["aux_margin_loss"])
    assert math.isfinite(result["anchor_policy_loss"])


def test_legacy_transformer_config_still_builds_without_aux_parameters() -> None:
    config = {
        "arch": "card_transformer",
        "obs_version": 2,
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 4,
        "ffn_multiplier": 1,
        "dropout": 0.0,
    }
    torch.manual_seed(708)
    direct = CardTokenNet(
        OBS_SIZE_V2,
        ACTION_SPACE_SIZE,
        d_model=16,
        n_layers=1,
        n_heads=4,
        ffn_multiplier=1,
    )
    torch.manual_seed(708)
    legacy = build_model(config, OBS_SIZE_V2, ACTION_SPACE_SIZE)
    assert getattr(legacy, "aux_margin_head") is None
    assert not any(name.startswith("aux_margin_head") for name in legacy.state_dict())
    for name, tensor in direct.state_dict().items():
        torch.testing.assert_close(legacy.state_dict()[name], tensor, rtol=0.0, atol=0.0)


class _CountingValueModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[int] = []
        self.grad_enabled: list[bool] = []

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(int(obs.shape[0]))
        self.grad_enabled.append(torch.is_grad_enabled())
        values = obs[:, 0] * 0.25 - 0.5
        return torch.zeros((obs.shape[0], 3), dtype=obs.dtype, device=obs.device), values


def _sil_replay(*, seed: int, sil_weight: float) -> ReplayBuffer:
    replay = ReplayBuffer(
        capacity=16,
        obs_size=2,
        action_size=2,
        seed=seed,
        sil_weight=sil_weight,
        sil_fraction=1.0,
        sil_alpha=0.6,
    )
    obs = np.arange(12, dtype=np.float32).reshape(6, 2)
    policy = np.zeros((6, 2), dtype=np.float32)
    policy[:, 0] = 1.0
    legal = np.ones((6, 2), dtype=np.bool_)
    replay.add(obs, policy, np.linspace(-1.0, 1.0, 6, dtype=np.float32), legal)
    return replay


def test_sil_off_path_preserves_legacy_sampling_and_skips_priority_forward() -> None:
    replay = _sil_replay(seed=711, sil_weight=0.0)
    expected_rng = np.random.default_rng(711)
    expected_indices = expected_rng.integers(0, len(replay), size=9, endpoint=False)

    batch = replay.sample(9)
    np.testing.assert_array_equal(batch.obs, replay.obs[expected_indices])
    np.testing.assert_array_equal(batch.policy, replay.policy[expected_indices])
    np.testing.assert_array_equal(batch.value, replay.value[expected_indices])
    np.testing.assert_array_equal(batch.legal_mask, replay.legal_mask[expected_indices])
    np.testing.assert_array_equal(batch.margin, replay.margin[expected_indices])
    assert np.isnan(replay.last_sampled_priority_mean)
    assert np.isnan(replay.last_sampled_priority_max)

    model = _CountingValueModel()
    refresh_sil_priorities(
        replay,
        model,
        torch.device("cpu"),
        start_write=0,
        inserted_positions=len(replay),
    )
    assert model.calls == []


def test_sil_sampling_statistically_favors_high_priority_rows() -> None:
    replay = ReplayBuffer(
        capacity=2,
        obs_size=1,
        action_size=1,
        seed=712,
        sil_weight=1.0,
        sil_fraction=1.0,
        sil_alpha=0.6,
    )
    replay.add(
        np.zeros((2, 1), dtype=np.float32),
        np.ones((2, 1), dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.ones((2, 1), dtype=np.bool_),
        priority=np.asarray([1.0, 16.0], dtype=np.float32),
    )

    sampled = replay.sample(10_000)
    high_rate = float(np.mean(sampled.value == 1.0))
    expected_rate = 16.0**0.6 / (1.0 + 16.0**0.6)
    assert abs(high_rate - expected_rate) < 0.025
    assert replay.last_sampled_priority_max == 16.0


def test_sil_priority_forward_is_batched_and_matches_hand_computation() -> None:
    model = _CountingValueModel()
    model.train()
    obs = np.asarray([[0.0], [2.0], [4.0], [6.0], [8.0]], dtype=np.float32)
    targets = np.asarray([-1.0, 0.0, 0.75, 0.25, 2.0], dtype=np.float32)
    priorities = compute_sil_priorities(
        model,
        obs,
        targets,
        torch.device("cpu"),
        batch_size=2,
    )
    expected = np.maximum(SIL_PRIORITY_EPSILON, targets - (obs[:, 0] * 0.25 - 0.5))
    np.testing.assert_allclose(priorities, expected.astype(np.float32), rtol=0.0, atol=0.0)
    assert model.calls == [2, 2, 1]
    assert model.grad_enabled == [False, False, False]
    assert model.training

    replay = ReplayBuffer(8, 1, 1, seed=713, sil_weight=1.0)
    replay.add(obs, np.ones((5, 1), dtype=np.float32), targets, np.ones((5, 1), dtype=np.bool_))
    refresh_sil_priorities(
        replay,
        model,
        torch.device("cpu"),
        start_write=0,
        inserted_positions=5,
        batch_size=2,
    )
    np.testing.assert_allclose(replay.priority[:5], expected.astype(np.float32), rtol=0.0, atol=0.0)


def test_replay_priority_roundtrip_and_legacy_absence_warning(tmp_path: Path) -> None:
    source = _sil_replay(seed=714, sil_weight=1.0)
    source.priority[: len(source)] = np.linspace(0.25, 1.5, len(source), dtype=np.float32)
    current = save_replay_state(source, tmp_path / "priority-current.npz")
    restored = ReplayBuffer(16, 2, 2, seed=715, sil_weight=1.0)
    load_replay_state(restored, current)
    np.testing.assert_array_equal(restored.priority[: len(source)], source.priority[: len(source)])

    legacy = tmp_path / "priority-legacy.npz"
    with np.load(current, allow_pickle=False) as archive:
        np.savez_compressed(
            legacy,
            metadata=archive["metadata"],
            obs=archive["obs"],
            policy=archive["policy"],
            value=archive["value"],
            legal_mask=archive["legal_mask"],
            margin=archive["margin"],
        )
    legacy_restored = ReplayBuffer(16, 2, 2, seed=716, sil_weight=1.0)
    with pytest.warns(RuntimeWarning, match="no priority column"):
        load_replay_state(legacy_restored, legacy)
    np.testing.assert_array_equal(legacy_restored.priority[: len(source)], np.ones(len(source), dtype=np.float32))
