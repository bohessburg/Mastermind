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
