"""Behavior-cloning, persistent-anchor, and AWR regression tests.

Run directly with:
    PYTHONPATH=build ./.venv/bin/python src/v2/train/test_imitation.py
"""

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

from src.v2.train.config import TrainConfig, effective_anchor_weight, load_config
from src.v2.train.human_data import HumanBatch, load_human_tuples
from src.v2.train.model import DominionNet
from src.v2.train.replay import ReplayBuffer
from src.v2.train import train as train_module
from src.v2.train.train import (
    anchor_awr_weights,
    build_objects,
    load_full_checkpoint,
    run_human_pretrain,
    save_checkpoint,
    train_step,
)


def _toy_replay(seed: int) -> ReplayBuffer:
    replay = ReplayBuffer(capacity=16, obs_size=5, action_size=3, seed=seed)
    obs = np.arange(40, dtype=np.float32).reshape(8, 5) / 10.0
    policy = np.zeros((8, 3), dtype=np.float32)
    policy[np.arange(8), np.arange(8) % 3] = 1.0
    value = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    legal = np.ones((8, 3), dtype=np.bool_)
    replay.add(obs, policy, value, legal)
    return replay


def _toy_human_batch() -> HumanBatch:
    return HumanBatch(
        obs=np.arange(20, dtype=np.float32).reshape(4, 5) / 10.0,
        action=np.asarray([0, 1, 2, 0], dtype=np.int64),
        legal=np.ones((4, 3), dtype=np.bool_),
        value=np.asarray([-1.0, -0.25, 0.5, 1.0], dtype=np.float32),
    )


def test_anchor_train_step_reports_finite_human_losses() -> None:
    torch.manual_seed(501)
    model = DominionNet(5, 3, hidden_sizes=[8])
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    result = train_step(
        model,
        optimizer,
        _toy_replay(502),
        batch_size=4,
        device=torch.device("cpu"),
        human_batches=iter([_toy_human_batch()]),
        anchor_weight=0.25,
    )

    assert {"anchor_policy_loss", "anchor_value_loss"}.issubset(result)
    assert all(math.isfinite(value) for value in result.values())


def test_pretrain_accepts_per_example_policy_weights() -> None:
    torch.manual_seed(511)
    model = DominionNet(5, 3, hidden_sizes=[8])
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    batch = _toy_human_batch()
    weighted = HumanBatch(
        obs=batch.obs,
        action=batch.action,
        legal=batch.legal,
        value=batch.value,
        policy_weight=np.asarray([0.02, 0.51, 1.5, 3.0], dtype=np.float32),
    )

    history = run_human_pretrain(model, optimizer, iter([weighted]), steps=1, device=torch.device("cpu"))

    assert len(history) == 1
    assert all(math.isfinite(value) for value in history[0].values())


def test_disabled_anchor_is_the_unchanged_selfplay_train_step_and_does_not_touch_human_data() -> None:
    torch.manual_seed(503)
    first = DominionNet(5, 3, hidden_sizes=[8])
    second = DominionNet(5, 3, hidden_sizes=[8])
    second.load_state_dict(first.state_dict())
    first_optimizer = torch.optim.Adam(first.parameters(), lr=1.0e-3)
    second_optimizer = torch.optim.Adam(second.parameters(), lr=1.0e-3)
    first_replay = _toy_replay(504)
    second_replay = _toy_replay(504)

    class NeverRead:
        def __iter__(self):
            return self

        def __next__(self):
            raise AssertionError("disabled anchor read human data")

    expected = train_step(first, first_optimizer, first_replay, 4, torch.device("cpu"))
    actual = train_step(
        second,
        second_optimizer,
        second_replay,
        4,
        torch.device("cpu"),
        human_batches=NeverRead(),
        anchor_weight=0.0,
    )

    assert actual == expected
    assert set(actual) == {"loss", "policy_loss", "value_loss", "entropy"}
    for name, expected_value in first.state_dict().items():
        torch.testing.assert_close(second.state_dict()[name], expected_value, rtol=0.0, atol=0.0)
    assert first_replay.rng.bit_generator.state == second_replay.rng.bit_generator.state


def test_awr_weights_clamp_normalize_and_bypass_at_zero_beta() -> None:
    target = torch.tensor([1.0, -1.0, 0.5])
    prediction = torch.tensor([-10.0, 10.0, 0.5])
    beta = 0.1
    weights = anchor_awr_weights(target, prediction, beta)
    expected_log_weights = torch.clamp((target - prediction) / beta, max=3.0)
    expected = torch.exp(expected_log_weights) / torch.exp(expected_log_weights).mean()

    torch.testing.assert_close(weights, expected, rtol=1.0e-6, atol=1.0e-6)
    assert expected_log_weights[0].item() == pytest.approx(3.0)
    assert weights.mean().item() == pytest.approx(1.0)
    torch.testing.assert_close(anchor_awr_weights(target, None, 0.0), torch.ones_like(target))


def test_real_tuple_five_step_bc_smoke_and_checkpoint_records_imitation_config(tmp_path: Path) -> None:
    config = TrainConfig()
    config.device = "cpu"
    config.checkpoint_dir = str(tmp_path / "checkpoints")
    config.metrics_csv = str(tmp_path / "checkpoints" / "metrics.csv")
    config.model.hidden_sizes = [16]
    config.model.input_scale = 16.0
    config.selfplay.obs_version = 3
    config.imitation.human_tuples = str(ROOT / "exports" / "tuples")
    config.imitation.pretrain_steps = 5
    config.imitation.pretrain_batch_size = 16
    config.imitation.pretrain_lr = 5.0e-4
    model, optimizer, replay = build_objects(config, torch.device("cpu"))
    batches = load_human_tuples(config.imitation.human_tuples).minibatches(
        config.imitation.pretrain_batch_size,
        seed=505,
    )

    history = run_human_pretrain(
        model,
        optimizer,
        batches,
        steps=config.imitation.pretrain_steps,
        device=torch.device("cpu"),
        pretrain_lr=config.imitation.pretrain_lr,
    )
    assert len(history) == 5
    assert all(math.isfinite(step["loss"]) for step in history)

    checkpoint = save_checkpoint(config, 1, model, optimizer, replay)
    payload = load_full_checkpoint(checkpoint, "cpu")
    assert payload["config"]["imitation"] == config.to_dict()["imitation"]


def test_bc_pretrain_runs_before_a_fresh_generation_but_never_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TrainConfig()
    config.device = "cpu"
    config.generations = 1
    config.checkpoint_dir = str(tmp_path / "checkpoints")
    config.metrics_csv = str(tmp_path / "checkpoints" / "metrics.csv")
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
    config.optim.train_steps_per_generation = 0
    config.replay.capacity = 128
    config.imitation.pretrain_steps = 1
    calls: list[int] = []

    class FakeDataset:
        def minibatches(self, _batch_size: int, _seed: int):
            return iter([_toy_human_batch()])

    def fake_pretrain(*_args: object, **kwargs: object) -> list[dict[str, float]]:
        calls.append(int(kwargs["steps"]))
        return []

    monkeypatch.setattr(train_module, "load_human_dataset_for_config", lambda _config: FakeDataset())
    monkeypatch.setattr(train_module, "run_human_pretrain", fake_pretrain)

    train_module.run_training(config)
    checkpoint = Path(config.checkpoint_dir) / "gen_0001.pt"
    assert calls == [1]

    config.generations = 2
    train_module.run_training(config, resume=str(checkpoint))
    assert calls == [1]


def test_anchor_schedule_and_existing_campaign_configs_keep_imitation_off() -> None:
    assert effective_anchor_weight([[1, 0.40], [3, 0.20]], 0.10, 2) == pytest.approx(0.30)
    for filename in ("run_c18.json", "run_c19.json"):
        config = load_config(ROOT / "configs" / filename)
        assert config.imitation.pretrain_steps == 0
        assert config.imitation.anchor_weight == 0.0
        assert config.imitation.anchor_weight_schedule == []


if __name__ == "__main__":  # pragma: no cover - standalone test entry point
    raise SystemExit(pytest.main([str(Path(__file__)), "-q"]))
