from __future__ import annotations

import csv
import hashlib
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

import dominion_v2_py as dz

from .config import TrainConfig, load_config
from .observation import obs_version_for_checkpoint
from .selfplay import play_routed_games
from .train import (
    build_objects,
    load_checkpoint,
    load_full_checkpoint,
    resolve_resume_path,
    run_training,
    save_checkpoint,
)
from .gating import SelfPlaySegment
from .workers import ParallelSelfPlayPool, _pack_records, game_quotas


def tiny_config(tmp_path: Path, seed: int = 20260709, generations: int = 2) -> TrainConfig:
    cfg = TrainConfig()
    cfg.seed = seed
    cfg.generations = generations
    cfg.device = "cpu"
    cfg.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.metrics_csv = str(tmp_path / "ckpt" / "metrics.csv")
    cfg.model.hidden_sizes = [64]
    cfg.selfplay.n_games = 8
    cfg.selfplay.sims_per_move = 32
    cfg.selfplay.games_per_generation = 8
    cfg.selfplay.max_batch = 64
    cfg.selfplay.dirichlet_frac = 0.0
    cfg.selfplay.temp_moves = 4
    cfg.selfplay.kingdom_mode = "fixed"
    cfg.selfplay.max_recorded_moves = 256
    cfg.selfplay.max_tree_nodes = 1024
    cfg.optim.batch_size = 32
    cfg.optim.train_steps_per_generation = 2
    cfg.replay.capacity = 4096
    return cfg


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def state_tensors(path: Path) -> dict[str, torch.Tensor]:
    payload = load_full_checkpoint(path, "cpu")
    return {key: value.detach().cpu().clone() for key, value in payload["model"].items()}


def test_tiny_training_smoke_checkpoint_and_metrics(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path)
    result = run_training(cfg)

    assert len(result["metrics"]) == 2
    metrics = read_metrics(Path(cfg.metrics_csv))
    assert len(metrics) == 2
    assert (Path(cfg.checkpoint_dir) / "gen_0001.pt").exists()
    assert (Path(cfg.checkpoint_dir) / "gen_0002.pt").exists()

    for row in metrics:
        assert int(row["games"]) >= cfg.selfplay.games_per_generation
        assert int(row["positions"]) > 0
        assert math.isfinite(float(row["policy_loss"]))
        assert math.isfinite(float(row["value_loss"]))
        assert math.isfinite(float(row["entropy"]))


def test_tiny_training_v2_observation_smoke_records_v2_width(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, seed=20260711, generations=1)
    cfg.model.hidden_sizes = [8]
    cfg.selfplay.obs_version = 2
    cfg.selfplay.n_games = 2
    cfg.selfplay.games_per_generation = 2
    cfg.selfplay.sims_per_move = 4
    cfg.selfplay.max_batch = 8
    cfg.selfplay.max_recorded_moves = 128
    cfg.selfplay.max_tree_nodes = 256
    cfg.optim.batch_size = 8
    cfg.optim.train_steps_per_generation = 1
    cfg.replay.capacity = 512

    result = run_training(cfg)
    assert len(result["metrics"]) == 1
    checkpoint = Path(cfg.checkpoint_dir) / "gen_0001.pt"
    payload = load_full_checkpoint(checkpoint, "cpu")
    assert obs_version_for_checkpoint(payload) == 2
    assert payload["config"]["selfplay"]["obs_version"] == 2
    assert payload["model"]["trunk.0.weight"].shape[1] == dz.OBS_SIZE_V2

    with np.load(Path(cfg.checkpoint_dir) / "replay_state.npz", allow_pickle=False) as archive:
        assert archive["obs"].shape[0] > 0
        assert archive["obs"].shape[1] == dz.OBS_SIZE_V2


def test_parallel_v2_records_preserve_observation_rows_in_parent_replay(tmp_path: Path) -> None:
    """A spawned worker must not reinterpret v2 records with the v1 stride."""
    cfg = tiny_config(tmp_path, seed=20260721, generations=1)
    cfg.parallel_workers = 2
    cfg.worker_device = "cpu"
    cfg.model.hidden_sizes = [8]
    cfg.selfplay.obs_version = 2
    cfg.selfplay.n_games = 2
    cfg.selfplay.games_per_generation = 4
    cfg.selfplay.sims_per_move = 2
    cfg.selfplay.max_batch = 4
    cfg.selfplay.max_recorded_moves = 64
    cfg.selfplay.max_tree_nodes = 256
    cfg.replay.capacity = 512
    model, _, replay = build_objects(cfg, torch.device("cpu"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    expected_rows: Counter[bytes] = Counter()
    for worker_index in range(cfg.parallel_workers):
        _, records = play_routed_games(
            (model, model),
            cfg.selfplay,
            seed=(cfg.seed + worker_index) ^ (1 * 0x9E37),
            device=torch.device("cpu"),
            target_games=2,
            same_model_fast_path=True,
        )
        for record in records:
            for obs, policy, value in zip(
                record["observations"],
                record["policy_targets"],
                record["values"],
                strict=True,
            ):
                expected_rows[hashlib.sha256(obs.tobytes() + policy.tobytes() + value.tobytes()).digest()] += 1

    pool = ParallelSelfPlayPool(cfg)
    try:
        result = pool.generate(
            model,
            replay,
            generation=1,
            # Exercise the segmented path used by the campaign's configured
            # scripted-opponent schedule even while its effective fraction is
            # still zero.
            segments=[SelfPlaySegment(4, 0, 0)],
        )
    finally:
        pool.close()

    assert result.stats.games == 4
    assert len(replay) > 0
    stored = replay.obs[: len(replay)]
    assert stored.shape[1] == dz.obs_size_for(2)
    np.testing.assert_array_equal(stored[:, 0], np.full(len(stored), 2.0, dtype=np.float32))
    np.testing.assert_array_equal(
        stored[:, 1],
        np.full(len(stored), float(dz.obs_size_for(2)), dtype=np.float32),
    )
    actual_rows = Counter(
        hashlib.sha256(obs.tobytes() + policy.tobytes() + value.tobytes()).digest()
        for obs, policy, value in zip(
            stored,
            replay.policy[: len(replay)],
            replay.value[: len(replay)],
            strict=True,
        )
    )
    assert actual_rows == expected_rows


def test_parallel_record_packing_rejects_a_v1_row_for_a_v2_generation() -> None:
    """The parent transport must not silently accept a legacy observation stride."""
    record = {
        "observations": np.zeros((1, dz.OBS_SIZE_V1), dtype=np.float32),
        "policy_targets": np.zeros((1, dz.ACTION_SPACE_SIZE), dtype=np.float32),
        "values": np.zeros((1,), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="observation width"):
        _pack_records([record], dz.obs_size_for(2))


def test_split_checkpoint_roundtrip(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, seed=1234, generations=1)
    model, optimizer, replay = build_objects(cfg, torch.device("cpu"))
    count = 5
    obs = np.arange(count * replay.obs_size, dtype=np.float32).reshape(count, replay.obs_size)
    policy = np.zeros((count, replay.action_size), dtype=np.float32)
    policy[:, 0] = 1.0
    value = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    replay.add(obs, policy, value, policy > 0.0)
    expected = replay.state_dict()

    checkpoint = save_checkpoint(cfg, 1, model, optimizer, replay)
    replay_file = Path(cfg.checkpoint_dir) / "replay_state.npz"
    assert checkpoint.exists()
    assert replay_file.exists()
    assert not list(Path(cfg.checkpoint_dir).glob(".*.tmp"))

    payload = load_full_checkpoint(checkpoint, "cpu")
    assert "replay" not in payload
    assert {"generation", "config", "model", "optimizer"}.issubset(payload)

    _, generation, _, _, restored = load_checkpoint(checkpoint, torch.device("cpu"))
    actual = restored.state_dict()
    assert generation == 1
    for key in ("capacity", "obs_size", "action_size", "write", "size", "rng_state"):
        assert actual[key] == expected[key]
    for key in ("obs", "policy", "value", "legal_mask"):
        np.testing.assert_array_equal(actual[key], expected[key])


def test_resume_without_replay_file_warns_and_uses_empty_buffer(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, seed=1234, generations=1)
    run_training(cfg)
    first = Path(cfg.checkpoint_dir) / "gen_0001.pt"
    replay_file = Path(cfg.checkpoint_dir) / "replay_state.npz"
    replay_file.unlink()

    with pytest.warns(RuntimeWarning, match="replay buffer was not restored"):
        _, generation, _, _, replay = load_checkpoint(first, torch.device("cpu"))
    assert generation == 1
    assert len(replay) == 0

    resume_cfg = tiny_config(tmp_path / "resume", seed=1234, generations=2)
    with pytest.warns(RuntimeWarning, match="replay buffer was not restored"):
        run_training(resume_cfg, resume=str(first))
    assert (Path(resume_cfg.checkpoint_dir) / "gen_0002.pt").exists()


def test_resume_latest_resolves_newest_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "gen_0001.pt").write_bytes(b"old")
    (root / "gen_0010.pt").write_bytes(b"new")
    assert resolve_resume_path("latest", root) == str(root / "gen_0010.pt")


def test_config_ignores_comment_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"_comment": "ignored", "optim": {"_comment_lr": "ignored", "lr": 0.0002}}')
    cfg = load_config(path)
    assert cfg.optim.lr == 0.0002


def test_cpu_first_generation_is_deterministic(tmp_path: Path) -> None:
    cfg_a = tiny_config(tmp_path / "a", seed=777, generations=1)
    cfg_a.selfplay.n_games = 4
    cfg_a.selfplay.games_per_generation = 4
    cfg_a.selfplay.sims_per_move = 8
    cfg_a.selfplay.max_batch = 16
    cfg_a.optim.train_steps_per_generation = 1
    cfg_a.replay.capacity = 1024

    cfg_b = tiny_config(tmp_path / "b", seed=777, generations=1)
    cfg_b.selfplay.n_games = cfg_a.selfplay.n_games
    cfg_b.selfplay.games_per_generation = cfg_a.selfplay.games_per_generation
    cfg_b.selfplay.sims_per_move = cfg_a.selfplay.sims_per_move
    cfg_b.selfplay.max_batch = cfg_a.selfplay.max_batch
    cfg_b.optim.train_steps_per_generation = cfg_a.optim.train_steps_per_generation
    cfg_b.replay.capacity = cfg_a.replay.capacity

    run_training(cfg_a)
    run_training(cfg_b)

    tensors_a = state_tensors(Path(cfg_a.checkpoint_dir) / "gen_0001.pt")
    tensors_b = state_tensors(Path(cfg_b.checkpoint_dir) / "gen_0001.pt")
    assert tensors_a.keys() == tensors_b.keys()
    for key in tensors_a:
        torch.testing.assert_close(tensors_a[key], tensors_b[key], rtol=0.0, atol=0.0)


def test_parallel_selfplay_cpu_smoke_two_generations(tmp_path: Path) -> None:
    """Spawn workers on CPU; this stays small enough for normal CI runners."""
    cfg = tiny_config(tmp_path, seed=5150, generations=2)
    cfg.parallel_workers = 2
    cfg.worker_device = "cpu"
    cfg.model.hidden_sizes = [16]
    cfg.selfplay.n_games = 2
    cfg.selfplay.games_per_generation = 2
    cfg.selfplay.sims_per_move = 2
    cfg.selfplay.max_batch = 4
    cfg.selfplay.max_recorded_moves = 64
    cfg.selfplay.max_tree_nodes = 256
    cfg.optim.batch_size = 8
    cfg.optim.train_steps_per_generation = 1
    cfg.replay.capacity = 512

    start = time.monotonic()
    result = run_training(cfg)
    elapsed = time.monotonic() - start

    assert elapsed < 180.0
    assert len(result["metrics"]) == 2
    assert [row["games"] for row in result["metrics"]] == [2, 2]
    metrics = read_metrics(Path(cfg.metrics_csv))
    assert [int(row["workers"]) for row in metrics] == [2, 2]
    assert all(float(row["aggregate_games_per_hour"]) > 0.0 for row in metrics)


def test_parallel_game_quotas_are_exact() -> None:
    assert game_quotas(2048, 24) == [86] * 8 + [85] * 16
