"""Regression tests for the 2026-11 audit fixes (Python side)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.train.human_data import load_human_tuples
from src.v2.train.train import hard_label_policy_loss


def _tuple_root(root: Path, outcomes: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = 6
    np.savez_compressed(
        root / "tuples-00000.npz",
        obs=np.zeros((rows, 4), dtype=np.float32),
        action=np.zeros((rows,), dtype=np.int32),
        legal=np.ones((rows, 3), dtype=np.bool_),
        value=np.zeros((rows,), dtype=np.float32),
        # game 0 resigned (margin written as 0), game 1 normal, game 2 shared tie
        margin=np.asarray([0, 0, 5, -5, 0, 0], dtype=np.int16),
        winner=np.asarray([0, 0, 0, 0, -1, -1], dtype=np.int8),
        seat_index=np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int16),
        game_index=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32),
        ply_index=np.arange(rows, dtype=np.int32),
        turn_number=np.ones((rows,), dtype=np.int32),
    )
    manifest = {
        "schema_version": 1,
        "obs_width": 4,
        "action_width": 3,
        "shards": [{"path": "tuples-00000.npz", "tuples": rows}],
        "totals": {"tuples_exported": rows},
        "games": [
            {"index": i, "seat_kinds": ["human", "human"], "outcome": outcome}
            for i, outcome in enumerate(outcomes)
        ],
    }
    (root / "tuple_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_resigned_rows_get_the_real_winner_sign(tmp_path: Path) -> None:
    root = _tuple_root(
        tmp_path / "t",
        [
            {"margin_valid": False, "margins": [7, -7], "outcome_signs": [1, -1], "resigned": [False, True]},
            {"margin_valid": True, "margins": [5, -5], "outcome_signs": [1, -1]},
            {"margin_valid": True, "margins": [0, 0], "outcome_signs": [0, 0]},
        ],
    )
    data = load_human_tuples(root, value_scheme="margin")
    assert data.margin.tolist() == [7, -7, 5, -5, 0, 0]
    assert data.value[0] > 0.5 and data.value[1] < -0.5
    assert data.value[4] == 0.0 and data.value[5] == 0.0


def test_weighted_policy_loss_is_a_weighted_mean() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    legal = torch.ones((2, 2), dtype=torch.bool)
    actions = torch.tensor([0, 0])
    per_row = torch.nn.functional.cross_entropy(logits, actions, reduction="none")
    weights = torch.tensor([0.02, 0.02])
    # Uniformly scaling every weight must not rescale the loss.
    assert hard_label_policy_loss(logits, legal, actions, weights=weights) == pytest.approx(
        float(per_row.mean()), rel=1e-6
    )
    weights = torch.tensor([3.0, 1.0])
    expected = float((per_row * weights).sum() / weights.sum())
    assert hard_label_policy_loss(logits, legal, actions, weights=weights) == pytest.approx(expected, rel=1e-6)


def test_paired_seating_shares_boards_and_swaps_seats() -> None:
    from src.v2.train.honest_eval import HonestEvalConfig, _a_seat, _config_game_seed

    cfg = HonestEvalConfig(
        a="a", b="b", opponent=None, games=6, sims_a=1, determinizations_a=1, sims_b=1,
        determinizations_b=1, workers=1, device="cpu", seed=100, torch_threads=1, legacy_shim=False,
    )
    assert [_a_seat(cfg, i) for i in range(6)] == [0, 1, 0, 1, 0, 1]
    assert _config_game_seed(cfg, 0) == _config_game_seed(cfg, 1)
    assert _config_game_seed(cfg, 2) != _config_game_seed(cfg, 1)
    blocked = HonestEvalConfig(**{**cfg.__dict__, "seating": "blocked"})
    assert [_a_seat(blocked, i) for i in range(6)] == [0, 0, 0, 1, 1, 1]


def test_score_statistics_wilson_and_paired_ci() -> None:
    from src.v2.train.honest_eval import GameResult, score_statistics, wilson_interval

    low, high = wilson_interval(66, 200)
    assert 0.26 < low < 0.27 and 0.39 < high < 0.40
    results = [
        GameResult(index=i, game_seed=0, a_seat=i % 2, turns=1, result=r, end_condition="province",
                   wall_time_seconds=0.0)
        for i, r in enumerate(["W", "L", "W", "W", "L", "T", "L", "L"])
    ]
    stats = score_statistics(results, paired=True)
    assert stats["games_scored"] == 8
    assert stats["score_pct"] == pytest.approx(100 * 3.5 / 8)
    assert stats["pairs"] == 4
    assert stats["paired_score_pct"] == pytest.approx(stats["score_pct"])


def test_sprt_llr_signs() -> None:
    from src.v2.train.sprt import gsprt_llr, pair_scores, sprt_bounds

    games = [{"index": i, "result": r} for i, r in enumerate(["W", "W", "W", "L"] * 20)]
    scores = pair_scores(games)
    assert len(scores) == 40
    assert gsprt_llr(scores, 0.0, 35.0) > 0
    losing = [{"index": i, "result": r} for i, r in enumerate(["L", "L", "W", "L"] * 20)]
    assert gsprt_llr(pair_scores(losing), 0.0, 35.0) < 0
    lower, upper = sprt_bounds(0.05, 0.05)
    assert lower == pytest.approx(-upper)


def test_ladder_recovers_ordering(tmp_path: Path) -> None:
    from src.v2.train.ladder import fit_bradley_terry

    rng = np.random.default_rng(0)
    truth = {"a": 200.0, "b": 0.0, "c": -200.0}
    games = []
    for x, y in (("a", "b"), ("b", "c"), ("a", "c")):
        p = 1.0 / (1.0 + 10 ** (-(truth[x] - truth[y]) / 400))
        games += [(x, y, float(rng.random() < p)) for _ in range(600)]
    ratings = fit_bradley_terry(games)
    assert ratings["a"] > ratings["b"] > ratings["c"]
    assert ratings["a"] - ratings["c"] == pytest.approx(400, abs=80)


def test_heldout_value_metrics_reads_only_fresh_rows() -> None:
    from src.v2.train.replay import ReplayBuffer
    from src.v2.train.selfplay import heldout_value_metrics

    class Const(torch.nn.Module):
        def forward(self, obs):
            return torch.zeros((obs.shape[0], 5)), torch.full((obs.shape[0],), 0.5)

    replay = ReplayBuffer(capacity=10, obs_size=3, action_size=5, seed=0)
    obs = np.zeros((4, 3), dtype=np.float32)
    policy = np.full((4, 5), 0.2, dtype=np.float32)
    legal = np.ones((4, 5), dtype=np.bool_)
    replay.add(obs, policy, np.full((4,), -1.0, dtype=np.float32), legal, np.zeros((4,), dtype=np.int16))
    start = replay.write
    replay.add(obs[:2], policy[:2], np.asarray([0.5, 0.5], dtype=np.float32), legal[:2], np.zeros((2,), dtype=np.int16))
    metrics = heldout_value_metrics(replay, Const(), torch.device("cpu"), start_write=start, inserted_positions=2)
    assert metrics["heldout_rows"] == 2
    assert metrics["heldout_value_mse"] == pytest.approx(0.0)


def test_coalescing_trigger_ignores_workers_that_went_quiet() -> None:
    from src.v2.train.inference_server import _PerModelRequestQueues, _Request

    queues = _PerModelRequestQueues(worker_count=3, target_rows=1000, max_batch=1000, coalesce_s=1.0,
                                    active_window_s=0.1)

    def req(worker: int, t: float) -> _Request:
        return _Request(worker_id=worker, request_id=0, slot=0, count=4, model_id=0, submitted_at_s=t,
                        obs=np.zeros((4, 1), np.float32), masks=np.zeros((4, 1), np.bool_))

    # Startup: unseen workers count as live, so one request must wait.
    queues.add(req(0, 0.0))
    assert queues.ready_model(0.01) is None
    queues.add(req(1, 0.01))
    queues.add(req(2, 0.02))
    assert queues.ready_model(0.02).trigger == "all_workers_pending"
    queues.take_batch(0)
    # Worker 2 finishes and goes quiet; after the window, 0+1 suffice.
    queues.add(req(0, 0.5))
    queues.add(req(1, 0.5))
    assert queues.ready_model(0.5).trigger == "all_workers_pending"
