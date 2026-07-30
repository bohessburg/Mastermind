"""Tests for manifest-backed human imitation tuples.

Run directly with:
    PYTHONPATH=build ./.venv/bin/python src/v2/train/test_human_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.train.human_data import load_human_tuples, recompute_value_targets


def make_tuple_root(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    obs = np.arange(16, dtype=np.float32).reshape(4, 4)
    action = np.asarray([0, 1, 2, 0], dtype=np.int32)
    legal = np.ones((4, 3), dtype=np.bool_)
    margin = np.asarray([0, 1, -10, 30], dtype=np.int16)
    np.savez_compressed(
        root / "tuples-00000.npz",
        obs=obs,
        action=action,
        legal=legal,
        value=np.zeros((4,), dtype=np.float32),
        margin=margin,
        winner=np.asarray([-1, 0, 0, 1], dtype=np.int8),
        seat_index=np.asarray([0, 0, 1, 1], dtype=np.int16),
        game_index=np.asarray([0, 0, 1, 1], dtype=np.int32),
        ply_index=np.arange(4, dtype=np.int32),
        turn_number=np.asarray([1, 1, 2, 2], dtype=np.int32),
    )
    manifest = {
        "schema_version": 1,
        "obs_width": 4,
        "action_width": 3,
        "shards": [{"path": "tuples-00000.npz", "tuples": 4}],
        "totals": {"tuples_exported": 4},
        "games": [
            {"index": 0, "seat_kinds": ["human", "bot:bigmoney"]},
            {"index": 1, "seat_kinds": ["bot:engine3", "human"]},
        ],
    }
    (root / "tuple_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_manifest_filters_by_opponent_kind_and_seat(tmp_path: Path) -> None:
    root = make_tuple_root(tmp_path / "tuples")

    bigmoney = load_human_tuples(root, opponent_kinds=["bigmoney"])
    assert len(bigmoney) == 2
    np.testing.assert_array_equal(bigmoney.game_index, np.asarray([0, 0], dtype=np.int32))

    engine3_seat_one = load_human_tuples(root, opponent_kinds=["bot:engine3"], seat_indices=[1])
    assert len(engine3_seat_one) == 2
    np.testing.assert_array_equal(engine3_seat_one.game_index, np.asarray([1, 1], dtype=np.int32))
    np.testing.assert_array_equal(engine3_seat_one.seat_index, np.asarray([1, 1], dtype=np.int16))


def test_value_recomputation_matches_margin_blend_endpoints() -> None:
    margins = np.asarray([0, 1, -10, 30], dtype=np.int16)
    alpha_six = recompute_value_targets(margins, scheme="margin_blend", alpha=0.6, scale=20.0)
    alpha_zero = recompute_value_targets(margins, scheme="margin_blend", alpha=0.0, scale=20.0)

    np.testing.assert_allclose(alpha_six, [0.0, 0.81, -0.90, 1.0], rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(alpha_zero, [0.0, 0.525, -0.75, 1.0], rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        recompute_value_targets(margins, scheme="outcome"),
        [0.0, 1.0, -1.0, 1.0],
        rtol=0.0,
        atol=0.0,
    )


def test_seeded_minibatches_cycle_forever_and_match_across_iterators(tmp_path: Path) -> None:
    root = make_tuple_root(tmp_path / "tuples")
    first = load_human_tuples(root).minibatches(batch_size=3, seed=913)
    second = load_human_tuples(root).minibatches(batch_size=3, seed=913)

    for _ in range(6):  # Cross several reshuffle boundaries; no StopIteration.
        first_batch = next(first)
        second_batch = next(second)
        np.testing.assert_array_equal(first_batch.obs, second_batch.obs)
        np.testing.assert_array_equal(first_batch.action, second_batch.action)
        np.testing.assert_array_equal(first_batch.legal, second_batch.legal)
        np.testing.assert_array_equal(first_batch.value, second_batch.value)


if __name__ == "__main__":  # pragma: no cover - standalone test entry point
    raise SystemExit(pytest.main([str(Path(__file__)), "-q"]))
