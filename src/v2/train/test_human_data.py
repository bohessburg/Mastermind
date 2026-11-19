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

from src.v2.train.human_data import (
    POLICY_WEIGHT_EPSILON,
    policy_weight_for_rating,
    load_human_tuples,
    recompute_value_targets,
)


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


def test_skip_batches_lands_where_the_same_number_of_draws_would(tmp_path: Path) -> None:
    root = make_tuple_root(tmp_path / "tuples")
    # Batch 3 over 4 rows: every skip count below crosses reshuffle boundaries.
    for skipped in (0, 1, 2, 5, 9):
        drawn = load_human_tuples(root).minibatches(batch_size=3, seed=913)
        for _ in range(skipped):
            next(drawn)
        jumped = load_human_tuples(root).minibatches(batch_size=3, seed=913)
        jumped.skip_batches(skipped)
        for _ in range(5):
            expected = next(drawn)
            actual = next(jumped)
            np.testing.assert_array_equal(actual.obs, expected.obs)
            np.testing.assert_array_equal(actual.action, expected.action)

    with pytest.raises(ValueError, match="non-negative"):
        load_human_tuples(root).minibatches(batch_size=3, seed=913).skip_batches(-1)


def test_resumed_anchor_stream_continues_the_corpus_pass_instead_of_replaying_it(tmp_path: Path) -> None:
    from src.v2.train.config import TrainConfig
    from src.v2.train.train import anchor_batches_for_generation

    root = make_tuple_root(tmp_path / "tuples")
    config = TrainConfig()
    config.seed = 31
    config.imitation.anchor_batch_size = 3
    config.optim.train_steps_per_generation = 2

    # One uninterrupted trainer: generations 1..4, two anchor batches each.
    uninterrupted = anchor_batches_for_generation(load_human_tuples(root), config, 1)
    by_generation = {
        generation: [next(uninterrupted).obs for _ in range(config.optim.train_steps_per_generation)]
        for generation in (1, 2, 3, 4)
    }

    # A trainer restarted at each generation sees that generation's batches.
    for generation, expected in by_generation.items():
        resumed = anchor_batches_for_generation(load_human_tuples(root), config, generation)
        for batch in expected:
            np.testing.assert_array_equal(next(resumed).obs, batch)


def test_signed_policy_weight_curve_boundaries_and_deviation_discount() -> None:
    assert policy_weight_for_rating(39.0) == pytest.approx(POLICY_WEIGHT_EPSILON)
    assert policy_weight_for_rating(40.0) == pytest.approx(POLICY_WEIGHT_EPSILON)
    assert policy_weight_for_rating(45.0) == pytest.approx(0.51)
    # Continuous at 50: the expert weight ramps from 1.0 at 50 to 3.0 at 55.
    assert policy_weight_for_rating(49.99) == pytest.approx(policy_weight_for_rating(50.0), abs=1e-3)
    assert policy_weight_for_rating(50.0) == pytest.approx(1.0)
    assert policy_weight_for_rating(52.5) == pytest.approx(2.0)
    assert policy_weight_for_rating(55.0) == pytest.approx(3.0)
    assert policy_weight_for_rating(60.0) == pytest.approx(3.0)
    assert policy_weight_for_rating(None) == pytest.approx(POLICY_WEIGHT_EPSILON)
    # The ramp can start later: below it a player is value-only; 50+ is unchanged.
    assert policy_weight_for_rating(44.9, ramp_start=45.0) == pytest.approx(POLICY_WEIGHT_EPSILON)
    assert policy_weight_for_rating(47.5, ramp_start=45.0) == pytest.approx(0.51)
    assert policy_weight_for_rating(52.5, ramp_start=45.0) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="ramp_start"):
        policy_weight_for_rating(47.0, ramp_start=50.0)
    # Native Glicko deviation 0.5 is the observed sidecar scale ceiling and therefore
    # invokes the documented 0.5 confidence floor.
    assert policy_weight_for_rating(55.0, deviation=0.5) == pytest.approx(1.5)
    assert policy_weight_for_rating(45.0, deviation=0.25) == pytest.approx(0.255)


def test_disabled_skill_weighting_is_loader_bit_identical_and_skips_sidecar(tmp_path: Path) -> None:
    root = make_tuple_root(tmp_path / "tuples")
    baseline = load_human_tuples(root)
    disabled = load_human_tuples(
        root,
        skill_weighting=False,
        ratings_sidecar=tmp_path / "does-not-exist.json",
    )

    for field in ("obs", "action", "legal", "value", "margin", "winner", "seat_index", "game_index", "ply_index", "turn_number"):
        np.testing.assert_array_equal(getattr(disabled, field), getattr(baseline, field))
    np.testing.assert_array_equal(disabled.policy_weight, np.ones(len(disabled), dtype=np.float32))
    assert disabled.minibatches(2, 17).__next__().policy_weight is None


def test_skill_weighted_loader_joins_manifest_player_ids_by_acting_seat(tmp_path: Path) -> None:
    root = make_tuple_root(tmp_path / "tuples")
    manifest_path = root / "tuple_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["games"][0]["id"] = "first"
    manifest["games"][0]["player_ids"] = [101, 102]
    manifest["games"][1]["id"] = "second"
    manifest["games"][1]["player_ids"] = [201, 202]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with np.load(root / "tuples-00000.npz", allow_pickle=False) as shard:
        contents = {name: shard[name] for name in shard.files}
    contents["seat_index"] = np.asarray([0, 1, 0, 1], dtype=np.int16)
    np.savez_compressed(root / "tuples-00000.npz", **contents)
    sidecar_path = tmp_path / "ratings.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "first": {
                    "101": {"level": 39.0, "deviation": 0.0},
                    "102": {"level": 55.0, "deviation": 0.375},
                },
                "second": {
                    "201": {"rating": 45.0, "deviation": 0.0},
                    "202": None,
                },
            }
        ),
        encoding="utf-8",
    )

    dataset = load_human_tuples(root, skill_weighting=True, ratings_sidecar=sidecar_path)
    np.testing.assert_allclose(dataset.policy_weight, [0.02, 1.5, 0.51, 0.02], rtol=0.0, atol=1.0e-6)
    assert dataset.rating_band.tolist() == ["level_below_40", "level_50_plus", "level_40_to_50", "unrated_or_missing"]


if __name__ == "__main__":  # pragma: no cover - standalone test entry point
    raise SystemExit(pytest.main([str(Path(__file__)), "-q"]))
