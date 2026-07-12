from __future__ import annotations

import time

import numpy as np

import dominion_v2_py as dz

from .config import SelfPlayConfig
from .gating import SelfPlaySegment
from .selfplay import SelfPlayStats, make_runner_config
from .workers import (
    _record_manifest_outcomes,
    build_slot_manifest,
    index_selfplay_segments,
)


def _tiny_config() -> SelfPlayConfig:
    return SelfPlayConfig(
        n_games=8,
        games_per_generation=8,
        sims_per_move=1,
        max_batch=16,
        max_tree_nodes=128,
        max_recorded_moves=96,
        dirichlet_frac=0.0,
        temp_moves=0,
        kingdom_mode="random",
        scripted_threads=0,
    )


def _drive_zero_evaluator(runner: dz.SelfPlayRunner, wanted: int) -> tuple[list[dict], set[int]]:
    records: list[dict] = []
    model_ids: set[int] = set()
    for _ in range(100_000):
        obs, _masks = runner.collect_leaves(16)
        batch = int(obs.shape[0])
        if batch:
            model_ids.update(int(model_id) for model_id in runner.leaf_model_ids())
            runner.provide_evaluations(
                np.zeros((batch,), dtype=np.float32),
                np.zeros((batch, dz.ACTION_SPACE_SIZE), dtype=np.float32),
            )
        records.extend(runner.finished_games())
        if len(records) == wanted:
            return records, model_ids
    raise AssertionError(f"runner completed {len(records)} / {wanted} manifest slots")


def _record_fingerprint(record: dict) -> tuple:
    return (
        int(record["seed"]),
        int(record["game_index"]),
        record["winner"],
        record["scripted_nn_player"],
        int(record["seat0_model_id"]),
        int(record["seat1_model_id"]),
        int(record["scripted_bot"]),
        int(record["sims_override"]),
        tuple(record["kingdom"]),
        np.asarray(record["scores"], dtype=np.int16).tobytes(),
        np.asarray(record["observations"], dtype=np.float32).tobytes(),
        np.asarray(record["policy_targets"], dtype=np.float32).tobytes(),
        np.asarray(record["values"], dtype=np.float32).tobytes(),
        np.asarray(record["players"], dtype=np.uint8).tobytes(),
    )


def test_mixed_slot_manifest_preserves_model_tags_counters_and_kingdoms() -> None:
    cfg = _tiny_config()
    curriculum_pool = [
        dz.def_id(name)
        for name in (
            "Village",
            "Smithy",
            "Market",
            "Festival",
            "Laboratory",
            "Cellar",
            "Chapel",
            "Moat",
            "Council Room",
            "Throne Room",
        )
    ]
    segments = index_selfplay_segments(
        [
            SelfPlaySegment(1, 0, 0),
            SelfPlaySegment(1, 0, 1, league_opponent="league_one.pt"),
            SelfPlaySegment(1, 0, 2, league_opponent="league_two.pt"),
            SelfPlaySegment(1, 0, 0, kingdom_pool=curriculum_pool, kingdom_mode="random"),
            SelfPlaySegment(1, 0, 0, sims_override=2),
            SelfPlaySegment(1, 0, 0, scripted_kind="bigmoney", nn_player=0),
            SelfPlaySegment(1, 0, 0, scripted_kind="engine", nn_player=1),
        ]
    )
    slots = build_slot_manifest(segments)
    runner = dz.SelfPlayRunner(make_runner_config(cfg, 0x5E1F_1001, slot_manifest=slots))
    records, model_ids = _drive_zero_evaluator(runner, len(slots))

    assert len(records) == len(slots)
    # The actual leaf tags must expose both current and historical seats for
    # the Python worker's per-model batch grouping.
    assert {0, 1, 2}.issubset(model_ids)
    by_game_index = {int(record["game_index"]): record for record in records}
    by_slot = {slot.game_index: slot for slot in slots}
    assert set(by_game_index) == set(by_slot)
    for game_index, slot in by_slot.items():
        record = by_game_index[game_index]
        assert int(record["seat0_model_id"]) == slot.seat0_model_id
        assert int(record["seat1_model_id"]) == slot.seat1_model_id
        assert int(record["sims_override"]) == slot.sims_override
    curriculum_slot = next(slot for slot in slots if slot.kingdom_pool == curriculum_pool)
    assert set(by_game_index[curriculum_slot.game_index]["kingdom"]).issubset(set(curriculum_pool))

    stats = SelfPlayStats()
    _record_manifest_outcomes(stats, records, by_slot)
    assert stats.scripted_games == 2
    assert stats.scripted_by_kind["bigmoney"][0] == 1
    assert stats.scripted_by_kind["engine"][0] == 1
    assert stats.league_by_opponent["league_one.pt"][0] == 1
    assert stats.league_by_opponent["league_two.pt"][0] == 1
    assert stats.deep_games == 1
    assert stats.deep_positions >= 1


def test_pure_mirror_manifest_records_are_bitwise_legacy_equivalent() -> None:
    cfg = _tiny_config()
    cfg.n_games = 4
    cfg.games_per_generation = 4
    seed = 0x5E1F_1002

    legacy = dz.SelfPlayRunner(make_runner_config(cfg, seed))
    legacy_records, _ = _drive_zero_evaluator(legacy, 4)

    slots = build_slot_manifest(index_selfplay_segments([SelfPlaySegment(4, 0, 0)]))
    manifest = dz.SelfPlayRunner(make_runner_config(cfg, seed, slot_manifest=slots))
    manifest_records, _ = _drive_zero_evaluator(manifest, 4)

    assert [_record_fingerprint(record) for record in legacy_records] == [
        _record_fingerprint(record) for record in manifest_records
    ]


def test_mixed_manifest_matches_segment_sequential_trajectories_by_seed() -> None:
    cfg = _tiny_config()
    seed = 0x5E1F_1003
    pool = [
        dz.def_id(name)
        for name in (
            "Village",
            "Smithy",
            "Market",
            "Festival",
            "Laboratory",
            "Cellar",
            "Chapel",
            "Moat",
            "Council Room",
            "Throne Room",
        )
    ]
    segments = index_selfplay_segments(
        [
            SelfPlaySegment(2, 0, 0),
            SelfPlaySegment(1, 0, 1, league_opponent="league_one.pt"),
            SelfPlaySegment(1, 0, 2, league_opponent="league_two.pt"),
            SelfPlaySegment(1, 0, 0, kingdom_pool=pool, kingdom_mode="random"),
            SelfPlaySegment(1, 0, 0, sims_override=2),
        ]
    )
    all_slots = build_slot_manifest(segments)
    all_runner = dz.SelfPlayRunner(make_runner_config(cfg, seed, slot_manifest=all_slots))
    all_records, _ = _drive_zero_evaluator(all_runner, len(all_slots))

    sequential_records: list[dict] = []
    for segment in segments:
        segment_slots = build_slot_manifest([segment])
        runner = dz.SelfPlayRunner(make_runner_config(cfg, seed, slot_manifest=segment_slots))
        records, _ = _drive_zero_evaluator(runner, len(segment_slots))
        sequential_records.extend(records)

    all_by_seed = {int(record["seed"]): _record_fingerprint(record) for record in all_records}
    sequential_by_seed = {
        int(record["seed"]): _record_fingerprint(record) for record in sequential_records
    }
    assert all_by_seed == sequential_by_seed


def test_manifest_scaffold_slot_keeps_async_offload_live() -> None:
    cfg = _tiny_config()
    cfg.kingdom_mode = "fixed"
    cfg.scripted_threads = 1
    cfg.scaffold_sims = 1
    cfg.scaffold_determinizations = 1
    slots = build_slot_manifest(
        index_selfplay_segments(
            [
                SelfPlaySegment(1, 0, 0),
                SelfPlaySegment(1, 0, 0, scripted_kind="scaffold", nn_player=1),
            ]
        )
    )
    runner = dz.SelfPlayRunner(make_runner_config(cfg, 0x5E1F_1004, slot_manifest=slots))
    records: list[dict] = []
    for _ in range(100_000):
        obs, _masks = runner.collect_leaves(4)
        count = int(obs.shape[0])
        if count:
            runner.provide_evaluations(
                np.zeros((count,), dtype=np.float32),
                np.zeros((count, dz.ACTION_SPACE_SIZE), dtype=np.float32),
            )
        records.extend(runner.finished_games())
        if len(records) == len(slots):
            break
        if not count:
            time.sleep(0.0002)
    assert len(records) == len(slots)
    scaffold_record = next(record for record in records if record["scripted_nn_player"] is not None)
    assert scaffold_record["scripted_nn_player"] == 1
