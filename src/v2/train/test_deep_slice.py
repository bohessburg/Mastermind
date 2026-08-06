from __future__ import annotations

import csv
from pathlib import Path

import pytest

from .config import SelfPlayConfig, validate_deep_slice_config
from .gating import compact_selfplay_segments, plan_training_selfplay_segments
from .selfplay import make_runner_config
from .test_train_smoke import read_metrics, tiny_config
from .train import append_metrics, run_training
from .workers import game_quotas, split_segments_by_quotas


@pytest.mark.parametrize(
    ("total_games", "fraction", "expected_deep_games"),
    [
        (8, 0.25, 2),
        # Python's existing fraction convention uses bankers' rounding.
        (5, 0.50, 2),
        (3, 0.50, 2),
        (5, 0.10, 0),
        (7, 0.50, 4),
    ],
)
def test_deep_slice_game_counts_use_exact_existing_rounding(
    total_games: int,
    fraction: float,
    expected_deep_games: int,
) -> None:
    segments = plan_training_selfplay_segments(
        total_games=total_games,
        league_fraction=0.0,
        league_pool_size=0,
        scripted_opponents={},
        seed=991,
        deep_slice_fraction=fraction,
        deep_slice_sims=128,
        sims_per_move=16,
    )

    assert sum(segment.n_games for segment in segments) == total_games
    deep = [segment for segment in segments if segment.sims_override]
    assert sum(segment.n_games for segment in deep) == expected_deep_games
    assert all(segment.is_normal_mirror for segment in deep)
    assert all(segment.sims_override == 128 for segment in deep)


def test_deep_slice_only_carves_normal_mirror_games() -> None:
    segments = plan_training_selfplay_segments(
        total_games=40,
        league_fraction=0.20,
        league_pool_size=2,
        scripted_opponents={"bigmoney": 0.15, "engine": 0.05},
        seed=1234,
        deep_slice_fraction=0.25,
        deep_slice_sims=128,
        sims_per_move=16,
    )

    assert sum(segment.n_games for segment in segments) == 40
    assert sum(segment.n_games for segment in segments if segment.is_league) == 8
    assert sum(segment.n_games for segment in segments if segment.is_scripted) == 8
    assert sum(segment.n_games for segment in segments if segment.sims_override) == 6
    assert all(
        segment.sims_override == 0
        for segment in segments
        if segment.is_league or segment.is_scripted
    )


def test_parallel_deep_slice_is_evenly_split_and_assigned_across_workers() -> None:
    total_games = 1024
    parallel_workers = 8
    segments = plan_training_selfplay_segments(
        total_games=total_games,
        league_fraction=0.0,
        league_pool_size=0,
        scripted_opponents={},
        seed=991,
        deep_slice_fraction=0.05,
        deep_slice_sims=4096,
        sims_per_move=64,
        parallel_workers=parallel_workers,
    )

    deep = [segment for segment in segments if segment.sims_override]
    assert len(deep) == parallel_workers
    assert sum(segment.n_games for segment in segments) == total_games
    assert sum(segment.n_games for segment in deep) == 51
    assert max(segment.n_games for segment in deep) - min(segment.n_games for segment in deep) <= 1
    assert {segment.sims_override for segment in deep} == {4096}

    # Repeated deep normal-mirror segments leave the model table unchanged.
    compacted, history_indices = compact_selfplay_segments(segments)
    assert history_indices == []
    assert sum(segment.n_games for segment in compacted if segment.sims_override) == 51

    assigned = split_segments_by_quotas(segments, game_quotas(total_games, parallel_workers))
    deep_by_worker = [
        [segment for segment in worker_segments if segment.sims_override]
        for worker_segments in assigned
    ]
    assert all(len(worker_segments) == 1 for worker_segments in deep_by_worker)
    assert [segment.n_games for worker_segments in deep_by_worker for segment in worker_segments] == [
        segment.n_games for segment in deep
    ]


@pytest.mark.parametrize(
    "config, message",
    [
        (SelfPlayConfig(deep_slice_fraction=-0.01), "between zero and one"),
        (SelfPlayConfig(deep_slice_fraction=1.01), "between zero and one"),
        (SelfPlayConfig(deep_slice_fraction=float("nan")), "between zero and one"),
        (SelfPlayConfig(sims_per_move=32, deep_slice_fraction=0.25, deep_slice_sims=32), "must exceed"),
        (SelfPlayConfig(sims_per_move=32, deep_slice_fraction=0.25, deep_slice_sims=0), "must exceed"),
    ],
)
def test_deep_slice_config_validation(config: SelfPlayConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_deep_slice_config(config)


def test_deep_runner_config_uses_override_scaled_capacity_and_search_flags() -> None:
    config = SelfPlayConfig(
        sims_per_move=32,
        max_tree_nodes=128,
        deep_slice_fraction=0.25,
        deep_slice_sims=512,
        tree_reuse=True,
        min_new_sims=7,
        expand_top_k=8,
    )

    normal = make_runner_config(config, 101)
    deep = make_runner_config(config, 102, sims_override=config.deep_slice_sims)

    assert normal.sims_per_move == 32
    assert normal.max_tree_nodes == 128
    assert deep.sims_per_move == 512
    assert deep.max_tree_nodes == 1024
    assert deep.tree_reuse is True
    assert deep.min_new_sims == 7
    assert deep.expand_top_k == 8


def test_append_metrics_migrates_legacy_header_for_deep_columns(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text("generation,games\n1,8\n")

    append_metrics(
        metrics_path,
        {"generation": 2, "games": 8, "deep_games": 2, "deep_positions": 123},
    )

    with metrics_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    assert "deep_games" in fieldnames
    assert "deep_positions" in fieldnames
    assert rows[0]["generation"] == "1"
    assert rows[0]["deep_games"] == ""
    assert rows[1]["deep_games"] == "2"
    assert rows[1]["deep_positions"] == "123"


def test_parallel_deep_slice_generation_reports_metrics(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, seed=7878, generations=1)
    cfg.parallel_workers = 2
    cfg.worker_device = "cpu"
    cfg.model.hidden_sizes = [16]
    cfg.selfplay.n_games = 1
    cfg.selfplay.games_per_generation = 4
    cfg.selfplay.sims_per_move = 2
    cfg.selfplay.deep_slice_fraction = 0.50
    cfg.selfplay.deep_slice_sims = 4
    cfg.selfplay.max_batch = 4
    cfg.selfplay.max_recorded_moves = 64
    cfg.selfplay.max_tree_nodes = 256
    cfg.optim.batch_size = 8
    cfg.optim.train_steps_per_generation = 1
    cfg.replay.capacity = 512

    result = run_training(cfg)

    row = result["metrics"][0]
    csv_row = read_metrics(Path(cfg.metrics_csv))[0]
    assert row["games"] == 4
    assert row["deep_games"] == 2
    assert row["deep_positions"] > 0
    assert int(csv_row["deep_games"]) == 2
    assert int(csv_row["deep_positions"]) > 0
