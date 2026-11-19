from __future__ import annotations

import numpy as np
import pytest
import torch

import dominion_v2_py as dz

from .config import SelfPlayConfig
from .gating import (
    SelfPlaySegment,
    assign_kingdom_phase_to_segments,
    effective_kingdom_phase,
)
from .selfplay import play_routed_games
from .test_train_smoke import read_metrics, tiny_config
from .train import run_training


ENGINE_POOL = [
    "Village",
    "Smithy",
    "Laboratory",
    "Market",
    "Festival",
    "Cellar",
    "Chapel",
    "Moat",
    "Council Room",
    "Throne Room",
    "Harbinger",
    "Vassal",
]


class _ZeroModel(torch.nn.Module):
    def evaluate(self, observations: torch.Tensor, _masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((observations.shape[0], dz.ACTION_SPACE_SIZE), dtype=torch.float32, device=observations.device),
            torch.zeros((observations.shape[0],), dtype=torch.float32, device=observations.device),
        )


def test_effective_kingdom_phase_uses_last_inclusive_match_and_base_between_ranges() -> None:
    curriculum = [
        {"generations": [2, 3], "mode": "pool", "pool": ENGINE_POOL, "pool_fraction": 0.5},
        {"generations": [3, 3], "mode": "random"},
        {"generations": [5, 6], "mode": "random"},
    ]

    assert effective_kingdom_phase(curriculum, "fixed", 1).label == "base:fixed"
    assert effective_kingdom_phase(curriculum, "fixed", 2).mode == "pool"
    assert effective_kingdom_phase(curriculum, "fixed", 2).pool_fraction == pytest.approx(0.5)
    # The second phase wins at its inclusive overlap with the first.
    assert effective_kingdom_phase(curriculum, "fixed", 3).label == "3-3:random"
    assert effective_kingdom_phase(curriculum, "fixed", 4).label == "base:fixed"
    assert effective_kingdom_phase(curriculum, "fixed", 5).label == "5-6:random"
    assert effective_kingdom_phase(curriculum, "fixed", 7).label == "base:fixed"


@pytest.mark.parametrize(
    "curriculum, match",
    [
        ([{"generations": [1, 2], "mode": "pool", "pool": [*ENGINE_POOL[:-1], "No Such Card"]}], "unknown"),
        ([{"generations": [1, 2], "mode": "pool", "pool": ENGINE_POOL[:9]}], "at least 10"),
        ([{"generations": [1, 2], "mode": "pool", "pool": ["Copper", *ENGINE_POOL[:9]]}], "unimplemented"),
    ],
)
def test_kingdom_curriculum_rejects_invalid_pools(curriculum: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        effective_kingdom_phase(curriculum, "random", 1)


def test_kingdom_curriculum_rejects_invalid_generation_range() -> None:
    with pytest.raises(ValueError, match="inclusive ranges"):
        effective_kingdom_phase(
            [{"generations": [5, 4], "mode": "random"}],
            "random",
            1,
        )


def test_kingdom_curriculum_mixture_splits_exact_generation_counts() -> None:
    phase = effective_kingdom_phase(
        [{"generations": [1, 1], "mode": "pool", "pool": ENGINE_POOL, "pool_fraction": 0.35}],
        "random",
        1,
    )
    assigned = assign_kingdom_phase_to_segments(
        [SelfPlaySegment(7, 0, 0), SelfPlaySegment(13, 0, 1)],
        phase,
        seed=12345,
    )

    assert sum(segment.n_games for segment in assigned) == 20
    assert sum(segment.n_games for segment in assigned if segment.kingdom_pool is not None) == 7
    assert sum(segment.n_games for segment in assigned if segment.kingdom_pool is None) == 13
    assert all(segment.kingdom_mode == "random" for segment in assigned)
    assert all(
        segment.kingdom_pool == [dz.def_id(name) for name in ENGINE_POOL]
        for segment in assigned
        if segment.kingdom_pool is not None
    )
    assert sum(segment.n_games for segment in assigned if (segment.seat0_model_id, segment.seat1_model_id) == (0, 0)) == 7
    assert sum(segment.n_games for segment in assigned if (segment.seat0_model_id, segment.seat1_model_id) == (0, 1)) == 13


def test_pool_restricted_segment_games_only_contain_pool_cards() -> None:
    config = SelfPlayConfig(
        n_games=1,
        sims_per_move=2,
        max_batch=4,
        max_recorded_moves=64,
        max_tree_nodes=256,
        kingdom_mode="random",
        dirichlet_frac=0.0,
    )
    model = _ZeroModel()
    phase = effective_kingdom_phase(
        [{"generations": [1, 1], "mode": "pool", "pool": ENGINE_POOL}], "random", 1
    )
    segment = assign_kingdom_phase_to_segments([SelfPlaySegment(1, 0, 0)], phase, seed=0xC14B)[0]
    assert segment.kingdom_pool is not None
    _stats, records = play_routed_games(
        (model, model),
        config,
        seed=0xC14B,
        device=torch.device("cpu"),
        target_games=1,
        kingdom_pool=segment.kingdom_pool,
        kingdom_mode=segment.kingdom_mode,
    )

    assert len(records) == 1
    assert len(records[0]["kingdom"]) == 10
    assert set(records[0]["kingdom"]).issubset(set(segment.kingdom_pool))
    assert np.asarray(records[0]["kingdom"]).size == 10


def test_training_metrics_record_the_active_kingdom_phase(tmp_path) -> None:
    cfg = tiny_config(tmp_path, seed=0xC14C, generations=1)
    cfg.model.hidden_sizes = [8]
    cfg.selfplay.n_games = 1
    cfg.selfplay.games_per_generation = 2
    cfg.selfplay.sims_per_move = 2
    cfg.selfplay.max_batch = 4
    cfg.selfplay.max_recorded_moves = 64
    cfg.selfplay.max_tree_nodes = 256
    cfg.optim.batch_size = 4
    cfg.optim.train_steps_per_generation = 1
    cfg.replay.capacity = 256
    cfg.kingdom_curriculum = [
        {"generations": [1, 1], "mode": "pool", "pool": ENGINE_POOL, "pool_fraction": 0.5}
    ]

    result = run_training(cfg)

    expected = "1-1:pool:0.500000:" + ",".join(ENGINE_POOL)
    assert result["metrics"][0]["kingdom_phase"] == expected
    assert read_metrics(tmp_path / "ckpt" / "metrics.csv")[0]["kingdom_phase"] == expected
