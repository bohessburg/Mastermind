"""Training self-play turn-cap configuration checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import dominion_v2_py as dz

from src.v2.train.config import TrainConfig, load_config, save_config
from src.v2.train.selfplay import make_runner_config


def test_selfplay_max_turns_round_trip_and_native_plumbing(tmp_path: Path) -> None:
    config = TrainConfig()
    assert config.selfplay.max_turns == 0

    zero_path = tmp_path / "selfplay_max_turns_zero.json"
    zero_path.write_text(json.dumps({"selfplay": {"max_turns": 0}}), encoding="utf-8")
    assert load_config(zero_path).selfplay.max_turns == 0

    config.selfplay.max_turns = 20
    path = tmp_path / "selfplay_max_turns.json"
    save_config(config, path)

    loaded = load_config(path)
    assert loaded.selfplay.max_turns == 20
    native = make_runner_config(loaded.selfplay, 0xCA20)
    assert native.selfplay_max_turns == 20
    assert dz.SelfPlayConfig().selfplay_max_turns == 0


@pytest.mark.parametrize("max_turns", [10, 250])
def test_selfplay_max_turns_rejects_out_of_range_values(tmp_path: Path, max_turns: int) -> None:
    path = tmp_path / "invalid_selfplay_max_turns.json"
    path.write_text(json.dumps({"selfplay": {"max_turns": max_turns}}), encoding="utf-8")
    with pytest.raises(ValueError, match="max_turns"):
        load_config(path)


def test_selfplay_turn_cap_record_is_truncated_with_zero_value_targets() -> None:
    config = dz.SelfPlayConfig(
        n_games=1,
        sims_per_move=2,
        max_batch=8,
        seed=0xCA20,
        kingdom_mode=dz.SelfPlayKingdomMode.Fixed,
        max_recorded_moves=256,
        max_tree_nodes=256,
        selfplay_max_turns=20,
        auto_play_treasures=True,
        prune_treasure_plays=True,
    )
    runner = dz.SelfPlayRunner(config)
    for _ in range(20_000):
        observations, _masks = runner.collect_leaves(config.max_batch)
        if observations.shape[0] > 0:
            runner.provide_evaluations(
                np.zeros((observations.shape[0],), dtype=np.float32),
                np.zeros((observations.shape[0], dz.ACTION_SPACE_SIZE), dtype=np.float32),
            )
        finished = runner.finished_games()
        if finished:
            record = finished[0]
            assert record["turn_counter"] == 20
            assert record["truncated"] is True
            assert np.all(record["values"] == 0.0)
            return
    raise AssertionError("selfplay turn cap did not finish a game")
