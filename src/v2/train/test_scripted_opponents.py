from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

import dominion_v2_py as dz

from .config import load_config
from .gating import SelfPlaySegment, effective_scripted_fractions, plan_training_selfplay_segments
from .inference_server import serialize_cpu_state_dict
from .test_train_smoke import read_metrics, tiny_config
from .train import build_objects, run_training
from .workers import ParallelSelfPlayPool


def _finish_one_scripted_game(
    nn_player: int,
    scripted_bot: dz.SelfPlayScriptedBotKind = dz.SelfPlayScriptedBotKind.BigMoney,
) -> tuple[dict, set[int]]:
    config = dz.SelfPlayConfig(
        n_games=1,
        sims_per_move=2,
        max_batch=4,
        seed=0x5C71,
        kingdom_mode=dz.SelfPlayKingdomMode.Fixed,
        scripted_bot=scripted_bot,
        scripted_nn_player=nn_player,
    )
    runner = dz.SelfPlayRunner(config)
    emitted_leaf_players: set[int] = set()
    for _ in range(20_000):
        obs, _masks = runner.collect_leaves(config.max_batch)
        if obs.shape[0] > 0:
            players = runner.leaf_players()
            emitted_leaf_players.update(int(player) for player in players)
            assert np.all(players == nn_player)
            runner.provide_evaluations(
                np.zeros((obs.shape[0],), dtype=np.float32),
                np.zeros((obs.shape[0], dz.ACTION_SPACE_SIZE), dtype=np.float32),
            )
        finished = runner.finished_games()
        if finished:
            return finished[0], emitted_leaf_players
    raise AssertionError("scripted self-play game did not finish")


@pytest.mark.parametrize(
    ("scripted_bot", "nn_player"),
    [
        (dz.SelfPlayScriptedBotKind.BigMoney, 1),
        (dz.SelfPlayScriptedBotKind.Engine, 0),
    ],
)
def test_scripted_runner_filters_to_nn_records_and_uses_nn_outcome_perspective(
    scripted_bot: dz.SelfPlayScriptedBotKind,
    nn_player: int,
) -> None:
    record, emitted_leaf_players = _finish_one_scripted_game(
        nn_player=nn_player,
        scripted_bot=scripted_bot,
    )

    assert emitted_leaf_players == {nn_player}
    assert record["scripted_nn_player"] == nn_player
    assert record["players"].dtype == np.uint8
    assert set(record["players"].tolist()) == {nn_player}
    assert record["observations"].shape[0] == record["policy_targets"].shape[0] == record["values"].shape[0]
    winner = record["winner"]
    expected_value = 0.0 if winner is None else (1.0 if int(winner) == nn_player else -1.0)
    np.testing.assert_array_equal(
        record["values"],
        np.full(record["values"].shape, expected_value, dtype=np.float32),
    )


def test_scripted_segment_fractions_account_exactly_and_seat_swap() -> None:
    segments = plan_training_selfplay_segments(
        total_games=40,
        league_fraction=0.2,
        league_pool_size=2,
        scripted_opponents={"bigmoney": 0.15, "engine": 0.05},
        seed=9876,
    )

    assert sum(segment.n_games for segment in segments) == 40
    assert sum(segment.n_games for segment in segments if segment.is_league) == 8
    scripted = [segment for segment in segments if segment.is_scripted]
    assert sum(segment.n_games for segment in scripted) == 8
    assert Counter({kind: sum(s.n_games for s in scripted if s.scripted_kind == kind) for kind in {"bigmoney", "engine"}}) == {
        "bigmoney": 6,
        "engine": 2,
    }
    assert {segment.nn_player for segment in scripted if segment.scripted_kind == "bigmoney"} == {0, 1}
    assert {segment.nn_player for segment in scripted if segment.scripted_kind == "engine"} == {0, 1}
    assert sum(segment.n_games for segment in segments if not segment.is_league and not segment.is_scripted) == 24


def test_effective_scripted_fractions_interpolates_breakpoints() -> None:
    schedule = {"bigmoney": [[10, 0.0], [11, 0.05], [25, 0.20]]}

    assert effective_scripted_fractions(schedule, {}, 0) == {}
    assert effective_scripted_fractions(schedule, {}, 10) == {}
    assert effective_scripted_fractions(schedule, {}, 11) == {"bigmoney": 0.05}
    assert effective_scripted_fractions(schedule, {}, 18) == pytest.approx({"bigmoney": 0.125})
    assert effective_scripted_fractions(schedule, {}, 25) == {"bigmoney": 0.20}
    assert effective_scripted_fractions(schedule, {}, 100) == {"bigmoney": 0.20}


def test_effective_scripted_fractions_schedule_wins_over_constant() -> None:
    fractions = effective_scripted_fractions(
        {"bigmoney": [[1, 0.0], [2, 0.25]]},
        {"bigmoney": 0.8, "engine": 0.1},
        2,
    )

    assert fractions == {"bigmoney": 0.25, "engine": 0.1}


@pytest.mark.parametrize(
    "schedule",
    [
        {"bigmoney": []},
        {"bigmoney": [[0, 0.0, 1.0]]},
        {"bigmoney": [[-1, 0.0]]},
        {"bigmoney": [[0, -0.1]]},
        {"bigmoney": [[0, 0.0], [0, 0.1]]},
    ],
)
def test_effective_scripted_fractions_rejects_invalid_breakpoints(schedule: dict[str, list]) -> None:
    with pytest.raises(ValueError):
        effective_scripted_fractions(schedule, {}, 1)


def test_parallel_workers_collect_scripted_segments(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, seed=5151, generations=1)
    cfg.parallel_workers = 2
    cfg.worker_device = "cpu"
    cfg.model.hidden_sizes = [16]
    cfg.selfplay.n_games = 1
    cfg.selfplay.games_per_generation = 2
    cfg.selfplay.sims_per_move = 2
    cfg.selfplay.max_batch = 4
    cfg.selfplay.max_recorded_moves = 64
    cfg.selfplay.max_tree_nodes = 256
    cfg.replay.capacity = 512
    model, _, replay = build_objects(cfg, torch.device("cpu"))
    segments = [
        SelfPlaySegment(1, 0, 0, scripted_kind="random", nn_player=0),
        SelfPlaySegment(1, 0, 0, scripted_kind="random", nn_player=1),
    ]

    pool = ParallelSelfPlayPool(cfg)
    try:
        result = pool.generate(
            model,
            replay,
            generation=1,
            segments=segments,
            model_state_payloads=[serialize_cpu_state_dict(model)],
        )
    finally:
        pool.close()

    assert result.stats.games == 2
    assert result.stats.scripted_games == 2
    assert 0 <= result.stats.scripted_wins <= 2
    assert len(replay) > 0


def test_ungated_training_accepts_scripted_opponents_and_reports_metrics(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, seed=6161, generations=1)
    cfg.model.hidden_sizes = [16]
    cfg.selfplay.n_games = 2
    cfg.selfplay.games_per_generation = 4
    cfg.selfplay.sims_per_move = 2
    cfg.selfplay.max_batch = 8
    cfg.selfplay.max_recorded_moves = 64
    cfg.selfplay.max_tree_nodes = 256
    cfg.optim.batch_size = 8
    cfg.optim.train_steps_per_generation = 1
    cfg.replay.capacity = 512
    cfg.scripted_opponents = {"bigmoney": 0.5}
    assert cfg.gate_games == 0

    result = run_training(cfg)
    row = result["metrics"][0]
    csv_row = read_metrics(Path(cfg.metrics_csv))[0]
    assert row["scripted_games"] == 2
    assert 0 <= row["scripted_wins"] <= row["scripted_games"]
    assert int(csv_row["scripted_games"]) == 2
    assert 0 <= int(csv_row["scripted_wins"]) <= 2


def test_ungated_training_uses_scripted_opponent_schedule_each_generation(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, seed=6262, generations=2)
    cfg.model.hidden_sizes = [16]
    cfg.selfplay.n_games = 2
    cfg.selfplay.games_per_generation = 4
    cfg.selfplay.sims_per_move = 2
    cfg.selfplay.max_batch = 8
    cfg.selfplay.max_recorded_moves = 64
    cfg.selfplay.max_tree_nodes = 256
    cfg.optim.batch_size = 8
    cfg.optim.train_steps_per_generation = 1
    cfg.replay.capacity = 512
    cfg.scripted_opponent_schedule = {"bigmoney": [[1, 0.0], [2, 0.5]]}

    result = run_training(cfg)

    assert [row["scripted_games"] for row in result["metrics"]] == [0, 2]
    assert "scripted_opponent_fractions" not in result["metrics"][0]
    assert result["metrics"][1]["scripted_opponent_fractions"] == {"bigmoney": 0.5}


def test_campaign5_config_loads_with_scripted_bigmoney_mix() -> None:
    config = load_config(Path(__file__).resolve().parents[3] / "configs" / "run_c5.json")
    assert config.gate_games == 0
    assert config.scripted_opponents == {"bigmoney": 0.2}
    assert config.model.hidden_sizes == [1536, 1536, 768]
    assert config.selfplay.sims_per_move == 256
    assert config.replay.capacity == 2_000_000


def test_campaign6_config_loads_with_scripted_bigmoney_schedule() -> None:
    config = load_config(Path(__file__).resolve().parents[3] / "configs" / "run_c6.json")
    assert config.seed == 20260714
    assert config.checkpoint_dir == "/workspace/checkpoints/campaign6"
    assert config.scripted_opponents == {}
    assert config.scripted_opponent_schedule == {"bigmoney": [[10, 0.0], [11, 0.05], [25, 0.20]]}


def test_campaign9_config_loads_with_two_opponent_curriculum() -> None:
    config = load_config(Path(__file__).resolve().parents[3] / "configs" / "run_c9.json")

    assert config.seed == 20260717
    assert config.checkpoint_dir == "/workspace/checkpoints/campaign9"
    assert config.selfplay.auto_play_treasures is True
    assert config.selfplay.prune_treasure_plays is True
    assert config.scripted_opponent_schedule == {
        "bigmoney": [[5, 0.0], [6, 0.01], [25, 0.20]],
        "engine": [[10, 0.0], [11, 0.01], [30, 0.20]],
    }
    assert effective_scripted_fractions(
        config.scripted_opponent_schedule,
        config.scripted_opponents,
        12,
    ) == pytest.approx({"bigmoney": 0.07, "engine": 0.02})
