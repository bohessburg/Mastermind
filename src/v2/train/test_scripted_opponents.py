from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

import numpy as np
import pytest
import torch

import dominion_v2_py as dz

from .config import SelfPlayConfig, load_config
from .gating import SelfPlaySegment, effective_scripted_fractions, plan_training_selfplay_segments
from .inference_server import serialize_cpu_state_dict
from .selfplay import SelfPlayStats, _record_scripted_outcomes, make_runner_config
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


def test_scaffold_scripted_runner_smoke_records_only_nn_seat() -> None:
    nn_player = 1
    config = dz.SelfPlayConfig(
        n_games=1,
        sims_per_move=4,
        scaffold_sims=32,
        max_batch=8,
        seed=0x5CAFF01D,
        kingdom_mode=dz.SelfPlayKingdomMode.Fixed,
        max_tree_nodes=512,
        scripted_bot=dz.SelfPlayScriptedBotKind.Scaffold,
        scripted_nn_player=nn_player,
        auto_play_treasures=True,
        prune_treasure_plays=True,
    )
    runner = dz.SelfPlayRunner(config)

    for _ in range(30_000):
        observations, _masks = runner.collect_leaves(config.max_batch)
        if observations.shape[0] > 0:
            players = runner.leaf_players()
            assert np.all(players == nn_player)
            runner.provide_evaluations(
                np.zeros((observations.shape[0],), dtype=np.float32),
                np.zeros((observations.shape[0], dz.ACTION_SPACE_SIZE), dtype=np.float32),
            )
        else:
            time.sleep(0.001)
        finished = runner.finished_games()
        if finished:
            record = finished[0]
            break
    else:
        raise AssertionError("Scaffold scripted self-play smoke did not finish a game")

    assert record["scripted_nn_player"] == nn_player
    assert record["players"].shape[0] > 0
    assert np.all(record["players"] == nn_player)
    winner = record["winner"]
    expected_value = 0.0 if winner is None else (1.0 if int(winner) == nn_player else -1.0)
    np.testing.assert_array_equal(
        record["values"],
        np.full(record["values"].shape, expected_value, dtype=np.float32),
    )


def _collect_uniform_scripted_records(
    scripted_bot: dz.SelfPlayScriptedBotKind,
    *,
    scripted_threads: int,
    n_games: int,
    seed: int = 0x5CAFF0D,
) -> list[dict]:
    config = dz.SelfPlayConfig(
        n_games=n_games,
        sims_per_move=2,
        scaffold_sims=2,
        scripted_threads=scripted_threads,
        max_batch=32,
        seed=seed,
        kingdom_mode=dz.SelfPlayKingdomMode.Fixed,
        dirichlet_frac=0.0,
        temp_moves=0,
        max_recorded_moves=256,
        max_tree_nodes=256,
        scripted_bot=scripted_bot,
        scripted_nn_player=1,
        auto_play_treasures=True,
        prune_treasure_plays=True,
    )
    runner = dz.SelfPlayRunner(config)
    mask = (1 << 64) - 1
    initial_seeds = {
        (seed + index * 0xD1B54A32D192ED03) & mask
        for index in range(n_games)
    }
    records: dict[int, dict] = {}
    deadline = time.monotonic() + 90.0
    while set(records) != initial_seeds:
        if time.monotonic() >= deadline:
            raise AssertionError("uniform scripted self-play did not finish the initial slots")
        observations, _masks = runner.collect_leaves(config.max_batch)
        if observations.shape[0] > 0:
            players = runner.leaf_players()
            assert np.all(players == 1)
            runner.provide_evaluations(
                np.zeros((observations.shape[0],), dtype=np.float32),
                np.zeros((observations.shape[0], dz.ACTION_SPACE_SIZE), dtype=np.float32),
            )
        else:
            # All slots may briefly be in Scaffold jobs. Let those workers run
            # without turning the test's wait loop into a hot CPU spin.
            time.sleep(0.001)
        for record in runner.finished_games():
            record_seed = int(record["seed"])
            if record_seed in initial_seeds:
                records[record_seed] = record
    return [records[record_seed] for record_seed in sorted(records)]


def _assert_records_identical_by_seed(reference: list[dict], candidate: list[dict]) -> None:
    assert [int(record["seed"]) for record in reference] == [int(record["seed"]) for record in candidate]
    for expected, actual in zip(reference, candidate, strict=True):
        assert expected["winner"] == actual["winner"]
        assert expected["scripted_nn_player"] == actual["scripted_nn_player"]
        np.testing.assert_array_equal(expected["observations"], actual["observations"])
        np.testing.assert_array_equal(expected["policy_targets"], actual["policy_targets"])
        np.testing.assert_array_equal(expected["values"], actual["values"])
        np.testing.assert_array_equal(expected["players"], actual["players"])


def test_threaded_scaffold_selfplay_matches_synchronous_records_by_seed() -> None:
    threaded = _collect_uniform_scripted_records(
        dz.SelfPlayScriptedBotKind.Scaffold,
        scripted_threads=2,
        n_games=8,
    )
    synchronous = _collect_uniform_scripted_records(
        dz.SelfPlayScriptedBotKind.Scaffold,
        scripted_threads=0,
        n_games=8,
    )
    single_worker = _collect_uniform_scripted_records(
        dz.SelfPlayScriptedBotKind.Scaffold,
        scripted_threads=1,
        n_games=8,
    )

    _assert_records_identical_by_seed(synchronous, threaded)
    _assert_records_identical_by_seed(synchronous, single_worker)
    synchronous_stats = SelfPlayStats()
    _record_scripted_outcomes(synchronous_stats, synchronous, "scaffold")
    for records in (threaded, single_worker):
        stats = SelfPlayStats()
        _record_scripted_outcomes(stats, records, "scaffold")
        assert stats.scripted_games == synchronous_stats.scripted_games == 8
        assert stats.scripted_wins == synchronous_stats.scripted_wins
        assert stats.scripted_by_kind == synchronous_stats.scripted_by_kind


def test_bigmoney_scripted_run_is_unaffected_by_scripted_threads() -> None:
    synchronous = _collect_uniform_scripted_records(
        dz.SelfPlayScriptedBotKind.BigMoney,
        scripted_threads=0,
        n_games=2,
        seed=0xB16B00B5,
    )
    configured_threads = _collect_uniform_scripted_records(
        dz.SelfPlayScriptedBotKind.BigMoney,
        scripted_threads=2,
        n_games=2,
        seed=0xB16B00B5,
    )

    _assert_records_identical_by_seed(synchronous, configured_threads)


def _scripted_uniform_outcomes(
    scripted_bot: dz.SelfPlayScriptedBotKind,
    nn_player: int,
    games: int = 8,
) -> list[dict]:
    config = dz.SelfPlayConfig(
        n_games=1,
        sims_per_move=16,
        max_batch=64,
        seed=0x51A7 + nn_player + (100 if scripted_bot == dz.SelfPlayScriptedBotKind.BigMoney else 0),
        kingdom_mode=dz.SelfPlayKingdomMode.Fixed,
        dirichlet_alpha=0.3,
        dirichlet_frac=0.25,
        temp_moves=20,
        max_tree_nodes=512,
        scripted_bot=scripted_bot,
        scripted_nn_player=nn_player,
        auto_play_treasures=True,
        prune_treasure_plays=True,
    )
    runner = dz.SelfPlayRunner(config)
    records: list[dict] = []
    for _ in range(100_000):
        observations, _masks = runner.collect_leaves(config.max_batch)
        if observations.shape[0] > 0:
            runner.provide_evaluations(
                np.zeros((observations.shape[0],), dtype=np.float32),
                np.zeros((observations.shape[0], dz.ACTION_SPACE_SIZE), dtype=np.float32),
            )
        records.extend(runner.finished_games())
        if len(records) >= games:
            return records[:games]
    raise AssertionError("uniform scripted self-play did not finish enough games")


@pytest.mark.parametrize(
    "scripted_bot",
    [dz.SelfPlayScriptedBotKind.Engine, dz.SelfPlayScriptedBotKind.BigMoney],
)
def test_uniform_nn_does_not_receive_spurious_scripted_wins(
    scripted_bot: dz.SelfPlayScriptedBotKind,
) -> None:
    records = [
        *_scripted_uniform_outcomes(scripted_bot, nn_player=0),
        *_scripted_uniform_outcomes(scripted_bot, nn_player=1),
    ]

    assert all(record["winner"] is not None for record in records)
    assert all(record["winner"] != record["scripted_nn_player"] for record in records)


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
        SelfPlaySegment(1, 0, 0, scripted_kind="bigmoney", nn_player=0),
        SelfPlaySegment(1, 0, 0, scripted_kind="engine", nn_player=1),
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
    assert result.stats.scripted_by_kind["bigmoney"][0] == 1
    assert result.stats.scripted_by_kind["engine"][0] == 1
    assert sum(wins for _games, wins in result.stats.scripted_by_kind.values()) == result.stats.scripted_wins
    assert len(replay) > 0


def test_ungated_training_accepts_scripted_opponents_and_reports_metrics(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, seed=6161, generations=1)
    cfg.model.hidden_sizes = [16]
    cfg.selfplay.n_games = 2
    cfg.selfplay.games_per_generation = 8
    cfg.selfplay.sims_per_move = 2
    cfg.selfplay.max_batch = 8
    cfg.selfplay.max_recorded_moves = 64
    cfg.selfplay.max_tree_nodes = 256
    cfg.optim.batch_size = 8
    cfg.optim.train_steps_per_generation = 1
    cfg.replay.capacity = 512
    cfg.scripted_opponents = {"bigmoney": 0.25, "engine": 0.25}
    assert cfg.gate_games == 0

    result = run_training(cfg)
    row = result["metrics"][0]
    csv_row = read_metrics(Path(cfg.metrics_csv))[0]
    assert row["scripted_games"] == 4
    assert 0 <= row["scripted_wins"] <= row["scripted_games"]
    assert row["scripted_games_bigmoney"] == row["scripted_games_engine"] == 2
    assert row["scripted_wins"] == row["scripted_wins_bigmoney"] + row["scripted_wins_engine"]
    assert int(csv_row["scripted_games"]) == 4
    assert 0 <= int(csv_row["scripted_wins"]) <= 4
    assert int(csv_row["scripted_games_bigmoney"]) == int(csv_row["scripted_games_engine"]) == 2
    assert int(csv_row["scripted_wins"]) == (
        int(csv_row["scripted_wins_bigmoney"]) + int(csv_row["scripted_wins_engine"])
    )


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
    assert [row["scripted_games_bigmoney"] for row in result["metrics"]] == [0, 2]
    csv_rows = read_metrics(Path(cfg.metrics_csv))
    assert [int(row["scripted_games_bigmoney"]) for row in csv_rows] == [0, 2]
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


def test_campaign11_config_loads_with_scaffold_curriculum() -> None:
    config = load_config(Path(__file__).resolve().parents[3] / "configs" / "run_c11.json")

    assert config.seed == 20260719
    assert config.checkpoint_dir == "checkpoints/campaign11"
    assert config.metrics_csv == "checkpoints/campaign11/metrics.csv"
    assert config.selfplay.scaffold_sims == 400
    assert config.scripted_opponent_schedule == {
        "bigmoney": [[5, 0.0], [6, 0.01], [25, 0.20], [40, 0.20], [60, 0.05]],
        "scaffold": [[10, 0.0], [11, 0.01], [35, 0.25]],
    }
    assert effective_scripted_fractions(
        config.scripted_opponent_schedule,
        config.scripted_opponents,
        12,
    ) == pytest.approx({"bigmoney": 0.07, "scaffold": 0.02})


def test_campaign12_config_uses_margin_targets_with_the_scaffold_curriculum() -> None:
    config = load_config(Path(__file__).resolve().parents[3] / "configs" / "run_c12.json")

    assert config.seed == 20260720
    assert config.checkpoint_dir == "checkpoints/campaign12"
    assert config.metrics_csv == "checkpoints/campaign12/metrics.csv"
    assert config.selfplay.value_target == "margin"
    assert config.selfplay.margin_scale == 20.0
    assert config.selfplay.scaffold_sims_opening == 0
    assert config.selfplay.scaffold_determinizations == 2
    assert config.selfplay.scripted_threads == 2
    assert config.scripted_opponent_schedule == {
        "bigmoney": [[5, 0.0], [6, 0.01], [25, 0.20], [40, 0.20], [60, 0.05]],
        "scaffold": [[10, 0.0], [11, 0.01], [35, 0.25]],
    }


def test_campaign14_config_loads_warm_start_league_and_curriculum() -> None:
    config = load_config(Path(__file__).resolve().parents[3] / "configs" / "run_c14.json")

    assert config.seed == 20260722
    assert config.init_weights == "checkpoints/campaign13/gen_0065.pt"
    assert config.checkpoint_dir == "checkpoints/campaign14"
    assert config.metrics_csv == "checkpoints/campaign14/metrics.csv"
    assert config.model.hidden_sizes == [1536, 1536, 768]
    assert config.model.input_scale == 16.0
    assert config.selfplay.obs_version == 2
    assert config.selfplay.sims_per_move == 512
    assert config.selfplay.temp_moves == 20
    assert config.selfplay.value_target == "margin"
    assert config.selfplay.margin_scale == 20.0
    assert config.selfplay.auto_play_treasures is True
    assert config.selfplay.prune_treasure_plays is True
    assert config.selfplay.scripted_threads == 2
    assert config.selfplay.deep_slice_fraction == pytest.approx(0.05)
    assert config.selfplay.deep_slice_sims == 4096
    assert config.selfplay.tree_reuse is True
    assert config.selfplay.expand_top_k == 8
    assert config.selfplay.scaffold_determinizations == 1
    assert config.scripted_opponent_schedule == {}
    assert config.eval.eval_sentinels == [{"opponent": "bigmoney", "games": 100}]
    assert config.league_pool_size == 16
    assert config.league_self_every == 10
    assert config.league_schedule == [[1, 0.10], [10, 0.25], [30, 0.30]]
    assert config.league_seed_checkpoints == [
        "checkpoints/campaign13/gen_0010.pt",
        "checkpoints/campaign13/gen_0020.pt",
        "checkpoints/campaign13/gen_0025.pt",
        "checkpoints/campaign13/gen_0030.pt",
        "checkpoints/campaign13/gen_0040.pt",
        "checkpoints/campaign13/gen_0050.pt",
        "checkpoints/campaign13/gen_0060.pt",
        "checkpoints/campaign13/gen_0065.pt",
        "checkpoints/campaign13/gen_0070.pt",
        "checkpoints/campaign13/gen_0080.pt",
        "checkpoints/campaign13/gen_0090.pt",
        "checkpoints/campaign13/gen_0096.pt",
    ]
    assert config.kingdom_curriculum[0]["generations"] == [1, 15]
    assert config.kingdom_curriculum[0]["pool_fraction"] == pytest.approx(0.6)
    assert config.kingdom_curriculum[1]["generations"] == [16, 40]
    assert config.kingdom_curriculum[1]["pool_fraction"] == pytest.approx(0.3)
    assert config.kingdom_curriculum[2] == {"generations": [41, 100], "mode": "random"}


def test_make_runner_config_maps_and_validates_value_targets() -> None:
    native_defaults = dz.SelfPlayConfig()
    assert native_defaults.scripted_threads == 2
    assert native_defaults.scaffold_determinizations == 2
    assert native_defaults.scaffold_sims_opening == 0
    assert native_defaults.tree_reuse is False
    assert native_defaults.expand_top_k == 0

    config = SelfPlayConfig(
        value_target="margin",
        margin_scale=17.5,
        scaffold_sims=37,
        scaffold_sims_opening=5,
        scaffold_determinizations=3,
        scripted_threads=2,
        tree_reuse=True,
        expand_top_k=8,
    )
    runner_config = make_runner_config(config, 12345, scripted_kind="scaffold")

    assert runner_config.value_target == dz.SelfPlayValueTarget.Margin
    assert runner_config.margin_scale == pytest.approx(17.5)
    assert runner_config.scaffold_sims == 37
    assert runner_config.scaffold_sims_opening == 5
    assert runner_config.scaffold_determinizations == 3
    assert runner_config.scripted_threads == 2
    assert runner_config.tree_reuse is True
    assert runner_config.expand_top_k == 8
    assert runner_config.scripted_bot == dz.SelfPlayScriptedBotKind.Scaffold

    with pytest.raises(ValueError, match="unknown value target"):
        make_runner_config(SelfPlayConfig(value_target="rank"), 12345)
    with pytest.raises(ValueError, match="margin_scale"):
        make_runner_config(SelfPlayConfig(margin_scale=0.0), 12345)
