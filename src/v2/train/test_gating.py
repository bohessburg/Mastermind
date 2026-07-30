from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

import dominion_v2_py as dz
from .config import TrainConfig
from .gating import (
    GateStats,
    SelfPlaySegment,
    archive_previous_best,
    archive_self_checkpoint,
    best_checkpoint_path,
    compact_selfplay_segments,
    effective_league_fraction,
    gate_result,
    initialize_best_checkpoint,
    league_checkpoint_paths,
    load_best_checkpoint,
    league_opponent_weights,
    plan_selfplay_segments,
    run_gate_match,
    sample_league_games,
    save_best_checkpoint,
    seed_league_checkpoints,
)
from .inference_server import serialize_cpu_state_dict
from .selfplay import make_runner_config, play_routed_games, route_leaf_evaluations
from .test_train_smoke import read_metrics, state_tensors, tiny_config
from .train import build_objects, load_checkpoint, run_training, save_checkpoint
from .workers import ParallelSelfPlayPool, game_quotas, split_segments_by_quotas


def test_gate_accept_reject_uses_decisive_win_rate() -> None:
    assert gate_result(GateStats(wins=11, losses=9, ties=40), 0.55) == "accepted"
    assert gate_result(GateStats(wins=10, losses=10, ties=40), 0.55) == "rejected"
    assert gate_result(GateStats(wins=0, losses=0, ties=60), 0.55) == "rejected"


def test_league_sampling_is_exact_uniform_and_seeded() -> None:
    first = sample_league_games(1000, 0.2, 4, seed=9876)
    second = sample_league_games(1000, 0.2, 4, seed=9876)
    assert first == second
    sampled = [game for game in first if game is not None]
    assert len(sampled) == 200
    counts = Counter(game.opponent_index for game in sampled)
    assert set(counts) == {0, 1, 2, 3}
    assert max(counts.values()) - min(counts.values()) < 40
    # League self-play is deliberately routed current-best as player zero and
    # the archived opponent as player one. Gate matches seat-swap separately.
    assert all(game.best_player == 0 for game in sampled)


def test_strength_matched_league_sampling_favors_lower_win_rate_seededly() -> None:
    names = ["hard.pt", "easy.pt"]
    # Tuples are (games, current-seat-zero wins): hard is only 10%, while
    # easy is fully beaten and remains selectable via the +0.1 floor.
    weights = league_opponent_weights(names, {"hard.pt": (100, 10), "easy.pt": (100, 100)})
    assert weights == pytest.approx([1.0, 0.1])
    first = sample_league_games(1_000, 1.0, 2, seed=12345, opponent_weights=weights)
    second = sample_league_games(1_000, 1.0, 2, seed=12345, opponent_weights=weights)
    assert first == second
    counts = Counter(game.opponent_index for game in first if game is not None)
    assert counts[0] > counts[1] * 5
    assert counts[1] > 0


def test_league_opponents_per_gen_caps_distinct_opponents_and_exact_counts() -> None:
    names = [f"opponent_{index}.pt" for index in range(8)]
    weights = league_opponent_weights(names, {})
    segments = plan_selfplay_segments(
        100,
        0.4,
        len(names),
        seed=0xBADC0DE,
        opponent_weights=weights,
        opponent_names=names,
        league_opponents_per_gen=3,
    )

    league_segments = [segment for segment in segments if segment.is_league]
    assert len(league_segments) == 3
    assert len({segment.league_opponent for segment in league_segments}) == 3
    assert all(isinstance(segment.n_games, int) and segment.n_games > 0 for segment in league_segments)
    assert sum(segment.n_games for segment in league_segments) == 40


def test_capped_league_sampling_is_weighted_without_replacement_and_seeded() -> None:
    names = ["hard.pt", "medium.pt", "easy.pt", "retired.pt"]
    weights = league_opponent_weights(
        names,
        {"hard.pt": (100, 10), "medium.pt": (100, 50), "easy.pt": (100, 100)},
    )
    first = sample_league_games(
        80,
        0.5,
        len(names),
        seed=0xC0FFEE,
        opponent_weights=weights,
        league_opponents_per_gen=3,
    )
    second = sample_league_games(
        80,
        0.5,
        len(names),
        seed=0xC0FFEE,
        opponent_weights=weights,
        league_opponents_per_gen=3,
    )
    assert first == second
    counts = Counter(game.opponent_index for game in first if game is not None)
    assert len(counts) == 3
    assert sum(counts.values()) == 40


def test_capped_league_sampling_rotates_uniform_pool_coverage_across_generations() -> None:
    pool_size = 8
    seen: set[int] = set()
    for generation in range(1, 9):
        games = sample_league_games(
            100,
            0.3,
            pool_size,
            seed=20260722 ^ (generation * 0xC0FFEE),
            opponent_weights=[1.1] * pool_size,
            league_opponents_per_gen=3,
        )
        seen.update(game.opponent_index for game in games if game is not None)
    assert seen == set(range(pool_size))


def test_uncapped_league_sampling_preserves_legacy_per_game_draw() -> None:
    games = sample_league_games(
        20,
        0.5,
        4,
        seed=9876,
        opponent_weights=[1.1, 0.8, 0.5, 0.2],
        league_opponents_per_gen=0,
    )
    assert [(index, game.opponent_index) for index, game in enumerate(games) if game is not None] == [
        (0, 0),
        (2, 0),
        (3, 0),
        (8, 3),
        (9, 2),
        (10, 0),
        (11, 0),
        (13, 0),
        (14, 1),
        (16, 2),
    ]


def test_league_schedule_interpolates_and_rejects_invalid_breakpoints() -> None:
    schedule = [[1, 0.0], [3, 0.5], [5, 1.0]]
    assert effective_league_fraction(schedule, 0.2, 0) == 0.0
    assert effective_league_fraction(schedule, 0.2, 2) == 0.25
    assert effective_league_fraction(schedule, 0.2, 5) == 1.0
    with pytest.raises(ValueError, match="strictly increasing"):
        effective_league_fraction([[1, 0.0], [1, 0.5]], 0.2, 1)


def test_league_mix_segments_honor_fraction_and_compact_model_table() -> None:
    raw_segments = plan_selfplay_segments(1000, 0.2, 4, seed=9876)
    assert sum(segment.n_games for segment in raw_segments) == 1000
    assert sum(segment.n_games for segment in raw_segments if segment.is_league) == 200
    assert all(segment.seat0_model_id == 0 for segment in raw_segments)

    segments, history_indices = compact_selfplay_segments(raw_segments)
    assert sum(segment.n_games for segment in segments) == 1000
    assert sum(segment.n_games for segment in segments if segment.is_league) == 200
    assert history_indices == [0, 1, 2, 3]
    assert {segment.seat1_model_id for segment in segments if segment.is_league} == {1, 2, 3, 4}


def test_single_opponent_league_stays_contiguous_until_worker_assignment() -> None:
    total_games = 80
    parallel_workers = 4
    raw_segments = plan_selfplay_segments(
        total_games,
        0.4,
        1,
        seed=9876,
        opponent_names=["best_0001.pt"],
        parallel_workers=parallel_workers,
    )

    assert [segment.n_games for segment in raw_segments] == [48, 32]
    assert raw_segments[1].is_league
    assert sum(segment.n_games for segment in raw_segments) == total_games
    assert {segment.league_opponent for segment in raw_segments if segment.is_league} == {"best_0001.pt"}

    compacted, history_indices = compact_selfplay_segments(raw_segments)
    assert history_indices == [0]
    assert {segment.seat1_model_id for segment in compacted if segment.is_league} == {1}
    assigned = split_segments_by_quotas(raw_segments, game_quotas(total_games, parallel_workers))
    assert [sum(segment.n_games for segment in worker_segments) for worker_segments in assigned] == game_quotas(
        total_games,
        parallel_workers,
    )
    league_games_by_worker = [
        sum(segment.n_games for segment in worker_segments if segment.is_league)
        for worker_segments in assigned
    ]
    assert league_games_by_worker == [0, 0, 12, 20]
    assert sum(games > 0 for games in league_games_by_worker) == 2


def test_worker_keeps_kingdom_split_league_model_blocks_together() -> None:
    segments = [
        SelfPlaySegment(40, 0, 0),
        SelfPlaySegment(10, 0, 1, kingdom_pool=[1], kingdom_mode="random", league_opponent="one.pt"),
        SelfPlaySegment(10, 0, 1, kingdom_mode="random", league_opponent="one.pt"),
        SelfPlaySegment(10, 0, 2, kingdom_pool=[1], kingdom_mode="random", league_opponent="two.pt"),
        SelfPlaySegment(10, 0, 2, kingdom_mode="random", league_opponent="two.pt"),
    ]

    assigned = split_segments_by_quotas(segments, game_quotas(80, 4))
    workers_by_model = {
        model_id: {
            worker_index
            for worker_index, worker_segments in enumerate(assigned)
            if any(segment.seat1_model_id == model_id for segment in worker_segments)
        }
        for model_id in (1, 2)
    }
    assert workers_by_model == {1: {0}, 2: {1}}
    assert [
        [segment.n_games for segment in worker_segments if segment.is_league]
        for worker_segments in assigned
    ] == [[10, 10], [10, 10], [], []]


def test_parallel_pool_routes_each_seat_to_its_model_table_entry(tmp_path: Path) -> None:
    """The pool must retain per-seat routing when worker processes are used."""
    cfg = tiny_config(tmp_path, seed=7171, generations=1)
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
    best, _, replay = build_objects(cfg, torch.device("cpu"))
    historical, _, _ = build_objects(cfg, torch.device("cpu"))
    # Make the two serialized model tables observably distinct. The worker's
    # routing audit is incremented next to route_leaf_evaluations, so it proves
    # every collected leaf selected the matching per-seat table entry.
    with torch.no_grad():
        next(historical.parameters()).add_(0.125)

    pool = ParallelSelfPlayPool(cfg)
    try:
        result = pool.generate(
            best,
            replay,
            generation=1,
            segments=[SelfPlaySegment(2, 0, 1)],
            model_state_payloads=[
                serialize_cpu_state_dict(best),
                serialize_cpu_state_dict(historical),
            ],
        )
    finally:
        pool.close()

    audit = result.seat_model_evals or {}
    assert result.stats.games == 2
    assert result.league_games == 2
    assert audit.get((0, 0), 0) > 0
    assert audit.get((1, 1), 0) > 0
    assert audit.get((0, 1), 0) == 0
    assert audit.get((1, 0), 0) == 0


class _TaggedEvaluator(torch.nn.Module):
    def __init__(self, tag: float):
        super().__init__()
        self.tag = float(tag)

    def evaluate(self, obs: torch.Tensor, _masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.full((obs.shape[0], dz.ACTION_SPACE_SIZE), self.tag, dtype=torch.float32, device=obs.device),
            torch.full((obs.shape[0],), self.tag, dtype=torch.float32, device=obs.device),
        )


def test_per_seat_model_routing_uses_leaf_attribution() -> None:
    obs = np.zeros((5, dz.OBS_SIZE), dtype=np.float32)
    masks = np.ones((5, dz.ACTION_SPACE_SIZE), dtype=np.bool_)
    players = np.asarray([0, 1, 1, 0, 1], dtype=np.uint8)
    logits, values = route_leaf_evaluations(
        (_TaggedEvaluator(3.0), _TaggedEvaluator(-2.0)),
        obs,
        masks,
        players,
        torch.device("cpu"),
    )
    np.testing.assert_array_equal(values, np.asarray([3.0, -2.0, -2.0, 3.0, -2.0], dtype=np.float32))
    np.testing.assert_array_equal(logits[:, 0], values)


def test_same_model_routing_fast_path_is_bit_exact_for_outputs_and_games(tmp_path: Path) -> None:
    """A same-model route must be exactly equivalent to legacy split/scatter."""
    cfg = tiny_config(tmp_path, seed=9191, generations=1)
    cfg.model.hidden_sizes = [16]
    cfg.selfplay.n_games = 2
    cfg.selfplay.games_per_generation = 2
    cfg.selfplay.sims_per_move = 2
    cfg.selfplay.max_batch = 4
    cfg.selfplay.max_recorded_moves = 64
    cfg.selfplay.max_tree_nodes = 256
    model, _, _ = build_objects(cfg, torch.device("cpu"))
    model.eval()
    rng = np.random.default_rng(12345)
    obs = rng.standard_normal((7, dz.OBS_SIZE), dtype=np.float32)
    masks = np.ones((7, dz.ACTION_SPACE_SIZE), dtype=np.bool_)
    players = np.asarray([0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)

    fast_logits, fast_values = route_leaf_evaluations(
        (model, model),
        obs,
        masks,
        None,
        torch.device("cpu"),
        same_model_fast_path=True,
    )
    split_logits, split_values = route_leaf_evaluations(
        (model, model),
        obs,
        masks,
        players,
        torch.device("cpu"),
        same_model_fast_path=False,
    )
    np.testing.assert_array_equal(fast_logits, split_logits)
    np.testing.assert_array_equal(fast_values, split_values)

    fast_stats, fast_records = play_routed_games(
        (model, model),
        cfg.selfplay,
        seed=cfg.seed,
        device=torch.device("cpu"),
        target_games=2,
        same_model_fast_path=True,
    )
    split_stats, split_records = play_routed_games(
        (model, model),
        cfg.selfplay,
        seed=cfg.seed,
        device=torch.device("cpu"),
        target_games=2,
        same_model_fast_path=False,
    )
    assert fast_stats.routed_fast_path_batches > 0
    assert fast_stats.routed_split_batches == 0
    assert split_stats.routed_fast_path_batches == 0
    assert split_stats.routed_split_batches > 0
    assert len(fast_records) == len(split_records) == 2
    for fast_record, split_record in zip(fast_records, split_records):
        assert fast_record.keys() == split_record.keys()
        for field in ("observations", "policy_targets", "values"):
            np.testing.assert_array_equal(fast_record[field], split_record[field])


def test_per_seat_model_routing_uses_real_runner_leaf_seats(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, generations=1)
    cfg.selfplay.n_games = 2
    cfg.selfplay.sims_per_move = 2
    cfg.selfplay.max_batch = 4
    cfg.selfplay.max_tree_nodes = 256
    runner = dz.SelfPlayRunner(make_runner_config(cfg.selfplay, cfg.seed))
    models = (_TaggedEvaluator(7.0), _TaggedEvaluator(-4.0))
    seen: set[int] = set()
    for _ in range(200):
        obs, masks = runner.collect_leaves(cfg.selfplay.max_batch)
        if obs.shape[0] == 0:
            continue
        players = runner.leaf_players()
        logits, values = route_leaf_evaluations(models, obs, masks, players, torch.device("cpu"))
        expected = np.where(np.asarray(players) == 0, 7.0, -4.0).astype(np.float32)
        np.testing.assert_array_equal(values, expected)
        runner.provide_evaluations(values, logits)
        seen.update(int(player) for player in players)
        if seen == {0, 1}:
            break
    assert seen == {0, 1}


def test_best_checkpoint_persists_across_candidate_resume(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, generations=1)
    candidate, optimizer, replay = build_objects(cfg, torch.device("cpu"))
    with torch.no_grad():
        for parameter in candidate.parameters():
            parameter.fill_(0.25)
    best_path = best_checkpoint_path(cfg.checkpoint_dir)
    save_best_checkpoint(cfg, 3, candidate, best_path)

    with torch.no_grad():
        for parameter in candidate.parameters():
            parameter.fill_(-0.5)
    candidate_checkpoint = save_checkpoint(cfg, 4, candidate, optimizer, replay, best_generation=3)
    payload = torch.load(candidate_checkpoint, map_location="cpu", weights_only=False)
    assert payload["best_generation"] == 3

    _, generation, resumed_candidate, _, _ = load_checkpoint(candidate_checkpoint, torch.device("cpu"))
    resumed_best, best_generation, destination = initialize_best_checkpoint(
        cfg,
        resumed_candidate,
        torch.device("cpu"),
        source_path=best_path,
        start_generation=generation,
    )
    assert destination == best_path
    assert best_generation == 3
    assert generation == 4
    best_state = resumed_best.state_dict()
    candidate_state = resumed_candidate.state_dict()
    assert any(not torch.equal(best_state[key], candidate_state[key]) for key in best_state)
    loaded_best, loaded_generation = load_best_checkpoint(cfg, torch.device("cpu"), best_path)
    assert loaded_generation == 3
    for key, value in loaded_best.state_dict().items():
        torch.testing.assert_close(value, best_state[key], rtol=0.0, atol=0.0)


def test_gated_training_resume_keeps_rejected_best(monkeypatch, tmp_path: Path) -> None:
    from . import train as train_module

    cfg = tiny_config(tmp_path, seed=6060, generations=1)
    cfg.selfplay.n_games = 2
    cfg.selfplay.games_per_generation = 2
    cfg.selfplay.sims_per_move = 2
    cfg.selfplay.max_batch = 4
    cfg.selfplay.max_tree_nodes = 256
    cfg.optim.batch_size = 8
    cfg.optim.train_steps_per_generation = 1
    cfg.gate_games = 2
    cfg.gate_sims = 1
    monkeypatch.setattr(train_module, "run_gate_match", lambda *_args, **_kwargs: GateStats(2, 0, 0))
    first = run_training(cfg)
    assert first["metrics"][0]["gate_result"] == "accepted"
    assert first["metrics"][0]["best_generation"] == 1
    best_file = Path(cfg.checkpoint_dir) / "best.pt"
    before = state_tensors(best_file)

    resume_cfg = tiny_config(tmp_path, seed=6060, generations=2)
    resume_cfg.selfplay.n_games = cfg.selfplay.n_games
    resume_cfg.selfplay.games_per_generation = cfg.selfplay.games_per_generation
    resume_cfg.selfplay.sims_per_move = cfg.selfplay.sims_per_move
    resume_cfg.selfplay.max_batch = cfg.selfplay.max_batch
    resume_cfg.selfplay.max_tree_nodes = cfg.selfplay.max_tree_nodes
    resume_cfg.optim.batch_size = cfg.optim.batch_size
    resume_cfg.optim.train_steps_per_generation = cfg.optim.train_steps_per_generation
    resume_cfg.gate_games = 2
    resume_cfg.gate_sims = 1
    monkeypatch.setattr(train_module, "run_gate_match", lambda *_args, **_kwargs: GateStats(0, 2, 0))
    resumed = run_training(resume_cfg, resume=str(Path(cfg.checkpoint_dir) / "gen_0001.pt"))
    assert resumed["metrics"][0]["gate_result"] == "rejected"
    assert resumed["metrics"][0]["best_generation"] == 1
    after = state_tensors(best_file)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0.0, atol=0.0)


def test_parallel_cpu_gated_pool_uses_archived_best_after_warmup(monkeypatch, tmp_path: Path) -> None:
    from . import train as train_module

    cfg = tiny_config(tmp_path, seed=7070, generations=2)
    cfg.parallel_workers = 2
    cfg.worker_device = "cpu"
    cfg.model.hidden_sizes = [16]
    cfg.selfplay.n_games = 1
    cfg.selfplay.games_per_generation = 2
    cfg.selfplay.sims_per_move = 2
    cfg.selfplay.max_batch = 4
    cfg.selfplay.max_recorded_moves = 64
    cfg.selfplay.max_tree_nodes = 256
    cfg.optim.batch_size = 8
    cfg.optim.train_steps_per_generation = 1
    cfg.replay.capacity = 512
    cfg.gate_games = 2
    cfg.gate_sims = 1
    cfg.gate_warmup_generations = 1
    cfg.league_fraction = 0.5
    cfg.league_pool_size = 2
    real_gate_match = train_module.run_gate_match
    gate_calls: list[int] = []

    def counted_gate_match(*args, **kwargs):
        gate_calls.append(1)
        return real_gate_match(*args, **kwargs)

    monkeypatch.setattr(train_module, "run_gate_match", counted_gate_match)

    result = run_training(cfg)
    assert result["metrics"][0]["gate_result"] == "warmup_accepted"
    assert result["metrics"][1]["gate_result"] in {"accepted", "rejected"}
    assert len(gate_calls) == 1
    assert (Path(cfg.checkpoint_dir) / "league" / "best_0000.pt").exists()
    assert [row["games"] for row in result["metrics"]] == [2, 2]
    assert [row["league_games"] for row in result["metrics"]] == [0, 1]
    assert result["metrics"][0]["routed_fast_path_batches"] > 0
    assert result["metrics"][0]["routed_split_batches"] == 0
    assert result["metrics"][1]["routed_fast_path_batches"] > 0
    assert result["metrics"][1]["routed_split_batches"] > 0
    csv_rows = read_metrics(Path(cfg.metrics_csv))
    assert int(csv_rows[0]["routed_fast_path_batches"]) > 0
    assert int(csv_rows[0]["routed_split_batches"]) == 0
    assert int(csv_rows[1]["routed_fast_path_batches"]) > 0
    assert int(csv_rows[1]["routed_split_batches"]) > 0
    assert all(float(row["aggregate_games_per_hour"]) > 0.0 for row in result["metrics"])


def test_temperature_sampled_gate_match_is_seeded_and_noise_free(monkeypatch, tmp_path: Path) -> None:
    """Temperature sampling remains reproducible for a fixed match seed."""
    from . import selfplay as selfplay_module

    cfg = tiny_config(tmp_path, seed=8181, generations=1)
    cfg.model.hidden_sizes = [16]
    cfg.gate_games = 2
    cfg.gate_sims = 1
    cfg.gate_temp_moves = 8
    cfg.selfplay.n_games = 1
    cfg.selfplay.max_batch = 4
    cfg.selfplay.max_recorded_moves = 64
    cfg.selfplay.max_tree_nodes = 256
    candidate, _, _ = build_objects(cfg, torch.device("cpu"))
    best, _, _ = build_objects(cfg, torch.device("cpu"))

    # The engine's normal temperature sampler is deterministic when its seed
    # is fixed, unlike temp-zero argmax behavior per position.
    first = run_gate_match(candidate, best, cfg, generation=3, device=torch.device("cpu"))
    second = run_gate_match(candidate, best, cfg, generation=3, device=torch.device("cpu"))
    assert first == second

    # Inspect the cloned gate config at the self-play boundary: gate sampling
    # uses the requested first-move window while Dirichlet exploration stays
    # disabled for every gate game.
    captured: list[tuple[int, float, int]] = []

    def capture_play(_models, gate_config, *, seed, device, target_games):
        del device
        captured.append((gate_config.temp_moves, gate_config.dirichlet_frac, seed))
        return None, [{"winner": 0} for _ in range(target_games)]

    monkeypatch.setattr(selfplay_module, "play_routed_games", capture_play)
    replayed = run_gate_match(candidate, best, cfg, generation=3, device=torch.device("cpu"))
    assert replayed == GateStats(wins=1, losses=1, ties=0)
    assert [(temp_moves, noise) for temp_moves, noise, _seed in captured] == [(8, 0.0), (8, 0.0)]
    assert captured[1][2] - captured[0][2] == 0x10001


def _small_gated_config(tmp_path: Path, *, generations: int) -> TrainConfig:
    cfg = tiny_config(tmp_path, seed=9292, generations=generations)
    cfg.model.hidden_sizes = [16]
    cfg.selfplay.n_games = 1
    cfg.selfplay.games_per_generation = 1
    cfg.selfplay.sims_per_move = 1
    cfg.selfplay.max_batch = 4
    cfg.selfplay.max_recorded_moves = 64
    cfg.selfplay.max_tree_nodes = 256
    cfg.optim.batch_size = 8
    cfg.optim.train_steps_per_generation = 1
    cfg.replay.capacity = 512
    cfg.gate_games = 2
    cfg.gate_sims = 1
    return cfg


def test_league_seed_checkpoint_is_available_and_sampled_from_generation_one(monkeypatch, tmp_path: Path) -> None:
    from . import train as train_module

    source_cfg = _small_gated_config(tmp_path / "source", generations=1)
    source_model, source_optimizer, source_replay = build_objects(source_cfg, torch.device("cpu"))
    with torch.no_grad():
        for parameter in source_model.parameters():
            parameter.fill_(0.125)
    source_checkpoint = save_checkpoint(source_cfg, 25, source_model, source_optimizer, source_replay)

    cfg = _small_gated_config(tmp_path / "target", generations=1)
    cfg.league_fraction = 1.0
    cfg.league_seed_checkpoints = [str(source_checkpoint)]
    monkeypatch.setattr(train_module, "run_gate_match", lambda *_args, **_kwargs: GateStats(0, 2, 0))

    result = run_training(cfg)
    seed_path = Path(cfg.checkpoint_dir) / "league" / "seed_0.pt"
    assert seed_path.exists()
    assert league_checkpoint_paths(cfg) == [seed_path]
    assert all(game is not None and game.opponent_index == 0 for game in sample_league_games(4, 1.0, 1, seed=17))
    assert result["metrics"][0]["league_games"] == 1
    seeded = state_tensors(seed_path)
    source = state_tensors(source_checkpoint)
    for key in source:
        torch.testing.assert_close(seeded[key], source[key], rtol=0.0, atol=0.0)


def test_league_seed_width_mismatch_raises_hard_clear_error(tmp_path: Path) -> None:
    source_cfg = _small_gated_config(tmp_path / "source", generations=1)
    source_model, source_optimizer, source_replay = build_objects(source_cfg, torch.device("cpu"))
    source_checkpoint = save_checkpoint(source_cfg, 1, source_model, source_optimizer, source_replay)

    target_cfg = _small_gated_config(tmp_path / "target", generations=1)
    target_cfg.model.hidden_sizes = [32]
    target_cfg.league_seed_checkpoints = [str(source_checkpoint)]
    with pytest.raises(ValueError, match=r"league seed checkpoint .*hidden_sizes \[16\].*current run requires \[32\]"):
        seed_league_checkpoints(target_cfg, torch.device("cpu"))


def test_league_seed_observation_version_mismatch_raises_hard_error(tmp_path: Path) -> None:
    source_cfg = _small_gated_config(tmp_path / "source", generations=1)
    source_cfg.selfplay.obs_version = 1
    source_model, source_optimizer, source_replay = build_objects(source_cfg, torch.device("cpu"))
    source_checkpoint = save_checkpoint(source_cfg, 1, source_model, source_optimizer, source_replay)

    target_cfg = _small_gated_config(tmp_path / "target", generations=1)
    target_cfg.selfplay.obs_version = 2
    target_cfg.league_seed_checkpoints = [str(source_checkpoint)]
    with pytest.raises(ValueError, match=r"stored obs_version 1 and input width .*current run requires obs_version 2"):
        seed_league_checkpoints(target_cfg, torch.device("cpu"))


def _card_transformer_config(tmp_path: Path, *, obs_version: int) -> TrainConfig:
    config = _small_gated_config(tmp_path, generations=1)
    config.model.arch = "card_transformer"
    config.model.obs_version = obs_version
    config.model.d_model = 16
    config.model.n_layers = 1
    config.model.n_heads = 4
    config.model.ffn_multiplier = 1
    config.model.dropout = 0.0
    config.selfplay.obs_version = obs_version
    return config


def test_league_observation_validation_allows_only_v3_to_v2_downgrade(tmp_path: Path) -> None:
    """A v2 CardTokenNet is a v3 league opponent, never an upgraded model."""
    v2_config = _card_transformer_config(tmp_path / "v2", obs_version=2)
    v2_model, v2_optimizer, v2_replay = build_objects(v2_config, torch.device("cpu"))
    v2_checkpoint = save_checkpoint(v2_config, 1, v2_model, v2_optimizer, v2_replay)

    v3_target = _card_transformer_config(tmp_path / "v3-target", obs_version=3)
    v3_target.league_seed_checkpoints = [str(v2_checkpoint)]
    copied = seed_league_checkpoints(v3_target, torch.device("cpu"))
    loaded_v2, _ = load_best_checkpoint(v3_target, torch.device("cpu"), copied[0])
    assert loaded_v2.obs_version == 2
    assert loaded_v2._dominion_model_config["obs_version"] == 2

    v1_config = _small_gated_config(tmp_path / "v1", generations=1)
    v1_config.selfplay.obs_version = 1
    v1_model, v1_optimizer, v1_replay = build_objects(v1_config, torch.device("cpu"))
    v1_checkpoint = save_checkpoint(v1_config, 1, v1_model, v1_optimizer, v1_replay)
    v3_target.league_seed_checkpoints = [str(v1_checkpoint)]
    with pytest.raises(ValueError, match=r"obs-v1 remains incompatible"):
        seed_league_checkpoints(v3_target, torch.device("cpu"))

    v3_source = _card_transformer_config(tmp_path / "v3-source", obs_version=3)
    v3_model, v3_optimizer, v3_replay = build_objects(v3_source, torch.device("cpu"))
    v3_checkpoint = save_checkpoint(v3_source, 1, v3_model, v3_optimizer, v3_replay)
    v2_target = _card_transformer_config(tmp_path / "v2-target", obs_version=2)
    v2_target.league_seed_checkpoints = [str(v3_checkpoint)]
    with pytest.raises(ValueError, match=r"no observation upgrade path exists"):
        seed_league_checkpoints(v2_target, torch.device("cpu"))


def test_v3_league_loads_real_legacy_checkpoint_with_generation_metadata(tmp_path: Path) -> None:
    """League routing can apply the Python generation-1 constant restoration."""
    checkpoint = Path(__file__).resolve().parents[3] / "checkpoints/remote/campaign15/gen_0045.pt"
    assert checkpoint.is_file(), f"required real league checkpoint is missing: {checkpoint}"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "encoder_generation" not in payload

    config = _card_transformer_config(tmp_path / "campaign18-smoke", obs_version=3)
    config.league_seed_checkpoints = [str(checkpoint)]
    copied = seed_league_checkpoints(config, torch.device("cpu"))
    model, _ = load_best_checkpoint(config, torch.device("cpu"), copied[0])
    assert model._dominion_encoder_generation == 1
    assert model._dominion_model_config["encoder_generation"] == 1


def test_ungated_league_schedule_uses_exact_per_generation_counts(tmp_path: Path) -> None:
    source_cfg = _small_gated_config(tmp_path / "source", generations=1)
    source_model, source_optimizer, source_replay = build_objects(source_cfg, torch.device("cpu"))
    source_checkpoint = save_checkpoint(source_cfg, 1, source_model, source_optimizer, source_replay)

    cfg = _small_gated_config(tmp_path / "target", generations=2)
    cfg.gate_games = 0
    cfg.selfplay.n_games = 2
    cfg.selfplay.games_per_generation = 4
    cfg.selfplay.sims_per_move = 2
    cfg.league_seed_checkpoints = [str(source_checkpoint)]
    cfg.league_schedule = [[1, 0.25], [2, 0.75]]

    result = run_training(cfg)
    assert [row["league_games"] for row in result["metrics"]] == [1, 3]
    csv_rows = read_metrics(Path(cfg.metrics_csv))
    assert [int(row["league_games_seed_0.pt"]) for row in csv_rows] == [1, 3]
    assert all("gate_result" not in row for row in result["metrics"])


def test_periodic_self_checkpoints_are_fifo_capped_without_evicting_seeds(tmp_path: Path) -> None:
    cfg = _small_gated_config(tmp_path, generations=1)
    cfg.league_pool_size = 2
    cfg.league_self_every = 1
    source_model, source_optimizer, source_replay = build_objects(cfg, torch.device("cpu"))
    seeds = tmp_path / "seed.pt"
    torch.save(
        {
            "generation": 0,
            "config": cfg.to_dict(),
            "encoder_generation": int(dz.ENCODER_GENERATION),
            "model": source_model.state_dict(),
        },
        seeds,
    )
    cfg.league_seed_checkpoints = [str(seeds)]
    seed_league_checkpoints(cfg, torch.device("cpu"))

    for generation in range(1, 4):
        checkpoint = save_checkpoint(cfg, generation, source_model, source_optimizer, source_replay)
        archive_self_checkpoint(cfg, checkpoint, generation)

    assert [path.name for path in league_checkpoint_paths(cfg)] == ["seed_0.pt", "self_0002.pt", "self_0003.pt"]


def test_self_pool_fifo_orders_periodic_and_gated_history_by_addition_age(tmp_path: Path) -> None:
    cfg = _small_gated_config(tmp_path, generations=1)
    cfg.league_pool_size = 2
    cfg.league_self_every = 1
    model, optimizer, replay = build_objects(cfg, torch.device("cpu"))
    generation_one = save_checkpoint(cfg, 1, model, optimizer, replay)
    archive_self_checkpoint(cfg, generation_one, 1)
    save_best_checkpoint(cfg, 1, model, best_checkpoint_path(cfg.checkpoint_dir))
    archive_previous_best(cfg, best_checkpoint_path(cfg.checkpoint_dir), 1)
    generation_two = save_checkpoint(cfg, 2, model, optimizer, replay)
    archive_self_checkpoint(cfg, generation_two, 2)

    assert [path.name for path in league_checkpoint_paths(cfg)] == ["best_0001.pt", "self_0002.pt"]


def test_gate_force_accepts_only_after_configured_stale_generations(monkeypatch, tmp_path: Path) -> None:
    from . import train as train_module

    cfg = _small_gated_config(tmp_path, generations=3)
    cfg.gate_force_accept_every = 2
    monkeypatch.setattr(train_module, "run_gate_match", lambda *_args, **_kwargs: GateStats(0, 2, 0))

    result = run_training(cfg)
    rows = result["metrics"]
    assert [row["gate_result"] for row in rows] == ["rejected", "rejected", "forced_accepted"]
    assert [row["best_generation"] for row in rows] == [0, 0, 3]
    csv_rows = read_metrics(Path(cfg.metrics_csv))
    assert [row["gate_result"] for row in csv_rows] == ["rejected", "rejected", "forced_accepted"]
