from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch

import dominion_v2_py as dz

from .gating import (
    GateStats,
    best_checkpoint_path,
    gate_result,
    initialize_best_checkpoint,
    load_best_checkpoint,
    sample_league_games,
    save_best_checkpoint,
)
from .selfplay import make_runner_config, route_leaf_evaluations
from .test_train_smoke import state_tensors, tiny_config
from .train import build_objects, load_checkpoint, run_training, save_checkpoint


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
    assert abs(sum(game.best_player == 0 for game in sampled) - 100) <= 1


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


def test_parallel_cpu_league_uses_archived_best(monkeypatch, tmp_path: Path) -> None:
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
    cfg.league_fraction = 0.5
    cfg.league_pool_size = 2
    outcomes = iter((GateStats(2, 0, 0), GateStats(0, 2, 0)))
    monkeypatch.setattr(train_module, "run_gate_match", lambda *_args, **_kwargs: next(outcomes))

    result = run_training(cfg)
    assert [row["gate_result"] for row in result["metrics"]] == ["accepted", "rejected"]
    assert (Path(cfg.checkpoint_dir) / "league" / "best_0000.pt").exists()
    assert [row["games"] for row in result["metrics"]] == [2, 2]
