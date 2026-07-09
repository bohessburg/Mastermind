from __future__ import annotations

from pathlib import Path

from .evaluate import evaluate_checkpoint
from .test_train_smoke import read_metrics, tiny_config
from .train import run_training


def test_checkpoint_eval_vs_random_smoke_and_determinism(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, seed=9090, generations=1)
    cfg.selfplay.n_games = 4
    cfg.selfplay.games_per_generation = 4
    cfg.selfplay.sims_per_move = 8
    cfg.selfplay.max_batch = 16
    cfg.optim.train_steps_per_generation = 1
    cfg.replay.capacity = 1024
    run_training(cfg)
    checkpoint = Path(cfg.checkpoint_dir) / "gen_0001.pt"

    first = evaluate_checkpoint(
        checkpoint,
        opponent="random",
        games=8,
        sims=8,
        kingdoms="fixed",
        seed=12345,
        device_name="cpu",
        n_games=4,
        max_batch=16,
    )
    second = evaluate_checkpoint(
        checkpoint,
        opponent="random",
        games=8,
        sims=8,
        kingdoms="fixed",
        seed=12345,
        device_name="cpu",
        n_games=4,
        max_batch=16,
    )

    assert first.games == 8
    assert first.games == first.wins + first.losses + first.ties
    assert first.wins == second.wins
    assert first.losses == second.losses
    assert first.ties == second.ties
    assert first.truncated == second.truncated


def test_training_metrics_include_periodic_eval(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, seed=6060, generations=1)
    cfg.selfplay.n_games = 2
    cfg.selfplay.games_per_generation = 2
    cfg.selfplay.sims_per_move = 4
    cfg.selfplay.max_batch = 8
    cfg.optim.train_steps_per_generation = 0
    cfg.eval.eval_every_n_generations = 1
    cfg.eval.eval_games = 4
    cfg.eval.eval_sims = 4
    cfg.eval.eval_opponent = "random"
    cfg.eval.eval_kingdoms = "fixed"
    cfg.eval.eval_n_games = 2
    cfg.eval.eval_max_batch = 8

    run_training(cfg)
    rows = read_metrics(Path(cfg.metrics_csv))
    assert len(rows) == 1
    assert rows[0]["eval_opponent"] == "random"
    assert int(rows[0]["eval_games"]) == 4
    total = int(rows[0]["eval_wins"]) + int(rows[0]["eval_losses"]) + int(rows[0]["eval_ties"])
    assert total == 4
