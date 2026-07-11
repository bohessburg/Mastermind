from __future__ import annotations

from pathlib import Path

import pytest
import torch

import dominion_v2_py as dz

from . import evaluate
from .evaluate import classify_game_end, evaluate_checkpoint
from .test_train_smoke import read_metrics, tiny_config
from .train import build_objects, run_training, save_checkpoint


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
    assert first.end_province == second.end_province
    assert first.end_piles == second.end_piles
    assert first.end_trunc == second.end_trunc
    assert first.end_province + first.end_piles + first.end_trunc == first.games


class _FinishedGame:
    def __init__(self, supply: list[tuple[int, int]], truncated: bool = False) -> None:
        self._supply = supply
        self._truncated = truncated

    def supply(self) -> list[tuple[int, int]]:
        return self._supply

    def truncated(self) -> bool:
        return self._truncated


def test_eval_end_forensics_classify_supply_and_turn_cap() -> None:
    province = _FinishedGame([(dz.DEF_PROVINCE, 0), (dz.DEF_DUCHY, 4), (dz.DEF_ESTATE, 4)])
    piles = _FinishedGame([(dz.DEF_PROVINCE, 4), (dz.DEF_DUCHY, 0), (dz.DEF_ESTATE, 0), (dz.DEF_CURSE, 0)])
    trunc = _FinishedGame([(dz.DEF_PROVINCE, 0)], truncated=True)

    assert classify_game_end(province) == "province"
    assert classify_game_end(piles) == "piles"
    assert classify_game_end(trunc) == "trunc"
    with pytest.raises(ValueError, match="recognized end condition"):
        classify_game_end(_FinishedGame([(dz.DEF_PROVINCE, 1), (dz.DEF_DUCHY, 1)]))


def test_checkpoint_eval_vs_phase6_mcts_smoke(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, seed=9191, generations=1)
    cfg.selfplay.n_games = 2
    cfg.selfplay.games_per_generation = 2
    cfg.selfplay.sims_per_move = 4
    cfg.selfplay.max_batch = 8
    cfg.optim.train_steps_per_generation = 0
    cfg.replay.capacity = 1024
    run_training(cfg)
    checkpoint = Path(cfg.checkpoint_dir) / "gen_0001.pt"

    stats = evaluate_checkpoint(
        checkpoint,
        opponent="mcts",
        games=2,
        sims=4,
        kingdoms="fixed",
        seed=54321,
        device_name="cpu",
        n_games=1,
        max_batch=8,
    )

    assert stats.games == 2
    assert stats.games == stats.wins + stats.losses + stats.ties
    assert stats.end_province + stats.end_piles + stats.end_trunc == stats.games


def test_checkpoint_eval_inherits_treasure_collapse_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tiny_config(tmp_path, generations=1)
    cfg.selfplay.auto_play_treasures = True
    cfg.selfplay.prune_treasure_plays = True
    model, optimizer, replay = build_objects(cfg, torch.device("cpu"))
    checkpoint = save_checkpoint(cfg, 1, model, optimizer, replay)
    captured: dict[str, object] = {}
    sentinel = object()

    def capture_eval(_model: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(evaluate, "evaluate_model", capture_eval)

    assert evaluate.evaluate_checkpoint(checkpoint, device_name="cpu") is sentinel
    assert captured["auto_play_treasures"] is True
    assert captured["prune_treasure_plays"] is True


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
    assert int(rows[0]["eval_end_province"]) + int(rows[0]["eval_end_piles"]) + int(rows[0]["eval_end_trunc"]) == 4
