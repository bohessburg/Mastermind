from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from .diagnostics import (
    RAW_POLICY_SIMS,
    greedy_stats,
    match_selfplay_config,
    play_manifest_games,
    run_paired_match,
    zero_value_copy,
)
from .test_train_smoke import tiny_config
from .train import build_objects

CPU = torch.device("cpu")


def _model(tmp_path: Path, seed: int = 3):
    cfg = tiny_config(tmp_path)
    cfg.selfplay.max_turns = 24
    torch.manual_seed(seed)
    model, _, _ = build_objects(cfg, CPU)
    model.eval()
    return cfg, model


def test_match_config_is_greedy_and_fresh_tree_by_default(tmp_path: Path) -> None:
    cfg, _ = _model(tmp_path)
    cfg.selfplay.temp_mode = "per_seat_buy"
    cfg.selfplay.tree_reuse = True
    cfg.selfplay.dirichlet_frac = 0.25
    clean = match_selfplay_config(cfg.selfplay, sims=16)
    assert (clean.temp_mode, clean.temp_moves, clean.temp_final) == ("legacy", 0, 0.0)
    assert clean.tree_reuse is False and clean.dirichlet_frac == 0.0 and clean.sims_per_move == 16
    legacy = match_selfplay_config(cfg.selfplay, sims=16, greedy=False, tree_reuse=True)
    assert legacy.temp_mode == "per_seat_buy" and legacy.tree_reuse is True
    # The caller's config is never modified.
    assert cfg.selfplay.tree_reuse is True and cfg.selfplay.dirichlet_frac == 0.25


def test_manifest_games_are_played_once_and_replay_identically(tmp_path: Path) -> None:
    cfg, model = _model(tmp_path)
    config = match_selfplay_config(cfg.selfplay, sims=8)
    first, evals = play_manifest_games((model, model), config, seed=11, device=CPU, game_indices=[5, 2, 9])
    again, _ = play_manifest_games((model, model), config, seed=11, device=CPU, game_indices=[9, 5, 2])
    assert [int(r["game_index"]) for r in first] == [2, 5, 9]
    assert evals > 0
    for a, b in zip(first, again):
        assert int(a["seed"]) == int(b["seed"])
        np.testing.assert_array_equal(a["sampled_actions"], b["sampled_actions"])
    assert greedy_stats(first)["non_greedy"] == 0


def test_training_temperature_schedule_is_not_greedy(tmp_path: Path) -> None:
    """The confound: per_seat_buy ignores temp_moves = 0 and samples openings."""
    cfg, model = _model(tmp_path)
    # A freshly initialised net has a prior peaked enough that 16 sims all land
    # on one child, which leaves nothing to sample. Zero weights give a uniform
    # prior, so visits spread and the temperature decides the move.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    cfg.selfplay.temp_mode = "per_seat_buy"
    cfg.selfplay.temp_buy_turns = 6
    sampled = match_selfplay_config(cfg.selfplay, sims=16, greedy=False)
    assert sampled.temp_moves == 0
    records, _ = play_manifest_games(
        (model, model), sampled, seed=4, device=CPU, game_indices=range(4), independent_search=False
    )
    assert greedy_stats(records)["non_greedy"] > 0
    greedy, _ = play_manifest_games(
        (model, model),
        match_selfplay_config(cfg.selfplay, sims=16),
        seed=4,
        device=CPU,
        game_indices=range(4),
        independent_search=False,
    )
    assert greedy_stats(greedy)["non_greedy"] == 0


def test_identical_models_mirror_every_pair(tmp_path: Path) -> None:
    """Routing check: a distinct copy of the same weights must split each pair."""
    cfg, model = _model(tmp_path)
    twin = copy.deepcopy(model)
    config = match_selfplay_config(cfg.selfplay, sims=8)
    match = run_paired_match(model, twin, config, pairs=4, seed=7, device=CPU, chunk=3)
    summary = match.summary()
    assert summary["pairs"] == 4 and summary["games"] == 8
    for seat0, seat1 in zip(match.a_as_seat0, match.a_as_seat1):
        assert seat0 + seat1 == pytest.approx(1.0)
    assert summary["a_score"] == pytest.approx(0.5)
    assert summary["paired_se"] == pytest.approx(0.0)
    assert summary["non_greedy_pct"] == 0.0


def test_raw_policy_side_plays_the_prior_argmax(tmp_path: Path) -> None:
    cfg, model = _model(tmp_path)
    other = zero_value_copy(model)
    config = match_selfplay_config(cfg.selfplay, sims=16)
    match = run_paired_match(
        model, other, config, pairs=2, seed=9, device=CPU, sims_a=16, sims_b=RAW_POLICY_SIMS
    )
    agree, total = match.raw_agree["b"]
    assert total > 0 and agree == total
    assert "a" not in match.raw_agree


def test_zero_value_copy_only_changes_the_value(tmp_path: Path) -> None:
    _, model = _model(tmp_path)
    clone = zero_value_copy(model)
    obs = torch.randn(5, next(model.parameters()).shape[1])
    with torch.no_grad():
        logits, values = model(obs)[:2]
        zero_logits, zero_values = clone(obs)[:2]
    torch.testing.assert_close(logits, zero_logits)
    assert torch.count_nonzero(zero_values) == 0 and torch.count_nonzero(values) > 0


def test_independent_search_rejects_tree_reuse(tmp_path: Path) -> None:
    cfg, model = _model(tmp_path)
    config = match_selfplay_config(cfg.selfplay, sims=8, tree_reuse=True)
    with pytest.raises(ValueError, match="tree_reuse"):
        play_manifest_games((model, model), config, seed=1, device=CPU, game_indices=[0])
