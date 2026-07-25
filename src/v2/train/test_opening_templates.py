from __future__ import annotations

import json

import pytest

import dominion_v2_py as dz

from .config import (
    TrainConfig,
    load_config,
    opening_template_schedule,
    save_config,
    scheduled_opening_selfplay_config,
)
from .selfplay import make_runner_config
from .train import run_training


def test_opening_template_config_round_trip_and_native_plumbing(tmp_path) -> None:
    config = TrainConfig()
    config.selfplay.opening_templates_enabled = True
    config.selfplay.opening_lambda = 0.42
    config.selfplay.opening_turn_window = 11
    config.selfplay.template_weights = [0.25, 0.15, 0.10, 0.10, 0.10, 0.15, 0.15]
    config.selfplay.opening_lambda_initial = 0.6
    config.selfplay.opening_lambda_final = 0.1
    config.selfplay.opening_anneal_gens = 20
    config.selfplay.opening_p_unconstrained_initial = 0.3
    config.selfplay.opening_p_unconstrained_final = 0.9

    path = tmp_path / "opening_templates.json"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.selfplay.opening_templates_enabled is True
    assert loaded.selfplay.opening_lambda == pytest.approx(0.42)
    assert loaded.selfplay.opening_turn_window == 11
    assert loaded.selfplay.template_weights == pytest.approx(config.selfplay.template_weights)
    assert loaded.selfplay.opening_anneal_gens == 20

    native = make_runner_config(loaded.selfplay, 0xC18)
    assert native.opening_templates_enabled is True
    assert native.opening_lambda == pytest.approx(0.42)
    assert native.opening_turn_window == 11
    assert native.template_weights == pytest.approx(config.selfplay.template_weights)

    path.write_text(json.dumps({"selfplay": {"template_weights": [1.0, 0.0]}}))
    with pytest.raises(ValueError, match="template_weights"):
        load_config(path)


def test_opening_template_linear_schedule_and_generation_config() -> None:
    config = TrainConfig().selfplay
    config.opening_templates_enabled = True
    config.template_weights = [0.3, 0.5, 0.1, 0.1, 0.0, 0.0, 0.0]
    config.opening_lambda_initial = 0.6
    config.opening_lambda_final = 0.0
    config.opening_p_unconstrained_initial = 0.3
    config.opening_p_unconstrained_final = 1.0
    config.opening_anneal_gens = 10

    assert opening_template_schedule(config, 0) == pytest.approx((0.6, 0.3))
    assert opening_template_schedule(config, 5) == pytest.approx((0.3, 0.65))
    assert opening_template_schedule(config, 10) == pytest.approx((0.0, 1.0))
    assert opening_template_schedule(config, 100) == pytest.approx((0.0, 1.0))

    scheduled = scheduled_opening_selfplay_config(config, 5)
    assert scheduled is not config
    assert scheduled.opening_lambda == pytest.approx(0.3)
    assert scheduled.template_weights[0] == pytest.approx(0.65)
    assert sum(scheduled.template_weights[1:]) == pytest.approx(0.35)
    assert scheduled.template_weights[1:] == pytest.approx([0.25, 0.05, 0.05, 0.0, 0.0, 0.0])
    # The base campaign config remains declarative and checkpoint-safe.
    assert config.opening_lambda == pytest.approx(0.6)

    config.opening_anneal_gens = 0
    assert opening_template_schedule(config, 0) == pytest.approx((0.0, 1.0))


def test_native_opening_template_defaults_are_disabled() -> None:
    native = dz.SelfPlayConfig()
    assert native.opening_templates_enabled is False
    assert native.opening_turn_window == 8
    assert len(native.template_weights) == 7


def test_opening_template_metrics_are_emitted(tmp_path) -> None:
    config = TrainConfig()
    config.seed = 0xC18001
    config.generations = 1
    config.device = "cpu"
    config.checkpoint_dir = str(tmp_path / "checkpoints")
    config.metrics_csv = str(tmp_path / "checkpoints" / "metrics.csv")
    config.model.hidden_sizes = [8]
    config.selfplay.n_games = 2
    config.selfplay.games_per_generation = 2
    config.selfplay.sims_per_move = 4
    config.selfplay.max_batch = 8
    config.selfplay.dirichlet_frac = 0.0
    config.selfplay.temp_moves = 0
    config.selfplay.kingdom_mode = "fixed"
    config.selfplay.fixed_kingdom = ["Chapel"]
    config.selfplay.max_recorded_moves = 128
    config.selfplay.max_tree_nodes = 256
    config.selfplay.auto_play_treasures = True
    config.selfplay.prune_treasure_plays = True
    config.selfplay.opening_templates_enabled = True
    config.selfplay.opening_lambda_initial = 1.0
    config.selfplay.opening_lambda_final = 1.0
    config.selfplay.opening_p_unconstrained_initial = 0.0
    config.selfplay.opening_p_unconstrained_final = 0.0
    config.selfplay.opening_anneal_gens = 1
    config.selfplay.template_weights = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    config.optim.train_steps_per_generation = 0
    config.replay.capacity = 512

    result = run_training(config)
    row = result["metrics"][0]
    assert row["opening_templates_enabled"] is True
    assert row["opening_lambda"] == pytest.approx(1.0)
    assert row["opening_template_games_t1"] == 4
    assert sum(row[f"opening_template_games_t{template_id}"] for template_id in range(7)) == 4
    assert "mean_cards_trashed_per_game" in row
    assert "unconstrained_buys_chapel" in row
