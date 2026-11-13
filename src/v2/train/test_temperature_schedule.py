"""Decision-kind-aware self-play temperature configuration coverage."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import dominion_v2_py as dz

from .config import SelfPlayConfig, TrainConfig, load_config, save_config
from .model import build_model
from .observation import obs_size_for_config
from .selfplay import make_runner_config, play_routed_games


def test_temperature_schedule_config_round_trip_and_native_plumbing(tmp_path) -> None:
    config = TrainConfig()
    config.selfplay.temp_moves = 23
    config.selfplay.temp_mode = "per_seat_buy"
    config.selfplay.temp_buy_turns = 17
    config.selfplay.temp_action_plies = 11
    config.selfplay.temp_effect_plies = 7
    config.selfplay.temp_final = 0.25

    path = tmp_path / "temperature.json"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.selfplay.temp_moves == 23
    assert loaded.selfplay.temp_mode == "per_seat_buy"
    assert loaded.selfplay.temp_buy_turns == 17
    assert loaded.selfplay.temp_action_plies == 11
    assert loaded.selfplay.temp_effect_plies == 7
    assert loaded.selfplay.temp_final == pytest.approx(0.25)

    native = make_runner_config(loaded.selfplay, 0xC20)
    assert native.temp_moves == 23
    assert native.temp_mode == "per_seat_buy"
    assert native.temp_buy_turns == 17
    assert native.temp_action_plies == 11
    assert native.temp_effect_plies == 7
    assert native.temp_final == pytest.approx(0.25)

    defaults = dz.SelfPlayConfig()
    assert defaults.temp_moves == 20
    assert defaults.temp_mode == "legacy"
    assert defaults.temp_buy_turns == 14
    assert defaults.temp_action_plies == 10
    assert defaults.temp_effect_plies == 6
    assert defaults.temp_final == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("temp_mode", "per_player", "temp_mode"),
        ("temp_moves", -1, "temp_moves"),
        ("temp_buy_turns", -1, "temp_buy_turns"),
        ("temp_action_plies", -1, "temp_action_plies"),
        ("temp_effect_plies", -1, "temp_effect_plies"),
        ("temp_final", -0.1, "temp_final"),
    ],
)
def test_temperature_schedule_config_rejects_invalid_values(
    tmp_path,
    field: str,
    value: object,
    message: str,
) -> None:
    path = tmp_path / "invalid_temperature.json"
    path.write_text(json.dumps({"selfplay": {field: value}}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_per_seat_buy_selfplay_smoke_completes() -> None:
    selfplay = SelfPlayConfig(
        n_games=1,
        games_per_generation=1,
        sims_per_move=2,
        max_batch=8,
        dirichlet_frac=0.0,
        temp_mode="per_seat_buy",
        kingdom_mode="fixed",
        fixed_kingdom=["Village"],
        max_recorded_moves=128,
        max_tree_nodes=256,
        auto_play_treasures=True,
        prune_treasure_plays=True,
    )
    training = TrainConfig()
    training.model.hidden_sizes = [8]
    training.selfplay = selfplay
    torch.manual_seed(0xC20)
    model = build_model(training.model, obs_size_for_config(training), dz.ACTION_SPACE_SIZE)

    stats, records = play_routed_games(
        (model, model),
        selfplay,
        seed=0xC200001,
        device=torch.device("cpu"),
        target_games=1,
    )

    assert stats.games == 1
    assert stats.positions > 0
    assert len(records) == 1
    policy = records[0]["policy_targets"]
    assert policy.shape[0] > 0
    assert np.isfinite(policy).all()
    np.testing.assert_allclose(policy.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-5)
