from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import dominion_v2_py as dz
from src.v2.arena.bot.policy import NNCheckpointError, _legacy_obs_transform, load_policy
from src.v2.arena.config import ArenaConfig
from src.v2.arena.main import _load_policy_or_raise
from src.v2.train.config import TrainConfig
from src.v2.train.train import build_objects, load_full_checkpoint, save_checkpoint


def test_load_policy_rejects_missing_checkpoint(tmp_path: Path) -> None:
    try:
        load_policy(tmp_path / "missing-policy.pt")
    except NNCheckpointError as error:
        assert str(error) == "neural-network checkpoint is unavailable"
    else:
        raise AssertionError("missing checkpoint unexpectedly loaded")


def _checkpoint(tmp_path: Path, *, obs_version: int = 2) -> Path:
    config = TrainConfig()
    config.device = "cpu"
    config.checkpoint_dir = str(tmp_path / "checkpoints")
    config.metrics_csv = str(tmp_path / "checkpoints" / "metrics.csv")
    config.model.hidden_sizes = [8]
    config.selfplay.obs_version = obs_version
    model, optimizer, replay = build_objects(config, torch.device("cpu"))
    return save_checkpoint(config, 1, model, optimizer, replay)


def test_legacy_shim_changes_exactly_the_native_layout_derived_positions() -> None:
    game = dz.new_game(
        dz.Setup(players=2, kingdom=["Village", "Smithy", "Market", "Chapel"]),
        0xEC0D_0001,
    )
    for version in (1, 2, 3):
        observation = game.encode(0, version)
        transformed = _legacy_obs_transform(version)(observation)
        layout = dz.encoder_layout(version)

        expected = set(
            range(
                int(layout["landscape_offset"]),
                int(layout["landscape_offset"]) + int(layout["landscape_id_size"]),
            )
        )
        expected.add(
            int(layout["landscape_offset"]) + int(layout["landscape_prophecy_offset"])
        )
        supply_offset = int(layout["supply_offset"])
        block_size = int(layout["pile_block_size"])
        for pile in range(int(layout["supply_size"]) // block_size):
            row = supply_offset + (pile * block_size)
            populated = (
                observation[row + int(layout["pile_count_field"])] != 0.0
                or observation[row + int(layout["pile_base_field"])] != 0.0
            )
            if populated:
                expected.add(row + int(layout["pile_trait_field"]))

        changed = set(np.flatnonzero(transformed != observation).tolist())
        assert changed == expected
        assert np.all(transformed[list(expected)] == 1.0)
        np.testing.assert_array_equal(observation, game.encode(0, version))


def test_load_policy_requires_explicit_legacy_shim_for_unstamped_checkpoint(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    payload = load_full_checkpoint(checkpoint, "cpu")
    del payload["encoder_generation"]
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)

    with pytest.raises(
        NNCheckpointError,
        match=r"encoder generation 1.*encoder generation 2.*legacy_shim=True",
    ):
        load_policy(legacy)

    policy = load_policy(legacy, legacy_shim=True)
    assert policy.encoder_generation == 1
    assert policy.obs_transform is not None


def test_stamped_current_checkpoint_loads_without_shim(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    payload = load_full_checkpoint(checkpoint, "cpu")
    assert payload["encoder_generation"] == int(dz.ENCODER_GENERATION)

    policy = load_policy(checkpoint)
    assert policy.encoder_generation == int(dz.ENCODER_GENERATION)
    assert policy.obs_transform is None


def test_arena_policy_loader_auto_shims_legacy_checkpoint(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    payload = load_full_checkpoint(checkpoint, "cpu")
    del payload["encoder_generation"]
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    caplog.set_level("WARNING", logger="src.v2.arena.bot.policy")

    policy = _load_policy_or_raise(
        ArenaConfig(checkpoint_path=legacy, obs_version=2)
    )

    assert policy.encoder_generation == 1
    assert policy.obs_transform is not None
    expected = (
        f"legacy checkpoint {legacy} (encoder generation 1) served via "
        "compatibility shim on generation 2 engine"
    )
    assert [record.getMessage() for record in caplog.records].count(expected) == 1


def test_policy_searches_on_the_terminal_scale_the_checkpoint_was_trained_on(tmp_path: Path) -> None:
    plain = load_policy(_checkpoint(tmp_path / "plain"))
    assert (plain.terminal_margin_scale, plain.terminal_margin_alpha) == (0.0, 0.0)

    config = TrainConfig()
    config.device = "cpu"
    config.checkpoint_dir = str(tmp_path / "margin")
    config.metrics_csv = str(tmp_path / "margin" / "metrics.csv")
    config.model.hidden_sizes = [8]
    config.selfplay.obs_version = 2
    config.selfplay.value_target = "margin"
    config.selfplay.scaled_terminal_values = True
    config.selfplay.margin_scale = 20.0
    model, optimizer, replay = build_objects(config, torch.device("cpu"))
    policy = load_policy(save_checkpoint(config, 1, model, optimizer, replay))
    assert (policy.terminal_margin_scale, policy.terminal_margin_alpha) == (20.0, 0.0)


def test_arena_search_hands_the_terminal_scale_to_the_native_searcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.v2.arena.bot import policy as policy_module

    policy = load_policy(_checkpoint(tmp_path))
    policy.terminal_margin_scale = 20.0
    policy.terminal_margin_alpha = 0.25
    seen: list[dict] = []
    real = dz.DecisionSearcher

    def capture(game, seat, config):
        seen.append(dict(config))
        return real(game, seat, config)

    monkeypatch.setattr(policy_module.dz, "DecisionSearcher", capture)
    kingdom = ["Village", "Smithy", "Market", "Festival", "Laboratory", "Cellar", "Chapel", "Militia", "Witch", "Moat"]
    game = dz.new_game(dz.Setup(players=2, kingdom=kingdom), 5)
    seat = int(game.current_decision()["player"])
    action = policy_module.choose_nnmcts_action(game, seat, policy, sims=4, determinizations=1)

    assert bool(game.legal_mask()[action])
    assert seen[0]["terminal_margin_scale"] == 20.0
    assert seen[0]["terminal_margin_alpha"] == 0.25
