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
