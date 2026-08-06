from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import dominion_v2_py as dz

from src.v2.encoder_compat import legacy_constant_layout, restore_legacy_constants

from .config import load_config
from .observation import downgrade_v3_observations
from .selfplay import route_leaf_evaluations
from .train import validate_model_config


KINGDOM = [
    "Sentry",
    "Library",
    "Throne Room",
    "Bandit",
    "Witch",
    "Moat",
    "Village",
    "Smithy",
    "Market",
    "Remodel",
]


def test_v3_downgrade_is_the_exact_v2_encoding_from_live_game_state() -> None:
    """Exercise live binding encodes, not a synthetic prefix fixture."""
    game = dz.new_game(dz.Setup(players=2, kingdom=KINGDOM), 0xD06E6A0)
    for _ in range(3):
        v3 = game.encode(0, 3)[None, :]
        v2 = game.encode(0, 2)[None, :]

        downgraded_numpy = downgrade_v3_observations(v3)
        np.testing.assert_array_equal(downgraded_numpy, v2)
        assert v3[0, 0] == 3.0
        assert v3[0, 1] == float(dz.OBS_SIZE_V3)

        downgraded_torch = downgrade_v3_observations(torch.from_numpy(v3))
        torch.testing.assert_close(downgraded_torch, torch.from_numpy(v2), rtol=0.0, atol=0.0)

        legal = np.flatnonzero(game.legal_mask())
        assert legal.size > 0
        game.step(int(legal[0]))


def test_campaign18_config_loads_and_validates_its_v2_league_seed_list() -> None:
    config_path = Path(__file__).resolve().parents[3] / "configs" / "run_c18.json"
    config = load_config(config_path)
    validate_model_config(config)

    assert config.selfplay.obs_version == 3
    assert config.model.obs_version == 3
    assert config.league_seed_checkpoints == [
        "checkpoints/campaign15/gen_0045.pt",
        "checkpoints/campaign15/gen_0055.pt",
        "checkpoints/campaign16/gen_0030.pt",
        "checkpoints/campaign17/gen_0015.pt",
    ]


class _RecordingLeagueModel:
    def __init__(self, obs_version: int, encoder_generation: int) -> None:
        self._dominion_model_config = {"obs_version": obs_version}
        self._dominion_encoder_generation = encoder_generation
        self.seen: list[np.ndarray] = []

    def eval(self) -> _RecordingLeagueModel:
        return self

    def evaluate(self, obs: torch.Tensor, legal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen.append(obs.detach().cpu().numpy().copy())
        return (
            torch.zeros((obs.shape[0], legal_mask.shape[1]), dtype=obs.dtype, device=obs.device),
            torch.zeros((obs.shape[0],), dtype=obs.dtype, device=obs.device),
        )


def _generation_two_v3_batch() -> np.ndarray:
    observations = np.zeros((2, dz.OBS_SIZE_V3), dtype=np.float32)
    observations[:, 0] = 3.0
    observations[:, 1] = float(dz.OBS_SIZE_V3)
    layout = legacy_constant_layout(3)
    piles = observations[:, layout.supply_offset : layout.supply_offset + layout.supply_size].reshape(
        2, -1, layout.pile_block_size
    )
    # These are structurally populated supply rows. Their trait sentinels are
    # zero in native generation-2 observations and become one only for a
    # generation-1 opponent.
    piles[0, 0, layout.pile_count_field] = 10.0
    piles[1, 1, layout.pile_base_field] = 1.0
    return observations


def _assert_restored_positions(observations: np.ndarray, obs_version: int) -> None:
    layout = legacy_constant_layout(obs_version)
    landscape_ids = slice(layout.landscape_offset, layout.landscape_offset + layout.landscape_id_size)
    prophecy = layout.landscape_offset + layout.landscape_prophecy_offset
    np.testing.assert_array_equal(observations[..., landscape_ids], 1.0)
    np.testing.assert_array_equal(observations[..., prophecy], 1.0)
    piles = observations[..., layout.supply_offset : layout.supply_offset + layout.supply_size].reshape(
        *observations.shape[:-1], -1, layout.pile_block_size
    )
    populated = (piles[..., layout.pile_count_field] != 0.0) | (piles[..., layout.pile_base_field] != 0.0)
    traits = piles[..., layout.pile_trait_field]
    np.testing.assert_array_equal(traits[populated], 1.0)
    np.testing.assert_array_equal(traits[~populated], 0.0)


def test_league_router_restores_generation_one_v2_downgrade_without_touching_candidate() -> None:
    observations = _generation_two_v3_batch()
    native = observations.copy()
    candidate = _RecordingLeagueModel(obs_version=3, encoder_generation=2)
    legacy_ancestor = _RecordingLeagueModel(obs_version=2, encoder_generation=1)
    masks = np.ones((2, dz.ACTION_SPACE_SIZE), dtype=np.bool_)

    route_leaf_evaluations(
        (candidate, legacy_ancestor),
        observations,
        masks,
        np.asarray([0, 1], dtype=np.uint8),
        torch.device("cpu"),
    )

    np.testing.assert_array_equal(candidate.seen[0], native[:1])
    expected_ancestor = restore_legacy_constants(downgrade_v3_observations(native[1:]), legacy_constant_layout(2))
    np.testing.assert_array_equal(legacy_ancestor.seen[0], expected_ancestor)
    _assert_restored_positions(legacy_ancestor.seen[0], 2)
    np.testing.assert_array_equal(observations, native)


def test_league_router_restores_generation_one_v3_input_and_leaves_generation_two_untouched() -> None:
    observations = _generation_two_v3_batch()
    legacy_ancestor = _RecordingLeagueModel(obs_version=3, encoder_generation=1)
    generation_two_ancestor = _RecordingLeagueModel(obs_version=3, encoder_generation=2)
    masks = np.ones((2, dz.ACTION_SPACE_SIZE), dtype=np.bool_)

    route_leaf_evaluations(
        (generation_two_ancestor, legacy_ancestor),
        observations,
        masks,
        np.asarray([0, 1], dtype=np.uint8),
        torch.device("cpu"),
    )

    np.testing.assert_array_equal(generation_two_ancestor.seen[0], observations[:1])
    expected_legacy = restore_legacy_constants(observations[1:], legacy_constant_layout(3))
    np.testing.assert_array_equal(legacy_ancestor.seen[0], expected_legacy)
    _assert_restored_positions(legacy_ancestor.seen[0], 3)
