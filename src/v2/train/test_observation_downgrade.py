from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import dominion_v2_py as dz

from .config import load_config
from .observation import downgrade_v3_observations
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
