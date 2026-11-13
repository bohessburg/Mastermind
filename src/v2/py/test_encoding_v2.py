from __future__ import annotations

import hashlib

import numpy as np

import dominion_v2_py as dz


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


# These cover the opening and three real decision states, including the
# landscape-sentinel correction, and protect v1's byte-level encoding ABI.
V1_GOLDEN_SHA256 = [
    (
        "b86808b3b9b24938e12bddb0052be726ee78b5c64fb9865aa9683ba4113e81d7",
        "a4732f358d37a400fc3328bd0a56a3d983b6d40763d35fbd0e14a8c3e544c8a0",
    ),
    (
        "21dbb2b3e64c1d233260dfc55e414ef44dd649934a721a8c119f5ed720a354c8",
        "299475e1769f55668d8995f1fc5b3708de1511b1ddd123b3c3290a6f744402ed",
    ),
    (
        "7adb587f5868b1fb0d0fc6ede5dfe6bdb40ba7c06da6b5af971e02340ad60d00",
        "212d280cc43376df3b18a7cdfbe98125a480a8048d89847db710360ce764abba",
    ),
    (
        "9830934ad7dbf26cb958781b8abd402f1397c330c8d6de9d4e8ecc8bd5ddcb03",
        "85411b4e822bb8a4cfd35f44686f5d4c89a6c3515743832b51168c744db6496e",
    ),
]


def _sha256(obs: np.ndarray) -> str:
    return hashlib.sha256(obs.tobytes()).hexdigest()


def test_v1_default_encoding_matches_pre_versioning_golden_snapshots() -> None:
    game = dz.new_game(dz.Setup(players=2, kingdom=KINGDOM), 0xE4C00001)
    for expected_player0, expected_player1 in V1_GOLDEN_SHA256:
        default0 = game.encode(0)
        default1 = game.encode(1)
        explicit0 = game.encode(0, 1)
        explicit1 = game.encode(1, 1)
        assert default0.shape == (dz.OBS_SIZE_V1,)
        assert default1.shape == (dz.OBS_SIZE_V1,)
        np.testing.assert_array_equal(default0, explicit0)
        np.testing.assert_array_equal(default1, explicit1)
        assert _sha256(default0) == expected_player0
        assert _sha256(default1) == expected_player1
        legal = np.flatnonzero(game.legal_mask())
        assert legal.size > 0
        game.step(int(legal[0]))


def test_v2_game_and_runner_version_selection() -> None:
    assert dz.OBS_VERSION == 1
    assert dz.ENCODER_GENERATION == 2
    assert dz.OBS_SIZE == dz.OBS_SIZE_V1 == 1141
    assert dz.OBS_SIZE_V2 == 1717
    assert dz.OBS_SIZE_V3 == 1788
    assert dz.obs_size_for(1) == dz.OBS_SIZE_V1
    assert dz.obs_size_for(2) == dz.OBS_SIZE_V2
    assert dz.obs_size_for(3) == dz.OBS_SIZE_V3
    assert dz.encoder_layout(1)["supply_offset"] != dz.encoder_layout(2)["supply_offset"]
    assert dz.encoder_layout(2) == dz.encoder_layout(3)

    game = dz.new_game(dz.Setup(players=2, kingdom=KINGDOM), 0xE4C00002)
    obs = game.encode(0, 2)
    assert obs.shape == (dz.OBS_SIZE_V2,)
    assert obs[0] == 2.0
    assert obs[1] == float(dz.OBS_SIZE_V2)

    obs_v3 = game.encode(0, 3)
    assert obs_v3.shape == (dz.OBS_SIZE_V3,)
    assert obs_v3[0] == 3.0
    assert obs_v3[1] == float(dz.OBS_SIZE_V3)
    np.testing.assert_array_equal(obs_v3[2 : dz.OBS_SIZE_V2], obs[2:])

    config = dz.SelfPlayConfig(
        n_games=2,
        sims_per_move=2,
        max_batch=4,
        seed=0xE4C00003,
        kingdom_mode=dz.SelfPlayKingdomMode.Fixed,
        kingdom=KINGDOM,
        dirichlet_frac=0.0,
        obs_version=2,
    )
    assert config.obs_version == 2
    runner = dz.SelfPlayRunner(config)
    leaves, masks = runner.collect_leaves(config.max_batch)
    assert leaves.shape == (masks.shape[0], dz.OBS_SIZE_V2)
    assert np.all(leaves[:, 0] == 2.0)
    if leaves.shape[0]:
        runner.provide_evaluations(
            np.zeros((leaves.shape[0],), dtype=np.float32),
            np.zeros((leaves.shape[0], dz.ACTION_SPACE_SIZE), dtype=np.float32),
        )

    config_v3 = dz.SelfPlayConfig(
        n_games=2,
        sims_per_move=2,
        max_batch=4,
        seed=0xE4C00005,
        kingdom_mode=dz.SelfPlayKingdomMode.Fixed,
        kingdom=KINGDOM,
        dirichlet_frac=0.0,
        obs_version=3,
    )
    runner_v3 = dz.SelfPlayRunner(config_v3)
    leaves_v3, masks_v3 = runner_v3.collect_leaves(config_v3.max_batch)
    assert leaves_v3.shape == (masks_v3.shape[0], dz.OBS_SIZE_V3)
    assert np.all(leaves_v3[:, 0] == 3.0)
    if leaves_v3.shape[0]:
        runner_v3.provide_evaluations(
            np.zeros((leaves_v3.shape[0],), dtype=np.float32),
            np.zeros((leaves_v3.shape[0], dz.ACTION_SPACE_SIZE), dtype=np.float32),
        )

    searcher = dz.DecisionSearcher(
        game,
        0,
        {"sims": 2, "c_puct": 1.25, "determinizations": 1, "seed": 0xE4C00004, "obs_version": 2},
    )
    leaves, masks = searcher.collect_leaves()
    assert leaves.shape == (masks.shape[0], dz.OBS_SIZE_V2)
    assert np.all(leaves[:, 0] == 2.0)

    searcher_v3 = dz.DecisionSearcher(
        game,
        0,
        {"sims": 2, "c_puct": 1.25, "determinizations": 1, "seed": 0xE4C00006, "obs_version": 3},
    )
    leaves_v3, masks_v3 = searcher_v3.collect_leaves()
    assert leaves_v3.shape == (masks_v3.shape[0], dz.OBS_SIZE_V3)
    assert np.all(leaves_v3[:, 0] == 3.0)
