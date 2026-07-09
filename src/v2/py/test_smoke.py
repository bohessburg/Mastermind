import time

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


def play_random_game(seed: int):
    setup = dz.Setup(players=2, kingdom=KINGDOM)
    game = dz.new_game(setup, seed)
    rng = np.random.default_rng(seed ^ 0xA501)
    checked_encoding = False
    steps = 0

    while not game.game_over():
        mask = game.legal_mask()
        assert mask.dtype == np.bool_
        assert mask.shape == (dz.ACTION_SPACE_SIZE,)
        legal = np.flatnonzero(mask)
        assert legal.size > 0

        if not checked_encoding:
            obs = game.encode(0)
            assert obs.dtype == np.float32
            assert obs.shape == (dz.OBS_SIZE,)
            assert obs[0] == dz.OBS_VERSION
            checked_encoding = True

        action = int(legal[int(rng.integers(legal.size))])
        done = game.step(action)
        steps += 1
        assert done == game.game_over()
        assert steps < 10000

    assert checked_encoding
    return (game.score(0), game.score(1), game.winner(), game.turn())


def random_actions(masks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    counts = masks.sum(axis=1)
    assert np.all(counts > 0)
    picks = np.floor(rng.random(counts.shape[0]) * counts).astype(np.int64)
    cumulative = np.cumsum(masks, axis=1)
    return np.argmax(cumulative > picks[:, None], axis=1).astype(np.int32)


def batch_run(seed_base: int, steps: int):
    runner = dz.BatchRunner(KINGDOM, 64, seed_base)
    rng = np.random.default_rng(seed_base ^ 0xBA7C)
    reward_sequence = []

    obs = runner.observations()
    assert obs.shape == (64, dz.OBS_SIZE)
    assert obs.dtype == np.float32

    players = runner.current_players()
    assert players.shape == (64,)
    assert players.dtype == np.int32

    started = time.perf_counter()
    for _ in range(steps):
        masks = runner.legal_masks()
        assert masks.shape == (64, dz.ACTION_SPACE_SIZE)
        assert masks.dtype == np.bool_
        actions = random_actions(masks, rng)
        dones, rewards = runner.step(actions)
        assert dones.shape == (64,)
        assert dones.dtype == np.bool_
        assert rewards.shape == (64,)
        assert rewards.dtype == np.float32
        reward_sequence.extend(float(value) for value in rewards[dones])

    elapsed = time.perf_counter() - started
    completed = runner.games_completed()
    games_per_sec = completed / elapsed if elapsed > 0 else 0.0
    return completed, games_per_sec, tuple(reward_sequence)


def main():
    setup = dz.Setup(players=2, kingdom=[dz.def_id(name) for name in KINGDOM])
    game = dz.new_game(setup, 0x5EED)

    mask = game.legal_mask()
    assert mask.shape == (dz.ACTION_SPACE_SIZE,)
    assert mask.any()

    obs = game.encode(0)
    assert obs.shape == (dz.OBS_SIZE,)
    assert obs.dtype == np.float32

    clone = game.clone()
    assert clone.phase() == game.phase()
    clone.step(0)
    assert clone.phase() != game.phase()

    sampled = game.clone()
    own_hand = dict(sampled.hand(0))
    before = sampled.encode(0).copy()
    sampled.determinize(0xD37E)
    np.testing.assert_array_equal(before, sampled.encode(0))
    assert dict(sampled.hand(0)) == own_hand

    assert isinstance(game.current_decision(), dict)
    assert isinstance(game.hand(0), dict)
    assert isinstance(game.supply(), list)
    assert isinstance(game.in_play(0), list)
    assert isinstance(game.resources(), dict)

    result_a = play_random_game(0xCAFE)
    result_b = play_random_game(0xCAFE)
    assert result_a == result_b

    completed, games_per_sec, _ = batch_run(0xBADC0DE, 10000)
    print(f"batch_self_play_games_per_sec={games_per_sec:.2f} completed={completed}")

    det_a = batch_run(0xBADC0DE, 512)[2]
    det_b = batch_run(0xBADC0DE, 512)[2]
    assert det_a == det_b


if __name__ == "__main__":
    main()
