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


def selfplay_run(seed: int, batches: int):
    config = dz.SelfPlayConfig(
        n_games=64,
        sims_per_move=64,
        max_batch=256,
        seed=seed,
        kingdom_mode=dz.SelfPlayKingdomMode.Fixed,
        kingdom=KINGDOM,
        dirichlet_frac=0.0,
    )
    runner = dz.SelfPlayRunner(config)
    leaves = 0
    records_seen = 0

    started = time.perf_counter()
    for _ in range(batches):
        obs, masks = runner.collect_leaves(config.max_batch)
        assert obs.dtype == np.float32
        assert masks.dtype == np.bool_
        assert obs.ndim == 2 and obs.shape[1] == dz.OBS_SIZE
        assert masks.ndim == 2 and masks.shape[1] == dz.ACTION_SPACE_SIZE
        assert obs.shape[0] == masks.shape[0]
        if obs.shape[0] == 0:
            continue

        values = np.zeros((obs.shape[0],), dtype=np.float32)
        policies = np.zeros((obs.shape[0], dz.ACTION_SPACE_SIZE), dtype=np.float32)
        runner.provide_evaluations(values, policies)
        assert runner.total_virtual_loss() == 0.0
        leaves += obs.shape[0]

        for record in runner.finished_games():
            records_seen += 1
            rec_obs = record["observations"]
            rec_policy = record["policy_targets"]
            rec_values = record["values"]
            assert rec_obs.dtype == np.float32
            assert rec_policy.dtype == np.float32
            assert rec_values.dtype == np.float32
            assert rec_obs.shape[1] == dz.OBS_SIZE
            assert rec_policy.shape[1] == dz.ACTION_SPACE_SIZE
            assert rec_obs.shape[0] == rec_policy.shape[0] == rec_values.shape[0]
            if rec_policy.shape[0] > 0:
                np.testing.assert_allclose(rec_policy.sum(axis=1), 1.0, atol=1e-4)
                assert np.all((rec_values == -1.0) | (rec_values == 0.0) | (rec_values == 1.0))

    elapsed = time.perf_counter() - started
    leaves_per_sec = leaves / elapsed if elapsed > 0 else 0.0
    return leaves, leaves_per_sec, records_seen


def selfplay_record_smoke(seed: int):
    config = dz.SelfPlayConfig(
        n_games=4,
        sims_per_move=4,
        max_batch=16,
        seed=seed,
        kingdom_mode=dz.SelfPlayKingdomMode.Fixed,
        kingdom=KINGDOM,
        dirichlet_frac=0.0,
        max_recorded_moves=512,
    )
    runner = dz.SelfPlayRunner(config)
    for _ in range(20000):
        obs, masks = runner.collect_leaves(config.max_batch)
        assert obs.shape[1] == dz.OBS_SIZE
        assert masks.shape[1] == dz.ACTION_SPACE_SIZE
        if obs.shape[0] == 0:
            continue
        runner.provide_evaluations(
            np.zeros((obs.shape[0],), dtype=np.float32),
            np.zeros((obs.shape[0], dz.ACTION_SPACE_SIZE), dtype=np.float32),
        )
        records = runner.finished_games()
        if records:
            record = records[0]
            rec_obs = record["observations"]
            rec_policy = record["policy_targets"]
            rec_values = record["values"]
            assert rec_obs.shape[1] == dz.OBS_SIZE
            assert rec_policy.shape[1] == dz.ACTION_SPACE_SIZE
            assert rec_obs.shape[0] == rec_policy.shape[0] == rec_values.shape[0]
            assert rec_obs.dtype == np.float32
            assert rec_policy.dtype == np.float32
            assert rec_values.dtype == np.float32
            np.testing.assert_allclose(rec_policy.sum(axis=1), 1.0, atol=1e-4)
            assert np.all((rec_values == -1.0) | (rec_values == 0.0) | (rec_values == 1.0))
            return len(records)
    raise AssertionError("selfplay record smoke did not finish a game")


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

    leaves, leaves_per_sec, records = selfplay_run(0x51E1F, 300)
    assert leaves > 0
    print(
        f"selfplay_py_leaves_per_sec={leaves_per_sec:.2f} "
        f"n=64 sims=64 max_batch=256 leaves={leaves} records={records}"
    )
    record_count = selfplay_record_smoke(0x51E1F + 1)
    print(f"selfplay_py_record_smoke_records={record_count}")


if __name__ == "__main__":
    main()
