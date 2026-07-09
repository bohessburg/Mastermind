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

    assert isinstance(game.current_decision(), dict)
    assert isinstance(game.hand(0), dict)
    assert isinstance(game.supply(), list)
    assert isinstance(game.in_play(0), list)
    assert isinstance(game.resources(), dict)

    result_a = play_random_game(0xCAFE)
    result_b = play_random_game(0xCAFE)
    assert result_a == result_b


if __name__ == "__main__":
    main()
