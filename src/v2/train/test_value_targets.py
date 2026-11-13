from __future__ import annotations

import numpy as np

import dominion_v2_py as dz


def _finish_record(value_target: dz.SelfPlayValueTarget) -> dict:
    config = dz.SelfPlayConfig(
        n_games=1,
        sims_per_move=2,
        max_batch=8,
        seed=0xA11CE,
        kingdom_mode=dz.SelfPlayKingdomMode.Fixed,
        kingdom=["Village"],
        dirichlet_frac=0.0,
        temp_moves=0,
        max_tree_nodes=256,
        scripted_bot=dz.SelfPlayScriptedBotKind.BigMoney,
        scripted_nn_player=0,
        auto_play_treasures=True,
        prune_treasure_plays=True,
        value_target=value_target,
        margin_scale=20.0,
    )
    runner = dz.SelfPlayRunner(config)
    for _ in range(20_000):
        observations, _masks = runner.collect_leaves(config.max_batch)
        if observations.shape[0] > 0:
            runner.provide_evaluations(
                np.zeros((observations.shape[0],), dtype=np.float32),
                np.zeros((observations.shape[0], dz.ACTION_SPACE_SIZE), dtype=np.float32),
            )
        finished = runner.finished_games()
        if finished:
            return finished[0]
    raise AssertionError("deterministic value-target self-play game did not finish")


def test_margin_value_targets_are_sign_preserving_hybrid() -> None:
    # v = sign(margin) * (0.5 + 0.5 * min(|margin|, scale)/scale); 0 for ties.
    # Wins never train below +0.5: narrow wins are legitimate outcomes, and
    # margin only adds gradient within the win/loss categories.
    record = _finish_record(dz.SelfPlayValueTarget.Margin)
    scale = 20.0
    players = record["players"].astype(np.intp, copy=False)
    scores = record["scores"].astype(np.float32, copy=False)
    margin = scores[players] - scores[1 - players]
    graded = np.clip(np.abs(margin), 0.0, scale) / scale
    expected = np.where(margin == 0.0, 0.0, np.sign(margin) * (0.5 + 0.5 * graded))

    assert record["scores"].dtype == np.int16
    np.testing.assert_allclose(record["values"], expected.astype(np.float32), rtol=0.0, atol=1e-6)


def test_outcome_value_targets_remain_winner_based() -> None:
    record = _finish_record(dz.SelfPlayValueTarget.Outcome)
    winner = record["winner"]
    players = record["players"]
    expected = np.zeros(players.shape, dtype=np.float32)
    if winner is not None:
        expected[players == int(winner)] = 1.0
        expected[players != int(winner)] = -1.0

    np.testing.assert_array_equal(record["values"], expected)
