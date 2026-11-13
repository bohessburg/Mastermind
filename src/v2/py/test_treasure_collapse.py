from __future__ import annotations

import numpy as np

import dominion_v2_py as dz


KINGDOM = [
    "Village",
    "Smithy",
    "Market",
    "Festival",
    "Laboratory",
    "Cellar",
    "Chapel",
    "Militia",
    "Witch",
    "Moat",
]

TREASURE_ACTIONS = np.array(
    [
        dz.A_PLAY_BASE + dz.DEF_COPPER,
        dz.A_PLAY_BASE + dz.DEF_SILVER,
        dz.A_PLAY_BASE + dz.DEF_GOLD,
    ],
    dtype=np.intp,
)


def _config(*, auto_play_treasures: bool, prune_treasure_plays: bool) -> dz.SelfPlayConfig:
    return dz.SelfPlayConfig(
        n_games=1,
        sims_per_move=4,
        max_batch=8,
        seed=0x7EA5,
        kingdom_mode=dz.SelfPlayKingdomMode.Fixed,
        kingdom=KINGDOM,
        dirichlet_frac=0.0,
        temp_moves=0,
        max_recorded_moves=2048,
        max_tree_nodes=1024,
        auto_play_treasures=auto_play_treasures,
        prune_treasure_plays=prune_treasure_plays,
    )


def _treasure_preferring_evaluations(batch: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((batch,), dtype=np.float32)
    policies = np.zeros((batch, dz.ACTION_SPACE_SIZE), dtype=np.float32)
    policies[:, TREASURE_ACTIONS] = 10.0
    return values, policies


def _run_one_game(config: dz.SelfPlayConfig) -> dict:
    runner = dz.SelfPlayRunner(config)
    for _ in range(10_000):
        observations, _ = runner.collect_leaves(config.max_batch)
        batch = int(observations.shape[0])
        if batch:
            values, policies = _treasure_preferring_evaluations(batch)
            runner.provide_evaluations(values, policies)
        finished = runner.finished_games()
        if finished:
            return finished[0]
    raise AssertionError("treasure-collapse self-play smoke did not finish a game")


def test_treasure_collapse_records_fewer_non_treasure_decisions() -> None:
    baseline = _run_one_game(_config(auto_play_treasures=False, prune_treasure_plays=False))
    collapsed = _run_one_game(_config(auto_play_treasures=True, prune_treasure_plays=True))

    assert collapsed["policy_targets"].shape[0] < baseline["policy_targets"].shape[0]
    assert float(collapsed["policy_targets"][:, TREASURE_ACTIONS].sum()) == 0.0


def test_treasure_collapse_selfplay_is_deterministic() -> None:
    first = _run_one_game(_config(auto_play_treasures=True, prune_treasure_plays=True))
    second = _run_one_game(_config(auto_play_treasures=True, prune_treasure_plays=True))

    assert first["seed"] == second["seed"]
    np.testing.assert_array_equal(first["observations"], second["observations"])
    np.testing.assert_array_equal(first["policy_targets"], second["policy_targets"])
    np.testing.assert_array_equal(first["values"], second["values"])
    np.testing.assert_array_equal(first["players"], second["players"])


def test_decision_searcher_auto_plays_treasures_in_ascending_definition_order() -> None:
    game = dz.new_game(dz.Setup(players=2, kingdom=KINGDOM), 0x7EA5)
    game.step(0)  # Action phase pass; the opening hand's Coppers are now legal treasures.

    played_defs: list[int] = []
    while True:
        legal = game.legal_mask()
        legal_treasures = [action for action in TREASURE_ACTIONS if legal[action]]
        if not legal_treasures:
            break
        searcher = dz.DecisionSearcher(
            game,
            int(game.current_decision()["player"]),
            {
                "sims": 4,
                "c_puct": 1.25,
                "determinizations": 1,
                "seed": 0x7EA5,
                "auto_play_treasures": True,
            },
        )
        assert searcher.done()
        observations, masks = searcher.collect_leaves()
        assert observations.shape[0] == masks.shape[0] == 0

        action = int(searcher.best_action())
        assert action in legal_treasures
        played_defs.append(action - dz.A_PLAY_BASE)
        game.step(action)

    assert played_defs
    assert played_defs == sorted(played_defs)
