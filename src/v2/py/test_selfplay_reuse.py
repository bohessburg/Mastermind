from __future__ import annotations

import numpy as np

import dominion_v2_py as dz


def _run_reused_selfplay(seed: int) -> dict:
    # SelfPlayRunner's NN tree is intentionally K=1; spell out the Scaffold
    # setting too so this smoke fixture cannot accidentally exercise a
    # multi-determinization reuse configuration if that path changes later.
    config = dz.SelfPlayConfig(
        n_games=1,
        sims_per_move=4,
        max_batch=8,
        seed=seed,
        kingdom_mode=dz.SelfPlayKingdomMode.Fixed,
        max_recorded_moves=256,
        max_tree_nodes=512,
        scaffold_determinizations=1,
        dirichlet_alpha=0.30,
        dirichlet_frac=0.25,
        tree_reuse=True,
        # Keep this small smoke fixture close to its historical four-sim
        # workload while exercising the adopted-root visit-target path.
        min_new_sims=2,
        expand_top_k=8,
    )
    runner = dz.SelfPlayRunner(config)
    for _ in range(30_000):
        observations, _masks = runner.collect_leaves(config.max_batch)
        if observations.shape[0] > 0:
            runner.provide_evaluations(
                np.zeros((observations.shape[0],), dtype=np.float32),
                np.zeros((observations.shape[0], dz.ACTION_SPACE_SIZE), dtype=np.float32),
            )
        finished = runner.finished_games()
        if finished:
            return finished[0]
    raise AssertionError("tree-reuse self-play did not finish a game")


def test_selfplay_tree_reuse_topk_is_seed_deterministic() -> None:
    first = _run_reused_selfplay(0xC14_0001)
    second = _run_reused_selfplay(0xC14_0001)

    assert first["seed"] == second["seed"]
    assert first["winner"] == second["winner"]
    assert first["kingdom"] == second["kingdom"]
    for field in ("observations", "policy_targets", "values", "players"):
        np.testing.assert_array_equal(first[field], second[field])


def test_selfplay_tree_reuse_min_new_sims_binding_defaults_to_64() -> None:
    assert dz.SelfPlayConfig().min_new_sims == 64
    assert dz.SelfPlayConfig(min_new_sims=2).min_new_sims == 2
