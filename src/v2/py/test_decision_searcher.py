from __future__ import annotations

import numpy as np
import torch

import dominion_v2_py as dz

from src.v2.train.model import DominionNet


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


def _scripted_position():
    game = dz.new_game(dz.Setup(players=2, kingdom=KINGDOM), 0xD3C1510)
    # Advance a short, fixed prefix so the search is exercised from a real
    # in-progress decision rather than only the opening state.
    for _ in range(4):
        legal = np.flatnonzero(game.legal_mask())
        assert legal.size > 0
        game.step(int(legal[0]))
    return game


def _fixed_tiny_net() -> DominionNet:
    model = DominionNet(dz.OBS_SIZE, dz.ACTION_SPACE_SIZE, hidden_sizes=[8])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # Fixed, non-uniform logits make this an actual tiny neural policy
        # while retaining a completely reproducible test fixture.
        model.policy_head.bias.copy_(torch.linspace(-0.75, 0.75, dz.ACTION_SPACE_SIZE))
        model.value_head[0].bias.fill_(0.125)
    model.eval()
    return model


def _search_action(game, model: DominionNet) -> int:
    perspective = int(game.current_decision()["player"])
    searcher = dz.DecisionSearcher(
        game,
        perspective,
        {"sims": 24, "c_puct": 1.25, "determinizations": 2, "seed": 0xD3C1510},
    )
    for _ in range(200):
        if searcher.done():
            action = int(searcher.best_action())
            assert bool(game.legal_mask()[action])
            return action
        obs, masks = searcher.collect_leaves()
        assert obs.dtype == np.float32
        assert masks.dtype == np.bool_
        assert obs.shape == (masks.shape[0], dz.OBS_SIZE)
        assert masks.shape[1] == dz.ACTION_SPACE_SIZE
        if obs.shape[0] == 0:
            continue
        with torch.no_grad():
            logits, values = model.evaluate(
                torch.as_tensor(obs, dtype=torch.float32),
                torch.as_tensor(masks, dtype=torch.bool),
            )
        searcher.provide_evaluations(
            values.cpu().numpy().astype(np.float32, copy=False),
            logits.cpu().numpy().astype(np.float32, copy=False),
        )
    raise AssertionError("DecisionSearcher did not complete")


def test_fixed_tiny_network_decision_search_is_deterministic_and_legal() -> None:
    game = _scripted_position()
    model = _fixed_tiny_net()

    first = _search_action(game.clone(), model)
    second = _search_action(game.clone(), model)

    assert first == second
    assert bool(game.legal_mask()[first])
