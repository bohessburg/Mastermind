"""Standalone correctness checks for the C15 Python card-token prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# This file is intentionally runnable as
# PYTHONPATH=build ./.venv/bin/python tests/v2/py/test_card_transformer.py
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dominion_v2_py as dz
from src.v2.train.card_transformer import CardTokenNet


KINGDOM = [
    "Village",
    "Smithy",
    "Market",
    "Witch",
    "Moat",
    "Cellar",
    "Chapel",
    "Workshop",
    "Remodel",
    "Mine",
]


def _game_and_model() -> tuple[object, CardTokenNet, torch.Tensor]:
    game = dz.new_game(dz.Setup(players=2, kingdom=KINGDOM), 0xC15CAFE)
    model = CardTokenNet(dz.OBS_SIZE_V2, dz.ACTION_SPACE_SIZE)
    model.eval()
    obs = torch.from_numpy(game.encode(0, 2)).unsqueeze(0)
    return game, model, obs


def _token_index(tokenized, def_id: int) -> int:
    matches = torch.nonzero(tokenized.card_mask[0] & (tokenized.def_ids[0] == int(def_id)), as_tuple=False)
    assert matches.shape == (1, 1), f"expected one active token for def {def_id}, got {matches.tolist()}"
    return int(matches.item())


def _feature_count(tokenized, token_index: int, feature_index: int) -> float:
    # Card count features are intentionally log1p conditioned in the model.
    return float(torch.expm1(tokenized.card_features[0, token_index, feature_index]))


def test_tokenizer_matches_real_game_state() -> None:
    game, model, obs = _game_and_model()
    tokenized = model.tokenize(obs)

    assert tokenized.card_mask.sum().item() == len(game.supply())
    assert tokenized.def_ids.dtype == torch.long

    hand = {int(def_id): int(count) for def_id, count in dict(game.hand(0)).items()}
    supply = {int(def_id): int(count) for def_id, count in game.supply()}
    # At a freshly created game the only cards in the deck are the ten known
    # starting cards, so composition is exactly recoverable from hand + deck.
    expected_deck = {
        int(dz.DEF_COPPER): 7 - hand.get(int(dz.DEF_COPPER), 0),
        int(dz.DEF_ESTATE): 3 - hand.get(int(dz.DEF_ESTATE), 0),
    }
    assert game.deck_count(0) == sum(expected_deck.values())

    for def_id in (dz.DEF_COPPER, dz.DEF_ESTATE, dz.DEF_SILVER, dz.DEF_VILLAGE):
        token_index = _token_index(tokenized, int(def_id))
        np.testing.assert_allclose(
            _feature_count(tokenized, token_index, model.OWN_HAND),
            hand.get(int(def_id), 0),
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            _feature_count(tokenized, token_index, model.OWN_DECK),
            expected_deck.get(int(def_id), 0),
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            _feature_count(tokenized, token_index, model.SUPPLY_REMAINING),
            supply[int(def_id)],
            atol=1.0e-5,
        )


def test_evaluate_shapes_and_pointer_scatter_against_native_actions() -> None:
    game, model, obs = _game_and_model()
    legal_mask = torch.from_numpy(game.legal_mask()).unsqueeze(0)
    masked_logits, values = model.evaluate(obs, legal_mask)

    assert masked_logits.shape == (1, dz.ACTION_SPACE_SIZE)
    assert values.shape == (1,)
    assert torch.isfinite(masked_logits).all()
    assert torch.isfinite(values).all()
    assert torch.all(masked_logits[~legal_mask] == model.POINTER_FILL)

    # Compute the same pointer-head values from the encoded token and confirm
    # they land at the exact C++ flat action indices, rather than Slot indices.
    assert dz.A_PLAY_BASE == 1
    assert dz.A_BUY_BASE == 206
    tokenized = model.tokenize(obs)
    encoded = model.encode_tokens(tokenized)
    raw_logits, _ = model.forward_tokenized(tokenized)
    for def_id in (dz.DEF_COPPER, dz.DEF_ESTATE, dz.DEF_VILLAGE):
        token_index = _token_index(tokenized, int(def_id))
        card_token = encoded[:, 1 + token_index]
        expected_play = model.play_head(card_token).squeeze(-1)
        expected_buy = model.buy_head(card_token).squeeze(-1)
        torch.testing.assert_close(raw_logits[:, dz.A_PLAY_BASE + int(def_id)], expected_play)
        torch.testing.assert_close(raw_logits[:, dz.A_BUY_BASE + int(def_id)], expected_buy)


def main() -> None:
    test_tokenizer_matches_real_game_state()
    test_evaluate_shapes_and_pointer_scatter_against_native_actions()
    print("test_card_transformer: PASS")


if __name__ == "__main__":
    main()
