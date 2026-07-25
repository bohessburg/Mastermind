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
from src.v2.train.card_transformer import OBS_DECISION_OFFSET, SELECT_SEMANTIC_COUNT, CardTokenNet
from src.v2.train.model import build_model


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


def _game_at_chapel_trash_choice() -> object:
    """Drive a native game to a live Chapel trash-selection decision."""

    game = dz.new_game(dz.Setup(players=2, kingdom=KINGDOM), 0xC15CAFE)
    chapel_play = int(dz.A_PLAY_BASE + dz.DEF_CHAPEL)
    chapel_buy = int(dz.A_BUY_BASE + dz.DEF_CHAPEL)
    for _ in range(512):
        legal = game.legal_mask()
        select_actions = np.flatnonzero(legal[dz.A_SELECT_BASE : dz.A_OPTION_BASE])
        if select_actions.size:
            game.step(int(dz.A_SELECT_BASE + select_actions[0]))
        elif legal[chapel_play]:
            game.step(chapel_play)
            chapel_selects = np.flatnonzero(game.legal_mask()[dz.A_SELECT_BASE : dz.A_OPTION_BASE])
            if chapel_selects.size:
                return game
        elif legal[chapel_buy]:
            game.step(chapel_buy)
        else:
            plays = np.flatnonzero(legal[dz.A_PLAY_BASE : dz.A_BUY_BASE])
            if plays.size:
                game.step(int(dz.A_PLAY_BASE + plays[0]))
            else:
                game.step(int(np.flatnonzero(legal)[0]))
    raise AssertionError("failed to drive a Chapel trash-choice in the native engine")


def _game_after_trashing() -> object:
    """Resolve a native Chapel decision by trashing a live card."""

    game = _game_at_chapel_trash_choice()
    select_actions = np.flatnonzero(game.legal_mask()[dz.A_SELECT_BASE : dz.A_OPTION_BASE])
    assert select_actions.size
    game.step(int(dz.A_SELECT_BASE + select_actions[0]))
    assert game.trash()
    return game


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


def test_v2_tokenizer_regression_keeps_its_feature_layout() -> None:
    game, model, obs = _game_and_model()
    tokenized = model.tokenize(obs)

    # v2 keeps its original 19 per-card and 68 global features. In
    # particular it does not acquire the v3 trash scalar/total feature.
    assert model.obs_version == 2
    assert tokenized.card_features.shape[-1] == CardTokenNet.CARD_FEATURE_SIZE_V2 == 19
    assert tokenized.global_features.shape[-1] == CardTokenNet.GLOBAL_FEATURE_SIZE_V2 == 68
    expected_meta = torch.tensor(
        [[2.0, np.log1p(dz.OBS_SIZE_V2), 0.0, np.log1p(float(obs[0, 3]))]], dtype=torch.float32
    )
    torch.testing.assert_close(tokenized.global_features[:, :4], expected_meta)
    copper_index = _token_index(tokenized, int(dz.DEF_COPPER))
    assert tokenized.card_features[0, copper_index, model.TRASH_COUNT - 1].item() in (0.0, 1.0)
    assert game.trash() == {}


def test_v3_tokenizer_exposes_live_trash_counts() -> None:
    game = _game_after_trashing()
    model = CardTokenNet(dz.OBS_SIZE_V3, dz.ACTION_SPACE_SIZE)
    model.eval()
    obs = torch.from_numpy(game.encode(0, 3)).unsqueeze(0)
    tokenized = model.tokenize(obs)
    trash = {int(def_id): int(count) for def_id, count in dict(game.trash()).items()}

    assert model.obs_version == 3
    assert tokenized.card_features.shape[-1] == CardTokenNet.CARD_FEATURE_SIZE_V3 == 20
    assert tokenized.global_features.shape[-1] == CardTokenNet.GLOBAL_FEATURE_SIZE_V3 == 76
    assert trash
    for def_id, count in trash.items():
        token_index = _token_index(tokenized, def_id)
        np.testing.assert_allclose(
            _feature_count(tokenized, token_index, model.TRASH_COUNT),
            count,
            atol=1.0e-5,
        )
    expected_total = torch.tensor([[np.log1p(sum(trash.values()))]], dtype=torch.float32)
    torch.testing.assert_close(
        tokenized.global_features[:, model.GLOBAL_TRASH_TOTAL : model.GLOBAL_TRASH_TOTAL + 1],
        expected_total,
    )


def test_v3_tokenizer_exposes_select_semantics_and_decision_source() -> None:
    game = _game_at_chapel_trash_choice()
    model = CardTokenNet(dz.OBS_SIZE_V3, dz.ACTION_SPACE_SIZE)
    model.eval()
    obs = torch.from_numpy(game.encode(0, 3)).unsqueeze(0)
    tokenized = model.tokenize(obs)

    # Chapel's generic Choose instruction applies Then::Trash. Its source is
    # encoded as def + 1, then embedded and added only to the v3 global token.
    expected_semantic = torch.zeros((1, SELECT_SEMANTIC_COUNT), dtype=torch.float32)
    expected_semantic[0, 3] = 1.0  # SelectSemantic::Trash
    torch.testing.assert_close(
        tokenized.global_features[
            :, model.GLOBAL_SELECT_SEMANTIC_OFFSET : model.GLOBAL_SELECT_SEMANTIC_OFFSET + SELECT_SEMANTIC_COUNT
        ],
        expected_semantic,
    )
    assert obs[0, OBS_DECISION_OFFSET + 12].item() == float(dz.DEF_CHAPEL + 1)
    source_component = tokenized.global_token - model.global_projection(tokenized.global_features).unsqueeze(1)
    expected_source = model.def_embedding.weight[int(dz.DEF_CHAPEL)].reshape(1, 1, -1)
    torch.testing.assert_close(source_component, expected_source)

    # A phase decision has no card source, so it contributes no source
    # embedding even though it still has the SelectSemantic::None one-hot.
    # (The frozen v2 prefix serializes that legacy source value as 1, so the
    # v3 tokenizer recognizes source-less phase kinds explicitly.)
    no_source_game = dz.new_game(dz.Setup(players=2, kingdom=KINGDOM), 0xC15CB00)
    no_source_obs = torch.from_numpy(no_source_game.encode(0, 3)).unsqueeze(0)
    no_source = model.tokenize(no_source_obs)
    assert no_source_obs[0, OBS_DECISION_OFFSET + 1 : OBS_DECISION_OFFSET + 4].sum().item() == 1.0
    no_source_component = no_source.global_token - model.global_projection(no_source.global_features).unsqueeze(1)
    torch.testing.assert_close(no_source_component, torch.zeros_like(no_source_component))

    no_decision_obs = no_source_obs.clone()
    no_decision_obs[:, OBS_DECISION_OFFSET : OBS_DECISION_OFFSET + 15] = 0.0
    no_decision = model.tokenize(no_decision_obs)
    no_decision_component = no_decision.global_token - model.global_projection(no_decision.global_features).unsqueeze(1)
    torch.testing.assert_close(no_decision_component, torch.zeros_like(no_decision_component))


def test_model_factory_selects_the_v3_tokenizer_from_config() -> None:
    model = build_model(
        {"arch": "card_transformer", "obs_version": 3, "d_model": 16, "n_layers": 1, "n_heads": 4},
        dz.OBS_SIZE_V3,
        dz.ACTION_SPACE_SIZE,
    )
    assert isinstance(model, CardTokenNet)
    assert model.obs_version == 3
    assert model.card_projection.in_features == CardTokenNet.CARD_FEATURE_SIZE_V3
    assert model.global_projection.in_features == CardTokenNet.GLOBAL_FEATURE_SIZE_V3


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
    test_v2_tokenizer_regression_keeps_its_feature_layout()
    test_v3_tokenizer_exposes_live_trash_counts()
    test_v3_tokenizer_exposes_select_semantics_and_decision_source()
    test_model_factory_selects_the_v3_tokenizer_from_config()
    test_evaluate_shapes_and_pointer_scatter_against_native_actions()
    print("test_card_transformer: PASS")


if __name__ == "__main__":
    main()
