from __future__ import annotations

import pytest

from src.v2.arena.protocol.cards import (
    CARD_NAMES,
    CARD_SYMBOLS,
    CARD_TABLE,
    CardId,
    card_id,
    card_name,
    card_symbol,
)


BASE_KINGDOM = {
    "Artisan",
    "Bandit",
    "Bureaucrat",
    "Cellar",
    "Chapel",
    "Council Room",
    "Festival",
    "Gardens",
    "Harbinger",
    "Laboratory",
    "Library",
    "Market",
    "Merchant",
    "Militia",
    "Mine",
    "Moat",
    "Moneylender",
    "Poacher",
    "Remodel",
    "Sentry",
    "Smithy",
    "Throne Room",
    "Vassal",
    "Village",
    "Witch",
    "Workshop",
}


def test_bundle_card_table_is_complete_and_symbol_unique() -> None:
    assert len(CARD_TABLE) == 881
    assert len(CARD_SYMBOLS) == len(set(CARD_SYMBOLS)) == len(CardId)
    assert len(CARD_NAMES) == 881
    assert set(CARD_NAMES[8:34]) == BASE_KINGDOM


def test_base_cards_are_present_and_spot_checked() -> None:
    required = BASE_KINGDOM | {
        "Copper",
        "Silver",
        "Gold",
        "Estate",
        "Duchy",
        "Province",
        "Curse",
    }
    assert required <= set(CARD_NAMES)
    assert CardId.BACK == 0
    assert CardId.CURSE == 1
    assert CardId.COPPER == 2
    assert CardId.ARTISAN == 8
    assert CardId.WORKSHOP == 33
    assert card_name(23) == "Moat"
    assert card_symbol(29) == "THRONE_ROOM"
    assert card_id("Workshop") == CardId.WORKSHOP
    assert card_id("WORKSHOP") == CardId.WORKSHOP


def test_reserved_display_name_requires_unique_symbol() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        card_id("unused")
    assert card_id("UNUSED_SLOT_27").name == "UNUSED_SLOT_27"


@pytest.mark.parametrize("wire_id", [-1, 881])
def test_unknown_wire_id_is_rejected(wire_id: int) -> None:
    with pytest.raises(ValueError):
        card_name(wire_id)
