from __future__ import annotations

from src.v2.arena.protocol.events import (
    Buy,
    DecisionResolved,
    Discard,
    Draw,
    Gain,
    PendingDecision,
    PileUpdate,
    Play,
    ResourceUpdate,
    Shuffle,
    Topdeck,
    TurnStart,
)
from src.v2.arena.undo_policy import classify_undo_request


def _resolved(index: int, seat: int = 0) -> DecisionResolved:
    return DecisionResolved(question_index=index, answers=(0,), seat=seat, auto_played=False)


def _move(kind, card: str, source: str, destination: str, seat: int = 0):
    return kind(
        seat=seat,
        cards=(card,),
        count=1,
        from_zone=source,
        to_zone=destination,
        from_zone_index=0,
        to_zone_index=1,
    )


def _workshop_gain() -> list:
    """Seat 0 plays Workshop at decision 10 and gains a Silver at decision 11."""
    return [
        _resolved(9),
        _resolved(10),
        _move(Play, "Workshop", "hand", "in-play"),
        ResourceUpdate(seat=0, resource="actions", counter_index=0, value=0),
        _resolved(11),
        _move(Gain, "Silver", "supply", "discard"),
        PileUpdate(zone="supply", zone_index=3, owner=None, top_card="Silver"),
    ]


def test_grants_an_undo_that_only_took_back_a_gain() -> None:
    assert classify_undo_request(_workshop_gain(), 0, 11).grant
    assert classify_undo_request(_workshop_gain(), 0, 10).grant


def test_grants_a_buy_and_a_topdeck_from_hand() -> None:
    history = [
        _resolved(20),
        Buy(seat=0, cards=("Smithy",), count=1),
        _move(Gain, "Smithy", "supply", "discard"),
        _move(Topdeck, "Gold", "hand", "deck"),
    ]
    assert classify_undo_request(history, 0, 20).grant


def test_denies_anything_that_drew_or_shuffled() -> None:
    drew = [*_workshop_gain(), Draw(seat=0, count=1, cards=(), from_zone="deck", to_zone="hand", from_zone_index=0, to_zone_index=1)]
    assert not classify_undo_request(drew, 0, 11).grant
    shuffled = [*_workshop_gain(), Shuffle(seat=0, included_discard=True)]
    assert not classify_undo_request(shuffled, 0, 11).grant


def test_denies_a_card_taken_off_the_deck() -> None:
    vassal = [_resolved(30), _move(Discard, "Estate", "deck", "discard")]
    verdict = classify_undo_request(vassal, 0, 30)
    assert not verdict.grant
    assert "deck" in verdict.reason


def test_allows_set_aside_cards_back_to_the_deck() -> None:
    sentry = [_resolved(40), _move(Topdeck, "Copper", "set-aside", "deck")]
    assert classify_undo_request(sentry, 0, 40).grant


def test_denies_our_decisions_attacks_and_turn_boundaries() -> None:
    ours = [*_workshop_gain(), _resolved(12, seat=1)]
    assert not classify_undo_request(ours, 0, 11).grant
    asked = [
        *_workshop_gain(),
        PendingDecision(
            question_index=12,
            decision_type="SELECT_CARDS",
            question_id="q",
            offered=(),
            minimum=0,
            maximum=1,
            association=None,
        ),
    ]
    assert not classify_undo_request(asked, 0, 11).grant
    turn = [*_workshop_gain(), TurnStart(seat=1, turn_number=5, turn_type=0, controller_seat=1)]
    assert not classify_undo_request(turn, 0, 11).grant


def test_denies_a_decision_missing_from_the_history() -> None:
    verdict = classify_undo_request(_workshop_gain(), 0, 3)
    assert not verdict.grant
    assert "not in the retained history" in verdict.reason


def test_a_mid_turn_draw_is_denied() -> None:
    smithy = [
        _resolved(60),
        _move(Play, "Smithy", "hand", "in-play"),
        Draw(seat=0, count=3, cards=(), from_zone="deck", to_zone="hand", from_zone_index=0, to_zone_index=1),
        _move(Discard, "Copper", "in-play", "discard"),
    ]
    assert not classify_undo_request(smithy, 0, 60).grant


def test_denies_undoing_a_buy_after_cleanup_drew_the_next_hand() -> None:
    history = [
        _resolved(50),
        Buy(seat=0, cards=("Smithy",), count=1),
        _move(Gain, "Smithy", "supply", "discard"),
        _move(Discard, "Copper", "in-play", "discard"),
        Draw(seat=0, count=5, cards=(), from_zone="deck", to_zone="hand", from_zone_index=0, to_zone_index=1),
    ]
    assert not classify_undo_request(history, 0, 50).grant
