from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from src.v2.arena.protocol.events import Draw, FullState, PendingDecision
from src.v2.arena.protocol.recording import newest_recording, parse_recording
from src.v2.arena.shadow.tracker import Tracker, TrackerError, TrackerSnapshot


HAND_QUESTION_IDS = frozenset(
    {
        "ARTISAN_TOPDECK",
        "CHAPEL",
        "POACHER",
        "REMODEL_TRASH",
        "THRONE_ROOM",
    }
)


def _recording_or_skip() -> Path:
    recording = newest_recording()
    if recording is None:
        pytest.skip("no arena-recordings/*/frames.jsonl fixture is available")
    return recording


def _counter(multiset: tuple[tuple[str, int], ...]) -> Counter[str]:
    return Counter(dict(multiset))


def _assert_non_negative(snapshot: TrackerSnapshot) -> None:
    assert snapshot.trash_count >= 0
    assert snapshot.trash_anonymous >= 0
    assert all(count >= 0 for _, count in snapshot.supply)
    assert all(count >= 0 for _, count in snapshot.trash)
    for seat in snapshot.seats:
        counts = (
            seat.hand_count,
            seat.hand_anonymous,
            seat.deck_count,
            seat.deck_anonymous,
            seat.hand_deck_count,
            seat.hand_deck_unresolved_count,
            seat.discard_count,
            seat.discard_anonymous,
            seat.in_play_count,
            seat.in_play_anonymous,
            seat.set_aside_revealed_count,
            seat.set_aside_anonymous,
            seat.actions,
            seat.buys,
            seat.coins,
        )
        assert all(count >= 0 for count in counts)
        for multiset in (
            seat.hand,
            seat.deck,
            seat.hand_deck,
            seat.discard,
            seat.in_play,
            seat.set_aside_revealed,
            seat.owned,
        ):
            assert all(count >= 0 for _, count in multiset)


def _assert_conservation(snapshot: TrackerSnapshot) -> None:
    if not snapshot.card_totals:
        return
    accounted = _counter(snapshot.supply)
    accounted.update(_counter(snapshot.trash))
    for seat in snapshot.seats:
        accounted.update(_counter(seat.owned))
        assert sum(dict(seat.owned).values()) == (
            seat.hand_count
            + seat.deck_count
            + seat.discard_count
            + seat.in_play_count
            + seat.set_aside_revealed_count
        )
    assert accounted == _counter(snapshot.card_totals)


def _assert_full_state(
    snapshot: TrackerSnapshot,
    full_state: FullState,
) -> None:
    supply = Counter()
    trash = Counter()
    expected_counts: Counter[tuple[int | None, str]] = Counter()
    expected_known: dict[tuple[int | None, str], Counter[str]] = {}
    for zone in full_state.zones:
        if zone.kind == "supply":
            assert zone.display_name is not None
            supply[zone.display_name] = len(zone.contents)
        elif zone.kind == "trash":
            trash.update(zone.contents)
            expected_counts[(None, "trash")] += (
                len(zone.contents) + zone.anonymous_count
            )
        elif zone.kind in {
            "hand",
            "deck",
            "discard",
            "in-play",
            "set-aside",
        }:
            key = (zone.owner, zone.kind)
            expected_counts[key] += len(zone.contents) + zone.anonymous_count
            expected_known.setdefault(key, Counter()).update(zone.contents)

    assert len(snapshot.supply) == 17
    assert dict(snapshot.supply) == dict(supply)
    assert _counter(snapshot.trash) == trash
    assert snapshot.trash_count == expected_counts[(None, "trash")]

    for seat in snapshot.seats:
        observed = {
            "hand": (
                seat.hand_count,
                _counter(seat.hand),
            ),
            "deck": (
                seat.deck_count,
                _counter(seat.deck),
            ),
            "discard": (
                seat.discard_count,
                _counter(seat.discard),
            ),
            "in-play": (
                seat.in_play_count,
                _counter(seat.in_play),
            ),
            "set-aside": (
                seat.set_aside_revealed_count,
                _counter(seat.set_aside_revealed),
            ),
        }
        for kind, (count, known) in observed.items():
            key = (seat.seat, kind)
            assert count == expected_counts[key]
            assert expected_known.get(key, Counter()) <= known

    resources = {
        (counter.owner, counter.name): counter.value
        for counter in full_state.counters
        if counter.owner is not None
    }
    for seat in snapshot.seats:
        assert seat.actions == resources[(seat.seat, "actions")]
        assert seat.buys == resources[(seat.seat, "buys")]
        assert seat.coins == resources[(seat.seat, "coins")]


def _offered_hand_cards(event: PendingDecision) -> tuple[str, ...]:
    if event.question_id == "GAME_ACTION_PHASE":
        return tuple(offered.rsplit(":", 1)[-1] for offered in event.offered)
    if event.question_id == "GAME_BUY_PHASE":
        return tuple(
            offered.rsplit(":", 1)[-1]
            for offered in event.offered
            if offered.startswith("1:0:")
        )
    if event.question_id == "THRONE_ROOM":
        return tuple(
            offered.rsplit(":", 1)[-1]
            for offered in event.offered
            if ":" in offered
        )
    if event.question_id in HAND_QUESTION_IDS:
        return event.offered
    return ()


def test_tracker_golden_recording_reconciles_and_conserves_cards() -> None:
    result = parse_recording(_recording_or_skip())
    tracker = Tracker()
    game_ids: set[int] = set()
    full_states = 0
    replacements = 0
    hand_questions = 0

    for event in result.events:
        tracker.consume(event)
        snapshot = tracker.snapshot()

        if isinstance(event, FullState):
            full_states += 1
            replacements += int(event.replacement)
            game_ids.add(event.game_id)
            _assert_full_state(snapshot, event)

        if isinstance(event, PendingDecision):
            offered = _offered_hand_cards(event)
            if offered:
                hand_questions += 1
                our_seat = snapshot.our_seat
                assert our_seat is not None
                hand = _counter(snapshot.seats[our_seat].hand)
                assert Counter(offered) <= hand

        _assert_non_negative(snapshot)
        _assert_conservation(snapshot)

    assert len(game_ids) == 3
    assert full_states == 4
    assert replacements == 1
    assert hand_questions > 300


def test_movement_count_divergence_raises_tracker_error() -> None:
    tracker = Tracker()
    for event in parse_recording(_recording_or_skip()).events:
        if isinstance(event, Draw):
            with pytest.raises(TrackerError, match="source has"):
                tracker.consume(replace(event, count=11))
            return
        tracker.consume(event)
    raise AssertionError("recording contained no Draw event")
