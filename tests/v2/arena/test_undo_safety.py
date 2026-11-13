"""Regression coverage for the arena's deny-only undo safety policy."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.v2.arena.actuate.clicks import (
    MODAL_CONTAINER_SELECTOR,
    UNDO_DECLINE_SELECTOR,
    UNDO_MODAL_BUTTON_SELECTOR,
    MockActuator,
    PlaywrightActuator,
)
from src.v2.arena.archive import GameArchive
from src.v2.arena.fsm.game import RecordedDecisionProvider, run_game_loop
from src.v2.arena.protocol.events import (
    FullState,
    GameEvent,
    TimeoutOffer,
    UndoRequest,
    UndoResolved,
    UnknownFrame,
)
from src.v2.arena.protocol.recording import (
    load_recording_sessions,
    parse_recording,
)
from src.v2.arena.shadow.tracker import Tracker, TrackerError


UNDO_ARCHIVE = Path(
    "exports/arena/20260725T002942.738597Z/frames.jsonl"
)


def _undo_archive_or_skip() -> Path:
    if not UNDO_ARCHIVE.is_file():
        pytest.skip(f"missing undo regression fixture: {UNDO_ARCHIVE}")
    return UNDO_ARCHIVE


def test_archived_message_35_frames_decode_without_unknowns() -> None:
    archive = _undo_archive_or_skip()
    result = parse_recording(archive)
    frames = [
        frame
        for session in load_recording_sessions(archive)
        for frame in session.frames
        if frame.direction.value == "in" and frame.msg_type == 35
    ]
    events = [
        event
        for event in result.events
        if isinstance(event, (UndoRequest, UndoResolved, TimeoutOffer))
    ]

    assert len(frames) == 7
    assert events == [
        UndoRequest(
            requester_seat=1,
            decision_index=1,
            timestamp_ms=1784939597671,
        ),
        UndoResolved(
            resolution="denied",
            actor_seat=0,
            decision_index=1,
            timestamp_ms=1784939602226,
        ),
        UndoRequest(
            requester_seat=1,
            decision_index=1,
            timestamp_ms=1784939621938,
        ),
        UndoResolved(
            resolution="denied",
            actor_seat=0,
            decision_index=1,
            timestamp_ms=1784939628817,
        ),
        UndoRequest(
            requester_seat=0,
            decision_index=21,
            timestamp_ms=1784939729029,
        ),
        UndoResolved(
            resolution="cancelled",
            actor_seat=0,
            decision_index=21,
            timestamp_ms=1784939734579,
        ),
        UndoRequest(
            requester_seat=0,
            decision_index=39,
            timestamp_ms=1784939772714,
        ),
    ]
    assert not any(
        isinstance(event, UnknownFrame) and event.msg_type == 35
        for event in result.events
    )


@dataclass
class _Button:
    handler: str
    label: str
    clicked: bool = False


class _FakeLocator:
    def __init__(
        self,
        page: "_FakeModalPage",
        selector: str,
        index: int | None = None,
    ) -> None:
        self.page = page
        self.selector = selector
        self.index = index

    def nth(self, index: int) -> "_FakeLocator":
        return _FakeLocator(self.page, self.selector, index)

    async def count(self) -> int:
        return len(self.page.matches(self.selector))

    async def is_visible(self) -> bool:
        return True

    async def click(self) -> None:
        button = self.page.matches(self.selector)[self.index or 0]
        assert isinstance(button, _Button)
        button.clicked = True

    async def get_attribute(self, name: str) -> str | None:
        assert name == "ng-click"
        button = self.page.matches(self.selector)[self.index or 0]
        assert isinstance(button, _Button)
        return button.handler

    async def inner_text(self) -> str:
        button = self.page.matches(self.selector)[self.index or 0]
        assert isinstance(button, _Button)
        return button.label

    async def evaluate(self, _script: str) -> dict[str, object]:
        return self.page.modal_descriptor


class _FakeModalPage:
    def __init__(
        self,
        *,
        exact_decline: _Button | None = None,
        undo_buttons: tuple[_Button, ...] = (),
        modal_descriptor: dict[str, object] | None = None,
    ) -> None:
        self.exact_decline = exact_decline
        self.undo_buttons = undo_buttons
        self.modal_descriptor = modal_descriptor or {
            "owner": "undo-request",
            "text": "Opponent requests an undo. Grant Deny",
            "handlers": ["$ctrl.grant()", "$ctrl.decline()"],
        }
        self.modal_visible = modal_descriptor is not None

    def locator(self, selector: str) -> _FakeLocator:
        return _FakeLocator(self, selector)

    def matches(self, selector: str) -> list[Any]:
        if selector == UNDO_DECLINE_SELECTOR:
            return [] if self.exact_decline is None else [self.exact_decline]
        if selector == UNDO_MODAL_BUTTON_SELECTOR:
            return list(self.undo_buttons)
        if selector == MODAL_CONTAINER_SELECTOR and self.modal_visible:
            return [self.modal_descriptor]
        return []


def test_undo_actuator_clicks_decline_and_never_grant() -> None:
    grant = _Button("$ctrl.grant()", "Grant")
    decline = _Button("$ctrl.decline()", "Deny")
    page = _FakeModalPage(
        exact_decline=decline,
        undo_buttons=(grant, decline),
    )
    actuator = PlaywrightActuator(page)

    clicked = asyncio.run(
        actuator.deny_undo_request(
            UndoRequest(requester_seat=1, decision_index=39)
        )
    )

    assert clicked
    assert decline.clicked
    assert not grant.clicked


def test_unresolved_undo_modal_snapshots_warns_and_does_not_click(
    caplog: pytest.LogCaptureFixture,
) -> None:
    grant = _Button("$ctrl.grant()", "Grant")
    ambiguous = _Button("$ctrl.wait()", "Later")
    page = _FakeModalPage(undo_buttons=(grant, ambiguous))
    snapshots: list[str] = []

    async def snapshot_dom(label: str) -> str:
        snapshots.append(label)
        return f"{label}.html"

    actuator = PlaywrightActuator(
        page,
        snapshot_dom=snapshot_dom,
        actuation_timeout_seconds=0.0,
    )
    with caplog.at_level("ERROR"):
        clicked = asyncio.run(
            actuator.deny_undo_request(
                UndoRequest(requester_seat=1, decision_index=39)
            )
        )

    assert not clicked
    assert not grant.clicked
    assert not ambiguous.clicked
    assert snapshots == ["undo-auto-deny-unresolved-seat-1-decision-39"]
    assert "AUTO-DENY FAILED SAFELY" in caplog.text
    assert UNDO_DECLINE_SELECTOR in caplog.text


def test_unknown_modal_snapshot_is_once_per_visible_occurrence() -> None:
    page = _FakeModalPage(
        modal_descriptor={
            "owner": "future-modal",
            "text": "A newly deployed prompt",
            "handlers": ["$ctrl.choose()"],
        }
    )
    snapshots: list[str] = []

    async def snapshot_dom(label: str) -> str:
        snapshots.append(label)
        return f"{label}.html"

    actuator = PlaywrightActuator(
        page,
        snapshot_dom=snapshot_dom,
        actuation_timeout_seconds=0,
    )
    asyncio.run(actuator.inspect_unknown_modals())
    asyncio.run(actuator.inspect_unknown_modals())
    page.modal_visible = False
    asyncio.run(actuator.inspect_unknown_modals())
    page.modal_visible = True
    asyncio.run(actuator.inspect_unknown_modals())

    assert snapshots == ["unknown-modal", "unknown-modal"]


def test_archive_undo_full_state_resyncs_but_same_state_without_signal_raises(
) -> None:
    events = parse_recording(_undo_archive_or_skip()).events
    target = next(
        event
        for event in events
        if isinstance(event, FullState)
        and event.timestamp_ms == 1784939776355
    )

    tracker = Tracker()
    for event in events:
        tracker.consume(event)
        if event is target:
            break
    assert tracker.last_undo_resync is not None
    assert tracker.last_undo_resync.requester_seat == 0
    assert tracker.last_undo_resync.decision_index == 39
    assert tracker.snapshot().seats[0].hand_count == 5

    no_signal = Tracker()
    for event in events:
        if isinstance(event, (UndoRequest, UndoResolved)):
            continue
        if event is target:
            with pytest.raises(TrackerError, match="FullState zone-count mismatch"):
                no_signal.consume(event)
            break
        no_signal.consume(event)
    else:
        raise AssertionError("undo archive did not contain the target FullState")


class _ResetCountingProvider(RecordedDecisionProvider):
    def __init__(self, events: tuple[GameEvent, ...]) -> None:
        super().__init__(events)
        self.resets = 0

    def reset_after_resync(self) -> None:
        self.resets += 1


def test_archive_loop_invalidates_shadow_and_records_undo_resync(
    tmp_path: Path,
) -> None:
    events = parse_recording(_undo_archive_or_skip()).events
    provider = _ResetCountingProvider(events)
    actuator = MockActuator(replay=True)

    results = asyncio.run(
        run_game_loop(
            events,
            actuator=actuator,
            decision_provider=provider,
            archive_factory=lambda game_id: GameArchive(
                tmp_path,
                game_id=game_id,
            ),
        )
    )

    assert len(results) == 2
    assert all(result.completed for result in results)
    assert not any(result.divergence_aborted for result in results)
    assert sum(result.undo_resyncs for result in results) == 1
    assert provider.resets == 1
    assert [(request.requester_seat, request.decision_index) for request in (
        actuator.undo_denials
    )] == [(1, 1), (1, 1), (0, 21), (0, 39)]
    archived = [
        json.loads(line)
        for path in tmp_path.glob("*-game-181368463/events.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(
        record["event_type"] == "UndoResync"
        for record in archived
    ) == 1


# Both runs aborted on an unrequested FullState: the opponent took back their
# own treasure plays and the server rewound without asking the other seat.
TREASURE_REWIND_ARCHIVES = {
    181415316: (
        Path("exports/arena/20260726T085210.446326Z/frames.jsonl"),
        2,
        2,
    ),
    185890558: (
        Path("exports/arena/20261113T094043.763381Z/frames.jsonl"),
        4,
        2,
    ),
}


def _replay_to_treasure_rewind(
    game_id: int,
) -> tuple[Tracker, tuple[GameEvent, ...], FullState]:
    archive = TREASURE_REWIND_ARCHIVES[game_id][0]
    if not archive.is_file():
        pytest.skip(f"missing treasure-rewind fixture: {archive}")
    events = parse_recording(archive).events
    target = [
        event
        for event in events
        if isinstance(event, FullState) and event.game_id == game_id
    ][-1]
    tracker = Tracker()
    for event in events:
        if event is target:
            break
        tracker.consume(event)
        assert tracker.last_undo_resync is None
    return tracker, events, target


@pytest.mark.parametrize("game_id", sorted(TREASURE_REWIND_ARCHIVES))
def test_unrequested_treasure_rewind_reseeds_the_tracker(game_id: int) -> None:
    tracker, _, target = _replay_to_treasure_rewind(game_id)
    _, hand_count, in_play_count = TREASURE_REWIND_ARCHIVES[game_id]

    tracker.consume(target)

    resync = tracker.last_undo_resync
    assert resync is not None
    assert not resync.signalled
    assert (resync.requester_seat, resync.decision_index) == (0, -1)
    opponent = tracker.snapshot().seats[0]
    assert opponent.hand_count == hand_count
    assert opponent.in_play_count == in_play_count


def test_treasure_rewind_shape_on_our_own_turn_still_raises() -> None:
    tracker, _, target = _replay_to_treasure_rewind(181415316)
    tracker.turn_owner = tracker.our_seat

    with pytest.raises(TrackerError, match="FullState zone-count mismatch"):
        tracker.consume(target)


def test_unrequested_rewind_of_a_non_treasure_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker, _, target = _replay_to_treasure_rewind(181415316)
    monkeypatch.setattr(
        "src.v2.arena.shadow.tracker.TREASURE_NAMES",
        frozenset({"Gold"}),
    )

    with pytest.raises(TrackerError, match="FullState zone-count mismatch"):
        tracker.consume(target)


def test_archive_loop_survives_an_unrequested_treasure_rewind(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, events, _ = _replay_to_treasure_rewind(181415316)

    with caplog.at_level("CRITICAL"):
        results = asyncio.run(
            run_game_loop(
                events,
                actuator=MockActuator(replay=True),
                decision_provider=RecordedDecisionProvider(events),
                archive_factory=lambda game_id: GameArchive(
                    tmp_path,
                    game_id=game_id,
                ),
            )
        )

    # The recording stops right after the rewind, so the rewound game itself
    # has no result; before the fix it ended the loop as a divergence abort.
    assert results
    assert not any(result.divergence_aborted for result in results)
    assert sum(
        "unrequested FullState reseeded game 181415316" in record.getMessage()
        for record in caplog.records
    ) == 1
