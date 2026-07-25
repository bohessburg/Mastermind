from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

import dominion_v2_py as dz
import pytest

from src.v2.arena.actuate.clicks import MockActuator, map_engine_actions
from src.v2.arena.archive import GameArchive
from src.v2.arena.fsm.game import BotDecisionProvider, run_game_loop
from src.v2.arena.protocol import events as protocol_events
from src.v2.arena.protocol.events import (
    FullStateCounter,
    FullStateZone,
    GameStart,
    PendingDecision,
    TurnStart,
    UnknownFrame,
)
from src.v2.arena.shadow.bridge import engine_snapshot, game_from_snapshot
from src.v2.arena.shadow.tracker import Tracker, TrackerSnapshot


MILITIA_ABORTS = (
    Path(
        "exports/arena/20260725T014909.665607Z/"
        "20260725T021917.936643Z-game-181370790"
    ),
    Path(
        "exports/arena/20260725T060501.751790Z/"
        "20260725T070852.019220Z-game-181376870"
    ),
    Path(
        "exports/arena/20260725T130411.465938Z/"
        "20260725T140834.589139Z-game-181385305"
    ),
)

_EVENT_CLASSES = {
    name: getattr(protocol_events, name)
    for name in (
        "Attack",
        "Buy",
        "Chat",
        "DecisionResolved",
        "Discard",
        "Draw",
        "FullState",
        "Gain",
        "GameResult",
        "GameStart",
        "PendingDecision",
        "PileReorder",
        "PileUpdate",
        "Play",
        "ResourceUpdate",
        "Shuffle",
        "TimeoutOffer",
        "TurnStart",
        "UnknownFrame",
        "ZoneTransfer",
    )
}
_TUPLE_FIELDS = {
    "Attack": ("targets",),
    "Buy": ("cards",),
    "DecisionResolved": ("answers",),
    "Discard": ("cards",),
    "Draw": ("cards",),
    "Gain": ("cards",),
    "GameResult": ("scores", "placings"),
    "GameStart": ("kingdom", "players", "player_ids"),
    "PendingDecision": ("offered",),
    "PileReorder": ("cards",),
    "Play": ("cards",),
    "ZoneTransfer": ("cards",),
}


def _event_from_archive_row(row: dict[str, object]) -> object:
    """Rehydrate the canonical archive event row without using raw frames."""
    event_type = str(row["event_type"])
    payload = dict(row["event"])
    if event_type == "FullState":
        payload["card_counts"] = tuple(
            (str(name), int(count))
            for name, count in payload["card_counts"]
        )
        payload["zones"] = tuple(
            FullStateZone(
                index=int(zone["index"]),
                kind=str(zone["kind"]),
                owner=zone["owner"],
                display_name=zone["display_name"],
                contents=tuple(str(card) for card in zone["contents"]),
                anonymous_count=int(zone["anonymous_count"]),
            )
            for zone in payload["zones"]
        )
        payload["counters"] = tuple(
            FullStateCounter(
                index=int(counter["index"]),
                name=str(counter["name"]),
                owner=counter["owner"],
                value=int(counter["value"]),
            )
            for counter in payload["counters"]
        )
    else:
        for field in _TUPLE_FIELDS.get(event_type, ()):
            payload[field] = tuple(payload[field])
        if event_type == "UnknownFrame":
            payload["raw"] = bytes.fromhex(str(payload["raw"]))

    return _EVENT_CLASSES[event_type](**payload)


def _archive_events(archive: Path) -> tuple[object, ...]:
    events_path = archive / "events.jsonl"
    assert events_path.is_file(), f"missing archived event fixture: {events_path}"
    return tuple(
        _event_from_archive_row(json.loads(line))
        for line in events_path.read_text(encoding="utf-8").splitlines()
    )


def _first_legal(game: Any, _seat: int) -> int:
    return next(
        action
        for action, legal in enumerate(game.legal_mask())
        if legal
    )


@pytest.mark.parametrize(
    "archive",
    MILITIA_ABORTS,
    ids=lambda path: path.name,
)
def test_archived_militia_shadow_replays_to_two_server_discards(
    archive: Path,
) -> None:
    events = _archive_events(archive)
    militia_indices = [
        index
        for index, event in enumerate(events)
        if isinstance(event, PendingDecision) and event.question_id == "MILITIA"
    ]
    assert len(militia_indices) == 1
    militia_index = militia_indices[0]
    assert not any(
        isinstance(event, TurnStart)
        for event in events[militia_index + 1 :]
    )

    tracker = Tracker()
    for event in events[: militia_index + 1]:
        tracker.consume(event)
    snapshot = tracker.snapshot()
    question = snapshot.pending_decision
    assert question is not None
    assert snapshot.our_seat is not None
    assert question.question_id == "MILITIA"
    assert (question.minimum, question.maximum) == (2, 2)

    our_hand = snapshot.seats[snapshot.our_seat]
    assert our_hand.hand_count == len(question.offered) == 5
    assert Counter(dict(our_hand.hand)) == Counter(question.offered)

    native_snapshot = engine_snapshot(snapshot)
    assert native_snapshot["players"][snapshot.our_seat]["hand"] == {
        int(dz.def_id(name)): count
        for name, count in Counter(question.offered).items()
    }

    game = game_from_snapshot(snapshot)
    assert game.validate()
    assert game.hand(snapshot.our_seat) == native_snapshot["players"][
        snapshot.our_seat
    ]["hand"]
    # The native engine correctly models DiscardDownTo as three keep choices.
    # The server asks for the complementary two discarded physical cards.
    assert game.current_decision()["min"] == 3
    assert game.current_decision()["max"] == 3

    plan = asyncio.run(
        BotDecisionProvider(_first_legal).plan(
            frame_index=militia_index,
            game=game,
            snapshot=snapshot,
            decision=question,
        )
    )
    assert len(plan.engine_actions) == 3
    assert plan.gesture_actions == plan.engine_actions
    mapped = map_engine_actions(plan.gesture_actions, question)
    assert len(mapped.answers) == 2
    assert all(0 <= answer < len(question.offered) for answer in mapped.answers)

    discard_before = game.discard_count(snapshot.our_seat)
    for action in plan.engine_actions:
        assert game.legal_mask()[action]
        game.step(action)
        assert game.validate()
    assert game.hand_count(snapshot.our_seat) == 3
    assert game.discard_count(snapshot.our_seat) - discard_before == 2


def test_recorded_rejection_frame_after_answer_aborts_loudly(
    tmp_path: Path,
) -> None:
    """The captured msg-type-1 answer rejection must not wait for a timeout."""
    screenshots: list[tuple[int, int | None]] = []

    def screenshot(
        frame_index: int,
        snapshot: TrackerSnapshot,
    ) -> Path:
        destination = tmp_path / "server-rejection.png"
        destination.write_text("rejection", encoding="utf-8")
        screenshots.append((frame_index, snapshot.game_id))
        return destination

    events = (
        GameStart(
            game_id=999,
            kingdom=("Militia",),
            players=("bot", "opponent"),
            player_ids=(1, 2),
            our_seat=0,
        ),
        PendingDecision(
            question_index=1,
            decision_type="CHOOSE_MODE",
            question_id="question-411",
            offered=("card-mode-97",),
            minimum=1,
            maximum=1,
            association=None,
        ),
        UnknownFrame(
            msg_type=1,
            direction="in",
            raw=bytes.fromhex("00000055ffffffff"),
            sequence=19176,
            reason="unhandled message type",
        ),
    )
    actuator = MockActuator(replay=True)
    results = asyncio.run(
        run_game_loop(
            events,
            actuator=actuator,
            decision_provider=BotDecisionProvider(_first_legal),
            screenshot_hook=screenshot,
            archive_factory=lambda game_id: GameArchive(tmp_path, game_id=game_id),
        )
    )

    assert len(results) == 1
    result = results[0]
    assert result.divergence_aborted
    assert not result.stall_aborted
    assert not result.completed
    assert "server rejected submitted answer" in result.reason
    assert "00000055ffffffff" in result.reason
    assert result.divergence_report is not None
    assert result.divergence_report.question_index == 1
    assert result.divergence_report.screenshot_path == str(
        tmp_path / "server-rejection.png"
    )
    assert screenshots == [(2, 999)]
    assert actuator.stopped
    assert [gesture.answer_indices for gesture in actuator.gestures] == [(0,)]

    assert result.archive_dir is not None
    archived_events = [
        json.loads(line)
        for line in (Path(result.archive_dir) / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert any(
        event["event_type"] == "UnknownFrame"
        and event["event"]["msg_type"] == 1
        and event["event"]["raw"] == "00000055ffffffff"
        for event in archived_events
    )
