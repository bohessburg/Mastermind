"""The game channel moves between WebSockets; the event stream must follow."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import pytest

from src.v2.arena.actuate.clicks import MockActuator
from src.v2.arena.fsm.game import DecisionPlan, run_game_loop
from src.v2.arena.protocol.events import (
    DecisionResolved,
    GameResult,
    GameStart,
    PendingDecision,
    Reconnect,
    SessionStart,
    TurnStart,
    UnknownFrame,
)
from src.v2.arena.protocol.live import events_from_raw_records
from src.v2.arena.protocol.recording import (
    GAME_CHANNEL_OPENING_MSG_TYPES,
    GameChannelSelector,
    parse_recording,
)


# 2026-11-11: after game 185802155 on socket 3 (alpha) the client was moved to
# the beta server on socket 4 for game 185802340. The driver listened only to
# socket 3, never answered the start handshake, and was resigned on timeout.
SERVER_MOVE_ARCHIVE = Path("exports/arena/20261111T084201.458865Z/frames.jsonl")
MOVED_GAME_ID = 185802340
# An in-page reconnect: probes on sockets 4/5 and 7/8, logins on 6 and 9.
IN_PAGE_RECONNECT_ARCHIVE = Path(
    "exports/arena/20260728T223441.580276Z/frames.jsonl"
)


def _record(sock: int, kind: str, direction: str = "in", msg_type: int = 0) -> dict[str, Any]:
    data = ""
    if kind == "binary":
        body = msg_type.to_bytes(4, "big")
        if direction == "in":
            body = (0).to_bytes(4, "big") + body
        data = base64.b64encode(body).decode("ascii")
    return {
        "ts": 0,
        "sock": sock,
        "dir": direction,
        "kind": kind,
        "url": f"wss://example/{sock}",
        "data": data,
        "b64": kind == "binary",
    }


def _selected_sockets(records: list[dict[str, Any]]) -> list[tuple[int, str]]:
    selector = GameChannelSelector(3)
    return [
        (int(selected["sock"]), str(selected["kind"]))
        for record in records
        for selected in selector.select(record)
    ]


def test_selector_ignores_probe_sockets_and_keeps_the_page_load_socket() -> None:
    assert GAME_CHANNEL_OPENING_MSG_TYPES == {24, 46}
    records = [
        _record(1, "open"),
        _record(1, "binary", "out", 45),
        _record(1, "binary", "in", 1),
        _record(1, "close"),
        _record(3, "open"),
        _record(3, "binary", "out", 24),
        _record(4, "open"),
        _record(4, "binary", "out", 45),
        _record(4, "binary", "in", 1),
        _record(3, "binary", "in", 7),
    ]

    assert _selected_sockets(records) == [
        (3, "open"),
        (3, "binary"),
        (3, "binary"),
    ]


@pytest.mark.parametrize("opening", sorted(GAME_CHANNEL_OPENING_MSG_TYPES))
def test_selector_follows_a_login_or_server_move_to_a_new_socket(opening: int) -> None:
    records = [
        _record(3, "open"),
        _record(3, "binary", "out", 24),
        _record(3, "close"),
        _record(4, "open"),
        _record(4, "binary", "out", opening),
        _record(4, "binary", "in", 7),
        _record(3, "binary", "in", 7),
    ]

    assert _selected_sockets(records) == [
        (3, "open"),
        (3, "binary"),
        (3, "close"),
        (4, "open"),
        (4, "binary"),
        (4, "binary"),
    ]


def test_selector_does_not_adopt_a_socket_whose_open_was_not_seen() -> None:
    records = [
        _record(3, "open"),
        _record(5, "binary", "out", 46),
        _record(5, "binary", "in", 7),
    ]

    assert _selected_sockets(records) == [(3, "open")]


def test_selector_handles_socket_ids_restarting_after_a_page_reload() -> None:
    records = [
        _record(3, "open"),
        _record(4, "open"),
        _record(4, "binary", "out", 46),
        # Page reload: the hook numbers sockets from 1 again.
        _record(1, "open"),
        _record(1, "binary", "out", 45),
        _record(3, "open"),
        _record(3, "binary", "out", 24),
        # A later probe reuses id 4 and must not be parsed as the game channel.
        _record(4, "open"),
        _record(4, "binary", "out", 45),
        _record(3, "binary", "in", 7),
    ]

    assert _selected_sockets(records) == [
        (3, "open"),
        (4, "open"),
        (4, "binary"),
        (3, "open"),
        (3, "binary"),
        (3, "binary"),
    ]


def _archive_or_skip(path: Path) -> Path:
    if not path.is_file():
        pytest.skip(f"missing arena archive: {path}")
    return path


def test_server_move_archive_yields_the_game_on_the_new_socket() -> None:
    result = parse_recording(_archive_or_skip(SERVER_MOVE_ARCHIVE))

    assert [session.socket for session in result.sessions] == [3, 4]
    starts = [event for event in result.events if isinstance(event, GameStart)]
    assert [start.game_id for start in starts] == [185802155, MOVED_GAME_ID]
    assert starts[1].our_seat == 1
    reconnects = [event for event in result.events if isinstance(event, Reconnect)]
    assert [event.session_index for event in reconnects] == [1]


def test_live_stream_matches_the_offline_parse_across_a_server_move() -> None:
    path = _archive_or_skip(SERVER_MOVE_ARCHIVE)

    async def collect() -> list[object]:
        async def records() -> Any:
            with path.open(encoding="utf-8") as frames:
                for line in frames:
                    yield json.loads(line)

        return [event async for event in events_from_raw_records(records())]

    assert tuple(asyncio.run(collect())) == parse_recording(path).events


def test_in_page_reconnect_archive_follows_the_login_sockets_only() -> None:
    result = parse_recording(_archive_or_skip(IN_PAGE_RECONNECT_ARCHIVE))

    assert [session.socket for session in result.sessions] == [3, 6, 9]
    assert [
        event.socket for event in result.events if isinstance(event, SessionStart)
    ] == [3, 6, 9]


class _MustNotPlanProvider:
    async def plan(self, **_: Any) -> DecisionPlan:
        raise AssertionError("the start handshake is answered without the provider")


def test_game_after_a_server_move_gets_its_start_handshake_answered() -> None:
    events = list(parse_recording(_archive_or_skip(SERVER_MOVE_ARCHIVE)).events)
    session_start = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, SessionStart) and event.socket == 4
    )
    question_position = next(
        index
        for index in range(session_start, len(events))
        if isinstance(events[index], PendingDecision)
        and events[index].question_id == "question-411"
    )
    question = events[question_position]
    assert isinstance(question, PendingDecision)
    result_position = next(
        index
        for index in range(question_position, len(events))
        if isinstance(events[index], GameResult)
    )
    # The archive has no answer because the old driver never saw the game.
    assert not any(
        isinstance(event, DecisionResolved)
        and event.question_index == question.question_index
        for event in events[question_position:result_position]
    )

    # Replay from the end of the previous game through the question, then
    # supply the continuation the answer unlocks.
    previous_result = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, GameResult) and event.game_id == 185802155
    )
    replay = [
        event
        for event in events[previous_result + 1 : question_position + 1]
        if not (isinstance(event, UnknownFrame) and event.direction == "out")
    ]
    replay += [
        DecisionResolved(
            question_index=question.question_index,
            answers=(0,),
            seat=1,
            auto_played=False,
        ),
        TurnStart(seat=0, turn_number=1, turn_type=0, controller_seat=0),
    ]
    actuator = MockActuator(replay=True)
    asyncio.run(
        run_game_loop(
            tuple(replay),
            actuator=actuator,
            decision_provider=_MustNotPlanProvider(),
        )
    )

    assert [
        (gesture.question_index, gesture.answer_indices)
        for gesture in actuator.gestures
    ] == [(question.question_index, (0,))]
