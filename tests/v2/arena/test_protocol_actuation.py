from __future__ import annotations

import asyncio
from pathlib import Path

import dominion_v2_py as dz

from src.v2.arena.actuate.clicks import possible_answer_indices
from src.v2.arena.actuate.protocol import ProtocolActuator
from src.v2.arena.fsm.game import RecordedDecisionProvider, run_game_loop
from src.v2.arena.protocol.events import DecisionResolved, PendingDecision
from src.v2.arena.protocol.frames import Direction, Writer, decode_frame
from src.v2.arena.protocol.messages import ANSWER_QUESTION, encode_answer
from src.v2.arena.protocol.parser import ArenaParser
from src.v2.arena.protocol.recording import (
    load_recording_sessions,
    parse_recording,
)
from src.v2.arena.shadow.tracker import PendingDecisionSnapshot


REFERENCE_RECORDING = Path(
    "arena-recordings/20260724T142103.096991Z/frames.jsonl"
)
MILITIA_ABORT = Path(
    "exports/arena/20260725T013442.698102Z/frames.jsonl"
)


class _FakeSession:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    async def send_frame(self, msg_type: int, payload_bytes: bytes) -> None:
        self.sent.append(Writer().u32(msg_type).bytes(payload_bytes).build())


def _snapshot(question: PendingDecision) -> PendingDecisionSnapshot:
    return PendingDecisionSnapshot(
        question_index=question.question_index,
        decision_type=question.decision_type,
        question_id=question.question_id,
        offered=question.offered,
        minimum=question.minimum,
        maximum=question.maximum,
        association=question.association,
    )


def _recorded_answer_frames(path: Path) -> tuple[bytes, ...]:
    return tuple(
        frame.raw
        for session in load_recording_sessions(path)
        for frame in session.frames
        if frame.direction is Direction.OUTBOUND
        and frame.msg_type == ANSWER_QUESTION
    )


def test_answer_encoder_round_trips_and_matches_all_recorded_frames() -> None:
    recorded = _recorded_answer_frames(REFERENCE_RECORDING)
    parser = ArenaParser()
    rebuilt: list[bytes] = []

    for raw in recorded:
        frame = decode_frame(raw, Direction.OUTBOUND)
        assert frame is not None
        parsed = parser.parse_frame(frame)
        assert len(parsed) == 1
        answer = parsed[0]
        assert isinstance(answer, DecisionResolved)

        payload = encode_answer(
            answer.question_index,
            answer.answers,
            auto_played=answer.auto_played,
        )
        rebuilt_raw = Writer().u32(ANSWER_QUESTION).bytes(payload).build()
        rebuilt_frame = decode_frame(rebuilt_raw, Direction.OUTBOUND)
        assert rebuilt_frame is not None
        round_trip = parser.parse_frame(rebuilt_frame)
        assert round_trip[-1] == answer
        rebuilt.append(rebuilt_raw)

    assert len(recorded) == 932
    assert tuple(rebuilt) == recorded


def test_three_game_replay_sends_every_answer_through_protocol() -> None:
    events = parse_recording(REFERENCE_RECORDING).events
    provider = RecordedDecisionProvider(events)
    valid_answers: list[tuple[tuple[int, ...], ...]] = []
    for frame_index in provider.recorded_answers:
        question = events[frame_index]
        plan = provider.plan_for_frame(frame_index)
        assert isinstance(question, PendingDecision)
        assert plan is not None
        valid_answers.append(
            possible_answer_indices(
                plan.gesture_actions or (int(dz.A_PASS),),
                _snapshot(question),
            )
        )
    fake_session = _FakeSession()
    actuator = ProtocolActuator(fake_session.send_frame)

    results = asyncio.run(
        run_game_loop(
            events,
            actuator=actuator,
            decision_provider=provider,
        )
    )

    recorded = _recorded_answer_frames(REFERENCE_RECORDING)
    assert len(results) == 3
    assert all(result.completed for result in results)
    assert not any(result.divergence_aborted for result in results)
    assert len(fake_session.sent) == len(actuator.gestures) == len(recorded) == 932

    exact = 0
    parser = ArenaParser()
    for sent, expected, gesture, possible in zip(
        fake_session.sent,
        recorded,
        actuator.gestures,
        valid_answers,
        strict=True,
    ):
        sent_frame = decode_frame(sent, Direction.OUTBOUND)
        expected_frame = decode_frame(expected, Direction.OUTBOUND)
        assert sent_frame is not None
        assert expected_frame is not None
        sent_event = parser.parse_frame(sent_frame)[0]
        expected_event = parser.parse_frame(expected_frame)[0]
        assert isinstance(sent_event, DecisionResolved)
        assert isinstance(expected_event, DecisionResolved)
        assert sent_event.question_index == expected_event.question_index
        assert sent_event.answers == gesture.answer_indices
        assert sent_event.auto_played is False
        assert sent_event.answers in possible
        if gesture.answer_indices == expected_event.answers:
            assert sent == expected
            exact += 1
        else:
            assert expected_event.answers in possible

    assert exact == 807


def test_militia_question_50_is_one_protocol_frame_without_dom_actuation() -> None:
    events = parse_recording(MILITIA_ABORT).events
    question = next(
        event
        for event in events
        if isinstance(event, PendingDecision) and event.question_index == 50
    )
    assert question.question_id == "MILITIA"
    assert question.offered == (
        "Copper",
        "Silver",
        "Silver",
        "Copper",
        "Estate",
    )
    fake_session = _FakeSession()
    actuator = ProtocolActuator(fake_session.send_frame)
    copper = int(dz.A_SELECT_BASE + dz.def_id("Copper"))
    silver = int(dz.A_SELECT_BASE + dz.def_id("Silver"))

    gesture = asyncio.run(
        actuator.act(
            silver,
            _snapshot(question),
            question.offered,
            prior_actions=(copper, silver),
        )
    )

    assert gesture.answer_indices == (3, 4)
    assert fake_session.sent == [
        Writer()
        .u32(ANSWER_QUESTION)
        .bytes(encode_answer(50, (3, 4), auto_played=False))
        .build()
    ]
