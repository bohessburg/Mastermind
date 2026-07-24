from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
from pathlib import Path

from src.v2.arena.actuate.clicks import (
    MockActuator,
    possible_answer_indices,
)
from src.v2.arena.fsm.game import (
    RecordedDecisionProvider,
    run_game_loop,
)
from src.v2.arena.protocol.events import DecisionResolved, PendingDecision
from src.v2.arena.protocol.recording import parse_recording
from src.v2.arena.shadow.tracker import PendingDecisionSnapshot


RECORDING = Path(
    "arena-recordings/20260724T142103.096991Z/frames.jsonl"
)


def _fixture_events() -> tuple[object, ...]:
    assert RECORDING.is_file(), f"missing required arena fixture: {RECORDING}"
    return parse_recording(RECORDING).events


def _snapshot(event: PendingDecision) -> PendingDecisionSnapshot:
    return PendingDecisionSnapshot(
        question_index=event.question_index,
        decision_type=event.decision_type,
        question_id=event.question_id,
        offered=event.offered,
        minimum=event.minimum,
        maximum=event.maximum,
        association=event.association,
    )


def test_answer_mapper_covers_every_recorded_answer_shape() -> None:
    events = _fixture_events()
    provider = RecordedDecisionProvider(events)
    coverage: Counter[str] = Counter()
    ambiguous_questions: Counter[str] = Counter()
    questions = tuple(
        event for event in events if isinstance(event, PendingDecision)
    )

    assert len(questions) == 933
    assert len(provider.recorded_answers) == 932

    for frame_index, recorded in provider.recorded_answers.items():
        question = events[frame_index]
        assert isinstance(question, PendingDecision)
        plan = provider.plan_for_frame(frame_index)
        assert plan is not None
        actions = plan.gesture_actions or (0,)
        possible = possible_answer_indices(actions, _snapshot(question))
        assert recorded in possible, (
            frame_index,
            question.question_index,
            question.question_id,
            recorded,
            possible,
        )
        coverage["mapped"] += 1
        coverage["total"] += 1
        if len(possible) > 1:
            coverage["ambiguous"] += 1
            ambiguous_questions[question.question_id] += 1

    assert coverage == Counter(
        {
            "mapped": 932,
            "total": 932,
            "ambiguous": 152,
        }
    )
    # Ambiguity is physical-copy/order ambiguity, never an unmapped answer.
    assert ambiguous_questions == Counter(
        {
            "GAME_ACTION_PHASE": 64,
            "GAME_BUY_PHASE": 28,
            "CHAPEL": 19,
            "THRONE_ROOM": 18,
            "SENTRY_TRASH": 16,
            "POACHER": 3,
            "ARTISAN_TOPDECK": 2,
            "SENTRY_DISCARD": 2,
        }
    )


def test_full_loop_replays_all_three_games_with_validated_rigged_steps() -> None:
    events = _fixture_events()
    provider = RecordedDecisionProvider(events)
    actuator = MockActuator(replay=True)

    results = asyncio.run(
        run_game_loop(
            events,
            actuator=actuator,
            decision_provider=provider,
        )
    )

    assert len(results) == 3
    assert all(result.completed for result in results)
    assert not any(result.divergence_aborted for result in results)
    assert [result.game_id for result in results] == [
        181347875,
        181347888,
        181348150,
    ]
    assert sum(result.decisions for result in results) == 929
    assert sum(result.validations for result in results) >= 929
    assert sum(result.rigged_steps for result in results) == 259
    assert len(actuator.gestures) == 929
    assert not actuator.stopped
    assert all(gesture.answer_indices is not None for gesture in actuator.gestures)


def test_corrupt_recorded_resolution_aborts_before_another_action() -> None:
    events = list(_fixture_events())
    provider = RecordedDecisionProvider(tuple(events))
    pending_index = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, PendingDecision)
        and event.question_id == "GAME_ACTION_PHASE"
    )
    question = events[pending_index]
    assert isinstance(question, PendingDecision)
    resolution_index = next(
        index
        for index in range(pending_index + 1, len(events))
        if isinstance(events[index], DecisionResolved)
        and events[index].question_index == question.question_index
    )
    resolution = events[resolution_index]
    assert isinstance(resolution, DecisionResolved)
    events[resolution_index] = replace(resolution, answers=(99,))
    actuator = MockActuator(replay=True)

    results = asyncio.run(
        run_game_loop(
            tuple(events),
            actuator=actuator,
            decision_provider=provider,
        )
    )

    assert results[-1].divergence_aborted
    assert not results[-1].completed
    assert results[-1].divergence_report is not None
    assert results[-1].divergence_report.error_type == "DivergenceError"
    assert "DecisionResolved answers differ" in results[-1].reason
    assert actuator.stopped
    assert actuator.gestures
    assert actuator.gestures[-1].question_index == question.question_index
