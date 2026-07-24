from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.v2.arena.actuate.clicks import (
    MockActuator,
    possible_answer_indices,
)
from src.v2.arena.fsm.game import (
    DecisionPlan,
    RecordedDecisionProvider,
    run_game_loop,
)
from src.v2.arena.protocol.events import (
    DecisionResolved,
    GameEnd,
    GameStart,
    PendingDecision,
)
from src.v2.arena.protocol.recording import parse_recording
from src.v2.arena.shadow.tracker import (
    PendingDecisionSnapshot,
    TrackerSnapshot,
)


RECORDING = Path(
    "arena-recordings/20260724T142103.096991Z/frames.jsonl"
)
LIVE_GAME_5_ARCHIVE = Path(
    "exports/arena/20260724T214053.394532Z/frames.jsonl"
)
LIVE_GAME_5_DECISIONS = Path(
    "exports/arena/20260724T214053.394532Z/"
    "20260724T215129.423605Z-game-181363699/decisions.jsonl"
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


class _SubmittedArchiveDecisionProvider:
    """Replay the action encodings recorded at submission time."""

    def __init__(
        self,
        events: tuple[object, ...],
        submissions: dict[int, tuple[int, ...]],
    ) -> None:
        self.recorded = RecordedDecisionProvider(events)
        self.submissions = submissions

    async def plan(
        self,
        *,
        frame_index: int,
        game: Any,
        snapshot: TrackerSnapshot,
        decision: PendingDecisionSnapshot,
    ) -> DecisionPlan | None:
        plan = await self.recorded.plan(
            frame_index=frame_index,
            game=game,
            snapshot=snapshot,
            decision=decision,
        )
        if plan is None:
            return None
        answer_hint = self.submissions.get(frame_index)
        return (
            replace(plan, answer_hint=answer_hint)
            if answer_hint is not None
            else plan
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


class _ChatSilentPage:
    """A page-shaped sentinel: only resign can be reached by the game loop."""

    def __init__(self) -> None:
        self.chat_sends: list[str] = []
        self.resign_calls = 0

    async def resign(self) -> None:
        self.resign_calls += 1


def test_replay_and_kingdom_gate_never_send_chat() -> None:
    events = _fixture_events()
    gate = (
        GameStart(
            game_id=999,
            kingdom=("Unknown Test Card",),
            players=("bot", "opponent"),
            player_ids=(1, 2),
            our_seat=0,
        ),
        GameEnd(game_id=999, reason="resigned"),
    )
    replay = (*gate, *events)
    page = _ChatSilentPage()
    actuator = MockActuator(replay=True)

    results = asyncio.run(
        run_game_loop(
            replay,
            actuator=actuator,
            decision_provider=RecordedDecisionProvider(replay),
            resign_hook=page.resign,
        )
    )

    assert results[0].kingdom_rejected
    assert page.resign_calls == 1
    assert page.chat_sends == []
    assert len(actuator.gestures) == 929


def test_arena_source_has_no_chat_send_plumbing() -> None:
    source_root = Path("src/v2/arena")
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in source_root.rglob("*.py")
    )

    for forbidden in (
        "_send_chat",
        "chat_hook",
        "chat_announcement",
        "sendChat",
        "game-chat-input",
    ):
        assert forbidden not in source


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


def test_live_game_five_accepts_the_other_collapsed_witch_index() -> None:
    if not LIVE_GAME_5_ARCHIVE.is_file():
        pytest.skip(f"missing live arena fixture: {LIVE_GAME_5_ARCHIVE}")
    if not LIVE_GAME_5_DECISIONS.is_file():
        pytest.skip(f"missing live decision fixture: {LIVE_GAME_5_DECISIONS}")
    events = parse_recording(LIVE_GAME_5_ARCHIVE).events
    submissions = {
        record["frame_index"]: tuple(record["answers"])
        for line in LIVE_GAME_5_DECISIONS.read_text(encoding="utf-8").splitlines()
        if (record := json.loads(line))
    }
    assert submissions[2892] == (0, 1, 0, 0)
    provider = _SubmittedArchiveDecisionProvider(events, submissions)
    actuator = MockActuator(replay=True)

    results = asyncio.run(
        run_game_loop(
            events,
            actuator=actuator,
            decision_provider=provider,
        )
    )

    game_five = next(result for result in results if result.game_id == 181363699)
    assert game_five.completed
    assert not game_five.divergence_aborted
    assert not actuator.stopped
    assert any(
        gesture.question_index == 92
        and gesture.answer_indices == submissions[2892]
        for gesture in actuator.gestures
    )
