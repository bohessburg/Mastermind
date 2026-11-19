from __future__ import annotations

import dominion_v2_py as dz
import pytest

from src.v2.arena.actuate.clicks import (
    ClientGesture,
    possible_answer_indices,
)
from src.v2.arena.actuate.verify import (
    DivergenceError,
    IntendedAction,
    verify_action_resolution,
)
from src.v2.arena.protocol.events import DecisionResolved
from src.v2.arena.shadow.tracker import PendingDecisionSnapshot


def _duplicate_witch_intended() -> IntendedAction:
    decision = PendingDecisionSnapshot(
        question_index=92,
        decision_type="COMPLEX_OR",
        question_id="GAME_ACTION_PHASE",
        offered=("0:0:Witch", "0:0:Witch"),
        minimum=0,
        maximum=1,
        association=None,
    )
    witch = int(dz.A_PLAY_BASE + dz.def_id("Witch"))
    acceptable_answers = possible_answer_indices((witch,), decision)
    assert acceptable_answers == ((0, 1, 0, 0), (0, 1, 1, 0))
    return IntendedAction(
        decision=decision,
        engine_actions=(witch,),
        gesture_actions=(witch,),
        gesture=ClientGesture(
            question_index=decision.question_index,
            action=witch,
            prior_actions=(),
            answer_indices=acceptable_answers[0],
            selected_indices=(0,),
            click_button=False,
            labels=("Witch",),
        ),
        acceptable_answers=acceptable_answers,
        frame_index=2892,
    )


def _resolution(
    answers: tuple[int, ...],
    *,
    question_index: int = 92,
) -> DecisionResolved:
    return DecisionResolved(
        timestamp_ms=0,
        question_index=question_index,
        answers=answers,
        seat=1,
        auto_played=False,
    )


def test_resolution_accepts_a_different_mapper_equivalent_answer() -> None:
    intended = _duplicate_witch_intended()

    verify_action_resolution(
        intended,
        _resolution((0, 1, 1, 0)),
    )


def test_resolution_rejects_an_answer_outside_mapper_equivalence() -> None:
    intended = _duplicate_witch_intended()

    with pytest.raises(
        DivergenceError,
        match="DecisionResolved answers differ from the submitted answer",
    ):
        verify_action_resolution(
            intended,
            _resolution((0, 1, 2, 0)),
        )


def test_resolution_rejects_a_mismatched_question_index() -> None:
    intended = _duplicate_witch_intended()

    with pytest.raises(
        DivergenceError,
        match="DecisionResolved question differs from the submitted question",
    ):
        verify_action_resolution(
            intended,
            _resolution((0, 1, 1, 0), question_index=93),
        )
