"""Verify server observations after every client action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import dominion_v2_py as dz

from ..protocol.events import (
    Buy,
    DecisionResolved,
    Discard,
    Gain,
    GameEvent,
    Play,
    Topdeck,
    Trash,
)
from ..shadow.tracker import PendingDecisionSnapshot
from .clicks import ClientGesture, offered_name


class DivergenceError(RuntimeError):
    """The server's observed resolution disagreed with the intended action."""

    def __init__(
        self,
        message: str,
        *,
        frame_index: int | None = None,
        question_index: int | None = None,
        intended: tuple[int, ...] = (),
        observed: tuple[int, ...] = (),
    ) -> None:
        super().__init__(message)
        self.frame_index = frame_index
        self.question_index = question_index
        self.intended = intended
        self.observed = observed


@dataclass(frozen=True, kw_only=True)
class IntendedAction:
    """One submitted client answer and the engine plan that produced it."""

    decision: PendingDecisionSnapshot
    engine_actions: tuple[int, ...]
    gesture_actions: tuple[int, ...]
    gesture: ClientGesture
    frame_index: int


def verify_action_events(
    intended: IntendedAction,
    events: Iterable[GameEvent],
    *,
    frame_index: int | None = None,
) -> None:
    """Raise if a completed question or deterministic move disagrees."""
    observed_events = tuple(events)
    resolution = next(
        (
            event
            for event in observed_events
            if isinstance(event, DecisionResolved)
            and event.question_index == intended.decision.question_index
        ),
        None,
    )
    if resolution is None:
        raise DivergenceError(
            "no DecisionResolved event followed the submitted question",
            frame_index=frame_index,
            question_index=intended.decision.question_index,
            intended=intended.gesture.answer_indices,
        )
    if resolution.answers != intended.gesture.answer_indices:
        raise DivergenceError(
            "DecisionResolved answers differ from the submitted answer",
            frame_index=frame_index,
            question_index=intended.decision.question_index,
            intended=intended.gesture.answer_indices,
            observed=resolution.answers,
        )

    expected = _expected_move(intended)
    if expected is None:
        return
    event_type, names = expected
    observed_names: list[str] = []
    for event in observed_events:
        if isinstance(event, event_type):
            observed_names.extend(event.cards)
    missing = list(names)
    for name in observed_names:
        if name in missing:
            missing.remove(name)
    if missing:
        raise DivergenceError(
            f"{event_type.__name__} outcome is missing intended cards "
            f"{tuple(missing)}; observed {tuple(observed_names)}",
            frame_index=frame_index,
            question_index=intended.decision.question_index,
            intended=intended.gesture.answer_indices,
            observed=resolution.answers,
        )


def _expected_move(
    intended: IntendedAction,
) -> tuple[type[GameEvent], tuple[str, ...]] | None:
    decision = intended.decision
    answers = intended.gesture.answer_indices
    labels = tuple(
        offered_name(decision.offered[index])
        for index in intended.gesture.selected_indices
        if 0 <= index < len(decision.offered)
    )

    if decision.question_id == "GAME_ACTION_PHASE":
        if intended.engine_actions == (int(dz.A_PASS),):
            return None
        return (Play, labels)
    if decision.question_id == "GAME_BUY_PHASE":
        if answers == (2, 0):
            treasures = tuple(
                offered_name(value)
                for value in decision.offered
                if value.startswith("1:0:")
            )
            return (Play, treasures)
        if len(answers) == 4 and answers[:2] == (1, 1):
            return (Play, labels)
        if intended.engine_actions == (int(dz.A_PASS),):
            return None
        return (Buy, labels)

    event_type = {
        "ARTISAN_GAIN": Gain,
        "WORKSHOP": Gain,
        "REMODEL_GAIN": Gain,
        "ARTISAN_TOPDECK": Topdeck,
        "REMODEL_TRASH": Trash,
        "CHAPEL": Trash,
        "SENTRY_TRASH": Trash,
        "SENTRY_DISCARD": Discard,
        "POACHER": Discard,
    }.get(decision.question_id)
    if event_type is None or not labels:
        return None
    return (event_type, labels)
