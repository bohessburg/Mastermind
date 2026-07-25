"""Supervised in-game loop with a validated native shadow game."""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from collections import Counter
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

import dominion_v2_py as dz

from ..actuate.clicks import (
    ActionMappingError,
    Actuator,
    ActuationError,
    ClientGesture,
    MockActuator,
    expected_answer_count,
    is_start_confirmation_prompt,
    offered_name,
    possible_answer_indices,
)
from ..actuate.verify import (
    DivergenceError,
    IntendedAction,
    verify_action_events,
    verify_action_resolution,
)
from ..archive import DecisionRecord, GameArchive, ResultSummary
from ..bot.policy import NNPolicy, choose_nnmcts_action
from ..protocol.events import (
    DecisionResolved,
    Discard,
    Draw,
    Chat,
    GameEnd,
    GameEvent,
    GameResult,
    GameStart,
    PendingDecision,
    Play,
    Reconnect,
    Reveal,
    SessionStart,
    Shuffle,
    TimeoutOffer,
    Topdeck,
    UndoRequest,
    UndoResync,
    UnknownFrame,
    ZoneTransfer,
)
from ..shadow.bridge import BridgeError, game_from_snapshot, set_deck_order
from ..shadow.tracker import (
    PendingDecisionSnapshot,
    Tracker,
    TrackerError,
    TrackerSnapshot,
)


LOGGER = logging.getLogger(__name__)
MAX_AUTOPLAY_ATTEMPTS = 2


@dataclass(frozen=True, kw_only=True)
class DecisionPlan:
    """Native actions and client-facing actions for one server question."""

    engine_actions: tuple[int, ...]
    gesture_actions: tuple[int, ...]
    answer_hint: tuple[int, ...] | None = None


class DecisionProvider(Protocol):
    """Choose a complete response for a pending client question."""

    async def plan(
        self,
        *,
        frame_index: int,
        game: Any,
        snapshot: TrackerSnapshot,
        decision: PendingDecisionSnapshot,
    ) -> DecisionPlan | None:
        """Return ``None`` only for a recorded terminal unanswered question."""


@dataclass(frozen=True, kw_only=True)
class DivergenceReport:
    """Full stop context emitted by the loop's tripwire."""

    frame_index: int
    error_type: str
    message: str
    game_id: int | None
    question_index: int | None
    tracker_summary: dict[str, object]
    screenshot_path: str | None


@dataclass(frozen=True, kw_only=True)
class StallReport:
    """Watchdog context emitted when the local client owes progress."""

    frame_index: int
    game_id: int | None
    last_event_timestamp_ms: int | None
    silent_seconds: float
    tracker_summary: dict[str, object]
    pending_question: dict[str, object] | None
    screenshot_path: str | None
    dom_snapshot_path: str | None


@dataclass(frozen=True, kw_only=True)
class GameRunResult:
    """Offline/live outcome for one observed game."""

    game_id: int | None
    completed: bool
    divergence_aborted: bool
    kingdom_rejected: bool
    decisions: int
    validations: int
    rigged_steps: int
    undo_resyncs: int
    reason: str
    divergence_report: DivergenceReport | None = None
    stall_aborted: bool = False
    stall_report: StallReport | None = None
    our_seat: int | None = None
    opponent: str | None = None
    outcome: str = "unknown"
    scores: tuple[int, ...] = ()
    placings: tuple[int, ...] = ()
    winner_seat: int | None = None
    tie: bool = False
    archive_dir: str | None = None


@dataclass
class _ActiveGame:
    game_id: int | None
    players: tuple[str, ...] = ()
    our_seat: int | None = None
    game_result: GameResult | None = None
    kingdom_rejected: bool = False
    decisions: int = 0
    validations: int = 0
    rigged_steps: int = 0
    undo_resyncs: int = 0
    shadow: Any | None = None
    shadow_turn: tuple[int | None, int | None] | None = None
    turn_start_snapshot: TrackerSnapshot | None = None
    turn_history: list[_TurnStep] = field(default_factory=list)
    pending: IntendedAction | None = None
    pending_snapshot: TrackerSnapshot | None = None
    pending_plan: DecisionPlan | None = None
    pending_events: list[GameEvent] = field(default_factory=list)
    timeout_offer: _PendingTimeoutOffer | None = None
    archive: GameArchive | None = None


@dataclass(frozen=True, kw_only=True)
class _TurnStep:
    """A settled native plan retained for shuffle-time turn reconstruction."""

    snapshot: TrackerSnapshot
    actions: tuple[int, ...]
    events: tuple[GameEvent, ...]


@dataclass(frozen=True, kw_only=True)
class _DeferredBuy:
    """A buy/pass selected on the collapsed state before client autoplay."""

    game_id: int | None
    turn_number: int
    seat: int
    action: int
    source_question: PendingDecisionSnapshot
    source_actions: tuple[int, ...]
    acceptable_answers: tuple[tuple[int, ...], ...]
    resolved_answers: tuple[int, ...] | None
    autoplay_attempts: int


@dataclass(frozen=True, kw_only=True)
class _PendingTimeoutOffer:
    """One opponent offer kept live until it is cancelled or claimed."""

    offer: TimeoutOffer
    deadline_at: float
    deadline_timestamp_ms: int | None


@dataclass(frozen=True, kw_only=True)
class _WatchdogStall(GameEvent):
    """Internal sentinel produced instead of waiting forever for an event."""

    frame_index: int
    last_event_timestamp_ms: int | None
    silent_seconds: float


@dataclass(frozen=True, kw_only=True)
class _TimeoutOfferGraceExpired(GameEvent):
    """Internal sentinel emitted after an unresolved offer's grace period."""

    player_seat: int
    decision_index: int


class _StallWatchdogAbort(RuntimeError):
    def __init__(self, stall: _WatchdogStall) -> None:
        super().__init__("stall watchdog expired")
        self.stall = stall


class RecordedDecisionProvider:
    """Replay recorded human answers as native engine action plans."""

    def __init__(self, events: tuple[GameEvent, ...]) -> None:
        self.events = events
        self._answers = _paired_answers(events)
        self._plans = self._build_plans()

    async def plan(
        self,
        *,
        frame_index: int,
        game: Any,
        snapshot: TrackerSnapshot,
        decision: PendingDecisionSnapshot,
    ) -> DecisionPlan | None:
        del game, snapshot, decision
        return self._plans.get(frame_index)

    @property
    def paired_questions(self) -> int:
        return len(self._answers)

    @property
    def recorded_answers(self) -> dict[int, tuple[int, ...]]:
        """Return a copy keyed by the PendingDecision event index."""
        return dict(self._answers)

    def plan_for_frame(self, frame_index: int) -> DecisionPlan | None:
        """Inspect one offline plan without running the async loop."""
        return self._plans.get(frame_index)

    def reset_after_resync(self) -> None:
        """Recorded plans carry no mutable interpreter state."""

    def _build_plans(self) -> dict[int, DecisionPlan]:
        plans: dict[int, DecisionPlan] = {}
        pending_indices = [
            index
            for index, event in enumerate(self.events)
            if isinstance(event, PendingDecision) and index in self._answers
        ]
        pending_position = {
            index: position
            for position, index in enumerate(pending_indices)
        }
        handled: set[int] = set()

        for index in pending_indices:
            if index in handled:
                continue
            event = self.events[index]
            assert isinstance(event, PendingDecision)
            if event.question_id != "SENTRY_TRASH":
                plans[index] = _recorded_plan(event, self._answers[index])
                continue

            tags = _occurrence_tags(event.offered)
            trash_tags = {tags[position] for position in self._answers[index]}
            discard_tags: set[tuple[str, int]] = set()
            discard_index: int | None = None
            topdeck_index: int | None = None
            position = pending_position[index] + 1
            while position < len(pending_indices):
                candidate_index = pending_indices[position]
                candidate = self.events[candidate_index]
                assert isinstance(candidate, PendingDecision)
                if candidate.question_id == "SENTRY_DISCARD":
                    discard_index = candidate_index
                    remaining = [tag for tag in tags if tag not in trash_tags]
                    discard_tags = {
                        remaining[answer]
                        for answer in self._answers[candidate_index]
                    }
                elif candidate.question_id == "SENTRY_TOPDECK":
                    topdeck_index = candidate_index
                else:
                    break
                position += 1

            classifications = tuple(
                int(
                    dz.A_OPTION_BASE
                    + (
                        0
                        if tag in trash_tags
                        else 1
                        if tag in discard_tags
                        else 2
                    )
                )
                for tag in tags
            )
            has_later_stage = discard_index is not None or topdeck_index is not None
            plans[index] = DecisionPlan(
                engine_actions=() if has_later_stage else classifications,
                gesture_actions=classifications,
                answer_hint=self._answers[index],
            )
            handled.add(index)

            if discard_index is not None:
                remaining_actions = tuple(
                    action
                    for tag, action in zip(tags, classifications, strict=True)
                    if tag not in trash_tags
                )
                keep_count = sum(
                    action - int(dz.A_OPTION_BASE) == 2
                    for action in classifications
                )
                engine_actions = (
                    ()
                    if topdeck_index is not None
                    else (
                        *classifications,
                        *(
                            (int(dz.A_OPTION_BASE),)
                            if keep_count == 2
                            else ()
                        ),
                    )
                )
                plans[discard_index] = DecisionPlan(
                    engine_actions=engine_actions,
                    gesture_actions=remaining_actions,
                    answer_hint=self._answers[discard_index],
                )
                handled.add(discard_index)

            if topdeck_index is not None:
                answers = self._answers[topdeck_index]
                if answers == (0, 1):
                    order_action = int(dz.A_OPTION_BASE + 1)
                elif answers == (1, 0):
                    order_action = int(dz.A_OPTION_BASE)
                else:
                    raise ValueError(f"unsupported Sentry order answer {answers}")
                plans[topdeck_index] = DecisionPlan(
                    engine_actions=(*classifications, order_action),
                    gesture_actions=(order_action,),
                    answer_hint=answers,
                )
                handled.add(topdeck_index)
        return plans


class BotDecisionProvider:
    """Adapt one semantic policy callback to the client's batched questions."""

    def __init__(
        self,
        choose_action: Callable[[Any, int], int | Awaitable[int]],
    ) -> None:
        self.choose_action = choose_action
        self._sentry: tuple[
            tuple[tuple[str, int], ...],
            tuple[int, ...],
        ] | None = None
        self._deferred_buy: _DeferredBuy | None = None

    async def plan(
        self,
        *,
        frame_index: int,
        game: Any,
        snapshot: TrackerSnapshot,
        decision: PendingDecisionSnapshot,
    ) -> DecisionPlan:
        if snapshot.our_seat is None:
            raise BridgeError("cannot choose without a local seat")
        seat = snapshot.our_seat

        if self._deferred_buy is not None:
            deferred = self._deferred_buy
            same_context = (
                snapshot.game_id == deferred.game_id
                and snapshot.turn_number == deferred.turn_number
                and seat == deferred.seat
            )
            expected_follow_up = (
                decision.question_id == "GAME_BUY_PHASE"
                and same_context
                and not _offered_treasures(decision)
            )
            if expected_follow_up:
                try:
                    _require_buy_or_pass(game, deferred.action)
                except BridgeError:
                    self._log_deferred_buy_deviation(
                        frame_index=frame_index,
                        snapshot=snapshot,
                        decision=decision,
                        deferred=deferred,
                        reason="deferred buy is no longer legal",
                    )
                    self._deferred_buy = None
                else:
                    self._deferred_buy = None
                    return DecisionPlan(
                        engine_actions=(deferred.action,),
                        gesture_actions=(deferred.action,),
                    )
            elif not same_context:
                self._log_deferred_buy_deviation(
                    frame_index=frame_index,
                    snapshot=snapshot,
                    decision=decision,
                    deferred=deferred,
                    reason="deferred buy belongs to another game, turn, or seat",
                )
                self._deferred_buy = None
            elif decision.question_id == "GAME_BUY_PHASE":
                self._log_deferred_buy_deviation(
                    frame_index=frame_index,
                    snapshot=snapshot,
                    decision=decision,
                    deferred=deferred,
                    reason="follow-up buy question still offers hand treasures",
                )
                self._deferred_buy = None
                return await self._plan_buy(
                    game=game,
                    snapshot=snapshot,
                    decision=decision,
                    seat=seat,
                    autoplay_attempts=deferred.autoplay_attempts,
                )
            else:
                self._log_deferred_buy_deviation(
                    frame_index=frame_index,
                    snapshot=snapshot,
                    decision=decision,
                    deferred=deferred,
                    reason="interleaved question arrived before buy follow-up",
                )

        if decision.question_id == "SENTRY_DISCARD":
            if self._sentry is None:
                raise BridgeError("Sentry discard arrived without classifications")
            tags, classifications = self._sentry
            remaining = tuple(
                action
                for tag, action in zip(tags, classifications, strict=True)
                if action - int(dz.A_OPTION_BASE) != 0
            )
            keep_count = sum(
                action - int(dz.A_OPTION_BASE) == 2
                for action in classifications
            )
            identical_kept = (
                keep_count == 2
                and len(set(decision.offered)) == 1
            )
            engine_actions = (
                (*classifications, int(dz.A_OPTION_BASE))
                if identical_kept
                else ()
                if keep_count == 2
                else classifications
            )
            if engine_actions:
                self._sentry = None
            return DecisionPlan(
                engine_actions=engine_actions,
                gesture_actions=remaining,
            )

        if decision.question_id == "SENTRY_TRASH":
            planning = game.clone()
            actions: list[int] = []
            for _ in decision.offered:
                action = await self._choose(planning, seat)
                if not int(dz.A_OPTION_BASE) <= action < int(dz.A_CALL_BASE):
                    raise BridgeError(
                        f"Sentry classification policy returned {action}"
                    )
                actions.append(action)
                planning.step(action)
            result = tuple(actions)
            self._sentry = (_occurrence_tags(decision.offered), result)
            return DecisionPlan(
                engine_actions=(
                    result
                    if all(
                        action - int(dz.A_OPTION_BASE) == 0
                        for action in result
                    )
                    else ()
                ),
                gesture_actions=result,
            )

        if decision.question_id == "SENTRY_TOPDECK":
            if self._sentry is None:
                raise BridgeError("Sentry order arrived without classifications")
            _, classifications = self._sentry
            planning = game.clone()
            for classification in classifications:
                planning.step(classification)
            action = await self._choose(planning, seat)
            self._sentry = None
            return DecisionPlan(
                engine_actions=(*classifications, action),
                gesture_actions=(action,),
            )

        if decision.question_id == "GAME_BUY_PHASE":
            return await self._plan_buy(
                game=game,
                snapshot=snapshot,
                decision=decision,
                seat=seat,
                autoplay_attempts=0,
            )

        planning = game.clone()
        if expected_answer_count(decision) == 1:
            action = await self._choose(planning, seat)
            return DecisionPlan(
                engine_actions=(action,),
                gesture_actions=(action,),
            )
        initial = _decision_signature(planning)
        actions: list[int] = []
        for _ in range(max(1, decision.maximum + 1)):
            action = await self._choose(planning, seat)
            actions.append(action)
            planning.step(action)
            if _decision_signature(planning) != initial:
                break
            if decision.question_id in {
                "GAME_ACTION_PHASE",
                "GAME_BUY_PHASE",
                "THRONE_ROOM",
            }:
                break
        else:
            raise BridgeError(
                f"policy did not finish question {decision.question_index}"
            )
        result = tuple(actions)
        return DecisionPlan(engine_actions=result, gesture_actions=result)

    async def _choose(self, game: Any, seat: int) -> int:
        value = self.choose_action(game, seat)
        if inspect.isawaitable(value):
            value = await value
        action = int(value)
        legal = game.legal_mask()
        if not 0 <= action < len(legal) or not bool(legal[action]):
            raise BridgeError(f"policy returned illegal action {action}")
        return action

    async def _plan_buy(
        self,
        *,
        game: Any,
        snapshot: TrackerSnapshot,
        decision: PendingDecisionSnapshot,
        seat: int,
        autoplay_attempts: int,
    ) -> DecisionPlan:
        planning = game.clone()
        treasure_actions = _collapse_offered_treasures(planning, decision)
        action = await self._choose(planning, seat)
        _require_buy_or_pass(planning, action)
        if not treasure_actions:
            return DecisionPlan(
                engine_actions=(action,),
                gesture_actions=(action,),
            )
        if snapshot.turn_number is None:
            raise BridgeError("cannot defer a buy without a turn number")

        has_autoplay = any(
            value.endswith("AUTOPLAY_TREASURES")
            for value in decision.offered
        )
        use_autoplay = (
            has_autoplay and autoplay_attempts < MAX_AUTOPLAY_ATTEMPTS
        )
        gesture_actions = (
            treasure_actions if use_autoplay else treasure_actions[:1]
        )
        next_attempts = autoplay_attempts + int(use_autoplay)
        acceptable_answers = possible_answer_indices(
            gesture_actions,
            decision,
        )
        answer_hint = (
            None
            if use_autoplay
            else next(
                answers
                for answers in acceptable_answers
                if len(answers) == 4 and answers[:2] == (1, 1)
            )
        )
        self._deferred_buy = _DeferredBuy(
            game_id=snapshot.game_id,
            turn_number=snapshot.turn_number,
            seat=seat,
            action=action,
            source_question=decision,
            source_actions=gesture_actions,
            acceptable_answers=acceptable_answers,
            resolved_answers=None,
            autoplay_attempts=next_attempts,
        )
        if not use_autoplay:
            LOGGER.warning(
                "AUTOPLAY BUY FALLBACK: playing one treasure explicitly after "
                "%d autoplay attempts; snapshot=%r question=%r actions=%r "
                "answers=%r",
                autoplay_attempts,
                _buy_snapshot_context(snapshot),
                _question_context(decision),
                gesture_actions,
                acceptable_answers,
            )
        return DecisionPlan(
            engine_actions=gesture_actions,
            gesture_actions=gesture_actions,
            answer_hint=answer_hint,
        )

    def observe_resolution(self, resolution: DecisionResolved) -> None:
        """Retain the accepted answer for deviation diagnostics."""
        deferred = self._deferred_buy
        if (
            deferred is not None
            and deferred.source_question.question_index
            == resolution.question_index
        ):
            self._deferred_buy = replace(
                deferred,
                resolved_answers=resolution.answers,
            )

    def reset_for_game(self, game_id: int | None) -> None:
        """Discard question-chain state before a newly observed game."""
        del game_id
        self._reset_transient_state()

    def reset_after_game(self, game_id: int | None) -> None:
        """Discard question-chain state once a game-ending frame arrives."""
        del game_id
        self._reset_transient_state()

    def reset_after_resync(self) -> None:
        """Discard multi-question state derived from the pre-undo shadow."""
        self._reset_transient_state()

    def _reset_transient_state(self) -> None:
        self._sentry = None
        self._deferred_buy = None

    @staticmethod
    def _log_deferred_buy_deviation(
        *,
        frame_index: int,
        snapshot: TrackerSnapshot,
        decision: PendingDecisionSnapshot,
        deferred: _DeferredBuy,
        reason: str,
    ) -> None:
        LOGGER.warning(
            "AUTOPLAY BUY DEVIATION: %s; frame=%d deferred=%r current=%r",
            reason,
            frame_index,
            {
                "game_id": deferred.game_id,
                "turn_number": deferred.turn_number,
                "seat": deferred.seat,
                "buy_action": deferred.action,
                "autoplay_attempts": deferred.autoplay_attempts,
                "source_question": _question_context(
                    deferred.source_question
                ),
                "source_actions": deferred.source_actions,
                "acceptable_answers": deferred.acceptable_answers,
                "resolved_answers": deferred.resolved_answers,
            },
            {
                "snapshot": _buy_snapshot_context(snapshot),
                "question": _question_context(decision),
            },
        )


class NNMCTSDecisionProvider(BotDecisionProvider):
    """The production provider backed by the shared P4 NN-MCTS service."""

    def __init__(
        self,
        policy: NNPolicy,
        *,
        sims: int = 400,
        determinizations: int = 2,
        wall_clock_cap: float | None = None,
    ) -> None:
        def choose(game: Any, seat: int) -> int:
            return choose_nnmcts_action(
                game,
                seat,
                policy,
                sims=sims,
                determinizations=determinizations,
                wall_clock_cap=wall_clock_cap,
            )

        super().__init__(choose)


async def run_game_loop(
    events: Iterable[GameEvent] | AsyncIterable[GameEvent],
    *,
    actuator: Actuator,
    decision_provider: DecisionProvider,
    think_time_min_seconds: float = 0.0,
    think_time_max_seconds: float = 0.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    random_source: random.Random | None = None,
    screenshot_hook: Callable[
        [int, TrackerSnapshot], str | Path | None | Awaitable[str | Path | None]
    ]
    | None = None,
    resign_hook: Callable[[], object | Awaitable[object]] | None = None,
    archive_factory: Callable[[int | None], GameArchive] | None = None,
    max_games: int | None = None,
    auto_deny_undo: bool = True,
    timeout_offer_grace_seconds: float = 30.0,
    stall_watchdog_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    watchdog_poll_seconds: float = 0.25,
    stall_screenshot_hook: Callable[
        [int, TrackerSnapshot], str | Path | None | Awaitable[str | Path | None]
    ]
    | None = None,
    stall_dom_hook: Callable[
        [int, TrackerSnapshot], str | Path | None | Awaitable[str | Path | None]
    ]
    | None = None,
) -> tuple[GameRunResult, ...]:
    """Consume normalized events and play until every observed game ends.

    Any tracker/bridge/verifier failure stops the entire actuator immediately.
    Continuing after a divergence would risk making a move in an unknown state.
    """
    if think_time_min_seconds < 0:
        raise ValueError("minimum think time cannot be negative")
    if think_time_max_seconds < think_time_min_seconds:
        raise ValueError("maximum think time must be at least the minimum")
    if max_games is not None and max_games <= 0:
        raise ValueError("max_games must be positive or null")
    if timeout_offer_grace_seconds < 0:
        raise ValueError("timeout-offer grace must be non-negative")
    if stall_watchdog_seconds is not None and stall_watchdog_seconds <= 0:
        raise ValueError("stall watchdog must be positive or null")
    if watchdog_poll_seconds <= 0:
        raise ValueError("watchdog poll interval must be positive")
    if not auto_deny_undo:
        raise ValueError("undo auto-deny is the only supported arena behavior")
    rng = random_source or random.Random()
    tracker = Tracker()
    active: _ActiveGame | None = None
    results: list[GameRunResult] = []

    async def settle(frame_index: int) -> None:
        nonlocal active
        assert active is not None
        if active.pending is None:
            return
        assert active.pending_snapshot is not None
        assert active.pending_plan is not None
        current_snapshot = tracker.snapshot()
        partial_autoplay = _recoverable_partial_autoplay(
            active.pending,
            active.pending_events,
            active.pending_snapshot,
            current_snapshot,
        )
        if partial_autoplay:
            resolution = next(
                event
                for event in active.pending_events
                if isinstance(event, DecisionResolved)
                and event.question_index
                == active.pending.decision.question_index
            )
            # Keep answer verification strict; only the deterministic Play
            # outcome is allowed to be partial when the new prompt explicitly
            # offers the missing treasures again.
            verify_action_resolution(
                active.pending,
                resolution,
                frame_index=frame_index,
            )
            LOGGER.warning(
                "AUTOPLAY BUY PARTIAL OUTCOME: rebuilding from tracker; "
                "frame=%d submitted=%r current=%r observed_events=%r",
                frame_index,
                {
                    "question": _question_context(active.pending.decision),
                    "actions": active.pending.gesture_actions,
                    "submitted_answers": (
                        active.pending.gesture.answer_indices
                    ),
                    "acceptable_answers": (
                        active.pending.acceptable_answers
                    ),
                    "resolved_answers": resolution.answers,
                },
                {
                    "snapshot": _buy_snapshot_context(current_snapshot),
                    "question": _question_context(
                        current_snapshot.pending_decision
                    ),
                },
                tuple(active.pending_events),
            )
        else:
            verify_action_events(
                active.pending,
                active.pending_events,
                frame_index=frame_index,
            )
        crossed_turn_boundary = (
            current_snapshot.turn_number
            != active.pending_snapshot.turn_number
            or current_snapshot.turn_owner
            != active.pending_snapshot.turn_owner
        )
        if partial_autoplay:
            active.shadow = None
            active.shadow_turn = None
            active.turn_start_snapshot = None
            active.turn_history.clear()
        elif active.pending_plan.engine_actions:
            current_step = _TurnStep(
                snapshot=active.pending_snapshot,
                actions=active.pending_plan.engine_actions,
                events=tuple(active.pending_events),
            )
            if crossed_turn_boundary:
                # The next local decision is rebuilt authoritatively below;
                # no stale end-of-turn interpreter state is carried forward.
                active.shadow = None
                active.shadow_turn = None
                active.turn_start_snapshot = None
                active.turn_history.clear()
            elif _has_local_shuffle(
                active.pending_snapshot,
                active.pending_events,
            ):
                _rebuild_turn_after_shuffle(active, current_step)
                active.rigged_steps += 1
            else:
                if _rig_observed_draws(
                    active.shadow,
                    active.pending_snapshot,
                    active.pending_events,
                ):
                    active.rigged_steps += 1
                _step_actions(
                    active.shadow,
                    active.pending_plan.engine_actions,
                    frame_index=frame_index,
                    intended=active.pending,
                )
            if not crossed_turn_boundary:
                active.turn_history.append(current_step)
        active.pending = None
        active.pending_snapshot = None
        active.pending_plan = None
        active.pending_events.clear()

    async for frame_index, event in _indexed_events(
        events,
        obligation=lambda: _has_outstanding_obligation(active, tracker),
        timeout_offer=lambda: (
            active.timeout_offer if active is not None else None
        ),
        stall_watchdog_seconds=stall_watchdog_seconds,
        clock=clock,
        sleep=sleep,
        watchdog_poll_seconds=watchdog_poll_seconds,
    ):
        try:
            if isinstance(event, _WatchdogStall):
                raise _StallWatchdogAbort(event)
            if isinstance(event, _TimeoutOfferGraceExpired):
                if (
                    active is not None
                    and active.timeout_offer is not None
                    and active.timeout_offer.offer.player_seat
                    == event.player_seat
                    and active.timeout_offer.offer.decision_index
                    == event.decision_index
                ):
                    offer = active.timeout_offer.offer
                    # Clear before sending so a reentrant archive frame cannot
                    # produce a duplicate claim while the socket call awaits.
                    active.timeout_offer = None
                    claimed = await actuator.claim_timeout_offer(offer)
                    if claimed:
                        LOGGER.warning(
                            "CLAIMED opponent timeout after %.1fs: seat %d "
                            "decision %d",
                            timeout_offer_grace_seconds,
                            offer.player_seat,
                            offer.decision_index,
                        )
                    else:
                        LOGGER.error(
                            "TIMEOUT CLAIM FAILED SAFELY: no actuation for "
                            "opponent seat %d decision %d after %.1fs",
                            offer.player_seat,
                            offer.decision_index,
                            timeout_offer_grace_seconds,
                        )
                continue
            tracker.consume(event)
            undo_resync = tracker.last_undo_resync
            if active is not None:
                active.pending_events.append(event)
                if active.archive is not None:
                    active.archive.append_event(frame_index, event)
                if (
                    active.pending is not None
                    and isinstance(event, DecisionResolved)
                    and event.question_index
                    == active.pending.decision.question_index
                ):
                    verify_action_resolution(
                        active.pending,
                        event,
                        frame_index=frame_index,
                    )
                    observe_resolution = getattr(
                        decision_provider,
                        "observe_resolution",
                        None,
                    )
                    if observe_resolution is not None:
                        await _maybe_await(observe_resolution(event))

            if undo_resync is not None:
                if active is None:
                    raise TrackerError(
                        "undo FullState resync occurred without an active game"
                    )
                active.shadow = None
                active.shadow_turn = None
                active.turn_start_snapshot = None
                active.turn_history.clear()
                active.pending = None
                active.pending_snapshot = None
                active.pending_plan = None
                active.pending_events.clear()
                active.undo_resyncs += 1
                reset_provider = getattr(
                    decision_provider,
                    "reset_after_resync",
                    None,
                )
                if reset_provider is not None:
                    await _maybe_await(reset_provider())
                marker = UndoResync(
                    game_id=undo_resync.game_id,
                    requester_seat=undo_resync.requester_seat,
                    decision_index=undo_resync.decision_index,
                    reason="mismatching FullState after recent undo signal",
                    timestamp_ms=undo_resync.full_state_timestamp_ms,
                )
                if active.archive is not None:
                    active.archive.append_event(frame_index, marker)
                LOGGER.critical(
                    "UNDO RESYNC: authoritative FullState reseeded game %d "
                    "after seat %d requested decision %d; invalidated the "
                    "mid-turn shadow and pending actuation",
                    undo_resync.game_id,
                    undo_resync.requester_seat,
                    undo_resync.decision_index,
                )
                continue

            if isinstance(event, GameStart):
                if active is None or active.game_id != event.game_id:
                    reset_for_game = getattr(
                        decision_provider,
                        "reset_for_game",
                        None,
                    )
                    if reset_for_game is not None:
                        await _maybe_await(reset_for_game(event.game_id))
                    active = _ActiveGame(
                        game_id=event.game_id,
                        players=event.players,
                        our_seat=event.our_seat,
                        archive=(
                            archive_factory(event.game_id)
                            if archive_factory is not None
                            else None
                        ),
                    )
                    if active.archive is not None:
                        active.archive.append_event(frame_index, event)
                    unknown = _unknown_kingdom(event.kingdom)
                    if unknown:
                        active.kingdom_rejected = True
                        await _maybe_await(resign_hook)
                continue

            if active is None:
                continue

            if isinstance(event, GameResult):
                active.game_result = event

            if (
                active.timeout_offer is not None
                and _timeout_offer_was_cancelled(
                    event,
                    active.timeout_offer.offer,
                )
            ):
                active.timeout_offer = None

            if isinstance(event, TimeoutOffer):
                snapshot = tracker.snapshot()
                if snapshot.our_seat is None:
                    LOGGER.error(
                        "TIMEOUT OFFER IGNORED: local seat is unknown for "
                        "seat %d decision %d",
                        event.player_seat,
                        event.decision_index,
                    )
                elif _timeout_offer_is_for_our_seat(event, snapshot):
                    LOGGER.critical(
                        "TIMEOUT OFFER FOR OUR SEAT: seat %d decision %d; "
                        "never self-resigning through the timeout path",
                        event.player_seat,
                        event.decision_index,
                    )
                else:
                    active.timeout_offer = _PendingTimeoutOffer(
                        offer=event,
                        deadline_at=clock() + timeout_offer_grace_seconds,
                        deadline_timestamp_ms=(
                            None
                            if event.timestamp_ms is None
                            else event.timestamp_ms
                            + round(timeout_offer_grace_seconds * 1_000)
                        ),
                    )
                continue

            if isinstance(event, UndoRequest):
                snapshot = tracker.snapshot()
                if (
                    snapshot.our_seat is not None
                    and event.requester_seat != snapshot.our_seat
                ):
                    denied = await actuator.deny_undo_request(event)
                    if not denied:
                        LOGGER.error(
                            "AUTO-DENY did not click a control for opponent "
                            "seat %d decision %d; continuing safely while the "
                            "server request times out",
                            event.requester_seat,
                            event.decision_index,
                        )
                continue

            if (
                isinstance(event, UnknownFrame)
                and event.direction == "in"
                and event.msg_type == 37
            ):
                question_index = (
                    int.from_bytes(event.raw[:4], "big", signed=True)
                    if len(event.raw) >= 4
                    else None
                )
                raise DivergenceError(
                    "unparseable inbound questionAsked while a game is active: "
                    f"sequence={event.sequence} bytes={len(event.raw)} "
                    f"reason={event.reason or 'unknown parser failure'}",
                    frame_index=frame_index,
                    question_index=question_index,
                )

            if isinstance(event, Reconnect):
                active.shadow = None
                active.shadow_turn = None
                active.turn_start_snapshot = None
                active.turn_history.clear()
                active.pending = None
                active.pending_snapshot = None
                active.pending_plan = None
                active.pending_events.clear()
                active.timeout_offer = None
                continue

            if isinstance(event, GameEnd):
                active.timeout_offer = None
                if (
                    active.pending is None
                    or any(
                        isinstance(observed, DecisionResolved)
                        and observed.question_index
                        == active.pending.decision.question_index
                        for observed in active.pending_events
                    )
                ):
                    await settle(frame_index)
                else:
                    # A server-side game end supersedes an outstanding client
                    # prompt; the fixture contains both unanswered and
                    # post-end late-answer examples.
                    active.pending = None
                    active.pending_snapshot = None
                    active.pending_plan = None
                    active.pending_events.clear()
                result = GameRunResult(
                    game_id=active.game_id,
                    completed=True,
                    divergence_aborted=False,
                    kingdom_rejected=active.kingdom_rejected,
                    decisions=active.decisions,
                    validations=active.validations,
                    rigged_steps=active.rigged_steps,
                    undo_resyncs=active.undo_resyncs,
                    reason=event.reason,
                    **_result_fields(active),
                    archive_dir=(
                        str(active.archive.path)
                        if active.archive is not None
                        else None
                    ),
                )
                _finish_archive(active, result)
                results.append(result)
                reset_after_game = getattr(
                    decision_provider,
                    "reset_after_game",
                    None,
                )
                if reset_after_game is not None:
                    await _maybe_await(reset_after_game(active.game_id))
                active = None
                if max_games is not None and len(results) >= max_games:
                    return tuple(results)
                continue

            if not isinstance(event, PendingDecision):
                continue
            snapshot = tracker.snapshot()
            # questionAsked is a client-private inbound frame: the parser emits
            # PendingDecision only for the connected local seat. Opponent
            # questions appear here only as seat-tagged DecisionResolved events.
            if snapshot.pending_decision is None:
                raise DivergenceError(
                    "PendingDecision was not retained by the tracker; refusing "
                    "to leave a local client question unanswered",
                    frame_index=frame_index,
                    question_index=event.question_index,
                )
            if snapshot.our_seat is None:
                raise DivergenceError(
                    "PendingDecision arrived before the local seat was known; "
                    "refusing to leave the client question unanswered",
                    frame_index=frame_index,
                    question_index=event.question_index,
                )
            if active.kingdom_rejected:
                raise DivergenceError(
                    "PendingDecision arrived after the kingdom was rejected; "
                    "the resign path did not clear the local client question",
                    frame_index=frame_index,
                    question_index=event.question_index,
                )

            if (
                snapshot.turn_owner is None
                and snapshot.turn_number is None
                and is_start_confirmation_prompt(snapshot.pending_decision)
            ):
                await settle(frame_index)
                plan = DecisionPlan(
                    engine_actions=(),
                    gesture_actions=(int(dz.A_OPTION_BASE),),
                    answer_hint=(0,),
                )
                gesture = await _act_with_hint(
                    actuator,
                    plan.gesture_actions[0],
                    snapshot.pending_decision,
                    snapshot.pending_decision.offered,
                    prior_actions=(),
                    answer_hint=plan.answer_hint,
                )
                intended = IntendedAction(
                    decision=snapshot.pending_decision,
                    engine_actions=plan.engine_actions,
                    gesture_actions=plan.gesture_actions,
                    gesture=gesture,
                    acceptable_answers=possible_answer_indices(
                        gesture.actions,
                        snapshot.pending_decision,
                    ),
                    frame_index=frame_index,
                )
                active.pending = intended
                active.pending_snapshot = snapshot
                active.pending_plan = plan
                active.pending_events = []
                active.decisions += 1
                if active.archive is not None:
                    active.archive.append_decision(
                        DecisionRecord(
                            frame_index=frame_index,
                            question_index=event.question_index,
                            question_id=event.question_id,
                            engine_actions=plan.engine_actions,
                            gesture_actions=plan.gesture_actions,
                            answers=gesture.answer_indices,
                            offered=event.offered,
                        )
                    )
                continue
            if snapshot.turn_owner is None or snapshot.turn_number is None:
                raise DivergenceError(
                    "unexpected pre-turn PendingDecision addressed to the local "
                    "client; refusing to leave it unanswered: "
                    f"type={event.decision_type!r} id={event.question_id!r} "
                    f"offered={event.offered!r} minimum={event.minimum} "
                    f"maximum={event.maximum}",
                    frame_index=frame_index,
                    question_index=event.question_index,
                )

            await settle(frame_index)
            turn_key = (snapshot.game_id, snapshot.turn_number)
            our_turn = snapshot.turn_owner == snapshot.our_seat
            clean_phase_boundary = event.question_id in {
                "GAME_ACTION_PHASE",
                "GAME_BUY_PHASE",
            }
            if (
                active.shadow is None
                or (our_turn and active.shadow_turn != turn_key)
                or not our_turn
                or (our_turn and clean_phase_boundary)
            ):
                active.shadow = game_from_snapshot(snapshot)
                active.shadow_turn = turn_key if our_turn else None
                active.turn_start_snapshot = snapshot if our_turn else None
                active.turn_history.clear()
            active.shadow.validate()
            active.validations += 1
            forced_actions = _advance_forced_to_client_question(
                active.shadow,
                snapshot.pending_decision,
                frame_index=frame_index,
            )
            if forced_actions:
                active.turn_history.append(
                    _TurnStep(
                        snapshot=snapshot,
                        actions=forced_actions,
                        events=(),
                    )
                )
                active.shadow.validate()
                active.validations += 1

            plan = await decision_provider.plan(
                frame_index=frame_index,
                game=active.shadow,
                snapshot=snapshot,
                decision=snapshot.pending_decision,
            )
            if plan is None:
                # The reference fixture has one Sentry question superseded by
                # an immediate GameEnd.  A live provider never returns None.
                continue
            gesture_actions = plan.gesture_actions or (int(dz.A_PASS),)
            delay = rng.uniform(
                think_time_min_seconds,
                think_time_max_seconds,
            )
            if delay:
                await sleep(delay)
            gesture = await _act_with_hint(
                actuator,
                gesture_actions[-1],
                snapshot.pending_decision,
                snapshot.pending_decision.offered,
                prior_actions=gesture_actions[:-1],
                answer_hint=plan.answer_hint,
            )
            intended = IntendedAction(
                decision=snapshot.pending_decision,
                engine_actions=plan.engine_actions,
                gesture_actions=plan.gesture_actions,
                gesture=gesture,
                acceptable_answers=possible_answer_indices(
                    gesture.actions,
                    snapshot.pending_decision,
                ),
                frame_index=frame_index,
            )
            active.pending = intended
            active.pending_snapshot = snapshot
            active.pending_plan = plan
            active.pending_events = []
            active.decisions += 1
            if active.archive is not None:
                active.archive.append_decision(
                    DecisionRecord(
                        frame_index=frame_index,
                        question_index=event.question_index,
                        question_id=event.question_id,
                        engine_actions=plan.engine_actions,
                        gesture_actions=plan.gesture_actions,
                        answers=gesture.answer_indices,
                        offered=event.offered,
                    )
                )
        except _StallWatchdogAbort as error:
            snapshot = tracker.snapshot()
            screenshot = await _capture_artifact(
                stall_screenshot_hook,
                error.stall.frame_index,
                snapshot,
                label="stall screenshot",
            )
            dom_snapshot = await _capture_artifact(
                stall_dom_hook,
                error.stall.frame_index,
                snapshot,
                label="stall DOM snapshot",
            )
            report = StallReport(
                frame_index=error.stall.frame_index,
                game_id=snapshot.game_id,
                last_event_timestamp_ms=error.stall.last_event_timestamp_ms,
                silent_seconds=error.stall.silent_seconds,
                tracker_summary=_snapshot_summary(snapshot),
                pending_question=_pending_question_summary(snapshot),
                screenshot_path=screenshot,
                dom_snapshot_path=dom_snapshot,
            )
            LOGGER.critical(
                "STALL WATCHDOG: game=%s frame=%d silent=%.1fs "
                "last_event_timestamp_ms=%s pending=%s tracker=%s",
                snapshot.game_id,
                report.frame_index,
                report.silent_seconds,
                report.last_event_timestamp_ms,
                report.pending_question,
                report.tracker_summary,
            )
            if isinstance(actuator, MockActuator):
                actuator.stop()
            if active is None:
                active = _ActiveGame(game_id=snapshot.game_id)
            result = GameRunResult(
                game_id=active.game_id,
                completed=False,
                divergence_aborted=False,
                kingdom_rejected=active.kingdom_rejected,
                decisions=active.decisions,
                validations=active.validations,
                rigged_steps=active.rigged_steps,
                undo_resyncs=active.undo_resyncs,
                reason=(
                    "stall watchdog expired after "
                    f"{report.silent_seconds:.1f}s"
                ),
                stall_aborted=True,
                stall_report=report,
                **_result_fields(active),
                archive_dir=(
                    str(active.archive.path)
                    if active.archive is not None
                    else None
                ),
            )
            _finish_archive(active, result)
            results.append(result)
            return tuple(results)
        except (
            TrackerError,
            BridgeError,
            DivergenceError,
            ActionMappingError,
            ActuationError,
        ) as error:
            snapshot = tracker.snapshot()
            screenshot = await _capture_screenshot(
                screenshot_hook,
                frame_index,
                snapshot,
            )
            report = DivergenceReport(
                frame_index=frame_index,
                error_type=type(error).__name__,
                message=str(error),
                game_id=snapshot.game_id,
                question_index=(
                    error.question_index
                    if isinstance(error, DivergenceError)
                    and error.question_index is not None
                    else (
                        snapshot.pending_decision.question_index
                        if snapshot.pending_decision is not None
                        else None
                    )
                ),
                tracker_summary=_snapshot_summary(snapshot),
                screenshot_path=screenshot,
            )
            if isinstance(actuator, MockActuator):
                actuator.stop()
            if active is None:
                active = _ActiveGame(game_id=snapshot.game_id)
            result = GameRunResult(
                game_id=active.game_id,
                completed=False,
                divergence_aborted=True,
                kingdom_rejected=active.kingdom_rejected,
                decisions=active.decisions,
                validations=active.validations,
                rigged_steps=active.rigged_steps,
                undo_resyncs=active.undo_resyncs,
                reason=str(error),
                divergence_report=report,
                **_result_fields(active),
                archive_dir=(
                    str(active.archive.path)
                    if active.archive is not None
                    else None
                ),
            )
            _finish_archive(active, result)
            results.append(result)
            return tuple(results)

    return tuple(results)


async def _indexed_events(
    events: Iterable[GameEvent] | AsyncIterable[GameEvent],
    *,
    obligation: Callable[[], bool] | None = None,
    timeout_offer: Callable[[], _PendingTimeoutOffer | None] | None = None,
    stall_watchdog_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    watchdog_poll_seconds: float = 0.25,
) -> AsyncIterator[tuple[int, GameEvent]]:
    """Preserve the synchronous API while awaiting a live event source."""
    if isinstance(events, AsyncIterable):
        iterator = events.__aiter__()
        index = 0
        last_relevant_at = clock()
        last_event_timestamp_ms: int | None = None
        while True:
            next_event = asyncio.create_task(iterator.__anext__())
            while not next_event.done():
                now = clock()
                pending_timeout = (
                    timeout_offer() if timeout_offer is not None else None
                )
                if (
                    pending_timeout is not None
                    and now >= pending_timeout.deadline_at
                ):
                    yield index, _TimeoutOfferGraceExpired(
                        player_seat=pending_timeout.offer.player_seat,
                        decision_index=pending_timeout.offer.decision_index,
                        timestamp_ms=pending_timeout.deadline_timestamp_ms,
                    )
                    continue

                silent_seconds: float | None = None
                if (
                    stall_watchdog_seconds is not None
                    and obligation is not None
                    and obligation()
                ):
                    silent_seconds = now - last_relevant_at
                    if silent_seconds >= stall_watchdog_seconds:
                        next_event.cancel()
                        await asyncio.gather(next_event, return_exceptions=True)
                        yield index, _WatchdogStall(
                            frame_index=index,
                            last_event_timestamp_ms=last_event_timestamp_ms,
                            silent_seconds=silent_seconds,
                        )
                        return

                waits = [watchdog_poll_seconds]
                if pending_timeout is not None:
                    waits.append(pending_timeout.deadline_at - now)
                if (
                    silent_seconds is not None
                    and stall_watchdog_seconds is not None
                ):
                    waits.append(stall_watchdog_seconds - silent_seconds)
                poll_task = asyncio.create_task(sleep(min(waits)))
                done, _ = await asyncio.wait(
                    {next_event, poll_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if next_event in done:
                    poll_task.cancel()
                    await asyncio.gather(poll_task, return_exceptions=True)
                    break
                await poll_task
            try:
                event = await next_event
            except StopAsyncIteration:
                return
            if _is_game_relevant_event(event):
                last_relevant_at = clock()
                last_event_timestamp_ms = event.timestamp_ms
            yield index, event
            index += 1
        return

    next_index = 0
    for index, event in enumerate(events):
        next_index = index + 1
        while True:
            pending_timeout = (
                timeout_offer() if timeout_offer is not None else None
            )
            if not _timeout_offer_is_due_before_event(pending_timeout, event):
                break
            assert pending_timeout is not None
            yield index, _TimeoutOfferGraceExpired(
                player_seat=pending_timeout.offer.player_seat,
                decision_index=pending_timeout.offer.decision_index,
                timestamp_ms=pending_timeout.deadline_timestamp_ms,
            )
        yield index, event

    pending_timeout = timeout_offer() if timeout_offer is not None else None
    if pending_timeout is not None:
        yield next_index, _TimeoutOfferGraceExpired(
            player_seat=pending_timeout.offer.player_seat,
            decision_index=pending_timeout.offer.decision_index,
            timestamp_ms=pending_timeout.deadline_timestamp_ms,
        )


def _timeout_offer_is_due_before_event(
    pending_timeout: _PendingTimeoutOffer | None,
    event: GameEvent,
) -> bool:
    """Replay timeout grace from recorded frame timestamps when available."""
    return (
        pending_timeout is not None
        and pending_timeout.deadline_timestamp_ms is not None
        and event.timestamp_ms is not None
        and event.timestamp_ms >= pending_timeout.deadline_timestamp_ms
    )


def _timeout_offer_was_cancelled(
    event: GameEvent,
    offer: TimeoutOffer,
) -> bool:
    """Stand down once the offered player makes a decoded game decision.

    Client 2.2.8 has no distinct timeout-cancel metagame kind: it adds a
    timeout offer to ``permanentlyResignablePlayers`` and relies on game-end
    for normal cleanup.  A later decision resolved for that seat is therefore
    the stream-level proof that the opponent returned during our grace period.
    """
    return (
        isinstance(event, DecisionResolved)
        and event.seat == offer.player_seat
        and event.question_index > offer.decision_index
    )


def _timeout_offer_is_for_our_seat(
    offer: TimeoutOffer,
    snapshot: TrackerSnapshot,
) -> bool:
    """Identify the defensive self-timeout case without blocking an opponent.

    A self timeout is only plausible while the local player owns the active
    turn (or before the first turn is known).  The stalled capture's offer
    names seat 1 while the tracker has seat 0 active, so treating its field as
    a self-resign signal would contradict the live timeout prompt and leave
    the opponent's force-end control unhandled.
    """
    return (
        snapshot.our_seat is not None
        and offer.player_seat == snapshot.our_seat
        and snapshot.turn_owner in (None, snapshot.our_seat)
    )


def _is_game_relevant_event(event: GameEvent) -> bool:
    if isinstance(event, (Chat, SessionStart)):
        return False
    if isinstance(event, UnknownFrame):
        return event.direction == "in" and event.msg_type in {32, 33, 35, 37, 38}
    return True


def _has_outstanding_obligation(
    active: _ActiveGame | None,
    tracker: Tracker,
) -> bool:
    if active is None:
        return False
    snapshot = tracker.snapshot()
    if snapshot.ended:
        return False
    if snapshot.pending_decision is not None:
        return True
    return (
        snapshot.our_seat is not None
        and snapshot.turn_owner == snapshot.our_seat
    )


def _recorded_plan(
    decision: PendingDecision,
    answers: tuple[int, ...],
) -> DecisionPlan:
    snapshot = PendingDecisionSnapshot(
        question_index=decision.question_index,
        decision_type=decision.decision_type,
        question_id=decision.question_id,
        offered=decision.offered,
        minimum=decision.minimum,
        maximum=decision.maximum,
        association=decision.association,
    )
    actions: tuple[int, ...]
    question = decision.question_id
    if question == "GAME_MAY_REACT_WITH":
        if answers == (0, 1, 0, 0):
            actions = (int(dz.A_PASS),)
        elif len(answers) == 5 and answers[:1] == (1,):
            actions = (
                int(
                    dz.A_SELECT_BASE
                    + dz.def_id(offered_name(decision.offered[answers[1]]))
                ),
            )
        else:
            raise ValueError(f"unsupported Moat reaction answer {answers}")
    elif question == "GAME_ACTION_PHASE":
        if answers == (0, 0, 0):
            actions = (int(dz.A_PASS),)
        else:
            actions = (
                int(
                    dz.A_PLAY_BASE
                    + dz.def_id(offered_name(decision.offered[answers[2]]))
                ),
            )
    elif question == "GAME_BUY_PHASE":
        if answers == (0,):
            actions = (int(dz.A_PASS),)
        elif answers == (2, 0):
            actions = tuple(
                int(dz.A_PLAY_BASE + dz.def_id(offered_name(value)))
                for value in decision.offered
                if value.startswith("1:0:")
            )
        elif len(answers) == 4 and answers[:2] == (1, 1):
            treasures = tuple(
                value
                for value in decision.offered
                if value.startswith("1:0:")
            )
            actions = (
                int(
                    dz.A_PLAY_BASE
                    + dz.def_id(offered_name(treasures[answers[2]]))
                ),
            )
        else:
            actions = (
                int(
                    dz.A_BUY_BASE
                    + dz.def_id(offered_name(decision.offered[answers[1]]))
                ),
            )
    elif question == "THRONE_ROOM":
        if not answers:
            actions = (int(dz.A_PASS),)
        else:
            actions = (
                int(
                    dz.A_SELECT_BASE
                    + dz.def_id(offered_name(decision.offered[answers[1]]))
                ),
            )
    elif question == "SENTRY_TOPDECK":
        actions = (
            int(
                dz.A_OPTION_BASE
                + (1 if answers == (0, 1) else 0)
            ),
        )
    elif decision.decision_type == "ORDER_CARDS":
        remaining = list(range(len(decision.offered)))
        order: list[int] = []
        for original in answers:
            relative = remaining.index(original)
            order.append(int(dz.A_OPTION_BASE + relative))
            remaining.pop(relative)
        actions = tuple(order)
    elif decision.decision_type == "CHOOSE_MODE":
        actions = tuple(
            int(
                dz.A_OPTION_BASE
                + (1 - answer if question == "VASSAL" else answer)
            )
            for answer in answers
        )
    else:
        selected = tuple(
            int(
                dz.A_SELECT_BASE
                + dz.def_id(offered_name(decision.offered[answer]))
            )
            for answer in answers
        )
        need_pass = (
            decision.minimum == 0
            and len(answers) < min(decision.maximum, len(decision.offered))
        )
        actions = (*selected, int(dz.A_PASS)) if need_pass else selected
    possibilities = possible_answer_indices(actions, snapshot)
    if answers not in possibilities:
        raise ValueError(
            f"recorded answer {answers} is not producible for "
            f"{decision.question_index}: {possibilities}"
        )
    return DecisionPlan(
        engine_actions=actions,
        gesture_actions=actions,
        answer_hint=answers,
    )


def _paired_answers(
    events: tuple[GameEvent, ...],
) -> dict[int, tuple[int, ...]]:
    answers: dict[int, tuple[int, ...]] = {}
    for index, event in enumerate(events):
        if not isinstance(event, PendingDecision):
            continue
        for later in events[index + 1 :]:
            if isinstance(later, PendingDecision):
                break
            if (
                isinstance(later, DecisionResolved)
                and later.question_index == event.question_index
            ):
                answers[index] = later.answers
                break
    return answers


def _rig_observed_draws(
    game: Any,
    snapshot: TrackerSnapshot,
    events: Iterable[GameEvent],
) -> bool:
    if snapshot.our_seat is None:
        raise BridgeError("draw rigging has no local seat")
    seat = snapshot.our_seat
    observed: list[str] = []
    injected: Counter[str] = Counter()
    for event in events:
        if isinstance(event, Shuffle) and event.seat == seat:
            # set_deck_order cannot patch the post-shuffle discard order.  The
            # prefix before the shuffle is still safe to force.
            break
        if (
            isinstance(event, Topdeck)
            and event.seat == seat
            and event.to_zone == "deck"
        ):
            injected.update(event.cards)
            continue
        if (
            isinstance(event, (Draw, Reveal, Discard, ZoneTransfer))
            and event.seat == seat
            and event.from_zone == "deck"
        ):
            for name in event.cards:
                if injected[name]:
                    injected[name] -= 1
                else:
                    observed.append(name)
    if not observed:
        return False

    deck: list[str] = []
    for name, count in snapshot.seats[seat].deck:
        deck.extend([name] * count)
    if snapshot.seats[seat].deck_anonymous:
        raise BridgeError("cannot rig a local deck with anonymous cards")
    if int(game.deck_count(seat)) != len(deck):
        raise BridgeError(
            "shadow/tracker deck counts differ before rigged step: "
            f"{game.deck_count(seat)} vs {len(deck)}"
        )
    remaining = Counter(deck)
    for name in observed:
        if remaining[name] <= 0:
            raise BridgeError(
                f"observed draw {name!r} is absent from the shadow deck"
            )
        remaining[name] -= 1
    order = [*observed, *sorted(remaining.elements())]
    try:
        set_deck_order(game, seat, order)
    except (TypeError, ValueError) as error:
        raise BridgeError(
            "shadow/tracker deck compositions differ before rigged step; "
            f"observed prefix={tuple(observed)} tracker deck={tuple(deck)}"
        ) from error
    return True


def _has_local_shuffle(
    snapshot: TrackerSnapshot,
    events: Iterable[GameEvent],
) -> bool:
    if snapshot.our_seat is None:
        return False
    return any(
        isinstance(event, Shuffle) and event.seat == snapshot.our_seat
        for event in events
    )


def _rebuild_turn_after_shuffle(
    active: _ActiveGame,
    current: _TurnStep,
) -> None:
    """Recover an exact mid-turn frame when a draw crossed a shuffle.

    The binding can order the current deck but cannot order discard before the
    interpreter shuffles it.  Rebuilding the turn with its initial deck and
    discard pre-merged is equivalent once the first shuffle is observed, and
    lets Python prescribe the complete observed deck-source sequence.
    """
    start = active.turn_start_snapshot
    if start is None or start.our_seat is None:
        raise BridgeError("shuffle reconstruction has no turn-start snapshot")
    seat_index = start.our_seat
    seat = start.seats[seat_index]
    if seat.deck_anonymous or seat.discard_anonymous:
        raise BridgeError("shuffle reconstruction requires exact local zones")

    combined = Counter(dict(seat.deck))
    combined.update(dict(seat.discard))
    combined_cards = tuple(sorted(combined.items()))
    hidden = Counter(dict(seat.hand_deck))
    hidden.update(dict(seat.discard))
    rebuilt_seat = replace(
        seat,
        deck=combined_cards,
        deck_count=seat.deck_count + seat.discard_count,
        deck_anonymous=0,
        hand_deck=tuple(sorted(hidden.items())),
        hand_deck_count=seat.hand_deck_count + seat.discard_count,
        discard=(),
        discard_count=0,
        discard_anonymous=0,
    )
    seats = list(start.seats)
    seats[seat_index] = rebuilt_seat
    rebuilt_snapshot = replace(start, seats=tuple(seats))
    game = game_from_snapshot(rebuilt_snapshot)

    all_steps = (*active.turn_history, current)
    observed = _deck_source_sequence(
        seat_index,
        (
            event
            for step in all_steps
            for event in step.events
        ),
    )
    remaining = Counter(combined)
    for name in observed:
        if remaining[name] <= 0:
            raise BridgeError(
                f"shuffle reconstruction observed unavailable {name!r}"
            )
        remaining[name] -= 1
    order = [*observed, *sorted(remaining.elements())]
    set_deck_order(game, seat_index, order)

    for step in all_steps:
        _step_actions(
            game,
            step.actions,
            frame_index=current.snapshot.pending_decision.question_index
            if current.snapshot.pending_decision is not None
            else -1,
            intended=None,
        )
    game.validate()
    active.shadow = game


def _deck_source_sequence(
    seat: int,
    events: Iterable[GameEvent],
) -> list[str]:
    """Return actual cards originating in the turn-start deck/discard pool."""
    observed: list[str] = []
    injected: Counter[str] = Counter()
    discarded_from_deck: Counter[str] = Counter()
    for event in events:
        if (
            isinstance(event, Topdeck)
            and event.seat == seat
            and event.to_zone == "deck"
        ):
            injected.update(event.cards)
            continue
        if (
            isinstance(event, Play)
            and event.seat == seat
            and event.from_zone == "discard"
        ):
            for name in event.cards:
                if discarded_from_deck[name]:
                    discarded_from_deck[name] -= 1
            continue
        if (
            isinstance(event, ZoneTransfer)
            and event.seat == seat
            and event.from_zone == "discard"
            and event.to_zone == "deck"
        ):
            injected.update(discarded_from_deck)
            discarded_from_deck.clear()
            continue
        if (
            isinstance(event, (Draw, Reveal, Discard, ZoneTransfer))
            and event.seat == seat
            and event.from_zone == "deck"
        ):
            for name in event.cards:
                if injected[name]:
                    injected[name] -= 1
                else:
                    observed.append(name)
                if isinstance(event, Discard):
                    discarded_from_deck[name] += 1
    return observed


def _step_actions(
    game: Any,
    actions: tuple[int, ...],
    *,
    frame_index: int,
    intended: IntendedAction | None,
) -> None:
    for action_index, action in enumerate(actions):
        legal = game.legal_mask()
        if not 0 <= action < len(legal) or not bool(legal[action]):
            raise DivergenceError(
                f"recorded/chosen engine action {action} at plan offset "
                f"{action_index} became illegal; legal actions are "
                f"{tuple(index for index, value in enumerate(legal) if value)}",
                frame_index=frame_index,
                question_index=(
                    intended.decision.question_index
                    if intended is not None
                    else None
                ),
                intended=(
                    intended.gesture.answer_indices
                    if intended is not None
                    else ()
                ),
            )
        game.step(action)
        game.validate()


async def _act_with_hint(
    actuator: Actuator,
    action: int,
    decision: PendingDecisionSnapshot,
    offered: tuple[str, ...],
    *,
    prior_actions: tuple[int, ...],
    answer_hint: tuple[int, ...] | None,
) -> ClientGesture:
    if answer_hint is None:
        return await actuator.act(
            action,
            decision,
            offered,
            prior_actions=prior_actions,
        )
    possibilities = possible_answer_indices(
        (*prior_actions, action),
        decision,
        offered,
    )
    if answer_hint not in possibilities:
        raise DivergenceError(
            "recorded answer is outside the mapper's possible encodings",
            question_index=decision.question_index,
            intended=answer_hint,
        )
    if isinstance(actuator, MockActuator) and actuator.replay:
        mapping_gesture = await actuator.act(
            action,
            decision,
            offered,
            prior_actions=prior_actions,
        )
        if mapping_gesture.answer_indices == answer_hint:
            return mapping_gesture
        replacement = ClientGesture(
            question_index=mapping_gesture.question_index,
            action=mapping_gesture.action,
            prior_actions=mapping_gesture.prior_actions,
            answer_indices=answer_hint,
            selected_indices=_recorded_selected_indices(decision, answer_hint),
            click_button=mapping_gesture.click_button,
            labels=tuple(
                offered_name(offered[index])
                for index in _recorded_selected_indices(decision, answer_hint)
            ),
        )
        actuator.gestures[-1] = replacement
        return replacement
    return await actuator.act(
        action,
        decision,
        offered,
        prior_actions=prior_actions,
    )


def _recorded_selected_indices(
    decision: PendingDecisionSnapshot,
    answers: tuple[int, ...],
) -> tuple[int, ...]:
    if decision.question_id == "GAME_MAY_REACT_WITH":
        return (answers[1],) if len(answers) == 5 and answers[0] == 1 else ()
    if decision.question_id == "GAME_ACTION_PHASE":
        return (answers[2],) if len(answers) == 4 else ()
    if decision.question_id == "GAME_BUY_PHASE":
        if answers == (2, 0):
            return tuple(
                index
                for index, value in enumerate(decision.offered)
                if value.endswith("AUTOPLAY_TREASURES")
            )
        if len(answers) == 4 and answers[:2] == (1, 1):
            treasures = [
                index
                for index, value in enumerate(decision.offered)
                if value.startswith("1:0:")
            ]
            return (treasures[answers[2]],)
        return (answers[1],) if len(answers) == 2 else ()
    if decision.question_id == "THRONE_ROOM":
        return (answers[1],) if answers else ()
    return answers


def _decision_signature(game: Any) -> tuple[int, int, int]:
    decision = game.current_decision()
    return (
        int(decision.get("player", -1)),
        int(decision.get("kind", -1)),
        int(decision.get("source", -1)),
    )


def _offered_treasures(
    decision: PendingDecisionSnapshot,
) -> tuple[str, ...]:
    return tuple(
        value
        for value in decision.offered
        if value.startswith("1:0:")
    )


def _question_context(
    decision: PendingDecisionSnapshot | None,
) -> dict[str, object] | None:
    if decision is None:
        return None
    return {
        "question_index": decision.question_index,
        "decision_type": decision.decision_type,
        "question_id": decision.question_id,
        "offered": decision.offered,
        "minimum": decision.minimum,
        "maximum": decision.maximum,
        "association": decision.association,
    }


def _buy_snapshot_context(snapshot: TrackerSnapshot) -> dict[str, object]:
    return {
        "game_id": snapshot.game_id,
        "turn_number": snapshot.turn_number,
        "turn_owner": snapshot.turn_owner,
        "our_seat": snapshot.our_seat,
        "phase": snapshot.phase,
    }


def _recoverable_partial_autoplay(
    intended: IntendedAction,
    events: Iterable[GameEvent],
    submitted_snapshot: TrackerSnapshot,
    current_snapshot: TrackerSnapshot,
) -> bool:
    """Allow a verified autoplay answer to be retried from its new prompt."""
    current = current_snapshot.pending_decision
    if (
        intended.decision.question_id != "GAME_BUY_PHASE"
        or intended.gesture.answer_indices != (2, 0)
        or current is None
        or current.question_id != "GAME_BUY_PHASE"
        or current_snapshot.game_id != submitted_snapshot.game_id
        or current_snapshot.turn_number != submitted_snapshot.turn_number
        or current_snapshot.turn_owner != submitted_snapshot.turn_owner
        or not _offered_treasures(current)
    ):
        return False
    resolution = next(
        (
            event
            for event in events
            if isinstance(event, DecisionResolved)
            and event.question_index == intended.decision.question_index
        ),
        None,
    )
    if (
        resolution is None
        or resolution.answers not in intended.acceptable_answers
    ):
        return False
    expected = Counter(
        offered_name(value)
        for value in _offered_treasures(intended.decision)
    )
    observed = Counter(
        card
        for event in events
        if isinstance(event, Play)
        and event.seat == current_snapshot.our_seat
        for card in event.cards
    )
    remaining = Counter(
        offered_name(value) for value in _offered_treasures(current)
    )
    return (
        bool(remaining)
        and not (observed - expected)
        and expected - observed == remaining
    )


def _collapse_offered_treasures(
    game: Any,
    decision: PendingDecisionSnapshot,
) -> tuple[int, ...]:
    """Mirror SelfPlayRunner's ascending-definition treasure collapse."""
    counts = Counter(
        int(dz.def_id(offered_name(value)))
        for value in _offered_treasures(decision)
    )
    actions: list[int] = []
    for definition, count in sorted(counts.items()):
        action = int(dz.A_PLAY_BASE + definition)
        for _ in range(count):
            legal = game.legal_mask()
            if not 0 <= action < len(legal) or not bool(legal[action]):
                raise BridgeError(
                    "offered hand treasure is not legal in the shadow game"
                )
            game.step(action)
            actions.append(action)
    return tuple(actions)


def _require_buy_or_pass(game: Any, action: int) -> None:
    legal = game.legal_mask()
    if not 0 <= action < len(legal) or not bool(legal[action]):
        raise BridgeError(f"collapsed buy policy returned illegal action {action}")
    is_buy = int(dz.A_BUY_BASE) <= action < int(
        dz.A_BUY_BASE + dz.ACTION_DEF_COUNT
    )
    if action != int(dz.A_PASS) and not is_buy:
        raise BridgeError(
            f"collapsed buy policy returned non-buy action {action}"
        )


def _advance_forced_to_client_question(
    game: Any,
    decision: PendingDecisionSnapshot,
    *,
    frame_index: int,
) -> tuple[int, ...]:
    """Apply engine-only forced choices the web client auto-resolved.

    dominion.games omits a question when an effect has only one meaningful
    answer (for example the second forced Remodel trash in a Throne Room
    chain).  The native interpreter deliberately exposes that choice.  Decision
    kind is sufficient to bridge the gap without guessing a non-forced move.
    """
    expected = {
        "GAME_ACTION_PHASE": 1,
        "GAME_BUY_PHASE": 2,
        "TRASH": 4,
        "DISCARD": 4,
        "GAIN": 5,
        "TOPDECK": 4,
        "COMPLEX_AND": 4,
        "CHOOSE_MODE": 6,
        "ORDER_CARDS": 7,
    }.get(decision.decision_type)
    if decision.question_id in {"SENTRY_TRASH", "SENTRY_DISCARD"}:
        expected = 6
    if decision.question_id == "SENTRY_TOPDECK":
        if int(game.current_decision().get("kind", -1)) == 6:
            return ()
        expected = 7
    if expected is None:
        return ()

    forced: list[int] = []
    for _ in range(16):
        current = game.current_decision()
        if int(current.get("kind", -1)) == expected:
            return tuple(forced)
        legal = tuple(
            index
            for index, value in enumerate(game.legal_mask())
            if value
        )
        if len(legal) != 1:
            raise DivergenceError(
                "native/client decision kinds differ without a forced bridge: "
                f"native={current.get('kind')} client={expected} legal={legal}",
                frame_index=frame_index,
                question_index=decision.question_index,
            )
        game.step(legal[0])
        game.validate()
        forced.append(legal[0])
    raise DivergenceError(
        "too many consecutive client-auto-resolved native choices",
        frame_index=frame_index,
        question_index=decision.question_index,
    )


def _occurrence_tags(values: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    counts: Counter[str] = Counter()
    result: list[tuple[str, int]] = []
    for value in values:
        result.append((value, counts[value]))
        counts[value] += 1
    return tuple(result)


def _unknown_kingdom(kingdom: tuple[str, ...]) -> tuple[str, ...]:
    unknown: list[str] = []
    for name in kingdom:
        try:
            dz.def_id(name)
        except (TypeError, ValueError):
            unknown.append(name)
    return tuple(unknown)


async def _maybe_await(
    callback: Callable[..., object | Awaitable[object]] | None,
    *args: object,
) -> object | None:
    if callback is None:
        return None
    result = callback(*args)
    if inspect.isawaitable(result):
        return await result
    return result


async def _capture_screenshot(
    hook: Callable[
        [int, TrackerSnapshot], str | Path | None | Awaitable[str | Path | None]
    ]
    | None,
    frame_index: int,
    snapshot: TrackerSnapshot,
) -> str | None:
    value = await _maybe_await(hook, frame_index, snapshot)
    return None if value is None else str(value)


async def _capture_artifact(
    hook: Callable[
        [int, TrackerSnapshot], str | Path | None | Awaitable[str | Path | None]
    ]
    | None,
    frame_index: int,
    snapshot: TrackerSnapshot,
    *,
    label: str,
) -> str | None:
    try:
        return await _capture_screenshot(hook, frame_index, snapshot)
    except Exception as error:
        LOGGER.error("could not capture %s: %s", label, error)
        return None


def _pending_question_summary(
    snapshot: TrackerSnapshot,
) -> dict[str, object] | None:
    pending = snapshot.pending_decision
    if pending is None:
        return None
    return {
        "question_index": pending.question_index,
        "decision_type": pending.decision_type,
        "question_id": pending.question_id,
        "offered": pending.offered,
        "minimum": pending.minimum,
        "maximum": pending.maximum,
        "association": pending.association,
    }


def _snapshot_summary(snapshot: TrackerSnapshot) -> dict[str, object]:
    return {
        "game_id": snapshot.game_id,
        "our_seat": snapshot.our_seat,
        "turn_number": snapshot.turn_number,
        "turn_owner": snapshot.turn_owner,
        "phase": snapshot.phase,
        "pending_question": (
            snapshot.pending_decision.question_index
            if snapshot.pending_decision is not None
            else None
        ),
        "supply_piles": len(snapshot.supply),
        "seat_counts": tuple(
            {
                "seat": seat.seat,
                "hand": seat.hand_count,
                "deck": seat.deck_count,
                "discard": seat.discard_count,
                "in_play": seat.in_play_count,
            }
            for seat in snapshot.seats
        ),
    }


def _result_fields(active: _ActiveGame) -> dict[str, object]:
    opponent = next(
        (
            player
            for seat, player in enumerate(active.players)
            if seat != active.our_seat
        ),
        None,
    )
    result = active.game_result
    outcome = "unknown"
    if (
        result is not None
        and result.decoded
        and active.our_seat is not None
        and active.our_seat < len(result.scores)
    ):
        if result.tie:
            outcome = "tie"
        elif result.winner_seat == active.our_seat:
            outcome = "win"
        else:
            outcome = "loss"
    return {
        "our_seat": active.our_seat,
        "opponent": opponent,
        "outcome": outcome,
        "scores": () if result is None else result.scores,
        "placings": () if result is None else result.placings,
        "winner_seat": None if result is None else result.winner_seat,
        "tie": False if result is None else result.tie,
    }


def _finish_archive(active: _ActiveGame, result: GameRunResult) -> None:
    if active.archive is None:
        return
    active.archive.finish(
        ResultSummary(
            game_id=result.game_id,
            completed=result.completed,
            divergence_aborted=result.divergence_aborted,
            decisions=result.decisions,
            reason=result.reason,
            stall_aborted=result.stall_aborted,
            our_seat=result.our_seat,
            opponent=result.opponent,
            outcome=result.outcome,
            scores=result.scores,
            placings=result.placings,
            winner_seat=result.winner_seat,
            tie=result.tie,
        ),
        divergence_report=(
            None
            if result.divergence_report is None
            else {
                "frame_index": result.divergence_report.frame_index,
                "error_type": result.divergence_report.error_type,
                "message": result.divergence_report.message,
                "tracker_summary": result.divergence_report.tracker_summary,
                "screenshot_path": result.divergence_report.screenshot_path,
            }
        ),
        stall_report=(
            None
            if result.stall_report is None
            else {
                "frame_index": result.stall_report.frame_index,
                "game_id": result.stall_report.game_id,
                "last_event_timestamp_ms": (
                    result.stall_report.last_event_timestamp_ms
                ),
                "silent_seconds": result.stall_report.silent_seconds,
                "tracker_summary": result.stall_report.tracker_summary,
                "pending_question": result.stall_report.pending_question,
                "screenshot_path": result.stall_report.screenshot_path,
                "dom_snapshot_path": result.stall_report.dom_snapshot_path,
            }
        ),
    )
