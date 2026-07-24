"""Supervised in-game loop with a validated native shadow game."""

from __future__ import annotations

import asyncio
import inspect
import random
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
    GameEnd,
    GameEvent,
    GameStart,
    PendingDecision,
    Play,
    Reconnect,
    Reveal,
    Shuffle,
    Topdeck,
    ZoneTransfer,
)
from ..shadow.bridge import BridgeError, game_from_snapshot, set_deck_order
from ..shadow.tracker import (
    PendingDecisionSnapshot,
    Tracker,
    TrackerError,
    TrackerSnapshot,
)


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
class GameRunResult:
    """Offline/live outcome for one observed game."""

    game_id: int | None
    completed: bool
    divergence_aborted: bool
    kingdom_rejected: bool
    decisions: int
    validations: int
    rigged_steps: int
    reason: str
    divergence_report: DivergenceReport | None = None


@dataclass
class _ActiveGame:
    game_id: int | None
    kingdom_rejected: bool = False
    decisions: int = 0
    validations: int = 0
    rigged_steps: int = 0
    shadow: Any | None = None
    shadow_turn: tuple[int | None, int | None] | None = None
    turn_start_snapshot: TrackerSnapshot | None = None
    turn_history: list[_TurnStep] = field(default_factory=list)
    pending: IntendedAction | None = None
    pending_snapshot: TrackerSnapshot | None = None
    pending_plan: DecisionPlan | None = None
    pending_events: list[GameEvent] = field(default_factory=list)
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
        del frame_index
        if snapshot.our_seat is None:
            raise BridgeError("cannot choose without a local seat")
        seat = snapshot.our_seat

        if self._deferred_buy is not None:
            deferred = self._deferred_buy
            if not (
                decision.question_id == "GAME_BUY_PHASE"
                and snapshot.game_id == deferred.game_id
                and snapshot.turn_number == deferred.turn_number
                and seat == deferred.seat
                and not _offered_treasures(decision)
            ):
                raise BridgeError(
                    "client did not confirm autoplay with a treasure-free "
                    "follow-up buy question"
                )
            self._deferred_buy = None
            _require_buy_or_pass(game, deferred.action)
            return DecisionPlan(
                engine_actions=(deferred.action,),
                gesture_actions=(deferred.action,),
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
            planning = game.clone()
            treasure_actions = _collapse_offered_treasures(
                planning,
                decision,
            )
            action = await self._choose(planning, seat)
            _require_buy_or_pass(planning, action)
            if treasure_actions:
                if not any(
                    value.endswith("AUTOPLAY_TREASURES")
                    for value in decision.offered
                ):
                    raise BridgeError(
                        "buy question has hand treasures but no autoplay control"
                    )
                if snapshot.turn_number is None:
                    raise BridgeError("cannot defer a buy without a turn number")
                self._deferred_buy = _DeferredBuy(
                    game_id=snapshot.game_id,
                    turn_number=snapshot.turn_number,
                    seat=seat,
                    action=action,
                )
                return DecisionPlan(
                    engine_actions=treasure_actions,
                    gesture_actions=treasure_actions,
                )
            return DecisionPlan(
                engine_actions=(action,),
                gesture_actions=(action,),
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
    chat_hook: Callable[[str], object | Awaitable[object]] | None = None,
    resign_hook: Callable[[], object | Awaitable[object]] | None = None,
    kingdom_rejection_text: str = (
        "Sorry — this bot only supports the configured base-set kingdom. "
        "I will resign this game."
    ),
    archive_factory: Callable[[int | None], GameArchive] | None = None,
) -> tuple[GameRunResult, ...]:
    """Consume normalized events and play until every observed game ends.

    Any tracker/bridge/verifier failure stops the entire actuator immediately.
    Continuing after a divergence would risk making a move in an unknown state.
    """
    if think_time_min_seconds < 0:
        raise ValueError("minimum think time cannot be negative")
    if think_time_max_seconds < think_time_min_seconds:
        raise ValueError("maximum think time must be at least the minimum")
    rng = random_source or random.Random()
    tracker = Tracker()
    active: _ActiveGame | None = None
    results: list[GameRunResult] = []

    async def settle(frame_index: int) -> None:
        nonlocal active
        assert active is not None
        if active.pending is None:
            return
        verify_action_events(
            active.pending,
            active.pending_events,
            frame_index=frame_index,
        )
        assert active.pending_snapshot is not None
        assert active.pending_plan is not None
        current_snapshot = tracker.snapshot()
        crossed_turn_boundary = (
            current_snapshot.turn_number
            != active.pending_snapshot.turn_number
            or current_snapshot.turn_owner
            != active.pending_snapshot.turn_owner
        )
        if active.pending_plan.engine_actions:
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

    async for frame_index, event in _indexed_events(events):
        try:
            tracker.consume(event)
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

            if isinstance(event, GameStart):
                if active is None or active.game_id != event.game_id:
                    active = _ActiveGame(
                        game_id=event.game_id,
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
                        await _maybe_await(chat_hook, kingdom_rejection_text)
                        await _maybe_await(resign_hook)
                continue

            if active is None:
                continue

            if isinstance(event, Reconnect):
                active.shadow = None
                active.shadow_turn = None
                active.turn_start_snapshot = None
                active.turn_history.clear()
                active.pending = None
                active.pending_snapshot = None
                active.pending_plan = None
                active.pending_events.clear()
                continue

            if isinstance(event, GameEnd):
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
                    reason=event.reason,
                )
                _finish_archive(active, result)
                results.append(result)
                active = None
                continue

            if (
                not isinstance(event, PendingDecision)
                or active.kingdom_rejected
            ):
                continue
            snapshot = tracker.snapshot()
            if (
                snapshot.pending_decision is None
                or snapshot.our_seat is None
                or snapshot.turn_owner is None
                or snapshot.turn_number is None
            ):
                continue

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
            if plan.answer_hint is not None and (
                gesture.answer_indices != plan.answer_hint
            ):
                raise DivergenceError(
                    "replay actuator did not preserve the recorded answer shape",
                    frame_index=frame_index,
                    question_index=event.question_index,
                    intended=plan.answer_hint,
                    observed=gesture.answer_indices,
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
                reason=str(error),
                divergence_report=report,
            )
            _finish_archive(active, result)
            results.append(result)
            return tuple(results)

    return tuple(results)


async def _indexed_events(
    events: Iterable[GameEvent] | AsyncIterable[GameEvent],
) -> AsyncIterator[tuple[int, GameEvent]]:
    """Preserve the synchronous API while awaiting a live event source."""
    if isinstance(events, AsyncIterable):
        index = 0
        async for event in events:
            yield index, event
            index += 1
        return
    for index, event in enumerate(events):
        yield index, event


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
    if question == "GAME_ACTION_PHASE":
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
    )
