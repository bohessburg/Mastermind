from __future__ import annotations

import asyncio
import re
from collections import Counter
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import dominion_v2_py as dz
import pytest

from src.v2.arena.actuate.clicks import (
    ActionMappingError,
    ActuationError,
    DOMCardStack,
    DOMGameButton,
    MockActuator,
    PlaywrightActuator,
    card_stack_region,
    dom_click_target,
    game_button_role,
    gesture_click_targets,
    map_engine_actions,
    offered_name,
    possible_answer_indices,
    resolve_card_stack_index,
    resolve_submit_button_index,
    selected_card_counts,
)
from src.v2.arena.actuate.protocol import ProtocolActuator
from src.v2.arena.fsm import game as game_module
from src.v2.arena.fsm.game import (
    BotDecisionProvider,
    DecisionPlan,
    RecordedDecisionProvider,
    _recorded_plan,
    _step_actions,
    run_game_loop,
)
from src.v2.arena.protocol.events import (
    DecisionResolved,
    Discard,
    Draw,
    GameEnd,
    GameStart,
    PendingDecision,
    Play,
    ResourceUpdate,
    Shuffle,
    UnknownFrame,
)
from src.v2.arena.protocol.messages import ANSWER_QUESTION, encode_answer
from src.v2.arena.protocol.recording import parse_recording
from src.v2.arena.shadow.bridge import BridgeError, game_from_snapshot
from src.v2.arena.shadow.tracker import (
    PendingDecisionSnapshot,
    Tracker,
    TrackerSnapshot,
)


LIVE_ARCHIVE = Path(
    "exports/arena/20260724T202307.273720Z/frames.jsonl"
)
BUY_DOM = Path(
    "arena-recordings/20260724T142103.096991Z/dom-1508015.html"
)
SELECTED_DISCARD_DOM = Path(
    "arena-recordings/20260724T142103.096991Z/dom-4326014.html"
)
LIVE_GAME_2_ARCHIVE = Path(
    "exports/arena/20260724T204829.953926Z/frames.jsonl"
)
LIVE_GAME_3_ARCHIVE = Path(
    "exports/arena/20260724T211613.397353Z/frames.jsonl"
)
REFERENCE_RECORDING = Path(
    "arena-recordings/20260724T142103.096991Z/frames.jsonl"
)
AUTOPLAY_ABORT_ARCHIVE = Path(
    "exports/arena/20260725T053730.304380Z/"
    "20260725T055943.084274Z-game-181375685/frames.jsonl"
)


def _live_turn_one() -> tuple[
    tuple[object, ...],
    int,
    TrackerSnapshot,
    PendingDecisionSnapshot,
]:
    if not LIVE_ARCHIVE.is_file():
        pytest.skip(f"missing live arena fixture: {LIVE_ARCHIVE}")
    events = parse_recording(LIVE_ARCHIVE).events
    tracker = Tracker()
    for frame_index, event in enumerate(events):
        tracker.consume(event)
        if (
            isinstance(event, PendingDecision)
            and event.question_index == 3
        ):
            snapshot = tracker.snapshot()
            assert snapshot.pending_decision is not None
            return events, frame_index, snapshot, snapshot.pending_decision
    raise AssertionError("live arena fixture has no turn-one buy question")


def _without_treasures(
    decision: PendingDecisionSnapshot,
) -> PendingDecisionSnapshot:
    return replace(
        decision,
        offered=tuple(
            value
            for value in decision.offered
            if value.startswith("0:")
        ),
        maximum=1,
    )


def test_live_buy_provider_autoplays_then_uses_collapsed_buy() -> None:
    _, frame_index, snapshot, decision = _live_turn_one()
    silver = int(dz.A_BUY_BASE + dz.def_id("Silver"))
    copper_play = int(dz.A_PLAY_BASE + dz.def_id("Copper"))
    policy_calls = 0

    def collapsed_policy(game: Any, seat: int) -> int:
        nonlocal policy_calls
        del seat
        policy_calls += 1
        assert not bool(game.legal_mask()[copper_play])
        assert bool(game.legal_mask()[silver])
        return silver

    provider = BotDecisionProvider(collapsed_policy)
    game = game_from_snapshot(snapshot)
    first = asyncio.run(
        provider.plan(
            frame_index=frame_index,
            game=game,
            snapshot=snapshot,
            decision=decision,
        )
    )

    assert first.engine_actions == (copper_play,) * 3
    assert first.gesture_actions == first.engine_actions
    assert first.gesture_actions != (copper_play,)
    assert map_engine_actions(
        first.gesture_actions,
        decision,
    ).answers == (2, 0)

    for action in first.engine_actions:
        game.step(action)
    follow_up = _without_treasures(decision)
    second = asyncio.run(
        provider.plan(
            frame_index=frame_index + 1,
            game=game,
            snapshot=replace(snapshot, pending_decision=follow_up),
            decision=follow_up,
        )
    )

    assert second.engine_actions == (silver,)
    assert second.gesture_actions == (silver,)
    assert map_engine_actions(
        second.gesture_actions,
        follow_up,
    ).answers == (0, 2)
    assert policy_calls == 1


def test_buy_provider_handles_pass_and_already_collapsed_state() -> None:
    _, frame_index, snapshot, decision = _live_turn_one()
    copper_play = int(dz.A_PLAY_BASE + dz.def_id("Copper"))
    follow_up = _without_treasures(decision)

    pass_provider = BotDecisionProvider(
        lambda game, seat: int(dz.A_PASS)
    )
    pass_game = game_from_snapshot(snapshot)
    autoplay = asyncio.run(
        pass_provider.plan(
            frame_index=frame_index,
            game=pass_game,
            snapshot=snapshot,
            decision=decision,
        )
    )
    assert autoplay.gesture_actions == (copper_play,) * 3
    for action in autoplay.engine_actions:
        pass_game.step(action)
    decline = asyncio.run(
        pass_provider.plan(
            frame_index=frame_index + 1,
            game=pass_game,
            snapshot=replace(snapshot, pending_decision=follow_up),
            decision=follow_up,
        )
    )
    assert decline.gesture_actions == (int(dz.A_PASS),)
    assert map_engine_actions(
        decline.gesture_actions,
        follow_up,
    ).answers == (0,)

    silver = int(dz.A_BUY_BASE + dz.def_id("Silver"))
    direct_provider = BotDecisionProvider(lambda game, seat: silver)
    direct = asyncio.run(
        direct_provider.plan(
            frame_index=frame_index + 1,
            game=pass_game,
            snapshot=replace(snapshot, pending_decision=follow_up),
            decision=follow_up,
        )
    )
    assert direct.engine_actions == (silver,)
    assert direct.gesture_actions == (silver,)


def test_incident_archive_replays_past_partial_autoplay_follow_up(
    caplog: pytest.LogCaptureFixture,
) -> None:
    if not AUTOPLAY_ABORT_ARCHIVE.is_file():
        pytest.skip(
            f"missing autoplay incident fixture: {AUTOPLAY_ABORT_ARCHIVE}"
        )
    events = parse_recording(AUTOPLAY_ABORT_ARCHIVE).events
    tracker = Tracker()
    prior_snapshot: TrackerSnapshot | None = None
    in_prior_game = False
    for event in events:
        if (
            isinstance(event, GameStart)
            and event.game_id == 181375622
        ):
            in_prior_game = True
        if not in_prior_game:
            continue
        tracker.consume(event)
        if (
            isinstance(event, PendingDecision)
            and event.question_index == 12
        ):
            prior_snapshot = tracker.snapshot()
            break
    assert prior_snapshot is not None
    assert prior_snapshot.pending_decision is not None

    silver = int(dz.A_BUY_BASE + dz.def_id("Silver"))
    copper_play = int(dz.A_PLAY_BASE + dz.def_id("Copper"))
    policy_calls = 0

    def collapsed_policy(game: Any, seat: int) -> int:
        nonlocal policy_calls
        del seat
        policy_calls += 1
        assert not bool(game.legal_mask()[copper_play])
        return silver

    provider = BotDecisionProvider(collapsed_policy)
    stale_plan = asyncio.run(
        provider.plan(
            frame_index=5760,
            game=game_from_snapshot(prior_snapshot),
            snapshot=prior_snapshot,
            decision=prior_snapshot.pending_decision,
        )
    )
    assert map_engine_actions(
        stale_plan.gesture_actions,
        prior_snapshot.pending_decision,
    ).answers == (2, 0)

    incident_start = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, GameStart)
        and event.game_id == 181375685
    )
    replay = list(events[incident_start:])
    question = replay[-1]
    assert isinstance(question, PendingDecision)
    assert question.question_index == 5
    supply = tuple(
        value for value in question.offered if value.startswith("0:")
    )
    two_coppers = (
        *supply,
        "1:0:Copper",
        "1:0:Copper",
        "2:AUTOPLAY_TREASURES",
    )
    replay.extend(
        (
            DecisionResolved(
                question_index=5,
                answers=(2, 0),
                seat=1,
                auto_played=False,
            ),
            Play(
                seat=1,
                cards=("Copper",),
                count=1,
                from_zone="hand",
                to_zone="in-play",
                from_zone_index=24,
                to_zone_index=25,
            ),
            ResourceUpdate(
                seat=1,
                resource="coins",
                value=1,
                counter_index=12,
            ),
            replace(
                question,
                question_index=6,
                offered=two_coppers,
            ),
            DecisionResolved(
                question_index=6,
                answers=(2, 0),
                seat=1,
                auto_played=False,
            ),
            Play(
                seat=1,
                cards=("Copper",),
                count=1,
                from_zone="hand",
                to_zone="in-play",
                from_zone_index=24,
                to_zone_index=25,
            ),
            ResourceUpdate(
                seat=1,
                resource="coins",
                value=2,
                counter_index=12,
            ),
            Play(
                seat=1,
                cards=("Copper",),
                count=1,
                from_zone="hand",
                to_zone="in-play",
                from_zone_index=24,
                to_zone_index=25,
            ),
            ResourceUpdate(
                seat=1,
                resource="coins",
                value=3,
                counter_index=12,
            ),
            replace(
                question,
                question_index=7,
                offered=supply,
                maximum=1,
            ),
            GameEnd(game_id=181375685, reason="test-complete"),
        )
    )

    caplog.set_level("WARNING")
    actuator = MockActuator(replay=True)
    results = asyncio.run(
        run_game_loop(
            tuple(replay),
            actuator=actuator,
            decision_provider=provider,
        )
    )

    assert len(results) == 1
    assert results[0].completed
    assert not results[0].divergence_aborted
    assert results[0].decisions == 4
    assert [
        (gesture.question_index, gesture.answer_indices)
        for gesture in actuator.gestures
    ] == [
        (2, (0,)),
        (5, (2, 0)),
        (6, (2, 0)),
        (7, (0, 2)),
    ]
    assert policy_calls == 3
    assert "AUTOPLAY BUY PARTIAL OUTCOME" in caplog.text
    assert "follow-up buy question still offers hand treasures" in caplog.text
    assert "'resolved_answers': (2, 0)" in caplog.text


class _RepeatedModeDecisionGame:
    """A clone whose second mode question keeps the same engine signature."""

    def __init__(self, steps: int = 0) -> None:
        self.steps = steps

    def clone(self) -> _RepeatedModeDecisionGame:
        return _RepeatedModeDecisionGame(self.steps)

    def current_decision(self) -> dict[str, int]:
        return {
            "player": 1,
            "kind": 7,
            "source": 11 if self.steps < 2 else 12,
        }

    def legal_mask(self) -> list[bool]:
        legal = [False] * int(dz.A_CALL_BASE)
        legal[int(dz.A_OPTION_BASE + 1)] = True
        return legal

    def step(self, action: int) -> None:
        assert action == int(dz.A_OPTION_BASE + 1)
        self.steps += 1


class _BatchedDecisionGame:
    """Small engine stand-in with a stable decision until a chosen step."""

    def __init__(
        self,
        legal_actions: tuple[int, ...],
        changes_after: int,
        steps: int = 0,
    ) -> None:
        self.legal_actions = legal_actions
        self.changes_after = changes_after
        self.steps = steps

    def clone(self) -> _BatchedDecisionGame:
        return _BatchedDecisionGame(
            self.legal_actions,
            self.changes_after,
            self.steps,
        )

    def current_decision(self) -> dict[str, int]:
        return {
            "player": 1,
            "kind": 7,
            "source": 11 if self.steps < self.changes_after else 12,
        }

    def legal_mask(self) -> list[bool]:
        legal = [False] * (max(self.legal_actions) + 1)
        for action in self.legal_actions:
            legal[action] = True
        return legal

    def step(self, action: int) -> None:
        assert action in self.legal_actions
        self.steps += 1

    def decision_context(self) -> dict[str, object]:
        """A Sentry looking at Copper then Estate."""
        return {
            "source_def": int(dz.def_id("Sentry")),
            "subject_defs": [int(dz.def_id("Copper")), int(dz.def_id("Estate"))],
            "subject_index": min(self.steps, 1),
        }


class _ProviderSnapshot:
    our_seat = 1


def _library_mode_decision() -> PendingDecisionSnapshot:
    return PendingDecisionSnapshot(
        question_index=64,
        decision_type="CHOOSE_MODE",
        question_id="LIBRARY",
        offered=("card-mode-74", "card-mode-75"),
        minimum=1,
        maximum=1,
        association="Library",
    )


def test_buy_follow_up_with_treasures_retries_twice_then_plays_explicitly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, frame_index, snapshot, decision = _live_turn_one()
    copper_play = int(dz.A_PLAY_BASE + dz.def_id("Copper"))
    silver = int(dz.A_BUY_BASE + dz.def_id("Silver"))
    policy_calls = 0

    def collapsed_policy(game: Any, seat: int) -> int:
        nonlocal policy_calls
        del seat
        policy_calls += 1
        assert not bool(game.legal_mask()[copper_play])
        return silver

    caplog.set_level("WARNING")
    provider = BotDecisionProvider(collapsed_policy)
    game = game_from_snapshot(snapshot)
    first = asyncio.run(
        provider.plan(
            frame_index=frame_index,
            game=game,
            snapshot=snapshot,
            decision=decision,
        )
    )
    provider.observe_resolution(
        DecisionResolved(
            question_index=decision.question_index,
            answers=(2, 0),
            seat=snapshot.our_seat,
            auto_played=False,
        )
    )
    game.step(first.engine_actions[0])

    supply = tuple(
        value for value in decision.offered if value.startswith("0:")
    )
    retry = replace(
        decision,
        question_index=decision.question_index + 1,
        offered=(
            *supply,
            "1:0:Copper",
            "1:0:Copper",
            "2:AUTOPLAY_TREASURES",
        ),
    )
    second = asyncio.run(
        provider.plan(
            frame_index=frame_index + 1,
            game=game,
            snapshot=replace(snapshot, pending_decision=retry),
            decision=retry,
        )
    )
    assert second.gesture_actions == (copper_play, copper_play)
    assert map_engine_actions(second.gesture_actions, retry).answers == (2, 0)
    provider.observe_resolution(
        DecisionResolved(
            question_index=retry.question_index,
            answers=(2, 0),
            seat=snapshot.our_seat,
            auto_played=False,
        )
    )
    game.step(second.engine_actions[0])

    final_treasure = replace(
        retry,
        question_index=retry.question_index + 1,
        offered=(
            *supply,
            "1:0:Copper",
            "2:AUTOPLAY_TREASURES",
        ),
    )
    third = asyncio.run(
        provider.plan(
            frame_index=frame_index + 2,
            game=game,
            snapshot=replace(snapshot, pending_decision=final_treasure),
            decision=final_treasure,
        )
    )

    assert third.engine_actions == (copper_play,)
    assert third.gesture_actions == (copper_play,)
    assert third.answer_hint is not None
    assert len(third.answer_hint) == 4
    assert third.answer_hint[:2] == (1, 1)
    assert policy_calls == 3
    assert "AUTOPLAY BUY FALLBACK" in caplog.text
    assert "'resolved_answers': (2, 0)" in caplog.text


def test_interleaved_question_is_planned_before_deferred_buy_follow_up(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, frame_index, snapshot, decision = _live_turn_one()
    copper_play = int(dz.A_PLAY_BASE + dz.def_id("Copper"))
    silver = int(dz.A_BUY_BASE + dz.def_id("Silver"))
    option = int(dz.A_OPTION_BASE + 1)
    choices = iter((silver, option))
    provider = BotDecisionProvider(
        lambda game, seat: next(choices)
    )
    game = game_from_snapshot(snapshot)
    autoplay = asyncio.run(
        provider.plan(
            frame_index=frame_index,
            game=game,
            snapshot=snapshot,
            decision=decision,
        )
    )
    assert autoplay.engine_actions == (copper_play,) * 3
    for action in autoplay.engine_actions:
        game.step(action)
    provider.observe_resolution(
        DecisionResolved(
            question_index=decision.question_index,
            answers=(2, 0),
            seat=snapshot.our_seat,
            auto_played=False,
        )
    )

    caplog.set_level("WARNING")
    interrupt = _library_mode_decision()
    interrupt_plan = asyncio.run(
        provider.plan(
            frame_index=frame_index + 1,
            game=_BatchedDecisionGame((option,), changes_after=1),
            snapshot=replace(snapshot, pending_decision=interrupt),
            decision=interrupt,
        )
    )
    assert interrupt_plan.engine_actions == (option,)

    follow_up = _without_treasures(decision)
    buy_plan = asyncio.run(
        provider.plan(
            frame_index=frame_index + 2,
            game=game,
            snapshot=replace(snapshot, pending_decision=follow_up),
            decision=follow_up,
        )
    )

    assert buy_plan.engine_actions == (silver,)
    assert "interleaved question arrived before buy follow-up" in caplog.text
    assert "'question_id': 'LIBRARY'" in caplog.text
    assert "'resolved_answers': (2, 0)" in caplog.text


def test_single_resolution_collapse_replans_at_next_turn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, frame_index, snapshot, decision = _live_turn_one()
    silver = int(dz.A_BUY_BASE + dz.def_id("Silver"))
    option = int(dz.A_OPTION_BASE + 1)
    choices = iter((silver, option))
    provider = BotDecisionProvider(
        lambda game, seat: next(choices)
    )
    autoplay = asyncio.run(
        provider.plan(
            frame_index=frame_index,
            game=game_from_snapshot(snapshot),
            snapshot=snapshot,
            decision=decision,
        )
    )
    assert autoplay.gesture_actions
    provider.observe_resolution(
        DecisionResolved(
            question_index=decision.question_index,
            answers=(2, 0),
            seat=snapshot.our_seat,
            auto_played=False,
        )
    )

    caplog.set_level("WARNING")
    next_question = _library_mode_decision()
    next_plan = asyncio.run(
        provider.plan(
            frame_index=frame_index + 1,
            game=_BatchedDecisionGame((option,), changes_after=1),
            snapshot=replace(
                snapshot,
                turn_number=snapshot.turn_number + 1,
                pending_decision=next_question,
            ),
            decision=next_question,
        )
    )

    assert next_plan.engine_actions == (option,)
    assert "belongs to another game, turn, or seat" in caplog.text
    assert "'resolved_answers': (2, 0)" in caplog.text


def test_choose_mode_provider_stops_before_an_identical_next_prompt() -> None:
    snapshot = _ProviderSnapshot()
    decision = _library_mode_decision()
    game = _RepeatedModeDecisionGame()
    planning = game.clone()
    planning.step(int(dz.A_OPTION_BASE + 1))
    assert planning.current_decision() == game.current_decision()
    policy_calls = 0

    def choose_mode(_game: Any, _seat: int) -> int:
        nonlocal policy_calls
        policy_calls += 1
        return int(dz.A_OPTION_BASE + 1)

    plan = asyncio.run(
        BotDecisionProvider(choose_mode).plan(
            frame_index=844,
            game=game,
            snapshot=snapshot,
            decision=decision,
        )
    )

    assert plan.engine_actions == (int(dz.A_OPTION_BASE + 1),)
    assert plan.gesture_actions == plan.engine_actions
    assert policy_calls == 1


@pytest.mark.parametrize(
    (
        "question_id",
        "decision_type",
        "minimum",
        "maximum",
        "actions",
    ),
    (
        (
            "MILITIA",
            "DISCARD",
            2,
            2,
            (
                int(dz.A_SELECT_BASE + dz.def_id("Copper")),
                int(dz.A_SELECT_BASE + dz.def_id("Estate")),
                int(dz.A_SELECT_BASE + dz.def_id("Silver")),
            ),
        ),
        (
            "CELLAR",
            "DISCARD",
            0,
            3,
            (
                int(dz.A_SELECT_BASE + dz.def_id("Copper")),
                int(dz.A_SELECT_BASE + dz.def_id("Estate")),
                int(dz.A_PASS),
            ),
        ),
        (
            "CHAPEL",
            "TRASH",
            0,
            4,
            (
                int(dz.A_SELECT_BASE + dz.def_id("Copper")),
                int(dz.A_SELECT_BASE + dz.def_id("Estate")),
                int(dz.A_SELECT_BASE + dz.def_id("Silver")),
                int(dz.A_SELECT_BASE + dz.def_id("Gold")),
            ),
        ),
    ),
)
def test_provider_keeps_multi_answer_batches(
    question_id: str,
    decision_type: str,
    minimum: int,
    maximum: int,
    actions: tuple[int, ...],
) -> None:
    snapshot = _ProviderSnapshot()
    library = _library_mode_decision()
    decision = replace(
        library,
        question_id=question_id,
        decision_type=decision_type,
        offered=("Copper", "Estate", "Silver", "Gold"),
        minimum=minimum,
        maximum=maximum,
    )
    choices = iter(actions)
    plan = asyncio.run(
        BotDecisionProvider(lambda game, seat: next(choices)).plan(
            frame_index=decision.question_index,
            game=_BatchedDecisionGame(actions, len(actions)),
            snapshot=snapshot,
            decision=decision,
        )
    )

    assert plan.engine_actions == actions
    assert plan.gesture_actions == actions


def test_provider_keeps_sentry_stages_batched() -> None:
    snapshot = _ProviderSnapshot()
    library = _library_mode_decision()
    keep = int(dz.A_OPTION_BASE + 2)
    order = int(dz.A_OPTION_BASE)
    choices = iter((keep, keep, order))
    provider = BotDecisionProvider(lambda game, seat: next(choices))
    game = _BatchedDecisionGame((keep, order), changes_after=3)
    trash = replace(
        library,
        question_id="SENTRY_TRASH",
        decision_type="TRASH",
        offered=("Copper", "Estate"),
        minimum=0,
        maximum=2,
    )
    discard = replace(
        trash,
        question_id="SENTRY_DISCARD",
        decision_type="DISCARD",
    )
    topdeck = replace(
        trash,
        question_id="SENTRY_TOPDECK",
        decision_type="ORDER_CARDS",
        minimum=2,
        maximum=2,
    )

    trash_plan = asyncio.run(
        provider.plan(
            frame_index=trash.question_index,
            game=game,
            snapshot=snapshot,
            decision=trash,
        )
    )
    discard_plan = asyncio.run(
        provider.plan(
            frame_index=discard.question_index,
            game=game,
            snapshot=snapshot,
            decision=discard,
        )
    )
    topdeck_plan = asyncio.run(
        provider.plan(
            frame_index=topdeck.question_index,
            game=game,
            snapshot=snapshot,
            decision=topdeck,
        )
    )

    assert trash_plan.engine_actions == ()
    assert trash_plan.gesture_actions == (keep, keep)
    assert discard_plan.engine_actions == ()
    assert discard_plan.gesture_actions == (keep, keep)
    assert topdeck_plan.engine_actions == (keep, keep, order)
    assert topdeck_plan.gesture_actions == (order,)


def test_choose_mode_mapper_rejects_an_overlength_answer_plan() -> None:
    decision = _library_mode_decision()
    with pytest.raises(
        ActionMappingError,
        match=r"LIBRARY expects exactly 1 answer index, got 2",
    ):
        map_engine_actions(
            (int(dz.A_OPTION_BASE + 1),) * 2,
            decision,
        )


class _RecordedBadSubmission:
    async def plan(
        self,
        *,
        frame_index: int,
        game: Any,
        snapshot: TrackerSnapshot,
        decision: PendingDecisionSnapshot,
    ) -> DecisionPlan:
        del frame_index, game, snapshot
        assert decision.question_index == 3
        copper = int(dz.A_PLAY_BASE + dz.def_id("Copper"))
        return DecisionPlan(
            engine_actions=(copper,),
            gesture_actions=(copper,),
            answer_hint=(1, 1, 0, 0),
        )


def test_live_mismatch_aborts_on_question_three_first_resolution() -> None:
    events, _, _, _ = _live_turn_one()
    actuator = MockActuator(replay=True)

    results = asyncio.run(
        run_game_loop(
            events,
            actuator=actuator,
            decision_provider=_RecordedBadSubmission(),
        )
    )

    assert len(results) == 1
    result = results[0]
    assert result.divergence_aborted
    assert result.divergence_report is not None
    assert result.divergence_report.frame_index == 45
    assert result.divergence_report.question_index == 3
    assert "DecisionResolved answers differ" in result.reason
    assert [gesture.question_index for gesture in actuator.gestures] == [1, 3]
    assert actuator.gestures[0].answer_indices == (0,)
    assert actuator.gestures[1].answer_indices == (1, 1, 0, 0)


class _SavedDOMParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, tuple[str, ...]]] = []
        self.card_stacks_depth: int | None = None
        self.current_card: dict[str, object] | None = None
        self.cards: list[dict[str, object]] = []
        self.game_buttons_depth: int | None = None
        self.buttons: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = {name: value or "" for name, value in attrs}
        classes = tuple(attributes.get("class", "").split())
        depth = len(self.stack)
        if tag == "div" and "card-stacks" in classes:
            self.card_stacks_depth = depth
        elif (
            tag == "div"
            and self.card_stacks_depth is not None
            and depth == self.card_stacks_depth + 1
        ):
            self.current_card = {
                "depth": depth,
                "style": attributes.get("style", ""),
                "name": "",
                "has_visible_counter": False,
                "counter_text": "",
                "has_visible_all": False,
                "all_style": "",
                "selected": False,
            }
            self.cards.append(self.current_card)
        elif self.current_card is not None and "name-layer" in classes:
            if "invisible" not in classes:
                self.current_card["reading_name"] = True
        elif self.current_card is not None and "counter-layer" in classes:
            if "invisible" not in classes:
                self.current_card["has_visible_counter"] = True
                self.current_card["reading_counter"] = True
        elif self.current_card is not None and "all-button" in classes:
            if "invisible" not in classes:
                self.current_card["has_visible_all"] = True
                self.current_card["all_style"] = attributes.get("style", "")
        elif self.current_card is not None and tag == "selection-cross":
            self.current_card["selected"] = True

        if tag == "div" and "game-buttons" in classes:
            self.game_buttons_depth = depth
        elif (
            tag == "canvas"
            and self.game_buttons_depth is not None
            and depth == self.game_buttons_depth + 1
            and "display: none" not in attributes.get("style", "")
        ):
            self.buttons.append(attributes.get("style", ""))
        self.stack.append((tag, classes))

    def handle_data(self, data: str) -> None:
        if (
            self.current_card is not None
            and self.current_card.get("reading_name")
        ):
            self.current_card["name"] = (
                str(self.current_card["name"]) + data
            ).strip()
        if (
            self.current_card is not None
            and self.current_card.get("reading_counter")
        ):
            self.current_card["counter_text"] = (
                str(self.current_card["counter_text"]) + data
            ).strip()

    def handle_endtag(self, tag: str) -> None:
        if not self.stack:
            return
        depth = len(self.stack) - 1
        _, classes = self.stack.pop()
        if self.current_card is not None:
            if "name-layer" in classes:
                self.current_card.pop("reading_name", None)
            if "counter-layer" in classes:
                self.current_card.pop("reading_counter", None)
            if depth == self.current_card["depth"]:
                self.current_card = None
        if self.card_stacks_depth == depth:
            self.card_stacks_depth = None
        if self.game_buttons_depth == depth:
            self.game_buttons_depth = None


def _style_number(style: str, name: str) -> float:
    match = re.search(rf"(?:^|;)\s*{name}:\s*([0-9.]+)", style)
    assert match is not None, (name, style)
    return float(match.group(1))


def _style_translate_x(style: str) -> float:
    match = re.search(r"translateX\(([0-9.]+)px\)", style)
    assert match is not None, style
    return float(match.group(1))


def _dom_stack_facts(parser: _SavedDOMParser) -> tuple[DOMCardStack, ...]:
    facts: list[DOMCardStack] = []
    for card in parser.cards:
        style = str(card["style"])
        if not card["name"] or "display: none" in style:
            continue
        try:
            width = _style_number(style, "width")
            height = _style_number(style, "height")
            z_index = int(_style_number(style, "z-index"))
        except AssertionError:
            continue
        counter_text = str(card["counter_text"])
        selected_count = (
            int(counter_text)
            if card["selected"] and counter_text.isdigit()
            else 1
            if card["selected"]
            else 0
        )
        all_style = str(card["all_style"])
        all_covers_center = False
        if card["has_visible_all"]:
            all_left = _style_number(all_style, "left")
            all_top = _style_number(all_style, "top")
            all_width = _style_number(all_style, "width")
            all_height = _style_number(all_style, "height")
            all_covers_center = (
                all_left <= width / 2 <= all_left + all_width
                and all_top <= height / 2 <= all_top + all_height
            )
        facts.append(
            DOMCardStack(
                identity=str(card["name"]),
                width=width,
                height=height,
                z_index=z_index,
                has_visible_counter=bool(card["has_visible_counter"]),
                selected_count=selected_count,
                clickable="cursor: pointer" in style,
                has_visible_all=bool(card["has_visible_all"]),
                all_covers_center=all_covers_center,
            )
        )
    return tuple(facts)


def _dom_button_facts(parser: _SavedDOMParser) -> tuple[DOMGameButton, ...]:
    return tuple(
        DOMGameButton(
            x=_style_translate_x(style),
            width=_style_number(style, "width"),
            height=_style_number(style, "height"),
        )
        for style in parser.buttons
    )


def test_saved_dom_separates_supply_hand_and_autoplay_targets() -> None:
    if not BUY_DOM.is_file():
        pytest.skip(f"missing saved DOM fixture: {BUY_DOM}")
    parser = _SavedDOMParser()
    parser.feed(BUY_DOM.read_text(encoding="utf-8"))

    gold_regions: set[str] = set()
    for card in parser.cards:
        style = str(card["style"])
        if card["name"] != "Gold" or "display: none" in style:
            continue
        region = card_stack_region(
            width=_style_number(style, "width"),
            height=_style_number(style, "height"),
            z_index=int(_style_number(style, "z-index")),
            has_visible_counter=bool(card["has_visible_counter"]),
        )
        if region is not None:
            gold_regions.add(region)
    assert gold_regions == {"supply", "hand"}

    button_roles = {
        game_button_role(
            width=_style_number(style, "width"),
            height=_style_number(style, "height"),
        )
        for style in parser.buttons
    }
    assert button_roles == {"primary", "secondary"}

    _, _, _, decision = _live_turn_one()
    targets = (
        dom_click_target(decision, "0:Gold", offered_index=0),
        dom_click_target(decision, "1:0:Gold", offered_index=1),
        dom_click_target(
            decision,
            "2:AUTOPLAY_TREASURES",
            offered_index=2,
        ),
    )
    assert [target.region for target in targets] == [
        "supply",
        "hand",
        "autoplay-button",
    ]


def test_saved_discard_dom_resolves_collapsed_stack_selection_and_confirm() -> None:
    if not SELECTED_DISCARD_DOM.is_file():
        pytest.skip(f"missing saved DOM fixture: {SELECTED_DISCARD_DOM}")
    parser = _SavedDOMParser()
    parser.feed(SELECTED_DISCARD_DOM.read_text(encoding="utf-8"))
    stacks = _dom_stack_facts(parser)
    buttons = _dom_button_facts(parser)
    decision = PendingDecisionSnapshot(
        question_index=1246,
        decision_type="TRASH",
        question_id="CHAPEL",
        offered=("Silver", "Silver", "Province", "Silver"),
        minimum=0,
        maximum=4,
        association="Chapel",
    )

    assert selected_card_counts(stacks) == Counter({"Silver": 3})
    silver_target = dom_click_target(
        decision,
        "Silver",
        offered_index=0,
    )
    silver_index = resolve_card_stack_index(silver_target, stacks)
    assert stacks[silver_index].identity == "Silver"
    assert stacks[silver_index].z_index == 2000
    assert stacks[silver_index].has_visible_all
    assert not stacks[silver_index].all_covers_center

    assert [game_button_role(width=b.width, height=b.height) for b in buttons] == [
        "primary",
        "secondary",
    ]
    assert resolve_submit_button_index(decision, buttons) == 0

    ambiguous = (
        *stacks,
        replace(stacks[silver_index], z_index=2004),
    )
    with pytest.raises(ActuationError, match="found=2"):
        resolve_card_stack_index(silver_target, ambiguous)
    with pytest.raises(ActuationError, match="found=2"):
        resolve_submit_button_index(
            decision,
            (*buttons, replace(buttons[0], x=buttons[0].x + 20)),
        )


def _live_militia() -> tuple[
    tuple[object, ...],
    int,
    PendingDecision,
    PendingDecisionSnapshot,
]:
    if not LIVE_GAME_2_ARCHIVE.is_file():
        pytest.skip(f"missing live arena fixture: {LIVE_GAME_2_ARCHIVE}")
    events = parse_recording(LIVE_GAME_2_ARCHIVE).events
    tracker = Tracker()
    for frame_index, event in enumerate(events):
        tracker.consume(event)
        if isinstance(event, PendingDecision) and event.question_index == 55:
            snapshot = tracker.snapshot()
            assert snapshot.pending_decision is not None
            return events, frame_index, event, snapshot.pending_decision
    raise AssertionError("live game 2 fixture has no question 55")


def test_live_militia_and_reference_moat_have_executable_click_plans() -> None:
    _, _, _, militia = _live_militia()
    gold = int(dz.A_SELECT_BASE + dz.def_id("Gold"))
    copper = int(dz.A_SELECT_BASE + dz.def_id("Copper"))
    actuator = MockActuator(replay=True)
    gesture = asyncio.run(
        actuator.act(
            copper,
            militia,
            militia.offered,
            prior_actions=(gold, copper),
        )
    )
    targets = gesture_click_targets(gesture, militia, militia.offered)

    assert gesture.answer_indices == (3, 4)
    assert gesture.click_button
    assert [
        (target.region, target.identity, target.offered_index)
        for target in targets
    ] == [
        ("hand", "Copper", 3),
        ("hand", "Militia", 4),
        ("submit-button", "MILITIA", -1),
    ]

    if not REFERENCE_RECORDING.is_file():
        pytest.skip(f"missing reference fixture: {REFERENCE_RECORDING}")
    reference = parse_recording(REFERENCE_RECORDING).events
    assert any(
        isinstance(event, GameStart)
        and {"Witch", "Moat"} <= set(event.kingdom)
        for event in reference
    )
    moat = PendingDecisionSnapshot(
        question_index=1,
        decision_type="REVEAL",
        question_id="GAME_MAY_REACT_WITH",
        offered=("Moat",),
        minimum=0,
        maximum=1,
        association="Witch",
    )
    reveal = asyncio.run(
        MockActuator(replay=True).act(
            int(dz.A_SELECT_BASE + dz.def_id("Moat")),
            moat,
            moat.offered,
        )
    )
    decline = asyncio.run(
        MockActuator(replay=True).act(
            int(dz.A_PASS),
            moat,
            moat.offered,
        )
    )
    assert [
        (target.region, target.identity)
        for target in gesture_click_targets(reveal, moat, moat.offered)
    ] == [("hand", "Moat")]
    assert [
        (target.region, target.identity)
        for target in gesture_click_targets(decline, moat, moat.offered)
    ] == [("decline-button", "GAME_MAY_REACT_WITH")]


def test_live_game_2_replays_through_militia_question_55() -> None:
    events, _, question, _ = _live_militia()
    resolved = DecisionResolved(
        timestamp_ms=question.timestamp_ms + 1,
        question_index=question.question_index,
        answers=(3, 4),
        seat=1,
        auto_played=False,
    )
    replay_events = (*events, resolved)
    actuator = MockActuator(replay=True)

    results = asyncio.run(
        run_game_loop(
            replay_events,
            actuator=actuator,
            decision_provider=RecordedDecisionProvider(replay_events),
        )
    )

    assert not any(result.divergence_aborted for result in results)
    assert actuator.gestures[-1].question_index == 55
    assert actuator.gestures[-1].answer_indices == (3, 4)
    assert actuator.gestures[-1].click_button


class _RecordedReplayWithLibraryBot:
    """Use the live provider for the repeated Library mode prompts only."""

    def __init__(self, events: tuple[object, ...]) -> None:
        self.recorded = RecordedDecisionProvider(events)
        self.library = BotDecisionProvider(
            lambda game, seat: int(dz.A_OPTION_BASE + 1)
        )
        self.library_plans: dict[int, DecisionPlan] = {}

    async def plan(
        self,
        *,
        frame_index: int,
        game: Any,
        snapshot: TrackerSnapshot,
        decision: PendingDecisionSnapshot,
    ) -> DecisionPlan | None:
        if decision.question_id == "LIBRARY":
            plan = await self.library.plan(
                frame_index=frame_index,
                game=game,
                snapshot=snapshot,
                decision=decision,
            )
            self.library_plans[decision.question_index] = plan
            return plan
        return await self.recorded.plan(
            frame_index=frame_index,
            game=game,
            snapshot=snapshot,
            decision=decision,
        )


def test_live_game_3_replays_library_modes_as_separate_plans() -> None:
    if not LIVE_GAME_3_ARCHIVE.is_file():
        pytest.skip(f"missing live arena fixture: {LIVE_GAME_3_ARCHIVE}")
    events = parse_recording(LIVE_GAME_3_ARCHIVE).events
    provider = _RecordedReplayWithLibraryBot(events)
    actuator = MockActuator(replay=True)

    results = asyncio.run(
        run_game_loop(
            events,
            actuator=actuator,
            decision_provider=provider,
        )
    )

    mode_action = int(dz.A_OPTION_BASE + 1)
    assert len(results) == 1
    assert results[0].completed
    assert not results[0].divergence_aborted
    assert {
        question_index: plan.engine_actions
        for question_index, plan in provider.library_plans.items()
    } == {
        64: (mode_action,),
        65: (mode_action,),
    }
    assert [
        gesture.answer_indices
        for gesture in actuator.gestures
        if gesture.question_index in {64, 65}
    ] == [(1,), (1,)]


class _EmptyLocator:
    async def count(self) -> int:
        return 0


class _EmptyPage:
    def locator(self, selector: str) -> _EmptyLocator:
        assert selector == "div.card-stacks > div"
        return _EmptyLocator()


class _CanvasButton:
    def __init__(
        self,
        page: _StartModalPage,
        index: int,
        box: dict[str, float],
    ) -> None:
        self.page = page
        self.index = index
        self.box = box

    async def is_visible(self) -> bool:
        return True

    async def bounding_box(self) -> dict[str, float]:
        return self.box

    async def click(self) -> None:
        self.page.clicks.append(self.index)


class _CanvasButtons:
    def __init__(self, page: _StartModalPage) -> None:
        self.page = page

    async def count(self) -> int:
        return len(self.page.boxes)

    def nth(self, index: int) -> _CanvasButton:
        return _CanvasButton(self.page, index, self.page.boxes[index])


class _StartModalPage:
    def __init__(self, boxes: list[dict[str, float]]) -> None:
        self.boxes = boxes
        self.clicks: list[int] = []

    def locator(self, selector: str) -> _CanvasButtons:
        assert selector == "div.game-buttons canvas"
        return _CanvasButtons(self)


def _start_confirmation_decision() -> PendingDecisionSnapshot:
    return PendingDecisionSnapshot(
        question_index=2,
        decision_type="CHOOSE_MODE",
        question_id="question-session-relative",
        offered=("card-mode-97",),
        minimum=1,
        maximum=1,
        association=None,
    )


def test_start_confirmation_uses_the_single_primary_canvas_fallback() -> None:
    page = _StartModalPage(
        [
            {"x": 10.0, "width": 200.0, "height": 40.0},
            {"x": 220.0, "width": 40.0, "height": 40.0},
        ]
    )

    gesture = asyncio.run(
        PlaywrightActuator(page).act(
            int(dz.A_OPTION_BASE),
            _start_confirmation_decision(),
            ("card-mode-97",),
        )
    )

    assert gesture.answer_indices == (0,)
    assert page.clicks == [0]


def test_start_confirmation_failure_snapshots_dom_and_names_search() -> None:
    page = _StartModalPage(
        [
            {"x": 10.0, "width": 40.0, "height": 40.0},
            {"x": 60.0, "width": 40.0, "height": 40.0},
        ]
    )
    snapshots: list[str] = []

    async def snapshot_dom(label: str) -> str:
        snapshots.append(label)
        return "dom-start-confirmation.html"

    with pytest.raises(ActuationError) as caught:
        asyncio.run(
            PlaywrightActuator(
                page,
                snapshot_dom=snapshot_dom,
                actuation_timeout_seconds=0,
            ).act(
                int(dz.A_OPTION_BASE),
                _start_confirmation_decision(),
                ("card-mode-97",),
            )
        )

    message = str(caught.value)
    assert "start-confirmation searched div.game-buttons canvas" in message
    assert "sole visible mode/primary button" in message
    assert "DOM snapshot=dom-start-confirmation.html" in message
    assert snapshots == [
        "actuation-failure-start-confirmation-question-2"
    ]
    assert page.clicks == []


def test_actuation_error_reports_offered_targets_and_failed_step() -> None:
    _, _, _, decision = _live_militia()
    gold = int(dz.A_SELECT_BASE + dz.def_id("Gold"))
    copper = int(dz.A_SELECT_BASE + dz.def_id("Copper"))

    with pytest.raises(ActuationError) as caught:
        asyncio.run(
            PlaywrightActuator(
                _EmptyPage(),
                actuation_timeout_seconds=0,
            ).act(
                copper,
                decision,
                decision.offered,
                prior_actions=(gold, copper),
            )
        )

    message = str(caught.value)
    assert "gesture (3, 4)" in message
    assert f"offered={decision.offered!r}" in message
    assert "resolved_targets=" in message
    assert "step=1/3" in message
    assert "target=DOMClickTarget(region='hand', identity='Copper'" in message
    assert "status=not-found" in message
    assert "found=0" in message


CLEANUP_SHUFFLE_AT_GAME_END_ARCHIVE = Path(
    "exports/arena/20261111T090857.411949Z/frames.jsonl"
)


def test_game_ending_on_our_turn_with_a_cleanup_shuffle_completes() -> None:
    """Game 185802782: our Province buy ended the game and cleanup reshuffled.

    The final step never crosses a turn boundary, so it carries the cleanup
    shuffle, whose draw (four Provinces from this turn's hand and gain) is not
    in the turn-start deck/discard pool.  The driver aborted a won game with
    "shuffle reconstruction observed unavailable 'Province'".
    """
    if not CLEANUP_SHUFFLE_AT_GAME_END_ARCHIVE.is_file():
        pytest.skip(
            f"missing live arena fixture: {CLEANUP_SHUFFLE_AT_GAME_END_ARCHIVE}"
        )
    events = parse_recording(CLEANUP_SHUFFLE_AT_GAME_END_ARCHIVE).events

    results = asyncio.run(
        run_game_loop(
            events,
            actuator=MockActuator(replay=True),
            decision_provider=RecordedDecisionProvider(events),
        )
    )

    assert [result.game_id for result in results] == [185802619, 185802782]
    won = results[1]
    assert won.completed
    assert not won.divergence_aborted
    assert won.reason == "game-finished"
    assert won.scores == (30, 19)
    assert won.outcome == "win"


THRONED_CHAPEL_ARCHIVE = Path("exports/arena/20261111T092905.220503Z/frames.jsonl")
THRONED_CHAPEL_GAME_ID = 185803487
THRONED_CHAPEL_QUESTIONS = (100, 101)
_TRASH_ESTATE = 256
_TRASH_CURSE = 260


class _RecordedReplayWithChapelBot:
    """Use the live provider for the two Throne Room Chapel questions only."""

    def __init__(self, events: tuple[object, ...]) -> None:
        self.recorded = RecordedDecisionProvider(events)
        self.chapel = BotDecisionProvider(self._trash_junk_then_stop)
        self.chapel_plans: dict[int, DecisionPlan] = {}

    @staticmethod
    def _trash_junk_then_stop(game: Any, seat: int) -> int:
        legal = game.legal_mask()
        for action in (_TRASH_ESTATE, _TRASH_CURSE):
            if legal[action]:
                return action
        return int(dz.A_PASS)

    async def plan(
        self,
        *,
        frame_index: int,
        game: Any,
        snapshot: TrackerSnapshot,
        decision: PendingDecisionSnapshot,
    ) -> DecisionPlan | None:
        if (
            snapshot.game_id == THRONED_CHAPEL_GAME_ID
            and decision.question_index in THRONED_CHAPEL_QUESTIONS
        ):
            plan = await self.chapel.plan(
                frame_index=frame_index,
                game=game,
                snapshot=snapshot,
                decision=decision,
            )
            self.chapel_plans[decision.question_index] = plan
            return plan
        return await self.recorded.plan(
            frame_index=frame_index,
            game=game,
            snapshot=snapshot,
            decision=decision,
        )


def test_throne_room_chapel_plans_each_chapel_question_separately() -> None:
    """Game 185803487: Throne Room played Chapel with Estate, Curse, Silver.

    The second Chapel is a new native decision with the same player, kind and
    source as the first.  The planner ran on into it, answering "trash
    nothing" inside the first question's plan, so the shadow reached the buy
    phase while the client asked the second Chapel question and the driver
    aborted with "native/client decision kinds differ".
    """
    if not THRONED_CHAPEL_ARCHIVE.is_file():
        pytest.skip(f"missing live arena fixture: {THRONED_CHAPEL_ARCHIVE}")
    events = parse_recording(THRONED_CHAPEL_ARCHIVE).events
    provider = _RecordedReplayWithChapelBot(events)

    results = asyncio.run(
        run_game_loop(
            events,
            actuator=MockActuator(replay=True),
            decision_provider=provider,
        )
    )

    assert not any(result.divergence_aborted for result in results)
    assert {
        question_index: plan.engine_actions
        for question_index, plan in provider.chapel_plans.items()
    } == {
        100: (_TRASH_ESTATE, _TRASH_CURSE, int(dz.A_PASS)),
        101: (int(dz.A_PASS),),
    }


CELLAR_SHUFFLE_ARCHIVE = Path("exports/arena/20261111T100415.808272Z/frames.jsonl")
CELLAR_SHUFFLE_GAME_ID = 185803914
CELLAR_QUESTION = 115
ACTION_QUESTION_AFTER_CELLAR = 116


class _RecordedReplayWithQuestionLog:
    """Replay the recorded answers and note each question one game reached."""

    def __init__(self, events: tuple[object, ...], game_id: int) -> None:
        self.recorded = RecordedDecisionProvider(events)
        self.game_id = game_id
        self.questions: list[tuple[int, str]] = []

    async def plan(
        self,
        *,
        frame_index: int,
        game: Any,
        snapshot: TrackerSnapshot,
        decision: PendingDecisionSnapshot,
    ) -> DecisionPlan | None:
        if snapshot.game_id == self.game_id:
            self.questions.append((decision.question_index, decision.question_id))
        return await self.recorded.plan(
            frame_index=frame_index,
            game=game,
            snapshot=snapshot,
            decision=decision,
        )


def test_cellar_discards_shuffled_back_into_its_own_draw_do_not_abort() -> None:
    """Game 185803914: Cellar discarded five cards and drew through a shuffle.

    The deck was short, so the shuffle took in the Coppers and Estates that
    Cellar had just discarded from hand and the draw returned some of them.
    The turn-start deck/discard pool cannot supply those cards, and the driver
    aborted with "shuffle reconstruction observed unavailable 'Estate'" although
    the next question was a clean action phase that rebuilds the shadow from
    the tracker anyway.
    """
    if not CELLAR_SHUFFLE_ARCHIVE.is_file():
        pytest.skip(f"missing live arena fixture: {CELLAR_SHUFFLE_ARCHIVE}")
    events = parse_recording(CELLAR_SHUFFLE_ARCHIVE).events
    provider = _RecordedReplayWithQuestionLog(events, CELLAR_SHUFFLE_GAME_ID)

    results = asyncio.run(
        run_game_loop(
            events,
            actuator=MockActuator(replay=True),
            decision_provider=provider,
        )
    )

    assert not any(result.divergence_aborted for result in results)
    assert provider.questions[-2:] == [
        (CELLAR_QUESTION, "CELLAR"),
        (ACTION_QUESTION_AFTER_CELLAR, "GAME_ACTION_PHASE"),
    ]


MERCHANT_TRIGGERS_ARCHIVE = Path("exports/arena/20261111T103516.286873Z/frames.jsonl")
MERCHANT_TRIGGERS_GAME_ID = 185804954
MERCHANT_BUY_QUESTION = 166
_PLAY_COPPER = int(dz.A_PLAY_BASE + dz.def_id("Copper"))
_PLAY_SILVER = int(dz.A_PLAY_BASE + dz.def_id("Silver"))
_BUY_PROVINCE = int(dz.A_BUY_BASE + dz.def_id("Province"))


class _RecordedReplayWithBuyBot:
    """Use the live provider for one buy question and keep its shadow."""

    def __init__(self, events: tuple[object, ...]) -> None:
        self.recorded = RecordedDecisionProvider(events)
        self.bot = BotDecisionProvider(self._buy_province)
        self.shadow: Any = None
        self.plan_at_buy: DecisionPlan | None = None
        self.coins_when_choosing: int | None = None

    def _buy_province(self, game: Any, seat: int) -> int:
        self.coins_when_choosing = int(game.resources()["coins"])
        return _BUY_PROVINCE if game.legal_mask()[_BUY_PROVINCE] else int(dz.A_PASS)

    async def plan(
        self,
        *,
        frame_index: int,
        game: Any,
        snapshot: TrackerSnapshot,
        decision: PendingDecisionSnapshot,
    ) -> DecisionPlan | None:
        if (
            snapshot.game_id == MERCHANT_TRIGGERS_GAME_ID
            and decision.question_index == MERCHANT_BUY_QUESTION
        ):
            self.shadow = game.clone()
            self.plan_at_buy = await self.bot.plan(
                frame_index=frame_index,
                game=game,
                snapshot=snapshot,
                decision=decision,
            )
            return self.plan_at_buy
        return await self.recorded.plan(
            frame_index=frame_index,
            game=game,
            snapshot=snapshot,
            decision=decision,
        )


def test_silver_played_with_several_merchants_resolves_trigger_order() -> None:
    """Game 185804954: four Merchants in play, two Silvers in hand at $2.

    The first Silver triggers all four Merchants.  The native game asks in
    which order to resolve them; the client does not.  The buy planner left
    that choice pending, so the second Silver was illegal and the driver
    aborted with "offered hand treasure is not legal in the shadow game".
    """
    if not MERCHANT_TRIGGERS_ARCHIVE.is_file():
        pytest.skip(f"missing live arena fixture: {MERCHANT_TRIGGERS_ARCHIVE}")
    events = parse_recording(MERCHANT_TRIGGERS_ARCHIVE).events
    provider = _RecordedReplayWithBuyBot(events)

    results = asyncio.run(
        run_game_loop(
            events,
            actuator=MockActuator(replay=True),
            decision_provider=provider,
        )
    )

    assert not any(result.divergence_aborted for result in results)
    treasures = (*([_PLAY_COPPER] * 6), *([_PLAY_SILVER] * 2))
    assert provider.plan_at_buy is not None
    assert provider.plan_at_buy.gesture_actions == treasures
    # 2 + six Coppers + two Silvers + four Merchant bonuses.
    assert provider.coins_when_choosing == 16

    # The same plays applied to the live shadow once the client answers.
    _step_actions(provider.shadow, treasures, frame_index=0, intended=None)
    assert int(provider.shadow.current_decision()["kind"]) == 2
    assert int(provider.shadow.resources()["coins"]) == 16


THRONE_DECLINE_ARCHIVE = Path("exports/arena/20261111T124901.220957Z/frames.jsonl")
THRONE_DECLINE_GAME_ID = 185811117
THRONE_DECLINE_QUESTION = 159
HUMAN_THRONE_DECLINE_ARCHIVE = Path(
    "exports/arena/20260726T142200.662178Z/frames.jsonl"
)
HUMAN_THRONE_DECLINE_GAME_ID = 181422355
HUMAN_THRONE_DECLINE_QUESTION = 195


def _events_of_game(events: tuple[object, ...], game_id: int) -> tuple[object, ...]:
    start = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, GameStart) and event.game_id == game_id
    )
    end = next(
        (
            index
            for index in range(start + 1, len(events))
            if isinstance(events[index], GameStart)
        ),
        len(events),
    )
    return events[start:end]


def _throne_room_snapshot(
    offered: tuple[str, ...] = ("0:Council Room",),
) -> PendingDecisionSnapshot:
    return PendingDecisionSnapshot(
        question_index=THRONE_DECLINE_QUESTION,
        decision_type="COMPLEX_AND",
        question_id="THRONE_ROOM",
        offered=offered,
        minimum=0,
        maximum=2,
        association="Throne Room",
    )


def test_declining_a_throne_room_target_sends_an_empty_selection() -> None:
    """Game 185811117: the bot played Throne Room and chose to play nothing.

    The driver answered the COMPLEX_AND question with a bare ``[]``; the
    server replied with error message 1 and the run aborted.  The client's
    encoding is ``[selection length][selection][way length]``, so a decline
    is ``[0, 0]``.
    """
    decision = _throne_room_snapshot()

    mapping = map_engine_actions((int(dz.A_PASS),), decision)

    assert mapping.answers == (0, 0)
    assert mapping.selected_indices == ()
    assert not mapping.ambiguous
    council_room = int(dz.A_SELECT_BASE + dz.def_id("Council Room"))
    assert map_engine_actions((council_room,), decision).answers == (1, 0, 0)

    sent: list[tuple[int, bytes]] = []

    async def send_frame(msg_type: int, payload: bytes) -> None:
        sent.append((msg_type, payload))

    asyncio.run(
        ProtocolActuator(send_frame).act(int(dz.A_PASS), decision, decision.offered)
    )
    assert sent == [
        (
            ANSWER_QUESTION,
            encode_answer(THRONE_DECLINE_QUESTION, (0, 0), auto_played=False),
        )
    ]


def test_recorded_throne_room_decline_replays_as_a_pass() -> None:
    question = PendingDecision(
        question_index=THRONE_DECLINE_QUESTION,
        decision_type="COMPLEX_AND",
        question_id="THRONE_ROOM",
        offered=("0:Council Room",),
        minimum=0,
        maximum=2,
        association="Throne Room",
    )

    assert _recorded_plan(question, (0, 0)).engine_actions == (int(dz.A_PASS),)
    council_room = int(dz.A_SELECT_BASE + dz.def_id("Council Room"))
    assert _recorded_plan(question, (1, 0, 0)).engine_actions == (council_room,)


def test_live_archive_shows_the_server_rejecting_a_bare_empty_throne_answer() -> None:
    if not THRONE_DECLINE_ARCHIVE.is_file():
        pytest.skip(f"missing live arena fixture: {THRONE_DECLINE_ARCHIVE}")
    events = _events_of_game(
        parse_recording(THRONE_DECLINE_ARCHIVE).events, THRONE_DECLINE_GAME_ID
    )
    asked = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, PendingDecision)
        and event.question_index == THRONE_DECLINE_QUESTION
    )
    question = events[asked]
    assert isinstance(question, PendingDecision)
    assert question.question_id == "THRONE_ROOM"
    assert question.decision_type == "COMPLEX_AND"
    assert (question.minimum, question.maximum) == (0, 2)

    answer, reply = events[asked + 1], events[asked + 2]
    assert isinstance(answer, DecisionResolved)
    assert tuple(answer.answers) == ()
    assert isinstance(reply, UnknownFrame)
    assert (reply.direction, reply.msg_type) == ("in", 1)


def test_human_throne_room_decline_is_logged_as_zero_zero() -> None:
    """Game 181422355: an opponent's own client declined a Throne Room target.

    The server echoes every player's answers.  Decision 195 follows the
    Throne Room play directly, plays nothing, and is followed by the action
    phase pass ``[0, 0, 0]`` as decision 196.
    """
    if not HUMAN_THRONE_DECLINE_ARCHIVE.is_file():
        pytest.skip(f"missing live arena fixture: {HUMAN_THRONE_DECLINE_ARCHIVE}")
    events = _events_of_game(
        parse_recording(HUMAN_THRONE_DECLINE_ARCHIVE).events,
        HUMAN_THRONE_DECLINE_GAME_ID,
    )
    declined = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, DecisionResolved)
        and event.question_index == HUMAN_THRONE_DECLINE_QUESTION
    )
    answer = events[declined]
    assert isinstance(answer, DecisionResolved)
    assert tuple(answer.answers) == (0, 0)

    seat = answer.seat
    before = [
        event
        for event in events[:declined]
        if isinstance(event, Play) and event.seat == seat
    ]
    assert tuple(before[-1].cards) == ("Throne Room",)
    next_question = next(
        index
        for index in range(declined + 1, len(events))
        if isinstance(events[index], DecisionResolved)
        and events[index].question_index == HUMAN_THRONE_DECLINE_QUESTION + 1
    )
    assert tuple(events[next_question].answers) == (0, 0, 0)
    assert not any(
        isinstance(event, Play) and event.seat == seat
        for event in events[declined:next_question]
    )


THRONED_VASSAL_MONEYLENDER_ARCHIVE = Path(
    "exports/arena/20261111T160015.763535Z/frames.jsonl"
)
THRONED_VASSAL_MONEYLENDER_GAME_ID = 185817129
SECOND_VASSAL_QUESTION = 193


class _RecordedReplayWithDecisionLog:
    """Replay the recorded answers and note the native decision at each one."""

    def __init__(self, events: tuple[object, ...]) -> None:
        self.recorded = RecordedDecisionProvider(events)
        self.native: dict[int, tuple[int, tuple[int, ...]]] = {}

    async def plan(
        self,
        *,
        frame_index: int,
        game: Any,
        snapshot: TrackerSnapshot,
        decision: PendingDecisionSnapshot,
    ) -> DecisionPlan | None:
        self.native[decision.question_index] = (
            int(game.current_decision()["kind"]),
            tuple(index for index, legal in enumerate(game.legal_mask()) if legal),
        )
        return await self.recorded.plan(
            frame_index=frame_index,
            game=game,
            snapshot=snapshot,
            decision=decision,
        )


def test_moneylender_played_by_a_throned_vassal_follows_the_client_auto_trash() -> None:
    """Game 185817129: Throne Room on Vassal, whose first play hit Moneylender.

    dominion.games trashes Moneylender's Copper without asking (the answer is
    logged as auto-played).  The native interpreter asks.  After a Moneylender
    played from hand the next question is a clean action or buy phase, which
    rebuilds the shadow, so the gap never showed.  Here the next question was
    the second Vassal's, mid-chain, and the driver aborted with "native/client
    decision kinds differ without a forced bridge: native=4 client=6
    legal=(0, 251)".
    """
    if not THRONED_VASSAL_MONEYLENDER_ARCHIVE.is_file():
        pytest.skip(
            f"missing live arena fixture: {THRONED_VASSAL_MONEYLENDER_ARCHIVE}"
        )
    events = _events_of_game(
        parse_recording(THRONED_VASSAL_MONEYLENDER_ARCHIVE).events,
        THRONED_VASSAL_MONEYLENDER_GAME_ID,
    )
    provider = _RecordedReplayWithDecisionLog(events)

    results = asyncio.run(
        run_game_loop(
            events,
            actuator=MockActuator(replay=True),
            decision_provider=provider,
        )
    )

    assert not any(result.divergence_aborted for result in results)
    kind, legal = provider.native[SECOND_VASSAL_QUESTION]
    assert kind == 6
    assert len(legal) == 2


THRONED_SENTRY_ARCHIVE = Path(
    "exports/arena/20261111T181356.236018Z/frames.jsonl"
)
THRONED_SENTRY_GAME_ID = 185820625
SECOND_SENTRY_TRASH_QUESTION = 173
SECOND_SENTRY_DISCARD_QUESTION = 174
SECOND_SENTRY_ORDER_QUESTION = 175


class _RecordedReplayWithShadowLog:
    """Replay the recorded answers and keep the shadow seen at each question."""

    def __init__(self, events: tuple[object, ...]) -> None:
        self.recorded = RecordedDecisionProvider(events)
        self.seen: dict[
            int,
            tuple[int, Any, TrackerSnapshot, PendingDecisionSnapshot],
        ] = {}

    async def plan(
        self,
        *,
        frame_index: int,
        game: Any,
        snapshot: TrackerSnapshot,
        decision: PendingDecisionSnapshot,
    ) -> DecisionPlan | None:
        self.seen[decision.question_index] = (
            frame_index,
            game.clone(),
            snapshot,
            decision,
        )
        return await self.recorded.plan(
            frame_index=frame_index,
            game=game,
            snapshot=snapshot,
            decision=decision,
        )


def _replay_throned_sentry_game() -> tuple[
    _RecordedReplayWithShadowLog,
    tuple[Any, ...],
]:
    if not THRONED_SENTRY_ARCHIVE.is_file():
        pytest.skip(f"missing live arena fixture: {THRONED_SENTRY_ARCHIVE}")
    events = _events_of_game(
        parse_recording(THRONED_SENTRY_ARCHIVE).events,
        THRONED_SENTRY_GAME_ID,
    )
    provider = _RecordedReplayWithShadowLog(events)
    results = asyncio.run(
        run_game_loop(
            events,
            actuator=MockActuator(replay=True),
            decision_provider=provider,
        )
    )
    return provider, tuple(results)


def test_throned_sentry_orders_cards_by_name_not_by_offer_position() -> None:
    """Game 185820625: Throne Room on Sentry, both plays keeping both cards.

    The first Sentry put Vassal and Artisan back.  The second drew Artisan and
    looked at that Vassal plus a fresh Sentry.  The shadow looks at them top
    first (Vassal, Sentry); the client offered them as (Sentry, Vassal).  The
    order answer was applied by position, so the shadow put Vassal on top
    where the client put Sentry, and the next Vassal play aborted with
    "shadow/tracker deck compositions differ before rigged step".
    """
    provider, results = _replay_throned_sentry_game()

    assert not any(result.divergence_aborted for result in results)
    _, shadow, _, decision = provider.seen[SECOND_SENTRY_TRASH_QUESTION]
    looked_at = tuple(
        int(card) for card in shadow.decision_context()["subject_defs"]
    )
    assert decision.offered == ("Sentry", "Vassal")
    assert looked_at == (int(dz.def_id("Vassal")), int(dz.def_id("Sentry")))


def test_bot_sentry_choices_reach_the_client_by_card_name() -> None:
    """The policy classifies the shadow's cards; the client must get the same cards."""
    provider, _ = _replay_throned_sentry_game()
    vassal = int(dz.def_id("Vassal"))
    trash = int(dz.A_OPTION_BASE)
    keep = int(dz.A_OPTION_BASE + 2)

    def trash_the_vassal(game: Any, seat: int) -> int:
        context = game.decision_context()
        subject = int(context["subject_defs"][context["subject_index"]])
        return trash if subject == vassal else keep

    frame_index, shadow, snapshot, decision = provider.seen[
        SECOND_SENTRY_TRASH_QUESTION
    ]
    trash_plan = asyncio.run(
        BotDecisionProvider(trash_the_vassal).plan(
            frame_index=frame_index,
            game=shadow.clone(),
            snapshot=snapshot,
            decision=decision,
        )
    )
    assert possible_answer_indices(trash_plan.gesture_actions, decision) == (
        (decision.offered.index("Vassal"),),
    )

    # Keep both, then put the shadow's first card (Vassal) on top.
    def vassal_on_top(game: Any, seat: int) -> int:
        if int(game.current_decision()["kind"]) == 7:
            return int(dz.A_OPTION_BASE)
        return keep

    bot = BotDecisionProvider(vassal_on_top)
    order_plan = None
    for question in (
        SECOND_SENTRY_TRASH_QUESTION,
        SECOND_SENTRY_DISCARD_QUESTION,
        SECOND_SENTRY_ORDER_QUESTION,
    ):
        frame_index, _, snapshot, decision = provider.seen[question]
        order_plan = asyncio.run(
            bot.plan(
                frame_index=frame_index,
                game=shadow.clone(),
                snapshot=snapshot,
                decision=decision,
            )
        )
    assert order_plan is not None
    # The client puts the last answered card on top.
    answers = possible_answer_indices(order_plan.gesture_actions, decision)
    assert answers == ((0, 1),)
    assert decision.offered[answers[0][-1]] == "Vassal"
    # The Sentry was played by a Throne Room that a Throne Room'd Vassal had
    # played, so the Vassal's second play follows and discards the top card.
    stepped = shadow.clone()
    for action in order_plan.engine_actions:
        stepped.step(action)
    assert int(stepped.current_decision()["source"]) == vassal
    assert int(stepped.discard_top(snapshot.our_seat)) == vassal


THRONED_MILITIA_ARCHIVE = Path(
    "exports/arena/20261111T194423.643533Z/frames.jsonl"
)
THRONED_MILITIA_GAME_ID = 185826880
THRONED_MILITIA_PLAY_QUESTION = 138
SECOND_THRONE_AFTER_MILITIA_QUESTION = 140


def _replay_throned_militia_game() -> tuple[
    _RecordedReplayWithShadowLog,
    tuple[Any, ...],
]:
    if not THRONED_MILITIA_ARCHIVE.is_file():
        pytest.skip(f"missing live arena fixture: {THRONED_MILITIA_ARCHIVE}")
    events = _events_of_game(
        parse_recording(THRONED_MILITIA_ARCHIVE).events,
        THRONED_MILITIA_GAME_ID,
    )
    provider = _RecordedReplayWithShadowLog(events)
    results = asyncio.run(
        run_game_loop(
            events,
            actuator=MockActuator(replay=True),
            decision_provider=provider,
        )
    )
    return provider, tuple(results)


def test_opponent_militia_discard_is_resolved_inside_a_throne_room_chain() -> None:
    """Game 185826880: Throne Room, Throne Room, Moat, then Militia.

    The first Militia play made the opponent discard.  The native interpreter
    asks the opponent which cards to keep; the client only reports the discard
    and then asks us what the second Throne Room plays.  Both are card
    selections, so the shadow was left on the opponent's choice, the bot picked
    "keep Gold" for the opponent, and the send failed with "select action def 2
    is absent from offered elements".
    """
    provider, results = _replay_throned_militia_game()

    assert not any(result.divergence_aborted for result in results)
    _, shadow, snapshot, decision = provider.seen[
        SECOND_THRONE_AFTER_MILITIA_QUESTION
    ]
    current = shadow.current_decision()
    opponent = 1 - snapshot.our_seat
    assert decision.question_id == "THRONE_ROOM"
    assert int(current["player"]) == snapshot.our_seat
    assert int(current["source"]) == int(dz.def_id("Throne Room"))
    assert shadow.hand_count(opponent) == 3
    assert snapshot.seats[opponent].hand_count == 3
    legal = {
        index for index, value in enumerate(shadow.legal_mask()) if value
    }
    assert legal == {
        int(dz.A_PASS),
        *(
            int(dz.A_SELECT_BASE + dz.def_id(offered_name(value)))
            for value in decision.offered
        ),
    }


FOUR_CARD_MILITIA_ARCHIVE = Path(
    "exports/arena/20261112T214112.021802Z/"
    "20261112T233826.547837Z-game-185871928/frames.jsonl"
)
FOUR_CARD_MILITIA_GAME_ID = 185871928
FOUR_CARD_MILITIA_QUESTION = 46


def test_militia_against_a_four_card_hand_plans_three_keeps() -> None:
    """Game 185871928: our whole deck was Chapel, Vassal, Vassal, Militia.

    The opponent's Militia asked for one discard.  One client answer was taken
    to mean one native action, so the plan held a single keep and the send
    failed with "expected 3 keep actions for 1 discards, got 1".
    """
    if not FOUR_CARD_MILITIA_ARCHIVE.is_file():
        pytest.skip(f"missing live arena fixture: {FOUR_CARD_MILITIA_ARCHIVE}")
    events = _events_of_game(
        parse_recording(FOUR_CARD_MILITIA_ARCHIVE).events,
        FOUR_CARD_MILITIA_GAME_ID,
    )
    provider = _RecordedReplayWithShadowLog(events)
    asyncio.run(
        run_game_loop(
            events,
            actuator=MockActuator(replay=True),
            decision_provider=provider,
        )
    )
    frame_index, shadow, snapshot, decision = provider.seen[
        FOUR_CARD_MILITIA_QUESTION
    ]
    assert decision.question_id == "MILITIA"
    assert decision.offered == ("Chapel", "Vassal", "Vassal", "Militia")
    assert (decision.minimum, decision.maximum) == (1, 1)
    assert shadow.hand_count(snapshot.our_seat) == 4

    def first_legal(game: Any, seat: int) -> int:
        del seat
        return next(
            index for index, value in enumerate(game.legal_mask()) if value
        )

    plan = asyncio.run(
        BotDecisionProvider(first_legal).plan(
            frame_index=frame_index,
            game=shadow,
            snapshot=snapshot,
            decision=decision,
        )
    )

    assert len(plan.engine_actions) == 3
    answers = possible_answer_indices(plan.engine_actions, decision)
    assert answers
    assert all(len(answer) == 1 for answer in answers)
    kept = Counter(
        int(action) - int(dz.A_SELECT_BASE) for action in plan.engine_actions
    )
    for answer in answers:
        left = Counter(
            int(dz.def_id(name))
            for index, name in enumerate(decision.offered)
            if index != answer[0]
        )
        assert left == kept
    after = shadow.clone()
    _step_actions(after, plan.engine_actions, frame_index=0, intended=None)
    assert after.hand_count(snapshot.our_seat) == 3


def test_opponent_response_is_stepped_only_on_our_own_turn() -> None:
    """Without the local seat the plan is stepped alone, as on their turn."""
    provider, _ = _replay_throned_militia_game()
    _, shadow, snapshot, _ = provider.seen[THRONED_MILITIA_PLAY_QUESTION]
    opponent = 1 - snapshot.our_seat
    play_militia = (int(dz.A_SELECT_BASE + dz.def_id("Militia")),)
    discard = Discard(
        seat=opponent,
        cards=("Copper",),
        count=2,
        from_zone="hand",
        to_zone="discard",
        from_zone_index=19,
        to_zone_index=22,
    )

    alone = shadow.clone()
    _step_actions(alone, play_militia, frame_index=0, intended=None)
    assert int(alone.current_decision()["player"]) == opponent
    assert int(alone.current_decision()["source"]) == int(dz.def_id("Militia"))

    answered = shadow.clone()
    coppers_before = int(answered.hand(opponent).get(int(dz.def_id("Copper")), 0))
    _step_actions(
        answered,
        play_militia,
        frame_index=0,
        intended=None,
        our_seat=snapshot.our_seat,
        events=(discard,),
    )
    assert int(answered.current_decision()["player"]) == snapshot.our_seat
    assert answered.hand_count(opponent) == 3
    coppers_after = int(answered.hand(opponent).get(int(dz.def_id("Copper")), 0))
    if coppers_before:
        assert coppers_after < coppers_before


def test_local_question_is_never_answered_from_an_opponent_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shadow left on the opponent's choice aborts instead of answering it."""
    monkeypatch.setattr(
        game_module,
        "_resolve_opponent_responses",
        lambda game, our_seat, observed: None,
    )
    provider, results = _replay_throned_militia_game()

    assert SECOND_THRONE_AFTER_MILITIA_QUESTION not in provider.seen
    assert [result.divergence_aborted for result in results] == [True]
    report = results[0].divergence_report
    assert report is not None
    assert report.error_type == "DivergenceError"
    assert report.question_index == SECOND_THRONE_AFTER_MILITIA_QUESTION
    assert "shadow waits on another decision" in report.message


THRONED_CELLAR_ARCHIVE = Path("exports/arena/20261111T221203.193765Z/frames.jsonl")
THRONED_CELLAR_GAME_ID = 185829033
SECOND_CELLAR_QUESTION = 80
ACTION_QUESTION_AFTER_THRONED_CELLAR = 81


def _replay_throned_cellar_game() -> tuple[
    _RecordedReplayWithShadowLog,
    tuple[Any, ...],
]:
    if not THRONED_CELLAR_ARCHIVE.is_file():
        pytest.skip(f"missing live arena fixture: {THRONED_CELLAR_ARCHIVE}")
    events = _events_of_game(
        parse_recording(THRONED_CELLAR_ARCHIVE).events,
        THRONED_CELLAR_GAME_ID,
    )
    provider = _RecordedReplayWithShadowLog(events)
    results = asyncio.run(
        run_game_loop(
            events,
            actuator=MockActuator(replay=True),
            decision_provider=provider,
        )
    )
    return provider, tuple(results)


def test_cards_discarded_before_a_mid_turn_shuffle_return_to_the_shadow_deck() -> None:
    """Game 185829033: Throne Room on Cellar, the first Cellar drew over a shuffle.

    The first Cellar discarded three Coppers and its draw shuffled them back
    into the deck with the rest of the discard pile.  The shadow rebuilt at
    the shuffle kept the three Coppers in its discard pile, the second Cellar
    question arrived mid-chain so the shadow was not rebuilt from the tracker,
    and settling it aborted with "deck counts differ before rigged step: 12 vs
    15".
    """
    provider, results = _replay_throned_cellar_game()

    assert not any(result.divergence_aborted for result in results)
    for question in (SECOND_CELLAR_QUESTION, ACTION_QUESTION_AFTER_THRONED_CELLAR):
        _, shadow, snapshot, _ = provider.seen[question]
        seat = snapshot.seats[snapshot.our_seat]
        assert shadow.deck_count(snapshot.our_seat) == seat.deck_count
        assert shadow.discard_count(snapshot.our_seat) == seat.discard_count
    _, shadow, snapshot, decision = provider.seen[SECOND_CELLAR_QUESTION]
    assert decision.question_id == "CELLAR"
    assert shadow.discard_count(snapshot.our_seat) == 0
    assert int(shadow.current_decision()["source"]) == int(dz.def_id("Cellar"))


def _shadow_hand_names(shadow: Any, snapshot: TrackerSnapshot) -> dict[str, int]:
    seat = snapshot.seats[snapshot.our_seat]
    names = {
        int(dz.def_id(name)): name
        for name, _ in (*seat.hand, *seat.deck, *seat.discard)
    }
    return {
        names[int(def_id)]: int(count)
        for def_id, count in shadow.hand(snapshot.our_seat).items()
        if int(count)
    }


RESHUFFLED_HAND_CARD_ARCHIVE = Path(
    "exports/arena/20261111T231744.478115Z/frames.jsonl"
)
RESHUFFLED_HAND_CARD_GAME_ID = 185844266
SECOND_CELLAR_AFTER_RESHUFFLE_QUESTION = 175


def _replay_reshuffled_hand_card_game() -> tuple[
    _RecordedReplayWithShadowLog,
    tuple[Any, ...],
]:
    if not RESHUFFLED_HAND_CARD_ARCHIVE.is_file():
        pytest.skip(f"missing live arena fixture: {RESHUFFLED_HAND_CARD_ARCHIVE}")
    events = _events_of_game(
        parse_recording(RESHUFFLED_HAND_CARD_ARCHIVE).events,
        RESHUFFLED_HAND_CARD_GAME_ID,
    )
    provider = _RecordedReplayWithShadowLog(events)
    results = asyncio.run(
        run_game_loop(
            events,
            actuator=MockActuator(replay=True),
            decision_provider=provider,
        )
    )
    return provider, tuple(results)


def test_card_discarded_from_hand_and_redrawn_over_a_shuffle_does_not_abort() -> None:
    """Game 185844266: Throne Room on Cellar, the first Cellar redrew its discard.

    The first Cellar discarded Chapel and Moat from hand with one Village left
    in the deck.  Its draw took the Village, shuffled the discard pile (the
    Moat among it) and drew that Moat back.  The shadow used to be rebuilt
    from the last clean question with deck and discard merged, where the Moat
    was still in hand, so the draw was "unavailable" and the game aborted with
    "shuffle reconstruction observed unavailable 'Moat'".  The live shadow now
    takes the step itself with its reshuffle told which cards come out on top.
    """
    provider, results = _replay_reshuffled_hand_card_game()

    assert not any(result.divergence_aborted for result in results)
    _, shadow, snapshot, decision = provider.seen[
        SECOND_CELLAR_AFTER_RESHUFFLE_QUESTION
    ]
    seat = snapshot.seats[snapshot.our_seat]
    assert decision.question_id == "CELLAR"
    assert int(shadow.current_decision()["source"]) == int(dz.def_id("Cellar"))
    assert _shadow_hand_names(shadow, snapshot) == dict(seat.hand)
    assert dict(seat.hand)["Moat"] == 1
    assert shadow.deck_count(snapshot.our_seat) == seat.deck_count == 3
    assert shadow.discard_count(snapshot.our_seat) == seat.discard_count == 0


def test_local_shuffle_draws_separate_the_old_deck_from_the_shuffled_cards() -> None:
    """The draw after a Shuffle event still holds the old deck's last cards."""
    if not RESHUFFLED_HAND_CARD_ARCHIVE.is_file():
        pytest.skip(f"missing live arena fixture: {RESHUFFLED_HAND_CARD_ARCHIVE}")
    provider, _ = _replay_reshuffled_hand_card_game()
    _, _, snapshot, _ = provider.seen[SECOND_CELLAR_AFTER_RESHUFFLE_QUESTION - 1]
    seat = snapshot.our_seat
    assert dict(snapshot.seats[seat].deck) == {"Village": 1}

    shuffle = Shuffle(seat=seat, included_discard=False)
    draw = Draw(
        seat=seat,
        cards=("Moat", "Village"),
        count=2,
        from_zone="deck",
        to_zone="hand",
        from_zone_index=0,
        to_zone_index=0,
    )
    assert game_module._local_shuffle_draws(snapshot, (shuffle, draw)) == ["Moat"]
    with pytest.raises(BridgeError, match="undrawn"):
        game_module._local_shuffle_draws(
            snapshot,
            (shuffle, replace(draw, cards=("Moat",), count=1)),
        )
    with pytest.raises(BridgeError, match="two local shuffles"):
        game_module._local_shuffle_draws(snapshot, (shuffle, draw, shuffle))


def test_shadow_reshuffle_that_cannot_supply_the_observed_draw_is_an_error() -> None:
    with pytest.raises(BridgeError, match="could not supply"):
        game_module._check_local_shuffle(
            {"consumed": 1, "pending": 0, "failed": True},
            1,
        )
    with pytest.raises(BridgeError, match="did not reshuffle"):
        game_module._check_local_shuffle(
            {"consumed": 0, "pending": 1, "failed": False},
            1,
        )
    game_module._check_local_shuffle(
        {"consumed": 1, "pending": 0, "failed": False},
        1,
    )
    # An announced shuffle of an empty discard pile draws nothing new.
    game_module._check_local_shuffle(
        {"consumed": 0, "pending": 1, "failed": False},
        0,
    )


THRONED_LIBRARY_ARCHIVE = Path("exports/arena/20261111T230432.642140Z/frames.jsonl")
THRONED_LIBRARY_GAME_ID = 185829493
FIRST_LIBRARY_QUESTION = 155
SECOND_LIBRARY_QUESTION = 156


def test_throned_library_asks_about_the_first_action_it_draws() -> None:
    """Game 185829493: three Throne Rooms, then Library drew an Artisan.

    This was a native engine bug, not an arena one.  Resuming the Throne Room
    choice ran the interpreter up to Library's keep-or-skip question, and the
    step loop then ran the suspended Library frame a second time.  Library's
    step had no case for that, ended the resolution, and Throne Room's repeat
    started it again: the Artisan was set aside unasked and the shadow asked
    about the next Action instead, one card further down the deck ("deck
    counts differ before rigged step: 12 vs 13").
    """
    if not THRONED_LIBRARY_ARCHIVE.is_file():
        pytest.skip(f"missing live arena fixture: {THRONED_LIBRARY_ARCHIVE}")
    events = _events_of_game(
        parse_recording(THRONED_LIBRARY_ARCHIVE).events,
        THRONED_LIBRARY_GAME_ID,
    )
    provider = _RecordedReplayWithShadowLog(events)

    results = asyncio.run(
        run_game_loop(
            events,
            actuator=MockActuator(replay=True),
            decision_provider=provider,
        )
    )

    assert not any(result.divergence_aborted for result in results)
    _, shadow, snapshot, decision = provider.seen[FIRST_LIBRARY_QUESTION]
    seat = snapshot.seats[snapshot.our_seat]
    assert decision.question_id == "LIBRARY"
    assert shadow.deck_count(snapshot.our_seat) == seat.deck_count
    assert shadow.set_aside(snapshot.our_seat) == [int(dz.def_id("Artisan"))]
    assert shadow.decision_context()["subject_defs"] == [int(dz.def_id("Artisan"))]
    _, shadow, snapshot, _ = provider.seen[SECOND_LIBRARY_QUESTION]
    seat = snapshot.seats[snapshot.our_seat]
    assert shadow.deck_count(snapshot.our_seat) == seat.deck_count
    assert shadow.set_aside(snapshot.our_seat) == [int(dz.def_id("Throne Room"))]
    assert shadow.hand(snapshot.our_seat)[int(dz.def_id("Artisan"))] == 1
