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
    resolve_card_stack_index,
    resolve_submit_button_index,
    selected_card_counts,
)
from src.v2.arena.fsm.game import (
    BotDecisionProvider,
    DecisionPlan,
    RecordedDecisionProvider,
    run_game_loop,
)
from src.v2.arena.protocol.events import (
    DecisionResolved,
    GameStart,
    PendingDecision,
)
from src.v2.arena.protocol.recording import parse_recording
from src.v2.arena.shadow.bridge import game_from_snapshot
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
    assert [gesture.question_index for gesture in actuator.gestures] == [3]
    assert actuator.gestures[0].answer_indices == (1, 1, 0, 0)


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

    assert gesture.answer_indices == (2, 0, 1)
    assert gesture.click_button
    assert [
        (target.region, target.identity, target.offered_index)
        for target in targets
    ] == [
        ("hand", "Gold", 2),
        ("hand", "Copper", 0),
        ("hand", "Copper", 1),
        ("submit-button", "MILITIA", -1),
    ]

    three_coppers = replace(
        militia,
        offered=("Copper", "Copper", "Copper"),
        minimum=3,
        maximum=3,
    )
    copper_gesture = asyncio.run(
        MockActuator(replay=True).act(
            copper,
            three_coppers,
            three_coppers.offered,
            prior_actions=(copper, copper),
        )
    )
    assert [
        (target.region, target.identity)
        for target in gesture_click_targets(
            copper_gesture,
            three_coppers,
            three_coppers.offered,
        )
    ] == [
        ("hand", "Copper"),
        ("hand", "Copper"),
        ("hand", "Copper"),
        ("submit-button", "MILITIA"),
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
    ] == [("submit-button", "GAME_MAY_REACT_WITH")]


def test_live_game_2_replays_through_militia_question_55() -> None:
    events, _, question, _ = _live_militia()
    resolved = DecisionResolved(
        timestamp_ms=question.timestamp_ms + 1,
        question_index=question.question_index,
        answers=(2, 0, 1),
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
    assert actuator.gestures[-1].answer_indices == (2, 0, 1)
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


def test_actuation_error_reports_offered_targets_and_failed_step() -> None:
    _, _, _, decision = _live_militia()
    gold = int(dz.A_SELECT_BASE + dz.def_id("Gold"))
    copper = int(dz.A_SELECT_BASE + dz.def_id("Copper"))

    with pytest.raises(ActuationError) as caught:
        asyncio.run(
            PlaywrightActuator(_EmptyPage()).act(
                copper,
                decision,
                decision.offered,
                prior_actions=(gold, copper),
            )
        )

    message = str(caught.value)
    assert "gesture (2, 0, 1)" in message
    assert f"offered={decision.offered!r}" in message
    assert "resolved_targets=" in message
    assert "step=1/4" in message
    assert "target=DOMClickTarget(region='hand', identity='Gold'" in message
    assert "status=not-found" in message
    assert "found=0" in message
