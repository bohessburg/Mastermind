from __future__ import annotations

import asyncio
import re
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import dominion_v2_py as dz
import pytest

from src.v2.arena.actuate.clicks import (
    MockActuator,
    card_stack_region,
    dom_click_target,
    game_button_role,
    map_engine_actions,
)
from src.v2.arena.fsm.game import (
    BotDecisionProvider,
    DecisionPlan,
    run_game_loop,
)
from src.v2.arena.protocol.events import PendingDecision
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
            }
            self.cards.append(self.current_card)
        elif self.current_card is not None and "name-layer" in classes:
            if "invisible" not in classes:
                self.current_card["reading_name"] = True
        elif self.current_card is not None and "counter-layer" in classes:
            if "invisible" not in classes:
                self.current_card["has_visible_counter"] = True

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

    def handle_endtag(self, tag: str) -> None:
        if not self.stack:
            return
        depth = len(self.stack) - 1
        _, classes = self.stack.pop()
        if self.current_card is not None:
            if "name-layer" in classes:
                self.current_card.pop("reading_name", None)
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
    assert button_roles == {"autoplay", "submit"}

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
