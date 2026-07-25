from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dominion_v2_py as dz
import pytest

from src.v2.arena.actuate.clicks import (
    ActuationError,
    MockActuator,
    PlaywrightActuator,
    gesture_click_targets,
)
from src.v2.arena.protocol.events import PendingDecision
from src.v2.arena.protocol.recording import parse_recording
from src.v2.arena.shadow.tracker import PendingDecisionSnapshot, Tracker


FIRST_ABORT = Path(
    "exports/arena/20260724T204829.953926Z/frames.jsonl"
)
SECOND_ABORT = Path(
    "exports/arena/20260724T235933.669422Z/frames.jsonl"
)


@dataclass
class _CardNode:
    node_id: int
    identity: str
    selected_count: int
    clickable: bool


class _CardLocator:
    def __init__(self, page: _RerenderingPage, node: _CardNode) -> None:
        self.page = page
        self.node = node

    async def is_visible(self) -> bool:
        return True

    async def evaluate(self, _script: str) -> dict[str, object]:
        return {
            "name": self.node.identity,
            "width": 112.0,
            "height": 176.0,
            "zIndex": 2000,
            "hasVisibleCounter": True,
            "selectedCount": self.node.selected_count,
            "clickable": self.node.clickable,
            "hasVisibleAll": False,
            "allCoversCenter": False,
        }

    async def click(self, *, position: dict[str, float]) -> None:
        assert position == {"x": 56.0, "y": 88.0}
        self.page.click_card(self.node)


class _CardLocators:
    def __init__(self, page: _RerenderingPage) -> None:
        self.page = page

    async def count(self) -> int:
        return len(self.page.nodes)

    def nth(self, index: int) -> _CardLocator:
        return _CardLocator(self.page, self.page.nodes[index])

    async def evaluate_all(self, _script: str) -> list[list[object]]:
        return [
            [identity, count]
            for identity, count in self.page.selected.items()
            if count
        ]


class _ButtonLocator:
    def __init__(
        self,
        page: _RerenderingPage,
        index: int,
        box: dict[str, float],
    ) -> None:
        self.page = page
        self.index = index
        self.box = box

    async def is_visible(self) -> bool:
        return True

    async def is_enabled(self) -> bool:
        return self.page.button_enabled

    async def evaluate(self, _script: str) -> bool:
        return self.page.button_enabled

    async def bounding_box(self) -> dict[str, float]:
        return self.box

    async def click(self) -> None:
        self.page.clicks.append(("button", "confirm", self.index))


class _ButtonLocators:
    def __init__(self, page: _RerenderingPage) -> None:
        self.page = page

    async def count(self) -> int:
        return len(self.page.button_boxes)

    def nth(self, index: int) -> _ButtonLocator:
        return _ButtonLocator(
            self.page,
            index,
            self.page.button_boxes[index],
        )


class _RerenderingPage:
    def __init__(
        self,
        *,
        click_has_effect: bool = True,
        replace_nodes: bool = False,
    ) -> None:
        self.click_has_effect = click_has_effect
        self.replace_nodes = replace_nodes
        self.remaining = {"Estate": 1, "Copper": 3, "Silver": 1}
        self.selected: dict[str, int] = {}
        self.cooldowns: dict[str, int] = {}
        self.next_node_id = 1
        self.nodes: list[_CardNode] = []
        self.clicks: list[tuple[str, str, int]] = []
        self.polls = 0
        self.button_enabled = True
        self.button_cooldown = 0
        self.button_boxes = [
            {"x": 10.0, "width": 200.0, "height": 40.0},
            {"x": 220.0, "width": 40.0, "height": 40.0},
        ]
        self._render()

    def locator(self, selector: str) -> Any:
        if selector == "div.card-stacks > div":
            return _CardLocators(self)
        if selector == "div.game-buttons canvas":
            return _ButtonLocators(self)
        raise AssertionError(selector)

    async def wait_for_timeout(self, _milliseconds: float) -> None:
        self.polls += 1
        for identity in tuple(self.cooldowns):
            self.cooldowns[identity] -= 1
            if self.cooldowns[identity] <= 0:
                del self.cooldowns[identity]
        if self.button_cooldown:
            self.button_cooldown -= 1
            if not self.button_cooldown:
                self.button_enabled = True
        self._render()

    def click_card(self, node: _CardNode) -> None:
        self.clicks.append(("card", node.identity, node.node_id))
        if not self.click_has_effect:
            return
        assert node.selected_count == 0
        assert node.clickable
        self.remaining[node.identity] -= 1
        self.selected[node.identity] = self.selected.get(node.identity, 0) + 1
        self.cooldowns[node.identity] = 1
        if sum(self.selected.values()) == 3:
            self.button_enabled = False
            self.button_cooldown = 1
        self._render()

    def _render(self) -> None:
        if not self.replace_nodes and self.nodes:
            by_key = {
                (node.identity, bool(node.selected_count)): node
                for node in self.nodes
            }
        else:
            by_key = {}
        nodes: list[_CardNode] = []
        for identity in ("Estate", "Copper", "Silver"):
            if self.remaining[identity]:
                nodes.append(
                    self._node(
                        by_key,
                        identity,
                        selected_count=0,
                        clickable=identity not in self.cooldowns,
                    )
                )
            selected_count = self.selected.get(identity, 0)
            if selected_count:
                nodes.append(
                    self._node(
                        by_key,
                        identity,
                        selected_count=selected_count,
                        clickable=False,
                    )
                )
        self.nodes = nodes

    def _node(
        self,
        by_key: dict[tuple[str, bool], _CardNode],
        identity: str,
        *,
        selected_count: int,
        clickable: bool,
    ) -> _CardNode:
        key = (identity, bool(selected_count))
        node = by_key.get(key)
        if node is None:
            node = _CardNode(
                node_id=self.next_node_id,
                identity=identity,
                selected_count=selected_count,
                clickable=clickable,
            )
            self.next_node_id += 1
        else:
            node.selected_count = selected_count
            node.clickable = clickable
        return node


def _militia_decision() -> PendingDecisionSnapshot:
    return PendingDecisionSnapshot(
        question_index=42,
        decision_type="DISCARD",
        question_id="MILITIA",
        offered=("Copper", "Copper", "Silver", "Copper", "Estate"),
        minimum=2,
        maximum=2,
        association="Militia",
    )


def _run_militia(page: _RerenderingPage) -> None:
    copper = int(dz.A_SELECT_BASE + dz.def_id("Copper"))
    estate = int(dz.A_SELECT_BASE + dz.def_id("Estate"))
    decision = _militia_decision()
    asyncio.run(
        PlaywrightActuator(
            page,
            actuation_timeout_seconds=0.01,
            poll_interval_seconds=0.001,
        ).act(
            copper,
            decision,
            decision.offered,
            prior_actions=(estate, copper),
        )
    )


def test_multiselect_waits_for_rerendered_duplicate_then_confirms() -> None:
    page = _RerenderingPage()

    _run_militia(page)

    assert [(kind, identity) for kind, identity, _ in page.clicks] == [
        ("card", "Estate"),
        ("card", "Copper"),
        ("card", "Copper"),
        ("button", "confirm"),
    ]
    assert page.selected == {"Estate": 1, "Copper": 2}
    assert page.polls >= 1


def test_multiselect_retries_one_no_effect_click_then_fails_with_step() -> None:
    page = _RerenderingPage(click_has_effect=False)

    with pytest.raises(ActuationError) as caught:
        _run_militia(page)

    message = str(caught.value)
    assert "step=1/4" in message
    assert "target=DOMClickTarget(region='hand', identity='Estate'" in message
    assert "status=no-effect" in message
    assert "after one retry" in message
    assert [(kind, identity) for kind, identity, _ in page.clicks] == [
        ("card", "Estate"),
        ("card", "Estate"),
    ]


def test_multiselect_resolves_replaced_stack_node_just_in_time() -> None:
    page = _RerenderingPage(replace_nodes=True)

    _run_militia(page)

    copper_node_ids = [
        node_id
        for kind, identity, node_id in page.clicks
        if kind == "card" and identity == "Copper"
    ]
    assert len(copper_node_ids) == 2
    assert copper_node_ids[0] != copper_node_ids[1]
    assert page.clicks[-1][:2] == ("button", "confirm")


def _archived_decision(
    path: Path,
    question_index: int,
) -> PendingDecisionSnapshot:
    if not path.is_file():
        pytest.skip(f"missing archived abort: {path}")
    tracker = Tracker()
    for event in parse_recording(path).events:
        tracker.consume(event)
        if (
            isinstance(event, PendingDecision)
            and event.question_index == question_index
        ):
            decision = tracker.snapshot().pending_decision
            assert decision is not None
            return decision
    raise AssertionError(f"{path} has no question {question_index}")


@pytest.mark.parametrize(
    ("path", "question_index", "labels", "answers"),
    [
        (
            FIRST_ABORT,
            55,
            ("Gold", "Copper", "Copper"),
            (2, 0, 1),
        ),
        (
            SECOND_ABORT,
            42,
            ("Estate", "Copper", "Copper"),
            (4, 0, 1),
        ),
    ],
)
def test_archived_militia_aborts_keep_the_correct_four_step_plan(
    path: Path,
    question_index: int,
    labels: tuple[str, str, str],
    answers: tuple[int, int, int],
) -> None:
    decision = _archived_decision(path, question_index)
    actions = tuple(
        int(dz.A_SELECT_BASE + dz.def_id(label))
        for label in labels
    )
    gesture = asyncio.run(
        MockActuator(replay=True).act(
            actions[-1],
            decision,
            decision.offered,
            prior_actions=actions[:-1],
        )
    )
    targets = gesture_click_targets(gesture, decision, decision.offered)

    assert gesture.answer_indices == answers
    assert [
        (target.region, target.identity)
        for target in targets
    ] == [
        ("hand", labels[0]),
        ("hand", labels[1]),
        ("hand", labels[2]),
        ("submit-button", "MILITIA"),
    ]
