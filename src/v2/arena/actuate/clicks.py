"""Map native engine actions onto dominion.games client gestures.

The site encodes answers as positions in the current question's offered
elements.  The engine instead uses semantic action regions (play/buy/select/
option/pass).  The pure mapping functions in this module are the single source
of truth for both the browser actuator and offline replay.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import permutations, product
from typing import Any, Iterable

import dominion_v2_py as dz

from ..shadow.tracker import PendingDecisionSnapshot


class ActionMappingError(ValueError):
    """An engine action cannot be represented by the current client question."""


class ActuationError(RuntimeError):
    """A concrete client gesture could not be completed."""


@dataclass(frozen=True, kw_only=True)
class AnswerMapping:
    """One complete ANSWER_QUESTION payload shape and its UI implications."""

    answers: tuple[int, ...]
    selected_indices: tuple[int, ...]
    click_button: bool
    ambiguous: bool


@dataclass(frozen=True, kw_only=True)
class ClientGesture:
    """The concrete gesture intended for one client question."""

    question_index: int
    action: int
    prior_actions: tuple[int, ...]
    answer_indices: tuple[int, ...]
    selected_indices: tuple[int, ...]
    click_button: bool
    labels: tuple[str, ...]

    @property
    def actions(self) -> tuple[int, ...]:
        return (*self.prior_actions, self.action)


def offered_name(value: str) -> str:
    """Strip complex-question path prefixes from an offered element."""
    return value.rsplit(":", 1)[-1]


def possible_answer_indices(
    actions: Iterable[int],
    decision: PendingDecisionSnapshot,
    offered_elements: tuple[str, ...] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return every client answer encoding consistent with engine actions.

    Multiple encodings arise for duplicate physical copies, multiple legal
    card orders, and manual treasure clicks equivalent to autoplay. The engine
    deliberately addresses cards by definition, because physical copies are
    game-equivalent.
    """
    actions_tuple = tuple(int(action) for action in actions)
    offered = decision.offered if offered_elements is None else offered_elements
    question = decision.question_id

    if question == "GAME_ACTION_PHASE":
        if actions_tuple == (int(dz.A_PASS),):
            return ((0, 0, 0),)
        _require_length(actions_tuple, 1, decision)
        indices = _card_assignments(actions_tuple, offered, region="play")
        return _unique((0, 1, index_tuple[0], 0) for index_tuple in indices)

    if question == "GAME_BUY_PHASE":
        if actions_tuple == (int(dz.A_PASS),):
            return ((0,),)
        if actions_tuple and all(
            _action_region(action) == "play" for action in actions_tuple
        ):
            offered_treasures = tuple(
                value for value in offered if value.startswith("1:0:")
            )
            assignments = _card_assignments(
                actions_tuple,
                offered_treasures,
                region="play",
            )
            if not assignments:
                raise ActionMappingError(
                    "autoplay actions do not match the offered treasures"
                )
            encodings: list[tuple[int, ...]] = []
            if (
                len(actions_tuple) == len(offered_treasures)
                and any(
                    value.endswith("AUTOPLAY_TREASURES") for value in offered
                )
            ):
                encodings.append((2, 0))
            if len(actions_tuple) == 1:
                encodings.extend(
                    (1, 1, assignment[0], 0)
                    for assignment in assignments
                )
            if encodings:
                return _unique(encodings)
            raise ActionMappingError(
                "client can only play one treasure or use autoplay"
            )
        _require_length(actions_tuple, 1, decision)
        supply = tuple(value for value in offered if value.startswith("0:"))
        indices = _card_assignments(actions_tuple, supply, region="buy")
        return _unique((0, index_tuple[0]) for index_tuple in indices)

    if question == "THRONE_ROOM":
        if actions_tuple == (int(dz.A_PASS),):
            return ((),)
        _require_length(actions_tuple, 1, decision)
        indices = _card_assignments(actions_tuple, offered, region="select")
        return _unique((1, index_tuple[0], 0) for index_tuple in indices)

    if question in {"SENTRY_TRASH", "SENTRY_DISCARD"} and all(
        _action_region(action) == "option" for action in actions_tuple
    ):
        if len(actions_tuple) != len(offered):
            raise ActionMappingError(
                f"{question} classification needs one option per offered card"
            )
        wanted = 0 if question == "SENTRY_TRASH" else 1
        answers = tuple(
            index
            for index, action in enumerate(actions_tuple)
            if action - int(dz.A_OPTION_BASE) == wanted
        )
        return _unique(permutations(answers))

    if question == "SENTRY_TOPDECK":
        _require_length(actions_tuple, 1, decision)
        option = _option_index(actions_tuple[0])
        if len(offered) != 2 or option not in (0, 1):
            raise ActionMappingError("Sentry order needs two cards and option 0/1")
        return ((1, 0),) if option == 0 else ((0, 1),)

    if decision.decision_type == "ORDER_CARDS":
        answers = _order_answers(actions_tuple, len(offered))
        return (answers,)

    if decision.decision_type == "CHOOSE_MODE":
        if question == "VASSAL":
            answers = tuple(1 - _option_index(action) for action in actions_tuple)
        else:
            answers = tuple(_option_index(action) for action in actions_tuple)
        return (answers,)

    selected = tuple(
        action for action in actions_tuple if action != int(dz.A_PASS)
    )
    if not selected:
        return ((),)
    indices = _card_assignments(selected, offered, region="select")
    return _unique(indices)


def map_engine_actions(
    actions: Iterable[int],
    decision: PendingDecisionSnapshot,
    offered_elements: tuple[str, ...] | None = None,
) -> AnswerMapping:
    """Choose a deterministic client encoding for a complete engine plan."""
    offered = decision.offered if offered_elements is None else offered_elements
    possibilities = possible_answer_indices(actions, decision, offered)
    if not possibilities:
        raise ActionMappingError(
            f"question {decision.question_index} has no answer mapping"
        )
    answers = possibilities[0]
    selected = _selected_indices(decision, answers)
    click_button = _needs_button(decision, selected)
    return AnswerMapping(
        answers=answers,
        selected_indices=selected,
        click_button=click_button,
        ambiguous=len(possibilities) > 1,
    )


def map_engine_action(
    action: int,
    decision: PendingDecisionSnapshot,
    offered_elements: tuple[str, ...] | None = None,
    *,
    prior_actions: Iterable[int] = (),
) -> AnswerMapping:
    """Map one engine action plus already accumulated actions to an answer."""
    return map_engine_actions(
        (*tuple(prior_actions), int(action)),
        decision,
        offered_elements,
    )


class Actuator(ABC):
    """One-action-at-a-time client actuation boundary."""

    @abstractmethod
    async def act(
        self,
        action: int,
        decision: PendingDecisionSnapshot,
        offered_elements: tuple[str, ...],
        *,
        prior_actions: tuple[int, ...] = (),
    ) -> ClientGesture:
        """Perform the gesture represented by the accumulated action plan."""


class MockActuator(Actuator):
    """Offline actuator that records exactly what would have been clicked."""

    def __init__(self, *, replay: bool = False) -> None:
        self.replay = replay
        self.gestures: list[ClientGesture] = []
        self.stopped = False

    async def act(
        self,
        action: int,
        decision: PendingDecisionSnapshot,
        offered_elements: tuple[str, ...],
        *,
        prior_actions: tuple[int, ...] = (),
    ) -> ClientGesture:
        if self.stopped:
            raise ActuationError("mock actuator was stopped after divergence")
        gesture = _gesture(
            action,
            decision,
            offered_elements,
            prior_actions=prior_actions,
        )
        self.gestures.append(gesture)
        return gesture

    def stop(self) -> None:
        self.stopped = True


class PlaywrightActuator(Actuator):
    """Click the canvas/card-stack DOM recorded from client bundle 2.2.8.

    Mappings verified from the saved DOM snapshots:

    * Offered cards are the visible, pointer-enabled direct children of
      ``div.card-stacks``.  Their nested ``.name-layer`` text is the display
      name used by the protocol parser.
    * Choice/decline/done controls are canvas elements under
      ``div.game-buttons``.  The client renders only the currently actionable
      canvases with ``display != none``.
    * Card-mode choices use the same visible ``game-buttons`` canvas row.

    The client is canvas-heavy and exposes no stable test ids.  These selectors
    intentionally stay behind this interface so a client update changes one
    module rather than the game loop.
    """

    def __init__(self, page: Any | None) -> None:
        self.page = page

    async def act(
        self,
        action: int,
        decision: PendingDecisionSnapshot,
        offered_elements: tuple[str, ...],
        *,
        prior_actions: tuple[int, ...] = (),
    ) -> ClientGesture:
        if self.page is None:
            raise ActuationError("Playwright page is unavailable")
        gesture = _gesture(
            action,
            decision,
            offered_elements,
            prior_actions=prior_actions,
        )
        try:
            if (
                decision.question_id == "GAME_BUY_PHASE"
                and gesture.answer_indices == (2, 0)
            ):
                await self._visible_buttons().first.click()
            elif decision.decision_type == "CHOOSE_MODE":
                for index in gesture.selected_indices:
                    await self._visible_buttons().nth(index).click()
            else:
                used: dict[str, int] = {}
                for index in gesture.selected_indices:
                    name = offered_name(offered_elements[index])
                    occurrence = used.get(name, 0)
                    used[name] = occurrence + 1
                    await self._click_card(name, occurrence)
            if gesture.click_button:
                await self._visible_buttons().first.click()
        except Exception as error:
            raise ActuationError(
                f"failed question {decision.question_index} gesture "
                f"{gesture.answer_indices}"
            ) from error
        return gesture

    def _visible_buttons(self) -> Any:
        return self.page.locator(
            "div.game-buttons canvas:not([style*='display: none'])"
        )

    async def _click_card(self, name: str, occurrence: int) -> None:
        cards = self.page.locator("div.card-stacks > div").filter(has_text=name)
        visible: list[Any] = []
        for index in range(await cards.count()):
            candidate = cards.nth(index)
            if await candidate.is_visible():
                visible.append(candidate)
        if occurrence >= len(visible):
            raise ActuationError(
                f"no visible clickable occurrence {occurrence} of {name!r}"
            )
        await visible[occurrence].click()


def _gesture(
    action: int,
    decision: PendingDecisionSnapshot,
    offered: tuple[str, ...],
    *,
    prior_actions: tuple[int, ...],
) -> ClientGesture:
    mapping = map_engine_action(
        action,
        decision,
        offered,
        prior_actions=prior_actions,
    )
    return ClientGesture(
        question_index=decision.question_index,
        action=int(action),
        prior_actions=prior_actions,
        answer_indices=mapping.answers,
        selected_indices=mapping.selected_indices,
        click_button=mapping.click_button,
        labels=tuple(
            offered_name(offered[index]) for index in mapping.selected_indices
        ),
    )


def _action_region(action: int) -> str:
    if action == int(dz.A_PASS):
        return "pass"
    if int(dz.A_PLAY_BASE) <= action < int(dz.A_PLAY_BASE + dz.ACTION_DEF_COUNT):
        return "play"
    if int(dz.A_BUY_BASE) <= action < int(dz.A_BUY_BASE + dz.ACTION_DEF_COUNT):
        return "buy"
    if int(dz.A_SELECT_BASE) <= action < int(
        dz.A_SELECT_BASE + dz.ACTION_DEF_COUNT
    ):
        return "select"
    if int(dz.A_OPTION_BASE) <= action < int(dz.A_CALL_BASE):
        return "option"
    raise ActionMappingError(f"unsupported engine action id {action}")


def _action_def(action: int, region: str) -> int:
    actual = _action_region(action)
    if actual != region:
        raise ActionMappingError(
            f"expected {region} action, got {actual} action {action}"
        )
    base = {
        "play": int(dz.A_PLAY_BASE),
        "buy": int(dz.A_BUY_BASE),
        "select": int(dz.A_SELECT_BASE),
    }[region]
    return action - base


def _option_index(action: int) -> int:
    if _action_region(action) != "option":
        raise ActionMappingError(f"expected option action, got {action}")
    return action - int(dz.A_OPTION_BASE)


def _card_assignments(
    actions: tuple[int, ...],
    offered: tuple[str, ...],
    *,
    region: str,
) -> tuple[tuple[int, ...], ...]:
    choices: list[tuple[int, ...]] = []
    for action in actions:
        wanted = _action_def(action, region)
        matching = tuple(
            index
            for index, value in enumerate(offered)
            if _offered_def(value) == wanted
        )
        if not matching:
            raise ActionMappingError(
                f"{region} action def {wanted} is absent from offered elements"
            )
        choices.append(matching)

    assignments = tuple(
        candidate
        for candidate in product(*choices)
        if len(set(candidate)) == len(candidate)
    )
    if not assignments:
        raise ActionMappingError("card actions reuse an unavailable offered copy")
    return _unique(assignments)


def _offered_def(value: str) -> int | None:
    name = offered_name(value)
    if name.startswith(
        ("card-mode-", "zone-", "game-button-", "AUTOPLAY_", "hidden-card-")
    ):
        return None
    try:
        return int(dz.def_id(name))
    except (TypeError, ValueError):
        return None


def _order_answers(actions: tuple[int, ...], count: int) -> tuple[int, ...]:
    remaining = list(range(count))
    answers: list[int] = []
    for action in actions:
        relative = _option_index(action)
        if not 0 <= relative < len(remaining):
            raise ActionMappingError(
                f"order option {relative} is invalid with {len(remaining)} remaining"
            )
        answers.append(remaining.pop(relative))
    return tuple(answers)


def _selected_indices(
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


def _needs_button(
    decision: PendingDecisionSnapshot,
    selected: tuple[int, ...],
) -> bool:
    if decision.question_id in {
        "GAME_ACTION_PHASE",
        "GAME_BUY_PHASE",
        "THRONE_ROOM",
    }:
        return not selected
    if decision.decision_type in {"CHOOSE_MODE", "ORDER_CARDS"}:
        return False
    required_to_auto_submit = min(decision.maximum, len(decision.offered))
    return len(selected) < required_to_auto_submit


def _require_length(
    actions: tuple[int, ...],
    expected: int,
    decision: PendingDecisionSnapshot,
) -> None:
    if len(actions) != expected:
        raise ActionMappingError(
            f"{decision.question_id} expects {expected} engine action, "
            f"got {len(actions)}"
        )


def _unique(values: Iterable[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    return tuple(dict.fromkeys(values))
