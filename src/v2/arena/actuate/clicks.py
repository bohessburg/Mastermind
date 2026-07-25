"""Map native engine actions onto dominion.games client gestures.

The site encodes answers as positions in the current question's offered
elements.  The engine instead uses semantic action regions (play/buy/select/
option/pass).  The pure mapping functions in this module are the single source
of truth for both the browser actuator and offline replay.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from itertools import permutations, product
from typing import Any, Awaitable, Callable, Iterable, Sequence

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


@dataclass(frozen=True, kw_only=True)
class DOMClickTarget:
    """Region-qualified physical target for one offered protocol element."""

    region: str
    identity: str
    offered_index: int


@dataclass(frozen=True, kw_only=True)
class DOMCardStack:
    """Recorded facts for one visible direct child of ``div.card-stacks``."""

    identity: str
    width: float
    height: float
    z_index: int
    has_visible_counter: bool
    selected_count: int
    clickable: bool
    has_visible_all: bool
    all_covers_center: bool


@dataclass(frozen=True, kw_only=True)
class DOMGameButton:
    """Recorded geometry for one visible ``div.game-buttons`` canvas."""

    x: float
    width: float
    height: float
    enabled: bool = True


def offered_name(value: str) -> str:
    """Strip complex-question path prefixes from an offered element."""
    return value.rsplit(":", 1)[-1]


_EXPLICIT_MULTI_ANSWER_TYPES = frozenset(
    {
        "COMPLEX_AND",
        "COMPLEX_OR",
        "ORDER_CARDS",
    }
)


def expected_answer_count(
    decision: PendingDecisionSnapshot,
) -> int | None:
    """Return the fixed client answer arity for simple questions.

    Complex client prompts encode commands and selections in one answer tuple,
    so their protocol arity is not their card-selection minimum/maximum.
    """
    if decision.decision_type == "CHOOSE_MODE":
        return 1
    if (
        decision.minimum == decision.maximum == 1
        and decision.decision_type not in _EXPLICIT_MULTI_ANSWER_TYPES
    ):
        return 1
    return None


def is_start_confirmation_prompt(
    decision: PendingDecisionSnapshot,
) -> bool:
    """Recognize the pre-turn client handshake by its protocol shape.

    The observed localization key is ``question-411``, but it is deliberately
    not required: that identifier may move between client sessions/releases.
    """
    return (
        decision.decision_type == "CHOOSE_MODE"
        and len(decision.offered) == 1
        and decision.offered[0].startswith("card-mode-")
        and decision.minimum == decision.maximum == 1
    )


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

    if question == "GAME_MAY_REACT_WITH":
        _require_length(actions_tuple, 1, decision)
        if actions_tuple == (int(dz.A_PASS),):
            # The client bundle flattens the three-part COMPLEX_AND as
            # [selection length][selection][reject length][reject][way length].
            return ((0, 1, 0, 0),)
        indices = _card_assignments(actions_tuple, offered, region="select")
        return _unique(
            (1, assignment[0], 1, 0, 0)
            for assignment in indices
        )

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
    expected = expected_answer_count(decision)
    if expected is not None and len(answers) != expected:
        raise ActionMappingError(
            f"{decision.question_id} expects exactly {expected} answer "
            f"index, got {len(answers)}"
        )
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

    * ``0:<card>`` buy elements map to landscape supply piles under
      ``div.card-stacks``. They have a visible pile counter.
    * ``1:0:<card>`` play elements map to portrait local-hand stacks under
      ``div.card-stacks``. Client 2.2.8 gives those stacks z-indices 2000–2999.
    * ``2:AUTOPLAY_TREASURES`` maps to the wide (at least 3.5:1)
      ``div.game-buttons`` canvas. The separate rightmost canvas submits a
      decline/end-phase answer.
    * Other hand/supply questions use the same physical-region rules;
      revealed-card choices use the high-z-index display-card region.

    The client is canvas-heavy and exposes no stable test ids.  These selectors
    intentionally stay behind this interface so a client update changes one
    module rather than the game loop.
    """

    def __init__(
        self,
        page: Any | None,
        *,
        snapshot_dom: Callable[[str], Awaitable[Any]] | None = None,
        actuation_timeout_seconds: float = 3.0,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        if actuation_timeout_seconds < 0:
            raise ValueError("actuation timeout cannot be negative")
        if poll_interval_seconds <= 0:
            raise ValueError("actuation poll interval must be positive")
        self.page = page
        self.snapshot_dom = snapshot_dom
        self.actuation_timeout_seconds = actuation_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds

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
        try:
            gesture = _gesture(
                action,
                decision,
                offered_elements,
                prior_actions=prior_actions,
            )
            targets = gesture_click_targets(
                gesture,
                decision,
                offered_elements,
            )
        except Exception as error:
            raise ActuationError(
                f"failed question {decision.question_index} gesture planning; "
                f"offered={offered_elements!r}; step=plan status=not-found; "
                f"detail={error}"
            ) from error

        planned_selected: Counter[str] = Counter()
        verify_card_effects = _verify_selection_before_submit(decision)
        for step, target in enumerate(targets, start=1):
            try:
                if target.region == "autoplay-button":
                    await self._wait_for_actionable_click(
                        self._click_autoplay_button
                    )
                elif target.region == "mode-button":
                    await self._wait_for_actionable_click(
                        lambda: self._click_mode_button(
                            target.offered_index,
                            len(offered_elements),
                            start_confirmation=is_start_confirmation_prompt(
                                decision
                            ),
                        )
                    )
                elif target.region == "submit-button":
                    if _verify_selection_before_submit(decision):
                        await self._verify_selected_cards(gesture.labels)
                    await self._wait_for_actionable_click(
                        lambda: self._click_submit_button(decision)
                    )
                elif target.region == "decline-button":
                    await self._wait_for_actionable_click(
                        self._click_reaction_decline
                    )
                else:
                    if verify_card_effects:
                        await self._click_card_with_verification(
                            target,
                            expected_selected=planned_selected[
                                target.identity
                            ],
                        )
                        planned_selected[target.identity] += 1
                    else:
                        await self._click_card(target)
            except _GestureStepError as error:
                start_confirmation_context = (
                    await self._start_confirmation_failure_context(decision)
                    if is_start_confirmation_prompt(decision)
                    else ""
                )
                raise ActuationError(
                    f"failed question {decision.question_index} gesture "
                    f"{gesture.answer_indices}; offered={offered_elements!r}; "
                    f"resolved_targets={targets!r}; step={step}/{len(targets)} "
                    f"target={target!r} status={error.status}; "
                    f"detail={error.detail}{start_confirmation_context}"
                ) from error
            except Exception as error:
                start_confirmation_context = (
                    await self._start_confirmation_failure_context(decision)
                    if is_start_confirmation_prompt(decision)
                    else ""
                )
                raise ActuationError(
                    f"failed question {decision.question_index} gesture "
                    f"{gesture.answer_indices}; offered={offered_elements!r}; "
                    f"resolved_targets={targets!r}; step={step}/{len(targets)} "
                    f"target={target!r} status=click-error; detail={error}"
                    f"{start_confirmation_context}"
                ) from error
        return gesture

    async def _click_card(self, target: DOMClickTarget) -> None:
        candidate, stack = await self._wait_for_card_target(
            target,
            expected_selected=None,
        )
        await self._click_resolved_card(candidate, stack)

    async def _click_card_with_verification(
        self,
        target: DOMClickTarget,
        *,
        expected_selected: int,
    ) -> None:
        expected_after = expected_selected + 1
        last_error: _GestureStepError | None = None
        for attempt in range(2):
            candidate, stack = await self._wait_for_card_target(
                target,
                expected_selected=expected_selected,
            )
            await self._click_resolved_card(candidate, stack)
            try:
                await self._wait_for_selection_count(
                    target.identity,
                    expected_after,
                )
                return
            except _GestureStepError as error:
                if error.status != "no-effect":
                    raise
                last_error = error
                if attempt == 0:
                    continue
        assert last_error is not None
        raise _GestureStepError(
            "no-effect",
            "card click had no observable selection effect after one retry; "
            f"{last_error.detail}",
        )

    async def _wait_for_card_target(
        self,
        target: DOMClickTarget,
        *,
        expected_selected: int | None,
    ) -> tuple[Any, DOMCardStack]:
        deadline = self._deadline()
        last_error: _GestureStepError | None = None
        while True:
            try:
                return await self._resolve_card_target(
                    target,
                    expected_selected=expected_selected,
                )
            except _GestureStepError as error:
                if error.status != "not-found":
                    raise
                last_error = error
            if not await self._wait_for_next_poll(deadline):
                assert last_error is not None
                raise last_error

    async def _resolve_card_target(
        self,
        target: DOMClickTarget,
        *,
        expected_selected: int | None,
    ) -> tuple[Any, DOMCardStack]:
        cards = self.page.locator("div.card-stacks > div")
        candidates: list[Any] = []
        facts: list[DOMCardStack] = []
        try:
            for index in range(await cards.count()):
                candidate = cards.nth(index)
                if not await candidate.is_visible():
                    continue
                observed = await candidate.evaluate(
                    """element => {
                        const name = Array.from(
                            element.querySelectorAll(
                                ".name-layer:not(.invisible)"
                            )
                        ).map(layer => layer.textContent.trim())
                            .filter(Boolean);
                        const counters = Array.from(
                            element.querySelectorAll(
                                ".counter-layer:not(.invisible)"
                            )
                        ).filter(
                            layer => getComputedStyle(layer).display !== "none"
                        );
                        const selected = (
                            element.querySelector("selection-cross") !== null
                        );
                        const selectedCount = selected
                            ? Number.parseInt(
                                counters.map(
                                    layer => layer.textContent.trim()
                                ).find(text => /^\\d+$/.test(text)) || "1",
                                10
                            )
                            : 0;
                        const all = Array.from(
                            element.querySelectorAll(
                                ".all-button:not(.invisible)"
                            )
                        ).find(
                            layer => getComputedStyle(layer).display !== "none"
                        );
                        const rect = element.getBoundingClientRect();
                        const style = getComputedStyle(element);
                        let allCoversCenter = false;
                        if (all) {
                            const allRect = all.getBoundingClientRect();
                            const centerX = rect.left + rect.width / 2;
                            const centerY = rect.top + rect.height / 2;
                            allCoversCenter = (
                                allRect.left <= centerX
                                && centerX <= allRect.right
                                && allRect.top <= centerY
                                && centerY <= allRect.bottom
                            );
                        }
                        return {
                            name: name.length === 1 ? name[0] : null,
                            width: rect.width,
                            height: rect.height,
                            zIndex: Number.parseInt(style.zIndex, 10) || 0,
                            hasVisibleCounter: counters.length > 0,
                            selectedCount,
                            clickable: (
                                style.cursor === "pointer"
                                && style.pointerEvents !== "none"
                            ),
                            hasVisibleAll: Boolean(all),
                            allCoversCenter,
                        };
                    }"""
                )
                if observed["name"] is None:
                    continue
                candidates.append(candidate)
                facts.append(
                    DOMCardStack(
                        identity=str(observed["name"]),
                        width=float(observed["width"]),
                        height=float(observed["height"]),
                        z_index=int(observed["zIndex"]),
                        has_visible_counter=bool(
                            observed["hasVisibleCounter"]
                        ),
                        selected_count=int(observed["selectedCount"]),
                        clickable=bool(observed["clickable"]),
                        has_visible_all=bool(observed["hasVisibleAll"]),
                        all_covers_center=bool(observed["allCoversCenter"]),
                    )
                )
        except Exception as error:
            raise _GestureStepError(
                "not-found",
                f"card DOM changed during live resolution: {error}",
            ) from error
        observed_selected = selected_card_counts(facts)
        if (
            expected_selected is not None
            and observed_selected[target.identity] != expected_selected
        ):
            raise _GestureStepError(
                "not-found",
                "duplicate-selection accounting expected "
                f"{expected_selected} selected {target.identity!r}, "
                f"observed={observed_selected[target.identity]}; "
                f"selected={dict(observed_selected)!r}; stacks={tuple(facts)!r}",
            )
        try:
            match = resolve_card_stack_index(target, facts)
        except ActuationError as error:
            raise _GestureStepError("not-found", str(error)) from error
        stack = facts[match]
        if not stack.clickable or stack.selected_count:
            raise _GestureStepError(
                "not-found",
                f"found=1 but target is not an unselected clickable stack: {stack!r}",
            )
        if stack.all_covers_center:
            raise _GestureStepError(
                "not-found",
                "found=1 but the distinct All control covers the stack "
                f"center: {stack!r}",
            )
        return candidates[match], stack

    async def _click_resolved_card(
        self,
        candidate: Any,
        stack: DOMCardStack,
    ) -> None:
        try:
            await candidate.click(
                position={
                    "x": stack.width / 2,
                    "y": stack.height / 2,
                }
            )
        except Exception as error:
            raise _GestureStepError(
                "click-error",
                f"found=1 stack={stack!r}; click={error}",
            ) from error

    async def _wait_for_selection_count(
        self,
        identity: str,
        expected: int,
    ) -> None:
        deadline = self._deadline()
        observed: Counter[str] = Counter()
        detail = "selection DOM was not readable"
        while True:
            try:
                observed = await self._selected_card_counts()
                actual = observed[identity]
                detail = (
                    f"selection-effect expected {identity!r} count={expected}, "
                    f"observed={actual}; selected={dict(observed)!r}"
                )
                if actual == expected:
                    return
                if actual > expected:
                    raise _GestureStepError("selection-mismatch", detail)
            except _GestureStepError:
                raise
            except Exception as error:
                detail = f"selection DOM changed during verification: {error}"
            if not await self._wait_for_next_poll(deadline):
                raise _GestureStepError("no-effect", detail)

    async def _wait_for_actionable_click(
        self,
        click: Callable[[], Awaitable[None]],
    ) -> None:
        deadline = self._deadline()
        last_error: _GestureStepError | None = None
        while True:
            try:
                await click()
                return
            except _GestureStepError as error:
                if error.status != "not-found":
                    raise
                last_error = error
            if not await self._wait_for_next_poll(deadline):
                assert last_error is not None
                raise last_error

    def _deadline(self) -> float:
        return (
            asyncio.get_running_loop().time()
            + self.actuation_timeout_seconds
        )

    async def _wait_for_next_poll(self, deadline: float) -> bool:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return False
        delay = min(self.poll_interval_seconds, remaining)
        wait_for_timeout = getattr(self.page, "wait_for_timeout", None)
        if wait_for_timeout is None:
            await asyncio.sleep(delay)
        else:
            await wait_for_timeout(delay * 1000)
        return True

    async def _visible_button_boxes(
        self,
    ) -> tuple[list[Any], list[DOMGameButton]]:
        buttons = self.page.locator("div.game-buttons canvas")
        locators: list[Any] = []
        facts: list[DOMGameButton] = []
        for index in range(await buttons.count()):
            candidate = buttons.nth(index)
            if not await candidate.is_visible():
                continue
            box = await candidate.bounding_box()
            if box is None or box["height"] <= 0:
                continue
            locators.append(candidate)
            facts.append(
                DOMGameButton(
                    x=float(box["x"]),
                    width=float(box["width"]),
                    height=float(box["height"]),
                    enabled=await self._button_is_enabled(candidate),
                )
            )
        return locators, facts

    async def _button_is_enabled(self, candidate: Any) -> bool:
        is_enabled = getattr(candidate, "is_enabled", None)
        try:
            if is_enabled is not None and not await is_enabled():
                return False
            evaluate = getattr(candidate, "evaluate", None)
            if evaluate is None:
                return True
            return bool(
                await evaluate(
                    """element => {
                        const style = getComputedStyle(element);
                        return (
                            style.pointerEvents !== "none"
                            && !element.hasAttribute("disabled")
                            && element.getAttribute("aria-disabled") !== "true"
                        );
                    }"""
                )
            )
        except Exception:
            return False

    async def _click_autoplay_button(self) -> None:
        locators, buttons = await self._visible_button_boxes()
        matches = [
            index
            for index, button in enumerate(buttons)
            if game_button_role(
                width=button.width,
                height=button.height,
            )
            == "primary"
            and button.enabled
        ]
        if len(matches) != 1:
            raise _GestureStepError(
                "not-found",
                f"expected one primary autoplay canvas, found={len(matches)}; "
                f"buttons={buttons!r}",
            )
        try:
            await locators[matches[0]].click()
        except Exception as error:
            raise _GestureStepError(
                "click-error",
                f"found=1 autoplay button={buttons[matches[0]]!r}; click={error}",
            ) from error

    async def _click_submit_button(
        self,
        decision: PendingDecisionSnapshot,
    ) -> None:
        locators, buttons = await self._visible_button_boxes()
        if not buttons:
            raise _GestureStepError("not-found", "no visible submit canvas; found=0")
        try:
            match = resolve_submit_button_index(decision, buttons)
        except ActuationError as error:
            raise _GestureStepError("not-found", str(error)) from error
        try:
            await locators[match].click()
        except Exception as error:
            raise _GestureStepError(
                "click-error",
                f"found=1 submit button={buttons[match]!r}; click={error}",
            ) from error

    async def _click_reaction_decline(self) -> None:
        """Click the one visible DONE_REACTING control, or fail loudly."""
        locators, buttons = await self._visible_button_boxes()
        matches = [
            index
            for index, button in enumerate(buttons)
            if game_button_role(
                width=button.width,
                height=button.height,
            )
            == "primary"
            and button.enabled
        ]
        if len(matches) != 1:
            raise _GestureStepError(
                "not-found",
                "DONE_REACTING control is not unambiguous; "
                f"found={len(matches)}; buttons={buttons!r}",
            )
        try:
            await locators[matches[0]].click()
        except Exception as error:
            raise _GestureStepError(
                "click-error",
                f"found=1 DONE_REACTING button="
                f"{buttons[matches[0]]!r}; click={error}",
            ) from error

    async def _click_mode_button(
        self,
        index: int,
        offered_count: int,
        *,
        start_confirmation: bool = False,
    ) -> None:
        locators, facts = await self._visible_button_boxes()
        buttons = sorted(
            zip(locators, facts, strict=True),
            key=lambda item: item[1].x,
        )
        if len(buttons) == offered_count:
            match = index
            if not buttons[match][1].enabled:
                raise _GestureStepError(
                    "not-found",
                    f"mode canvas index={match} is present but disabled; "
                    f"buttons={facts!r}",
                )
        elif start_confirmation and offered_count == 1 and index == 0:
            primary = [
                button_index
                for button_index, (_, fact) in enumerate(buttons)
                if game_button_role(width=fact.width, height=fact.height)
                == "primary"
                and fact.enabled
            ]
            if len(primary) != 1:
                raise _GestureStepError(
                    "not-found",
                    "start-confirmation searched div.game-buttons canvas for "
                    "one visible mode canvas or one unambiguous primary canvas; "
                    f"visible={len(buttons)} primary={len(primary)}; "
                    f"buttons={facts!r}",
                )
            match = primary[0]
        else:
            raise _GestureStepError(
                "not-found",
                f"expected {offered_count} mode canvases, found={len(buttons)}; "
                f"buttons={facts!r}",
            )
        try:
            await buttons[match][0].click()
        except Exception as error:
            raise _GestureStepError(
                "click-error",
                f"found={len(buttons)} mode button index={match}; click={error}",
            ) from error

    async def _start_confirmation_failure_context(
        self,
        decision: PendingDecisionSnapshot,
    ) -> str:
        searched = (
            "start-confirmation searched div.game-buttons canvas for the sole "
            "visible mode/primary button"
        )
        if self.snapshot_dom is None:
            return f"; {searched}; DOM snapshot unavailable (not configured)"
        label = (
            "actuation-failure-start-confirmation-"
            f"question-{decision.question_index}"
        )
        try:
            destination = await self.snapshot_dom(label)
        except Exception as error:
            return f"; {searched}; DOM snapshot failed: {error}"
        return f"; {searched}; DOM snapshot={destination}"

    async def _verify_selected_cards(self, labels: tuple[str, ...]) -> None:
        wanted = Counter(labels)
        observed: Counter[str] = Counter()
        detail = "selection DOM was not readable"
        deadline = self._deadline()
        while True:
            try:
                observed = await self._selected_card_counts()
                detail = (
                    f"selection-check expected={dict(wanted)!r} "
                    f"observed={dict(observed)!r}"
                )
                if observed == wanted:
                    return
            except Exception as error:
                detail = f"selection-check DOM changed: {error}"
            if not await self._wait_for_next_poll(deadline):
                break
        raise _GestureStepError(
            "not-found",
            detail,
        )

    async def _selected_card_counts(self) -> Counter[str]:
        raw = await self.page.locator("div.card-stacks > div").evaluate_all(
            """elements => elements.flatMap(element => {
                if (element.querySelector("selection-cross") === null) {
                    return [];
                }
                const name = Array.from(
                    element.querySelectorAll(".name-layer:not(.invisible)")
                ).map(layer => layer.textContent.trim()).filter(Boolean);
                if (name.length !== 1) {
                    return [];
                }
                const count = Array.from(
                    element.querySelectorAll(".counter-layer:not(.invisible)")
                ).map(layer => layer.textContent.trim())
                    .find(text => /^\\d+$/.test(text));
                return [[name[0], Number.parseInt(count || "1", 10)]];
            })"""
        )
        counts: Counter[str] = Counter()
        for name, count in raw:
            counts[str(name)] += int(count)
        return counts


_HAND_QUESTIONS = frozenset(
    {
        "ARTISAN_TOPDECK",
        "BUREAUCRAT",
        "CELLAR",
        "CHAPEL",
        "GAME_ACTION_PHASE",
        "GAME_MAY_REACT_WITH",
        "MILITIA",
        "MINE_TRASH",
        "MONEYLENDER",
        "POACHER",
        "REMODEL_TRASH",
        "THRONE_ROOM",
    }
)
_SUPPLY_QUESTIONS = frozenset(
    {
        "ARTISAN_GAIN",
        "REMODEL_GAIN",
        "WORKSHOP",
    }
)


def dom_click_target(
    decision: PendingDecisionSnapshot,
    offered_element: str,
    *,
    offered_index: int,
) -> DOMClickTarget:
    """Resolve protocol path/identity to a physical DOM region."""
    if decision.question_id == "GAME_BUY_PHASE":
        if offered_element.startswith("0:"):
            region = "supply"
        elif offered_element.startswith("1:0:"):
            region = "hand"
        elif offered_element.endswith("AUTOPLAY_TREASURES"):
            region = "autoplay-button"
        else:
            raise ActionMappingError(
                f"unknown buy-region element {offered_element!r}"
            )
    elif decision.decision_type == "CHOOSE_MODE":
        region = "mode-button"
    elif decision.question_id in _HAND_QUESTIONS:
        region = "hand"
    elif decision.question_id in _SUPPLY_QUESTIONS:
        region = "supply"
    else:
        region = "display"
    return DOMClickTarget(
        region=region,
        identity=offered_name(offered_element),
        offered_index=offered_index,
    )


def card_stack_region(
    *,
    width: float,
    height: float,
    z_index: int,
    has_visible_counter: bool,
) -> str | None:
    """Classify one client-2.2.8 direct card-stack child."""
    if width <= 0 or height <= 0:
        return None
    if width > height and has_visible_counter:
        return "supply"
    if height > width and 2000 <= z_index < 3000:
        return "hand"
    if height > width and z_index >= 10_000:
        return "display"
    return None


def resolve_card_stack_index(
    target: DOMClickTarget,
    stacks: Sequence[DOMCardStack],
) -> int:
    """Resolve a logical card target without conflating duplicate copies.

    The client collapses unselected duplicate cards into one stack. Selected
    copies may be rendered as a second, rotated stack with a
    ``selection-cross``; only the region-qualified unselected stack is a
    candidate for the next click.
    """
    matches = [
        index
        for index, stack in enumerate(stacks)
        if stack.identity == target.identity
        and stack.selected_count == 0
        and card_stack_region(
            width=stack.width,
            height=stack.height,
            z_index=stack.z_index,
            has_visible_counter=stack.has_visible_counter,
        )
        == target.region
    ]
    if len(matches) != 1:
        raise ActuationError(
            f"expected one {target.region} target for {target.identity!r}, "
            f"found={len(matches)}; stacks={tuple(stacks)!r}"
        )
    return matches[0]


def selected_card_counts(stacks: Iterable[DOMCardStack]) -> Counter[str]:
    """Return the selected multiset encoded by recorded selection stacks."""
    counts: Counter[str] = Counter()
    for stack in stacks:
        if stack.selected_count:
            counts[stack.identity] += stack.selected_count
    return counts


def game_button_role(*, width: float, height: float) -> str | None:
    """Classify recorded client-2.2.8 game-button canvas geometry."""
    if width <= 0 or height <= 0:
        return None
    ratio = width / height
    if ratio >= 3.5:
        # The same wide primary canvas is Autoplay Treasures in buy phase and
        # Confirm Trashing/Discarding for effect prompts.
        return "primary"
    return "secondary"


def resolve_submit_button_index(
    decision: PendingDecisionSnapshot,
    buttons: Sequence[DOMGameButton],
) -> int:
    """Resolve the recorded phase or effect confirmation canvas."""
    if decision.question_id in {
        "GAME_ACTION_PHASE",
        "GAME_BUY_PHASE",
    }:
        enabled = [
            (index, button)
            for index, button in enumerate(buttons)
            if button.enabled
        ]
        if not enabled:
            matches: list[int] = []
        else:
            rightmost_x = max(button.x for _, button in enabled)
            matches = [
                index
                for index, button in enabled
                if button.x == rightmost_x
            ]
    else:
        matches = [
            index
            for index, button in enumerate(buttons)
            if game_button_role(
                width=button.width,
                height=button.height,
            )
            == "primary"
            and button.enabled
        ]
    if len(matches) != 1:
        raise ActuationError(
            f"submit canvas is not unambiguous; found={len(matches)}; "
            f"buttons={tuple(buttons)!r}"
        )
    return matches[0]


def gesture_click_targets(
    gesture: ClientGesture,
    decision: PendingDecisionSnapshot,
    offered: tuple[str, ...],
) -> tuple[DOMClickTarget, ...]:
    """Return the complete ordered physical plan for a mapped gesture."""
    targets = [
        dom_click_target(
            decision,
            offered[index],
            offered_index=index,
        )
        for index in gesture.selected_indices
    ]
    if gesture.click_button:
        targets.append(
            DOMClickTarget(
                region=(
                    "decline-button"
                    if decision.question_id == "GAME_MAY_REACT_WITH"
                    else "submit-button"
                ),
                identity=decision.question_id,
                offered_index=-1,
            )
        )
    return tuple(targets)


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


def _needs_button(
    decision: PendingDecisionSnapshot,
    selected: tuple[int, ...],
) -> bool:
    if decision.question_id == "GAME_MAY_REACT_WITH":
        return not selected
    if decision.question_id in {
        "GAME_ACTION_PHASE",
        "GAME_BUY_PHASE",
        "THRONE_ROOM",
    }:
        return not selected
    if decision.decision_type in {"CHOOSE_MODE", "ORDER_CARDS"}:
        return False
    if decision.maximum > 1:
        # Recorded multi-select prompts retain the selection and expose a wide
        # primary confirmation canvas even after the maximum is selected.
        return True
    required_to_auto_submit = min(decision.maximum, len(decision.offered))
    return len(selected) < required_to_auto_submit


def _verify_selection_before_submit(
    decision: PendingDecisionSnapshot,
) -> bool:
    return (
        decision.maximum > 1
        and decision.question_id
        not in {
            "GAME_ACTION_PHASE",
            "GAME_BUY_PHASE",
            "THRONE_ROOM",
        }
        and decision.decision_type not in {"CHOOSE_MODE", "ORDER_CARDS"}
    )


class _GestureStepError(RuntimeError):
    """One planned physical gesture step failed with structured status."""

    def __init__(self, status: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


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
