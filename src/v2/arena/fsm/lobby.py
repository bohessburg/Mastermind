"""Recorded-DOM lobby automation for consecutive supervised arena games."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from html.parser import HTMLParser
from typing import Any, Awaitable, Callable

from ..config import LobbyConfig


LOGGER = logging.getLogger(__name__)


class LobbyError(RuntimeError):
    """The recorded lobby flow no longer matches the live browser."""


class LobbyState(str, Enum):
    """Explicit states for one automatch session."""

    HOMEPAGE = "homepage"
    SEARCHING = "searching"
    TABLE_WAITING = "table_waiting"
    IN_GAME = "in_game"
    GAME_OVER = "game_over"
    LEAVING = "leaving"


class LobbyControl(str, Enum):
    """Controls whose selectors were observed in the reference DOMs."""

    START_SEARCH = "start_search"
    START_GAME = "start_game"
    DISMISS_GAME_ENDED = "dismiss_game_ended"
    LEAVE_TABLE = "leave_table"
    RECONNECT_LIMIT_RECONNECT = "reconnect_limit_reconnect"
    RECONNECT_LIMIT_RETURN_TO_LOBBY = "reconnect_limit_return_to_lobby"


# Evidence: dom-34066.html/4341034.html have the first selector and its
# searchNow() handler.  dom-530325.html has the recorded table controls.
START_SEARCH_SELECTOR = (
    'button.automatch-button[ng-click="$ctrl.automatch.searchNow()"]'
)
START_GAME_SELECTOR = (
    'score-table-buttons button.lobby-button[ng-click="$ctrl.readyClick()"]'
)
LEAVE_TABLE_SELECTOR = (
    'score-table-buttons button.lobby-button[ng-click="$ctrl.leave()"]'
)
DISMISS_GAME_ENDED_SELECTOR = (
    'game-ended-notification modal-window '
    'button.lobby-button[ng-click="$ctrl.ok()"]'
)
# The reconnect-limit prompt is in the main-window ``reconnecting-failed``
# component.  ``:has(div.timeout)`` identifies its reconnect-count content,
# keeping its ``$ctrl.decline()`` separate from the in-game undo modal's
# identically named handler.  Target the inner ``div.modal-window`` rather
# than the ``<modal-window>`` custom element: the custom element defaults to
# ``display: inline`` around fixed-position children, so it has an empty box
# and never reports visible, which hid this modal from the poll loop live.
RECONNECT_LIMIT_MODAL_SELECTOR = "reconnecting-failed div.modal-window:has(div.timeout)"
RECONNECT_LIMIT_RECONNECT_SELECTOR = (
    f'{RECONNECT_LIMIT_MODAL_SELECTOR} '
    'button.lobby-button[ng-click="$ctrl.reconnect()"]'
)
RECONNECT_LIMIT_RETURN_TO_LOBBY_SELECTOR = (
    f'{RECONNECT_LIMIT_MODAL_SELECTOR} '
    'button.lobby-button[ng-click="$ctrl.decline()"]'
)
# dom-3978828.html and dom-2848087.html both contain the in-play board,
# ``game-area``.  dom-530325.html is a table-waiting snapshot with a
# message-input placeholder and score-table controls but no game-area, so chat
# alone is not a game signal.
IN_GAME_SELECTOR = "game-area"
GAME_CHAT_SELECTOR = 'input[placeholder="message"]'
TABLE_CONTAINER_SELECTOR = (
    'score-table, score-table-buttons, input[placeholder="message"]'
)
# These main-window component tags are always present in the client DOM, even
# when Angular has rendered only their empty comment shell.  A component is a
# blocker only when it is visible and has rendered human-visible content.
MODAL_WINDOW_SELECTOR = "modal-window"
RECONNECTING_SELECTOR = "reconnecting"
RECONNECTING_FAILED_SELECTOR = "reconnecting-failed"
CONNECTING_TO_GAME_SELECTOR = "connecting-to-game"
LOADING_GAME_SELECTOR = "loading-game"

_HOMEPAGE_TRANSIENT_BLOCKER_SELECTORS = (
    MODAL_WINDOW_SELECTOR,
    RECONNECTING_SELECTOR,
    CONNECTING_TO_GAME_SELECTOR,
    LOADING_GAME_SELECTOR,
)

_RETURN_TO_HOMEPAGE = object()

@dataclass(frozen=True, kw_only=True)
class _ControlSpec:
    selector: str
    label: str
    handler: str
    required_class: str
    ancestor_tag: str | None = None
    requires_timeout_descendant: bool = False


_CONTROL_SPECS = {
    LobbyControl.START_SEARCH: _ControlSpec(
        selector=START_SEARCH_SELECTOR,
        label="Start search",
        handler="$ctrl.automatch.searchNow()",
        required_class="automatch-button",
    ),
    # The client labels this button "Ready".  It is the only recorded
    # table-start control, so the FSM calls it the semantic Start game step.
    LobbyControl.START_GAME: _ControlSpec(
        selector=START_GAME_SELECTOR,
        label="Ready",
        handler="$ctrl.readyClick()",
        required_class="lobby-button",
        ancestor_tag="score-table-buttons",
    ),
    # Evidence: dom-4326014.html has this button inside the blocking
    # game-ended-notification modal, with the observed $ctrl.ok() handler.
    LobbyControl.DISMISS_GAME_ENDED: _ControlSpec(
        selector=DISMISS_GAME_ENDED_SELECTOR,
        label="Ok",
        handler="$ctrl.ok()",
        required_class="lobby-button",
        ancestor_tag="game-ended-notification",
    ),
    LobbyControl.LEAVE_TABLE: _ControlSpec(
        selector=LEAVE_TABLE_SELECTOR,
        label="Leave Table",
        handler="$ctrl.leave()",
        required_class="lobby-button",
        ancestor_tag="score-table-buttons",
    ),
    LobbyControl.RECONNECT_LIMIT_RECONNECT: _ControlSpec(
        selector=RECONNECT_LIMIT_RECONNECT_SELECTOR,
        label="Reconnect again",
        handler="$ctrl.reconnect()",
        required_class="lobby-button",
        ancestor_tag="reconnecting-failed",
        requires_timeout_descendant=True,
    ),
    LobbyControl.RECONNECT_LIMIT_RETURN_TO_LOBBY: _ControlSpec(
        selector=RECONNECT_LIMIT_RETURN_TO_LOBBY_SELECTOR,
        label="Return to lobby",
        handler="$ctrl.decline()",
        required_class="lobby-button",
        ancestor_tag="reconnecting-failed",
        requires_timeout_descendant=True,
    ),
}


class LobbyFSM:
    """Drive only controls and state signals seen in the reference capture.

    Automatch enters ``game-area`` directly; its apparent Start game modal is
    an in-feed game decision handled by the game loop. The recorded hosted-table
    ``Ready`` control remains supported for the separate table-waiting flow.
    Every missing-control failure leaves a DOM snapshot when supplied.
    """

    def __init__(
        self,
        page: Any,
        config: LobbyConfig,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        poll_seconds: float = 0.25,
        snapshot_dom: Callable[[str], Awaitable[Any]] | None = None,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("lobby poll interval must be positive")
        self.page = page
        self.config = config
        self.sleep = sleep
        self.clock = clock
        self.poll_seconds = poll_seconds
        self.snapshot_dom = snapshot_dom
        self.state = LobbyState.HOMEPAGE

    @property
    def max_games_per_session(self) -> int:
        """Return zero for an unlimited Ctrl-C-bounded session."""
        return self.config.max_games_per_session

    def reached_game_limit(self, completed_games: int) -> bool:
        """Whether normal completion should stop instead of requeueing."""
        limit = self.max_games_per_session
        return limit != 0 and completed_games >= limit

    async def queue_next_game(self) -> None:
        """Resume a retained board or search and start the next table."""
        await self.start_or_resume_game()

    async def start_or_resume_game(self) -> bool:
        """Poll startup controls until automatch starts or a board resumes.

        A reconnect-limit modal can render after the initial page load, so the
        homepage, board, and modal signals deliberately share one deadline and
        are all reconsidered after every poll.
        """
        self._require_state(LobbyState.HOMEPAGE)
        while True:
            start_search = await self._wait_for_homepage_start_search_or_game()
            if start_search is None:
                self._transition(LobbyState.IN_GAME)
                return True
            await self._click(start_search, LobbyControl.START_SEARCH)
            self._transition(LobbyState.SEARCHING)

            match = await self._wait_for_match()
            if match == "homepage":
                self._transition(LobbyState.HOMEPAGE)
                continue
            if match == "in-game":
                self._transition(LobbyState.IN_GAME)
                return False

            self._transition(LobbyState.TABLE_WAITING)
            start_game = await self._wait_for_table_start_or_game()
            if start_game is _RETURN_TO_HOMEPAGE:
                self._transition(LobbyState.HOMEPAGE)
                continue
            if start_game is None:
                self._transition(LobbyState.IN_GAME)
                return False
            await self._click(start_game, LobbyControl.START_GAME)
            self._transition(LobbyState.IN_GAME)
            return False

    async def resolve_startup_blocking_modal(self) -> bool:
        """Handle one known startup modal or report an unknown blocker."""
        self._require_state(LobbyState.HOMEPAGE)
        return await self._resolve_blocking_modal_if_present()

    async def _resolve_blocking_modal_if_present(self) -> bool:
        """Handle a visible modal while any lobby state is polling."""
        if await self._visible_selector_has_content(RECONNECT_LIMIT_MODAL_SELECTOR):
            control = (
                LobbyControl.RECONNECT_LIMIT_RETURN_TO_LOBBY
                if self.config.reconnect_limit_policy == "return_to_lobby"
                else LobbyControl.RECONNECT_LIMIT_RECONNECT
            )
            locator = await self._control_if_present(control)
            if locator is None:
                raise self._error(
                    "reconnect-limit modal was visible but its recorded "
                    f"{self._control_description(control)} was not actionable"
                )
            LOGGER.warning(
                "startup reconnect-limit modal: selecting %s",
                _CONTROL_SPECS[control].label,
            )
            await self._click(locator, control)
            await self._wait_for_control_to_be_gone(
                control,
                timeout_seconds=self.config.game_ended_dialog_timeout_seconds,
            )
            return True

        modal_texts = await self._visible_selector_texts(MODAL_WINDOW_SELECTOR)
        if modal_texts:
            await self._snapshot_failure("startup-unknown-modal")
            quoted = ", ".join(repr(text) for text in modal_texts)
            raise self._error(
                f"unrecognized blocking startup modal with visible text {quoted}"
            )
        return False

    async def resume_running_game_if_present(self) -> bool:
        """Enter the game loop when a reconnect retained an in-play board."""
        self._require_state(LobbyState.HOMEPAGE)
        if not await self._in_game_board_is_present():
            return False
        LOGGER.warning(
            "startup found an in-progress game board; waiting for its "
            "authoritative FullState instead of starting automatch"
        )
        self._transition(LobbyState.IN_GAME)
        return True

    async def recover_resumed_game_and_queue_next(self) -> None:
        """Leave a resumed game that never seeded a safe tracker state."""
        self._require_state(LobbyState.IN_GAME)
        LOGGER.critical(
            "RESUME RECOVERY: authoritative game state was unavailable or "
            "rejected; archiving the DOM and leaving the table before requeue"
        )
        await self._snapshot_dom("lobby-resume-recovery")
        # The recorded table exit is the safe path after a terminal game.  A
        # reconnect that cannot produce a trustworthy FullState is deliberately
        # abandoned rather than played from an unknown local state.
        self._transition(LobbyState.GAME_OVER)
        await self.leave_after_game(requeue=True)

    def game_ended(self) -> None:
        """Advance on the decoded ``GameEnd`` feed event."""
        self._require_state(LobbyState.IN_GAME)
        self._transition(LobbyState.GAME_OVER)

    async def leave_after_game(self, *, requeue: bool) -> None:
        """Leave the completed table and optionally start the next search."""
        self._require_state(LobbyState.GAME_OVER)
        dismiss_game_ended = await self._wait_for_optional_control(
            LobbyControl.DISMISS_GAME_ENDED,
            timeout_seconds=self.config.game_ended_dialog_timeout_seconds,
        )
        if dismiss_game_ended is None:
            leave_table = await self._control_if_present(LobbyControl.LEAVE_TABLE)
            if leave_table is None:
                spec = _CONTROL_SPECS[LobbyControl.LEAVE_TABLE]
                raise self._error(
                    "the game-ended notification was absent after "
                    f"{self.config.game_ended_dialog_timeout_seconds:.1f}s and "
                    f"{spec.label!r} ({spec.selector}) was not actionable"
                )
            LOGGER.warning(
                "game-ended notification was absent after %.1fs; recorded "
                "Leave Table control is actionable, proceeding",
                self.config.game_ended_dialog_timeout_seconds,
            )
        else:
            await self._click(
                dismiss_game_ended,
                LobbyControl.DISMISS_GAME_ENDED,
            )
            await self._wait_for_control_to_be_gone(
                LobbyControl.DISMISS_GAME_ENDED,
                timeout_seconds=self.config.game_ended_dialog_timeout_seconds,
            )
            leave_table = await self._wait_for_control(
                LobbyControl.LEAVE_TABLE,
                state=LobbyState.GAME_OVER,
            )
        self._transition(LobbyState.LEAVING)
        await self._click(leave_table, LobbyControl.LEAVE_TABLE)
        await self._wait_for_control(
            LobbyControl.START_SEARCH,
            state=LobbyState.LEAVING,
        )
        self._transition(LobbyState.HOMEPAGE)
        if requeue:
            await self.queue_next_game()

    async def _wait_for_match(self) -> str:
        """Wait for a table start control or an actual in-play board."""
        await self._snapshot_waiting(LobbyState.SEARCHING)
        deadline = self.clock() + self._timeout_for(LobbyState.SEARCHING)
        while True:
            if await self._resolve_blocking_modal_if_present():
                if self.config.reconnect_limit_policy == "return_to_lobby":
                    return "homepage"
                continue
            # Automatch lands directly in the game. Prefer that authoritative
            # board signal over any stale score-table controls.
            if await self._in_game_board_is_present():
                return "in-game"
            start_game = await self._control_if_present(LobbyControl.START_GAME)
            if start_game is not None:
                return "table"
            if await self._selector_count(TABLE_CONTAINER_SELECTOR) > 0:
                # A table has definitely landed, but its start control may be
                # rendering asynchronously, or automatch may be about to render
                # game-area. Apply the table-specific timeout to both signals.
                return "table"
            if self.clock() >= deadline:
                await self._snapshot_failure("searching-no-match-control")
                raise self._timeout_error(
                    LobbyState.SEARCHING,
                    "a hosted table's Ready control "
                    f"({START_GAME_SELECTOR}) or an in-play board "
                    f"({IN_GAME_SELECTOR})",
                )
            await self.sleep(self.poll_seconds)

    async def _wait_for_table_start_or_game(self) -> Any | None:
        """Return hosted-table Ready, or ``None`` when automatch enters play."""
        await self._snapshot_waiting(LobbyState.TABLE_WAITING)
        deadline = self.clock() + self._timeout_for(LobbyState.TABLE_WAITING)
        while True:
            if await self._resolve_blocking_modal_if_present():
                if self.config.reconnect_limit_policy == "return_to_lobby":
                    return _RETURN_TO_HOMEPAGE
                continue
            if await self._in_game_board_is_present():
                return None
            locator = await self._control_if_present(LobbyControl.START_GAME)
            if locator is not None:
                return locator
            if self.clock() >= deadline:
                await self._snapshot_failure(
                    "table_waiting-start_game-not-found"
                )
                raise self._timeout_error(
                    LobbyState.TABLE_WAITING,
                    self._control_description(LobbyControl.START_GAME)
                    + f" or an in-play board ({IN_GAME_SELECTOR})",
                )
            await self.sleep(self.poll_seconds)

    async def _wait_for_control(
        self,
        control: LobbyControl,
        *,
        state: LobbyState,
    ) -> Any:
        await self._snapshot_waiting(state)
        deadline = self.clock() + self._timeout_for(state)
        while True:
            if await self._resolve_blocking_modal_if_present():
                continue
            locator = await self._control_if_present(control)
            if locator is not None:
                return locator
            if self.clock() >= deadline:
                await self._snapshot_failure(
                    f"{state.value}-{control.value}-not-found"
                )
                raise self._timeout_error(state, self._control_description(control))
            await self.sleep(self.poll_seconds)

    async def _wait_for_homepage_start_search_or_game(self) -> Any | None:
        """Poll the startup modal, board, and homepage before one reload.

        The client retains empty ``reconnecting`` and ``loading-game`` custom
        elements in a healthy DOM.  Their presence alone must never suppress
        the bounded reload; only visible, rendered content is a blocker.
        """
        timeout_seconds = self._timeout_for(LobbyState.HOMEPAGE)
        started_at = self.clock()
        retry_at = started_at + timeout_seconds / 2
        deadline = started_at + timeout_seconds
        reloaded = False

        while True:
            if await self._resolve_blocking_modal_if_present():
                continue
            if await self._in_game_board_is_present():
                return None
            locator = await self._control_if_present(LobbyControl.START_SEARCH)
            if locator is not None:
                return locator

            reloaded_now = False
            if (
                not reloaded
                and await self._visible_selector_has_content(
                    RECONNECTING_FAILED_SELECTOR
                )
            ):
                await self._reload_homepage("reconnecting failed")
                reloaded = True
                reloaded_now = True
            elif (
                not reloaded
                and self.clock() >= retry_at
                and not await self._homepage_has_active_blocker()
            ):
                await self._reload_homepage(
                    f"Start search was absent after {timeout_seconds / 2:.1f}s"
                )
                reloaded = True
                reloaded_now = True

            if self.clock() >= deadline:
                await self._snapshot_failure("homepage-start_search-not-found")
                raise self._timeout_error(
                    LobbyState.HOMEPAGE,
                    self._control_description(LobbyControl.START_SEARCH),
                )
            if reloaded_now:
                continue
            await self.sleep(self.poll_seconds)

    async def _homepage_has_active_blocker(self) -> bool:
        """Whether a visible modal or main-window recovery state is active."""
        for selector in _HOMEPAGE_TRANSIENT_BLOCKER_SELECTORS:
            if await self._visible_selector_has_content(selector):
                return True
        return False

    async def _visible_selector_has_content(self, selector: str) -> bool:
        """Recognize rendered modal content while ignoring empty Angular shells."""
        return bool(await self._visible_selector_texts(selector))

    async def _visible_selector_texts(self, selector: str) -> tuple[str, ...]:
        """Return normalized text from visible rendered selector matches."""
        count = await self._selector_count(selector)
        locator = self.page.locator(selector)
        texts: list[str] = []
        for index in range(count):
            candidate = locator.nth(index)
            try:
                if not await candidate.is_visible():
                    continue
                text = " ".join((await candidate.inner_text()).split())
                if text:
                    texts.append(text)
            except Exception as error:
                raise self._error(
                    f"could not inspect visible content for {selector}: {error}"
                ) from error
        return tuple(texts)

    async def _reload_homepage(self, reason: str) -> None:
        """Reload once without resetting the homepage attempt's deadline."""
        LOGGER.warning("reloading homepage once: %s", reason)
        try:
            await self.page.reload(wait_until="domcontentloaded")
        except Exception as error:
            raise self._error(f"could not reload homepage: {error}") from error

    async def _wait_for_optional_control(
        self,
        control: LobbyControl,
        *,
        timeout_seconds: float,
    ) -> Any | None:
        deadline = self.clock() + timeout_seconds
        while True:
            locator = await self._control_if_present(control)
            if locator is not None:
                return locator
            if self.clock() >= deadline:
                return None
            await self.sleep(self.poll_seconds)

    async def _wait_for_control_to_be_gone(
        self,
        control: LobbyControl,
        *,
        timeout_seconds: float,
    ) -> None:
        deadline = self.clock() + timeout_seconds
        while True:
            if await self._control_if_present(control) is None:
                return
            if self.clock() >= deadline:
                spec = _CONTROL_SPECS[control]
                raise self._error(
                    f"{spec.label!r} ({spec.selector}) did not disappear after "
                    f"{timeout_seconds:.1f}s"
                )
            await self.sleep(self.poll_seconds)

    async def _control_if_present(self, control: LobbyControl) -> Any | None:
        spec = _CONTROL_SPECS[control]
        count = await self._selector_count(spec.selector)
        if count == 0:
            return None
        if count != 1:
            raise self._error(
                f"ambiguous {spec.label!r} control: selector {spec.selector} "
                f"matched {count} elements"
            )
        return self.page.locator(spec.selector)

    async def _in_game_board_is_present(self) -> bool:
        """Recognize only the recorded board, never the persistent chat pane."""
        count = await self._selector_count(IN_GAME_SELECTOR)
        if count == 0:
            return False
        if count != 1:
            raise self._error(
                "ambiguous in-game board signal: "
                f"{IN_GAME_SELECTOR} matched {count} elements"
            )
        chat_count = await self._selector_count(GAME_CHAT_SELECTOR)
        LOGGER.debug(
            "in-play board is present (%s); game-chat secondary signal count=%s",
            IN_GAME_SELECTOR,
            chat_count,
        )
        return True

    async def _selector_count(self, selector: str) -> int:
        try:
            return int(await self.page.locator(selector).count())
        except Exception as error:
            raise self._error(
                f"could not inspect recorded selector {selector}: {error}"
            ) from error

    async def _click(self, locator: Any, control: LobbyControl) -> None:
        spec = _CONTROL_SPECS[control]
        try:
            await locator.click()
        except Exception as error:
            raise self._error(
                f"could not click recorded {spec.label!r} control "
                f"({spec.selector}): {error}"
            ) from error

    def _control_description(self, control: LobbyControl) -> str:
        if control is LobbyControl.START_GAME:
            return (
                "hosted-table Ready control "
                f"({START_GAME_SELECTOR})"
            )
        spec = _CONTROL_SPECS[control]
        return f"{spec.label!r} ({spec.selector})"

    async def _snapshot_waiting(self, state: LobbyState) -> None:
        if state in {LobbyState.SEARCHING, LobbyState.TABLE_WAITING}:
            await self._snapshot_dom(
                f"lobby-{state.value.replace('_', '-')}-wait"
            )

    async def _snapshot_failure(self, label: str) -> None:
        """Capture a terminal failed resolution without masking its LobbyError."""
        await self._snapshot_dom(f"lobby-failure-{label}")

    async def _snapshot_dom(self, label: str) -> None:
        if self.snapshot_dom is None:
            return
        try:
            destination = await self.snapshot_dom(label)
        except Exception as error:
            LOGGER.warning("could not archive lobby DOM snapshot %s: %s", label, error)
        else:
            LOGGER.debug("archived lobby DOM snapshot %s at %s", label, destination)

    def _timeout_for(self, state: LobbyState) -> float:
        return {
            LobbyState.HOMEPAGE: self.config.homepage_timeout_seconds,
            LobbyState.SEARCHING: self.config.searching_timeout_seconds,
            LobbyState.TABLE_WAITING: self.config.table_waiting_timeout_seconds,
            LobbyState.IN_GAME: self.config.in_game_timeout_seconds,
            LobbyState.GAME_OVER: self.config.game_over_timeout_seconds,
            LobbyState.LEAVING: self.config.leaving_timeout_seconds,
        }[state]

    def _require_state(self, expected: LobbyState) -> None:
        if self.state is not expected:
            raise self._error(
                f"expected state {expected.value}, found {self.state.value}"
            )

    def _transition(self, target: LobbyState) -> None:
        self.state = target

    def _timeout_error(self, state: LobbyState, target: str) -> LobbyError:
        return self._error(
            f"{state.value} timed out after {self._timeout_for(state):.1f}s "
            f"waiting for {target}"
        )

    def _error(self, detail: str) -> LobbyError:
        return LobbyError(
            f"LOBBY ERROR [{self.state.value}]: {detail}. Browser left open; "
            "operator: inspect the visible lobby and either correct it or press Ctrl-C."
        )


class _RecordedControlParser(HTMLParser):
    """Tiny HTML reader used only to assert saved-DOM selector evidence."""

    def __init__(self, spec: _ControlSpec) -> None:
        super().__init__()
        self.spec = spec
        self._button_depth: int | None = None
        self._text: list[str] = []
        self.labels: list[str] = []
        self._depth = 0
        self._tags: list[str] = []
        self._reconnect_modal_depths: list[int] = []
        self._timeout_modal_depths: set[int] = set()

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        classes = attributes.get("class", "").split()
        if tag == "modal-window" and "reconnecting-failed" in self._tags:
            self._reconnect_modal_depths.append(self._depth)
        if (
            tag == "div"
            and "timeout" in classes
            and self._reconnect_modal_depths
        ):
            self._timeout_modal_depths.add(self._reconnect_modal_depths[-1])
        if (
            tag == "button"
            and attributes.get("ng-click") == self.spec.handler
            and self.spec.required_class in classes
            and (
                self.spec.ancestor_tag is None
                or self.spec.ancestor_tag in self._tags
            )
            and (
                not self.spec.requires_timeout_descendant
                or (
                    bool(self._reconnect_modal_depths)
                    and self._reconnect_modal_depths[-1]
                    in self._timeout_modal_depths
                )
            )
        ):
            self._button_depth = self._depth
            self._text = []
        self._tags.append(tag)
        self._depth += 1

    def handle_data(self, data: str) -> None:
        if self._button_depth is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        self._depth -= 1
        if tag == "button" and self._button_depth == self._depth:
            self.labels.append(" ".join("".join(self._text).split()))
            self._button_depth = None
            self._text = []
        if (
            tag == "modal-window"
            and self._reconnect_modal_depths
            and self._reconnect_modal_depths[-1] == self._depth
        ):
            modal_depth = self._reconnect_modal_depths.pop()
            self._timeout_modal_depths.discard(modal_depth)
        if self._tags:
            self._tags.pop()


def recorded_control_labels(html: str, control: LobbyControl) -> tuple[str, ...]:
    """Resolve a recorded control from a saved DOM without a browser.

    The live browser uses the exact CSS selector constants above.  This helper
    checks the matching observed Angular click handler and returns the button
    label so tests lock the selector evidence to the recording.
    """
    parser = _RecordedControlParser(_CONTROL_SPECS[control])
    parser.feed(html)
    parser.close()
    return tuple(parser.labels)
