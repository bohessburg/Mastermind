from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.v2.arena.actuate.clicks import UNDO_DECLINE_SELECTOR
from src.v2.arena.config import LobbyConfig
from src.v2.arena.fsm.lobby import (
    DISMISS_GAME_ENDED_SELECTOR,
    GAME_CHAT_SELECTOR,
    IN_GAME_SELECTOR,
    LEAVE_TABLE_SELECTOR,
    LOADING_GAME_SELECTOR,
    MODAL_WINDOW_SELECTOR,
    RECONNECT_LIMIT_MODAL_SELECTOR,
    RECONNECT_LIMIT_RECONNECT_SELECTOR,
    RECONNECT_LIMIT_RETURN_TO_LOBBY_SELECTOR,
    RECONNECTING_FAILED_SELECTOR,
    RECONNECTING_SELECTOR,
    START_GAME_SELECTOR,
    START_SEARCH_SELECTOR,
    TABLE_CONTAINER_SELECTOR,
    LobbyControl,
    LobbyError,
    LobbyFSM,
    LobbyState,
    recorded_control_labels,
)


RECORDING = Path("arena-recordings/20260724T142103.096991Z")
RESUME_FAILURE_DOM = Path(
    "exports/arena/20260725T025231.042167Z/"
    "dom-20260725T025402.631104Z-lobby-failure-homepage-start-search-not-found.html"
)
RECONNECT_LIMIT_DOM = Path(
    "exports/arena/20260725T032324.753710Z/"
    "dom-20260725T032456.384944Z-lobby-failure-homepage-start-search-not-found.html"
)


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    async def sleep(self, seconds: float) -> None:
        self.value += seconds


class _FakeLocator:
    def __init__(
        self,
        page: _FakePage,
        selector: str,
        index: int | None = None,
    ) -> None:
        self.page = page
        self.selector = selector
        self.index = index

    async def count(self) -> int:
        return self.page.count(self.selector)

    async def click(self) -> None:
        self.page.click(self.selector)

    def nth(self, index: int) -> _FakeLocator:
        return _FakeLocator(self.page, self.selector, index)

    async def is_visible(self) -> bool:
        return self.page.count(self.selector) > 0

    async def inner_text(self) -> str:
        return self.page.text(self.selector)


class _FakePage:
    def __init__(
        self,
        *,
        missing_table_start: bool = False,
        direct_automatch: bool = False,
        game_chat_only: bool = False,
        blank: bool = False,
        end_game_dialog: bool = True,
        unclickable_end_game_dialog: bool = False,
        clock: _FakeClock | None = None,
        start_search_at: float = 0.0,
        start_search_after_reload: bool = False,
        reconnecting_content: bool = False,
        reconnecting_failed_content: bool = False,
        loading_game_content: bool = False,
        modal_content: bool = False,
        reconnect_limit_modal: bool = False,
        unknown_modal_text: str | None = None,
    ) -> None:
        self.screen = (
            "reconnect_limit"
            if reconnect_limit_modal
            else "unknown_modal"
            if unknown_modal_text is not None
            else "blank"
            if blank
            else "homepage"
        )
        self.missing_table_start = missing_table_start
        self.direct_automatch = direct_automatch
        self.game_chat_only = game_chat_only
        self.end_game_dialog = end_game_dialog
        self.unclickable_end_game_dialog = unclickable_end_game_dialog
        self.clock = clock
        self.start_search_at = start_search_at
        self.start_search_after_reload = start_search_after_reload
        self.reconnecting_content = reconnecting_content
        self.reconnecting_failed_content = reconnecting_failed_content
        self.loading_game_content = loading_game_content
        self.modal_content = modal_content
        self.unknown_modal_text = unknown_modal_text
        self.reloaded = False
        self.reloads = 0
        self.clicks: list[str] = []

    def locator(self, selector: str) -> _FakeLocator:
        return _FakeLocator(self, selector)

    def count(self, selector: str) -> int:
        if self.screen == "reconnect_limit":
            if selector in {
                MODAL_WINDOW_SELECTOR,
                RECONNECT_LIMIT_MODAL_SELECTOR,
                RECONNECT_LIMIT_RECONNECT_SELECTOR,
                RECONNECT_LIMIT_RETURN_TO_LOBBY_SELECTOR,
            }:
                return 1
        if self.screen == "unknown_modal" and selector == MODAL_WINDOW_SELECTOR:
            return 1
        if (
            self.screen == "homepage"
            and selector == START_SEARCH_SELECTOR
            and self._start_search_is_rendered()
        ):
            return 1
        if selector == RECONNECTING_SELECTOR:
            return int(self.reconnecting_content)
        if selector == RECONNECTING_FAILED_SELECTOR:
            return int(self.reconnecting_failed_content)
        if selector == LOADING_GAME_SELECTOR:
            return int(self.loading_game_content)
        if selector == MODAL_WINDOW_SELECTOR:
            return int(self.modal_content)
        if (
            self.screen == "table"
            and not self.missing_table_start
            and selector == START_GAME_SELECTOR
        ):
            return 1
        if self.screen == "table":
            if selector == TABLE_CONTAINER_SELECTOR:
                return int(
                    not self.game_chat_only
                    or GAME_CHAT_SELECTOR in TABLE_CONTAINER_SELECTOR
                )
            if self.game_chat_only and selector == GAME_CHAT_SELECTOR:
                return 1
        if self.screen == "game":
            if selector in {IN_GAME_SELECTOR, GAME_CHAT_SELECTOR}:
                return 1
        if self.screen == "game_over":
            if selector == DISMISS_GAME_ENDED_SELECTOR and self.end_game_dialog:
                return 1
            if selector == LEAVE_TABLE_SELECTOR and not self.end_game_dialog:
                return 1
        return 0

    def text(self, selector: str) -> str:
        if self.screen == "reconnect_limit" and selector in {
            MODAL_WINDOW_SELECTOR,
            RECONNECT_LIMIT_MODAL_SELECTOR,
        }:
            return (
                "You've reconnected to this game 2 times. Reconnect again "
                "Return to lobby"
            )
        if self.screen == "unknown_modal" and selector == MODAL_WINDOW_SELECTOR:
            assert self.unknown_modal_text is not None
            return self.unknown_modal_text
        if selector == RECONNECTING_SELECTOR and self.reconnecting_content:
            return "Reconnecting to game server"
        if (
            selector == RECONNECTING_FAILED_SELECTOR
            and self.reconnecting_failed_content
        ):
            return "Could not reconnect"
        if selector == LOADING_GAME_SELECTOR and self.loading_game_content:
            return "Loading game"
        if selector == MODAL_WINDOW_SELECTOR and self.modal_content:
            return "A lobby modal is open"
        return ""

    async def reload(self, *, wait_until: str) -> None:
        assert wait_until == "domcontentloaded"
        self.reloads += 1
        self.reloaded = True

    def _start_search_is_rendered(self) -> bool:
        if self.start_search_after_reload and not self.reloaded:
            return False
        return self.clock is None or self.clock() >= self.start_search_at

    def click(self, selector: str) -> None:
        self.clicks.append(selector)
        if (
            self.screen == "reconnect_limit"
            and selector == RECONNECT_LIMIT_RETURN_TO_LOBBY_SELECTOR
        ):
            self.screen = "homepage"
        elif (
            self.screen == "reconnect_limit"
            and selector == RECONNECT_LIMIT_RECONNECT_SELECTOR
        ):
            self.screen = "game"
        elif self.screen == "homepage" and selector == START_SEARCH_SELECTOR:
            self.screen = "game" if self.direct_automatch else "table"
        elif self.screen == "table" and selector == START_GAME_SELECTOR:
            self.screen = "game"
        elif self.screen == "game_over" and selector == DISMISS_GAME_ENDED_SELECTOR:
            if self.unclickable_end_game_dialog:
                raise RuntimeError("modal button is obscured")
            self.end_game_dialog = False
        elif (
            self.screen == "game_over"
            and not self.end_game_dialog
            and selector == LEAVE_TABLE_SELECTOR
        ):
            self.screen = "homepage"
        else:
            raise AssertionError(f"unexpected click {selector} on {self.screen}")


def _config(**changes: object) -> LobbyConfig:
    values: dict[str, object] = {
        "max_games_per_session": 5,
        "homepage_timeout_seconds": 1.0,
        "searching_timeout_seconds": 1.0,
        "table_waiting_timeout_seconds": 1.0,
        "in_game_timeout_seconds": 1.0,
        "game_over_timeout_seconds": 1.0,
        "game_ended_dialog_timeout_seconds": 1.0,
        "leaving_timeout_seconds": 1.0,
    }
    values.update(changes)
    return LobbyConfig(**values)


def test_lobby_fsm_drives_the_full_recorded_happy_cycle() -> None:
    page = _FakePage()
    clock = _FakeClock()
    lobby = LobbyFSM(page, _config(), clock=clock, sleep=clock.sleep)

    asyncio.run(lobby.queue_next_game())

    assert lobby.state is LobbyState.IN_GAME
    assert page.clicks == [START_SEARCH_SELECTOR, START_GAME_SELECTOR]

    page.screen = "game_over"
    lobby.game_ended()
    asyncio.run(lobby.leave_after_game(requeue=True))

    assert lobby.state is LobbyState.IN_GAME
    assert page.clicks == [
        START_SEARCH_SELECTOR,
        START_GAME_SELECTOR,
        DISMISS_GAME_ENDED_SELECTOR,
        LEAVE_TABLE_SELECTOR,
        START_SEARCH_SELECTOR,
        START_GAME_SELECTOR,
    ]


def test_lobby_fsm_proceeds_when_the_game_ended_dialog_is_absent() -> None:
    page = _FakePage(end_game_dialog=False)
    clock = _FakeClock()
    lobby = LobbyFSM(page, _config(), clock=clock, sleep=clock.sleep)

    asyncio.run(lobby.queue_next_game())
    page.screen = "game_over"
    lobby.game_ended()
    asyncio.run(lobby.leave_after_game(requeue=False))

    assert lobby.state is LobbyState.HOMEPAGE
    assert page.clicks == [
        START_SEARCH_SELECTOR,
        START_GAME_SELECTOR,
        LEAVE_TABLE_SELECTOR,
    ]


def test_lobby_fsm_rejects_an_unclickable_game_ended_dialog() -> None:
    page = _FakePage(unclickable_end_game_dialog=True)
    clock = _FakeClock()
    lobby = LobbyFSM(page, _config(), clock=clock, sleep=clock.sleep)

    asyncio.run(lobby.queue_next_game())
    page.screen = "game_over"
    lobby.game_ended()

    with pytest.raises(LobbyError, match=r"\[game_over\].*could not click.*Ok"):
        asyncio.run(lobby.leave_after_game(requeue=False))

    assert page.clicks == [
        START_SEARCH_SELECTOR,
        START_GAME_SELECTOR,
        DISMISS_GAME_ENDED_SELECTOR,
    ]


def test_lobby_fsm_times_out_loudly_without_a_homepage_control() -> None:
    page = _FakePage(blank=True)
    clock = _FakeClock()
    snapshots: list[str] = []

    async def snapshot_dom(label: str) -> None:
        snapshots.append(label)

    lobby = LobbyFSM(
        page,
        _config(),
        clock=clock,
        sleep=clock.sleep,
        snapshot_dom=snapshot_dom,
    )

    with pytest.raises(LobbyError, match=r"\[homepage\].*Start search"):
        asyncio.run(lobby.queue_next_game())

    assert page.clicks == []
    assert page.reloads == 1
    assert clock.value == 1.0
    assert snapshots == ["lobby-failure-homepage-start_search-not-found"]


def test_lobby_fsm_waits_for_a_slow_cold_start_without_reloading() -> None:
    clock = _FakeClock()
    page = _FakePage(clock=clock, start_search_at=40.0)
    lobby = LobbyFSM(
        page,
        _config(homepage_timeout_seconds=90.0),
        clock=clock,
        sleep=clock.sleep,
    )

    asyncio.run(lobby.queue_next_game())

    assert lobby.state is LobbyState.IN_GAME
    assert page.reloads == 0
    assert clock.value == 40.0


def test_lobby_fsm_reloads_once_when_homepage_renders_only_after_retry() -> None:
    clock = _FakeClock()
    page = _FakePage(clock=clock, start_search_after_reload=True)
    lobby = LobbyFSM(
        page,
        _config(homepage_timeout_seconds=10.0),
        clock=clock,
        sleep=clock.sleep,
    )

    asyncio.run(lobby.queue_next_game())

    assert lobby.state is LobbyState.IN_GAME
    assert page.reloads == 1
    assert clock.value == 5.0


def test_lobby_fsm_does_not_reload_during_an_active_reconnect() -> None:
    clock = _FakeClock()
    page = _FakePage(
        clock=clock,
        start_search_at=6.0,
        reconnecting_content=True,
    )
    lobby = LobbyFSM(
        page,
        _config(homepage_timeout_seconds=10.0),
        clock=clock,
        sleep=clock.sleep,
    )

    asyncio.run(lobby.queue_next_game())

    assert lobby.state is LobbyState.IN_GAME
    assert page.reloads == 0
    assert clock.value == 6.0


def test_lobby_fsm_reloads_immediately_after_reconnect_failure() -> None:
    clock = _FakeClock()
    page = _FakePage(
        clock=clock,
        start_search_after_reload=True,
        reconnecting_failed_content=True,
    )
    lobby = LobbyFSM(
        page,
        _config(homepage_timeout_seconds=10.0),
        clock=clock,
        sleep=clock.sleep,
    )

    asyncio.run(lobby.queue_next_game())

    assert lobby.state is LobbyState.IN_GAME
    assert page.reloads == 1
    assert clock.value == 0.0


def test_lobby_fsm_rejects_a_missing_recorded_table_start_control() -> None:
    page = _FakePage(missing_table_start=True)
    clock = _FakeClock()
    lobby = LobbyFSM(page, _config(), clock=clock, sleep=clock.sleep)

    with pytest.raises(LobbyError, match=r"\[table_waiting\].*Ready control"):
        asyncio.run(lobby.queue_next_game())

    assert page.clicks == [START_SEARCH_SELECTOR]


def test_lobby_fsm_hands_direct_automatch_game_to_game_loop() -> None:
    page = _FakePage(direct_automatch=True)
    clock = _FakeClock()
    lobby = LobbyFSM(page, _config(), clock=clock, sleep=clock.sleep)

    asyncio.run(lobby.queue_next_game())

    assert lobby.state is LobbyState.IN_GAME
    assert page.clicks == [START_SEARCH_SELECTOR]


def test_lobby_fsm_returns_to_lobby_from_reconnect_limit_then_searches() -> None:
    page = _FakePage(reconnect_limit_modal=True)
    clock = _FakeClock()
    lobby = LobbyFSM(page, _config(), clock=clock, sleep=clock.sleep)

    assert asyncio.run(lobby.resolve_startup_blocking_modal())
    asyncio.run(lobby.queue_next_game())

    assert lobby.state is LobbyState.IN_GAME
    assert page.clicks == [
        RECONNECT_LIMIT_RETURN_TO_LOBBY_SELECTOR,
        START_SEARCH_SELECTOR,
        START_GAME_SELECTOR,
    ]


def test_lobby_fsm_reconnect_limit_policy_can_reconnect() -> None:
    page = _FakePage(reconnect_limit_modal=True)
    clock = _FakeClock()
    lobby = LobbyFSM(
        page,
        _config(reconnect_limit_policy="reconnect"),
        clock=clock,
        sleep=clock.sleep,
    )

    assert asyncio.run(lobby.resolve_startup_blocking_modal())
    assert asyncio.run(lobby.resume_running_game_if_present())

    assert lobby.state is LobbyState.IN_GAME
    assert page.clicks == [RECONNECT_LIMIT_RECONNECT_SELECTOR]


def test_lobby_fsm_reports_an_unknown_blocking_startup_modal() -> None:
    page = _FakePage(unknown_modal_text="Maintenance is in progress")
    clock = _FakeClock()
    snapshots: list[str] = []

    async def snapshot_dom(label: str) -> None:
        snapshots.append(label)

    lobby = LobbyFSM(
        page,
        _config(),
        clock=clock,
        sleep=clock.sleep,
        snapshot_dom=snapshot_dom,
    )

    with pytest.raises(LobbyError, match=r"'Maintenance is in progress'"):
        asyncio.run(lobby.resolve_startup_blocking_modal())

    assert page.clicks == []
    assert snapshots == ["lobby-failure-startup-unknown-modal"]


def test_lobby_fsm_resumes_a_running_game_before_homepage_search() -> None:
    page = _FakePage()
    page.screen = "game"
    clock = _FakeClock()
    lobby = LobbyFSM(page, _config(), clock=clock, sleep=clock.sleep)

    assert asyncio.run(lobby.resume_running_game_if_present())
    assert lobby.state is LobbyState.IN_GAME
    assert page.clicks == []


def test_lobby_fsm_resume_recovery_leaves_then_requeues() -> None:
    page = _FakePage(end_game_dialog=False)
    page.screen = "game"
    clock = _FakeClock()
    snapshots: list[str] = []

    async def snapshot_dom(label: str) -> None:
        snapshots.append(label)

    lobby = LobbyFSM(
        page,
        _config(),
        clock=clock,
        sleep=clock.sleep,
        snapshot_dom=snapshot_dom,
    )
    assert asyncio.run(lobby.resume_running_game_if_present())
    page.screen = "game_over"

    asyncio.run(lobby.recover_resumed_game_and_queue_next())

    assert lobby.state is LobbyState.IN_GAME
    assert page.clicks == [
        LEAVE_TABLE_SELECTOR,
        START_SEARCH_SELECTOR,
        START_GAME_SELECTOR,
    ]
    assert snapshots == [
        "lobby-resume-recovery",
        "lobby-searching-wait",
        "lobby-table-waiting-wait",
    ]


def test_lobby_fsm_does_not_treat_game_chat_without_board_as_in_game() -> None:
    page = _FakePage(missing_table_start=True, game_chat_only=True)
    clock = _FakeClock()
    snapshots: list[str] = []

    async def snapshot_dom(label: str) -> None:
        snapshots.append(label)

    lobby = LobbyFSM(
        page,
        _config(),
        clock=clock,
        sleep=clock.sleep,
        snapshot_dom=snapshot_dom,
    )

    with pytest.raises(LobbyError, match=r"\[table_waiting\].*Ready control"):
        asyncio.run(lobby.queue_next_game())

    assert lobby.state is LobbyState.TABLE_WAITING
    assert page.clicks == [START_SEARCH_SELECTOR]
    assert "lobby-searching-wait" in snapshots
    assert "lobby-table-waiting-wait" in snapshots
    assert "lobby-failure-table_waiting-start_game-not-found" in snapshots


def test_lobby_max_games_zero_is_unlimited_and_positive_limit_stops() -> None:
    page = _FakePage()
    clock = _FakeClock()
    finite = LobbyFSM(
        page,
        _config(max_games_per_session=1),
        clock=clock,
        sleep=clock.sleep,
    )
    unlimited = LobbyFSM(_FakePage(), _config(max_games_per_session=0))

    asyncio.run(finite.queue_next_game())
    page.screen = "game_over"
    finite.game_ended()
    asyncio.run(
        finite.leave_after_game(requeue=not finite.reached_game_limit(1))
    )

    assert finite.state is LobbyState.HOMEPAGE
    assert page.clicks == [
        START_SEARCH_SELECTOR,
        START_GAME_SELECTOR,
        DISMISS_GAME_ENDED_SELECTOR,
        LEAVE_TABLE_SELECTOR,
    ]
    assert not unlimited.reached_game_limit(10_000)


def test_recorded_lobby_control_selectors_resolve_saved_dom_snapshots() -> None:
    homepage = RECORDING / "dom-34066.html"
    game_over = RECORDING / "dom-530325.html"
    game_ended = RECORDING / "dom-4326014.html"
    in_game = RECORDING / "dom-3978828.html"
    later_in_game = RECORDING / "dom-2848087.html"
    if (
        not homepage.is_file()
        or not game_over.is_file()
        or not game_ended.is_file()
        or not in_game.is_file()
        or not later_in_game.is_file()
    ):
        pytest.skip("missing required recorded lobby DOM snapshots")

    assert recorded_control_labels(
        homepage.read_text(encoding="utf-8"),
        LobbyControl.START_SEARCH,
    ) == ("Start search",)
    scoreboard = game_over.read_text(encoding="utf-8")
    assert recorded_control_labels(scoreboard, LobbyControl.START_GAME) == ("Ready",)
    assert recorded_control_labels(scoreboard, LobbyControl.LEAVE_TABLE) == (
        "Leave Table",
    )
    assert recorded_control_labels(
        game_ended.read_text(encoding="utf-8"),
        LobbyControl.DISMISS_GAME_ENDED,
    ) == ("Ok",)
    chat_attribute = GAME_CHAT_SELECTOR.removeprefix("input[").removesuffix("]")
    assert chat_attribute in scoreboard
    assert f"<{IN_GAME_SELECTOR}" not in scoreboard
    assert f"<{IN_GAME_SELECTOR}" in in_game.read_text(encoding="utf-8")
    assert f"<{IN_GAME_SELECTOR}" in later_in_game.read_text(encoding="utf-8")


def test_restart_failure_dom_is_recognized_as_an_in_progress_game() -> None:
    assert RESUME_FAILURE_DOM.is_file(), "missing restart failure DOM evidence"

    html = RESUME_FAILURE_DOM.read_text(encoding="utf-8")

    assert f"<{IN_GAME_SELECTOR}" in html
    assert START_SEARCH_SELECTOR not in html


def test_reconnect_limit_evidence_uses_a_distinct_decline_scope() -> None:
    assert RECONNECT_LIMIT_DOM.is_file(), "missing reconnect-limit DOM evidence"

    html = RECONNECT_LIMIT_DOM.read_text(encoding="utf-8")

    assert "<reconnecting-failed>" in html
    assert '<div class="timeout">You\'ve reconnected to this game 2 times.</div>' in html
    assert recorded_control_labels(
        html,
        LobbyControl.RECONNECT_LIMIT_RECONNECT,
    ) == ("Reconnect again",)
    assert recorded_control_labels(
        html,
        LobbyControl.RECONNECT_LIMIT_RETURN_TO_LOBBY,
    ) == ("Return to lobby",)
    assert "reconnecting-failed modal-window:has(div.timeout)" in (
        RECONNECT_LIMIT_RETURN_TO_LOBBY_SELECTOR
    )
    assert "undo-request" in UNDO_DECLINE_SELECTOR
    assert RECONNECT_LIMIT_RETURN_TO_LOBBY_SELECTOR != UNDO_DECLINE_SELECTOR
