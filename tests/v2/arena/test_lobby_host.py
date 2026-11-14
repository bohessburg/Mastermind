from __future__ import annotations

import asyncio

import pytest

from src.v2.arena.config import ArenaConfig, LobbyConfig
from src.v2.arena.fsm.lobby import (
    DISMISS_GAME_ENDED_SELECTOR,
    HOST_CARD_POOL_INPUT_SELECTOR,
    HOST_LEAVE_TABLE_SELECTOR,
    HOST_PLAYER_LINE_SELECTOR,
    HOST_PLAYERS_TITLE_SELECTOR,
    HOST_READY_SELECTOR,
    HOST_SELF_READY_SELECTOR,
    IN_GAME_SELECTOR,
    MY_TABLE_SELECTOR,
    MY_TABLE_TAB_SELECTOR,
    NEW_TABLE_TAB_SELECTOR,
    LobbyError,
    LobbyFSM,
    LobbyState,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    async def sleep(self, seconds: float) -> None:
        self.value += seconds


class _Locator:
    def __init__(self, page: "_HostPage", selector: str) -> None:
        self.page = page
        self.selector = selector

    async def count(self) -> int:
        return self.page.count(self.selector)

    async def click(self) -> None:
        self.page.click(self.selector)

    def nth(self, _index: int) -> "_Locator":
        return self

    async def is_visible(self) -> bool:
        return self.page.count(self.selector) > 0

    async def inner_text(self) -> str:
        if self.selector == HOST_PLAYERS_TITLE_SELECTOR:
            return f"Players ({self.page.players()}/2)"
        return ""

    async def get_attribute(self, _name: str) -> str:
        return "lobby-button gold-button"

    async def input_value(self) -> str:
        return self.page.card_pool

    async def fill(self, value: str) -> None:
        self.page.card_pool = value

    async def dispatch_event(self, _name: str) -> None:
        return None


class _HostPage:
    """A lobby that creates a table on New Table and starts on two Readies."""

    def __init__(
        self,
        clock: _Clock,
        *,
        join_at: float,
        card_pool: str = "3",
        spectators: int = 0,
    ) -> None:
        self.clock = clock
        self.spectators = spectators
        self.join_at = join_at
        self.screen = "lobby"
        self.card_pool = card_pool
        self.self_ready = False
        self.loading_until = 0.0
        self.clicks: list[str] = []

    def locator(self, selector: str) -> _Locator:
        return _Locator(self, selector)

    def players(self) -> int:
        return 2 if self.clock() >= self.join_at else 1

    def count(self, selector: str) -> int:
        screen = self.screen
        if screen == "lobby":
            return int(selector == NEW_TABLE_TAB_SELECTOR)
        if screen == "table":
            if self.self_ready:
                # The live server starts even a solo game once all seated
                # players are ready; the table unmounts before the board.
                self.screen = "loading"
                self.loading_until = self.clock() + 2.0
                return self.count(selector)
            return {
                MY_TABLE_TAB_SELECTOR: 1,
                MY_TABLE_SELECTOR: 1,
                HOST_READY_SELECTOR: 1,
                HOST_LEAVE_TABLE_SELECTOR: 1,
                HOST_CARD_POOL_INPUT_SELECTOR: 1,
                HOST_PLAYER_LINE_SELECTOR: self.players(),
                HOST_PLAYERS_TITLE_SELECTOR: 1,
                # An unfiltered line selector would also see spectators.
                "my-table participant-list table-person-line": (
                    self.players() + self.spectators
                ),
                HOST_SELF_READY_SELECTOR: int(self.self_ready),
            }.get(selector, 0)
        if screen == "loading":
            if self.clock() < self.loading_until:
                return 0
            self.screen = "game"
            return self.count(selector)
        if screen == "game":
            return int(selector == IN_GAME_SELECTOR)
        if screen == "game_over":
            return int(selector == DISMISS_GAME_ENDED_SELECTOR)
        if screen == "post_game":
            return int(selector == HOST_LEAVE_TABLE_SELECTOR)
        return 0

    def click(self, selector: str) -> None:
        self.clicks.append(selector)
        if selector == NEW_TABLE_TAB_SELECTOR:
            self.screen = "table"
        elif selector == HOST_READY_SELECTOR:
            self.self_ready = not self.self_ready
        elif selector == DISMISS_GAME_ENDED_SELECTOR:
            self.screen = "post_game"
        elif selector == HOST_LEAVE_TABLE_SELECTOR:
            self.screen = "lobby"
            self.self_ready = False


def _fsm(page: _HostPage, clock: _Clock, **lobby: object) -> LobbyFSM:
    touches: list[float] = []
    fsm = LobbyFSM(
        page,
        LobbyConfig(mode="host", **lobby),
        sleep=clock.sleep,
        clock=clock,
        keepalive=lambda: touches.append(clock()),
    )
    fsm.touches = touches  # type: ignore[attr-defined]
    return fsm


def test_host_opens_level_one_table_and_readies_only_after_a_join() -> None:
    clock = _Clock()
    page = _HostPage(clock, join_at=5000.0)
    fsm = _fsm(page, clock)

    resumed = asyncio.run(fsm.queue_next_game())

    assert resumed is False
    assert fsm.state is LobbyState.IN_GAME
    assert page.card_pool == "1"
    assert page.clicks == [NEW_TABLE_TAB_SELECTOR, HOST_READY_SELECTOR]
    assert clock() >= 5000.0
    # The wait kept the idle watchdog fed for the whole empty-table period.
    assert fsm.touches[-1] >= 5000.0  # type: ignore[attr-defined]


def test_host_leaves_after_game_and_hosts_again() -> None:
    clock = _Clock()
    page = _HostPage(clock, join_at=0.0)
    fsm = _fsm(page, clock)
    asyncio.run(fsm.queue_next_game())
    page.screen = "game_over"
    fsm.game_ended()
    page.join_at = clock() + 100.0

    asyncio.run(fsm.leave_after_game(requeue=True))

    assert fsm.state is LobbyState.IN_GAME
    assert page.clicks.count(NEW_TABLE_TAB_SELECTOR) == 2
    assert HOST_LEAVE_TABLE_SELECTOR in page.clicks


def test_host_wait_timeout_fails_loudly() -> None:
    clock = _Clock()
    page = _HostPage(clock, join_at=float("inf"))
    fsm = _fsm(page, clock, host_wait_timeout_seconds=60.0)

    with pytest.raises(LobbyError, match="nobody joined"):
        asyncio.run(fsm.queue_next_game())
    assert HOST_READY_SELECTOR not in page.clicks


def test_host_config_validates_mode(tmp_path) -> None:
    path = tmp_path / "arena.json"
    path.write_text('{"lobby": {"mode": "lurk"}}')
    with pytest.raises(ValueError, match="mode"):
        ArenaConfig.load(path, environ={})
    path.write_text('{"lobby": {"mode": "host", "host_card_pool_level": 1}}')
    assert ArenaConfig.load(path, environ={}).lobby.mode == "host"


def test_host_does_not_ready_when_a_spectator_joins_an_open_seat() -> None:
    clock = _Clock()
    page = _HostPage(clock, join_at=float("inf"), spectators=1)
    fsm = _fsm(page, clock, host_wait_timeout_seconds=60.0)

    with pytest.raises(LobbyError, match="nobody joined"):
        asyncio.run(fsm.queue_next_game())
    assert HOST_READY_SELECTOR not in page.clicks
