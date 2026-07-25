"""Regression coverage for metagame opponent-timeout offers."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from src.v2.arena.actuate.clicks import (
    TIMEOUT_CLAIM_SELECTOR,
    PlaywrightActuator,
)
from src.v2.arena.actuate.protocol import ProtocolActuator
from src.v2.arena.fsm.game import RecordedDecisionProvider, run_game_loop
from src.v2.arena.protocol.events import (
    DecisionResolved,
    GameEnd,
    GameEvent,
    GameStart,
    TimeoutClaim,
    TimeoutOffer,
)
from src.v2.arena.protocol.frames import Direction, Writer, decode_frame
from src.v2.arena.protocol.messages import (
    TIMEOUT_REQUEST,
    encode_timeout_request,
)
from src.v2.arena.protocol.parser import ArenaParser
from src.v2.arena.protocol.recording import parse_recording


PARKED_ARCHIVE = Path(
    "exports/arena/20260725T014909.665607Z/frames.jsonl"
)


class _NoDecisions:
    async def plan(self, **_: Any) -> None:
        raise AssertionError("timeout fixtures do not contain local questions")


class _FakeSession:
    def __init__(self, clock: "_FakeClock | None" = None) -> None:
        self.clock = clock
        self.sent: list[bytes] = []
        self.claimed = asyncio.Event()
        self.claimed_at: float | None = None

    async def send_frame(self, msg_type: int, payload_bytes: bytes) -> None:
        self.sent.append(Writer().u32(msg_type).bytes(payload_bytes).build())
        if msg_type == TIMEOUT_REQUEST:
            self.claimed_at = None if self.clock is None else self.clock()
            self.claimed.set()


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        await asyncio.sleep(0)


def _start(*, our_seat: int = 0) -> GameStart:
    return GameStart(
        game_id=999,
        kingdom=("Chapel",),
        players=("bot", "opponent"),
        player_ids=(1, 2),
        our_seat=our_seat,
        timestamp_ms=1_000,
    )


def test_timeout_claim_encoder_and_parser_match_bundle_wire_layout() -> None:
    raw = (
        Writer()
        .u32(TIMEOUT_REQUEST)
        .bytes(encode_timeout_request(1))
        .build()
    )
    frame = decode_frame(raw, Direction.OUTBOUND, timestamp_ms=123)
    assert frame is not None

    events = ArenaParser().parse_frame(frame)

    assert events == [
        TimeoutClaim(
            player_seat=1,
            decision_index=-1,
            timestamp_ms=123,
        )
    ]


def test_opponent_timeout_claims_after_grace_via_protocol() -> None:
    session = _FakeSession()
    results = asyncio.run(
        run_game_loop(
            (
                _start(),
                TimeoutOffer(
                    player_seat=1,
                    decision_index=32,
                    timestamp_ms=2_000,
                ),
                GameEnd(game_id=999, reason="timeout-claimed", timestamp_ms=32_000),
            ),
            actuator=ProtocolActuator(session.send_frame),
            decision_provider=_NoDecisions(),
            timeout_offer_grace_seconds=30.0,
        )
    )

    assert results[0].completed
    assert session.sent == [
        Writer()
        .u32(TIMEOUT_REQUEST)
        .bytes(encode_timeout_request(1))
        .build()
    ]


def test_opponent_return_during_grace_cancels_timeout_claim() -> None:
    session = _FakeSession()
    asyncio.run(
        run_game_loop(
            (
                _start(),
                TimeoutOffer(
                    player_seat=1,
                    decision_index=32,
                    timestamp_ms=2_000,
                ),
                DecisionResolved(
                    question_index=33,
                    answers=(0,),
                    seat=1,
                    auto_played=False,
                    timestamp_ms=10_000,
                ),
                GameEnd(game_id=999, reason="opponent-returned", timestamp_ms=40_000),
            ),
            actuator=ProtocolActuator(session.send_frame),
            decision_provider=_NoDecisions(),
            timeout_offer_grace_seconds=30.0,
        )
    )

    assert session.sent == []


def test_our_timeout_offer_is_logged_loudly_without_self_resigning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _FakeSession()
    with caplog.at_level(logging.CRITICAL):
        asyncio.run(
            run_game_loop(
                (
                    _start(our_seat=0),
                    TimeoutOffer(
                        player_seat=0,
                        decision_index=32,
                        timestamp_ms=2_000,
                    ),
                    GameEnd(game_id=999, reason="self-timeout", timestamp_ms=40_000),
                ),
                actuator=ProtocolActuator(session.send_frame),
                decision_provider=_NoDecisions(),
                timeout_offer_grace_seconds=30.0,
            )
        )

    assert session.sent == []
    assert "TIMEOUT OFFER FOR OUR SEAT" in caplog.text


class _ClickButton:
    def __init__(self) -> None:
        self.clicked = False


class _ClickLocator:
    def __init__(self, button: _ClickButton, index: int | None = None) -> None:
        self.button = button
        self.index = index

    def nth(self, index: int) -> "_ClickLocator":
        return _ClickLocator(self.button, index)

    async def count(self) -> int:
        return 1

    async def is_visible(self) -> bool:
        return True

    async def click(self) -> None:
        self.button.clicked = True


class _ClickPage:
    def __init__(self, button: _ClickButton) -> None:
        self.button = button

    def locator(self, selector: str) -> _ClickLocator:
        assert selector == TIMEOUT_CLAIM_SELECTOR
        return _ClickLocator(self.button)


def test_click_actuator_uses_exact_timeout_claim_control() -> None:
    button = _ClickButton()
    clicked = asyncio.run(
        PlaywrightActuator(_ClickPage(button)).claim_timeout_offer(
            TimeoutOffer(player_seat=1, decision_index=32)
        )
    )

    assert clicked
    assert button.clicked


def test_parked_archive_claims_thirty_seconds_after_timeout_offer() -> None:
    if not PARKED_ARCHIVE.is_file():
        pytest.skip(f"missing parked timeout fixture: {PARKED_ARCHIVE}")
    events = parse_recording(PARKED_ARCHIVE).events
    clock = _FakeClock()
    session = _FakeSession(clock)
    offer_seen_at: list[float] = []

    async def parked_events() -> AsyncIterator[GameEvent]:
        for event in events:
            yield event
            if isinstance(event, TimeoutOffer):
                # Resume only after the loop has retained the offer and armed
                # its grace deadline.
                offer_seen_at.append(clock())
        await session.claimed.wait()

    results = asyncio.run(
        run_game_loop(
            parked_events(),
            actuator=ProtocolActuator(session.send_frame),
            decision_provider=RecordedDecisionProvider(events),
            timeout_offer_grace_seconds=30.0,
            clock=clock,
            sleep=clock.sleep,
            watchdog_poll_seconds=5.0,
        )
    )

    assert len(results) == 8
    assert len(offer_seen_at) == 1
    assert session.claimed_at is not None
    assert session.claimed_at - offer_seen_at[0] == pytest.approx(30.0)
    assert [
        frame
        for frame in session.sent
        if frame[:4] == TIMEOUT_REQUEST.to_bytes(4, "big")
    ] == [
        Writer()
        .u32(TIMEOUT_REQUEST)
        .bytes(encode_timeout_request(1))
        .build()
    ]
