from __future__ import annotations

from pathlib import Path

import pytest

from src.v2.arena.protocol.events import GameStart, UnknownFrame
from src.v2.arena.protocol.recording import newest_recording, parse_recording


EXPECTED_KINGDOM = (
    "Chapel",
    "Artisan",
    "Workshop",
    "Moat",
    "Market",
    "Remodel",
    "Sentry",
    "Poacher",
    "Moneylender",
    "Throne Room",
)

# P1's structural acceptance bar permits a small forward-compatibility margin
# while catching any field-boundary regression. The verified 2.2.8 fixture is
# expected to remain at 100% for inbound 32/33 and outbound 37.
MIN_RELEVANT_DECODE_COVERAGE = 0.995


def _recording_or_skip() -> Path:
    recording = newest_recording()
    if recording is None:
        pytest.skip("no arena-recordings/*/frames.jsonl fixture is available")
    return recording


def test_newest_recording_parses_both_sessions_and_expected_game() -> None:
    result = parse_recording(_recording_or_skip())

    assert len(result.sessions) == 2
    for session in result.sessions:
        inbound_sequences = [
            frame.sequence for frame in session.frames if frame.sequence is not None
        ]
        assert inbound_sequences[0] == 0

    starts = [event for event in result.events if isinstance(event, GameStart)]
    assert any(event.kingdom == EXPECTED_KINGDOM for event in starts)
    assert result.stats.relevant_total > 20_000
    assert result.stats.decode_coverage >= MIN_RELEVANT_DECODE_COVERAGE


def test_residual_unknown_frames_are_inspectable_and_not_relevant() -> None:
    result = parse_recording(_recording_or_skip())
    unknown = [
        event for event in result.events if isinstance(event, UnknownFrame)
    ]

    assert unknown
    assert result.unknown_msg_types
    assert all(isinstance(event.raw, bytes) and event.reason for event in unknown)
    assert not {
        (event.direction, event.msg_type)
        for event in unknown
    } & {("in", 32), ("in", 33), ("out", 37)}
