from __future__ import annotations

import math

import pytest

from src.v2.arena.protocol.frames import (
    Direction,
    ProtocolError,
    Reader,
    Writer,
    decode_frame,
)


def test_reader_writer_round_trip_primitives_and_arrays() -> None:
    writer = Writer()
    writer.u8(255)
    writer.boolean(True)
    writer.u32(0)
    writer.u32(2**32 - 1)
    writer.s32(-(2**31))
    writer.s32(2**31 - 1)
    writer.u64(2**64 - 1)
    writer.f64(math.pi)
    writer.string("")
    writer.string("Dominion ♛")
    writer.array([1, -2, 3], writer.s32)

    reader = Reader(writer.build())
    assert reader.u8() == 255
    assert reader.boolean() is True
    assert reader.u32() == 0
    assert reader.u32() == 2**32 - 1
    assert reader.s32() == -(2**31)
    assert reader.s32() == 2**31 - 1
    assert reader.u64() == 2**64 - 1
    assert reader.f64() == math.pi
    assert reader.string() == ""
    assert reader.string() == "Dominion ♛"
    assert reader.s32_array() == (1, -2, 3)
    reader.finish()


def test_frame_envelopes_are_direction_aware() -> None:
    inbound_raw = Writer().u32(9).u32(32).bytes(b"payload").build()
    inbound = decode_frame(inbound_raw, Direction.INBOUND, timestamp_ms=123)
    assert inbound is not None
    assert inbound.sequence == 9
    assert inbound.msg_type == 32
    assert inbound.payload == b"payload"
    assert inbound.timestamp_ms == 123

    outbound_raw = Writer().u32(37).bytes(b"answer").build()
    outbound = decode_frame(outbound_raw, "out")
    assert outbound is not None
    assert outbound.sequence is None
    assert outbound.msg_type == 37
    assert outbound.payload == b"answer"


def test_empty_frame_is_a_safe_noop() -> None:
    assert decode_frame(b"", "in") is None
    assert decode_frame(b"", "out") is None


@pytest.mark.parametrize(
    "data,operation",
    [
        (b"\x00\x00\x00", lambda reader: reader.u32()),
        (b"\x00\x00\x00\x05abc", lambda reader: reader.string()),
        (b"\x02", lambda reader: reader.boolean()),
        (b"\x00\x00\x00\x02\x00", lambda reader: reader.u32_array()),
    ],
)
def test_reader_rejects_truncated_or_invalid_values(data: bytes, operation: object) -> None:
    with pytest.raises(ProtocolError):
        operation(Reader(data))  # type: ignore[operator]


def test_finish_rejects_trailing_bytes() -> None:
    reader = Reader(b"\x00")
    with pytest.raises(ProtocolError, match="trailing"):
        reader.finish()
