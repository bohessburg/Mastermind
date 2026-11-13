"""Direction-aware Dominion P1 frame envelopes and binary primitives."""

from __future__ import annotations

import struct
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar


class ProtocolError(ValueError):
    """A malformed or truncated protocol value."""


class Direction(str, Enum):
    INBOUND = "in"
    OUTBOUND = "out"


@dataclass(frozen=True)
class DecodedFrame:
    direction: Direction
    msg_type: int
    payload: bytes
    raw: bytes
    sequence: int | None = None
    timestamp_ms: int | None = None


T = TypeVar("T")


class Reader:
    """Bounds-checked, cursor-based reader for the protocol's big-endian data."""

    def __init__(self, data: bytes | bytearray | memoryview) -> None:
        self._data = memoryview(data)
        self.offset = 0

    @property
    def remaining(self) -> int:
        return len(self._data) - self.offset

    @property
    def at_end(self) -> bool:
        return self.remaining == 0

    def _take(self, size: int) -> memoryview:
        if size < 0:
            raise ProtocolError(f"negative read size {size}")
        end = self.offset + size
        if end > len(self._data):
            raise ProtocolError(
                f"truncated value at offset {self.offset}: "
                f"need {size} bytes, have {self.remaining}"
            )
        value = self._data[self.offset:end]
        self.offset = end
        return value

    def bytes(self, size: int) -> bytes:
        return self._take(size).tobytes()

    def u8(self) -> int:
        return self._take(1)[0]

    def boolean(self) -> bool:
        value = self.u8()
        if value not in (0, 1):
            raise ProtocolError(f"invalid boolean {value} at offset {self.offset - 1}")
        return bool(value)

    def u32(self) -> int:
        return struct.unpack(">I", self._take(4))[0]

    def s32(self) -> int:
        return struct.unpack(">i", self._take(4))[0]

    def u64(self) -> int:
        return struct.unpack(">Q", self._take(8))[0]

    def f64(self) -> float:
        return struct.unpack(">d", self._take(8))[0]

    def string(self) -> str:
        size = self.u32()
        try:
            return self.bytes(size).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProtocolError(
                f"invalid UTF-8 string ending at offset {self.offset}"
            ) from error

    def array(self, read_item: Callable[[], T]) -> tuple[T, ...]:
        count = self.u32()
        # Every observed array item consumes at least one byte. This guard
        # catches corrupt counts before a long loop, without constraining
        # legitimate empty-item parsers used by callers.
        if count > self.remaining + 1:
            raise ProtocolError(
                f"implausible array count {count} with {self.remaining} bytes remaining"
            )
        return tuple(read_item() for _ in range(count))

    def u32_array(self) -> tuple[int, ...]:
        return self.array(self.u32)

    def s32_array(self) -> tuple[int, ...]:
        return self.array(self.s32)

    def string_array(self) -> tuple[str, ...]:
        return self.array(self.string)

    def finish(self) -> None:
        if not self.at_end:
            raise ProtocolError(
                f"{self.remaining} trailing bytes at offset {self.offset}"
            )


class Writer:
    """Small counterpart used by primitive and fixture tests."""

    def __init__(self) -> None:
        self._parts: list[bytes] = []

    def u8(self, value: int) -> "Writer":
        self._parts.append(struct.pack(">B", value))
        return self

    def boolean(self, value: bool) -> "Writer":
        return self.u8(int(value))

    def u32(self, value: int) -> "Writer":
        self._parts.append(struct.pack(">I", value))
        return self

    def s32(self, value: int) -> "Writer":
        self._parts.append(struct.pack(">i", value))
        return self

    def u64(self, value: int) -> "Writer":
        self._parts.append(struct.pack(">Q", value))
        return self

    def f64(self, value: float) -> "Writer":
        self._parts.append(struct.pack(">d", value))
        return self

    def bytes(self, value: bytes) -> "Writer":
        self._parts.append(value)
        return self

    def string(self, value: str) -> "Writer":
        encoded = value.encode("utf-8")
        return self.u32(len(encoded)).bytes(encoded)

    def array(self, values: Iterable[T], write_item: Callable[[T], None]) -> "Writer":
        materialized = tuple(values)
        self.u32(len(materialized))
        for value in materialized:
            write_item(value)
        return self

    def build(self) -> bytes:
        return b"".join(self._parts)


def decode_frame(
    raw: bytes,
    direction: Direction | str,
    *,
    timestamp_ms: int | None = None,
) -> DecodedFrame | None:
    """Decode one WS binary body.

    Empty captures are legal recorder artifacts and return ``None``. Inbound
    bodies carry a sequence before the message type; outbound bodies do not.
    """
    direction = Direction(direction)
    if not raw:
        return None
    reader = Reader(raw)
    sequence = reader.u32() if direction is Direction.INBOUND else None
    msg_type = reader.u32()
    payload = reader.bytes(reader.remaining)
    return DecodedFrame(
        direction=direction,
        msg_type=msg_type,
        payload=payload,
        raw=raw,
        sequence=sequence,
        timestamp_ms=timestamp_ms,
    )
