"""Binary protocol support for the dominion.games arena stream."""

from .events import GameEvent
from .frames import DecodedFrame, Direction, Reader, decode_frame
from .live import events_from_queue, events_from_raw_records
from .messages import (
    ANSWER_QUESTION,
    TIMEOUT_REQUEST,
    encode_answer,
    encode_timeout_request,
)
from .parser import ArenaParser

__all__ = [
    "ANSWER_QUESTION",
    "TIMEOUT_REQUEST",
    "ArenaParser",
    "DecodedFrame",
    "Direction",
    "GameEvent",
    "Reader",
    "decode_frame",
    "encode_answer",
    "encode_timeout_request",
    "events_from_queue",
    "events_from_raw_records",
]
