"""Writers for stateless client-to-server arena messages."""

from __future__ import annotations

from collections.abc import Iterable

from .frames import Writer


ANSWER_QUESTION = 37


def encode_answer(
    question_index: int,
    answers: Iterable[int],
    *,
    auto_played: bool = False,
) -> bytes:
    """Encode the payload of one client ``ANSWER_QUESTION`` message."""
    writer = Writer().s32(question_index)
    writer.array(tuple(answers), writer.s32)
    writer.boolean(auto_played)
    return writer.build()
