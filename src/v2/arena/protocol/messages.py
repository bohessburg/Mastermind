"""Writers for stateless client-to-server arena messages."""

from __future__ import annotations

from collections.abc import Iterable

from .frames import Writer


ANSWER_QUESTION = 37
TIMEOUT_REQUEST = 40
UNDO_GRANT = 41
UNDO_DENY = 42


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


def encode_undo_reply(requester_seat: int, decision_index: int) -> bytes:
    """Encode the body of a client ``UNDO_GRANT`` or ``UNDO_DENY`` message.

    Every recorded reply is ``[requester-seat:s32][decision-index:s32]``: 208
    modal declines (message 42, both seats) and the one grant (message 41) in
    game 181368463.
    """
    return Writer().s32(requester_seat).s32(decision_index).build()


def encode_timeout_request(player_seat: int) -> bytes:
    """Encode client 2.2.8's force-end request for an eligible opponent.

    The client discards the offer's source decision and sends ``-1`` followed
    by the eligible player's seat.  This is the body used by the
    ``timeout-request`` modal's ``$ctrl.click()`` handler.
    """
    return Writer().s32(-1).s32(player_seat).build()
