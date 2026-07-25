"""Stateless question actuation through the client's game WebSocket."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ..protocol.events import UndoRequest
from ..protocol.messages import ANSWER_QUESTION, encode_answer
from ..shadow.tracker import PendingDecisionSnapshot
from .clicks import (
    ActuationError,
    Actuator,
    ClientGesture,
    map_client_gesture,
)


FrameSender = Callable[[int, bytes], Awaitable[None]]


class ProtocolActuator(Actuator):
    """Send mapped ``ANSWER_QUESTION`` bodies without touching the game DOM."""

    def __init__(
        self,
        send_frame: FrameSender,
        *,
        modal_actuator: Actuator | None = None,
    ) -> None:
        self.send_frame = send_frame
        self.modal_actuator = modal_actuator
        self.gestures: list[ClientGesture] = []

    async def act(
        self,
        action: int,
        decision: PendingDecisionSnapshot,
        offered_elements: tuple[str, ...],
        *,
        prior_actions: tuple[int, ...] = (),
    ) -> ClientGesture:
        try:
            gesture = map_client_gesture(
                action,
                decision,
                offered_elements,
                prior_actions=prior_actions,
            )
            payload = encode_answer(
                decision.question_index,
                gesture.answer_indices,
                auto_played=False,
            )
            await self.send_frame(ANSWER_QUESTION, payload)
        except Exception as error:
            raise ActuationError(
                f"failed question {decision.question_index} protocol send; "
                f"offered={offered_elements!r}; detail={error}"
            ) from error
        self.gestures.append(gesture)
        return gesture

    async def deny_undo_request(self, request: UndoRequest) -> bool:
        """Keep the existing modal-click implementation for undo denial."""
        if self.modal_actuator is None:
            return False
        return await self.modal_actuator.deny_undo_request(request)

    async def inspect_unknown_modals(self) -> None:
        """Keep modal monitoring on the existing Playwright actuator."""
        if self.modal_actuator is not None:
            await self.modal_actuator.inspect_unknown_modals()
