"""Stateless question actuation through the client's game WebSocket."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ..protocol.events import TimeoutOffer, UndoRequest
from ..protocol.messages import (
    ANSWER_QUESTION,
    TIMEOUT_REQUEST,
    encode_answer,
    encode_timeout_request,
)
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

    async def claim_timeout_offer(self, offer: TimeoutOffer) -> bool:
        """Use the bundle's stateless timeout request without touching the DOM."""
        try:
            await self.send_frame(
                TIMEOUT_REQUEST,
                encode_timeout_request(offer.player_seat),
            )
        except Exception as error:
            raise ActuationError(
                "failed timeout claim protocol send for opponent seat "
                f"{offer.player_seat} decision {offer.decision_index}; "
                f"detail={error}"
            ) from error
        return True

    async def inspect_unknown_modals(self) -> None:
        """Keep modal monitoring on the existing Playwright actuator."""
        if self.modal_actuator is not None:
            await self.modal_actuator.inspect_unknown_modals()
