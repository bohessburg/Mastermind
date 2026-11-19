"""Decide whether an opponent's undo request can be granted safely.

An undo to decision ``N`` rewinds the server to just before ``N`` was
answered, so the requester keeps everything they saw after it.  A grant is
safe only when nothing after that point showed anyone a hidden card or asked
us anything:

* no draw, reveal, shuffle, or deck reorder, and no card moved off a deck;
* no turn boundary (cleanup draws a new hand) or game state reload;
* every resolved decision in the range is the requester's own, and we were
  never asked a question or targeted by an attack.

Everything else, including a target decision that is not in the retained
history (for example from before an earlier resync), is denied.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


# Events that never move a card or show hidden information.
_NEUTRAL_EVENTS = frozenset(
    {
        "Buy",
        "Chat",
        "PileUpdate",
        "ResourceUpdate",
        "UndoRequest",
        "UndoResolved",
        "UnknownFrame",
    }
)
# Card moves that are safe when they stay on the requester's side and never
# leave a deck.
_CARD_MOVES = frozenset({"Discard", "Gain", "Play", "Topdeck", "Trash", "ZoneTransfer"})
# A set-aside card got there from a deck by an earlier (checked) move, so the
# requester already knows it when it leaves again, e.g. Sentry's topdecks.
_PUBLIC_SOURCES = frozenset({"discard", "hand", "in-play", "set-aside", "supply", "trash"})


@dataclass(frozen=True)
class UndoVerdict:
    grant: bool
    reason: str


def classify_undo_request(
    history: Sequence[Any],
    requester_seat: int,
    decision_index: int,
) -> UndoVerdict:
    """Return whether rewinding ``history`` to ``decision_index`` is safe.

    ``history`` is the game's events in arrival order, ending before the
    request.  Events are matched by class name so archived records can be
    classified with the same rule.
    """
    start = next(
        (
            position
            for position, event in enumerate(history)
            if type(event).__name__ == "DecisionResolved"
            and event.question_index == decision_index
        ),
        None,
    )
    if start is None:
        return UndoVerdict(False, f"decision {decision_index} is not in the retained history")
    for event in history[start:]:
        name = type(event).__name__
        if name in _NEUTRAL_EVENTS:
            continue
        if name == "DecisionResolved":
            if event.seat != requester_seat:
                return UndoVerdict(False, f"decision {event.question_index} belongs to seat {event.seat}")
            continue
        if name in _CARD_MOVES:
            if event.seat != requester_seat:
                return UndoVerdict(False, f"{name} by seat {event.seat}")
            if event.from_zone not in _PUBLIC_SOURCES:
                return UndoVerdict(False, f"{name} from {event.from_zone}")
            continue
        return UndoVerdict(False, f"{name} after decision {decision_index}")
    return UndoVerdict(True, f"{len(history) - start} events since decision {decision_index} reveal nothing")
