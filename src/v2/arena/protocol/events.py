"""Stable, normalized events emitted by the arena protocol parser."""

from __future__ import annotations

from dataclasses import dataclass


ZoneKind = str


@dataclass(frozen=True, kw_only=True)
class GameEvent:
    timestamp_ms: int | None = None


@dataclass(frozen=True)
class SessionStart(GameEvent):
    session_index: int
    socket: int
    url: str


@dataclass(frozen=True)
class Reconnect(GameEvent):
    session_index: int
    previous_sequence: int | None


@dataclass(frozen=True)
class GameStart(GameEvent):
    game_id: int
    kingdom: tuple[str, ...]
    players: tuple[str, ...]
    player_ids: tuple[int, ...]
    our_seat: int | None


@dataclass(frozen=True, kw_only=True)
class FullStateZone:
    index: int
    kind: ZoneKind
    owner: int | None
    display_name: str | None
    contents: tuple[str, ...]
    anonymous_count: int


@dataclass(frozen=True, kw_only=True)
class FullStateCounter:
    index: int
    name: str
    owner: int | None
    value: int


@dataclass(frozen=True, kw_only=True)
class FullState(GameEvent):
    game_id: int
    replacement: bool
    card_counts: tuple[tuple[str, int], ...]
    zones: tuple[FullStateZone, ...]
    counters: tuple[FullStateCounter, ...]


@dataclass(frozen=True)
class TurnStart(GameEvent):
    seat: int
    turn_number: int
    turn_type: int
    controller_seat: int


@dataclass(frozen=True)
class Play(GameEvent):
    seat: int | None
    cards: tuple[str, ...]
    count: int
    from_zone: ZoneKind
    to_zone: ZoneKind
    from_zone_index: int
    to_zone_index: int


@dataclass(frozen=True)
class Buy(GameEvent):
    seat: int | None
    cards: tuple[str, ...]
    count: int
    from_zone: ZoneKind = "supply"
    to_zone: ZoneKind | None = None


@dataclass(frozen=True)
class Gain(GameEvent):
    seat: int | None
    cards: tuple[str, ...]
    count: int
    from_zone: ZoneKind
    to_zone: ZoneKind
    from_zone_index: int
    to_zone_index: int


@dataclass(frozen=True)
class Trash(GameEvent):
    seat: int | None
    cards: tuple[str, ...]
    count: int
    from_zone: ZoneKind
    to_zone: ZoneKind
    from_zone_index: int
    to_zone_index: int


@dataclass(frozen=True)
class Discard(GameEvent):
    seat: int | None
    cards: tuple[str, ...]
    count: int
    from_zone: ZoneKind
    to_zone: ZoneKind
    from_zone_index: int
    to_zone_index: int


@dataclass(frozen=True)
class Draw(GameEvent):
    seat: int | None
    count: int
    cards: tuple[str, ...]
    from_zone: ZoneKind
    to_zone: ZoneKind
    from_zone_index: int
    to_zone_index: int


@dataclass(frozen=True)
class Reveal(GameEvent):
    seat: int | None
    cards: tuple[str, ...]
    count: int
    from_zone: ZoneKind
    to_zone: ZoneKind
    from_zone_index: int
    to_zone_index: int


@dataclass(frozen=True, kw_only=True)
class Topdeck(GameEvent):
    seat: int | None
    cards: tuple[str, ...]
    count: int
    from_zone: ZoneKind
    to_zone: ZoneKind
    from_zone_index: int
    to_zone_index: int


@dataclass(frozen=True, kw_only=True)
class ZoneTransfer(GameEvent):
    seat: int | None
    movement: str
    cards: tuple[str, ...]
    count: int
    from_zone: ZoneKind
    to_zone: ZoneKind
    from_zone_index: int
    to_zone_index: int


@dataclass(frozen=True, kw_only=True)
class PileUpdate(GameEvent):
    zone_index: int
    zone: ZoneKind
    owner: int | None
    top_card: str | None


@dataclass(frozen=True, kw_only=True)
class PileReorder(GameEvent):
    zone_index: int
    zone: ZoneKind
    owner: int | None
    cards: tuple[str, ...]
    count: int
    anonymous_count: int


@dataclass(frozen=True)
class Shuffle(GameEvent):
    seat: int
    included_discard: bool


@dataclass(frozen=True)
class Attack(GameEvent):
    seat: int | None
    card: str
    targets: tuple[int, ...]


@dataclass(frozen=True)
class ReactionWindow(GameEvent):
    question_index: int
    seat: int | None
    offered: tuple[str, ...]


@dataclass(frozen=True)
class PendingDecision(GameEvent):
    question_index: int
    decision_type: str
    question_id: str
    offered: tuple[str, ...]
    minimum: int
    maximum: int
    association: str | None


@dataclass(frozen=True)
class DecisionResolved(GameEvent):
    question_index: int
    answers: tuple[int, ...]
    seat: int | None
    auto_played: bool


@dataclass(frozen=True, kw_only=True)
class UndoRequest(GameEvent):
    """One player asked to rewind to a prior decision."""

    requester_seat: int
    decision_index: int


@dataclass(frozen=True, kw_only=True)
class UndoResolved(GameEvent):
    """The pending undo was denied or cancelled without rewinding."""

    resolution: str
    actor_seat: int
    decision_index: int


@dataclass(frozen=True, kw_only=True)
class TimeoutOffer(GameEvent):
    """One player became eligible for a metagame timeout request."""

    player_seat: int
    decision_index: int


@dataclass(frozen=True, kw_only=True)
class UndoResync(GameEvent):
    """Runtime evidence that a post-undo FullState reseeded the tracker."""

    game_id: int
    requester_seat: int
    decision_index: int
    reason: str


@dataclass(frozen=True)
class ResourceUpdate(GameEvent):
    seat: int | None
    resource: str
    value: int
    counter_index: int


@dataclass(frozen=True)
class GameEnd(GameEvent):
    game_id: int | None
    reason: str


@dataclass(frozen=True)
class Chat(GameEvent):
    sender: str
    receiver: str
    message: str
    outbound: bool


@dataclass(frozen=True)
class UnknownFrame(GameEvent):
    msg_type: int
    direction: str
    raw: bytes
    sequence: int | None
    reason: str = ""
