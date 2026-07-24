"""Stateful conversion from decoded Dominion frames to normalized events."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .cards import card_name
from .events import (
    Attack,
    Buy,
    Chat,
    DecisionResolved,
    Discard,
    Draw,
    FullState,
    FullStateCounter,
    FullStateZone,
    Gain,
    GameEnd,
    GameEvent,
    GameStart,
    PendingDecision,
    PileReorder,
    PileUpdate,
    Play,
    ReactionWindow,
    ResourceUpdate,
    Reveal,
    Shuffle,
    Topdeck,
    Trash,
    TurnStart,
    UnknownFrame,
    ZoneTransfer,
)
from .frames import DecodedFrame, Direction, ProtocolError, Reader


MOVEMENT_TYPES = (
    "PLAY",
    "DRAW",
    "DISCARD",
    "GAIN",
    "TRASH",
    "TOPDECK",
    "REVEAL",
    "REVEAL_SEARCHING",
    "LOOK_AT",
    "PUT_IN_HAND",
    "SET_ASIDE",
    "PUT_ON_MAT",
    "DECK_TO_DISCARD",
    "BACK_ON_DECK",
    "SHUFFLE_INTO_DECK",
    "INSERT_IN_DECK",
    "EXCHANGE_RETURN",
    "EXCHANGE_RECEIVE",
    "PASS",
    "RETURN_TO",
    "PUT_ON_BOTTOM_OF",
    "STARTS_WITH",
    "TAKE",
    "RETURN",
    "EXILE",
)

QUESTION_TYPES = (
    "PLAY",
    "GAIN",
    "DISCARD",
    "TRASH",
    "TRASH_UNIQUE",
    "SELECT_ZONE",
    "SELECT_TURN",
    "RESOLVE_ABILITY",
    "ORDER_CARDS",
    "BUY",
    "CHOOSE_MODE",
    "HOW_MUCH_REPAY",
    "HOW_MANY_SPEND",
    "TOPDECK",
    "BOTTOMDECK",
    "SET_ASIDE",
    "REVEAL",
    "PASS",
    "COMPLEX_OR",
    "COMPLEX_AND",
    "WISH",
    "DELAYED",
    "PUT_IN_HAND",
    "KEEP_IN_DISCARD",
    "BID",
    "OVERPAY",
    "SHUFFLE_INTO_DECK",
    "INSERT_INTO_DECK",
    "LOCATION",
    "DONATE",
    "TAKE_HOW_MANY",
    "RETURN_TO_SUPPLY",
    "GAIN_PRIZE",
    "PRIORITIZE_CARDS",
    "BUTTON",
    "CLEANUP",
    "IMPERSONATE",
    "RECEIVE",
    "START_GAME",
    "PLAY_USING_VILLAGER",
    "EXILE",
    "USE_VILLAGERS",
    "EXILE_WITH_SAME_NAME",
    "WAY",
    "HOW_MANY_FAVORS",
    "ROTATE",
    "KEEP",
    "OVERPAY_COINS",
    "OVERPAY_POTIONS",
    "REJECT_ABILITY",
    "REVEAL_SELECT",
)

# Base questions occupy the beginning of the bundle's 733-entry QuestionIds
# table. Unknown later ordinals remain losslessly inspectable as question-N.
QUESTION_IDS = (
    "GAME_RESOLVE_ABILITY",
    "GAME_MAY_RESOLVE_ABILITY",
    "GAME_MAY_REACT_WITH",
    "GAME_TURN_SELECT",
    "GAME_MAY_BUY",
    "GAME_MAY_START_BUYING",
    "GAME_MAY_BUY_NONCARDS",
    "GAME_MAY_START_BUYING_NONCARDS",
    "GAME_REPAY_HOW_MUCH",
    "GAME_SPEND_COIN_TOKENS",
    "GAME_MAY_PLAY_ACTION",
    "GAME_MAY_PLAY_TREASURES",
    "GAME_PLAY_ALL_TREASURES",
    "GAME_ACTION_PHASE",
    "GAME_BUY_PHASE",
    "GAME_BUY_PHASE_NONCARDS",
    "GAME_CLEANUP_PHASE",
    "GAME_USE_VILLAGER",
    "GAME_USE_COFFERS",
    "GAME_PAY_OFF_DEBT",
    "ARTISAN_GAIN",
    "ARTISAN_TOPDECK",
    "BANDIT",
    "BUREAUCRAT",
    "CELLAR",
    "CHAPEL",
    "HARBINGER_TOPDECK",
    "LIBRARY",
    "MILITIA",
    "MINE_TRASH",
    "MINE_GAIN",
    "MONEYLENDER",
    "POACHER",
    "REMODEL_TRASH",
    "REMODEL_GAIN",
    "SENTRY_TRASH",
    "SENTRY_DISCARD",
    "SENTRY_TOPDECK",
    "THRONE_ROOM",
    "VASSAL",
    "WORKSHOP",
)

GAME_BUTTONS = (
    "AUTOPLAY_TREASURES",
    "USE_VILLAGER",
    "USE_FAVOR_TOPDECK",
    "USE_FAVOR_DISCARD",
    "USE_COFFERS",
    "PAY_OFF_DEBT",
    "SELECT_ALL",
    "FORCE_END_TURN",
)

COUNTER_NAMES = (
    "actions",
    "coins",
    "potions",
    "buys",
    "points",
    "coffers",
    "debt",
    "vp_tokens",
    "pirate_strikes",
    "embargo_tokens",
    "trade_route_coins",
    "trade_route_mat",
    "villagers",
    "sinister_plot",
    "blockade_tokens",
    "favors",
    "card_tokens",
    "remaining_hand_plays",
    "sun_tokens",
)

ATTACK_CARDS = frozenset({"Bandit", "Bureaucrat", "Militia", "Witch"})
BASE_NON_KINGDOM_IDS = frozenset(range(1, 8))
ZONE_KINDS = {
    0: "hand",
    1: "deck",
    2: "discard",
    3: "trash",
    4: "supply",
    5: "supply",
    6: "supply",
    8: "in-play",
    23: "set-aside",
}
RELEVANT_KEYS = frozenset(
    {
        (Direction.INBOUND, 32),
        (Direction.INBOUND, 33),
        (Direction.OUTBOUND, 37),
    }
)


def _enum_name(names: tuple[str, ...], ordinal: int, prefix: str) -> str:
    if 0 <= ordinal < len(names):
        return names[ordinal]
    return f"{prefix}-{ordinal}"


@dataclass
class ParseStats:
    frame_counts: dict[tuple[str, int], int] = field(default_factory=dict)
    decoded_counts: dict[tuple[str, int], int] = field(default_factory=dict)
    unknown_counts: dict[tuple[str, int], int] = field(default_factory=dict)
    empty_frames: int = 0

    @property
    def relevant_total(self) -> int:
        return sum(
            self.frame_counts.get((direction.value, msg_type), 0)
            for direction, msg_type in RELEVANT_KEYS
        )

    @property
    def relevant_decoded(self) -> int:
        return sum(
            self.decoded_counts.get((direction.value, msg_type), 0)
            for direction, msg_type in RELEVANT_KEYS
        )

    @property
    def decode_coverage(self) -> float:
        if not self.relevant_total:
            return 1.0
        return self.relevant_decoded / self.relevant_total


@dataclass(frozen=True)
class _Question:
    question_type: str
    question_id: str
    association: int
    offered: tuple[str, ...]
    minimum: int
    maximum: int


class ArenaParser:
    """Parse one or more consecutive sock-3 sessions.

    State deliberately survives session boundaries so a reconnect can resume
    the same game's instance-card and zone mappings.
    """

    def __init__(self) -> None:
        self.stats = ParseStats()
        self.our_player_id: int | None = None
        self.our_seat: int | None = None
        self.game_id: int | None = None
        self.player_ids: tuple[int, ...] = ()
        self.player_names_by_id: dict[int, str] = {}
        self.card_by_instance: dict[int, str] = {}
        self.zone_owner: dict[int, int] = {}
        self.zone_type: dict[int, int] = {}
        self.zone_display: dict[int, str | None] = {}
        self.counter_info: dict[int, tuple[int | None, str]] = {}

    def parse_frames(self, frames: Iterable[DecodedFrame]) -> tuple[GameEvent, ...]:
        """Parse an ordered session (or consecutive sessions) in one call."""
        events: list[GameEvent] = []
        for frame in frames:
            events.extend(self.parse_frame(frame))
        return tuple(events)

    def parse_frame(self, frame: DecodedFrame) -> list[GameEvent]:
        key = (frame.direction.value, frame.msg_type)
        self.stats.frame_counts[key] = self.stats.frame_counts.get(key, 0) + 1
        try:
            events = self._parse_frame(frame)
        except (ProtocolError, IndexError, UnicodeDecodeError, ValueError) as error:
            self.stats.unknown_counts[key] = self.stats.unknown_counts.get(key, 0) + 1
            return [
                UnknownFrame(
                    msg_type=frame.msg_type,
                    direction=frame.direction.value,
                    raw=frame.payload,
                    sequence=frame.sequence,
                    reason=str(error),
                    timestamp_ms=frame.timestamp_ms,
                )
            ]

        self.stats.decoded_counts[key] = self.stats.decoded_counts.get(key, 0) + 1
        return events

    def _parse_frame(self, frame: DecodedFrame) -> list[GameEvent]:
        if frame.direction is Direction.INBOUND:
            if frame.msg_type == 0:
                return [self._parse_chat(frame, outbound=False)]
            if frame.msg_type == 2:
                return self._parse_login_success(frame)
            if frame.msg_type == 10:
                self._parse_table_details_prefix(frame.payload)
                return []
            if frame.msg_type == 14:
                return [
                    GameEnd(
                        game_id=self.game_id,
                        reason="game-finished",
                        timestamp_ms=frame.timestamp_ms,
                    )
                ]
            if frame.msg_type == 32:
                return self._parse_game_event(frame)
            if frame.msg_type == 33:
                return self._parse_log(frame)
            if frame.msg_type == 34:
                self._parse_ticker(frame.payload)
                return []
            if frame.msg_type == 36:
                Reader(frame.payload).finish()
                return []
            if frame.msg_type == 37:
                return self._parse_question(frame)
            if frame.msg_type == 38:
                return self._parse_full_state(frame)
            if frame.msg_type == 41:
                self._parse_timer_update(frame.payload)
                return []
            if frame.msg_type == 47 and frame.payload == b"\x00\x00\x00\x00":
                return []
        else:
            if frame.msg_type == 0:
                return [self._parse_chat(frame, outbound=True)]
            if frame.msg_type == 37:
                return [self._parse_answer(frame)]
            if frame.msg_type == 44:
                Reader(frame.payload).finish()
                return []

        self.stats.unknown_counts[(frame.direction.value, frame.msg_type)] = (
            self.stats.unknown_counts.get((frame.direction.value, frame.msg_type), 0) + 1
        )
        return [
            UnknownFrame(
                msg_type=frame.msg_type,
                direction=frame.direction.value,
                raw=frame.payload,
                sequence=frame.sequence,
                reason="unhandled message type",
                timestamp_ms=frame.timestamp_ms,
            )
        ]

    def _parse_login_success(self, frame: DecodedFrame) -> list[GameEvent]:
        reader = Reader(frame.payload)
        self.our_player_id = reader.u32()
        reader.string()
        if self.game_id is None:
            return []

        # A reconnect login embeds the current fullGameState as a queued
        # message: [msgType=38][full-state payload].  The rest of login-success
        # is lobby state and remains deliberately prefix-decoded.
        game_id_bytes = self.game_id.to_bytes(8, "big")
        for wire_flag in (b"\x00", b"\x01"):
            marker = (38).to_bytes(4, "big") + wire_flag + game_id_bytes
            offset = frame.payload.find(marker, reader.offset)
            if offset >= 0:
                return self._parse_full_state(
                    frame,
                    payload=frame.payload[offset + 4 :],
                    replacement_override=True,
                )
        return []

    def _parse_table_details_prefix(self, payload: bytes) -> None:
        reader = Reader(payload)
        reader.u64()
        reader.s32()
        players = reader.array(lambda: (reader.s32(), reader.string()))
        self.player_names_by_id.update(players)

    def _parse_chat(self, frame: DecodedFrame, *, outbound: bool) -> Chat:
        reader = Reader(frame.payload)
        receiver = reader.string()
        sender = reader.string()
        message = reader.string()
        reader.finish()
        return Chat(
            sender=sender,
            receiver=receiver,
            message=message,
            outbound=outbound,
            timestamp_ms=frame.timestamp_ms,
        )

    def _parse_full_state(
        self,
        frame: DecodedFrame,
        *,
        payload: bytes | None = None,
        replacement_override: bool = False,
    ) -> list[GameEvent]:
        reader = Reader(frame.payload if payload is None else payload)
        replacement = reader.boolean() or replacement_override
        game_id = reader.u64()

        player_ids: list[int] = []
        for _ in range(reader.u32()):
            player_ids.append(reader.s32())
            reader.array(lambda: (reader.u32(), reader.s32()))
            reader.array(lambda: (reader.u32(), reader.boolean(), reader.s32()))
            reader.u32_array()

        state_card_names = reader.u32_array()
        altered_names = reader.u32_array()
        card_counts = reader.u32_array()
        if not (
            len(state_card_names) == len(altered_names) == len(card_counts)
        ):
            raise ProtocolError("full-state card arrays have different lengths")

        card_by_instance: dict[int, str] = {}
        instance_id = 0
        for wire_id, count in zip(state_card_names, card_counts, strict=True):
            name = card_name(wire_id)
            for _ in range(count):
                card_by_instance[instance_id] = name
                instance_id += 1

        zone_indices = reader.s32_array()
        zone_types = reader.u32_array()
        zone_display_names = reader.u32_array()
        zone_owners = reader.s32_array()
        zone_count = reader.u32()
        if not (
            len(zone_indices)
            == len(zone_types)
            == len(zone_display_names)
            == len(zone_owners)
            == zone_count
        ):
            raise ProtocolError("full-state zone arrays have different lengths")
        zone_contents = tuple(reader.s32_array() for _ in range(zone_count))
        reader.s32_array()  # creating card
        reader.u32_array()  # anonymous-card face
        reader.array(reader.boolean)

        # Pile markers.
        reader.array(lambda: (reader.u32(), reader.s32()))
        # Token name, owner, zone, and flipped arrays.
        reader.u32_array()
        reader.s32_array()
        reader.s32_array()
        reader.array(reader.boolean)

        counter_indices = reader.s32_array()
        counter_names = reader.u32_array()
        counter_owners = reader.s32_array()
        counter_values = reader.s32_array()
        reader.s32_array()  # associated zones
        reader.s32_array()  # associated cards
        reader.array(reader.boolean)
        if not (
            len(counter_indices)
            == len(counter_names)
            == len(counter_owners)
            == len(counter_values)
        ):
            raise ProtocolError("full-state counter arrays have different lengths")

        self.game_id = game_id
        self.player_ids = tuple(player_ids)
        self.our_seat = (
            player_ids.index(self.our_player_id)
            if self.our_player_id in player_ids
            else None
        )
        self.card_by_instance = card_by_instance
        self.zone_owner = dict(zip(zone_indices, zone_owners, strict=True))
        self.zone_type = dict(zip(zone_indices, zone_types, strict=True))
        self.zone_display = {
            index: (
                None
                if card_name(display_name) == "Back"
                else card_name(display_name)
            )
            for index, display_name in zip(
                zone_indices, zone_display_names, strict=True
            )
        }
        self.counter_info = {
            index: (
                owner if owner >= 0 else None,
                _enum_name(COUNTER_NAMES, name, "counter"),
            )
            for index, name, owner in zip(
                counter_indices, counter_names, counter_owners, strict=True
            )
        }

        kingdom = tuple(
            card_name(wire_id)
            for wire_id in state_card_names
            if wire_id not in BASE_NON_KINGDOM_IDS and wire_id != 0
        )
        players = tuple(
            self.player_names_by_id.get(player_id, str(player_id))
            for player_id in player_ids
        )
        game_start = GameStart(
            game_id=game_id,
            kingdom=kingdom,
            players=players,
            player_ids=tuple(player_ids),
            our_seat=self.our_seat,
            timestamp_ms=frame.timestamp_ms,
        )
        zones = tuple(
            FullStateZone(
                index=index,
                kind=self._zone_kind(index),
                owner=owner if owner >= 0 else None,
                display_name=self.zone_display[index],
                contents=self._instance_names(contents),
                anonymous_count=sum(
                    instance_id < 0 or instance_id not in card_by_instance
                    for instance_id in contents
                ),
            )
            for index, owner, contents in zip(
                zone_indices, zone_owners, zone_contents, strict=True
            )
        )
        counters = tuple(
            FullStateCounter(
                index=index,
                name=_enum_name(COUNTER_NAMES, name, "counter"),
                owner=owner if owner >= 0 else None,
                value=value,
            )
            for index, name, owner, value in zip(
                counter_indices,
                counter_names,
                counter_owners,
                counter_values,
                strict=True,
            )
        )
        full_state = FullState(
            game_id=game_id,
            replacement=replacement,
            card_counts=tuple(
                (card_name(wire_id), count)
                for wire_id, count in zip(
                    state_card_names, card_counts, strict=True
                )
            ),
            zones=zones,
            counters=counters,
            timestamp_ms=frame.timestamp_ms,
        )
        return [game_start, full_state]

    def _parse_game_event(self, frame: DecodedFrame) -> list[GameEvent]:
        reader = Reader(frame.payload)
        subtype = reader.u32()
        timestamp = frame.timestamp_ms

        if subtype == 0:
            from_zone = reader.s32()
            to_zone = reader.s32()
            card_ids = reader.s32_array()
            visible_ids = reader.s32_array()
            movement = reader.u32()
            reader.u32()  # animation class
            reader.finish()
            seat = self._movement_seat(movement, from_zone, to_zone)
            count = max(len(card_ids), len(visible_ids))
            cards = self._movement_cards(
                movement=movement,
                seat=seat,
                from_zone=from_zone,
                card_ids=card_ids,
                visible_ids=visible_ids,
                count=count,
            )
            from_kind = self._zone_kind(from_zone)
            to_kind = self._zone_kind(to_zone)
            movement_name = _enum_name(MOVEMENT_TYPES, movement, "movement")
            if movement_name == "PLAY":
                events: list[GameEvent] = [
                    Play(
                        seat=seat,
                        cards=cards,
                        count=count,
                        from_zone=from_kind,
                        to_zone=to_kind,
                        from_zone_index=from_zone,
                        to_zone_index=to_zone,
                        timestamp_ms=timestamp,
                    )
                ]
                for attack_card in (card for card in cards if card in ATTACK_CARDS):
                    targets = tuple(
                        index
                        for index in range(len(self.player_ids))
                        if index != seat
                    )
                    events.append(
                        Attack(
                            seat=seat,
                            card=attack_card,
                            targets=targets,
                            timestamp_ms=timestamp,
                        )
                    )
                return events
            if movement_name == "DRAW":
                private_cards = cards if seat == self.our_seat else ()
                return [
                    Draw(
                        seat=seat,
                        count=count,
                        cards=private_cards,
                        from_zone=from_kind,
                        to_zone=to_kind,
                        from_zone_index=from_zone,
                        to_zone_index=to_zone,
                        timestamp_ms=timestamp,
                    )
                ]
            if movement_name == "DISCARD":
                return [
                    Discard(
                        seat=seat,
                        cards=cards,
                        count=count,
                        from_zone=from_kind,
                        to_zone=to_kind,
                        from_zone_index=from_zone,
                        to_zone_index=to_zone,
                        timestamp_ms=timestamp,
                    )
                ]
            if movement_name == "GAIN":
                return [
                    Gain(
                        seat=seat,
                        cards=cards,
                        count=count,
                        from_zone=from_kind,
                        to_zone=to_kind,
                        from_zone_index=from_zone,
                        to_zone_index=to_zone,
                        timestamp_ms=timestamp,
                    )
                ]
            if movement_name == "TRASH":
                return [
                    Trash(
                        seat=seat,
                        cards=cards,
                        count=count,
                        from_zone=from_kind,
                        to_zone=to_kind,
                        from_zone_index=from_zone,
                        to_zone_index=to_zone,
                        timestamp_ms=timestamp,
                    )
                ]
            if movement_name in {"REVEAL", "REVEAL_SEARCHING", "LOOK_AT"}:
                return [
                    Reveal(
                        seat=seat,
                        cards=cards,
                        count=count,
                        from_zone=from_kind,
                        to_zone=to_kind,
                        from_zone_index=from_zone,
                        to_zone_index=to_zone,
                        timestamp_ms=timestamp,
                    )
                ]
            if movement_name == "TOPDECK":
                return [
                    Topdeck(
                        seat=seat,
                        cards=cards,
                        count=count,
                        from_zone=from_kind,
                        to_zone=to_kind,
                        from_zone_index=from_zone,
                        to_zone_index=to_zone,
                        timestamp_ms=timestamp,
                    )
                ]
            return [
                ZoneTransfer(
                    seat=seat,
                    movement=movement_name,
                    cards=cards,
                    count=count,
                    from_zone=from_kind,
                    to_zone=to_kind,
                    from_zone_index=from_zone,
                    to_zone_index=to_zone,
                    timestamp_ms=timestamp,
                )
            ]

        if subtype == 1:
            counter_index = reader.s32()
            value = reader.s32()
            reader.finish()
            seat, resource = self.counter_info.get(
                counter_index, (None, f"counter-{counter_index}")
            )
            return [
                ResourceUpdate(
                    seat=seat,
                    resource=resource,
                    value=value,
                    counter_index=counter_index,
                    timestamp_ms=timestamp,
                )
            ]

        if subtype == 2:
            zone_index = reader.s32()
            top_card_id = reader.s32()
            reader.finish()
            top_card = self.card_by_instance.get(top_card_id)
            if top_card is None and self._zone_kind(zone_index) == "supply":
                top_card = self.zone_display.get(zone_index)
            owner = self.zone_owner.get(zone_index, -1)
            return [
                PileUpdate(
                    zone_index=zone_index,
                    zone=self._zone_kind(zone_index),
                    owner=owner if owner >= 0 else None,
                    top_card=top_card,
                    timestamp_ms=timestamp,
                )
            ]

        if subtype == 3:
            owner = reader.s32()
            turn_number = reader.s32()
            turn_type = reader.s32()
            controller = reader.s32()
            reader.finish()
            return [
                TurnStart(
                    seat=owner,
                    turn_number=turn_number,
                    turn_type=turn_type,
                    controller_seat=controller,
                    timestamp_ms=timestamp,
                )
            ]

        if subtype == 4:
            owner = reader.s32()
            include_discard = reader.boolean()
            reader.finish()
            return [
                Shuffle(
                    seat=owner,
                    included_discard=include_discard,
                    timestamp_ms=timestamp,
                )
            ]

        if subtype == 8:
            zone_index = reader.s32()
            owner = reader.s32()
            zone_type = reader.u32()
            reader.s32()
            reader.finish()
            self.zone_owner[zone_index] = owner
            self.zone_type[zone_index] = zone_type
            self.zone_display[zone_index] = None
            return []

        if subtype == 19:
            zone_index = reader.s32()
            card_ids = reader.s32_array()
            reader.finish()
            if zone_index not in self.zone_owner:
                self.zone_owner[zone_index] = -1
            owner = self.zone_owner[zone_index]
            public = owner == self.our_seat or self._zone_kind(zone_index) in {
                "supply",
                "trash",
                "discard",
                "in-play",
            }
            cards = self._instance_names(card_ids) if public else ()
            return [
                PileReorder(
                    zone_index=zone_index,
                    zone=self._zone_kind(zone_index),
                    owner=owner if owner >= 0 else None,
                    cards=cards,
                    count=len(card_ids),
                    anonymous_count=len(card_ids) - len(cards),
                    timestamp_ms=timestamp,
                )
            ]

        raise ProtocolError(f"unknown game-event subtype {subtype}")

    def _movement_seat(
        self, movement: int, from_zone: int, to_zone: int
    ) -> int | None:
        if movement in (0, 2, 4):
            owner = self.zone_owner.get(from_zone, -1)
        else:
            owner = self.zone_owner.get(to_zone, -1)
        return owner if owner >= 0 else None

    def _zone_kind(self, zone_index: int) -> str:
        zone_type = self.zone_type.get(zone_index)
        if zone_type is None:
            return f"zone-{zone_index}"
        return ZONE_KINDS.get(zone_type, f"zone-type-{zone_type}")

    def _movement_cards(
        self,
        *,
        movement: int,
        seat: int | None,
        from_zone: int,
        card_ids: tuple[int, ...],
        visible_ids: tuple[int, ...],
        count: int,
    ) -> tuple[str, ...]:
        if movement == 1 and seat != self.our_seat:
            return ()
        if seat == self.our_seat:
            cards = self._instance_names(card_ids)
        else:
            cards = self._instance_names(visible_ids)

        if (
            movement == 3
            and self._zone_kind(from_zone) == "supply"
            and len(cards) < count
        ):
            pile_name = self.zone_display.get(from_zone)
            if pile_name is not None:
                cards = cards + (pile_name,) * (count - len(cards))
        return cards

    def _instance_names(self, instance_ids: tuple[int, ...]) -> tuple[str, ...]:
        return tuple(
            self.card_by_instance[instance_id]
            for instance_id in instance_ids
            if instance_id >= 0 and instance_id in self.card_by_instance
        )

    def _parse_log(self, frame: DecodedFrame) -> list[GameEvent]:
        reader = Reader(frame.payload)
        start_index = reader.u32()
        entry_count = reader.u32()
        events: list[GameEvent] = []
        for offset in range(entry_count):
            entry_type = reader.u32()
            if entry_type == 0:
                name = reader.u32()
                reader.s32()  # depth
                arguments = reader.array(lambda: self._read_log_argument(reader))
                if name == 30:  # BUY
                    seat = self._argument_player(arguments)
                    cards = self._argument_cards(arguments)
                    events.append(
                        Buy(
                            seat=seat,
                            cards=cards,
                            count=len(cards),
                            timestamp_ms=frame.timestamp_ms,
                        )
                    )
            elif entry_type == 1:
                decision_index = reader.s32()
                seat = reader.s32()
                answers = reader.s32_array()
                auto_played = reader.boolean()
                events.append(
                    DecisionResolved(
                        question_index=decision_index,
                        answers=answers,
                        seat=seat,
                        auto_played=auto_played,
                        timestamp_ms=frame.timestamp_ms,
                    )
                )
            else:
                raise ProtocolError(
                    f"unknown log entry type {entry_type} at {start_index + offset}"
                )
        reader.finish()
        return events

    def _parse_ticker(self, payload: bytes) -> None:
        reader = Reader(payload)
        entry_type = reader.u32()
        if entry_type == 0:
            reader.u32()  # log name
            reader.s32()  # depth
            reader.array(lambda: self._read_log_argument(reader))
        elif entry_type == 1:
            reader.s32()
            reader.s32()
            reader.s32_array()
            reader.boolean()
        else:
            raise ProtocolError(f"unknown ticker entry type {entry_type}")
        reader.boolean()  # start-game marker
        reader.finish()

    def _parse_timer_update(self, payload: bytes) -> None:
        reader = Reader(payload)
        for _ in range(reader.u32()):
            reader.s32()
            reader.u64()
            reader.boolean()
            reader.u64()
            reader.u64()
        reader.finish()

    def _read_log_argument(self, reader: Reader) -> tuple[int, object]:
        argument_type = reader.u32()
        if argument_type == 0:  # CARD_NAMES
            value = reader.array(lambda: (reader.u32(), reader.s32()))
        elif argument_type in (1, 2, 4, 5, 10, 11, 12):
            value = reader.s32()
        elif argument_type == 3:  # OWN_ZONE
            value = (reader.s32(), reader.s32())
        elif argument_type == 6:  # TURN_DESCRIPTION
            value = tuple(reader.s32() for _ in range(4))
        elif argument_type == 7:  # COST
            value = tuple(reader.s32() for _ in range(3))
        elif argument_type == 13:  # DIRECTIONAL_ZONE
            value = (reader.s32(), reader.s32())
        elif argument_type == 14:  # PLAYERS
            value = reader.s32_array()
        elif argument_type == 15:  # METAGAME_INFO
            value = self._read_metagame_info(reader)
        else:
            raise ProtocolError(f"unknown log argument type {argument_type}")
        return argument_type, value

    def _read_metagame_info(self, reader: Reader) -> tuple[object, ...]:
        game_id = reader.u64()
        rated = reader.boolean()
        card_pool_level = reader.s32()
        level_map = reader.array(lambda: (reader.s32(), reader.f64()))
        changed = reader.array(lambda: (reader.s32(), reader.u32_array()))
        timer_preset = reader.s32()
        return game_id, rated, card_pool_level, level_map, changed, timer_preset

    def _argument_player(
        self, arguments: tuple[tuple[int, object], ...]
    ) -> int | None:
        for argument_type, value in arguments:
            if argument_type == 1:
                return int(value)
        return None

    def _argument_cards(
        self, arguments: tuple[tuple[int, object], ...]
    ) -> tuple[str, ...]:
        cards: list[str] = []
        for argument_type, value in arguments:
            if argument_type != 0:
                continue
            for wire_id, frequency in value:  # type: ignore[union-attr]
                cards.extend([card_name(wire_id)] * frequency)
        return tuple(cards)

    def _parse_question(self, frame: DecodedFrame) -> list[GameEvent]:
        reader = Reader(frame.payload)
        question_index = reader.s32()
        question_class = reader.u32()
        question = self._read_question(reader, question_class)
        reader.finish()
        association = self.card_by_instance.get(question.association)
        pending = PendingDecision(
            question_index=question_index,
            decision_type=question.question_type,
            question_id=question.question_id,
            offered=question.offered,
            minimum=question.minimum,
            maximum=question.maximum,
            association=association,
            timestamp_ms=frame.timestamp_ms,
        )
        events: list[GameEvent] = [pending]
        if question.question_id == "GAME_MAY_REACT_WITH":
            events.append(
                ReactionWindow(
                    question_index=question_index,
                    seat=self.our_seat,
                    offered=question.offered,
                    timestamp_ms=frame.timestamp_ms,
                )
            )
        return events

    def _read_question(self, reader: Reader, question_class: int) -> _Question:
        if question_class == 0:
            question_type, association, question_id = self._read_description(reader)
            minimum = reader.s32()
            maximum = reader.s32()
            offered = self._read_question_elements(reader)
            reader.s32()  # decline button
            reader.s32_array()  # accumulated answers
            reader.s32_array()  # affected cards
            return _Question(
                question_type=question_type,
                question_id=question_id,
                association=association,
                offered=offered,
                minimum=minimum,
                maximum=maximum,
            )

        if question_class == 1:
            question_type, association, question_id = self._read_description(reader)
            minimum = reader.s32()
            maximum = reader.s32()
            reader.s32()  # default
            reader.s32()  # decline button
            offered = tuple(str(value) for value in range(minimum, maximum + 1))
            return _Question(
                question_type=question_type,
                question_id=question_id,
                association=association,
                offered=offered,
                minimum=minimum,
                maximum=maximum,
            )

        if question_class == 2:
            question_type, association, question_id = self._read_description(reader)
            subquestions = tuple(
                self._read_question(reader, reader.u32())
                for _ in range(reader.u32())
            )
            offered = tuple(
                f"{index}:{choice}"
                for index, subquestion in enumerate(subquestions)
                for choice in subquestion.offered
            )
            return _Question(
                question_type=question_type,
                question_id=question_id,
                association=association,
                offered=offered,
                minimum=0,
                maximum=len(subquestions),
            )

        if question_class == 3:
            question_type, association, question_id = self._read_description(reader)
            reader.s32()
            return _Question(
                question_type=question_type,
                question_id=question_id,
                association=association,
                offered=("accept", "decline"),
                minimum=0,
                maximum=1,
            )

        if question_class == 4:
            question_type, association, question_id = self._read_description(reader)
            return _Question(
                question_type=question_type,
                question_id=question_id,
                association=association,
                offered=(),
                minimum=1,
                maximum=1,
            )

        raise ProtocolError(f"unknown question class {question_class}")

    def _read_description(self, reader: Reader) -> tuple[str, int, str]:
        question_type = _enum_name(
            QUESTION_TYPES, reader.u32(), "question-type"
        )
        association = reader.s32()
        question_id = self._read_story(reader)
        self._read_story(reader)
        return question_type, association, question_id

    def _read_story(self, reader: Reader) -> str:
        question_id_ordinal = reader.u32()
        reader.array(lambda: self._read_log_argument(reader))
        if reader.boolean():
            self._read_log_argument(reader)
        return _enum_name(QUESTION_IDS, question_id_ordinal, "question")

    def _read_question_elements(self, reader: Reader) -> tuple[str, ...]:
        count = reader.u32()
        if not count:
            return ()
        element_type = reader.u32()
        offered: list[str] = []
        if element_type in (0, 3):
            for _ in range(count):
                value = reader.s32()
                if element_type == 0:
                    offered.append(
                        self.card_by_instance.get(value, f"hidden-card-{value}")
                    )
                else:
                    offered.append(f"zone-{value}")
        elif element_type == 2:
            for _ in range(count):
                association = reader.s32()
                reader.s32()
                turn_type = reader.s32()
                reader.s32()
                offered.append(f"extra-turn-{turn_type}@{association}")
        elif element_type == 4:
            offered.extend(f"card-mode-{reader.s32()}" for _ in range(count))
        elif element_type == 5:
            offered.extend(card_name(reader.u32()) for _ in range(count))
        elif element_type == 6:
            offered.extend(
                _enum_name(GAME_BUTTONS, reader.u32(), "game-button")
                for _ in range(count)
            )
        elif element_type == 8:
            for _ in range(count):
                instance_id = reader.s32()
                reader.boolean()
                offered.append(
                    self.card_by_instance.get(instance_id, f"card-{instance_id}")
                )
        else:
            raise ProtocolError(f"unknown question element type {element_type}")
        return tuple(offered)

    def _parse_answer(self, frame: DecodedFrame) -> DecisionResolved:
        reader = Reader(frame.payload)
        question_index = reader.s32()
        answers = reader.s32_array()
        auto_played = reader.boolean()
        reader.finish()
        return DecisionResolved(
            question_index=question_index,
            answers=answers,
            seat=self.our_seat,
            auto_played=auto_played,
            timestamp_ms=frame.timestamp_ms,
        )
