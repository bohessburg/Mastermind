"""Fold normalized arena events into deterministic public game state."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..protocol.events import (
    Buy,
    DecisionResolved,
    Discard,
    Draw,
    FullState,
    FullStateZone,
    Gain,
    GameEnd,
    GameEvent,
    GameStart,
    PendingDecision,
    PileReorder,
    PileUpdate,
    Play,
    ResourceUpdate,
    Reveal,
    Shuffle,
    Topdeck,
    Trash,
    TurnStart,
    UndoRequest,
    UndoResolved,
    ZoneTransfer,
)


CardMultiset = tuple[tuple[str, int], ...]
TRACKED_ZONE_KINDS = frozenset(
    {"hand", "deck", "discard", "in-play", "set-aside"}
)
_SET_ASIDE_ZONE_KINDS = frozenset({"set-aside", "zone-type-24"})
RESOURCE_NAMES = frozenset({"actions", "buys", "coins"})
TREASURE_NAMES = frozenset({"Copper", "Silver", "Gold", "Potion"})
UNDO_SIGNAL_MAX_AGE_MS = 30_000
UNDO_SIGNAL_MAX_EVENT_GAP = 64


class TrackerError(RuntimeError):
    """The normalized stream disagreed with the tracked public state."""


@dataclass(frozen=True, kw_only=True)
class UndoResyncInfo:
    """Evidence retained for the loop when an undo-authorized reseed occurs."""

    game_id: int
    requester_seat: int
    decision_index: int
    signal_timestamp_ms: int | None
    full_state_timestamp_ms: int | None


@dataclass(frozen=True, kw_only=True)
class PendingDecisionSnapshot:
    question_index: int
    decision_type: str
    question_id: str
    offered: tuple[str, ...]
    minimum: int
    maximum: int
    association: str | None


@dataclass(frozen=True, kw_only=True)
class SeatSnapshot:
    seat: int
    hand: CardMultiset
    hand_count: int
    hand_anonymous: int
    deck: CardMultiset
    deck_count: int
    deck_anonymous: int
    hand_deck: CardMultiset
    hand_deck_count: int
    hand_deck_unresolved_count: int
    discard: CardMultiset
    discard_count: int
    discard_anonymous: int
    in_play: CardMultiset
    in_play_count: int
    in_play_anonymous: int
    set_aside_revealed: CardMultiset
    set_aside_revealed_count: int
    set_aside_anonymous: int
    owned: CardMultiset
    actions: int
    buys: int
    coins: int


@dataclass(frozen=True, kw_only=True)
class TrackerSnapshot:
    game_id: int | None
    kingdom: tuple[str, ...]
    players: tuple[str, ...]
    player_ids: tuple[int, ...]
    our_seat: int | None
    supply: CardMultiset
    seats: tuple[SeatSnapshot, ...]
    trash: CardMultiset
    trash_count: int
    trash_anonymous: int
    card_totals: CardMultiset
    turn_number: int | None
    turn_owner: int | None
    controller_seat: int | None
    phase: str | None
    pending_decision: PendingDecisionSnapshot | None
    ended: bool


@dataclass
class _ZoneState:
    known: Counter[str] = field(default_factory=Counter)
    anonymous: int = 0

    @property
    def count(self) -> int:
        return sum(self.known.values()) + self.anonymous


def _multiset(counter: Counter[str]) -> CardMultiset:
    return tuple(sorted((name, count) for name, count in counter.items() if count))


def _tracked_zone_kind(kind: str) -> str:
    if kind in _SET_ASIDE_ZONE_KINDS:
        return "set-aside"
    return kind


def _subtract(
    minuend: Counter[str],
    subtrahend: Counter[str],
    *,
    context: str,
) -> Counter[str]:
    result = minuend.copy()
    for name, count in subtrahend.items():
        if result[name] < count:
            raise TrackerError(
                f"{context}: need {count} {name!r}, have {result[name]}"
            )
        result[name] -= count
        if not result[name]:
            del result[name]
    return result


class Tracker:
    """Maintain the public state needed to build a P3 engine snapshot."""

    def __init__(self) -> None:
        self.game_id: int | None = None
        self.kingdom: tuple[str, ...] = ()
        self.players: tuple[str, ...] = ()
        self.player_ids: tuple[int, ...] = ()
        self.our_seat: int | None = None
        self.turn_number: int | None = None
        self.turn_owner: int | None = None
        self.controller_seat: int | None = None
        self.phase: str | None = None
        self.pending_decision: PendingDecisionSnapshot | None = None
        self.ended = False
        self.last_undo_resync: UndoResyncInfo | None = None

        self._initialized = False
        self._recent_undo_request: UndoRequest | None = None
        self._events_since_undo_request: int | None = None
        self._card_totals: Counter[str] = Counter()
        self._owned: dict[int, Counter[str]] = {}
        self._resources: dict[int, dict[str, int]] = {}
        self._zones: dict[int, _ZoneState] = {}
        self._zone_kind: dict[int, str] = {}
        self._zone_owner: dict[int, int | None] = {}
        self._supply_name: dict[int, str] = {}

    def consume(self, event: GameEvent) -> None:
        """Consume one event and raise immediately on any count divergence."""
        self.last_undo_resync = None
        if self._events_since_undo_request is not None:
            self._events_since_undo_request += 1
        try:
            self._consume(event)
            if self._initialized:
                self._validate(event)
        except TrackerError:
            raise
        except (KeyError, ValueError) as error:
            raise TrackerError(
                f"{type(event).__name__}: invalid tracker transition: {error}"
            ) from error

    def snapshot(self) -> TrackerSnapshot:
        """Return an immutable, deterministically ordered public snapshot."""
        seat_snapshots = tuple(
            self._seat_snapshot(seat) for seat in range(len(self.players))
        )
        trash = self._aggregate_global("trash")
        return TrackerSnapshot(
            game_id=self.game_id,
            kingdom=self.kingdom,
            players=self.players,
            player_ids=self.player_ids,
            our_seat=self.our_seat,
            supply=self._supply_snapshot(),
            seats=seat_snapshots,
            trash=_multiset(trash.known),
            trash_count=trash.count,
            trash_anonymous=trash.anonymous,
            card_totals=_multiset(self._card_totals),
            turn_number=self.turn_number,
            turn_owner=self.turn_owner,
            controller_seat=self.controller_seat,
            phase=self.phase,
            pending_decision=self.pending_decision,
            ended=self.ended,
        )

    def _consume(self, event: GameEvent) -> None:
        if isinstance(event, GameStart):
            self._on_game_start(event)
        elif isinstance(event, FullState):
            self._on_full_state(event)
        elif isinstance(event, TurnStart):
            self.turn_owner = event.seat
            self.turn_number = event.turn_number
            self.controller_seat = event.controller_seat
            self.phase = "action" if event.turn_type == 0 else "cleanup"
        elif isinstance(event, Play):
            self._move(event)
            if any(card in TREASURE_NAMES for card in event.cards):
                self.phase = "buy"
            elif self.phase is None:
                self.phase = "action"
        elif isinstance(event, Buy):
            if event.count != len(event.cards):
                raise TrackerError(
                    f"Buy reports count {event.count} for {len(event.cards)} cards"
                )
            self.phase = "buy"
        elif isinstance(event, Gain):
            self._move(event)
        elif isinstance(event, Trash):
            self._move(event)
        elif isinstance(event, Discard):
            self._move(event)
            if event.from_zone == "in-play":
                self.phase = "cleanup"
        elif isinstance(event, (Draw, Reveal, Topdeck, ZoneTransfer)):
            self._move(event)
        elif isinstance(event, Shuffle):
            self._on_shuffle(event)
        elif isinstance(event, ResourceUpdate):
            self._on_resource(event)
        elif isinstance(event, PendingDecision):
            self.pending_decision = PendingDecisionSnapshot(
                question_index=event.question_index,
                decision_type=event.decision_type,
                question_id=event.question_id,
                offered=event.offered,
                minimum=event.minimum,
                maximum=event.maximum,
                association=event.association,
            )
            if "ACTION_PHASE" in event.question_id:
                self.phase = "action"
            elif "BUY_PHASE" in event.question_id:
                self.phase = "buy"
            elif "CLEANUP_PHASE" in event.question_id:
                self.phase = "cleanup"
        elif isinstance(event, DecisionResolved):
            if (
                self.pending_decision is not None
                and self.pending_decision.question_index == event.question_index
            ):
                self.pending_decision = None
        elif isinstance(event, UndoRequest):
            self._recent_undo_request = event
            self._events_since_undo_request = 0
        elif isinstance(event, UndoResolved):
            if (
                self._recent_undo_request is not None
                and event.decision_index
                == self._recent_undo_request.decision_index
            ):
                self._recent_undo_request = None
                self._events_since_undo_request = None
        elif isinstance(event, PileReorder):
            self._on_pile_reorder(event)
        elif isinstance(event, PileUpdate):
            self._on_pile_update(event)
        elif isinstance(event, GameEnd):
            self.ended = True
            self.pending_decision = None

    def _on_game_start(self, event: GameStart) -> None:
        if self.game_id != event.game_id:
            self._reset_game_state()
        self.game_id = event.game_id
        self.kingdom = event.kingdom
        self.players = event.players
        self.player_ids = event.player_ids
        self.our_seat = event.our_seat
        self.ended = False

    def _reset_game_state(self) -> None:
        self.turn_number = None
        self.turn_owner = None
        self.controller_seat = None
        self.phase = None
        self.pending_decision = None
        self.ended = False
        self._initialized = False
        self._recent_undo_request = None
        self._events_since_undo_request = None
        self.last_undo_resync = None
        self._card_totals.clear()
        self._owned.clear()
        self._resources.clear()
        self._zones.clear()
        self._zone_kind.clear()
        self._zone_owner.clear()
        self._supply_name.clear()

    def _on_full_state(self, event: FullState) -> None:
        if self.game_id is not None and event.game_id != self.game_id:
            raise TrackerError(
                f"FullState game {event.game_id} follows GameStart {self.game_id}"
            )
        reported_totals = Counter(dict(event.card_counts))
        reported_zones = self._zones_from_full_state(event.zones)
        if self._initialized:
            if event.replacement:
                self._replace_full_state(event, reported_totals, reported_zones)
            else:
                self._reconcile_or_resync_after_undo(
                    event,
                    reported_totals,
                    reported_zones,
                )
        else:
            self._seed_full_state(event, reported_totals, reported_zones)

    def _reconcile_or_resync_after_undo(
        self,
        event: FullState,
        totals: Counter[str],
        zones: dict[int, _ZoneState],
    ) -> None:
        old_zones = {
            index: _ZoneState(known=zone.known.copy(), anonymous=zone.anonymous)
            for index, zone in self._zones.items()
        }
        old_zone_kind = self._zone_kind.copy()
        old_zone_owner = self._zone_owner.copy()
        old_supply_name = self._supply_name.copy()
        old_resources = {
            seat: resources.copy()
            for seat, resources in self._resources.items()
        }
        try:
            self._reconcile_full_state(event, totals, zones)
            return
        except TrackerError:
            self._zones = old_zones
            self._zone_kind = old_zone_kind
            self._zone_owner = old_zone_owner
            self._supply_name = old_supply_name
            self._resources = old_resources
            if not self._undo_signal_is_recent(event):
                raise

        request = self._recent_undo_request
        assert request is not None
        self._replace_full_state(event, totals, zones)
        self.pending_decision = None
        self.last_undo_resync = UndoResyncInfo(
            game_id=event.game_id,
            requester_seat=request.requester_seat,
            decision_index=request.decision_index,
            signal_timestamp_ms=request.timestamp_ms,
            full_state_timestamp_ms=event.timestamp_ms,
        )
        self._recent_undo_request = None
        self._events_since_undo_request = None

    def _undo_signal_is_recent(self, full_state: FullState) -> bool:
        request = self._recent_undo_request
        gap = self._events_since_undo_request
        if (
            request is None
            or gap is None
            or gap > UNDO_SIGNAL_MAX_EVENT_GAP
            or self.turn_number is None
            or self.ended
        ):
            return False
        if request.timestamp_ms is None or full_state.timestamp_ms is None:
            return True
        age_ms = full_state.timestamp_ms - request.timestamp_ms
        return 0 <= age_ms <= UNDO_SIGNAL_MAX_AGE_MS

    def _zones_from_full_state(
        self, zones: tuple[FullStateZone, ...]
    ) -> dict[int, _ZoneState]:
        result: dict[int, _ZoneState] = {}
        for zone in zones:
            state = _ZoneState(
                known=Counter(zone.contents),
                anonymous=zone.anonymous_count,
            )
            if state.count != len(zone.contents) + zone.anonymous_count:
                raise TrackerError(
                    f"FullState zone {zone.index} has inconsistent contents"
                )
            result[zone.index] = state
        return result

    def _seed_full_state(
        self,
        event: FullState,
        totals: Counter[str],
        zones: dict[int, _ZoneState],
    ) -> None:
        self._install_zone_metadata(event.zones)
        self._zones = zones
        self._card_totals = totals
        self._resources = {seat: {} for seat in range(len(self.players))}
        for counter in event.counters:
            if counter.owner is not None:
                self._resources.setdefault(counter.owner, {})[
                    counter.name
                ] = counter.value

        known_owned = {
            seat: self._known_player_cards(seat)
            for seat in range(len(self.players))
        }
        accounted = self._supply()
        accounted.update(self._aggregate_global("trash").known)
        for cards in known_owned.values():
            accounted.update(cards)
        residual = _subtract(
            totals, accounted, context="FullState initial card table"
        )
        anonymous_by_seat = {
            seat: sum(
                zone.anonymous
                for index, zone in self._zones.items()
                if self._zone_owner[index] == seat
            )
            for seat in range(len(self.players))
        }
        anonymous_seats = [
            seat for seat, count in anonymous_by_seat.items() if count
        ]
        if sum(residual.values()) != sum(anonymous_by_seat.values()):
            raise TrackerError(
                "FullState anonymous card count does not match card-table residual: "
                f"{sum(anonymous_by_seat.values())} zones vs "
                f"{sum(residual.values())} cards"
            )
        if len(anonymous_seats) > 1 and residual:
            raise TrackerError(
                "cannot seed ownership: anonymous cards span multiple seats"
            )

        self._owned = {seat: cards.copy() for seat, cards in known_owned.items()}
        if anonymous_seats:
            seat = anonymous_seats[0]
            self._owned[seat].update(residual)
            anonymous_zones = [
                index
                for index, zone in self._zones.items()
                if self._zone_owner[index] == seat and zone.anonymous
            ]
            if len(anonymous_zones) == 1:
                zone = self._zones[anonymous_zones[0]]
                if zone.anonymous == sum(residual.values()):
                    zone.known.update(residual)
                    zone.anonymous = 0
        self._initialized = True

    def _reconcile_full_state(
        self,
        event: FullState,
        totals: Counter[str],
        zones: dict[int, _ZoneState],
    ) -> None:
        if totals != self._card_totals:
            raise TrackerError(
                "FullState card table changed during a game: "
                f"expected {_multiset(self._card_totals)}, got {_multiset(totals)}"
            )
        old_supply = self._supply()
        reported_supply = Counter()
        for zone in event.zones:
            if zone.kind == "supply":
                reported_supply.update(zones[zone.index].known)
        if reported_supply != old_supply:
            raise TrackerError(
                "FullState supply mismatch: "
                f"tracked {_multiset(old_supply)}, "
                f"reported {_multiset(reported_supply)}"
            )

        old_counts = self._zone_counts_by_owner_kind()
        new_counts: Counter[tuple[int | None, str]] = Counter()
        for zone in event.zones:
            kind = _tracked_zone_kind(zone.kind)
            if kind in TRACKED_ZONE_KINDS or kind == "trash":
                new_counts[(zone.owner, kind)] += zones[zone.index].count
        relevant_old = Counter(
            {
                key: value
                for key, value in old_counts.items()
                if key[1] in TRACKED_ZONE_KINDS or key[1] == "trash"
            }
        )
        if new_counts != relevant_old:
            raise TrackerError(
                "FullState zone-count mismatch: "
                f"tracked {dict(relevant_old)}, reported {dict(new_counts)}"
            )

        for counter in event.counters:
            if counter.owner is None:
                continue
            previous = self._resources.get(counter.owner, {}).get(counter.name)
            if previous is not None and previous != counter.value:
                raise TrackerError(
                    f"FullState counter mismatch for seat {counter.owner} "
                    f"{counter.name}: tracked {previous}, reported {counter.value}"
                )

        merged: dict[int, _ZoneState] = {}
        old_metadata = (
            self._zone_kind.copy(),
            self._zone_owner.copy(),
            self._supply_name.copy(),
        )
        self._install_zone_metadata(event.zones)
        for index, reported in zones.items():
            old = self._zones.get(index)
            if old is None or old.count != reported.count:
                merged[index] = reported
                continue
            merged[index] = self._merge_known_zone(
                old,
                reported,
                context=f"FullState zone {index}",
            )
        self._zones = merged
        for counter in event.counters:
            if counter.owner is not None:
                self._resources.setdefault(counter.owner, {})[
                    counter.name
                ] = counter.value

        # Metadata replacement is authoritative. The saved values are only
        # included in divergence messages via this sanity check.
        if any(index not in self._zone_kind for index in self._zones):
            self._zone_kind, self._zone_owner, self._supply_name = old_metadata
            raise TrackerError("FullState omitted metadata for a reported zone")

    def _replace_full_state(
        self,
        event: FullState,
        totals: Counter[str],
        zones: dict[int, _ZoneState],
    ) -> None:
        """Install a reconnect state after an expected recording/feed gap."""
        if totals != self._card_totals:
            raise TrackerError(
                "replacement FullState changed the game's card table"
            )
        old_zones = self._zones
        old_kind = self._zone_kind
        old_owner = self._zone_owner
        active_seat = (
            self.controller_seat
            if self.controller_seat is not None
            else self.turn_owner
        )

        self._install_zone_metadata(event.zones)
        merged: dict[int, _ZoneState] = {}
        for index, reported in zones.items():
            owner = self._zone_owner[index]
            kind = self._zone_kind[index]
            old = old_zones.get(index)
            unchanged_private_seat = (
                owner is not None
                and owner != active_seat
                and kind in TRACKED_ZONE_KINDS
            )
            if (
                unchanged_private_seat
                and old is not None
                and old_kind.get(index) == kind
                and old_owner.get(index) == owner
                and old.count == reported.count
            ):
                merged[index] = self._merge_known_zone(
                    old,
                    reported,
                    context=f"replacement FullState zone {index}",
                )
            else:
                merged[index] = reported
        self._zones = merged

        if active_seat is None or active_seat not in self._owned:
            raise TrackerError(
                "replacement FullState has no active seat for ownership resync"
            )
        accounted = self._supply()
        accounted.update(self._aggregate_global("trash").known)
        replacement_owned: dict[int, Counter[str]] = {}
        for seat, cards in self._owned.items():
            if seat == active_seat:
                continue
            replacement_owned[seat] = cards.copy()
            accounted.update(cards)
        replacement_owned[active_seat] = _subtract(
            totals,
            accounted,
            context=f"replacement FullState seat {active_seat} ownership",
        )
        self._owned = replacement_owned

        for counter in event.counters:
            if counter.owner is not None:
                self._resources.setdefault(counter.owner, {})[
                    counter.name
                ] = counter.value

    def _install_zone_metadata(
        self, zones: tuple[FullStateZone, ...]
    ) -> None:
        self._zone_kind = {
            zone.index: _tracked_zone_kind(zone.kind) for zone in zones
        }
        self._zone_owner = {zone.index: zone.owner for zone in zones}
        self._supply_name = {
            zone.index: zone.display_name
            for zone in zones
            if zone.kind == "supply" and zone.display_name is not None
        }

    def _merge_known_zone(
        self,
        old: _ZoneState,
        reported: _ZoneState,
        *,
        context: str,
    ) -> _ZoneState:
        known = Counter(
            {
                name: max(old.known[name], reported.known[name])
                for name in old.known.keys() | reported.known.keys()
            }
        )
        if sum(known.values()) > old.count:
            raise TrackerError(
                f"{context}: incompatible known contents "
                f"{_multiset(old.known)} vs {_multiset(reported.known)}"
            )
        return _ZoneState(known=known, anonymous=old.count - sum(known.values()))

    def _install_event_zone(
        self,
        index: int,
        kind: str,
        *,
        owner: int | None,
        destination: bool,
    ) -> _ZoneState:
        if index in self._zones:
            if self._zone_kind[index] != kind:
                raise TrackerError(
                    f"zone {index} changed kind from "
                    f"{self._zone_kind[index]} to {kind}"
                )
            if owner is not None:
                known_owner = self._zone_owner[index]
                if known_owner is not None and known_owner != owner:
                    raise TrackerError(
                        f"zone {index} owner {known_owner} != event seat {owner}"
                    )
            return self._zones[index]
        if not destination:
            raise TrackerError(f"movement source zone {index} is unknown")
        self._zone_kind[index] = kind
        self._zone_owner[index] = owner
        self._zones[index] = _ZoneState()
        return self._zones[index]

    def _move(
        self,
        event: Play | Gain | Trash | Discard | Draw | Reveal | Topdeck | ZoneTransfer,
    ) -> None:
        if not self._initialized:
            raise TrackerError(f"{type(event).__name__} arrived before FullState")
        if event.count < 0 or len(event.cards) > event.count:
            raise TrackerError(
                f"{type(event).__name__} count {event.count} cannot carry "
                f"{len(event.cards)} known cards"
            )
        source_owner = self._zone_owner.get(event.from_zone_index)
        destination_owner = self._zone_owner.get(event.to_zone_index)
        inferred_owner = event.seat
        source = self._install_event_zone(
            event.from_zone_index,
            _tracked_zone_kind(event.from_zone),
            owner=inferred_owner if source_owner is None else source_owner,
            destination=False,
        )
        destination = self._install_event_zone(
            event.to_zone_index,
            _tracked_zone_kind(event.to_zone),
            owner=(
                inferred_owner
                if destination_owner is None and event.to_zone != "trash"
                else destination_owner
            ),
            destination=True,
        )
        allow_anonymize = (
            event.seat != self.our_seat
            and _tracked_zone_kind(event.from_zone)
            in {"hand", "deck", "set-aside"}
        )
        moved_known, moved_anonymous = self._take_cards(
            source,
            cards=event.cards,
            count=event.count,
            allow_anonymize=allow_anonymize,
            context=(
                f"{type(event).__name__} seat={event.seat} "
                f"{event.from_zone}[{event.from_zone_index}]"
            ),
        )
        destination.known.update(moved_known)
        destination.anonymous += moved_anonymous

        if event.from_zone == "supply" and event.seat is not None:
            if moved_anonymous:
                raise TrackerError(
                    f"Gain from supply moved {moved_anonymous} anonymous cards"
                )
            self._owned[event.seat].update(moved_known)
        if event.to_zone == "trash" and event.seat is not None:
            if moved_anonymous:
                raise TrackerError(
                    f"Trash from seat {event.seat} hid "
                    f"{moved_anonymous} card identities"
                )
            self._owned[event.seat] = _subtract(
                self._owned[event.seat],
                moved_known,
                context=f"Trash ownership seat {event.seat}",
            )
        if (
            event.to_zone == "supply"
            and event.from_zone != "supply"
            and event.seat is not None
        ):
            if moved_anonymous:
                raise TrackerError("return to supply hid card identities")
            self._owned[event.seat] = _subtract(
                self._owned[event.seat],
                moved_known,
                context=f"return-to-supply ownership seat {event.seat}",
            )

    def _take_cards(
        self,
        source: _ZoneState,
        *,
        cards: tuple[str, ...],
        count: int,
        allow_anonymize: bool,
        context: str,
    ) -> tuple[Counter[str], int]:
        if source.count < count:
            raise TrackerError(
                f"{context}: move reports {count} cards, source has {source.count}"
            )
        moved = Counter[str]()
        for card in cards:
            if source.known[card]:
                source.known[card] -= 1
                if not source.known[card]:
                    del source.known[card]
            elif source.anonymous:
                source.anonymous -= 1
            else:
                raise TrackerError(
                    f"{context}: reported {card!r} is absent from source"
                )
            moved[card] += 1

        hidden = count - len(cards)
        if hidden == source.count:
            moved.update(source.known)
            moved_anonymous = source.anonymous
            source.known.clear()
            source.anonymous = 0
            return moved, moved_anonymous

        if hidden and allow_anonymize and source.known:
            source.anonymous += sum(source.known.values())
            source.known.clear()
        if source.anonymous < hidden:
            if not allow_anonymize:
                raise TrackerError(
                    f"{context}: {hidden} hidden cards requested but only "
                    f"{source.anonymous} source cards are anonymous"
                )
            source.anonymous += sum(source.known.values())
            source.known.clear()
        source.anonymous -= hidden
        return moved, hidden

    def _on_shuffle(self, event: Shuffle) -> None:
        if not event.included_discard:
            return
        discard_indices = self._zone_indices(event.seat, "discard")
        deck_indices = self._zone_indices(event.seat, "deck")
        if len(discard_indices) != 1 or len(deck_indices) != 1:
            raise TrackerError(
                f"Shuffle seat {event.seat} needs one deck and discard, got "
                f"{deck_indices} and {discard_indices}"
            )
        discard = self._zones[discard_indices[0]]
        deck = self._zones[deck_indices[0]]
        deck.known.update(discard.known)
        deck.anonymous += discard.anonymous
        discard.known.clear()
        discard.anonymous = 0

    def _on_resource(self, event: ResourceUpdate) -> None:
        if event.seat is None:
            return
        if event.value < 0 and event.resource in RESOURCE_NAMES:
            raise TrackerError(
                f"negative {event.resource} for seat {event.seat}: {event.value}"
            )
        self._resources.setdefault(event.seat, {})[event.resource] = event.value

    def _on_pile_reorder(self, event: PileReorder) -> None:
        zone = self._zones.get(event.zone_index)
        if zone is None:
            return
        if zone.count != event.count:
            raise TrackerError(
                f"PileReorder zone {event.zone_index} reports {event.count} "
                f"cards, tracked {zone.count}"
            )
        for name, count in Counter(event.cards).items():
            if zone.known[name] + zone.anonymous < count:
                raise TrackerError(
                    f"PileReorder zone {event.zone_index} reports "
                    f"{count} {name!r}, tracked {zone.known[name]} known and "
                    f"{zone.anonymous} anonymous"
                )

    def _on_pile_update(self, event: PileUpdate) -> None:
        if event.zone != "supply" or event.top_card is None:
            return
        pile_name = self._supply_name.get(event.zone_index)
        if pile_name is not None and event.top_card != pile_name:
            raise TrackerError(
                f"PileUpdate zone {event.zone_index} top is "
                f"{event.top_card!r}, expected pile {pile_name!r}"
            )

    def _seat_snapshot(self, seat: int) -> SeatSnapshot:
        hand = self._aggregate_seat(seat, "hand")
        deck = self._aggregate_seat(seat, "deck")
        discard = self._aggregate_seat(seat, "discard")
        in_play = self._aggregate_seat(seat, "in-play")
        set_aside = self._aggregate_seat(seat, "set-aside")
        owned = self._owned.get(seat, Counter())

        located_outside_private = discard.known.copy()
        located_outside_private.update(in_play.known)
        located_outside_private.update(set_aside.known)
        hand_deck = _subtract(
            owned,
            located_outside_private,
            context=f"seat {seat} private composition",
        )
        hand_deck_count = hand.count + deck.count
        unresolved = sum(hand_deck.values()) - hand_deck_count
        if unresolved < 0:
            raise TrackerError(
                f"seat {seat} known private composition has "
                f"{sum(hand_deck.values())} cards but zones have "
                f"{hand_deck_count}"
            )
        resources = self._resources.get(seat, {})
        return SeatSnapshot(
            seat=seat,
            hand=_multiset(hand.known),
            hand_count=hand.count,
            hand_anonymous=hand.anonymous,
            deck=_multiset(deck.known),
            deck_count=deck.count,
            deck_anonymous=deck.anonymous,
            hand_deck=_multiset(hand_deck),
            hand_deck_count=hand_deck_count,
            hand_deck_unresolved_count=unresolved,
            discard=_multiset(discard.known),
            discard_count=discard.count,
            discard_anonymous=discard.anonymous,
            in_play=_multiset(in_play.known),
            in_play_count=in_play.count,
            in_play_anonymous=in_play.anonymous,
            set_aside_revealed=_multiset(set_aside.known),
            set_aside_revealed_count=set_aside.count,
            set_aside_anonymous=set_aside.anonymous,
            owned=_multiset(owned),
            actions=resources.get("actions", 0),
            buys=resources.get("buys", 0),
            coins=resources.get("coins", 0),
        )

    def _validate(self, event: GameEvent) -> None:
        for index, zone in self._zones.items():
            if zone.anonymous < 0 or any(count <= 0 for count in zone.known.values()):
                raise TrackerError(
                    f"after {type(event).__name__}, zone {index} has invalid "
                    f"state known={dict(zone.known)} anonymous={zone.anonymous}"
                )
        for seat in range(len(self.players)):
            physical_count = sum(
                zone.count
                for index, zone in self._zones.items()
                if self._zone_owner[index] == seat
                and self._zone_kind[index] in TRACKED_ZONE_KINDS
            )
            owned_count = sum(self._owned.get(seat, Counter()).values())
            if physical_count != owned_count:
                raise TrackerError(
                    f"after {type(event).__name__}, seat {seat} owns "
                    f"{owned_count} cards but zones contain {physical_count}"
                )

        accounted = self._supply()
        trash = self._aggregate_global("trash")
        if trash.anonymous:
            raise TrackerError(
                f"after {type(event).__name__}, trash has "
                f"{trash.anonymous} anonymous cards"
            )
        accounted.update(trash.known)
        for cards in self._owned.values():
            accounted.update(cards)
        if accounted != self._card_totals:
            missing = self._card_totals - accounted
            extra = accounted - self._card_totals
            raise TrackerError(
                f"after {type(event).__name__}, card conservation failed: "
                f"missing={_multiset(missing)} extra={_multiset(extra)}"
            )

        if self.our_seat is not None:
            hand = self._aggregate_seat(self.our_seat, "hand")
            if hand.anonymous:
                raise TrackerError(
                    f"after {type(event).__name__}, our hand has "
                    f"{hand.anonymous} anonymous cards"
                )

    def _known_player_cards(self, seat: int) -> Counter[str]:
        result: Counter[str] = Counter()
        for index, zone in self._zones.items():
            if (
                self._zone_owner[index] == seat
                and self._zone_kind[index] in TRACKED_ZONE_KINDS
            ):
                result.update(zone.known)
        return result

    def _zone_indices(self, seat: int, kind: str) -> list[int]:
        return [
            index
            for index in self._zones
            if self._zone_owner[index] == seat and self._zone_kind[index] == kind
        ]

    def _aggregate_seat(self, seat: int, kind: str) -> _ZoneState:
        result = _ZoneState()
        for index in self._zone_indices(seat, kind):
            result.known.update(self._zones[index].known)
            result.anonymous += self._zones[index].anonymous
        return result

    def _aggregate_global(self, kind: str) -> _ZoneState:
        result = _ZoneState()
        for index, zone in self._zones.items():
            if self._zone_kind[index] == kind and self._zone_owner[index] is None:
                result.known.update(zone.known)
                result.anonymous += zone.anonymous
        return result

    def _supply(self) -> Counter[str]:
        result: Counter[str] = Counter()
        for index, zone in self._zones.items():
            if self._zone_kind[index] != "supply":
                continue
            if zone.anonymous:
                raise TrackerError(
                    f"supply zone {index} contains {zone.anonymous} anonymous cards"
                )
            result.update(zone.known)
        return result

    def _supply_snapshot(self) -> CardMultiset:
        return tuple(
            sorted(
                (
                    pile_name,
                    self._zones[index].count,
                )
                for index, pile_name in self._supply_name.items()
            )
        )

    def _zone_counts_by_owner_kind(
        self,
    ) -> Counter[tuple[int | None, str]]:
        counts: Counter[tuple[int | None, str]] = Counter()
        for index, zone in self._zones.items():
            counts[(self._zone_owner[index], self._zone_kind[index])] += zone.count
        return counts
