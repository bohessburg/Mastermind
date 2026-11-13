"""Typed schema for the unified, analysis-oriented game record format."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


SCHEMA_VERSION = "1.0"
Visibility = Literal["known", "counts_only", "unknown"]
Source = Literal["local", "arena"]
Outcome = Literal["win", "loss", "tie", "unknown"]


@dataclass(frozen=True, kw_only=True)
class CardRef:
    """One stable engine definition id paired with its display name."""

    def_id: int
    name: str


@dataclass(frozen=True, kw_only=True)
class SeatInfo:
    """Identity and controller information for one zero-based seat."""

    index: int
    kind: str
    display_name: str | None
    controlled: bool
    bot: bool


@dataclass(frozen=True, kw_only=True)
class SeatResult:
    """Final standing for one seat, or an explicitly unknown standing."""

    seat: int
    visibility: Visibility
    vp: int | None
    placing: int | None
    outcome: Outcome


@dataclass(frozen=True, kw_only=True)
class ObservedCards:
    """A card consequence with honest card-identity observability."""

    visibility: Visibility
    count: int | None
    cards: tuple[CardRef, ...]


@dataclass(frozen=True, kw_only=True)
class Resources:
    """The active seat's resources after a record."""

    visibility: Visibility
    actions: int | None
    buys: int | None
    coins: int | None


@dataclass(frozen=True, kw_only=True)
class ZoneCount:
    """One zone size after a record.

    Arena opponent hand/deck entries use ``counts_only`` even though their
    numeric sizes are known, because their card identities are hidden.
    """

    seat: int | None
    zone: str
    visibility: Visibility
    count: int | None


@dataclass(frozen=True, kw_only=True)
class ActionRecord:
    """One ordered decision, action, consequence, or turn marker."""

    index: int
    source_index: int
    timestamp_ms: int | None
    timestamp_ms_visibility: Visibility
    record_type: str
    event: str
    turn_number: int | None
    turn_number_visibility: Visibility
    active_seat: int | None
    active_seat_visibility: Visibility
    phase: str | None
    phase_visibility: Visibility
    actor_seat: int | None
    actor_seat_visibility: Visibility
    engine_action_ids: tuple[int, ...]
    engine_action_ids_visibility: Visibility
    action_labels: tuple[str, ...]
    action_labels_visibility: Visibility
    played: ObservedCards
    bought: ObservedCards
    gained: ObservedCards
    trashed: ObservedCards
    discarded: ObservedCards
    resources_after: Resources
    zone_counts_after: tuple[ZoneCount, ...]


@dataclass(frozen=True, kw_only=True)
class GameRecord:
    """One self-describing game with a flat, ordered record body."""

    schema_version: str
    source: Source
    provenance: str
    game_id: str
    timestamp: str | None
    timestamp_visibility: Visibility
    kingdom: tuple[CardRef, ...]
    player_count: int
    seats: tuple[SeatInfo, ...]
    obs_version: int | None
    obs_version_visibility: Visibility
    controlled_seat: int | None
    controlled_seat_visibility: Visibility
    results: tuple[SeatResult, ...]
    records: tuple[ActionRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON object shape."""
        return asdict(self)


def unknown_cards() -> ObservedCards:
    """Return a consequence for which the source contains no evidence."""
    return ObservedCards(visibility="unknown", count=None, cards=())


def known_cards(cards: tuple[CardRef, ...]) -> ObservedCards:
    """Return an exactly observed card consequence."""
    return ObservedCards(visibility="known", count=len(cards), cards=cards)


def validate_record(record: GameRecord | Mapping[str, Any]) -> None:
    """Validate schema invariants without an optional validation dependency."""
    data = record.to_dict() if isinstance(record, GameRecord) else dict(record)
    _require(data.get("schema_version") == SCHEMA_VERSION, "schema_version")
    _require(data.get("source") in {"local", "arena"}, "source")
    _require(isinstance(data.get("provenance"), str), "provenance")
    _require(isinstance(data.get("game_id"), str), "game_id")
    _visibility_pair(data, "timestamp")
    _visibility_pair(data, "obs_version")
    _visibility_pair(data, "controlled_seat")

    player_count = data.get("player_count")
    _require(isinstance(player_count, int) and player_count >= 2, "player_count")
    kingdom = data.get("kingdom")
    _require(isinstance(kingdom, (list, tuple)), "kingdom")
    for card in kingdom:
        _validate_card(card)

    seats = data.get("seats")
    results = data.get("results")
    _require(isinstance(seats, (list, tuple)), "seats")
    _require(len(seats) == player_count, "seats length")
    _require(isinstance(results, (list, tuple)), "results")
    _require(len(results) == player_count, "results length")
    for index, seat in enumerate(seats):
        _require(seat.get("index") == index, "seat index")
        _require(isinstance(seat.get("kind"), str), "seat kind")
        _require(isinstance(seat.get("controlled"), bool), "seat controlled")
        _require(isinstance(seat.get("bot"), bool), "seat bot")
    for index, result in enumerate(results):
        _require(result.get("seat") == index, "result seat")
        _require(result.get("visibility") in _VISIBILITIES, "result visibility")
        _require(
            result.get("outcome") in {"win", "loss", "tie", "unknown"},
            "result outcome",
        )
        if result.get("visibility") == "unknown":
            _require(result.get("vp") is None, "unknown result vp")
            _require(result.get("placing") is None, "unknown result placing")
            _require(result.get("outcome") == "unknown", "unknown result outcome")

    records = data.get("records")
    _require(isinstance(records, (list, tuple)), "records")
    for index, item in enumerate(records):
        _require(item.get("index") == index, "record index")
        _validate_action_record(item, player_count)


_VISIBILITIES = {"known", "counts_only", "unknown"}


def _require(condition: bool, field: str) -> None:
    if not condition:
        raise ValueError(f"invalid game record field: {field}")


def _visibility_pair(data: Mapping[str, Any], name: str) -> None:
    visibility = data.get(f"{name}_visibility")
    _require(visibility in _VISIBILITIES, f"{name}_visibility")
    if visibility == "unknown":
        _require(data.get(name) is None, f"unknown {name}")


def _validate_card(card: Mapping[str, Any]) -> None:
    _require(isinstance(card.get("def_id"), int), "card def_id")
    _require(isinstance(card.get("name"), str) and bool(card["name"]), "card name")


def _validate_observed_cards(group: Mapping[str, Any], name: str) -> None:
    visibility = group.get("visibility")
    count = group.get("count")
    cards = group.get("cards")
    _require(visibility in _VISIBILITIES, f"{name} visibility")
    _require(isinstance(cards, (list, tuple)), f"{name} cards")
    for card in cards:
        _validate_card(card)
    if visibility == "unknown":
        _require(count is None and not cards, f"unknown {name}")
    else:
        _require(isinstance(count, int) and count >= len(cards), f"{name} count")
        if visibility == "known":
            _require(count == len(cards), f"known {name} count")
        if visibility == "counts_only":
            _require(not cards, f"counts-only {name} cards")


def _validate_action_record(item: Mapping[str, Any], player_count: int) -> None:
    _require(isinstance(item.get("source_index"), int), "source_index")
    _require(isinstance(item.get("record_type"), str), "record_type")
    _require(isinstance(item.get("event"), str), "event")
    for field in (
        "timestamp_ms",
        "turn_number",
        "active_seat",
        "phase",
        "actor_seat",
    ):
        _visibility_pair(item, field)
    for field in ("active_seat", "actor_seat"):
        seat = item.get(field)
        _require(
            seat is None or isinstance(seat, int) and 0 <= seat < player_count,
            field,
        )
    for field in ("engine_action_ids", "action_labels"):
        visibility = item.get(f"{field}_visibility")
        values = item.get(field)
        _require(visibility in _VISIBILITIES, f"{field}_visibility")
        _require(isinstance(values, (list, tuple)), field)
        if visibility == "unknown":
            _require(not values, f"unknown {field}")
    _require(
        len(item["engine_action_ids"]) == len(item["action_labels"])
        or item["engine_action_ids_visibility"] == "unknown"
        or item["action_labels_visibility"] == "unknown",
        "action id/label length",
    )
    for field in ("played", "bought", "gained", "trashed", "discarded"):
        _validate_observed_cards(item[field], field)

    resources = item.get("resources_after")
    _require(resources.get("visibility") in _VISIBILITIES, "resources visibility")
    resource_values = tuple(resources.get(name) for name in ("actions", "buys", "coins"))
    if resources["visibility"] == "unknown":
        _require(resource_values == (None, None, None), "unknown resources")
    else:
        _require(all(isinstance(value, int) for value in resource_values), "resources")

    zones = item.get("zone_counts_after")
    _require(isinstance(zones, (list, tuple)), "zone_counts_after")
    for zone in zones:
        _require(zone.get("visibility") in _VISIBILITIES, "zone visibility")
        _require(isinstance(zone.get("zone"), str), "zone name")
        seat = zone.get("seat")
        _require(
            seat is None or isinstance(seat, int) and 0 <= seat < player_count,
            "zone seat",
        )
        if zone["visibility"] == "unknown":
            _require(zone.get("count") is None, "unknown zone count")
        else:
            _require(
                isinstance(zone.get("count"), int) and zone["count"] >= 0,
                "zone count",
            )
