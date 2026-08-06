"""Fold partial-observability arena archives into unified game records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import dominion_v2_py as dz

from src.v2.web.server.defs import def_id, def_name, load_defs

from .model import (
    SCHEMA_VERSION,
    ActionRecord,
    CardRef,
    GameRecord,
    ObservedCards,
    Resources,
    SeatInfo,
    SeatResult,
    ZoneCount,
    known_cards,
    unknown_cards,
)


_MOVEMENT_EVENTS = {
    "Play",
    "Gain",
    "Trash",
    "Discard",
    "Draw",
    "Reveal",
    "Topdeck",
    "ZoneTransfer",
}
_ACTION_EVENTS = {"Play", "Buy", "Attack"}
_CONSEQUENCE_EVENTS = {
    "Gain",
    "Trash",
    "Discard",
    "Draw",
    "Reveal",
    "Topdeck",
    "ZoneTransfer",
    "Shuffle",
    "ResourceUpdate",
    "PileUpdate",
    "PileReorder",
}
_BODY_EVENTS = {
    "FullState",
    "TurnStart",
    "PendingDecision",
    "ReactionWindow",
    "Play",
    "Buy",
    "Attack",
    "Gain",
    "Trash",
    "Discard",
    "Draw",
    "Reveal",
    "Topdeck",
    "ZoneTransfer",
    "Shuffle",
    "ResourceUpdate",
    "PileReorder",
    "GameEnd",
    "GameResult",
}


@dataclass
class _ArenaState:
    player_count: int
    our_seat: int | None
    turn_number: int | None = None
    active_seat: int | None = None
    phase: str | None = None
    zones: dict[int, tuple[int | None, str, int]] = field(default_factory=dict)
    resources: dict[int, dict[str, int]] = field(default_factory=dict)


def is_arena_game_dir(path: Path | str) -> bool:
    """Return whether a directory has the stable arena source files."""
    candidate = Path(path)
    return (
        candidate.is_dir()
        and (candidate / "events.jsonl").is_file()
        and (candidate / "decisions.jsonl").is_file()
        and (candidate / "result.json").is_file()
    )


def convert_arena_archive(path: Path | str) -> GameRecord:
    """Convert one arena game directory without inventing hidden state."""
    archive = Path(path)
    if not is_arena_game_dir(archive):
        raise ValueError(f"not an arena game archive: {archive}")

    events = _read_jsonl(archive / "events.jsonl")
    decisions = _read_jsonl(archive / "decisions.jsonl")
    result = _read_object_if_present(archive / "result.json")
    game_start = next(
        (row["event"] for row in events if row.get("event_type") == "GameStart"),
        {},
    )

    players = tuple(str(value) for value in game_start.get("players", ()))
    scores = tuple(int(value) for value in result.get("scores", ()))
    placings = tuple(int(value) for value in result.get("placings", ()))
    player_count = len(players) or len(scores)
    if player_count < 2:
        raise ValueError(f"arena archive has no usable player roster: {archive}")
    if not players:
        players = tuple("" for _ in range(player_count))
    our_seat = _optional_int(
        game_start.get("our_seat", result.get("our_seat"))
    )
    game_id_value = game_start.get("game_id", result.get("game_id"))
    game_id = (
        str(game_id_value)
        if game_id_value is not None
        else _game_id_from_directory(archive)
    )
    timestamp_ms = _optional_int(game_start.get("timestamp_ms"))
    timestamp = (
        datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).isoformat()
        if timestamp_ms is not None
        else None
    )
    kingdom_names = tuple(str(value) for value in game_start.get("kingdom", ()))
    decision_by_question = {
        int(row["question_index"]): row
        for row in decisions
        if row.get("question_index") is not None
    }

    state = _ArenaState(player_count=player_count, our_seat=our_seat)
    body: list[ActionRecord] = []
    for source_index, row in enumerate(events):
        event_type = str(row.get("event_type", "UnknownEvent"))
        event = row.get("event")
        if not isinstance(event, dict):
            event = {}
        if event_type == "GameStart":
            continue
        _update_arena_context(state, event_type, event)
        if event_type == "FullState":
            _apply_full_state(state, event)
        elif event_type in _MOVEMENT_EVENTS:
            _apply_movement(state, event)
        elif event_type == "ResourceUpdate":
            _apply_resource_update(state, event)

        if event_type not in _BODY_EVENTS:
            continue
        body.append(
            _arena_record(
                state=state,
                index=len(body),
                source_index=source_index,
                event_type=event_type,
                event=event,
                decision=(
                    decision_by_question.get(int(event["question_index"]))
                    if event_type == "PendingDecision"
                    and event.get("question_index") is not None
                    else None
                ),
            )
        )

    return GameRecord(
        schema_version=SCHEMA_VERSION,
        source="arena",
        provenance=str(archive),
        game_id=game_id,
        timestamp=timestamp,
        timestamp_visibility="known" if timestamp is not None else "unknown",
        kingdom=tuple(_card_from_name(name) for name in kingdom_names),
        player_count=player_count,
        seats=tuple(
            SeatInfo(
                index=seat,
                kind=(
                    "bot:nnmcts"
                    if seat == our_seat
                    else "human"
                ),
                display_name=players[seat] or None,
                controlled=seat == our_seat,
                bot=seat == our_seat,
            )
            for seat in range(player_count)
        ),
        obs_version=None,
        obs_version_visibility="unknown",
        controlled_seat=our_seat,
        controlled_seat_visibility=(
            "known" if our_seat is not None else "unknown"
        ),
        results=_arena_results(
            player_count=player_count,
            scores=scores,
            placings=placings,
            result=result,
        ),
        records=tuple(body),
    )


def _arena_record(
    *,
    state: _ArenaState,
    index: int,
    source_index: int,
    event_type: str,
    event: Mapping[str, Any],
    decision: Mapping[str, Any] | None,
) -> ActionRecord:
    actor = _event_actor(state, event_type, event, decision)
    timestamp_ms = _optional_int(event.get("timestamp_ms"))
    action_ids, labels, action_visibility = _arena_actions(
        event_type, event, decision
    )
    empty = unknown_cards()
    played = empty
    bought = empty
    gained = empty
    trashed = empty
    discarded = empty
    observed = _observed_event_cards(event)
    if event_type == "Play":
        played = observed
    elif event_type == "Buy":
        bought = observed
    elif event_type == "Gain":
        gained = observed
    elif event_type == "Trash":
        trashed = observed
    elif event_type == "Discard":
        discarded = observed

    return ActionRecord(
        index=index,
        source_index=source_index,
        timestamp_ms=timestamp_ms,
        timestamp_ms_visibility=(
            "known" if timestamp_ms is not None else "unknown"
        ),
        record_type=_record_type(event_type),
        event=event_type,
        turn_number=state.turn_number,
        turn_number_visibility=(
            "known" if state.turn_number is not None else "unknown"
        ),
        active_seat=state.active_seat,
        active_seat_visibility=(
            "known" if state.active_seat is not None else "unknown"
        ),
        phase=state.phase,
        phase_visibility="known" if state.phase is not None else "unknown",
        actor_seat=actor,
        actor_seat_visibility="known" if actor is not None else "unknown",
        engine_action_ids=action_ids,
        engine_action_ids_visibility=action_visibility,
        action_labels=labels,
        action_labels_visibility=(
            "known" if labels else "unknown"
        ),
        played=played,
        bought=bought,
        gained=gained,
        trashed=trashed,
        discarded=discarded,
        resources_after=_arena_resources(state, actor),
        zone_counts_after=_arena_zone_counts(state),
    )


def _update_arena_context(
    state: _ArenaState,
    event_type: str,
    event: Mapping[str, Any],
) -> None:
    if event_type == "TurnStart":
        state.turn_number = _optional_int(event.get("turn_number"))
        state.active_seat = _optional_int(event.get("seat"))
        turn_type = _optional_int(event.get("turn_type"))
        state.phase = "action" if turn_type == 0 else "cleanup" if turn_type == 1 else None
    elif event_type == "PendingDecision":
        question_id = str(event.get("question_id", ""))
        if "ACTION_PHASE" in question_id:
            state.phase = "action"
        elif "BUY_PHASE" in question_id:
            state.phase = "buy"
        elif "NIGHT_PHASE" in question_id:
            state.phase = "night"
        elif "CLEANUP_PHASE" in question_id:
            state.phase = "cleanup"
    elif event_type == "Play":
        names = tuple(str(value) for value in event.get("cards", ()))
        if any(
            "Treasure" in load_defs()["by_name"].get(name, {}).get("types", ())
            for name in names
        ):
            state.phase = "buy"
        elif state.phase is None:
            state.phase = "action"
    elif event_type == "Buy":
        state.phase = "buy"
    elif event_type == "Discard" and event.get("from_zone") == "in-play":
        state.phase = "cleanup"


def _apply_full_state(
    state: _ArenaState,
    event: Mapping[str, Any],
) -> None:
    state.zones.clear()
    for raw in event.get("zones", ()):
        if not isinstance(raw, dict) or raw.get("index") is None:
            continue
        contents = raw.get("contents", ())
        count = len(contents) + int(raw.get("anonymous_count", 0))
        state.zones[int(raw["index"])] = (
            _optional_int(raw.get("owner")),
            str(raw.get("kind", "unknown")),
            count,
        )
    for raw in event.get("counters", ()):
        if not isinstance(raw, dict):
            continue
        owner = _optional_int(raw.get("owner"))
        name = str(raw.get("name", ""))
        if owner is None or name not in {"actions", "buys", "coins"}:
            continue
        state.resources.setdefault(owner, {})[name] = int(raw.get("value", 0))


def _apply_movement(
    state: _ArenaState,
    event: Mapping[str, Any],
) -> None:
    count = int(event.get("count", 0))
    source = _optional_int(event.get("from_zone_index"))
    destination = _optional_int(event.get("to_zone_index"))
    if source in state.zones:
        owner, kind, old_count = state.zones[source]
        state.zones[source] = (owner, kind, max(0, old_count - count))
    if destination in state.zones:
        owner, kind, old_count = state.zones[destination]
        state.zones[destination] = (owner, kind, old_count + count)


def _apply_resource_update(
    state: _ArenaState,
    event: Mapping[str, Any],
) -> None:
    seat = _optional_int(event.get("seat"))
    resource = str(event.get("resource", ""))
    if seat is None or resource not in {"actions", "buys", "coins"}:
        return
    state.resources.setdefault(seat, {})[resource] = int(event.get("value", 0))


def _arena_actions(
    event_type: str,
    event: Mapping[str, Any],
    decision: Mapping[str, Any] | None,
) -> tuple[tuple[int, ...], tuple[str, ...], str]:
    if decision is not None:
        actions = tuple(int(value) for value in decision.get("engine_actions", ()))
        if actions:
            return actions, tuple(_generic_action_label(action) for action in actions), "known"
        labels = (f"Answer {event.get('question_id', 'question')}",)
        return (), labels, "unknown"
    if event_type in {"Play", "Buy"}:
        base = int(dz.A_PLAY_BASE if event_type == "Play" else dz.A_BUY_BASE)
        cards = tuple(str(value) for value in event.get("cards", ()))
        try:
            actions = tuple(base + def_id(name) for name in cards)
        except KeyError:
            actions = ()
        labels = tuple(f"{event_type} {name}" for name in cards)
        return actions, labels, "known" if len(actions) == len(cards) else "unknown"

    cards = tuple(str(value) for value in event.get("cards", ()))
    if cards:
        return (), (f"{event_type} {', '.join(cards)}",), "unknown"
    if event_type == "TurnStart":
        return (), ("Start turn",), "unknown"
    if event_type == "PendingDecision":
        return (), (f"Decision {event.get('question_id', 'unknown')}",), "unknown"
    return (), (), "unknown"


def _generic_action_label(action: int) -> str:
    if action == int(dz.A_PASS):
        return "Pass"
    if int(dz.A_PLAY_BASE) <= action < int(dz.A_BUY_BASE):
        return f"Play {def_name(action - int(dz.A_PLAY_BASE))}"
    if int(dz.A_BUY_BASE) <= action < int(dz.A_SELECT_BASE):
        return f"Buy {def_name(action - int(dz.A_BUY_BASE))}"
    if int(dz.A_SELECT_BASE) <= action < int(dz.A_OPTION_BASE):
        return f"Select {def_name(action - int(dz.A_SELECT_BASE))}"
    if int(dz.A_OPTION_BASE) <= action < int(dz.A_CALL_BASE):
        return f"Choose option {action - int(dz.A_OPTION_BASE) + 1}"
    return f"Engine action {action}"


def _observed_event_cards(event: Mapping[str, Any]) -> ObservedCards:
    count = _optional_int(event.get("count"))
    names = tuple(str(value) for value in event.get("cards", ()))
    if count is None:
        return unknown_cards()
    if len(names) != count:
        return ObservedCards(visibility="counts_only", count=count, cards=())
    return known_cards(tuple(_card_from_name(name) for name in names))


def _arena_resources(state: _ArenaState, actor: int | None) -> Resources:
    seat = actor if actor is not None else state.active_seat
    values = state.resources.get(seat, {}) if seat is not None else {}
    if not all(name in values for name in ("actions", "buys", "coins")):
        return Resources(
            visibility="unknown",
            actions=None,
            buys=None,
            coins=None,
        )
    return Resources(
        visibility="known",
        actions=values["actions"],
        buys=values["buys"],
        coins=values["coins"],
    )


def _arena_zone_counts(state: _ArenaState) -> tuple[ZoneCount, ...]:
    by_key: dict[tuple[int | None, str], int] = {}
    for owner, kind, count in state.zones.values():
        if kind in {"hand", "deck", "discard", "in-play", "set-aside", "trash"}:
            by_key[(owner, kind)] = by_key.get((owner, kind), 0) + count

    zones: list[ZoneCount] = []
    for seat in range(state.player_count):
        for zone in ("hand", "deck", "discard", "in-play", "set-aside"):
            count = by_key.get((seat, zone))
            visibility = (
                "counts_only"
                if count is not None
                and seat != state.our_seat
                and zone in {"hand", "deck"}
                else "known"
                if count is not None
                else "unknown"
            )
            zones.append(
                ZoneCount(
                    seat=seat,
                    zone=zone,
                    visibility=visibility,
                    count=count,
                )
            )
    trash_count = by_key.get((None, "trash"))
    zones.append(
        ZoneCount(
            seat=None,
            zone="trash",
            visibility="known" if trash_count is not None else "unknown",
            count=trash_count,
        )
    )
    return tuple(zones)


def _event_actor(
    state: _ArenaState,
    event_type: str,
    event: Mapping[str, Any],
    decision: Mapping[str, Any] | None,
) -> int | None:
    if event.get("seat") is not None:
        return _optional_int(event.get("seat"))
    if event_type == "TurnStart":
        return _optional_int(event.get("seat"))
    if decision is not None:
        return state.our_seat
    return None


def _record_type(event_type: str) -> str:
    if event_type == "TurnStart":
        return "turn"
    if event_type in {"PendingDecision", "DecisionResolved", "ReactionWindow"}:
        return "decision"
    if event_type in _ACTION_EVENTS:
        return "action"
    if event_type in _CONSEQUENCE_EVENTS:
        return "consequence"
    return "state"


def _arena_results(
    *,
    player_count: int,
    scores: tuple[int, ...],
    placings: tuple[int, ...],
    result: Mapping[str, Any],
) -> tuple[SeatResult, ...]:
    if len(scores) != player_count or len(placings) != player_count:
        return tuple(
            SeatResult(
                seat=seat,
                visibility="unknown",
                vp=None,
                placing=None,
                outcome="unknown",
            )
            for seat in range(player_count)
        )
    best = min(placings)
    tied_winners = sum(placing == best for placing in placings) > 1 or bool(
        result.get("tie", False)
    )
    return tuple(
        SeatResult(
            seat=seat,
            visibility="known",
            vp=scores[seat],
            placing=placings[seat],
            outcome=(
                "tie"
                if placings[seat] == best and tied_winners
                else "win"
                if placings[seat] == best
                else "loss"
            ),
        )
        for seat in range(player_count)
    )


def _card_from_name(name: str) -> CardRef:
    card_id = def_id(name)
    return CardRef(def_id=card_id, name=name)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(value)
    return rows


def _read_object_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"non-object JSON file: {path}")
    return value


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _game_id_from_directory(path: Path) -> str:
    marker = "-game-"
    return path.name.split(marker, 1)[1] if marker in path.name else path.name
