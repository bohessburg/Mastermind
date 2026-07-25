"""Convert seed-replay web exports into unified game records."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import dominion_v2_py as dz

from src.v2.web.server.defs import def_name
from src.v2.web.server.observer import action_category, label_for_action

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
)


PHASE_NAMES = {
    0: "action",
    1: "buy",
    2: "night",
    3: "cleanup",
    4: "over",
}


@dataclass(frozen=True, kw_only=True)
class _LocalSnapshot:
    hands: tuple[Counter[int], ...]
    decks: tuple[int, ...]
    discards: tuple[Counter[int], ...]
    in_play: tuple[Counter[int], ...]
    set_aside: tuple[Counter[int], ...]
    supply: Counter[int]
    trash: Counter[int]


def is_local_export_data(data: object) -> bool:
    """Return whether a decoded object has the unchanged web export shape."""
    return (
        isinstance(data, dict)
        and isinstance(data.get("seed"), int)
        and isinstance(data.get("kingdom"), list)
        and isinstance(data.get("seats"), list)
        and isinstance(data.get("actions"), list)
    )


def is_complete_local_export_data(data: object) -> bool:
    """Return whether an export is a complete replay under this engine build."""
    if not is_local_export_data(data):
        return False
    assert isinstance(data, dict)
    if not data["actions"]:
        return False
    try:
        game = dz.new_game(
            dz.Setup(
                players=len(data["seats"]),
                kingdom=[int(value) for value in data["kingdom"]],
            ),
            int(data["seed"]),
        )
        for raw_action in data["actions"]:
            action = int(raw_action)
            mask = game.legal_mask()
            if action < 0 or action >= len(mask) or not bool(mask[action]):
                return False
            game.step(action)
        expected = data.get("final_state_hash")
        return (
            bool(game.game_over())
            and (
                expected is None
                or str(expected).lower() == f"0x{int(game.state_hash()):016x}"
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def convert_local_export(path: Path | str) -> GameRecord:
    """Replay one web export and derive every ordered action record."""
    source_path = Path(path)
    data = json.loads(source_path.read_text(encoding="utf-8"))
    if not is_local_export_data(data):
        raise ValueError(f"not a local web export: {source_path}")
    return convert_local_data(data, provenance=str(source_path), game_id=source_path.stem)


def convert_local_data(
    data: Mapping[str, Any],
    *,
    provenance: str,
    game_id: str,
) -> GameRecord:
    """Convert already decoded local export data."""
    kingdom_ids = tuple(int(value) for value in data["kingdom"])
    seat_kinds = tuple(str(value) for value in data["seats"])
    setup = dz.Setup(players=len(seat_kinds), kingdom=list(kingdom_ids))
    game = dz.new_game(setup, int(data["seed"]))
    records: list[ActionRecord] = []

    for source_index, raw_action in enumerate(data["actions"]):
        action = int(raw_action)
        mask = game.legal_mask()
        if action < 0 or action >= len(mask) or not bool(mask[action]):
            raise ValueError(
                f"illegal local action {action} at index {source_index}"
            )
        decision = dict(game.current_decision())
        context = game.decision_context()
        actor_seat = int(decision["player"])
        turn_counter = int(game.turn())
        active_seat = turn_counter % len(seat_kinds)
        turn_number = turn_counter // len(seat_kinds) + 1
        phase = PHASE_NAMES.get(int(game.phase()), f"phase-{int(game.phase())}")
        label, _ = label_for_action(action, decision, context)
        before = _snapshot(game)

        game.step(action)
        after = _snapshot(game)
        records.append(
            _local_action_record(
                game=game,
                before=before,
                after=after,
                index=len(records),
                source_index=source_index,
                action=action,
                label=label,
                turn_number=turn_number,
                active_seat=active_seat,
                phase=phase,
                actor_seat=actor_seat,
            )
        )

    expected_hash = data.get("final_state_hash")
    actual_hash = f"0x{int(game.state_hash()):016x}"
    if expected_hash is not None and str(expected_hash).lower() != actual_hash:
        raise ValueError(
            f"local replay hash mismatch: {actual_hash} != {expected_hash}"
        )

    scores = tuple(int(game.score(seat)) for seat in range(len(seat_kinds)))
    results = _results_from_scores(scores)
    seats = tuple(
        SeatInfo(
            index=index,
            kind=kind,
            display_name=None,
            controlled=False,
            bot=kind.startswith("bot"),
        )
        for index, kind in enumerate(seat_kinds)
    )
    return GameRecord(
        schema_version=SCHEMA_VERSION,
        source="local",
        provenance=provenance,
        game_id=game_id,
        timestamp=None,
        timestamp_visibility="unknown",
        kingdom=tuple(_card(def_id) for def_id in kingdom_ids),
        player_count=len(seat_kinds),
        seats=seats,
        obs_version=int(data["obs_version"]) if data.get("obs_version") is not None else None,
        obs_version_visibility=(
            "known" if data.get("obs_version") is not None else "unknown"
        ),
        controlled_seat=None,
        controlled_seat_visibility="unknown",
        results=results,
        records=tuple(records),
    )


def _snapshot(game: Any) -> _LocalSnapshot:
    players = range(int(game.num_players()))
    return _LocalSnapshot(
        hands=tuple(_counter(game.hand(seat)) for seat in players),
        decks=tuple(int(game.deck_count(seat)) for seat in players),
        discards=tuple(Counter(int(card) for card in game.discard(seat)) for seat in players),
        in_play=tuple(Counter(int(card) for card in game.in_play(seat)) for seat in players),
        set_aside=tuple(
            Counter(int(card) for card in game.set_aside(seat)) for seat in players
        ),
        supply=Counter(
            {int(card): int(count) for card, count in game.supply()}
        ),
        trash=Counter(
            {int(card): int(count) for card, count in game.trash().items()}
        ),
    )


def _counter(values: Mapping[int, int]) -> Counter[int]:
    return Counter({int(card): int(count) for card, count in values.items()})


def _positive_delta(after: Counter[int], before: Counter[int]) -> Counter[int]:
    return Counter(
        {
            card: count - before.get(card, 0)
            for card, count in after.items()
            if count > before.get(card, 0)
        }
    )


def _cards_from_counter(values: Counter[int]) -> tuple[CardRef, ...]:
    return tuple(
        _card(card)
        for card, count in sorted(values.items())
        for _ in range(max(0, count))
    )


def _local_action_record(
    *,
    game: Any,
    before: _LocalSnapshot,
    after: _LocalSnapshot,
    index: int,
    source_index: int,
    action: int,
    label: str,
    turn_number: int,
    active_seat: int,
    phase: str,
    actor_seat: int,
) -> ActionRecord:
    played_counts = Counter()
    for before_zone, after_zone in zip(before.in_play, after.in_play):
        played_counts.update(_positive_delta(after_zone, before_zone))

    bought_counts: Counter[int] = Counter()
    if action_category(action) == "buy":
        bought_counts[action - int(dz.A_BUY_BASE)] += 1

    gained_counts = _positive_delta(before.supply, after.supply)
    trashed_counts = _positive_delta(after.trash, before.trash)
    discarded_counts = Counter()
    for before_zone, after_zone in zip(before.discards, after.discards):
        discarded_counts.update(_positive_delta(after_zone, before_zone))
    # A gain to discard is represented by ``gained``, not also by ``discarded``.
    for card, count in gained_counts.items():
        discarded_counts[card] = max(0, discarded_counts[card] - count)

    resources = game.resources()
    return ActionRecord(
        index=index,
        source_index=source_index,
        timestamp_ms=None,
        timestamp_ms_visibility="unknown",
        record_type="action",
        event="EngineAction",
        turn_number=turn_number,
        turn_number_visibility="known",
        active_seat=active_seat,
        active_seat_visibility="known",
        phase=phase,
        phase_visibility="known",
        actor_seat=actor_seat,
        actor_seat_visibility="known",
        engine_action_ids=(action,),
        engine_action_ids_visibility="known",
        action_labels=(label,),
        action_labels_visibility="known",
        played=known_cards(_cards_from_counter(played_counts)),
        bought=known_cards(_cards_from_counter(bought_counts)),
        gained=known_cards(_cards_from_counter(gained_counts)),
        trashed=known_cards(_cards_from_counter(trashed_counts)),
        discarded=known_cards(_cards_from_counter(discarded_counts)),
        resources_after=Resources(
            visibility="known",
            actions=int(resources["actions"]),
            buys=int(resources["buys"]),
            coins=int(resources["coins"]),
        ),
        zone_counts_after=_local_zone_counts(after),
    )


def _local_zone_counts(snapshot: _LocalSnapshot) -> tuple[ZoneCount, ...]:
    zones: list[ZoneCount] = []
    for seat in range(len(snapshot.hands)):
        for name, count in (
            ("hand", sum(snapshot.hands[seat].values())),
            ("deck", snapshot.decks[seat]),
            ("discard", sum(snapshot.discards[seat].values())),
            ("in-play", sum(snapshot.in_play[seat].values())),
            ("set-aside", sum(snapshot.set_aside[seat].values())),
        ):
            zones.append(
                ZoneCount(
                    seat=seat,
                    zone=name,
                    visibility="known",
                    count=int(count),
                )
            )
    zones.append(
        ZoneCount(
            seat=None,
            zone="trash",
            visibility="known",
            count=sum(snapshot.trash.values()),
        )
    )
    return tuple(zones)


def _results_from_scores(scores: tuple[int, ...]) -> tuple[SeatResult, ...]:
    ordered = sorted(set(scores), reverse=True)
    placings = tuple(ordered.index(score) + 1 for score in scores)
    best = max(scores)
    winners = sum(score == best for score in scores)
    return tuple(
        SeatResult(
            seat=seat,
            visibility="known",
            vp=score,
            placing=placings[seat],
            outcome=(
                "tie"
                if score == best and winners > 1
                else "win"
                if score == best
                else "loss"
            ),
        )
        for seat, score in enumerate(scores)
    )


def _card(def_id: int) -> CardRef:
    return CardRef(def_id=int(def_id), name=def_name(int(def_id)))
