from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import dominion_v2_py as dz

from src.v2.arena.protocol import events as protocol_events
from src.v2.arena.protocol.events import (
    Buy,
    FullStateCounter,
    FullStateZone,
    PendingDecision,
    Play,
)
from src.v2.arena.protocol.recording import parse_recording
from src.v2.arena.shadow.bridge import engine_snapshot, game_from_snapshot
from src.v2.arena.shadow.tracker import Tracker


RECORDING = Path(
    "arena-recordings/20260724T142103.096991Z/frames.jsonl"
)
PHASE_QUESTIONS = frozenset({"GAME_ACTION_PHASE", "GAME_BUY_PHASE"})
ARCHIVED_ORDER_REGRESSION = Path(
    "exports/arena/20260727T195055.733005Z/"
    "20260727T204214.059414Z-game-181476737/events.jsonl"
)
CANONICAL_BASE_SUPPLY = (
    "Copper",
    "Silver",
    "Gold",
    "Estate",
    "Duchy",
    "Province",
    "Curse",
)

# Keep these derived from encoder.h's fixed v2 layout so this regression checks
# the actual supply block instead of only Game.supply()'s diagnostic view.
_OBS_V2_SUPPLY_OFFSET = 4 + (5 * 64) + (3 * (75 + (3 * 64)))
_OBS_PILE_BLOCK_SIZE = 11
_OBS_V2_TURN_OFFSET = _OBS_V2_SUPPLY_OFFSET + (48 * _OBS_PILE_BLOCK_SIZE) + 27 + 12

_ARCHIVE_TUPLE_FIELDS = {
    "Attack": ("targets",),
    "Buy": ("cards",),
    "DecisionResolved": ("answers",),
    "Discard": ("cards",),
    "Draw": ("cards",),
    "Gain": ("cards",),
    "GameResult": ("scores", "placings"),
    "GameStart": ("kingdom", "players", "player_ids"),
    "PendingDecision": ("offered",),
    "PileReorder": ("cards",),
    "Play": ("cards",),
    "ZoneTransfer": ("cards",),
}


def _event_from_archive_row(row: dict[str, object]) -> object:
    """Rehydrate one canonical arena archive event without rewriting it."""
    event_type = str(row["event_type"])
    payload = dict(row["event"])
    if event_type == "FullState":
        payload["card_counts"] = tuple(
            (str(name), int(count))
            for name, count in payload["card_counts"]
        )
        payload["zones"] = tuple(
            FullStateZone(
                index=int(zone["index"]),
                kind=str(zone["kind"]),
                owner=zone["owner"],
                display_name=zone["display_name"],
                contents=tuple(str(card) for card in zone["contents"]),
                anonymous_count=int(zone["anonymous_count"]),
            )
            for zone in payload["zones"]
        )
        payload["counters"] = tuple(
            FullStateCounter(
                index=int(counter["index"]),
                name=str(counter["name"]),
                owner=counter["owner"],
                value=int(counter["value"]),
            )
            for counter in payload["counters"]
        )
    else:
        for field in _ARCHIVE_TUPLE_FIELDS.get(event_type, ()):
            payload[field] = tuple(payload[field])
    event_class = getattr(protocol_events, event_type)
    return event_class(**payload)


def _archived_second_buy_snapshot() -> object:
    assert ARCHIVED_ORDER_REGRESSION.is_file(), (
        f"missing required archived order fixture: {ARCHIVED_ORDER_REGRESSION}"
    )
    tracker = Tracker()
    buy_questions = 0
    for line in ARCHIVED_ORDER_REGRESSION.read_text(encoding="utf-8").splitlines():
        event = _event_from_archive_row(json.loads(line))
        tracker.consume(event)
        if isinstance(event, PendingDecision) and event.question_id == "GAME_BUY_PHASE":
            buy_questions += 1
            if buy_questions == 2:
                snapshot = tracker.snapshot()
                assert snapshot.pending_decision is not None
                assert snapshot.pending_decision.question_index == 4
                return snapshot
    raise AssertionError("archive has no second GAME_BUY_PHASE decision")


def _encoded_supply_defs(observation: object, count: int) -> tuple[int, ...]:
    return tuple(
        int(observation[_OBS_V2_SUPPLY_OFFSET + (pile * _OBS_PILE_BLOCK_SIZE) + 1]) - 1
        for pile in range(count)
    )


def _taken_phase_action(
    events: tuple[object, ...],
    index: int,
    our_seat: int,
) -> Play | Buy | None:
    for event in events[index + 1 :]:
        if isinstance(event, PendingDecision):
            return None
        if isinstance(event, (Play, Buy)) and event.seat == our_seat:
            return event
    return None


def test_recording_decisions_build_valid_legal_shadow_games() -> None:
    assert RECORDING.is_file(), f"missing required arena fixture: {RECORDING}"
    events = parse_recording(RECORDING).events
    tracker = Tracker()
    covered: Counter[str] = Counter()

    for index, event in enumerate(events):
        tracker.consume(event)
        if not isinstance(event, PendingDecision):
            continue
        snapshot = tracker.snapshot()
        if not (
            snapshot.pending_decision is not None
            and snapshot.our_seat is not None
            and snapshot.turn_owner == snapshot.our_seat
            and snapshot.phase in {"action", "buy"}
        ):
            continue

        game = game_from_snapshot(snapshot)
        assert game.validate()
        legal = game.legal_mask()
        assert sum(bool(value) for value in legal) > 0
        covered["decision_points"] += 1

        # Mid-card questions need the rigged-stepping shadow frame from P5;
        # P3 intentionally rebuilds them as clean public states. Count them
        # explicitly, while phase plays/buys below have direct engine actions.
        if event.question_id not in PHASE_QUESTIONS:
            covered["excluded_internal_questions"] += 1
            continue

        taken = _taken_phase_action(events, index, snapshot.our_seat)
        if taken is None:
            covered["passes_or_unmapped_phase_answers"] += 1
            continue

        if isinstance(taken, Buy):
            covered["buy_decisions"] += 1
            for name in taken.cards:
                action = int(dz.A_BUY_BASE + dz.def_id(name))
                assert legal[action], (event.question_index, name, action)
                covered["buy_actions"] += 1
        elif event.question_id == "GAME_ACTION_PHASE":
            covered["action_play_decisions"] += 1
            for name in taken.cards:
                action = int(dz.A_PLAY_BASE + dz.def_id(name))
                assert legal[action], (event.question_index, name, action)
                covered["action_play_actions"] += 1
        else:
            covered["treasure_play_decisions"] += 1
            for name in taken.cards:
                action = int(dz.A_PLAY_BASE + dz.def_id(name))
                assert legal[action], (event.question_index, name, action)
                covered["treasure_play_actions"] += 1

    assert covered == Counter(
        {
            "decision_points": 930,
            "excluded_internal_questions": 337,
            "action_play_decisions": 305,
            "action_play_actions": 305,
            "buy_decisions": 130,
            "buy_actions": 130,
            "treasure_play_decisions": 94,
            "treasure_play_actions": 94,
            "passes_or_unmapped_phase_answers": 64,
        }
    )


def test_archived_game_181476737_rebuild_keeps_dealt_supply_order() -> None:
    snapshot = _archived_second_buy_snapshot()
    expected_names = (*CANONICAL_BASE_SUPPLY, *snapshot.kingdom)
    expected_defs = tuple(int(dz.def_id(name)) for name in expected_names)
    expected_def_sorted = (*expected_defs[:7], *sorted(expected_defs[7:]))

    # The server's TurnDescription is one-based; the engine's first turn is 0.
    assert snapshot.turn_number == 1
    native_snapshot = engine_snapshot(snapshot)
    assert tuple(native_snapshot["supply"]) == expected_defs
    assert tuple(native_snapshot["kingdom_order"]) == expected_defs[7:]
    assert native_snapshot["turn_number"] == 0

    game = game_from_snapshot(snapshot)
    assert game.turn() == 0
    assert tuple(def_id for def_id, _ in game.supply()) == expected_defs
    observation = game.encode(snapshot.our_seat, 3)
    assert _encoded_supply_defs(observation, len(expected_defs)) == expected_defs
    assert _encoded_supply_defs(observation, len(expected_defs)) != expected_def_sorted
    assert observation[_OBS_V2_TURN_OFFSET + 7] == 0.0

    # Seed 778 reaches the same visible opening after passing Action and
    # playing its three Coppers. This catches both the pile ABI and the arena
    # one-based turn conversion without loading a checkpoint or torch.
    organic = dz.new_game(
        dz.Setup(players=2, kingdom=list(snapshot.kingdom)),
        778,
    )
    copper = int(dz.def_id("Copper"))
    estate = int(dz.def_id("Estate"))
    assert organic.hand(0) == {copper: 3, estate: 2}
    organic.step(int(dz.A_PASS))
    copper_play = int(dz.A_PLAY_BASE + copper)
    while organic.legal_mask()[copper_play]:
        organic.step(copper_play)
    assert organic.turn() == game.turn() == 0
    assert (organic.encode(0, 3) == observation).all()
