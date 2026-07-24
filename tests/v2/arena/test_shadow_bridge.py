from __future__ import annotations

from collections import Counter
from pathlib import Path

import dominion_v2_py as dz

from src.v2.arena.protocol.events import Buy, PendingDecision, Play
from src.v2.arena.protocol.recording import parse_recording
from src.v2.arena.shadow.bridge import game_from_snapshot
from src.v2.arena.shadow.tracker import Tracker


RECORDING = Path(
    "arena-recordings/20260724T142103.096991Z/frames.jsonl"
)
PHASE_QUESTIONS = frozenset({"GAME_ACTION_PHASE", "GAME_BUY_PHASE"})


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
