from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from src.v2.arena.actuate.clicks import MockActuator
from src.v2.arena.archive import GameArchive
from src.v2.arena.fsm.game import GameRunResult, run_game_loop
from src.v2.arena.main import (
    EXIT_DIVERGENCE,
    EXIT_LOBBY,
    EXIT_STALL,
    EXIT_UNEXPECTED,
    _exit_code_for_game_result,
    write_exit_summary,
)
from src.v2.arena.protocol.events import (
    FullState,
    GameEnd,
    GameResult,
    GameStart,
    PendingDecision,
    ResourceUpdate,
    TurnStart,
)
from src.v2.arena.protocol.frames import DecodedFrame, Direction
from src.v2.arena.protocol.parser import ArenaParser
from src.v2.arena.protocol.recording import parse_recording


FIVE_GAME_ARCHIVE = Path(
    "exports/arena/20260724T220259.308785Z/frames.jsonl"
)
EXPECTED_RESULTS = {
    181364095: ((33, 44), (2, 1), 1, "win"),
    181364271: ((25, 39), (2, 1), 1, "loss"),
    181364518: ((27, 28), (2, 1), 1, "loss"),
    # Rank is authoritative in these equal-VP resignations.
    181364739: ((3, 3), (2, 1), 1, "win"),
    181364786: ((3, 3), (1, 2), 0, "loss"),
}


class _UnusedProvider:
    async def plan(self, **_: Any) -> None:
        raise AssertionError("synthetic terminal streams contain no decisions")


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        await asyncio.sleep(0)


def _start(*, our_seat: int = 0) -> GameStart:
    return GameStart(
        game_id=999,
        kingdom=("Chapel",),
        players=("bot", "opponent"),
        player_ids=(1, 2),
        our_seat=our_seat,
        timestamp_ms=1_000,
    )


async def _our_turn_then_silence() -> AsyncIterator[object]:
    yield _start()
    yield TurnStart(
        seat=0,
        turn_number=1,
        turn_type=0,
        controller_seat=0,
        timestamp_ms=2_000,
    )
    await asyncio.Event().wait()


def test_our_turn_watchdog_archives_stall_and_maps_to_exit_3(
    tmp_path: Path,
) -> None:
    clock = _FakeClock()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    async def screenshot(frame_index: int, _snapshot: object) -> Path:
        destination = run_dir / f"stall-{frame_index}.png"
        destination.write_bytes(b"png")
        return destination

    async def dom(frame_index: int, _snapshot: object) -> Path:
        destination = run_dir / f"stall-{frame_index}.html"
        destination.write_text("<html>stall</html>", encoding="utf-8")
        return destination

    results = asyncio.run(
        run_game_loop(
            _our_turn_then_silence(),
            actuator=MockActuator(replay=True),
            decision_provider=_UnusedProvider(),
            archive_factory=lambda game_id: GameArchive(
                run_dir,
                game_id=game_id,
            ),
            stall_watchdog_seconds=2.0,
            watchdog_poll_seconds=0.5,
            clock=clock,
            sleep=clock.sleep,
            stall_screenshot_hook=screenshot,
            stall_dom_hook=dom,
        )
    )

    assert len(results) == 1
    result = results[0]
    assert result.stall_aborted
    assert not result.divergence_aborted
    assert _exit_code_for_game_result(result) == EXIT_STALL
    assert result.stall_report is not None
    assert result.stall_report.frame_index == 2
    assert result.stall_report.last_event_timestamp_ms == 2_000
    assert result.stall_report.tracker_summary["turn_owner"] == 0
    assert result.stall_report.pending_question is None
    assert result.archive_dir is not None
    stall = json.loads(
        (Path(result.archive_dir) / "stall.json").read_text(encoding="utf-8")
    )
    assert stall["silent_seconds"] >= 2.0
    assert Path(stall["screenshot_path"]).is_file()
    assert Path(stall["dom_snapshot_path"]).is_file()


async def _opponent_turn_then_end(clock: _FakeClock) -> AsyncIterator[object]:
    yield _start()
    yield TurnStart(
        seat=1,
        turn_number=1,
        turn_type=0,
        controller_seat=1,
        timestamp_ms=2_000,
    )
    await clock.sleep(20.0)
    yield GameEnd(
        game_id=999,
        reason="opponent-finished-thinking",
        timestamp_ms=22_000,
    )


def test_opponent_turn_silence_does_not_trigger_watchdog() -> None:
    clock = _FakeClock()
    results = asyncio.run(
        run_game_loop(
            _opponent_turn_then_end(clock),
            actuator=MockActuator(replay=True),
            decision_provider=_UnusedProvider(),
            stall_watchdog_seconds=2.0,
            watchdog_poll_seconds=0.5,
            clock=clock,
            sleep=clock.sleep,
        )
    )

    assert len(results) == 1
    assert results[0].completed
    assert not results[0].stall_aborted


async def _start_question_then_silence() -> AsyncIterator[object]:
    yield _start()
    yield PendingDecision(
        question_index=7,
        decision_type="CHOOSE_MODE",
        question_id="question-session-relative",
        offered=("card-mode-97",),
        minimum=1,
        maximum=1,
        association=None,
        timestamp_ms=3_000,
    )
    await asyncio.Event().wait()


def test_unresolved_pre_game_question_is_a_watchdog_obligation() -> None:
    clock = _FakeClock()
    results = asyncio.run(
        run_game_loop(
            _start_question_then_silence(),
            actuator=MockActuator(replay=True),
            decision_provider=_UnusedProvider(),
            stall_watchdog_seconds=1.0,
            watchdog_poll_seconds=0.25,
            clock=clock,
            sleep=clock.sleep,
        )
    )

    assert results[0].stall_aborted
    assert results[0].stall_report is not None
    assert results[0].stall_report.pending_question is not None
    assert results[0].stall_report.pending_question["question_index"] == 7


@pytest.mark.parametrize(
    ("exit_code", "reason"),
    [
        (EXIT_UNEXPECTED, "unexpected-crash"),
        (EXIT_STALL, "stall-watchdog"),
        (EXIT_DIVERGENCE, "divergence-or-actuation"),
        (EXIT_LOBBY, "lobby-error"),
    ],
)
def test_exit_summary_writer_covers_every_nonzero_contract_code(
    tmp_path: Path,
    exit_code: int,
    reason: str,
) -> None:
    destination = write_exit_summary(
        tmp_path,
        exit_code=exit_code,
        exit_reason=reason,
        game_id=123,
        archive_dir=tmp_path / "game",
    )

    assert destination == tmp_path / "exit.json"
    assert destination.read_text(encoding="utf-8").count("\n") == 1
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "archive_dir": str(tmp_path / "game"),
        "exit_code": exit_code,
        "exit_reason": reason,
        "game_id": 123,
    }


def test_message_14_golden_scores_match_final_feed_points() -> None:
    assert FIVE_GAME_ARCHIVE.is_file()
    current_game: int | None = None
    points: dict[int, int] = {}
    observed: dict[int, GameResult] = {}

    for event in parse_recording(FIVE_GAME_ARCHIVE).events:
        if isinstance(event, GameStart):
            current_game = event.game_id
            points = {}
        elif isinstance(event, FullState):
            for counter in event.counters:
                if counter.name == "points" and counter.owner is not None:
                    points[counter.owner] = counter.value
        elif (
            isinstance(event, ResourceUpdate)
            and event.resource == "points"
            and event.seat is not None
        ):
            points[event.seat] = event.value
        elif isinstance(event, GameResult):
            assert event.game_id == current_game
            assert event.decoded, event.error
            assert event.scores == tuple(
                points[seat] for seat in range(len(event.scores))
            )
            observed[event.game_id] = event

    assert set(observed) == set(EXPECTED_RESULTS)
    for game_id, (scores, placings, winner, _outcome) in EXPECTED_RESULTS.items():
        assert observed[game_id].scores == scores
        assert observed[game_id].placings == placings
        assert observed[game_id].winner_seat == winner
        assert not observed[game_id].tie


def test_completed_games_append_correct_ledger_outcome_from_our_seat(
    tmp_path: Path,
) -> None:
    parsed = parse_recording(FIVE_GAME_ARCHIVE).events
    terminal_events = tuple(
        event
        for event in parsed
        if isinstance(event, (GameStart, GameResult, GameEnd))
    )
    run_dir = tmp_path / "session"
    run_dir.mkdir()
    ledger = tmp_path / "record.jsonl"

    results = asyncio.run(
        run_game_loop(
            terminal_events,
            actuator=MockActuator(replay=True),
            decision_provider=_UnusedProvider(),
            archive_factory=lambda game_id: GameArchive(
                run_dir,
                game_id=game_id,
                ledger_path=ledger,
            ),
        )
    )

    assert [result.outcome for result in results] == [
        expected[3] for expected in EXPECTED_RESULTS.values()
    ]
    lines = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
    ]
    assert len(lines) == 5
    for line, result in zip(lines, results, strict=True):
        assert line["game_id"] == result.game_id
        assert line["opponent"] == result.opponent
        assert line["our_seat"] == result.our_seat
        assert line["result"] == result.outcome
        assert line["scores"] == list(result.scores)
        result_json = json.loads(
            (Path(result.archive_dir) / "result.json").read_text(encoding="utf-8")
        )
        assert result_json["outcome"] == result.outcome
        assert result_json["scores"] == list(result.scores)
        assert result_json["opponent"] == result.opponent


def test_undecodable_message_14_records_unknown_and_ledger_never_raises(
    tmp_path: Path,
) -> None:
    parser = ArenaParser()
    parser.game_id = 999
    parser.player_ids = (1, 2)
    parser.our_seat = 0
    decoded = parser.parse_frame(
        DecodedFrame(
            direction=Direction.INBOUND,
            msg_type=14,
            payload=b"not-a-game-result",
            raw=b"",
            sequence=4,
            timestamp_ms=4_000,
        )
    )
    game_result = next(
        event for event in decoded if isinstance(event, GameResult)
    )
    assert not game_result.decoded
    assert game_result.error

    run_dir = tmp_path / "session"
    run_dir.mkdir()
    # A directory is deliberately not appendable as a file; the ledger path
    # must log the error without affecting the completed game result.
    bad_ledger = tmp_path / "bad-ledger"
    bad_ledger.mkdir()
    events = (_start(), game_result)
    results = asyncio.run(
        run_game_loop(
            events,
            actuator=MockActuator(replay=True),
            decision_provider=_UnusedProvider(),
            archive_factory=lambda game_id: GameArchive(
                run_dir,
                game_id=game_id,
                ledger_path=bad_ledger,
            ),
        )
    )

    assert results[0].completed
    assert results[0].outcome == "unknown"
