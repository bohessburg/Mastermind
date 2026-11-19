from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import dominion_v2_py as dz
import pytest

from src.v2.arena.actuate.clicks import (
    MockActuator,
    gesture_click_targets,
)
from src.v2.arena.fsm.game import (
    DecisionPlan,
    RecordedDecisionProvider,
    run_game_loop,
)
from src.v2.arena.protocol.events import (
    GameStart,
    PendingDecision,
    ReactionWindow,
    UnknownFrame,
)
from src.v2.arena.protocol.frames import DecodedFrame, Direction, Writer
from src.v2.arena.protocol.parser import ArenaParser
from src.v2.arena.protocol.recording import (
    load_recording_sessions,
    parse_recording,
)
from src.v2.arena.shadow.tracker import (
    PendingDecisionSnapshot,
    TrackerSnapshot,
)


MOAT_ARCHIVE = Path(
    "exports/arena/20260724T220259.308785Z/frames.jsonl"
)
MOAT_GAME_ID = 181364786
MOAT_SEQUENCE = 9756
MOAT_QUESTION_INDEX = 87


def _archive_or_skip() -> Path:
    if not MOAT_ARCHIVE.is_file():
        pytest.skip(f"missing live Moat fixture: {MOAT_ARCHIVE}")
    return MOAT_ARCHIVE


def _reaction(
    events: tuple[object, ...],
) -> PendingDecision:
    reactions = [
        event
        for event in events
        if isinstance(event, PendingDecision)
        and event.question_id == "GAME_MAY_REACT_WITH"
    ]
    assert len(reactions) == 1
    return reactions[0]


def test_archived_273_byte_frame_parses_as_moat_reaction() -> None:
    archive = _archive_or_skip()
    sessions = load_recording_sessions(archive)
    frame = next(
        frame
        for session in sessions
        for frame in session.frames
        if frame.sequence == MOAT_SEQUENCE
    )
    assert frame.msg_type == 37
    assert len(frame.payload) == 273

    result = parse_recording(archive)
    decision = _reaction(result.events)
    assert decision.question_index == MOAT_QUESTION_INDEX
    assert decision.decision_type == "REVEAL"
    assert decision.offered == ("Moat",)
    assert (decision.minimum, decision.maximum) == (0, 1)
    assert any(
        isinstance(event, ReactionWindow)
        and event.question_index == MOAT_QUESTION_INDEX
        and event.offered == ("Moat",)
        for event in result.events
    )
    assert not any(
        isinstance(event, UnknownFrame)
        and event.direction == "in"
        and event.msg_type == 37
        for event in result.events
    )


class _ReactionProvider:
    def __init__(
        self,
        events: tuple[object, ...],
        reaction_action: int,
    ) -> None:
        self.recorded = RecordedDecisionProvider(events)
        self.reaction_action = reaction_action
        self.reaction_decision: PendingDecisionSnapshot | None = None
        self.legal_actions: tuple[int, ...] = ()

    async def plan(
        self,
        *,
        frame_index: int,
        game: Any,
        snapshot: TrackerSnapshot,
        decision: PendingDecisionSnapshot,
    ) -> DecisionPlan | None:
        if decision.question_id == "GAME_MAY_REACT_WITH":
            self.reaction_decision = decision
            self.legal_actions = tuple(
                index
                for index, legal in enumerate(game.legal_mask())
                if legal
            )
            assert self.reaction_action in self.legal_actions
            return DecisionPlan(
                engine_actions=(self.reaction_action,),
                gesture_actions=(self.reaction_action,),
            )
        return await self.recorded.plan(
            frame_index=frame_index,
            game=game,
            snapshot=snapshot,
            decision=decision,
        )


@pytest.mark.parametrize(
    ("reaction_action", "answers", "target"),
    (
        (
            int(dz.A_SELECT_BASE + dz.def_id("Moat")),
            (1, 0, 1, 0, 0),
            ("hand", "Moat"),
        ),
        (
            int(dz.A_PASS),
            (0, 1, 0, 0),
            ("decline-button", "GAME_MAY_REACT_WITH"),
        ),
    ),
)
def test_full_archive_replays_through_moat_reaction(
    reaction_action: int,
    answers: tuple[int, ...],
    target: tuple[str, str],
) -> None:
    events = parse_recording(_archive_or_skip()).events
    provider = _ReactionProvider(events, reaction_action)
    actuator = MockActuator(replay=True)

    results = asyncio.run(
        run_game_loop(
            events,
            actuator=actuator,
            decision_provider=provider,
        )
    )

    game = next(result for result in results if result.game_id == MOAT_GAME_ID)
    assert game.completed
    assert not game.divergence_aborted
    assert provider.legal_actions == (
        int(dz.A_PASS),
        int(dz.A_SELECT_BASE + dz.def_id("Moat")),
    )
    assert provider.reaction_decision is not None
    gesture = actuator.gestures[-1]
    assert gesture.question_index == MOAT_QUESTION_INDEX
    assert gesture.answer_indices == answers
    assert [
        (click.region, click.identity)
        for click in gesture_click_targets(
            gesture,
            provider.reaction_decision,
            provider.reaction_decision.offered,
        )
    ] == [target]


class _UnexpectedPolicy:
    async def plan(self, **_: object) -> DecisionPlan:
        raise AssertionError("an unparseable question must abort before planning")


def test_unparseable_question_mid_game_aborts_loudly() -> None:
    payload = Writer().s32(123).u32(99).build()
    parsed = ArenaParser().parse_frame(
        DecodedFrame(
            direction=Direction.INBOUND,
            msg_type=37,
            payload=payload,
            raw=b"",
            sequence=456,
        )
    )
    assert len(parsed) == 1
    assert isinstance(parsed[0], UnknownFrame)
    screenshots: list[tuple[int, int | None]] = []

    def screenshot(
        frame_index: int,
        snapshot: TrackerSnapshot,
    ) -> str:
        screenshots.append((frame_index, snapshot.game_id))
        return "/tmp/unparseable-question.png"

    events = (
        GameStart(
            game_id=999,
            kingdom=("Bandit", "Moat"),
            players=("opponent", "bot"),
            player_ids=(1, 2),
            our_seat=1,
        ),
        parsed[0],
    )
    actuator = MockActuator()
    results = asyncio.run(
        run_game_loop(
            events,
            actuator=actuator,
            decision_provider=_UnexpectedPolicy(),
            screenshot_hook=screenshot,
        )
    )

    assert len(results) == 1
    result = results[0]
    assert result.game_id == 999
    assert result.divergence_aborted
    assert not result.completed
    assert "unparseable inbound questionAsked" in result.reason
    assert result.divergence_report is not None
    assert result.divergence_report.error_type == "DivergenceError"
    assert result.divergence_report.question_index == 123
    assert result.divergence_report.screenshot_path == (
        "/tmp/unparseable-question.png"
    )
    assert screenshots == [(1, 999)]
    assert actuator.stopped
    assert actuator.gestures == []


COPY_FIELDS_ARCHIVE = Path("exports/arena/20261111T151610.685744Z/frames.jsonl")
COPY_FIELDS_GAME_ID = 185812617
COPY_FIELDS_QUESTION_INDEX = 96
MOAT_INSTANCE = 209
WITCH_INSTANCE = 178


def _moat_reaction_payload(*, copy_fields: bool) -> bytes:
    """The Moat reaction question as the server sends it.

    Field for field the 281-byte frame of live game 185812617 (web client
    bundle 2.3.7).  Without ``copy_fields`` it is the 273-byte layout that
    bundle 2.2.8 recordings carry: the same frame minus the copyIndex and
    copyTotal that 2.3.7 appends to every ability offered as an element.
    """
    writer = Writer()

    def empty_story(question_id: int) -> None:
        writer.u32(question_id).u32(0).boolean(False)

    def ability(name: int, association: int) -> None:
        writer.u32(1).u32(name).s32(association)

    writer.s32(COPY_FIELDS_QUESTION_INDEX).u32(2)
    writer.u32(19).s32(-1)
    empty_story(1)
    empty_story(1)
    writer.u32(3)

    # May resolve ability: the Moat, in response to the Witch being played.
    writer.u32(0).u32(7).s32(-1)
    writer.u32(1).u32(1).u32(8)
    ability(101, MOAT_INSTANCE)
    writer.boolean(True).u32(9).u32(1)
    ability(3, WITCH_INSTANCE)
    writer.u32(2).u32(1).s32(1).u32(0).u32(1).u32(32).s32(1)
    empty_story(356)
    writer.s32(0).s32(1)
    writer.u32(1).u32(1)
    ability(101, MOAT_INSTANCE)
    writer.boolean(True)
    if copy_fields:
        writer.s32(0).s32(1)
    writer.s32(4).u32(0).u32(0)

    # Reject ability: a number question, 0 to 1, default 0.
    writer.u32(1).u32(49).s32(-1)
    empty_story(0)
    empty_story(0)
    writer.s32(0).s32(1).s32(0).s32(-1)

    # Way: nothing offered.
    writer.u32(0).u32(43).s32(-1)
    empty_story(0)
    empty_story(0)
    writer.s32(0).s32(0).u32(0).s32(-1).u32(0).u32(0)
    return writer.build()


@pytest.mark.parametrize(("copy_fields", "size"), ((True, 281), (False, 273)))
def test_moat_reaction_parses_with_and_without_ability_copy_fields(
    copy_fields: bool,
    size: int,
) -> None:
    payload = _moat_reaction_payload(copy_fields=copy_fields)
    assert len(payload) == size
    parser = ArenaParser()
    parser.card_by_instance[MOAT_INSTANCE] = "Moat"

    parsed = parser.parse_frame(
        DecodedFrame(
            direction=Direction.INBOUND,
            msg_type=37,
            payload=payload,
            raw=b"",
            sequence=13707,
        )
    )

    assert [type(event) for event in parsed] == [PendingDecision, ReactionWindow]
    decision = parsed[0]
    assert isinstance(decision, PendingDecision)
    assert decision.question_index == COPY_FIELDS_QUESTION_INDEX
    assert decision.question_id == "GAME_MAY_REACT_WITH"
    assert decision.decision_type == "REVEAL"
    assert decision.offered == ("Moat",)
    assert (decision.minimum, decision.maximum) == (0, 1)


def test_live_281_byte_frame_parses_as_moat_reaction() -> None:
    """Game 185812617: the opponent played Witch while the bot held a Moat.

    The question did not parse and the run aborted on the opponent's turn.
    """
    if not COPY_FIELDS_ARCHIVE.is_file():
        pytest.skip(f"missing live Moat fixture: {COPY_FIELDS_ARCHIVE}")
    events = parse_recording(COPY_FIELDS_ARCHIVE).events
    start = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, GameStart) and event.game_id == COPY_FIELDS_GAME_ID
    )
    reactions = [
        event
        for event in events[start:]
        if isinstance(event, PendingDecision)
        and event.question_id == "GAME_MAY_REACT_WITH"
    ]
    assert [reaction.question_index for reaction in reactions] == [
        COPY_FIELDS_QUESTION_INDEX
    ]
    assert reactions[0].offered == ("Moat",)
    assert any(
        isinstance(event, ReactionWindow)
        and event.question_index == COPY_FIELDS_QUESTION_INDEX
        for event in events[start:]
    )
    assert not any(
        isinstance(event, UnknownFrame)
        and event.direction == "in"
        and event.msg_type == 37
        for event in events
    )


def test_live_moat_question_reaches_a_legal_reveal_in_the_shadow_game() -> None:
    if not COPY_FIELDS_ARCHIVE.is_file():
        pytest.skip(f"missing live Moat fixture: {COPY_FIELDS_ARCHIVE}")
    events = parse_recording(COPY_FIELDS_ARCHIVE).events
    start = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, GameStart) and event.game_id == COPY_FIELDS_GAME_ID
    )
    game_events = events[start:]
    reveal = int(dz.A_SELECT_BASE + dz.def_id("Moat"))
    provider = _ReactionProvider(game_events, reveal)
    actuator = MockActuator(replay=True)

    results = asyncio.run(
        run_game_loop(
            game_events,
            actuator=actuator,
            decision_provider=provider,
        )
    )

    assert not any(result.divergence_aborted for result in results)
    assert provider.legal_actions == (int(dz.A_PASS), reveal)
    gesture = actuator.gestures[-1]
    assert gesture.question_index == COPY_FIELDS_QUESTION_INDEX
    assert gesture.answer_indices == (1, 0, 1, 0, 0)
