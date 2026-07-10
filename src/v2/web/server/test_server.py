from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

import dominion_v2_py as dz

from src.v2.web.server.main import (
    Seat,
    Session,
    _decision_message,
    _apply_action,
    _post_step_messages,
    _undo_previous_human_decision,
    _apply_validated_action,
    app,
    sessions,
)
from src.v2.web.server.observer import (
    BanditLogState,
    PlayerPublicSnapshot,
    PublicSnapshot,
    SentryLogState,
    bandit_resolution_logs,
    log_line,
    public_log_lines,
    sentry_resolution_logs,
)
from tests.v2.replay_export import verify_export_data


KINGDOM = [
    "Sentry",
    "Library",
    "Throne Room",
    "Bandit",
    "Witch",
    "Moat",
    "Village",
    "Smithy",
    "Market",
    "Remodel",
]
FIXTURE_THRONE_BANDIT = Path(__file__).with_name("fixtures_throne_bandit_replay.json")


@pytest.fixture
def tiny_nn_checkpoint(tmp_path: Path) -> Path:
    """A checkpoint with the same model/config keys as training output."""
    import torch

    from src.v2.train.model import DominionNet

    torch.manual_seed(0x5105)
    model = DominionNet(dz.OBS_SIZE, dz.ACTION_SPACE_SIZE, hidden_sizes=[32])
    checkpoint = tmp_path / "tiny-random-policy.pt"
    torch.save(
        {
            "config": {"model": {"hidden_sizes": [32]}},
            "model": model.state_dict(),
        },
        checkpoint,
    )
    return checkpoint


def by_type(messages: list[dict], message_type: str) -> dict:
    for message in messages:
        if message["type"] == message_type:
            return message
    raise AssertionError(f"missing {message_type}: {messages}")


def read_initial(websocket) -> list[dict]:
    return [websocket.receive_json() for _ in range(3)]


def read_update(websocket) -> list[dict]:
    return [websocket.receive_json() for _ in range(3)]


def read_until_decision_or_gameover(websocket) -> list[dict]:
    last_messages: list[dict] = []
    for _ in range(2000):
        last_messages = read_update(websocket)
        if any(message["type"] == "gameover" for message in last_messages):
            return last_messages
        decision = by_type(last_messages, "decision")
        if decision["options"]:
            return last_messages
    raise AssertionError(f"no active decision or gameover after updates: {last_messages}")


def read_broadcast(ws0, ws1) -> tuple[list[dict], list[dict]]:
    return read_pair_messages(ws0, ws1, 3)


def read_messages(websocket, count: int) -> list[dict]:
    return [websocket.receive_json() for _ in range(count)]


def read_pair_messages(ws0, ws1, count: int) -> tuple[list[dict], list[dict]]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        future0 = executor.submit(read_messages, ws0, count)
        future1 = executor.submit(read_messages, ws1, count)
        return future0.result(timeout=5), future1.result(timeout=5)


def active_decision(decisions: dict[int, dict]) -> tuple[int, dict]:
    for seat, decision in decisions.items():
        if decision["options"]:
            return seat, decision
    raise AssertionError(f"no active decision: {decisions}")


def player_snapshot(
    *,
    hand_count: int = 5,
    deck_count: int = 0,
    discard: tuple[int, ...] = (),
    set_aside_count: int = 0,
) -> PlayerPublicSnapshot:
    return PlayerPublicSnapshot(
        hand_count=hand_count,
        deck_count=deck_count,
        discard=discard,
        discard_top=discard[-1] if discard else None,
        set_aside_count=set_aside_count,
    )


def snapshot(players: tuple[PlayerPublicSnapshot, ...], trash: dict[int, int] | None = None) -> PublicSnapshot:
    return PublicSnapshot(players=players, trash=trash or {})


def decision(source: int, kind: int = 4, player: int = 0) -> dict:
    return {"player": player, "kind": kind, "source": source, "min": 0, "max": 0}


class FakeDecisionGame:
    def __init__(self, decision_value: dict, context: dict, actions: list[int]):
        self._decision = decision_value
        self._context = context
        self._actions = actions

    def current_decision(self):
        return self._decision

    def decision_context(self):
        return self._context

    def legal_mask(self):
        mask = np.zeros(dz.ACTION_SPACE_SIZE, dtype=bool)
        for action in self._actions:
            mask[action] = True
        return mask


class FakeViewGame(FakeDecisionGame):
    def __init__(self):
        super().__init__(decision(dz.DEF_SENTRY, kind=6), {}, [])

    def num_players(self):
        return 2

    def hand_count(self, player):
        return 5

    def deck_count(self, player):
        return 0

    def discard_count(self, player):
        return 0

    def discard_top(self, player):
        return None

    def in_play(self, player):
        return []

    def resources(self, player=-1):
        return {
            "actions": 1,
            "buys": 1,
            "coins": 0,
            "potion": 0,
            "debt": 0,
            "coffers": 0,
            "villagers": 0,
            "favors": 0,
            "vp_tokens": 0,
        }

    def hand(self, player):
        return {}

    def set_aside(self, player):
        return []

    def trash(self):
        return {}

    def supply(self):
        return []

    def phase(self):
        return 0

    def turn(self):
        return 0

    def game_over(self):
        return False


class FakeThronedBanditAutoGame:
    def __init__(self):
        self.before = True

    def current_decision(self):
        if self.before:
            return decision(dz.DEF_THRONE_ROOM, kind=4, player=0)
        return decision(0, kind=2, player=0)

    def decision_context(self):
        return {}

    def step(self, action):
        assert action == dz.A_SELECT_BASE + dz.DEF_BANDIT
        self.before = False
        return False

    def num_players(self):
        return 2

    def hand_count(self, player):
        return 5

    def deck_count(self, player):
        if player == 1:
            return 4 if self.before else 0
        return 0

    def discard(self, player):
        if self.before:
            return []
        if player == 0:
            return [dz.DEF_GOLD, dz.DEF_GOLD]
        return [dz.DEF_COPPER, dz.DEF_COPPER, dz.DEF_COPPER, dz.DEF_COPPER]

    def discard_top(self, player):
        cards = self.discard(player)
        return cards[-1] if cards else None

    def set_aside(self, player):
        return []

    def trash(self):
        return {}


def assert_filtered_state(message: dict) -> None:
    assert message["type"] == "state"
    view = message["view"]
    assert "myHand" in view
    for opponent in view["opponents"]:
        assert "hand" not in opponent
        assert "myHand" not in opponent
        assert "handCount" in opponent
        assert "deckCount" in opponent
        assert "discardTop" in opponent


def test_private_decision_context_labels_sentry_for_actor_only() -> None:
    game = FakeDecisionGame(
        decision(dz.DEF_SENTRY, kind=6, player=0),
        {"source_def": dz.DEF_SENTRY, "subject_defs": [dz.DEF_COPPER, dz.DEF_ESTATE], "subject_index": 0},
        [dz.A_OPTION_BASE, dz.A_OPTION_BASE + 1, dz.A_OPTION_BASE + 2],
    )
    session = SimpleNamespace(game=game)

    actor_message = _decision_message(session, 0)
    opponent_message = _decision_message(session, 1)

    assert actor_message["prompt"] == "Sentry: you look at Copper and Estate"
    assert [option["label"] for option in actor_message["options"]] == [
        "Trash Copper",
        "Discard Copper",
        "Keep Copper",
    ]
    assert "Copper" not in opponent_message["prompt"]
    assert "Estate" not in opponent_message["prompt"]
    assert opponent_message["options"] == []

    order_game = FakeDecisionGame(
        decision(dz.DEF_SENTRY, kind=7, player=0),
        {"source_def": dz.DEF_SENTRY, "subject_defs": [dz.DEF_COPPER, dz.DEF_ESTATE], "subject_index": None},
        [dz.A_OPTION_BASE, dz.A_OPTION_BASE + 1],
    )
    order_labels = [option["label"] for option in _decision_message(SimpleNamespace(game=order_game), 0)["options"]]
    assert order_labels == ["Put Copper on top (drawn next)", "Put Estate on top (drawn next)"]


def test_card_select_decisions_include_additive_visible_zone_metadata() -> None:
    cases = [
        (decision(dz.DEF_CELLAR, kind=4), dz.DEF_COPPER, "hand"),
        (decision(dz.DEF_WORKSHOP, kind=5), dz.DEF_SILVER, "supply"),
        (decision(dz.DEF_HARBINGER, kind=4), dz.DEF_SILVER, "discard"),
        (decision(dz.DEF_BANDIT, kind=4), dz.DEF_GOLD, "set_aside"),
        (decision(dz.DEF_MOAT, kind=8), dz.DEF_MOAT, "hand"),
    ]
    for decision_value, selected_def, expected_zone in cases:
        game = FakeDecisionGame(
            decision_value,
            {},
            [dz.A_SELECT_BASE + selected_def],
        )
        message = _decision_message(SimpleNamespace(game=game), 0)
        assert message["select_zone"] == expected_zone
        assert message["options"][0]["def"] == selected_def

    # It is wire-additive and actor-only: an observer still receives the
    # pre-existing inactive shape with no private card-choice metadata.
    inactive = _decision_message(
        SimpleNamespace(game=FakeDecisionGame(decision(dz.DEF_CELLAR, kind=4), {}, [dz.A_SELECT_BASE + dz.DEF_COPPER])),
        1,
    )
    assert "select_zone" not in inactive


def test_library_and_vassal_prompts_include_subject_card() -> None:
    library = FakeDecisionGame(
        decision(dz.DEF_LIBRARY, kind=6, player=0),
        {"source_def": dz.DEF_LIBRARY, "subject_defs": [dz.DEF_VILLAGE], "subject_index": 0},
        [dz.A_OPTION_BASE, dz.A_OPTION_BASE + 1],
    )
    library_message = _decision_message(SimpleNamespace(game=library), 0)
    assert library_message["prompt"] == "Library: drew Village - set it aside?"
    assert [option["label"] for option in library_message["options"]] == ["Keep Village", "Set aside Village"]

    vassal = FakeDecisionGame(
        decision(dz.DEF_VASSAL, kind=6, player=0),
        {"source_def": dz.DEF_VASSAL, "subject_defs": [dz.DEF_SMITHY], "subject_index": 0},
        [dz.A_OPTION_BASE, dz.A_OPTION_BASE + 1],
    )
    vassal_message = _decision_message(SimpleNamespace(game=vassal), 0)
    assert vassal_message["prompt"] == "Vassal: discarded Smithy - play it?"
    assert [option["label"] for option in vassal_message["options"]] == [
        "Do not play Smithy",
        "Play Smithy",
    ]


def test_sentry_resolution_log_summarizes_once_and_private_detail_is_seat_filtered() -> None:
    states: dict[int, SentryLogState] = {}
    before = snapshot((player_snapshot(), player_snapshot()))
    after = snapshot((player_snapshot(), player_snapshot()), {dz.DEF_COPPER: 1})
    first_decision = decision(dz.DEF_SENTRY, kind=6, player=0)
    first_context = {
        "source_def": dz.DEF_SENTRY,
        "subject_defs": [dz.DEF_COPPER, dz.DEF_ESTATE],
        "subject_index": 0,
    }
    next_sentry_decision = decision(dz.DEF_SENTRY, kind=6, player=0)

    first_public = public_log_lines(
        0,
        dz.A_OPTION_BASE,
        first_decision,
        before,
        after,
        first_context,
        next_sentry_decision,
    )
    first_summary, first_private = sentry_resolution_logs(
        states,
        0,
        dz.A_OPTION_BASE,
        first_decision,
        first_context,
        next_sentry_decision,
    )
    assert first_public == []
    assert first_summary == []
    assert first_private == {}

    second_decision = decision(dz.DEF_SENTRY, kind=6, player=0)
    second_context = {
        "source_def": dz.DEF_SENTRY,
        "subject_defs": [dz.DEF_COPPER, dz.DEF_ESTATE],
        "subject_index": 1,
    }
    done_decision = decision(0, kind=1, player=0)
    second_public = public_log_lines(
        0,
        dz.A_OPTION_BASE + 2,
        second_decision,
        after,
        after,
        second_context,
        done_decision,
    )
    summary, private = sentry_resolution_logs(
        states,
        0,
        dz.A_OPTION_BASE + 2,
        second_decision,
        second_context,
        done_decision,
    )
    assert second_public == []
    assert summary == ["P1 trashes Copper and keeps 1 card on top (Sentry)"]
    assert private == {0: ["You looked at Copper and Estate; trashed Copper; kept Estate on top"]}
    assert all("keeps" not in line for line in first_public + second_public)

    session = SimpleNamespace(game=FakeViewGame())
    actor_log = _post_step_messages(session, 0, summary, private)[0]["lines"]
    opponent_log = _post_step_messages(session, 1, summary, private)[0]["lines"]
    assert actor_log == [
        "P1 trashes Copper and keeps 1 card on top (Sentry)",
        "You looked at Copper and Estate; trashed Copper; kept Estate on top",
    ]
    assert opponent_log == ["P1 trashes Copper and keeps 1 card on top (Sentry)"]
    assert "You looked" not in " ".join(opponent_log)


def test_public_effect_logs_describe_bandit_militia_and_witch_without_private_leaks() -> None:
    bandit_state = BanditLogState()
    bandit_before = snapshot(
        (
            player_snapshot(discard=()),
            player_snapshot(deck_count=2, discard=()),
        )
    )
    bandit_after = snapshot(
        (
            player_snapshot(discard=(dz.DEF_GOLD,)),
            player_snapshot(deck_count=0, discard=(dz.DEF_ESTATE,)),
        ),
        {dz.DEF_SILVER: 1},
    )
    bandit_lines = public_log_lines(
        0,
        dz.A_PLAY_BASE + dz.DEF_BANDIT,
        decision(dz.DEF_BANDIT, kind=1),
        bandit_before,
        bandit_after,
    )
    bandit_lines.extend(bandit_resolution_logs(
        bandit_state,
        0,
        dz.A_PLAY_BASE + dz.DEF_BANDIT,
        decision(dz.DEF_BANDIT, kind=1),
        bandit_before,
        bandit_after,
        {},
    ))
    assert "P2 reveals Silver and Estate; trashes Silver" in bandit_lines

    militia_before = snapshot(
        (
            player_snapshot(),
            player_snapshot(discard=(dz.DEF_DUCHY,)),
        )
    )
    militia_after = snapshot(
        (
            player_snapshot(),
            player_snapshot(discard=(dz.DEF_DUCHY, dz.DEF_COPPER, dz.DEF_ESTATE)),
        )
    )
    militia_lines = public_log_lines(
        1,
        dz.A_SELECT_BASE + dz.DEF_GOLD,
        decision(dz.DEF_MILITIA, player=1),
        militia_before,
        militia_after,
    )
    assert militia_lines == ["P2 discards 2 cards; discard top is now Estate"]
    assert "Gold" not in " ".join(militia_lines)
    assert "Copper" not in " ".join(militia_lines)

    witch_before = snapshot(
        (
            player_snapshot(discard=()),
            player_snapshot(discard=()),
        )
    )
    witch_after = snapshot(
        (
            player_snapshot(discard=()),
            player_snapshot(discard=(dz.DEF_CURSE,)),
        )
    )
    witch_lines = public_log_lines(
        0,
        dz.A_PLAY_BASE + dz.DEF_WITCH,
        decision(dz.DEF_WITCH, kind=1),
        witch_before,
        witch_after,
    )
    assert "P2 gains a Curse" in witch_lines


def test_throned_bandit_logs_one_victim_attributed_line_per_hit() -> None:
    state = BanditLogState()
    empty = snapshot((player_snapshot(), player_snapshot()))

    start_decision = decision(dz.DEF_THRONE_ROOM, kind=4, player=0)
    start_lines = public_log_lines(
        0,
        dz.A_SELECT_BASE + dz.DEF_BANDIT,
        start_decision,
        empty,
        empty,
    )
    start_lines.extend(bandit_resolution_logs(
        state,
        0,
        dz.A_SELECT_BASE + dz.DEF_BANDIT,
        start_decision,
        empty,
        empty,
        {},
    ))
    assert start_lines == ["P1 plays Bandit"]

    first_before = snapshot((player_snapshot(), player_snapshot(deck_count=2, discard=())))
    first_after = snapshot(
        (player_snapshot(), player_snapshot(deck_count=0, discard=(dz.DEF_GOLD,))),
        {dz.DEF_SILVER: 1},
    )
    first_decision = decision(dz.DEF_BANDIT, kind=4, player=1)
    first_context = {
        "source_def": dz.DEF_BANDIT,
        "subject_defs": [dz.DEF_SILVER, dz.DEF_GOLD],
        "subject_index": None,
        "victim_player": 1,
        "attacker_player": 0,
    }
    first_lines = public_log_lines(
        1,
        dz.A_SELECT_BASE + dz.DEF_SILVER,
        first_decision,
        first_before,
        first_after,
        first_context,
    )
    first_lines.extend(bandit_resolution_logs(
        state,
        1,
        dz.A_SELECT_BASE + dz.DEF_SILVER,
        first_decision,
        first_before,
        first_after,
        first_context,
    ))
    assert first_lines == ["P2 reveals Silver and Gold; trashes Silver"]

    second_before = first_after
    second_after = snapshot(
        (player_snapshot(), player_snapshot(deck_count=0, discard=(dz.DEF_GOLD, dz.DEF_COPPER))),
        {dz.DEF_SILVER: 1, dz.DEF_GOLD: 1},
    )
    second_context = {
        "source_def": dz.DEF_BANDIT,
        "subject_defs": [dz.DEF_GOLD, dz.DEF_COPPER],
        "subject_index": None,
        "victim_player": 1,
        "attacker_player": 0,
    }
    second_lines = public_log_lines(
        1,
        dz.A_SELECT_BASE + dz.DEF_GOLD,
        first_decision,
        second_before,
        second_after,
        second_context,
    )
    second_lines.extend(bandit_resolution_logs(
        state,
        1,
        dz.A_SELECT_BASE + dz.DEF_GOLD,
        first_decision,
        second_before,
        second_after,
        second_context,
    ))
    assert second_lines == [
        "P1 plays Bandit (again)",
        "P2 reveals Gold and Copper; trashes Gold",
    ]

    all_lines = start_lines + first_lines + second_lines
    assert not any(line.startswith("P1 reveals") for line in all_lines)
    assert not any("Silver, Gold and Gold" in line or "Silver, Gold, Gold and Copper" in line for line in all_lines)


def test_fixture_replay_logs_throned_bandit_hits_through_server_session_path() -> None:
    data = json.loads(FIXTURE_THRONE_BANDIT.read_text())
    setup = dz.Setup(players=len(data["seats"]), kingdom=data["kingdom"])
    session = Session(
        session_id="fixture-throne-bandit",
        seats=[Seat(kind, f"seat-{index}") for index, kind in enumerate(data["seats"])],
        setup=setup,
        seed=int(data["seed"]),
        kingdom=[int(def_value) for def_value in data["kingdom"]],
        game=dz.new_game(setup, int(data["seed"])),
    )

    lines_by_index: dict[int, list[str]] = {}
    for index, action in enumerate(data["actions"]):
        player = int(session.game.current_decision()["player"])
        lines, private = _apply_validated_action(session, player, int(action))
        assert private == {} or index in {49, 51, 85}
        lines_by_index[index] = lines

    assert len(data["actions"]) == 108
    assert lines_by_index[87] == [
        "P1 plays Bandit",
        "P2 reveals Silver and Copper; trashes Silver",
        "P1 plays Bandit (again)",
        "P2 reveals Gold and Copper; trashes Gold",
    ]
    assert lines_by_index[107] == [
        "P1 plays Bandit",
        "P2 reveals Copper and Copper; trashes nothing",
    ]
    assert session.log_lines.count("P2 reveals Silver and Copper; trashes Silver") == 1
    assert session.log_lines.count("P1 plays Bandit (again)") == 1
    assert session.log_lines.count("P2 reveals Gold and Copper; trashes Gold") == 1
    assert session.log_lines.count("P2 reveals Copper and Copper; trashes nothing") == 1
    assert not any("Silver, Gold and Gold" in line or "Silver, Gold, Gold and Copper" in line for line in session.log_lines)


def test_throned_bandit_splits_two_auto_hits_from_one_server_step() -> None:
    session = SimpleNamespace(
        game=FakeThronedBanditAutoGame(),
        bandit_log_state=BanditLogState(),
        sentry_log_states={},
        action_log=[],
        action_log_line_counts=[],
        log_lines=[],
    )

    lines, private = _apply_action(session, 0, dz.A_SELECT_BASE + dz.DEF_BANDIT)

    assert private == {}
    assert lines == [
        "P1 plays Bandit",
        "P2 reveals Copper and Copper; trashes nothing",
        "P1 plays Bandit (again)",
        "P2 reveals Copper and Copper; trashes nothing",
    ]
    assert session.action_log == [dz.A_SELECT_BASE + dz.DEF_BANDIT]
    assert session.action_log_line_counts == [4]


def test_undo_clears_in_flight_bandit_log_state_before_continuing() -> None:
    setup = dz.Setup(players=2, kingdom=KINGDOM)
    session = Session(
        session_id="undo-bandit",
        seats=[Seat("human", "p1"), Seat("bot", "p2")],
        setup=setup,
        seed=0xBADA11,
        kingdom=[dz.def_id(name) for name in KINGDOM],
        game=dz.new_game(setup, 0xBADA11),
    )
    session.action_log = [12345]
    session.action_log_line_counts = [4]
    session.human_decision_prefixes = [0]
    session.log_lines = [
        "P1 plays Throne Room",
        "P1 plays Bandit",
        "P2 reveals Silver and Gold; trashes Silver",
        "P1 plays Bandit (again)",
    ]
    session.bandit_log_state.hits_by_attacker[0] = 2

    line = _undo_previous_human_decision(session, 0)
    assert line == "Undo: rewound to previous human decision"
    assert session.action_log == []
    assert session.action_log_line_counts == []
    assert session.log_lines == [line]
    assert session.bandit_log_state.hits_by_attacker == {}

    before = snapshot((player_snapshot(), player_snapshot(deck_count=2, discard=())))
    after = snapshot(
        (player_snapshot(), player_snapshot(deck_count=0, discard=(dz.DEF_GOLD,))),
        {dz.DEF_SILVER: 1},
    )
    context = {
        "source_def": dz.DEF_BANDIT,
        "subject_defs": [dz.DEF_SILVER, dz.DEF_GOLD],
        "subject_index": None,
        "victim_player": 1,
        "attacker_player": 0,
    }
    continued = bandit_resolution_logs(
        session.bandit_log_state,
        1,
        dz.A_SELECT_BASE + dz.DEF_SILVER,
        decision(dz.DEF_BANDIT, kind=4, player=1),
        before,
        after,
        context,
    )
    fresh = bandit_resolution_logs(
        BanditLogState(),
        1,
        dz.A_SELECT_BASE + dz.DEF_SILVER,
        decision(dz.DEF_BANDIT, kind=4, player=1),
        before,
        after,
        context,
    )
    assert continued == fresh == ["P2 reveals Silver and Gold; trashes Silver"]


def test_log_line_spells_trashes_correctly() -> None:
    line = log_line(1, dz.A_SELECT_BASE + dz.DEF_SILVER, decision(dz.DEF_BANDIT, player=1))
    assert line == "P2 trashes Silver"
    assert "trash" + "s" not in line


def test_sentry_public_log_never_names_kept_topdecked_cards() -> None:
    states: dict[int, SentryLogState] = {}
    before = snapshot((player_snapshot(deck_count=5, discard=(), set_aside_count=1), player_snapshot()))
    after = snapshot((player_snapshot(deck_count=6, discard=(), set_aside_count=0), player_snapshot()))
    context = {"source_def": dz.DEF_SENTRY, "subject_defs": [dz.DEF_SILVER], "subject_index": 0}
    done_decision = decision(0, kind=1, player=0)
    lines = public_log_lines(
        0,
        dz.A_OPTION_BASE + 2,
        decision(dz.DEF_SENTRY, kind=6),
        before,
        after,
        context,
        done_decision,
    )
    summary, private = sentry_resolution_logs(
        states,
        0,
        dz.A_OPTION_BASE + 2,
        decision(dz.DEF_SENTRY, kind=6),
        context,
        done_decision,
    )
    assert lines == []
    assert summary == ["P1 keeps 1 card on top (Sentry)"]
    assert private == {0: ["You looked at Silver; kept Silver on top"]}
    assert "Silver" not in " ".join(summary)


def choose_big_money_action(decision: dict) -> int:
    options = decision["options"]
    assert all("action" in option and "label" in option for option in options)

    for prefix in ("Play Platinum", "Play Gold", "Play Silver", "Play Copper", "Play Potion"):
        for option in options:
            if option["label"] == prefix:
                return int(option["action"])

    for prefix in ("Buy Province", "Buy Gold", "Buy Silver"):
        for option in options:
            if option["label"] == prefix:
                return int(option["action"])

    for option in options:
        if option["label"] == "Pass":
            return int(option["action"])
    return int(options[0]["action"])


def direct_big_money_action(game) -> int:
    mask = game.legal_mask()
    for def_id in (dz.DEF_PLATINUM, dz.DEF_GOLD, dz.DEF_SILVER, dz.DEF_COPPER, dz.DEF_POTION):
        action = dz.A_PLAY_BASE + def_id
        if action < dz.ACTION_SPACE_SIZE and mask[action]:
            return int(action)
    for def_id in (dz.DEF_PROVINCE, dz.DEF_GOLD, dz.DEF_SILVER):
        action = dz.A_BUY_BASE + def_id
        if action < dz.ACTION_SPACE_SIZE and mask[action]:
            return int(action)
    if mask[dz.A_PASS]:
        return int(dz.A_PASS)
    return int(next(iter(mask.nonzero()[0])))


def finish_direct_game(game) -> None:
    for _ in range(10000):
        if game.game_over():
            return
        game.step(direct_big_money_action(game))
    raise AssertionError("direct game did not finish")


def test_web_session_filters_validates_labels_and_broadcasts() -> None:
    sessions.clear()
    client = TestClient(app)

    response = client.post(
        "/api/session",
        json={"seats": ["human", "human"], "kingdom": KINGDOM, "seed": 0x5100},
    )
    assert response.status_code == 200
    created = response.json()
    assert "session_id" in created
    assert len(created["seat_tokens"]) == 2

    session_id = created["session_id"]
    tokens = created["seat_tokens"]
    session = sessions[session_id]

    with client.websocket_connect(f"/ws/{session_id}/{tokens[0]}") as ws0:
        with client.websocket_connect(f"/ws/{session_id}/{tokens[1]}") as ws1:
            initial0 = read_initial(ws0)
            initial1 = read_initial(ws1)

            assert by_type(initial0, "table")["kingdom"]
            assert by_type(initial1, "table")["seats"][1]["kind"] == "human"
            assert_filtered_state(by_type(initial0, "state"))
            assert_filtered_state(by_type(initial1, "state"))

            decisions = {
                0: by_type(initial0, "decision"),
                1: by_type(initial1, "decision"),
            }
            seat, decision = active_decision(decisions)
            assert decision["prompt"]
            assert decision["options"]
            assert all(option["label"] for option in decision["options"])

            before_log_len = len(session.action_log)
            (ws0 if seat == 0 else ws1).send_json(
                {"type": "act", "action": dz.ACTION_SPACE_SIZE + 100}
            )
            error = (ws0 if seat == 0 else ws1).receive_json()
            assert error["type"] == "error"
            assert "illegal" in error["message"]
            assert len(session.action_log) == before_log_len

            action = choose_big_money_action(decision)
            (ws0 if seat == 0 else ws1).send_json({"type": "act", "action": action})
            update0, update1 = read_broadcast(ws0, ws1)
            assert by_type(update0, "log")["lines"]
            assert by_type(update1, "log")["lines"]
            assert_filtered_state(by_type(update0, "state"))
            assert_filtered_state(by_type(update1, "state"))
            decisions = {
                0: by_type(update0, "decision"),
                1: by_type(update1, "decision"),
            }
            inactive_seat = 1 - active_decision(decisions)[0]
            assert decisions[inactive_seat]["options"] == []


def test_websocket_initial_state_can_report_gameover() -> None:
    sessions.clear()
    client = TestClient(app)
    response = client.post(
        "/api/session",
        json={"seats": ["human", "human"], "kingdom": KINGDOM, "seed": 0x5101},
    )
    assert response.status_code == 200
    created = response.json()
    session_id = created["session_id"]
    token = created["seat_tokens"][0]

    finish_direct_game(sessions[session_id].game)
    with client.websocket_connect(f"/ws/{session_id}/{token}") as websocket:
        messages = read_initial(websocket)
        assert by_type(messages, "table")["kingdom"]
        assert_filtered_state(by_type(messages, "state"))
        gameover = by_type(messages, "gameover")
        assert len(gameover["scores"]) == 2
        assert gameover["winner"] in (0, 1, None)
        assert gameover["truncated"] is False


def test_human_vs_bigmoney_bot_completes_over_websocket() -> None:
    sessions.clear()
    client = TestClient(app)
    response = client.post(
        "/api/session",
        json={
            "seats": ["human", "bot:bigmoney"],
            "kingdom": KINGDOM,
            "seed": 0x5102,
            "thinking_delay_ms": 0,
        },
    )
    assert response.status_code == 200
    created = response.json()
    session = sessions[created["session_id"]]

    with client.websocket_connect(f"/ws/{created['session_id']}/{created['seat_tokens'][0]}") as websocket:
        messages = read_initial(websocket)
        for _ in range(4000):
            if any(message["type"] == "gameover" for message in messages):
                break
            decision = by_type(messages, "decision")
            if not decision["options"]:
                messages = read_until_decision_or_gameover(websocket)
                continue
            websocket.send_json({"type": "act", "action": choose_big_money_action(decision)})
            messages = read_until_decision_or_gameover(websocket)
        else:
            raise AssertionError("web game did not complete")

    assert session.game.game_over()
    assert session.game.truncated() is False
    assert any(line.startswith("P2 ") for line in session.log_lines)
    assert len(session.action_log) > 0


def test_human_vs_nn_bot_completes_with_legal_nn_actions(tiny_nn_checkpoint: Path) -> None:
    sessions.clear()
    client = TestClient(app)
    response = client.post(
        "/api/session",
        json={
            "seats": ["human", f"bot:nn:{tiny_nn_checkpoint}"],
            "kingdom": KINGDOM,
            "seed": 0x5105,
            "thinking_delay_ms": 0,
        },
    )
    assert response.status_code == 200
    created = response.json()
    session = sessions[created["session_id"]]
    assert 1 in session.nn_policies

    with client.websocket_connect(f"/ws/{created['session_id']}/{created['seat_tokens'][0]}") as websocket:
        messages = read_initial(websocket)
        for _ in range(4000):
            if any(message["type"] == "gameover" for message in messages):
                break
            decision = by_type(messages, "decision")
            if not decision["options"]:
                messages = read_until_decision_or_gameover(websocket)
                continue
            websocket.send_json({"type": "act", "action": choose_big_money_action(decision)})
            messages = read_until_decision_or_gameover(websocket)
        else:
            raise AssertionError("web game against NN bot did not complete")

    replay = dz.new_game(session.setup, session.seed)
    nn_actions: list[int] = []
    for action in session.action_log:
        player = int(replay.current_decision()["player"])
        assert bool(replay.legal_mask()[action])
        if player == 1:
            nn_actions.append(action)
        replay.step(action)

    assert session.game.game_over()
    assert replay.game_over()
    assert nn_actions


def test_human_vs_nnmcts_bot_completes_with_legal_search_actions(
    monkeypatch: pytest.MonkeyPatch,
    tiny_nn_checkpoint: Path,
) -> None:
    """A scripted human game exercises the full parked-leaf NN-MCTS loop."""
    sessions.clear()
    monkeypatch.setenv("NN_MCTS_SIMS", "8")
    client = TestClient(app)
    response = client.post(
        "/api/session",
        json={
            "seats": ["human", f"bot:nnmcts:{tiny_nn_checkpoint}"],
            "kingdom": KINGDOM,
            "seed": 0x5108,
            "thinking_delay_ms": 0,
        },
    )
    assert response.status_code == 200
    created = response.json()
    session = sessions[created["session_id"]]
    assert session.seats[1].kind.startswith("bot:nnmcts")
    assert 1 in session.nn_policies

    with client.websocket_connect(f"/ws/{created['session_id']}/{created['seat_tokens'][0]}") as websocket:
        messages = read_initial(websocket)
        for _ in range(4000):
            if any(message["type"] == "gameover" for message in messages):
                break
            decision = by_type(messages, "decision")
            if not decision["options"]:
                messages = read_until_decision_or_gameover(websocket)
                continue
            websocket.send_json({"type": "act", "action": choose_big_money_action(decision)})
            messages = read_until_decision_or_gameover(websocket)
        else:
            raise AssertionError("web game against NN-MCTS bot did not complete")

    replay = dz.new_game(session.setup, session.seed)
    nnmcts_actions: list[int] = []
    for action in session.action_log:
        player = int(replay.current_decision()["player"])
        assert bool(replay.legal_mask()[action])
        if player == 1:
            nnmcts_actions.append(action)
        replay.step(action)

    assert session.game.game_over()
    assert replay.game_over()
    assert nnmcts_actions


def test_nn_bot_uses_configured_default_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tiny_nn_checkpoint: Path,
) -> None:
    sessions.clear()
    monkeypatch.setenv("DOMINION_NN_CHECKPOINT", str(tiny_nn_checkpoint))

    response = TestClient(app).post(
        "/api/session",
        json={"seats": ["human", "bot:nn"], "kingdom": KINGDOM, "seed": 0x5106},
    )

    assert response.status_code == 200
    assert 1 in sessions[response.json()["session_id"]].nn_policies


def test_nn_bot_missing_checkpoint_returns_clean_400(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sessions.clear()
    monkeypatch.setenv("DOMINION_NN_CHECKPOINT", str(tmp_path / "missing-policy.pt"))

    response = TestClient(app).post(
        "/api/session",
        json={"seats": ["human", "bot:nn"], "kingdom": KINGDOM, "seed": 0x5107},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "neural-network checkpoint is unavailable"}
    assert "traceback" not in response.text.lower()


def test_undo_rewinds_to_previous_human_decision() -> None:
    sessions.clear()
    client = TestClient(app)
    response = client.post(
        "/api/session",
        json={
            "seats": ["human", "bot:bigmoney"],
            "kingdom": KINGDOM,
            "seed": 0x5103,
            "thinking_delay_ms": 0,
        },
    )
    assert response.status_code == 200
    created = response.json()
    session = sessions[created["session_id"]]

    with client.websocket_connect(f"/ws/{created['session_id']}/{created['seat_tokens'][0]}") as websocket:
        initial = read_initial(websocket)
        initial_state = by_type(initial, "state")["view"]
        initial_decision = by_type(initial, "decision")
        websocket.send_json({"type": "act", "action": choose_big_money_action(initial_decision)})
        _ = read_until_decision_or_gameover(websocket)
        assert session.action_log

        websocket.send_json({"type": "undo_request"})
        undo_messages = read_update(websocket)
        assert "undo" in " ".join(by_type(undo_messages, "log")["lines"]).lower()
        assert by_type(undo_messages, "state")["view"] == initial_state
        assert by_type(undo_messages, "decision")["options"]

    replay = dz.new_game(session.setup, session.seed)
    assert session.action_log == []
    assert session.game.state_hash() == replay.state_hash()


def create_two_human_undo_session(client: TestClient, seed: int) -> tuple[dict, Session]:
    sessions.clear()
    response = client.post(
        "/api/session",
        json={"seats": ["human", "human"], "kingdom": KINGDOM, "seed": seed},
    )
    assert response.status_code == 200
    created = response.json()
    return created, sessions[created["session_id"]]


def play_opening_human_decision(ws0, ws1) -> tuple[list[dict], tuple[list[dict], list[dict]]]:
    initial0 = read_initial(ws0)
    initial1 = read_initial(ws1)
    seat, opening = active_decision({0: by_type(initial0, "decision"), 1: by_type(initial1, "decision")})
    assert seat == 0
    ws0.send_json({"type": "act", "action": choose_big_money_action(opening)})
    return initial0, read_broadcast(ws0, ws1)


def test_two_human_undo_accept_rewinds_and_full_refreshes() -> None:
    client = TestClient(app)
    created, session = create_two_human_undo_session(client, 0x5110)
    tokens = created["seat_tokens"]
    initial_hash = session.game.state_hash()

    with client.websocket_connect(f"/ws/{created['session_id']}/{tokens[0]}") as ws0:
        with client.websocket_connect(f"/ws/{created['session_id']}/{tokens[1]}") as ws1:
            initial0, _ = play_opening_human_decision(ws0, ws1)
            assert session.human_decision_seats == [0]

            ws0.send_json({"type": "undo_request"})
            assert ws0.receive_json() == {"type": "undo_pending", "seat": 0}
            assert ws1.receive_json() == {"type": "undo_offer", "seat": 0}

            ws1.send_json({"type": "undo_response", "accept": True})
            refreshed0, refreshed1 = read_pair_messages(ws0, ws1, 5)

            for messages in (refreshed0, refreshed1):
                assert messages[0] == {"type": "undo_result", "seat": 0, "accepted": True}
                assert messages[1]["type"] == "table"
                assert by_type(messages, "log")["lines"] == ["P1 undoes their last decision"]
            assert by_type(refreshed0, "state")["view"] == by_type(initial0, "state")["view"]
            assert session.action_log == []
            assert session.human_decision_prefixes == []
            assert session.human_decision_seats == []
            assert session.game.state_hash() == initial_hash


def test_two_human_undo_deny_preserves_game_and_notifies_both() -> None:
    client = TestClient(app)
    created, session = create_two_human_undo_session(client, 0x5111)
    tokens = created["seat_tokens"]

    with client.websocket_connect(f"/ws/{created['session_id']}/{tokens[0]}") as ws0:
        with client.websocket_connect(f"/ws/{created['session_id']}/{tokens[1]}") as ws1:
            _, _ = play_opening_human_decision(ws0, ws1)
            before_actions = list(session.action_log)
            before_hash = session.game.state_hash()

            ws0.send_json({"type": "undo_request"})
            _ = ws0.receive_json()
            _ = ws1.receive_json()
            ws1.send_json({"type": "undo_response", "accept": False})
            denied0, denied1 = read_pair_messages(ws0, ws1, 1)

            expected = {"type": "undo_result", "seat": 0, "accepted": False, "reason": "denied"}
            assert denied0 == [expected]
            assert denied1 == [expected]
            assert session.pending_undo is None
            assert session.action_log == before_actions
            assert session.game.state_hash() == before_hash


def test_two_human_undo_is_invalidated_when_the_game_advances() -> None:
    client = TestClient(app)
    created, session = create_two_human_undo_session(client, 0x5112)
    tokens = created["seat_tokens"]

    with client.websocket_connect(f"/ws/{created['session_id']}/{tokens[0]}") as ws0:
        with client.websocket_connect(f"/ws/{created['session_id']}/{tokens[1]}") as ws1:
            _, (after0, _) = play_opening_human_decision(ws0, ws1)
            ws0.send_json({"type": "undo_request"})
            _ = ws0.receive_json()
            _ = ws1.receive_json()

            ws0.send_json({"type": "act", "action": choose_big_money_action(by_type(after0, "decision"))})
            advanced0, advanced1 = read_pair_messages(ws0, ws1, 4)

            expected = {"type": "undo_result", "seat": 0, "accepted": False, "reason": "game_advanced"}
            assert advanced0[0] == expected
            assert advanced1[0] == expected
            assert by_type(advanced0[1:], "log")["lines"]
            assert by_type(advanced1[1:], "state")
            assert session.pending_undo is None


def test_two_human_undo_allows_only_one_pending_request() -> None:
    client = TestClient(app)
    created, session = create_two_human_undo_session(client, 0x5113)
    tokens = created["seat_tokens"]

    with client.websocket_connect(f"/ws/{created['session_id']}/{tokens[0]}") as ws0:
        with client.websocket_connect(f"/ws/{created['session_id']}/{tokens[1]}") as ws1:
            _, _ = play_opening_human_decision(ws0, ws1)
            ws0.send_json({"type": "undo_request"})
            _ = ws0.receive_json()
            _ = ws1.receive_json()

            ws0.send_json({"type": "undo_request"})
            error = ws0.receive_json()
            assert error["type"] == "error"
            assert "pending" in error["message"]
            assert session.pending_undo is not None


def test_two_human_undo_rejects_when_opponent_made_the_last_decision() -> None:
    client = TestClient(app)
    created, session = create_two_human_undo_session(client, 0x5114)
    tokens = created["seat_tokens"]

    with client.websocket_connect(f"/ws/{created['session_id']}/{tokens[0]}") as ws0:
        with client.websocket_connect(f"/ws/{created['session_id']}/{tokens[1]}") as ws1:
            _, updates = play_opening_human_decision(ws0, ws1)
            for _ in range(100):
                decisions = {0: by_type(updates[0], "decision"), 1: by_type(updates[1], "decision")}
                seat, decision_value = active_decision(decisions)
                websocket = ws0 if seat == 0 else ws1
                websocket.send_json({"type": "act", "action": choose_big_money_action(decision_value)})
                updates = read_broadcast(ws0, ws1)
                if seat == 1:
                    break
            else:
                raise AssertionError("P2 never made a decision")

            assert session.human_decision_seats[-1] == 1
            ws0.send_json({"type": "undo_request"})
            error = ws0.receive_json()
            assert error == {"type": "error", "message": "nothing to undo"}
            assert session.pending_undo is None


def test_export_endpoint_replays_deterministically() -> None:
    sessions.clear()
    client = TestClient(app)
    response = client.post(
        "/api/session",
        json={
            "seats": ["human", "bot:random"],
            "kingdom": KINGDOM,
            "seed": 0x5104,
            "thinking_delay_ms": 0,
        },
    )
    assert response.status_code == 200
    created = response.json()

    with client.websocket_connect(f"/ws/{created['session_id']}/{created['seat_tokens'][0]}") as websocket:
        messages = read_initial(websocket)
        decision = by_type(messages, "decision")
        websocket.send_json({"type": "act", "action": choose_big_money_action(decision)})
        _ = read_until_decision_or_gameover(websocket)

    export_response = client.get(f"/api/session/{created['session_id']}/export")
    assert export_response.status_code == 200
    data = export_response.json()
    assert data["seed"] == 0x5104
    assert data["kingdom"]
    assert data["seats"] == ["human", "bot:random"]
    assert data["actions"]
    assert data["obs_version"] == dz.OBS_VERSION
    assert verify_export_data(data) == int(data["final_state_hash"], 16)


def test_export_file_endpoint_writes_finished_game() -> None:
    sessions.clear()
    client = TestClient(app)

    unknown_response = client.post("/api/session/unknown/export-file")
    assert unknown_response.status_code == 404
    assert unknown_response.json() == {"detail": "session not found"}

    response = client.post(
        "/api/session",
        json={"seats": ["human", "human"], "kingdom": KINGDOM, "seed": 0x5109},
    )
    assert response.status_code == 200
    created = response.json()
    session_id = created["session_id"]

    unfinished_response = client.post(f"/api/session/{session_id}/export-file")
    assert unfinished_response.status_code == 409
    assert unfinished_response.json() == {"detail": "game is not over"}

    finish_direct_game(sessions[session_id].game)
    export_response = client.post(f"/api/session/{session_id}/export-file")
    assert export_response.status_code == 200
    path = Path(export_response.json()["path"])
    assert path.exists()
    assert json.loads(path.read_text()) == client.get(f"/api/session/{session_id}/export").json()


def test_random_kingdom_is_seeded_and_valid() -> None:
    from src.v2.web.server.defs import kingdom_def_ids

    with TestClient(app) as client:
        first = client.post(
            "/api/session",
            json={"seats": ["human", "bot"], "kingdom": "random", "seed": 99, "thinking_delay_ms": 0},
        ).json()
        second = client.post(
            "/api/session",
            json={"seats": ["human", "bot"], "kingdom": "random", "seed": 99, "thinking_delay_ms": 0},
        ).json()
        pool = set(kingdom_def_ids())
        kingdoms = []
        for created in (first, second):
            session = sessions[created["session_id"]]
            assert len(session.kingdom) == 10
            assert len(set(session.kingdom)) == 10
            assert set(session.kingdom) <= pool
            kingdoms.append(list(session.kingdom))
        assert kingdoms[0] == kingdoms[1]
