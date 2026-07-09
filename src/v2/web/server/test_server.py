from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

import dominion_v2_py as dz

from src.v2.web.server.main import app, sessions
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
    with ThreadPoolExecutor(max_workers=2) as executor:
        future0 = executor.submit(read_update, ws0)
        future1 = executor.submit(read_update, ws1)
        return future0.result(timeout=5), future1.result(timeout=5)


def active_decision(decisions: dict[int, dict]) -> tuple[int, dict]:
    for seat, decision in decisions.items():
        if decision["options"]:
            return seat, decision
    raise AssertionError(f"no active decision: {decisions}")


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
