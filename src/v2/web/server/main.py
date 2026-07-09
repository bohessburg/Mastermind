from __future__ import annotations

import asyncio
import random
import secrets
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

import dominion_v2_py as dz

from .defs import def_by_id, def_id, load_defs
from .observer import decision_kind_name, legal_options, log_line, prompt_for


DEFAULT_KINGDOM = [
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


@dataclass
class Seat:
    kind: str
    token: str


@dataclass
class Session:
    session_id: str
    seats: list[Seat]
    setup: dz.Setup
    seed: int
    kingdom: list[int]
    game: Any
    action_log: list[int] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    connections: dict[str, WebSocket] = field(default_factory=dict)
    bot_rng: random.Random = field(default_factory=random.Random)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


app = FastAPI(title="DominionZero v2 Web Server")
sessions: dict[str, Session] = {}


def _make_setup(players: int, kingdom: list[int]) -> dz.Setup:
    return dz.Setup(players=players, kingdom=kingdom)


def _parse_kingdom(payload: dict[str, Any]) -> list[int]:
    raw = payload.get("kingdom") or DEFAULT_KINGDOM
    kingdom: list[int] = []
    for item in raw:
        if isinstance(item, str):
            kingdom.append(def_id(item))
        else:
            kingdom.append(int(item))
    return kingdom


def _seat_index(session: Session, token: str) -> int:
    for index, seat in enumerate(session.seats):
        if seat.token == token:
            return index
    return -1


def _hand_entries(hand: dict[int, int]) -> list[dict[str, int]]:
    return [
        {"def": int(def_id_value), "count": int(count)}
        for def_id_value, count in sorted(hand.items())
        if int(count) > 0
    ]


def _trash_entries(game: Any) -> list[dict[str, int]]:
    return [
        {"def": int(def_id_value), "count": int(count)}
        for def_id_value, count in sorted(game.trash().items())
        if int(count) > 0
    ]


def _state_view(session: Session, seat: int) -> dict[str, Any]:
    game = session.game
    opponents: list[dict[str, Any]] = []
    for player in range(game.num_players()):
        if player == seat:
            continue
        opponents.append(
            {
                "seat": player,
                "handCount": game.hand_count(player),
                "deckCount": game.deck_count(player),
                "discardCount": game.discard_count(player),
                "discardTop": game.discard_top(player),
                "inPlay": game.in_play(player),
                "vp": game.score(player) if game.game_over() else None,
            }
        )

    return {
        "piles": [{"def": int(card_def), "count": int(count)} for card_def, count in game.supply()],
        "myHand": _hand_entries(game.hand(seat)),
        "myPlayArea": game.in_play(seat),
        "myDeckCount": game.deck_count(seat),
        "myDiscardCount": game.discard_count(seat),
        "myDiscardTop": game.discard_top(seat),
        "opponents": opponents,
        "trash": _trash_entries(game),
        "trashTop": _trash_entries(game)[-1]["def"] if _trash_entries(game) else None,
        "resources": game.resources(),
        "phase": game.phase(),
        "turn": game.turn(),
    }


def _table_message(session: Session) -> dict[str, Any]:
    return {
        "type": "table",
        "seats": [{"index": index, "kind": seat.kind} for index, seat in enumerate(session.seats)],
        "kingdom": session.kingdom,
        "landscapes": [],
    }


def _state_message(session: Session, seat: int) -> dict[str, Any]:
    return {"type": "state", "view": _state_view(session, seat)}


def _decision_message(session: Session, seat: int) -> dict[str, Any]:
    decision = session.game.current_decision()
    acting = int(decision["player"])
    source = int(decision["source"])
    message = {
        "type": "decision",
        "seat": acting,
        "kind": decision_kind_name(decision),
        "source": {"def": source, "name": def_by_id(source)["name"] if source in load_defs()["by_id"] else ""},
        "prompt": prompt_for(decision) if seat == acting else f"Waiting for P{acting + 1}",
        "options": [],
        "min": int(decision["min"]),
        "max": int(decision["max"]),
    }
    if seat == acting:
        message["options"] = legal_options(session.game.legal_mask(), decision)
    return message


def _gameover_message(session: Session) -> dict[str, Any]:
    scores = [session.game.score(player) for player in range(session.game.num_players())]
    return {
        "type": "gameover",
        "scores": scores,
        "winner": session.game.winner(),
        "truncated": session.game.truncated(),
    }


def _initial_messages(session: Session, seat: int) -> list[dict[str, Any]]:
    messages = [_table_message(session), _state_message(session, seat)]
    if session.game.game_over():
        messages.append(_gameover_message(session))
    else:
        messages.append(_decision_message(session, seat))
    return messages


def _post_step_messages(session: Session, seat: int, lines: list[str]) -> list[dict[str, Any]]:
    messages = [{"type": "log", "lines": lines}, _state_message(session, seat)]
    if session.game.game_over():
        messages.append(_gameover_message(session))
    else:
        messages.append(_decision_message(session, seat))
    return messages


def _is_legal(game: Any, action: int) -> bool:
    if action < 0 or action >= dz.ACTION_SPACE_SIZE:
        return False
    return bool(game.legal_mask()[action])


def _current_player(game: Any) -> int:
    return int(game.current_decision()["player"])


def _choose_bot_action(session: Session) -> int:
    legal = np.flatnonzero(session.game.legal_mask())
    if legal.size == 0:
        return int(dz.A_PASS)
    return int(legal[session.bot_rng.randrange(int(legal.size))])


def _apply_action(session: Session, seat: int, action: int) -> str:
    decision = session.game.current_decision()
    line = log_line(seat, action, decision)
    done = session.game.step(action)
    session.action_log.append(action)
    session.log_lines.append(line)
    if done:
        session.log_lines.append("Game over")
    return line


def _run_bots(session: Session) -> list[str]:
    lines: list[str] = []
    while not session.game.game_over():
        player = _current_player(session.game)
        if player >= len(session.seats) or session.seats[player].kind != "bot":
            break
        action = _choose_bot_action(session)
        if not _is_legal(session.game, action):
            break
        lines.append(_apply_action(session, player, action))
    return lines


async def _broadcast(session: Session, lines: list[str]) -> None:
    stale: list[str] = []
    for token, websocket in list(session.connections.items()):
        seat = _seat_index(session, token)
        if seat < 0:
            stale.append(token)
            continue
        try:
            for message in _post_step_messages(session, seat, lines):
                await websocket.send_json(message)
        except RuntimeError:
            stale.append(token)
    for token in stale:
        session.connections.pop(token, None)


@app.post("/api/session")
async def create_session(payload: dict[str, Any]) -> dict[str, Any]:
    seat_kinds = list(payload.get("seats") or ["human", "bot"])
    if len(seat_kinds) < 2 or len(seat_kinds) > dz.MAX_PLAYERS:
        raise HTTPException(status_code=400, detail="seats must have 2 to MAX_PLAYERS entries")
    for kind in seat_kinds:
        if kind not in {"human", "bot"}:
            raise HTTPException(status_code=400, detail="seat kind must be human or bot")

    seed = int(payload.get("seed", secrets.randbits(63)))
    kingdom = _parse_kingdom(payload)
    setup = _make_setup(len(seat_kinds), kingdom)
    session_id = secrets.token_urlsafe(12)
    seats = [Seat(kind=kind, token=secrets.token_urlsafe(16)) for kind in seat_kinds]
    session = Session(
        session_id=session_id,
        seats=seats,
        setup=setup,
        seed=seed,
        kingdom=kingdom,
        game=dz.new_game(setup, seed),
        bot_rng=random.Random(seed ^ 0xB07),
    )
    sessions[session_id] = session
    return {"session_id": session_id, "seat_tokens": [seat.token for seat in seats]}


@app.websocket("/ws/{session_id}/{seat_token}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, seat_token: str) -> None:
    session = sessions.get(session_id)
    if session is None or _seat_index(session, seat_token) < 0:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    seat = _seat_index(session, seat_token)
    async with session.lock:
        session.connections[seat_token] = websocket
        for message in _initial_messages(session, seat):
            await websocket.send_json(message)

    try:
        while True:
            payload = await websocket.receive_json()
            if payload.get("type") != "act":
                await websocket.send_json({"type": "error", "message": "unsupported message"})
                continue

            action = int(payload.get("action", -1))
            async with session.lock:
                current = _current_player(session.game)
                if current != seat:
                    await websocket.send_json({"type": "error", "message": "not your turn"})
                    continue
                if not _is_legal(session.game, action):
                    await websocket.send_json({"type": "error", "message": "illegal action"})
                    continue

                lines = [_apply_action(session, seat, action)]
                lines.extend(_run_bots(session))
                await _broadcast(session, lines)
    except WebSocketDisconnect:
        async with session.lock:
            if session.connections.get(seat_token) is websocket:
                session.connections.pop(seat_token, None)
