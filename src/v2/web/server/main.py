from __future__ import annotations

import asyncio
import random
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import dominion_v2_py as dz

from .defs import def_by_id, def_id, kingdom_def_ids, load_defs
from .observer import (
    SentryLogState,
    capture_public_snapshot,
    decision_kind_name,
    legal_options,
    prompt_for,
    public_log_lines,
    sentry_resolution_logs,
)


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
    action_log_line_counts: list[int] = field(default_factory=list)
    human_decision_prefixes: list[int] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    sentry_log_states: dict[int, SentryLogState] = field(default_factory=dict)
    connections: dict[str, WebSocket] = field(default_factory=dict)
    bot_rngs: list[random.Random] = field(default_factory=list)
    thinking_delay_ms: int = 600
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


app = FastAPI(title="DominionZero v2 Web Server")
sessions: dict[str, Session] = {}
SEAT_KINDS = {"human", "bot", "bot:bigmoney", "bot:random"}


def _make_setup(players: int, kingdom: list[int]) -> dz.Setup:
    return dz.Setup(players=players, kingdom=kingdom)


def _parse_kingdom(payload: dict[str, Any], seed: int) -> list[int]:
    raw = payload.get("kingdom") or DEFAULT_KINGDOM
    if raw == "random":
        # Seeded so the same session seed reproduces the same kingdom.
        return sorted(random.Random(seed).sample(kingdom_def_ids(), 10))
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


def _is_bot_kind(kind: str) -> bool:
    return kind == "bot" or kind.startswith("bot:")


def _bot_policy(kind: str) -> str:
    if kind == "bot":
        return "bigmoney"
    if kind.startswith("bot:"):
        return kind.split(":", 1)[1]
    return ""


def _is_human_vs_bot(session: Session) -> bool:
    humans = sum(1 for seat in session.seats if seat.kind == "human")
    bots = sum(1 for seat in session.seats if _is_bot_kind(seat.kind))
    return len(session.seats) == 2 and humans == 1 and bots == 1


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
                "resources": game.resources(player),
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
        "mySetAside": game.set_aside(seat),
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
    context = session.game.decision_context() if seat == acting else {}
    message = {
        "type": "decision",
        "seat": acting,
        "kind": decision_kind_name(decision),
        "source": {"def": source, "name": def_by_id(source)["name"] if source in load_defs()["by_id"] else ""},
        "prompt": prompt_for(decision, context) if seat == acting else f"Waiting for P{acting + 1}",
        "options": [],
        "min": int(decision["min"]),
        "max": int(decision["max"]),
    }
    if seat == acting:
        message["options"] = legal_options(session.game.legal_mask(), decision, context)
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


def _post_step_messages(
    session: Session,
    seat: int,
    lines: list[str],
    private_lines_by_seat: dict[int, list[str]] | None = None,
) -> list[dict[str, Any]]:
    seat_lines = [*lines, *(private_lines_by_seat or {}).get(seat, [])]
    messages = [{"type": "log", "lines": seat_lines}, _state_message(session, seat)]
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


def _legal_actions(game: Any) -> list[int]:
    return [int(action) for action in np.flatnonzero(game.legal_mask())]


def _is_treasure(def_value: int) -> bool:
    return "Treasure" in def_by_id(def_value)["types"]


def _coin_value(def_value: int) -> int:
    return int(def_by_id(def_value).get("coin_value", 0))


def _cost_coins(def_value: int) -> int:
    return int(def_by_id(def_value)["cost"]["coins"])


def _select_def(action: int) -> int:
    return int(action) - int(dz.A_SELECT_BASE)


def _option_index(action: int) -> int:
    return int(action) - int(dz.A_OPTION_BASE)


def _first_legal(legal: list[int]) -> int:
    if not legal:
        return int(dz.A_PASS)
    return int(legal[0])


def _choose_bigmoney_action(game: Any, legal: list[int]) -> int:
    decision = game.current_decision()
    kind = decision_kind_name(decision)
    source = int(decision["source"])
    source_name = def_by_id(source)["name"] if source in load_defs()["by_id"] else ""
    legal_set = set(legal)

    if kind == "ReactWindow":
        moat = int(dz.A_SELECT_BASE + dz.DEF_MOAT)
        if moat in legal_set:
            return moat
        if int(dz.A_PASS) in legal_set:
            return int(dz.A_PASS)

    if kind == "PhaseBuy":
        for def_value in (dz.DEF_PLATINUM, dz.DEF_GOLD, dz.DEF_SILVER, dz.DEF_COPPER, dz.DEF_POTION):
            action = int(dz.A_PLAY_BASE + def_value)
            if action in legal_set:
                return action
        for def_value in (dz.DEF_PROVINCE, dz.DEF_GOLD, dz.DEF_SILVER):
            action = int(dz.A_BUY_BASE + def_value)
            if action in legal_set:
                return action

    if kind == "Choose" and source_name == "Militia":
        selects = [action for action in legal if dz.A_SELECT_BASE <= action < dz.A_OPTION_BASE]
        if selects:
            return max(
                selects,
                key=lambda action: (
                    _coin_value(_select_def(action)),
                    1 if _is_treasure(_select_def(action)) else 0,
                    _cost_coins(_select_def(action)),
                    -_select_def(action),
                ),
            )

    if kind == "Choose" and source_name == "Bureaucrat":
        selects = [action for action in legal if dz.A_SELECT_BASE <= action < dz.A_OPTION_BASE]
        if selects:
            return min(selects, key=lambda action: (_cost_coins(_select_def(action)), _select_def(action)))

    if kind in {"Choose", "ChooseGain"} and int(dz.A_PASS) in legal_set:
        return int(dz.A_PASS)

    if kind == "ChooseOption":
        options = [action for action in legal if dz.A_OPTION_BASE <= action < dz.A_CALL_BASE]
        if options:
            return min(options, key=_option_index)
        if int(dz.A_PASS) in legal_set:
            return int(dz.A_PASS)

    if kind in {"PhaseAction", "PhaseNight"} and int(dz.A_PASS) in legal_set:
        return int(dz.A_PASS)

    if int(dz.A_PASS) in legal_set:
        return int(dz.A_PASS)
    return _first_legal(legal)


def _choose_bot_action(session: Session, seat: int) -> int:
    legal = _legal_actions(session.game)
    if not legal:
        return int(dz.A_PASS)

    policy = _bot_policy(session.seats[seat].kind)
    if policy == "random":
        rng = session.bot_rngs[seat]
        return int(legal[rng.randrange(len(legal))])
    return _choose_bigmoney_action(session.game, legal)


def _apply_action(session: Session, seat: int, action: int) -> tuple[list[str], dict[int, list[str]]]:
    decision = session.game.current_decision()
    decision_context = session.game.decision_context()
    before = capture_public_snapshot(session.game)
    done = session.game.step(action)
    after = capture_public_snapshot(session.game)
    after_decision = session.game.current_decision()
    lines = public_log_lines(seat, action, decision, before, after, decision_context, after_decision)
    resolution_lines, private_lines = sentry_resolution_logs(
        session.sentry_log_states,
        seat,
        action,
        decision,
        decision_context,
        after_decision,
    )
    lines.extend(resolution_lines)
    session.action_log.append(action)
    session.action_log_line_counts.append(len(lines) + (1 if done else 0))
    session.log_lines.extend(lines)
    if done:
        session.log_lines.append("Game over")
        lines = [*lines, "Game over"]
    return lines, private_lines


def _apply_validated_action(session: Session, seat: int, action: int) -> tuple[list[str], dict[int, list[str]]]:
    current = _current_player(session.game)
    if current != seat:
        raise ValueError("not your turn")
    if not _is_legal(session.game, action):
        raise ValueError("illegal action")
    return _apply_action(session, seat, action)


async def _run_bots(session: Session) -> None:
    while not session.game.game_over():
        player = _current_player(session.game)
        if player >= len(session.seats) or not _is_bot_kind(session.seats[player].kind):
            break
        if session.thinking_delay_ms > 0:
            await asyncio.sleep(session.thinking_delay_ms / 1000.0)
        action = _choose_bot_action(session, player)
        lines, private_lines = _apply_validated_action(session, player, action)
        await _broadcast(session, lines, private_lines)


def _replay_game(session: Session, actions: list[int]) -> Any:
    game = dz.new_game(session.setup, session.seed)
    for action in actions:
        if not _is_legal(game, int(action)):
            raise RuntimeError(f"recorded action is illegal during replay: {action}")
        game.step(int(action))
    return game


def _undo_previous_human_decision(session: Session, seat: int) -> str:
    if not _is_human_vs_bot(session):
        raise ValueError("undo is only available in human-vs-bot sessions")
    if seat < 0 or session.seats[seat].kind != "human":
        raise ValueError("only a human seat can undo")
    if not session.human_decision_prefixes:
        raise ValueError("nothing to undo")

    prefix = session.human_decision_prefixes.pop()
    log_prefix = sum(session.action_log_line_counts[:prefix])
    session.action_log = session.action_log[:prefix]
    session.action_log_line_counts = session.action_log_line_counts[:prefix]
    session.human_decision_prefixes = [
        previous for previous in session.human_decision_prefixes if previous < prefix
    ]
    session.game = _replay_game(session, session.action_log)
    session.log_lines = session.log_lines[:log_prefix]
    session.sentry_log_states.clear()
    line = "Undo: rewound to previous human decision"
    session.log_lines.append(line)
    return line


async def _broadcast(
    session: Session,
    lines: list[str],
    private_lines_by_seat: dict[int, list[str]] | None = None,
) -> None:
    stale: list[str] = []
    for token, websocket in list(session.connections.items()):
        seat = _seat_index(session, token)
        if seat < 0:
            stale.append(token)
            continue
        try:
            for message in _post_step_messages(session, seat, lines, private_lines_by_seat):
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
        if kind not in SEAT_KINDS:
            raise HTTPException(status_code=400, detail="seat kind must be human, bot, bot:bigmoney, or bot:random")

    seed = int(payload.get("seed", secrets.randbits(63)))
    thinking_delay_ms = int(payload.get("thinking_delay_ms", payload.get("thinkingDelayMs", 600)))
    if thinking_delay_ms < 0:
        raise HTTPException(status_code=400, detail="thinking delay must be non-negative")
    kingdom = _parse_kingdom(payload, seed)
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
        bot_rngs=[
            random.Random(seed ^ 0xB07 ^ ((index + 1) * 0x9E3779B97F4A7C15))
            for index in range(len(seats))
        ],
        thinking_delay_ms=thinking_delay_ms,
    )
    sessions[session_id] = session
    return {"session_id": session_id, "seat_tokens": [seat.token for seat in seats]}


@app.get("/api/session/{session_id}/export")
async def export_session(session_id: str) -> dict[str, Any]:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    async with session.lock:
        return {
            "seed": session.seed,
            "kingdom": session.kingdom,
            "seats": [seat.kind for seat in session.seats],
            "actions": session.action_log,
            "obs_version": int(dz.OBS_VERSION),
            "final_state_hash": f"0x{session.game.state_hash():016x}",
        }


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
            if payload.get("type") == "undo_request":
                async with session.lock:
                    try:
                        line = _undo_previous_human_decision(session, seat)
                    except ValueError as error:
                        await websocket.send_json({"type": "error", "message": str(error)})
                        continue
                    await _broadcast(session, [line])
                continue

            if payload.get("type") != "act":
                await websocket.send_json({"type": "error", "message": "unsupported message"})
                continue

            action = int(payload.get("action", -1))
            async with session.lock:
                try:
                    if session.seats[seat].kind == "human":
                        session.human_decision_prefixes.append(len(session.action_log))
                    lines, private_lines = _apply_validated_action(session, seat, action)
                except ValueError as error:
                    if session.human_decision_prefixes and session.human_decision_prefixes[-1] == len(session.action_log):
                        session.human_decision_prefixes.pop()
                    await websocket.send_json({"type": "error", "message": str(error)})
                    continue

                await _broadcast(session, lines, private_lines)
                await _run_bots(session)
    except WebSocketDisconnect:
        async with session.lock:
            if session.connections.get(seat_token) is websocket:
                session.connections.pop(seat_token, None)


CLIENT_DIST = Path(__file__).resolve().parents[1] / "client" / "dist"
if CLIENT_DIST.exists():
    assets_dir = CLIENT_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_client(full_path: str) -> FileResponse:
        candidate = (CLIENT_DIST / full_path).resolve()
        if full_path and candidate.is_file() and CLIENT_DIST in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(CLIENT_DIST / "index.html")
