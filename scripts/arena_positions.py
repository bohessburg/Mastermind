"""Rebuild the bot's decision positions from archived arena games.

Replays each game's frames.jsonl through the arena parser and tracker (the same
path the live bot uses) and, at every action- or buy-phase question on the
bot's own turn, stores the engine snapshot the bot would have searched from
plus deck-size features. Output: one JSON line per position.

  PYTHONPATH=build:. python scripts/arena_positions.py --root exports/arena --out bench/thin/positions.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import dominion_v2_py as dz  # noqa: E402
from src.v2.arena.protocol.events import GameStart, PendingDecision  # noqa: E402
from src.v2.arena.protocol.recording import parse_recording  # noqa: E402
from src.v2.arena.shadow.bridge import engine_snapshot  # noqa: E402
from src.v2.arena.shadow.tracker import Tracker  # noqa: E402

PHASE_QUESTIONS = {"GAME_ACTION_PHASE", "GAME_BUY_PHASE"}


def total(multiset) -> int:
    if hasattr(multiset, "values"):
        return int(sum(multiset.values()))
    return int(sum(int(c) for _, c in multiset))


def game_positions(game_dir: Path):
    result = json.loads((game_dir / "result.json").read_text())
    if not result.get("completed"):
        return [], "incomplete"
    game_id = int(result["game_id"])
    our = result.get("our_seat")
    outcome = result.get("outcome")
    events = parse_recording(game_dir / "frames.jsonl").events
    tracker = Tracker()
    started = False
    rows = []
    for event in events:
        if isinstance(event, GameStart):
            if started and int(event.game_id) != game_id:
                break
            started = int(event.game_id) == game_id
            if not started:
                continue
        if not started:
            continue
        try:
            tracker.consume(event)
        except Exception as exc:  # noqa: BLE001
            return rows, f"tracker:{type(exc).__name__}"
        if not isinstance(event, PendingDecision) or event.question_id not in PHASE_QUESTIONS:
            continue
        snap = tracker.snapshot()
        if snap.our_seat is None or snap.turn_owner != snap.our_seat:
            continue
        try:
            engine = engine_snapshot(snap)
            game = dz.game_from_snapshot(engine)
            game.validate()
        except Exception:  # noqa: BLE001
            continue
        decision = game.current_decision()
        if int(decision.get("player", -1)) != snap.our_seat:
            continue
        if int(game.legal_mask().sum()) < 2:
            continue
        me, opp = snap.seats[snap.our_seat], snap.seats[1 - snap.our_seat]
        rows.append({
            "game_dir": str(game_dir),
            "game_id": game_id,
            "outcome": outcome,
            "our_seat": snap.our_seat,
            "turn": snap.turn_number,
            "question": event.question_id,
            "legal": int(game.legal_mask().sum()),
            "our_owned": total(me.owned),
            "opp_owned": total(opp.owned),
            "our_draw": int(me.deck_count),
            "our_discard": int(me.discard_count),
            "snapshot": engine,
        })
    return rows, "ok"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="exports/arena")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    dirs = sorted(d.parent for d in Path(args.root).glob("*/*-game-*/result.json"))
    if args.limit:
        dirs = dirs[-args.limit:]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    status: dict[str, int] = {}
    n = 0
    start = time.time()
    with out.open("w") as fh:
        for i, d in enumerate(dirs):
            try:
                rows, st = game_positions(d)
            except Exception as exc:  # noqa: BLE001
                rows, st = [], f"error:{type(exc).__name__}"
            status[st] = status.get(st, 0) + 1
            for r in rows:
                fh.write(json.dumps(r, default=list) + "\n")
            n += len(rows)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(dirs)} games {n} positions {time.time() - start:.0f}s {status}", flush=True)
    print(f"DONE {len(dirs)} games {n} positions {status}", flush=True)


if __name__ == "__main__":
    main()
