#!/usr/bin/env python3
"""Find the arena bot's costliest moves in games it lost.

For each archived arena game, rebuild every action- and buy-phase position on
the bot's own turn (the arena parser and tracker, as in arena_positions.py),
look up the move the bot actually played (decisions.jsonl), and score it
against the policy's top-K moves with policy playouts (as in
c29_decision_regret.py): sample a world the bot cannot tell from the real one,
play the candidate, let the net's policy play both seats to the end.  All
candidates of a position share the same worlds, so differences are paired.

Reported per position, in win probability for the bot:
  q_played   playout value of the bot's move
  q_best     the best candidate, picked on one half of the playouts and
             scored on the other (both ways, averaged), so noise alone does
             not create a regret
  regret     q_best - q_played

A blunder in a winnable position is a large regret where q_best is high.
The playout policy is the raw net, weaker than the bot's 1024-sim search, so
treat single positions as leads to inspect rather than verdicts.

  PYTHONPATH=build python scripts/arena_blunders.py --rows-from 1064 \
      --results loss,tie --min-value 0 --out bench/arena_blunders/g51_losses.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
import dominion_v2_py as dz  # noqa: E402
from src.v2.arena.protocol.events import GameStart, PendingDecision  # noqa: E402
from src.v2.arena.protocol.recording import parse_recording  # noqa: E402
from src.v2.arena.shadow.bridge import engine_snapshot  # noqa: E402
from src.v2.arena.shadow.tracker import Tracker  # noqa: E402
from src.v2.train.evaluate import load_model  # noqa: E402
from src.v2.train.diagnostics import pick_device  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from c29_decision_regret import masked_logits  # noqa: E402

PHASE_QUESTIONS = {"GAME_ACTION_PHASE", "GAME_BUY_PHASE"}
HEARTBEAT_S = 20.0
CARD_NAMES = (
    "Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse",
    "Artisan", "Bandit", "Bureaucrat", "Cellar", "Chapel", "Council Room",
    "Festival", "Gardens", "Harbinger", "Laboratory", "Library", "Market",
    "Merchant", "Militia", "Mine", "Moat", "Moneylender", "Poacher", "Remodel",
    "Sentry", "Smithy", "Throne Room", "Vassal", "Village", "Witch", "Workshop",
)
TREASURE_PLAYS = {int(dz.A_PLAY_BASE) + int(dz.def_id(name)) for name in ("Copper", "Silver", "Gold")}


def action_label(action: int) -> str:
    names = {int(dz.def_id(name)): name for name in CARD_NAMES}
    if action == int(dz.A_PASS):
        return "pass"
    for base, verb, top in (
        (dz.A_PLAY_BASE, "play", dz.A_BUY_BASE),
        (dz.A_BUY_BASE, "buy", dz.A_SELECT_BASE),
        (dz.A_SELECT_BASE, "select", dz.A_OPTION_BASE),
    ):
        if int(base) <= action < int(top):
            return f"{verb} {names.get(action - int(base), action - int(base))}"
    return f"action {action}"


def game_positions(game_dir: Path) -> list[dict]:
    """Bot-turn phase positions with the bot's played first engine action."""
    result = json.loads((game_dir / "result.json").read_text())
    if not result.get("completed"):
        return []
    game_id = int(result["game_id"])
    played: dict[int, int] = {}
    for line in (game_dir / "decisions.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("engine_actions"):
            played[int(record["question_index"])] = int(record["engine_actions"][0])
    tracker, started, rows = Tracker(), False, []
    for event in parse_recording(game_dir / "frames.jsonl").events:
        if isinstance(event, GameStart):
            if started and int(event.game_id) != game_id:
                break
            started = int(event.game_id) == game_id
        if not started:
            continue
        try:
            tracker.consume(event)
        except Exception:  # noqa: BLE001
            return rows
        if not isinstance(event, PendingDecision) or event.question_id not in PHASE_QUESTIONS:
            continue
        action = played.get(int(event.question_index))
        if action is None or action in TREASURE_PLAYS:
            continue
        snap = tracker.snapshot()
        if snap.our_seat is None or snap.turn_owner != snap.our_seat:
            continue
        try:
            game = dz.game_from_snapshot(engine_snapshot(snap))
            game.validate()
        except Exception:  # noqa: BLE001
            continue
        mask = np.asarray(game.legal_mask(), dtype=bool)
        if int(game.current_decision().get("player", -1)) != snap.our_seat or mask.sum() < 2 or not mask[action]:
            continue
        rows.append({
            "game_dir": str(game_dir), "game_id": game_id, "result": result.get("outcome"),
            "turn": snap.turn_number, "question_index": int(event.question_index),
            "seat": snap.our_seat, "played": action, "game": game,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="default: configs/arena_c29.json checkpoint")
    parser.add_argument("--ledger", default="exports/arena/record.jsonl")
    parser.add_argument("--rows-from", type=int, default=0, help="first ledger row (gen 51 starts at 1064)")
    parser.add_argument("--results", default="loss,tie", help="comma list of win/loss/tie")
    parser.add_argument("--min-value", type=float, default=None,
                        help="keep positions whose value head (bot's view, -1..1) is at least this")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--playouts", type=int, default=128, help="per candidate; even, for split-half")
    parser.add_argument("--slots", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="bench/arena_blunders/report.json")
    args = parser.parse_args()
    if args.playouts < 2 or args.playouts % 2:
        parser.error("--playouts must be even")

    device = pick_device(args.device)
    checkpoint = Path(args.model or json.loads(Path("configs/arena_c29.json").read_text())["checkpoint_path"])
    model, config = load_model(checkpoint, device)
    selfplay = config.selfplay
    wanted = set(args.results.split(","))
    ledger = [json.loads(line) for line in Path(args.ledger).read_text().splitlines()][args.rows_from:]
    games = [row for row in ledger if row["result"] in wanted]
    started = time.time()
    positions: list[dict] = []
    for number, row in enumerate(games, start=1):
        for game_dir in glob.glob(f"{row['run_dir']}/*-game-{row['game_id']}"):
            positions.extend(game_positions(Path(game_dir)))
        if number % 10 == 0 or number == len(games):
            print(f"positions: {number}/{len(games)} games, {len(positions)} positions, {time.time() - started:.0f}s", flush=True)
    if not positions:
        raise SystemExit("no positions")

    obs = np.stack([np.asarray(p["game"].encode(p["seat"], int(selfplay.obs_version)), dtype=np.float32) for p in positions])
    masks = np.stack([np.asarray(p["game"].legal_mask(), dtype=bool) for p in positions])
    logits, values = masked_logits(model, obs, masks, device)
    if args.min_value is not None:
        keep = np.flatnonzero(values >= args.min_value)
        print(f"value >= {args.min_value}: {keep.size} of {len(positions)} positions", flush=True)
        positions = [positions[i] for i in keep]
        obs, masks, logits, values = obs[keep], masks[keep], logits[keep], values[keep]
    slots = args.top_k + 1
    cand = np.full((len(positions), slots), -1, dtype=np.int64)
    for row, position in enumerate(positions):
        legal = np.flatnonzero(masks[row])
        ranked = legal[np.argsort(-logits[row, legal], kind="stable")]
        order = [position["played"], *[int(a) for a in ranked[: args.top_k]]]
        unique = list(dict.fromkeys(order))[:slots]
        cand[row, : len(unique)] = unique
    n_cand = (cand >= 0).sum(axis=1)

    runner = dz.RolloutRunner(
        n_slots=args.slots, obs_version=int(selfplay.obs_version),
        auto_play_treasures=bool(selfplay.auto_play_treasures),
        prune_treasure_plays=bool(selfplay.prune_treasure_plays),
        max_turns=int(selfplay.max_turns), temperature=0.0,
    )
    position_ids = np.asarray([runner.add_position(p["game"]) for p in positions], dtype=np.uint32)
    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(1, 2**63 - 1, size=(len(positions), args.playouts), dtype=np.uint64)
    job_pos = np.repeat(np.arange(len(positions)), n_cand * args.playouts)
    job_cand = np.concatenate([np.repeat(np.arange(k), args.playouts) for k in n_cand])
    job_seed = np.concatenate([np.tile(np.arange(args.playouts), k) for k in n_cand])
    runner.add_jobs(
        position_ids[job_pos], cand[job_pos, job_cand].astype(np.uint32), seeds[job_pos, job_seed],
        np.arange(job_pos.size, dtype=np.uint32), determinize=True,
    )
    total = int(job_pos.size)
    print(f"{len(positions)} positions, {total} playouts", flush=True)
    playout_started = last_beat = time.time()
    while True:
        leaf_obs, leaf_masks = runner.collect_leaves()
        if leaf_obs.shape[0] == 0:
            break
        runner.provide_policies(masked_logits(model, leaf_obs, leaf_masks, device)[0])
        if time.time() - last_beat >= HEARTBEAT_S:
            last_beat = time.time()
            finished = total - runner.queued_jobs() - runner.active_rollouts()
            elapsed = last_beat - playout_started
            print(f"playouts {finished}/{total} {runner.decisions_evaluated() / max(elapsed, 1e-9):.0f} evals/s {elapsed:.0f}s", flush=True)
    results = runner.take_results()
    tags = np.asarray(results["tags"], dtype=np.int64)
    outcomes = np.full((len(positions), slots, args.playouts), np.nan, dtype=np.float32)
    outcomes[job_pos[tags], job_cand[tags], job_seed[tags]] = results["outcomes"]

    half = args.playouts // 2
    q = np.nanmean(outcomes, axis=2)
    regrets, bests = [], []
    for select, score in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
        q_select = np.nanmean(outcomes[:, :, select], axis=2)
        q_score = np.nanmean(outcomes[:, :, score], axis=2)
        best = np.nanargmax(np.where(np.isnan(q_select), -np.inf, q_select), axis=1)
        rows = np.arange(len(positions))
        regrets.append(q_score[rows, best] - q_score[rows, 0])
        bests.append(q_score[rows, best])
    regret = np.mean(regrets, axis=0)
    q_best = np.mean(bests, axis=0)
    report = []
    for row, position in enumerate(positions):
        report.append({
            "game_id": position["game_id"], "result": position["result"], "turn": position["turn"],
            "question_index": position["question_index"], "value": float(values[row]),
            "played": action_label(position["played"]), "q_played": float(q[row, 0]),
            "q_best": float(q_best[row]), "regret": float(regret[row]),
            "candidates": {action_label(int(a)): round(float(q[row, c]), 3) for c, a in enumerate(cand[row, : n_cand[row]])},
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(checkpoint), "rows_from": args.rows_from, "results": sorted(wanted),
        "games": len(games), "positions": len(positions), "playouts": args.playouts, "top_k": args.top_k,
        "wall_s": round(time.time() - started), "per_position": report,
    }, indent=1) + "\n")
    print(f"DONE {len(positions)} positions from {len(games)} games, {total} playouts, {time.time() - started:.0f}s -> {out}", flush=True)
    print(f"mean regret {100 * regret.mean():+.2f} pts; positions with regret > 10 pts: {(regret > 0.10).sum()}", flush=True)
    for entry in sorted(report, key=lambda e: -e["regret"])[:20]:
        print(f"  game {entry['game_id']} t{entry['turn']} {entry['played']:>16s}: q {entry['q_played']:.2f} -> best {entry['q_best']:.2f} "
              f"(regret {100 * entry['regret']:+.0f}) {entry['candidates']}", flush=True)


if __name__ == "__main__":
    main()
