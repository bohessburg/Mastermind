#!/usr/bin/env python3
"""Same checkpoint, different numbers of sampled hidden-information worlds.

The native ``SelfPlayRunner`` searches one sampled world per decision, so a
K-world comparison has to go through ``dz.DecisionSearcher``, the arena bot's
searcher (sims split across K worlds).  To run it at batch speed rather than
one decision at a time, each worker process plays ``--concurrency`` games at
once and sends every pending search leaf through one batched network forward,
as self-play does.  That changes speed, not play: each searcher gets exactly
the evaluations it asked for.  The arena's 30 s cap is not modelled.

Each board is a random 10-card base kingdom played in both seat orders. Every
finished game is appended to ``--out`` as one JSON line, so a run can be
stopped and resumed (finished games are skipped).

  python scripts/worlds_duel.py --a-worlds 8 --b-worlds 2 --sims 256 --boards 200 --workers 4
  python scripts/worlds_duel.py --a-worlds 2 --b-worlds 2 --a-sims 1024 --b-sims 400 --out bench/worlds/w2_1024_vs_400.jsonl
  python scripts/worlds_duel.py --summary --out bench/worlds/w8_vs_w2_256.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

BASE_CARDS = (
    "Artisan", "Bandit", "Bureaucrat", "Cellar", "Chapel", "Council Room", "Festival",
    "Gardens", "Harbinger", "Laboratory", "Library", "Market", "Merchant", "Militia",
    "Mine", "Moat", "Moneylender", "Poacher", "Remodel", "Sentry", "Smithy",
    "Throne Room", "Vassal", "Village", "Witch", "Workshop",
)
MAX_DECISIONS = 3000


def searcher_config(policy, game, seat: int, sims: int, worlds: int) -> dict:
    """``choose_nnmcts_action``'s DecisionSearcher settings, with a chosen world count."""
    return {
        "sims": sims,
        "c_puct": 1.25,
        "determinizations": worlds,
        "obs_version": policy.obs_version,
        "auto_play_treasures": True,
        "prune_treasure_plays": True,
        "terminal_margin_scale": float(policy.terminal_margin_scale),
        "terminal_margin_alpha": float(policy.terminal_margin_alpha),
        "seed": (int(game.state_hash()) ^ ((seat + 1) * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF,
    }


class Slot:
    def __init__(self, dz, board: int, kingdom: list[str], seed: int, a_seat: int):
        self.board, self.kingdom, self.seed, self.a_seat = board, kingdom, seed, a_seat
        self.game = dz.new_game(dz.Setup(players=2, kingdom=list(kingdom)), seed)
        self.searcher = None
        self.decisions = 0
        self.started = time.monotonic()

    def over(self) -> bool:
        return self.game.game_over() or self.game.truncated() or self.decisions >= MAX_DECISIONS

    def result(self) -> dict:
        game = self.game
        winner = game.winner()
        if winner is None or self.decisions >= MAX_DECISIONS:
            score = 0.5
        else:
            score = 1.0 if int(winner) == self.a_seat else 0.0
        return {
            "board": self.board, "a_seat": self.a_seat, "a_score": score,
            "scores": [int(game.score(0)), int(game.score(1))], "turns": int(game.turn()),
            "truncated": bool(game.truncated()), "decisions": self.decisions,
            "kingdom": self.kingdom, "seed": self.seed, "seconds": round(time.monotonic() - self.started, 1),
        }


def worker(index: int, args: argparse.Namespace, jobs: list[tuple[int, list[str], int, int]]) -> None:
    import dominion_v2_py as dz
    import numpy as np
    import torch
    from src.v2.arena.bot import policy as P

    policy = P.load_policy(Path(args.checkpoint), obs_version=3, device=args.device, legacy_shim="auto")
    queue = list(jobs)
    slots: list[Slot] = []
    action_space = int(dz.ACTION_SPACE_SIZE)

    def refill() -> None:
        while queue and len(slots) < args.concurrency:
            slots.append(Slot(dz, *queue.pop(0)))

    def advance(slot: Slot) -> None:
        # Forced moves are played directly; searching them cannot change the game.
        while slot.searcher is None and not slot.over():
            game = slot.game
            seat = int(game.current_decision()["player"])
            legal = np.flatnonzero(np.asarray(game.legal_mask()))
            if len(legal) == 1:
                game.step(int(legal[0]))
                slot.decisions += 1
                continue
            a_side = seat == slot.a_seat
            worlds = args.a_worlds if a_side else args.b_worlds
            sims = args.a_sims if a_side else args.b_sims
            slot.searcher = dz.DecisionSearcher(game, seat, searcher_config(policy, game, seat, sims, worlds))

    refill()
    while slots:
        for slot in slots:
            advance(slot)
        for slot in [s for s in slots if s.searcher is None and s.over()]:
            row = slot.result()
            row["worker"] = index
            with open(args.out, "a") as out:
                out.write(json.dumps(row) + "\n")
            slots.remove(slot)
        refill()
        batch, owners = [], []
        for slot in slots:
            if slot.searcher is None or slot.searcher.done():
                continue
            obs, masks = slot.searcher.collect_leaves()
            if obs.shape[0]:
                batch.append((obs, masks))
                owners.append(slot)
        if batch:
            obs = np.concatenate([b[0] for b in batch])
            masks = np.concatenate([b[1] for b in batch])
            with torch.no_grad():
                logits, values = policy.evaluate(
                    torch.as_tensor(obs, dtype=torch.float32, device=policy.device),
                    torch.as_tensor(masks, dtype=torch.bool, device=policy.device),
                )
            values = values.detach().cpu().numpy().astype(np.float32, copy=False)
            logits = logits.detach().cpu().numpy().astype(np.float32, copy=False)
            row = 0
            for (leaf_obs, _), slot in zip(batch, owners):
                count = int(leaf_obs.shape[0])
                slot.searcher.provide_evaluations(values[row : row + count], logits[row : row + count])
                row += count
        for slot in slots:
            if slot.searcher is not None and slot.searcher.done():
                action = int(slot.searcher.best_action())
                legal_mask = slot.game.legal_mask()
                if not (0 <= action < action_space and bool(legal_mask[action])):
                    action = int(np.flatnonzero(np.asarray(legal_mask))[0])
                slot.game.step(action)
                slot.decisions += 1
                slot.searcher = None


def summarize(path: Path, label: str) -> str:
    rows = [json.loads(line) for line in open(path)] if path.exists() else []
    if not rows:
        return f"{label}: no games yet"
    n = len(rows)
    score = sum(r["a_score"] for r in rows) / n
    wins = sum(r["a_score"] == 1.0 for r in rows)
    ties = sum(r["a_score"] == 0.5 for r in rows)
    by_board: dict[int, list[float]] = {}
    for r in rows:
        by_board.setdefault(r["board"], []).append(r["a_score"])
    pairs = [sum(v) / 2 for v in by_board.values() if len(v) == 2]
    if len(pairs) > 1:
        mean = sum(pairs) / len(pairs)
        se = math.sqrt(sum((p - mean) ** 2 for p in pairs) / (len(pairs) - 1) / len(pairs))
    else:
        se = math.sqrt(max(score * (1 - score), 1e-9) / n)
    clipped = min(max(score, 1e-3), 1 - 1e-3)
    elo = -400 * math.log10(1 / clipped - 1)
    return (
        f"{label}: {n} games, A {wins}W-{n - wins - ties}L-{ties}T, score {score:.3f} +/- {1.96 * se:.3f} (95%), "
        f"~{elo:+.0f} Elo, truncated {sum(r['truncated'] for r in rows)}, mean {sum(r['seconds'] for r in rows) / n:.0f}s/game"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/remote/campaign31/gen_0027.pt")
    parser.add_argument("--sims", type=int, default=256, help="total sims per decision, split across the worlds")
    parser.add_argument("--a-sims", type=int, default=None, help="side A's sims (default --sims)")
    parser.add_argument("--b-sims", type=int, default=None, help="side B's sims (default --sims)")
    parser.add_argument("--a-worlds", type=int, default=8)
    parser.add_argument("--b-worlds", type=int, default=2)
    parser.add_argument("--boards", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=50, help="games in flight per worker")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--out", type=Path, default=Path("bench/worlds/w8_vs_w2_256.jsonl"))
    parser.add_argument("--summary", action="store_true", help="print the tally of --out and exit")
    args = parser.parse_args()
    args.a_sims = args.sims if args.a_sims is None else args.a_sims
    args.b_sims = args.sims if args.b_sims is None else args.b_sims
    label = (
        f"{args.a_worlds} vs {args.b_worlds} worlds @ {args.sims} sims"
        if args.a_sims == args.b_sims
        else f"{args.a_worlds}x{args.a_sims} vs {args.b_worlds}x{args.b_sims} (worlds x total sims)"
    )
    if args.summary:
        print(summarize(args.out, label))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    boards = [(b, sorted(rng.sample(BASE_CARDS, 10)), rng.randrange(1, 2**31)) for b in range(args.boards)]
    done = {(r["board"], r["a_seat"]) for r in map(json.loads, open(args.out))} if args.out.exists() else set()
    jobs = [(b, k, s, seat) for b, k, s in boards for seat in (0, 1) if (b, seat) not in done]
    print(f"{label}: {len(jobs)} games to play ({len(done)} done), {args.workers} workers x {args.concurrency}", flush=True)
    chunks = [jobs[i :: args.workers] for i in range(args.workers)]
    procs = [mp.Process(target=worker, args=(i, args, chunk)) for i, chunk in enumerate(chunks) if chunk]
    for proc in procs:
        proc.start()
    last = -1
    while any(proc.is_alive() for proc in procs):
        time.sleep(60)
        n = sum(1 for _ in open(args.out)) if args.out.exists() else 0
        if n != last:
            print(time.strftime("%H:%M:%S", time.gmtime()), summarize(args.out, label), flush=True)
            last = n
    for proc in procs:
        proc.join()
    print("final", summarize(args.out, label), flush=True)
    return 0 if all(proc.exitcode == 0 for proc in procs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
