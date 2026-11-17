"""Does the sampled hidden world matter more when decks are thin?

For arena decision positions (scripts/arena_positions.py), runs the arena's own
DecisionSearcher at several (worlds, sims) settings with different seeds, plus
one large many-world reference search, and reports by deck size:

  flip   - share of seed pairs at 1 world x 128 that pick different moves
  agree  - share of runs whose move matches the reference's most-visited move
  regret - reference value of the reference move minus that of the chosen move

Many searchers run at once and their leaves share one inference batch.

  PYTHONPATH=build-thin:. python scripts/thin_worlds.py --positions bench/thin/positions.jsonl \
      --checkpoint checkpoints/remote/campaign33/gen_0051.pt --out bench/thin/results.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import dominion_v2_py as dz  # noqa: E402
from src.v2.arena.bot.policy import load_policy  # noqa: E402

BUCKETS = [(0, 6), (7, 9), (10, 12), (13, 17), (18, 25), (26, 999)]
# (label, worlds, total sims, seeds)
ARMS = [
    ("1x128", 1, 128, 6),
    ("2x512", 2, 1024, 3),   # what the arena plays
    ("8x128", 8, 1024, 3),
    ("1x1024", 1, 1024, 3),
]
REF = ("ref32x256", 32, 8192, 1)


def bucket_of(n: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 999 else f"{lo}+"
    return "?"


def int_keys(value):
    """JSON turned the snapshot's integer card-id keys into strings; undo that."""
    if isinstance(value, dict):
        return {(int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): int_keys(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [int_keys(v) for v in value]
    return value


def make_searcher(game, seat, policy, worlds, sims, seed):
    return dz.DecisionSearcher(game, seat, {
        "sims": sims, "c_puct": 1.25, "determinizations": worlds,
        "obs_version": policy.obs_version, "auto_play_treasures": True, "prune_treasure_plays": True,
        "terminal_margin_scale": float(policy.terminal_margin_scale),
        "terminal_margin_alpha": float(policy.terminal_margin_alpha),
        "seed": seed & 0xFFFFFFFFFFFFFFFF,
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--positions", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--per-bucket", type=int, default=150)
    p.add_argument("--concurrent", type=int, default=96)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rng = random.Random(args.seed)
    by_bucket = defaultdict(list)
    for line in open(args.positions):
        r = json.loads(line)
        by_bucket[bucket_of(int(r["our_owned"]))].append(r)
    chosen = []
    for b, rows in sorted(by_bucket.items()):
        rng.shuffle(rows)
        chosen.extend(rows[: args.per_bucket])
        print(f"bucket {b}: {len(rows)} positions, using {min(len(rows), args.per_bucket)}", flush=True)

    policy = load_policy(Path(args.checkpoint), device=args.device)
    torch = policy.torch

    jobs = []  # (position index, arm label, seed index, worlds, sims)
    for i, _ in enumerate(chosen):
        for label, worlds, sims, seeds in ARMS + [REF]:
            for s in range(seeds):
                jobs.append((i, label, s, worlds, sims))
    games = {}
    results = defaultdict(dict)  # position -> {(label, s): stats}
    pending = iter(jobs)
    active = []
    done_jobs = 0
    evals = 0
    start = last = time.time()

    def launch():
        try:
            i, label, s, worlds, sims = next(pending)
        except StopIteration:
            return False
        if i not in games:
            games[i] = dz.game_from_snapshot(int_keys(chosen[i]["snapshot"]))
        seat = int(chosen[i]["our_seat"])
        seed = (args.seed * 1000003 + i * 7919 + s * 104729 + worlds * 31 + sims) * 0x9E3779B97F4A7C15
        active.append([make_searcher(games[i], seat, policy, worlds, sims, seed), i, label, s])
        return True

    while len(active) < args.concurrent and launch():
        pass
    while active:
        obs_list, mask_list, owners = [], [], []
        for k, (searcher, *_rest) in enumerate(active):
            if searcher.done():
                continue
            obs, masks = searcher.collect_leaves()
            if obs.shape[0]:
                obs_list.append(obs)
                mask_list.append(masks)
                owners.append((k, obs.shape[0]))
        if obs_list:
            with torch.no_grad():
                logits, values = policy.evaluate(
                    torch.as_tensor(np.concatenate(obs_list), dtype=torch.float32, device=policy.device),
                    torch.as_tensor(np.concatenate(mask_list), dtype=torch.bool, device=policy.device),
                )
            logits = logits.detach().cpu().numpy().astype(np.float32, copy=False)
            values = values.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
            off = 0
            for k, n in owners:
                active[k][0].provide_evaluations(values[off:off + n], logits[off:off + n])
                off += n
            evals += off
        still = []
        for entry in active:
            searcher, i, label, s = entry
            if searcher.done():
                results[i][(label, s)] = {"best": int(searcher.best_action()),
                                          "stats": searcher.root_stats()}
                done_jobs += 1
            else:
                still.append(entry)
        active[:] = still
        while len(active) < args.concurrent and launch():
            pass
        if time.time() - last > 60:
            last = time.time()
            rate = evals / (last - start)
            print(f"{done_jobs}/{len(jobs)} searches {evals} evals {rate:.0f}/s "
                  f"eta {(len(jobs) - done_jobs) / max(done_jobs, 1) * (last - start) / 60:.0f} min", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for i, row in enumerate(chosen):
            meta = {k: row[k] for k in ("game_id", "outcome", "turn", "question", "legal",
                                        "our_owned", "opp_owned", "our_draw", "our_discard")}
            meta["runs"] = {f"{label}#{s}": v for (label, s), v in results[i].items()}
            fh.write(json.dumps(meta) + "\n")
    print(f"DONE {len(chosen)} positions {done_jobs} searches {evals} evals {time.time() - start:.0f}s", flush=True)
    summarize(out)


def summarize(path: Path) -> None:
    rows = [json.loads(l) for l in open(path)]
    table = defaultdict(lambda: defaultdict(list))
    for r in rows:
        b = bucket_of(int(r["our_owned"]))
        ref = r["runs"].get("ref32x256#0")
        if not ref:
            continue
        q = {int(a): float(v) for a, n, v in ref["stats"] if n > 0}
        ref_best = int(ref["best"])
        picks_128 = [r["runs"][k]["best"] for k in r["runs"] if k.startswith("1x128#")]
        pairs = list(itertools.combinations(picks_128, 2))
        if pairs:
            table[b]["flip"].append(sum(a != c for a, c in pairs) / len(pairs))
        for label, *_ in ARMS:
            for k, v in r["runs"].items():
                if k.split("#")[0] != label:
                    continue
                table[b][f"agree_{label}"].append(float(v["best"] == ref_best))
                if v["best"] in q and ref_best in q:
                    table[b][f"regret_{label}"].append(q[ref_best] - q[v["best"]])
        table[b]["n"].append(1)
    labels = [a[0] for a in ARMS]
    print("\nbucket   n    flip@1x128  " + "  ".join(f"agree/regret {l:>6}" for l in labels))
    order = [bucket_of(lo) for lo, _ in BUCKETS]
    for b in order:
        t = table.get(b)
        if not t:
            continue
        cells = []
        for l in labels:
            ag = np.mean(t[f"agree_{l}"]) if t[f"agree_{l}"] else float("nan")
            rg = np.mean(t[f"regret_{l}"]) if t[f"regret_{l}"] else float("nan")
            cells.append(f"{ag:6.1%} / {rg:.4f}    ")
        print(f"{b:7} {len(t['n']):4}  {np.mean(t['flip']):9.1%}   " + "  ".join(cells))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--summarize":
        summarize(Path(sys.argv[2]))
    else:
        main()
