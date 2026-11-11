"""Does batched (multi-leaf) search agree with sequential search?

Plays the same greedy games with one leaf in flight (the reference) and with
each ``--leaves`` setting, then compares the root visit distributions on the
decisions both runs reached with an identical history. Games use one seed per
index, so a batched game follows the reference move for move until the two
searches first prefer different actions; every decision up to and including
that one is a matched position.

Run this before training on any change to how leaves are batched
(docs/training-log.md, 2026-11-10: a flat virtual-loss penalty spread visits
evenly over the top moves and went unnoticed for four campaigns).

  python scripts/c29_leaf_parity.py --tag bca_512 --sims 512 --leaves 8 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.v2.train.config import load_config  # noqa: E402
from src.v2.train.diagnostics import match_selfplay_config, pick_device, play_manifest_games  # noqa: E402
from src.v2.train.evaluate import load_model  # noqa: E402


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1.0e-12, None))).sum(axis=-1)


def normalised(targets: np.ndarray) -> np.ndarray:
    total = targets.sum(axis=-1, keepdims=True)
    return targets / np.clip(total, 1.0e-12, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ckpt", default="checkpoints/campaign25/bc_a.pt")
    parser.add_argument("--sims", type=int, default=512)
    parser.add_argument("--leaves", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--max-turns", type=int, default=0, help="cap game length (0 = the config's)")
    parser.add_argument("--determinize", choices=("off", "per_decision", "per_turn"), default="per_decision")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--config", default="configs/run_c28.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="bench/c29_ladder")
    args = parser.parse_args()

    device = pick_device(args.device)
    config = load_config(args.config)
    model, _ = load_model(Path(args.ckpt), device)
    base = match_selfplay_config(config.selfplay, sims=args.sims, determinize=args.determinize)
    base.max_recorded_moves = 4096
    # The cap is exactly the requested number: no per-sims limit, no tail boost.
    base.leaves_min_sims_per_leaf = 0
    base.tail_leaves_in_flight = 0
    if args.max_turns > 0:
        base.max_turns = args.max_turns
    started = time.time()

    def play(leaves: int) -> list[dict]:
        run_config = match_selfplay_config(base, sims=args.sims)
        run_config.leaves_in_flight = leaves
        last = [time.time()]

        def tick(done: int, evals: int) -> None:
            now = time.time()
            if now - last[0] >= 30.0:
                last[0] = now
                print(f"[{args.tag}] leaves={leaves} .. {done}/{args.games} games  {evals} evals  "
                      f"{now - started:.0f}s", flush=True)

        records, evals = play_manifest_games(
            (model, model), run_config, seed=args.seed, device=device,
            game_indices=range(args.games), independent_search=True, tick=tick,
        )
        print(f"[{args.tag}] leaves={leaves} done: {len(records)} games  {evals} evals  "
              f"{time.time() - started:.0f}s", flush=True)
        return records

    def rows(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs = np.asarray(record["observations"], dtype=np.float32)
        masks = np.asarray(record["legal_mask"], dtype=bool)
        targets = normalised(np.asarray(record["policy_targets"], dtype=np.float64))
        actions = np.asarray(record["sampled_actions"]).astype(int)
        n = min(len(obs), len(masks), len(targets), len(actions))
        return obs[:n], masks[:n], targets[:n], actions[:n]

    reference = [rows(record) for record in play(1)]

    def overall(games: list[tuple]) -> dict[str, float]:
        targets = np.concatenate([t[m.sum(axis=-1) >= 2] for _, m, t, _ in games])
        return {
            "decisions": int(len(targets)),
            "entropy": float(entropy(targets).mean()),
            "top_share": float(targets.max(axis=-1).mean()),
        }

    result: dict = {
        "tag": args.tag, "ckpt": args.ckpt, "sims": args.sims, "games": args.games,
        "determinize": args.determinize, "max_turns": int(base.max_turns), "seed": args.seed,
        "reference": overall(reference), "leaves": {},
    }
    print(f"[{args.tag}] sequential: {result['reference']}", flush=True)
    for leaves in args.leaves:
        batched = [rows(record) for record in play(leaves)]
        ref_p, bat_p, full_games = [], [], 0
        for (r_obs, r_mask, r_t, r_a), (b_obs, b_mask, b_t, b_a) in zip(reference, batched):
            n = min(len(r_a), len(b_a))
            diverged = False
            for i in range(n):
                if not np.array_equal(r_obs[i], b_obs[i]):
                    diverged = True
                    break
                if r_mask[i].sum() >= 2:
                    ref_p.append(r_t[i])
                    bat_p.append(b_t[i])
                if r_a[i] != b_a[i]:
                    diverged = True
                    break
            full_games += int(not diverged and len(r_a) == len(b_a))
        ref = np.asarray(ref_p)
        bat = np.asarray(bat_p)
        ref_top = ref.argmax(axis=-1)
        summary = {
            "matched_decisions": int(len(ref)),
            "identical_games": full_games,
            "same_top_move": float(np.mean(ref_top == bat.argmax(axis=-1))),
            "total_variation": float(0.5 * np.abs(ref - bat).sum(axis=-1).mean()),
            "sequential_entropy": float(entropy(ref).mean()),
            "batched_entropy": float(entropy(bat).mean()),
            "sequential_top_share": float(ref.max(axis=-1).mean()),
            "batched_top_share": float(bat.max(axis=-1).mean()),
            # Share of batched visits on the move sequential search preferred.
            "batched_share_on_sequential_top": float(bat[np.arange(len(bat)), ref_top].mean()),
            "all_decisions": overall(batched),
        }
        result["leaves"][str(leaves)] = summary
        print(f"[{args.tag}] leaves={leaves}: {json.dumps(summary)}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"leaf_parity_{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[{args.tag}] DONE {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
