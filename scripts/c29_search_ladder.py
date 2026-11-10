"""c29 search ladder: one clean, paired matchup per invocation.

Answers "does search add strength over the raw policy, and does the value
head help search?" with the measurement confounds of the training gate
removed (see src/v2/train/diagnostics.py): greedy play, each side's tree
scored only by its own network, fresh trees, and every board played once in
each seat orientation.

Sides are a checkpoint plus a simulation budget. ``--a-sims raw`` plays the
policy head's argmax with no search. ``--a-zero-value`` zeroes the value
output so search follows the prior alone.

  # does 128-sim search beat the raw policy of the same net?
  python scripts/c29_search_ladder.py --tag bca_128_vs_raw \
      --a checkpoints/campaign25/bc_a.pt --a-sims 128 --b-sims raw
  # does the value head help that search?
  python scripts/c29_search_ladder.py --tag bca_128_vs_zerov \
      --a checkpoints/campaign25/bc_a.pt --a-sims 128 --b-sims 128 --b-zero-value
  # corrected gate for a candidate
  python scripts/c29_search_ladder.py --tag c28g1_vs_bca \
      --a checkpoints/campaign28/gen_0001.pt --b checkpoints/campaign25/bc_a.pt

``--legacy-gate`` reproduces the old gate conditions on the same paired
boards, to measure the measurement. It is shorthand for the three legacy
conditions, which can also be switched on one at a time:
  --cross-route   leaf scored by the model of the player deciding at the leaf
  --tree-reuse    one carried-over tree shared by both sides (needs --cross-route)
  --opening-temp  training temperature schedule (T=1 openings) instead of greedy
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.v2.train.config import load_config  # noqa: E402
from src.v2.train.diagnostics import (  # noqa: E402
    RAW_POLICY_SIMS,
    match_selfplay_config,
    pick_device,
    run_paired_match,
    zero_value_copy,
)
from src.v2.train.evaluate import load_model  # noqa: E402


def parse_sims(text: str) -> int:
    if text.lower() == "raw":
        return RAW_POLICY_SIMS
    sims = int(text)
    if sims < RAW_POLICY_SIMS:
        raise argparse.ArgumentTypeError("sims must be 'raw' or at least 2")
    return sims


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--a", required=True, help="checkpoint for side A")
    parser.add_argument("--b", default=None, help="checkpoint for side B (default: same as A)")
    parser.add_argument("--a-sims", type=parse_sims, default=128)
    parser.add_argument("--b-sims", type=parse_sims, default=128)
    parser.add_argument("--a-zero-value", action="store_true")
    parser.add_argument("--b-zero-value", action="store_true")
    parser.add_argument("--pairs", type=int, default=200, help="boards; each is played twice")
    parser.add_argument("--chunk", type=int, default=64, help="concurrent games")
    parser.add_argument("--determinize", choices=("off", "per_decision", "per_turn"), default=None,
                        help="search information model (default: the config's)")
    parser.add_argument("--legacy-gate", action="store_true")
    parser.add_argument("--cross-route", action="store_true")
    parser.add_argument("--tree-reuse", action="store_true")
    parser.add_argument("--opening-temp", action="store_true")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--config", default="configs/run_c28.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="bench/c29_ladder")
    args = parser.parse_args()

    device = pick_device(args.device)
    config = load_config(args.config)
    model_a, _ = load_model(Path(args.a), device)
    b_path = args.b or args.a
    # One object for both seats lets identical sides share a full inference batch.
    model_b = model_a if args.b is None else load_model(Path(b_path), device)[0]
    if args.a_zero_value:
        model_a = zero_value_copy(model_a)
    if args.b_zero_value:
        model_b = zero_value_copy(model_b)

    cross_route = bool(args.legacy_gate or args.cross_route or args.tree_reuse)
    tree_reuse = bool(args.legacy_gate or args.tree_reuse)
    opening_temp = bool(args.legacy_gate or args.opening_temp)
    if tree_reuse and args.a_sims != args.b_sims:
        parser.error("a reused tree is shared between the sides; budgets must match")
    match_config = match_selfplay_config(
        config.selfplay,
        sims=max(args.a_sims, args.b_sims),
        determinize=args.determinize,
        greedy=not opening_temp,
        tree_reuse=tree_reuse,
    )
    print(
        f"[{args.tag}] device={device} pairs={args.pairs} chunk={args.chunk} "
        f"A={args.a} sims={args.a_sims} zero_v={args.a_zero_value} | "
        f"B={b_path} sims={args.b_sims} zero_v={args.b_zero_value} | "
        f"determinize={match_config.determinize} cross_route={cross_route} "
        f"tree_reuse={tree_reuse} opening_temp={opening_temp}",
        flush=True,
    )

    started = time.time()

    def heartbeat(match) -> None:
        s = match.summary()
        rate = s["games"] / max(match.wall_time, 1.0e-9) * 3600.0
        print(
            f"[{args.tag}] {s['pairs']}/{args.pairs} pairs  A {s['wins']}W-{s['losses']}L-{s['ties']}T "
            f"score={s['a_score']:.3f}±{s['paired_se']:.3f}  {rate:.0f} games/hr  "
            f"{match.nn_evals / max(match.wall_time, 1.0e-9):.0f} evals/s  {time.time() - started:.0f}s",
            flush=True,
        )

    last_tick = [started]

    def tick(games_done: int, evals: int) -> None:
        now = time.time()
        if now - last_tick[0] < 30.0:
            return
        last_tick[0] = now
        print(f"[{args.tag}] .. {games_done}/{2 * args.pairs} games  "
              f"{evals / max(now - started, 1.0e-9):.0f} evals/s  {now - started:.0f}s", flush=True)

    match = run_paired_match(
        model_a,
        model_b,
        match_config,
        pairs=args.pairs,
        seed=args.seed,
        device=device,
        sims_a=0 if tree_reuse else args.a_sims,
        sims_b=0 if tree_reuse else args.b_sims,
        independent_search=not cross_route,
        chunk=args.chunk,
        progress=heartbeat,
        tick=tick,
    )
    summary = match.summary()
    result = {
        "tag": args.tag,
        "a": args.a,
        "b": b_path,
        "a_sims": args.a_sims,
        "b_sims": args.b_sims,
        "a_zero_value": args.a_zero_value,
        "b_zero_value": args.b_zero_value,
        "determinize": match_config.determinize,
        "cross_route": cross_route,
        "tree_reuse": tree_reuse,
        "opening_temp": opening_temp,
        "seed": args.seed,
        "config": args.config,
        "device": str(device),
        **summary,
        "a_scores_seat0": match.a_as_seat0,
        "a_scores_seat1": match.a_as_seat1,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    lo, hi = summary["ci95"]
    print(
        f"[{args.tag}] DONE A {summary['wins']}W-{summary['losses']}L-{summary['ties']}T "
        f"score={summary['a_score']:.3f} ci95=[{lo:.3f},{hi:.3f}] elo={summary['elo']:+.0f} "
        f"seat0={summary['seat0_score']:.3f} non_greedy={summary['non_greedy_pct']:.1f}% "
        f"turns={summary['mean_turns']:.1f} trunc={summary['truncated']} {summary['wall_s']}s",
        flush=True,
    )
    for side in ("a", "b"):
        key = f"raw_policy_agreement_{side}"
        if key in summary and summary[key] is not None:
            print(f"[{args.tag}] raw-policy check side {side.upper()}: {100.0 * summary[key]:.2f}% of moves = prior argmax", flush=True)


if __name__ == "__main__":
    main()
