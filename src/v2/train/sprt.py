"""Sequential probability ratio test for checkpoint go/no-go decisions.

Plays seat-swapped pairs on the canonical eval boards (see honest_eval) in
batches and stops as soon as the evidence is decisive, instead of a fixed
200-game duel whose +-6.5-point interval could not resolve most c22-c24
milestone questions.

The test is the pentanomial-style normal GSPRT used by chess engine testing
(fishtest): each seat-swapped pair is one observation x in {0, .25, .5, .75,
1}; under H_i the mean pair score is s_i = logistic(elo_i). With pair mean m
and variance v over N pairs,

    LLR = N * (s1 - s0) * (2m - s0 - s1) / (2v)

Accept H1 (A is at least elo1 stronger) at LLR >= log((1-beta)/alpha);
accept H0 at LLR <= log(beta/(1-alpha)).

Example (is c25 g10 better than the champion?)::

    PYTHONPATH=build .venv/bin/python -m src.v2.train.sprt \
        --a checkpoints/campaign25/gen_0010.pt \
        --b checkpoints/remote/campaign15/gen_0045.pt --legacy-shim \
        --elo0 0 --elo1 35 --batch-games 40 --max-games 2000 \
        --workers 24 --device cuda --out bench/sprt_c25g10_vs_champ.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.v2.train.honest_eval import (
    CANONICAL_EVAL_SEED,
    HonestEvalConfig,
    run_honest_eval,
    score_statistics,
)


def expected_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-float(elo) / 400.0))


def pair_scores(games: list[dict[str, Any]]) -> list[float]:
    """Mean score of each complete seat-swapped pair (errors drop the pair)."""
    by_index = {int(game["index"]): game["result"] for game in games}
    value = {"W": 1.0, "T": 0.5, "L": 0.0}
    scores: list[float] = []
    for index in sorted(by_index):
        if index % 2 != 0 or index + 1 not in by_index:
            continue
        first, second = by_index[index], by_index[index + 1]
        if first in value and second in value:
            scores.append(0.5 * (value[first] + value[second]))
    return scores


def gsprt_llr(scores: list[float], elo0: float, elo1: float) -> float:
    if len(scores) < 2:
        return 0.0
    arr = np.asarray(scores, dtype=np.float64)
    mean = float(arr.mean())
    var = float(arr.var(ddof=0))
    # A pair sample with zero variance (all identical) carries no usable
    # spread estimate; floor it at the variance of a single split pair.
    var = max(var, 1.0 / (4.0 * arr.size))
    s0, s1 = expected_score(elo0), expected_score(elo1)
    return arr.size * (s1 - s0) * (2.0 * mean - s0 - s1) / (2.0 * var)


def sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def run_sprt(
    base: HonestEvalConfig,
    *,
    elo0: float,
    elo1: float,
    alpha: float,
    beta: float,
    batch_games: int,
    max_games: int,
    out: Path,
) -> dict[str, Any]:
    lower, upper = sprt_bounds(alpha, beta)
    games: list[dict[str, Any]] = []
    verdict = "inconclusive"
    llr = 0.0
    started = time.perf_counter()
    scratch = out.with_suffix(".batch.json")
    next_index = int(base.first_index)
    while len(games) < max_games:
        count = min(batch_games, max_games - len(games))
        count -= count % 2
        if count <= 0:
            break
        batch = replace(base, games=count, first_index=next_index)
        payload = run_honest_eval(batch, scratch)
        games.extend(payload["games"])
        next_index += count
        scores = pair_scores(games)
        llr = gsprt_llr(scores, elo0, elo1)
        mean = 100.0 * float(np.mean(scores)) if scores else float("nan")
        print(
            f"sprt: games={len(games)} pairs={len(scores)} score={mean:.1f}% "
            f"llr={llr:.3f} bounds=[{lower:.3f}, {upper:.3f}] "
            f"elapsed={time.perf_counter() - started:.0f}s",
            flush=True,
        )
        if llr >= upper:
            verdict = "H1"
            break
        if llr <= lower:
            verdict = "H0"
            break
    scratch.unlink(missing_ok=True)

    class _Result:  # minimal adapter for score_statistics
        def __init__(self, game: dict[str, Any]) -> None:
            self.index = int(game["index"])
            self.result = str(game["result"])

    result = {
        "config": {
            "a": base.a,
            "b": base.b,
            "opponent": base.opponent,
            "seed": base.seed,
            "elo0": elo0,
            "elo1": elo1,
            "alpha": alpha,
            "beta": beta,
            "batch_games": batch_games,
            "max_games": max_games,
            "sims_a": base.sims_a,
            "sims_b": base.sims_b,
            "determinizations_a": base.determinizations_a,
            "determinizations_b": base.determinizations_b,
        },
        "verdict": verdict,
        "verdict_meaning": {
            "H1": f"A is stronger by >= ~{elo1:+.0f} Elo (accept)",
            "H0": f"A is not stronger than {elo0:+.0f} Elo (reject)",
            "inconclusive": "max_games reached before either bound",
        }[verdict],
        "llr": llr,
        "bounds": [lower, upper],
        "statistics": score_statistics([_Result(game) for game in games], paired=True),
        "wall_time_seconds": time.perf_counter() - started,
        "games": games,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"sprt verdict: {verdict} ({result['verdict_meaning']}) -> {out}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SPRT between two checkpoints on canonical paired boards")
    parser.add_argument("--a", required=True)
    counterpart = parser.add_mutually_exclusive_group(required=True)
    counterpart.add_argument("--b")
    counterpart.add_argument("--opponent", choices=("engine3", "engine2", "bigmoney", "thinner"))
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=35.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--batch-games", type=int, default=40)
    parser.add_argument("--max-games", type=int, default=2000)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--determinizations", type=int, default=2)
    parser.add_argument("--sims-b", type=int)
    parser.add_argument("--determinizations-b", type=int)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=CANONICAL_EVAL_SEED)
    parser.add_argument("--first-index", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--legacy-shim", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.batch_games < 2 or args.batch_games % 2:
        parser.error("--batch-games must be an even number >= 2")
    if args.first_index % 2:
        parser.error("--first-index must be even so pairs stay aligned")
    if not 0.0 < args.alpha < 0.5 or not 0.0 < args.beta < 0.5:
        parser.error("--alpha and --beta must be in (0, 0.5)")
    if args.elo1 <= args.elo0:
        parser.error("--elo1 must exceed --elo0")
    base = HonestEvalConfig(
        a=str(args.a),
        b=str(args.b) if args.b is not None else None,
        opponent=args.opponent,
        games=int(args.batch_games),
        sims_a=int(args.sims),
        determinizations_a=int(args.determinizations),
        sims_b=int(args.sims if args.sims_b is None else args.sims_b),
        determinizations_b=int(args.determinizations if args.determinizations_b is None else args.determinizations_b),
        workers=int(args.workers),
        device=str(args.device),
        seed=int(args.seed),
        torch_threads=int(args.torch_threads),
        legacy_shim=bool(args.legacy_shim),
        seating="paired",
        first_index=int(args.first_index),
    )
    run_sprt(
        base,
        elo0=float(args.elo0),
        elo1=float(args.elo1),
        alpha=float(args.alpha),
        beta=float(args.beta),
        batch_games=int(args.batch_games),
        max_games=int(args.max_games),
        out=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
