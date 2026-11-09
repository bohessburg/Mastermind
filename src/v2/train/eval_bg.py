"""Background generation eval: runs the main eval and its sentinels
concurrently, one single-runner process each, and appends rows to evals.csv.

Eval wall time is bounded by the serial 400-sim search chain inside each game,
not throughput, so splitting games across processes does not help; overlapping
eval with the next generations' selfplay removes it from the critical path.

Usage: python -m src.v2.train.eval_bg JOB.json
"""
from __future__ import annotations

import concurrent.futures
import csv
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

FIELDS = ["generation", "kind", "opponent", "games", "wins", "losses", "ties", "truncated",
          "end_province", "end_piles", "end_trunc", "win_pct_excl_ties", "wall_s", "finished_unix"]


def _run(kwargs: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    from src.v2.train.evaluate import evaluate_checkpoint

    return evaluate_checkpoint(**kwargs).__dict__


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    out = Path(job["evals_csv"])
    started = time.time()
    print(f"eval_bg gen {job['generation']}: {len(job['runs'])} runs started", flush=True)
    # fork, not spawn: resume.sh reaps "multiprocessing.spawn" orphans on
    # every restart, which killed spawn-started eval workers. The parent
    # never imports torch/CUDA, so forking is safe.
    ctx = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(job["runs"]), mp_context=ctx) as pool:
        futures = {pool.submit(_run, run["kwargs"]): run for run in job["runs"]}
        for future in concurrent.futures.as_completed(futures):
            run = futures[future]
            stats = future.result()
            decided = stats["wins"] + stats["losses"]
            row = {
                "generation": job["generation"],
                "kind": run["kind"],
                "opponent": stats["opponent"],
                **{k: stats[k] for k in ("games", "wins", "losses", "ties", "truncated",
                                         "end_province", "end_piles", "end_trunc")},
                "win_pct_excl_ties": (100.0 * stats["wins"] / decided) if decided else 0.0,
                "wall_s": round(time.time() - started, 1),
                "finished_unix": round(time.time(), 1),
            }
            new = not out.exists()
            with out.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow(row)
            print(f"eval_bg gen {job['generation']} {run['kind']}: {stats['opponent']} "
                  f"{stats['wins']}W-{stats['losses']}L-{stats['ties']}T, {row['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
