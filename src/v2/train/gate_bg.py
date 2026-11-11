"""Background third-party gate match: one checkpoint against the gate's
scripted opponent, score written to a JSON file.

The trainer starts this when a gate begins (gating.start_gate_third_party_jobs)
so the match overlaps the head-to-head instead of following it. No
multiprocessing: resume.sh reaps "multiprocessing.spawn" processes.

Usage: python -m src.v2.train.gate_bg JOB.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    tag = f"gate_bg gen {job['generation']} {job['label']}"
    kwargs = job["kwargs"]
    print(f"{tag}: {job['checkpoint']} vs {kwargs['opponent']}, {kwargs['games']} games, "
          f"{kwargs['sims']} sims", flush=True)
    started = time.time()

    import torch

    torch.set_num_threads(1)
    from src.v2.train.evaluate import evaluate_model, load_model
    from src.v2.train.gating import write_third_party_score
    from src.v2.train.train import select_device

    device = select_device(job["device_name"])
    model, _ = load_model(job["checkpoint"], device)
    stats = evaluate_model(model, device=device, **kwargs)
    score = (stats.wins + 0.5 * stats.ties) / stats.games if stats.games > 0 else 0.0
    wall_s = round(time.time() - started, 1)
    write_third_party_score(
        job["result"], score, kwargs,
        gate_generation=job["generation"], label=job["label"], checkpoint=job["checkpoint"],
        wins=stats.wins, losses=stats.losses, ties=stats.ties, games=stats.games, wall_s=wall_s,
    )
    print(f"{tag}: {stats.wins}W-{stats.losses}L-{stats.ties}T score {score:.4f}, {wall_s:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
