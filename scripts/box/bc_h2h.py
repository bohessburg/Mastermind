"""Level-1 BC seeds against bc_a with search: 800-game paired gate, c32 gate settings, 8 workers."""

import json
import sys
import time
from pathlib import Path

import torch

from src.v2.train.config import load_config
from src.v2.train.gating import load_best_checkpoint, run_paired_gate_match

OUT = Path("bench/bc_level1/h2h_vs_bca.jsonl")


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {message}", flush=True)


def main() -> None:
    cfg = load_config("configs/run_c32.json")
    cfg.gate_workers = 8
    device = torch.device("cuda")
    done = {json.loads(line)["arm"] for line in OUT.read_text().splitlines()} if OUT.exists() else set()
    bca, _ = load_best_checkpoint(cfg, device, "checkpoints/campaign25/bc_a.pt")
    for arm in (sys.argv[1:] or ["A", "B"]):
        if arm in done:
            continue
        candidate, _ = load_best_checkpoint(cfg, device, f"checkpoints/bc_level1_{arm}/bc_f100.pt")
        s = run_paired_gate_match(candidate, bca, cfg, 600, device, log=log)
        row = {"arm": arm, "score": s["a_score"], "se": s["paired_se"], "wins": s["wins"],
               "losses": s["losses"], "ties": s["ties"], "wall_s": s["wall_s"]}
        log("H2H " + json.dumps(row))
        with OUT.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
