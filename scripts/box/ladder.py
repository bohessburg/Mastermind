"""2000-game paired ladder of each train-steps arm's gen 6 candidate against c31 gen 27.

Same boards for every arm (fixed seed), c32 gate settings, 8 workers.
Resumable: arms already in ladder_results.jsonl are skipped.
"""

import json
import time
from pathlib import Path

import torch

from src.v2.train.config import load_config
from src.v2.train.gating import load_best_checkpoint, run_paired_gate_match

ARMS = {
    "s80 (c32)": "checkpoints/campaign32/gen_0006.pt",
    "s160": "checkpoints/campaign32_s160/gen_0006.pt",
    "s240": "checkpoints/campaign32_s240/gen_0006.pt",
    "s160h": "checkpoints/campaign32_s160h/gen_0006.pt",
}
BEST = "checkpoints/remote/campaign31/gen_0027.pt"
OUT = Path("checkpoints/ladder_c32_arms.jsonl")
SEED_GENERATION = 500  # fixes the boards; distinct from any training gate


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {message}", flush=True)


def main() -> None:
    cfg = load_config("configs/run_c32.json")
    cfg.gate_games = 2000
    cfg.gate_workers = 8
    device = torch.device("cuda")
    done = {json.loads(line)["arm"] for line in OUT.read_text().splitlines()} if OUT.exists() else set()
    best, _ = load_best_checkpoint(cfg, device, BEST)
    for arm, path in ARMS.items():
        if arm in done:
            continue
        candidate, generation = load_best_checkpoint(cfg, device, path)
        log(f"ladder {arm}: {path} (generation {generation}) vs gen 27, {cfg.gate_games} games")
        s = run_paired_gate_match(candidate, best, cfg, SEED_GENERATION, device, log=log)
        row = {"arm": arm, "path": path, "score": s["a_score"], "se": s["paired_se"],
               "wins": s["wins"], "losses": s["losses"], "ties": s["ties"], "wall_s": s["wall_s"]}
        log("LADDER " + json.dumps(row))
        with OUT.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
