#!/usr/bin/env python3
"""Measure how much MCTS search lands past the first hidden-information reveal.

Plays games with the given checkpoint through the native SelfPlayRunner (c32
self-play settings) with DZ_REVEAL_STATS=1; the runner prints a REVEAL_STATS
line to stderr when destroyed. One run per determinize mode.

  DZ_REVEAL_STATS=1 python scripts/reveal_stats.py CHECKPOINT [--games 16] [--batches 1500] [--device cuda]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

import dominion_v2_py as dz
from src.v2.train.config import load_config
from src.v2.train.gating import load_best_checkpoint
from src.v2.train.selfplay import make_runner_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="configs/run_c32.json")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--batches", type=int, default=1500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modes", default="off,per_decision")
    args = parser.parse_args()
    if os.environ.get("DZ_REVEAL_STATS") != "1":
        print("set DZ_REVEAL_STATS=1", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model, _ = load_best_checkpoint(cfg, device, args.checkpoint)
    model.eval()
    for mode in args.modes.split(","):
        sp = load_config(args.config).selfplay
        sp.determinize = mode
        sp.tree_reuse = mode == "off"
        slots = [{"game_index": i, "seat0_model_id": 0, "seat1_model_id": 0} for i in range(args.games)]
        runner = dz.SelfPlayRunner(make_runner_config(sp, 4242, slot_manifest=slots))
        started, batches, rows = time.perf_counter(), 0, 0
        with torch.inference_mode():
            while batches < args.batches and runner.games_completed() < args.games:
                obs, masks = runner.collect_leaves(0)[:2]
                n = obs.shape[0]
                if not n:
                    continue
                logits, values = model(torch.as_tensor(obs, device=device))[:2]
                logits = logits.float().masked_fill(~torch.as_tensor(masks, device=device), -1e9)
                policy = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
                runner.provide_evaluations(values.float().reshape(n).cpu().numpy(), policy)
                batches += 1
                rows += n
        print(f"mode={mode}: {batches} batches, {rows} evals, {runner.games_completed()} games done, "
              f"{time.perf_counter() - started:.0f}s", flush=True)
        sys.stderr.flush()
        del runner  # prints REVEAL_STATS
        sys.stderr.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
