"""One checkpoint, several search setups, against a native scripted bot.

Every arm plays the same game indices (same kingdoms and shuffles), with the
network seat alternating by index, so arms can be compared game by game.

  PYTHONPATH=build:. python scripts/worlds_scripted.py --a ckpt.pt --opponent engine3 \
      --arms 1x128,2x128,1x256 --games 1000 --out bench/worlds3
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
import dominion_v2_py as dz  # noqa: E402
from src.v2.train.config import load_config  # noqa: E402
from src.v2.train.diagnostics import match_selfplay_config, pick_device, route_leaf_evaluations  # noqa: E402
from src.v2.train.evaluate import load_model  # noqa: E402
from src.v2.train.selfplay import make_runner_config  # noqa: E402


def play(model, config, seed, indices, opponent, worlds, device, tag):
    cfg = copy.deepcopy(config)
    cfg.worlds = int(worlds)
    cfg.max_tree_nodes = 2 * int(cfg.sims_per_move) + 64
    manifest = [{"game_index": i, "scripted_kind": opponent, "nn_player": i % 2} for i in indices]
    native = make_runner_config(cfg, seed, slot_manifest=manifest)
    runner = dz.SelfPlayRunner(native)
    records, evals, last = [], 0, time.time()
    while len(records) < len(indices):
        obs, masks = runner.collect_leaves(cfg.max_batch)
        if int(obs.shape[0]) > 0:
            logits, values = route_leaf_evaluations([model, model], obs, masks, None, device,
                                                    same_model_fast_path=True)
            runner.provide_evaluations(values, logits)
            evals += int(obs.shape[0])
        finished = runner.finished_games()
        if not finished and int(obs.shape[0]) == 0:
            raise RuntimeError("runner stalled")
        records.extend(finished)
        if time.time() - last > 60:
            last = time.time()
            print(f"[{tag}] {len(records)}/{len(indices)} games {evals} evals", flush=True)
    return records, evals


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--opponent", required=True, choices=("engine3", "bigmoney"))
    p.add_argument("--arms", required=True, help="comma list of WORLDSxSIMS_PER_WORLD, e.g. 1x128,2x128")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--config", default="configs/run_c33.json")
    p.add_argument("--determinize", default="per_decision")
    p.add_argument("--seed", type=int, default=20261119)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = pick_device(args.device)
    base = load_config(args.config).selfplay
    model, _ = load_model(Path(args.a), device)
    model.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    indices = list(range(args.first, args.first + args.games))
    for arm in args.arms.split(","):
        worlds, per = (int(x) for x in arm.split("x"))
        sims = worlds * per
        tag = f"{args.opponent}_w{worlds}x{per}_b{args.first}"
        cfg = match_selfplay_config(base, sims=sims, determinize=args.determinize, greedy=True)
        print(f"[{tag}] worlds={worlds} sims={sims} games={len(indices)} first={args.first} "
              f"adaptive_per_legal={getattr(cfg, 'adaptive_sims_per_legal', None)}", flush=True)
        start = time.time()
        records, evals = [], 0
        with torch.no_grad():
            for c in range(0, len(indices), args.chunk):
                r, e = play(model, cfg, args.seed, indices[c:c + args.chunk], args.opponent, worlds, device, tag)
                records.extend(r)
                evals += e
        per_game = {}
        for rec in records:
            w = rec.get("winner")
            nn = int(rec["scripted_nn_player"])
            per_game[int(rec["game_index"])] = 0.5 if w is None else (1.0 if int(w) == nn else 0.0)
        n = len(per_game)
        score = sum(per_game.values()) / n
        se = math.sqrt(max(score * (1 - score), 1e-9) / n)
        wins = sum(1 for v in per_game.values() if v == 1.0)
        ties = sum(1 for v in per_game.values() if v == 0.5)
        payload = {"tag": tag, "opponent": args.opponent, "worlds": worlds, "sims": sims, "games": n,
                   "score": score, "se": se, "wins": wins, "losses": n - wins - ties, "ties": ties,
                   "nn_evals": evals, "wall_s": round(time.time() - start, 1), "per_game": per_game}
        (out / f"{tag}.json").write_text(json.dumps(payload))
        print(f"[{tag}] DONE {wins}W-{n - wins - ties}L-{ties}T score={score:.3f}+-{se:.3f} "
              f"evals/game={evals / n:.0f} {payload['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
