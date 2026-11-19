"""Gate-style duel between two checkpoints with per-side value ablation.

Same conditions as the training gate (seat-swapped, clairvoyant, Dirichlet
off, temperature 0, random kingdoms) but with configurable sims and an
optional zero-value wrapper per side: a zero-value net's search just follows
its policy prior, so BC vs BC-zero-value measures what value-guided search
adds over the raw policy.

Usage: python scripts/c26_duel.py --a A.pt --b B.pt [--b-zero-value]
       [--sims 128] [--games 200] [--tag NAME]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.v2.train.config import load_config  # noqa: E402
from src.v2.train.gating import run_seat_swapped_match  # noqa: E402
from src.v2.train.train import build_objects, load_full_checkpoint  # noqa: E402


def load(path: str, config, device, zero_value: bool) -> torch.nn.Module:
    # Architecture comes from the checkpoint itself (BC variants differ in size).
    from src.v2.train.evaluate import load_model

    model, _ = load_model(Path(path), device)
    model.eval()
    if zero_value:
        model.register_forward_hook(lambda _m, _i, out: (out[0], torch.zeros_like(out[1])) + tuple(out[2:]))
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--a-zero-value", action="store_true")
    parser.add_argument("--b-zero-value", action="store_true")
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--config", default="configs/run_c26.json")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    config = load_config(args.config)
    a = load(args.a, config, device, args.a_zero_value)
    b = load(args.b, config, device, args.b_zero_value)
    match_config = copy.deepcopy(config.selfplay)
    match_config.sims_per_move = int(args.sims)
    match_config.deep_slice_fraction = 0.0
    match_config.deep_slice_sims = 0
    match_config.dirichlet_frac = 0.0
    match_config.temp_moves = 0
    match_config.kingdom_mode = "random"
    started = time.time()
    match = run_seat_swapped_match(a, b, match_config, games=args.games, seed=args.seed, device=device)
    s = match.stats
    result = {"tag": args.tag, "a": args.a, "b": args.b, "a_zero_value": args.a_zero_value,
              "b_zero_value": args.b_zero_value, "sims": args.sims, "games": args.games,
              "wins": s.wins, "losses": s.losses, "ties": s.ties, "a_win_pct": s.win_pct,
              "wall_s": round(time.time() - started)}
    out = Path("bench/c26_duels")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[{args.tag}] A {s.wins}W-{s.losses}L-{s.ties}T a_win_pct={s.win_pct:.3f} {result['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
