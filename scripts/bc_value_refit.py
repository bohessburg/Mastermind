#!/usr/bin/env python3
"""Refit a BC checkpoint's value head on its final, frozen trunk.

The fixed BC recipe selects on validation policy loss, and by then the value
half has overfit (R40B: best validation value 0.329 at 12.5k steps, 0.347 at
24k). An older value head cannot be spliced in because the shared trunk kept
moving, so this freezes the trunk and policy head and retrains only
``value_head`` (value MSE, as in BC), early-stopping on validation value loss.
Two starts are tried: the existing head, and a re-initialized head. The best
validation value wins; the original head is kept if neither beats it.

usage: bc_value_refit.py <in.pt> <out.pt> [--steps N] [--eval-every N] [--patience N]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.v2.train.config import load_config
from src.v2.train.gating import load_best_checkpoint
from src.v2.train.human_data import load_human_tuples


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {message}", flush=True)


def val_value_loss(model, obs: np.ndarray, value: np.ndarray, device, batch: int = 4096) -> float:
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(obs), batch):
            o = torch.as_tensor(obs[start:start + batch], dtype=torch.float32, device=device)
            v = torch.as_tensor(value[start:start + batch], dtype=torch.float32, device=device)
            total += float(F.mse_loss(model(o)[1].float(), v, reduction="sum"))
    return total / len(obs)


def head_state(model) -> dict[str, torch.Tensor]:
    return {k: t.detach().clone() for k, t in model.value_head.state_dict().items()}


def reinit(module: torch.nn.Module) -> None:
    for layer in module.modules():
        if layer is not module and hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def fit(model, train, val, device, args, label: str) -> tuple[float, dict[str, torch.Tensor], int]:
    params = list(model.value_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    rng = np.random.default_rng(args.seed)
    best = val_value_loss(model, val.obs, val.value, device)
    best_state, best_step, stale = head_state(model), 0, 0
    log(f"{label}: start val value {best:.5f}")
    order, cursor = rng.permutation(len(train.obs)), 0
    for step in range(1, args.steps + 1):
        if cursor + args.batch > len(order):
            order, cursor = rng.permutation(len(train.obs)), 0
        idx = np.sort(order[cursor:cursor + args.batch])
        cursor += args.batch
        lr = args.lr * (0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        o = torch.as_tensor(train.obs[idx], dtype=torch.float32, device=device)
        v = torch.as_tensor(train.value[idx], dtype=torch.float32, device=device)
        loss = F.mse_loss(model(o)[1].float(), v)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.eval_every == 0:
            score = val_value_loss(model, val.obs, val.value, device)
            if score < best:
                best, best_state, best_step, stale = score, head_state(model), step, 0
            else:
                stale += 1
            log(f"{label}: step {step} train {float(loss):.5f} val value {score:.5f} best {best:.5f}@{best_step} stale {stale}")
            if stale >= args.patience:
                break
    return best, best_state, best_step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-tuples", default="exports/tuples_dgames_level1_bc/train")
    parser.add_argument("--val-tuples", default="exports/tuples_all_40plus/val")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("configs/run_c32.json")
    model, _ = load_best_checkpoint(cfg, device, args.input)
    model.eval()  # trunk features fixed (no dropout); only the value head trains
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.value_head.parameters():
        p.requires_grad_(True)
    log("loading tuples")
    train = load_human_tuples(args.train_tuples, value_scheme="margin")
    val = load_human_tuples(args.val_tuples, value_scheme="margin")
    log(f"train {len(train.obs)} rows, val {len(val.obs)} rows")

    original = head_state(model)
    base = val_value_loss(model, val.obs, val.value, device)
    results = {"original": (base, original, 0)}
    results["continue"] = fit(model, train, val, device, args, "continue")
    model.value_head.load_state_dict(original)
    reinit(model.value_head)
    results["reinit"] = fit(model, train, val, device, args, "reinit")
    choice = min(results, key=lambda k: results[k][0])
    log("summary " + json.dumps({k: {"val_value": r[0], "step": r[2]} for k, r in results.items()} | {"chosen": choice}))

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    for key, tensor in results[choice][1].items():
        payload["model"][f"value_head.{key}"] = tensor.cpu()
    payload["value_refit"] = {"source": str(args.input), "chosen": choice,
                              "val_value": results[choice][0], "original_val_value": base}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    # Sanity: reload and re-score.
    check, _ = load_best_checkpoint(cfg, device, args.output)
    log(f"saved {args.output}: val value {val_value_loss(check, val.obs, val.value, device):.5f} (original {base:.5f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
