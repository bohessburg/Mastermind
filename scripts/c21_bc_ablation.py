#!/usr/bin/env python3
"""Run the c21 CardTokenNet behavior-cloning size ablation.

Training is intentionally delegated to ``run_human_pretrain`` and evaluation
to the offline-fit human helpers.  This script only schedules fixed-size
pretrain chunks so it can evaluate the game-disjoint validation corpus and
early-stop without creating another optimizer loop.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import dominion_v2_py as dz
from src.v2.train.config import load_config, validate_aux_margin_config
from src.v2.train.human_data import HumanTupleDataset, load_human_tuples
from src.v2.train.model import build_model, count_parameters
from src.v2.train.observation import obs_size_for_config
from src.v2.train.offline_fit import evaluate_human, human_arrays
from src.v2.train.train import checkpoint_payload, run_human_pretrain, seed_everything, select_device


CONFIG_PATH = Path("configs/run_c21_draft.json")
TRAIN_TUPLES = Path("exports/tuples_all/train")
VAL_TUPLES = Path("exports/tuples_all/val")
OUTPUT_DIR = Path("checkpoints/c21_ablation")
REPORT_PATH = Path("bench/c21_bc_ablation.json")
ARMS = {
    "a": {"d_model": 192, "n_layers": 4, "n_heads": 6},
    "b": {"d_model": 320, "n_layers": 5, "n_heads": 8},
}


@torch.no_grad()
def policy_accuracy(
    model: torch.nn.Module,
    dataset: HumanTupleDataset,
    batch_size: int,
    device: torch.device,
) -> float:
    """Return masked argmax agreement; loss computation stays in offline_fit."""

    model.eval()
    matches = 0
    total = 0
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        obs = torch.as_tensor(dataset.obs[start:stop], dtype=torch.float32, device=device)
        legal = torch.as_tensor(dataset.legal[start:stop], dtype=torch.bool, device=device)
        actions = torch.as_tensor(dataset.action[start:stop], dtype=torch.long, device=device)
        logits, _ = model(obs)
        matches += int((logits.masked_fill(~legal, -1.0e9).argmax(dim=-1) == actions).sum().item())
        total += stop - start
    return matches / total


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _prepare_config(args: argparse.Namespace, arm: dict[str, int]):
    config = load_config(CONFIG_PATH)
    config.device = args.device
    config.checkpoint_dir = str(args.output_dir)
    config.metrics_csv = str(args.output_dir / "metrics.csv")
    config.imitation.human_tuples = str(args.train_tuples)
    config.imitation.pretrain_steps = 50 if args.smoke else args.steps
    config.model.d_model = arm["d_model"]
    config.model.n_layers = arm["n_layers"]
    config.model.n_heads = arm["n_heads"]
    # The draft already carries these settings. Keep the assertions local so a
    # future draft change fails loudly instead of silently changing this ablation.
    if config.model.arch != "card_transformer" or config.model.obs_version != 3 or config.selfplay.obs_version != 3:
        raise ValueError("c21 ablation requires the native card transformer with observation version 3")
    validate_aux_margin_config(config)
    return config


def _run_arm(
    label: str,
    arm: dict[str, int],
    args: argparse.Namespace,
    train_dataset: HumanTupleDataset,
    val_dataset: HumanTupleDataset,
    device: torch.device,
) -> dict[str, Any]:
    config = _prepare_config(args, arm)
    if train_dataset.obs_width != obs_size_for_config(config) or val_dataset.obs_width != obs_size_for_config(config):
        raise ValueError("tuple observation width does not match c21 observation version 3")
    if train_dataset.action_width != int(dz.ACTION_SPACE_SIZE) or val_dataset.action_width != int(dz.ACTION_SPACE_SIZE):
        raise ValueError("tuple action width does not match the native action space")

    seed_everything(int(config.seed), deterministic=True)
    model = build_model(config.model, obs_size_for_config(config), int(dz.ACTION_SPACE_SIZE)).to(device)
    optimizer_class = torch.optim.AdamW if config.optim.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    train_arrays = human_arrays(train_dataset)
    val_arrays = human_arrays(val_dataset)
    # A normal ablation reports the corpus-wide final train loss.  Smoke mode
    # intentionally keeps its diagnostic loss compact so the promised fast
    # CPU sanity check does not spend most of its time re-scoring 62K rows.
    train_metric_rows = min(len(train_arrays), 2_048) if args.smoke else len(train_arrays)
    train_indices = np.arange(train_metric_rows, dtype=np.intp)
    val_indices = np.arange(len(val_arrays), dtype=np.intp)
    eval_batch_size = int(config.imitation.pretrain_batch_size)
    train_batch_size = min(eval_batch_size, 32) if args.smoke else eval_batch_size
    total_steps = int(config.imitation.pretrain_steps)
    train_batches = train_dataset.minibatches(train_batch_size, int(config.seed) ^ 0x4855_4D41_4E)
    parameter_count = count_parameters(model)
    print(
        f"c21 BC arm={label} d_model={arm['d_model']} layers={arm['n_layers']} heads={arm['n_heads']} "
        f"params={parameter_count} train={len(train_dataset)} val={len(val_dataset)} steps={total_steps} "
        f"batch={train_batch_size} device={device}",
        flush=True,
    )

    started = time.monotonic()
    steps_run = 0
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_evaluations = 0
    last_train_loss = float("nan")
    eval_every = int(args.eval_every)
    while steps_run < total_steps:
        # The existing pretrain runner provides a 50-step flush-safe heartbeat.
        # Chunk only at a validation boundary; optimizer updates stay wholly in
        # the shared c21 behavior-cloning path.
        until_evaluation = total_steps - steps_run if args.smoke else min(eval_every - (steps_run % eval_every), total_steps - steps_run)
        chunk = min(50, until_evaluation)
        run_human_pretrain(
            model,
            optimizer,
            train_batches,
            steps=chunk,
            device=device,
            pretrain_lr=config.imitation.pretrain_lr,
        )
        steps_run += chunk
        if not args.smoke and steps_run % eval_every != 0 and steps_run != total_steps:
            continue

        val_metrics = evaluate_human(model, val_arrays, val_indices, eval_batch_size, device)
        improved = val_metrics.total < best_val_loss
        if improved:
            best_val_loss = val_metrics.total
            best_state = _cpu_state_dict(model)
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        print(
            f"c21 BC eval arm={label} step={steps_run}/{total_steps} val_loss={val_metrics.total:.6f} "
            f"best_val={best_val_loss:.6f} stale={stale_evaluations}",
            flush=True,
        )
        if not args.smoke and stale_evaluations >= args.patience:
            print(f"c21 BC early-stop arm={label} step={steps_run} patience={args.patience}", flush=True)
            break

    if best_state is None:
        raise AssertionError("no validation result was collected")
    model.load_state_dict(best_state)
    final_train = evaluate_human(model, train_arrays, train_indices, eval_batch_size, device)
    last_train_loss = final_train.total
    final_val = evaluate_human(model, val_arrays, val_indices, eval_batch_size, device)
    val_accuracy = policy_accuracy(model, val_dataset, eval_batch_size, device)
    checkpoint = checkpoint_payload(config, 0, model, optimizer)
    checkpoint["encoder_generation"] = 2
    checkpoint_path = args.output_dir / f"bc_{label}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    wall_seconds = time.monotonic() - started
    print(
        f"c21 BC complete arm={label} checkpoint={checkpoint_path} steps={steps_run} "
        f"train_loss={final_train.total:.6f} best_val={best_val_loss:.6f} val_accuracy={val_accuracy:.6f}",
        flush=True,
    )
    return {
        "label": label,
        "model": copy.deepcopy(config.model.__dict__),
        "final_train_loss": final_train.total,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val.total,
        "val_policy_accuracy": val_accuracy,
        "steps_run": steps_run,
        "wall_seconds": wall_seconds,
        "parameter_count": parameter_count,
        "train_batch_size": train_batch_size,
        "train_loss_rows": train_metric_rows,
        "checkpoint": str(checkpoint_path),
        "last_pre_restore_train_loss": last_train_loss,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true", help="run 50 steps per arm without early stopping")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--train-tuples", type=Path, default=TRAIN_TUPLES)
    parser.add_argument("--val-tuples", type=Path, default=VAL_TUPLES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0 or args.eval_every <= 0 or args.patience <= 0:
        raise SystemExit("--steps, --eval-every, and --patience must be positive")
    device = select_device(args.device)
    print(f"c21 BC loading train={args.train_tuples} val={args.val_tuples}", flush=True)
    train_dataset = load_human_tuples(args.train_tuples, value_scheme="margin")
    val_dataset = load_human_tuples(args.val_tuples, value_scheme="margin")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        _run_arm(label, arm, args, train_dataset, val_dataset, device)
        for label, arm in ARMS.items()
    ]
    report = {
        "config": str(CONFIG_PATH),
        "encoder_generation": 2,
        "observation_version": 3,
        "train_tuples": len(train_dataset),
        "val_tuples": len(val_dataset),
        "smoke": bool(args.smoke),
        "arms": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"c21 BC report={args.report}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
