#!/usr/bin/env python3
"""Human-data learning curve for the behavior-cloned init (bc_a recipe).

Two questions, answered by retraining the bc_a recipe on subsets of the
human corpus and scoring every arm on the same held-out games:

1. Does more of the same data still help?  Arms ``f25``, ``f50``, ``f100``
   train on a nested random 25% / 50% / 100% of the training GAMES.
2. Does more data from level 50+ players help?  Arms ``x0`` and ``x50`` keep
   every row played by a level 40-50 player and keep 0% / 50% of the
   (game, seat) units played by a level 50+ player.  ``f100`` is the 100%
   point of that curve too.

Training is ``scripts/c21_bc_ablation.py``'s arm runner unchanged (same
optimizer loop, early stopping and checkpoint format), so ``f100`` is a
re-run of bc_a.  Per arm the report adds held-out move accuracy split by the
rating band of the player who made the move.

  python scripts/bc_learning_curve.py                # all five arms
  python scripts/bc_learning_curve.py --arms f25,f50 # some of them

Playing strength is measured afterwards from the saved checkpoints
(checkpoints/bc_curve/bc_<arm>.pt); see docs/training-log.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.v2.train.human_data import (  # noqa: E402
    RATING_BAND_EXPERT,
    RATING_BAND_RAMP,
    HumanTupleDataset,
    load_human_tuples,
)
from src.v2.train.train import select_device  # noqa: E402

_spec = importlib.util.spec_from_file_location("c21_bc_ablation", REPOSITORY_ROOT / "scripts" / "c21_bc_ablation.py")
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)

ARRAY_FIELDS = (
    "obs", "action", "legal", "value", "margin", "winner", "seat_index",
    "game_index", "ply_index", "turn_number", "policy_weight", "rating_band",
)
ARMS = ("f25", "f50", "f100", "x0", "x50")
SUBSET_SEED = 20261110


def take(dataset: HumanTupleDataset, mask: np.ndarray) -> HumanTupleDataset:
    return dataclasses.replace(dataset, **{name: getattr(dataset, name)[mask] for name in ARRAY_FIELDS})


def game_fraction_mask(dataset: HumanTupleDataset, fraction: float) -> np.ndarray:
    """Rows of the first ``fraction`` of a fixed shuffle of the games (nested across fractions)."""
    games = np.unique(dataset.game_index)
    order = np.random.default_rng(SUBSET_SEED).permutation(games)
    keep = order[: max(1, int(round(fraction * len(order))))]
    return np.isin(dataset.game_index, keep)


def expert_fraction_mask(dataset: HumanTupleDataset, fraction: float) -> np.ndarray:
    """All non-expert rows plus the rows of ``fraction`` of the level-50+ (game, seat) units."""
    expert = dataset.rating_band == RATING_BAND_EXPERT
    units = dataset.game_index.astype(np.int64) * 16 + dataset.seat_index.astype(np.int64)
    expert_units = np.unique(units[expert])
    order = np.random.default_rng(SUBSET_SEED ^ 0x50).permutation(expert_units)
    keep = order[: int(round(fraction * len(order)))]
    return ~expert | np.isin(units, keep)


def arm_mask(dataset: HumanTupleDataset, arm: str) -> np.ndarray | None:
    if arm == "f100":
        return None
    if arm.startswith("f"):
        return game_fraction_mask(dataset, int(arm[1:]) / 100.0)
    if arm.startswith("x"):
        return expert_fraction_mask(dataset, int(arm[1:]) / 100.0)
    raise ValueError(f"unknown arm {arm!r}")


def composition(dataset: HumanTupleDataset) -> dict[str, float]:
    expert = dataset.rating_band == RATING_BAND_EXPERT
    units = dataset.game_index.astype(np.int64) * 16 + dataset.seat_index.astype(np.int64)
    weight = dataset.policy_weight.astype(np.float64)
    return {
        "tuples": int(len(dataset)),
        "games": int(np.unique(dataset.game_index).size),
        "expert_tuples": int(expert.sum()),
        "expert_seat_games": int(np.unique(units[expert]).size),
        "policy_effective_tuples": float(weight.sum()),
        "expert_share_of_policy_weight": float(weight[expert].sum() / max(weight.sum(), 1.0e-9)),
    }


@torch.no_grad()
def move_accuracy_by_band(model: torch.nn.Module, dataset: HumanTupleDataset, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = np.zeros(len(dataset), dtype=bool)
    for start in range(0, len(dataset), 2048):
        stop = min(start + 2048, len(dataset))
        obs = torch.as_tensor(dataset.obs[start:stop], dtype=torch.float32, device=device)
        legal = torch.as_tensor(dataset.legal[start:stop], dtype=torch.bool, device=device)
        logits, _ = model(obs)
        top = logits.masked_fill(~legal, -1.0e9).argmax(dim=-1).cpu().numpy()
        correct[start:stop] = top == dataset.action[start:stop]
    expert = dataset.rating_band == RATING_BAND_EXPERT
    ramp = dataset.rating_band == RATING_BAND_RAMP
    weight = dataset.policy_weight.astype(np.float64)
    return {
        "all": float(correct.mean()),
        "level_50_plus": float(correct[expert].mean()),
        "level_50_plus_rows": int(expert.sum()),
        "level_40_to_50": float(correct[ramp].mean()),
        "level_40_to_50_rows": int(ramp.sum()),
        "policy_weighted": float((correct * weight).sum() / weight.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--train-tuples", type=Path, default=Path("exports/tuples_all_40plus/train"))
    parser.add_argument("--val-tuples", type=Path, default=Path("exports/tuples_all_40plus/val"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/run_c25.json"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/bc_curve"))
    parser.add_argument("--report-dir", type=Path, default=Path("bench/bc_curve"))
    parser.add_argument("--smoke", action="store_true", help="50 steps per arm, to test the plumbing")
    args = parser.parse_args()
    arms = [arm for arm in args.arms.split(",") if arm]
    for arm in arms:
        if arm not in ARMS:
            parser.error(f"unknown arm {arm!r}; choose from {', '.join(ARMS)}")

    device = select_device(args.device)
    started = time.time()
    print(f"bc_curve loading train={args.train_tuples} val={args.val_tuples} device={device}", flush=True)
    train = load_human_tuples(args.train_tuples, value_scheme="margin", skill_weighting=True)
    val = load_human_tuples(args.val_tuples, value_scheme="margin", skill_weighting=True)
    print(f"bc_curve loaded {len(train)} train / {len(val)} val rows in {time.time() - started:.0f}s", flush=True)

    # The arm runner reads these from an argparse namespace.
    run_args = argparse.Namespace(
        device=str(device), smoke=args.smoke, skill_weighting=True, steps=args.steps,
        eval_every=args.eval_every, patience=args.patience, train_tuples=args.train_tuples,
        val_tuples=args.val_tuples, output_dir=args.output_dir, model_config=args.model_config,
        dropout=None, select_on="total", weight_decay=None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        mask = arm_mask(train, arm)
        subset = train if mask is None else take(train, mask)
        mix = composition(subset)
        print(f"bc_curve arm={arm} {json.dumps(mix)}", flush=True)
        result = bc._run_arm(arm, bc.ARMS["a"], run_args, subset, val, device)
        checkpoint = torch.load(result["checkpoint"], map_location=device, weights_only=False)
        config = bc._prepare_config(run_args, bc.ARMS["a"])
        model = bc.build_model(config.model, bc.obs_size_for_config(config), int(bc.dz.ACTION_SPACE_SIZE)).to(device)
        model.load_state_dict(checkpoint["model"])
        result["train_composition"] = mix
        result["val_move_accuracy"] = move_accuracy_by_band(model, val, device)
        result["arm"] = arm
        (args.report_dir / f"{arm}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"bc_curve arm={arm} DONE steps={result['steps_run']} val_loss={result['best_val_loss']:.4f} "
              f"acc={json.dumps(result['val_move_accuracy'])} {time.time() - started:.0f}s", flush=True)
        del subset, model, checkpoint
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
