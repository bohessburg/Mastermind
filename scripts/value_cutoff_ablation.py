#!/usr/bin/env python3
"""Which human games should train the value head when policy is 45+ only?

Arms (all share the policy data; only the value data differs):

  v45  value from rows whose mover is rated 45+
  v40  value from rows whose mover is rated 40+
  vall value from every row, unrated included
  v40nr   v40, but no value rows from games that ended in a resignation
  v40nrp  v40nr, and no policy rows from resigned games either

The resignation arms also get test slices that leave resigned games out.

Every optimizer step draws 512 policy rows from the 45+ training pool (same
seed, so every arm sees the same policy rows in the same order) plus 512
value-only rows from the arm's value pool.  Value MSE covers all 1024 rows;
the policy cross entropy is the usual rating-weighted mean over the 45+ rows
(value-only rows carry a negligible 1e-6 weight because the trainer rejects
zero weights).

The corpus is split game-disjoint 90/5/5: validation early-stops each arm on
45+ total loss, and the held-out test split is reported, chiefly value MSE on
50+ movers.

usage: value_cutoff_ablation.py [--tuples DIR] [--arms v45,v40,vall] [--smoke]
"""

from __future__ import annotations

import argparse
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
if str(REPOSITORY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import c21_bc_ablation as bc
import dominion_v2_py as dz
from src.v2.train.human_data import (
    DEFAULT_RATINGS_SIDECAR,
    HumanBatch,
    _decisive_outcome_margins,
    _manifest_rating_games,
    _rating_entry_level_and_deviation,
    policy_weight_for_rating,
    recompute_value_targets,
)
from src.v2.train.model import build_model, count_parameters
from src.v2.train.observation import obs_size_for_config
from src.v2.train.offline_fit import HumanArrays, evaluate_human
from src.v2.train.train import checkpoint_payload, run_human_pretrain, seed_everything, select_device

POLICY_FLOOR = 45.0
VALUE_ONLY_WEIGHT = 1.0e-6
HALF_BATCH = 512
SPLIT_SEED = 20261115
ARM_FLOORS: dict[str, float | None] = {"v45": 45.0, "v40": 40.0, "vall": None, "v40nr": 40.0, "v40nrp": 40.0}
# Arms that drop resigned games: (from the value pool, from the policy pool).
ARM_DROP_RESIGNED: dict[str, tuple[bool, bool]] = {"v40nr": (True, False), "v40nrp": (True, True)}


def load_corpus(tuple_dir: Path, smoke: bool) -> dict[str, Any]:
    manifest = json.loads((tuple_dir / "tuple_manifest.json").read_text())
    shards = manifest["shards"][:3] if smoke else manifest["shards"]
    rows = sum(int(shard["tuples"]) for shard in shards)
    obs_width = int(manifest["obs_width"])
    action_width = int(manifest["action_width"])
    print(f"loading {len(shards)} shards, {rows} rows, obs={obs_width} legal={action_width}", flush=True)
    # Preallocate: the stock loader concatenates lists and peaks near 2x.
    data = {
        "obs": np.empty((rows, obs_width), dtype=np.float32),
        "legal": np.empty((rows, action_width), dtype=np.bool_),
        "action": np.empty((rows,), dtype=np.int64),
        "margin": np.empty((rows,), dtype=np.int16),
        "seat_index": np.empty((rows,), dtype=np.int16),
        "game_index": np.empty((rows,), dtype=np.int32),
    }
    cursor = 0
    started = time.monotonic()
    for number, shard in enumerate(shards, start=1):
        with np.load(tuple_dir / shard["path"], allow_pickle=False) as arrays:
            count = int(arrays["action"].shape[0])
            for name, destination in data.items():
                destination[cursor : cursor + count] = arrays[name]
        cursor += count
        if number % 25 == 0 or number == len(shards):
            print(f"  loaded {number}/{len(shards)} shards ({cursor} rows, {time.monotonic() - started:.0f}s)", flush=True)
    if cursor != rows:
        raise ValueError(f"shard rows {cursor} != manifest rows {rows}")
    margins = _decisive_outcome_margins(manifest, data["margin"], data["game_index"], data["seat_index"])
    data["margin"] = margins.astype(np.int16, copy=False)
    data["value"] = np.ascontiguousarray(
        recompute_value_targets(margins, scheme="margin", scale=20.0), dtype=np.float32
    )
    data["manifest"] = manifest
    return data


def mover_ratings(manifest: dict[str, Any], game_index: np.ndarray, seat_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row mover rating (NaN if unrated) and its 45+ policy weight."""
    sidecar = json.loads(Path(DEFAULT_RATINGS_SIDECAR).read_text())
    games = _manifest_rating_games(manifest)
    level = np.full(game_index.shape, np.nan, dtype=np.float32)
    weight = np.full(game_index.shape, VALUE_ONLY_WEIGHT, dtype=np.float32)
    keys = game_index.astype(np.int64) * 8 + seat_index.astype(np.int64)
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    key_level = np.full(unique_keys.shape, np.nan, dtype=np.float32)
    key_weight = np.full(unique_keys.shape, VALUE_ONLY_WEIGHT, dtype=np.float32)
    for position, key in enumerate(unique_keys):
        game_id, player_ids = games[int(key // 8)]
        seat = int(key % 8)
        if player_ids is None or seat >= len(player_ids):
            continue
        entry = sidecar.get(game_id, {})
        entry = entry.get(str(player_ids[seat])) if isinstance(entry, dict) else None
        value, deviation = _rating_entry_level_and_deviation(entry)
        if value is None:
            continue
        key_level[position] = value
        if value >= POLICY_FLOOR:
            key_weight[position] = policy_weight_for_rating(value, deviation)
    level[:] = key_level[inverse]
    weight[:] = key_weight[inverse]
    return level, weight


class MixedBatches:
    """512 policy rows from the 45+ pool plus 512 value-only rows per step."""

    def __init__(self, data: dict[str, Any], weight: np.ndarray, policy_pool: np.ndarray, value_pool: np.ndarray, seed: int):
        self.data = data
        self.weight = weight
        self.policy_pool = policy_pool
        self.value_pool = value_pool
        # Separate streams: the policy rows are identical across arms.
        self.policy_rng = np.random.default_rng(seed)
        self.value_rng = np.random.default_rng(seed + 1)

    def __iter__(self) -> "MixedBatches":
        return self

    def __next__(self) -> HumanBatch:
        policy_rows = self.policy_pool[self.policy_rng.integers(0, len(self.policy_pool), HALF_BATCH)]
        value_rows = self.value_pool[self.value_rng.integers(0, len(self.value_pool), HALF_BATCH)]
        rows = np.concatenate([policy_rows, value_rows])
        weight = self.weight[rows].copy()
        weight[HALF_BATCH:] = VALUE_ONLY_WEIGHT
        return HumanBatch(
            obs=self.data["obs"][rows],
            action=self.data["action"][rows],
            legal=self.data["legal"][rows],
            value=self.data["value"][rows],
            margin=self.data["margin"][rows],
            policy_weight=weight,
        )


@torch.no_grad()
def accuracy(model: torch.nn.Module, data: dict[str, Any], rows: np.ndarray, device: torch.device) -> float:
    model.eval()
    matches = 0
    for start in range(0, len(rows), 1024):
        batch = rows[start : start + 1024]
        obs = torch.as_tensor(data["obs"][batch], device=device)
        legal = torch.as_tensor(data["legal"][batch], device=device)
        logits, _ = model(obs)
        predicted = logits.masked_fill(~legal, -1.0e9).argmax(dim=-1).cpu().numpy()
        matches += int((predicted == data["action"][batch]).sum())
    return matches / max(1, len(rows))


def slice_metrics(model, arrays: HumanArrays, data, slices: dict[str, np.ndarray], device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, rows in slices.items():
        if len(rows) == 0:
            continue
        metrics = evaluate_human(model, arrays, rows, 1024, device)
        result[name] = {"rows": int(len(rows)), "value_mse": metrics.value, "policy": metrics.policy}
        if name in ("45plus", "50plus", "45plus_nr", "50plus_nr"):
            result[name]["accuracy"] = accuracy(model, data, rows, device)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tuples", type=Path, default=Path("exports/tuples_dgames_level1"))
    parser.add_argument("--arms", default="v45,v40,vall")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/value_cutoff"))
    parser.add_argument("--report", type=Path, default=Path("bench/value_cutoff/report.json"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    if any(arm not in ARM_FLOORS for arm in arms):
        raise SystemExit(f"arms must be among {sorted(ARM_FLOORS)}")

    device = select_device(args.device)
    data = load_corpus(args.tuples, args.smoke)
    level, weight = mover_ratings(data["manifest"], data["game_index"], data["seat_index"])

    games = np.unique(data["game_index"])
    order = np.random.default_rng(SPLIT_SEED).permutation(games)
    cut_val, cut_test = int(len(order) * 0.90), int(len(order) * 0.95)
    split = np.zeros(int(games.max()) + 1, dtype=np.int8)
    split[order[cut_val:cut_test]] = 1
    split[order[cut_test:]] = 2
    row_split = split[data["game_index"]]
    rated = ~np.isnan(level)
    resigned_game = np.array([any(game["outcome"].get("resigned") or ()) for game in data["manifest"]["games"]], dtype=np.bool_)
    played_out = ~resigned_game[data["game_index"]]
    above = lambda floor: rated & (np.nan_to_num(level, nan=-1.0) >= floor)

    def rows(mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(mask).astype(np.intp)

    train = row_split == 0
    policy_pool = rows(train & above(POLICY_FLOOR))
    played_out_policy_pool = rows(train & above(POLICY_FLOOR) & played_out)
    eval_slices = {
        part: {
            "all": rows(row_split == code),
            "40plus": rows((row_split == code) & above(40.0)),
            "45plus": rows((row_split == code) & above(45.0)),
            "50plus": rows((row_split == code) & above(50.0)),
            "45plus_nr": rows((row_split == code) & above(45.0) & played_out),
            "50plus_nr": rows((row_split == code) & above(50.0) & played_out),
        }
        for part, code in (("val", 1), ("test", 2))
    }
    print(
        f"games train/val/test {cut_val}/{cut_test - cut_val}/{len(order) - cut_test}; "
        f"policy pool {len(policy_pool)} rows; "
        + "; ".join(f"{part} " + " ".join(f"{k}={len(v)}" for k, v in s.items()) for part, s in eval_slices.items()),
        flush=True,
    )
    arrays = HumanArrays(
        obs=data["obs"], action=data["action"], value=data["value"],
        legal_mask=data["legal"], margin=data["margin"], policy_weight=weight,
    )

    config_args = argparse.Namespace(
        device=args.device, output_dir=args.output_dir, train_tuples=args.tuples, skill_weighting=True,
        smoke=args.smoke, steps=args.steps, model_config=Path("configs/run_c31.json"), dropout=None, weight_decay=None,
    )
    report: dict[str, Any] = {"tuples": str(args.tuples), "policy_floor": POLICY_FLOOR, "split_seed": SPLIT_SEED, "arms": {}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        floor = ARM_FLOORS[arm]
        drop_value, drop_policy = ARM_DROP_RESIGNED.get(arm, (False, False))
        value_mask = train & (above(floor) if floor is not None else np.ones_like(train))
        value_pool = rows(value_mask & played_out if drop_value else value_mask)
        arm_policy_pool = played_out_policy_pool if drop_policy else policy_pool
        config = bc._prepare_config(config_args, bc.ARMS["a"])
        if obs_size_for_config(config) != data["obs"].shape[1]:
            raise ValueError("tuple observation width does not match the model config")
        seed_everything(int(config.seed), deterministic=True)
        model = build_model(config.model, obs_size_for_config(config), int(dz.ACTION_SPACE_SIZE)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
        batches = MixedBatches(data, weight, arm_policy_pool, value_pool, int(config.seed))
        total_steps = 50 if args.smoke else int(args.steps)
        eval_every = 25 if args.smoke else int(args.eval_every)
        print(
            f"arm={arm} value_floor={floor} value_pool={len(value_pool)} policy_pool={len(arm_policy_pool)} params={count_parameters(model)} "
            f"steps<={total_steps} device={device}",
            flush=True,
        )
        started = time.monotonic()
        steps_run, best, best_state, stale, history = 0, float("inf"), None, 0, []
        while steps_run < total_steps:
            chunk = min(50, eval_every - steps_run % eval_every, total_steps - steps_run)
            run_human_pretrain(model, optimizer, batches, steps=chunk, device=device, pretrain_lr=config.imitation.pretrain_lr)
            steps_run += chunk
            if steps_run % eval_every and steps_run != total_steps:
                continue
            val45 = evaluate_human(model, arrays, eval_slices["val"]["45plus"], 1024, device)
            val50 = evaluate_human(model, arrays, eval_slices["val"]["50plus"] if len(eval_slices["val"]["50plus"]) else eval_slices["val"]["45plus"], 1024, device)
            score = val45.policy + val45.value
            history.append({"step": steps_run, "val45_policy": val45.policy, "val45_value": val45.value, "val50_value": val50.value})
            if score < best:
                best, best_state, stale = score, bc._cpu_state_dict(model), 0
            else:
                stale += 1
            print(
                f"eval arm={arm} step={steps_run} val45_policy={val45.policy:.5f} val45_value={val45.value:.5f} "
                f"val50_value={val50.value:.5f} best={best:.5f} stale={stale} elapsed={time.monotonic() - started:.0f}s",
                flush=True,
            )
            if not args.smoke and stale >= args.patience:
                print(f"early-stop arm={arm} step={steps_run}", flush=True)
                break
        model.load_state_dict(best_state)
        test = slice_metrics(model, arrays, data, eval_slices["test"], device)
        checkpoint = checkpoint_payload(config, 0, model, optimizer)
        checkpoint["encoder_generation"] = 2
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.output_dir / f"bc_{arm}.pt")
        report["arms"][arm] = {
            "value_floor": floor, "value_pool_rows": int(len(value_pool)), "policy_pool_rows": int(len(arm_policy_pool)), "steps_run": steps_run,
            "best_val": best, "best_step": min(history, key=lambda h: h["val45_policy"] + h["val45_value"])["step"],
            "wall_seconds": time.monotonic() - started, "test": test, "history": history,
        }
        print(f"test arm={arm} " + " ".join(f"{k}_value={v['value_mse']:.5f}" for k, v in test.items())
              + " " + " ".join(f"{k}_acc={v['accuracy']:.4f}" for k, v in test.items() if "accuracy" in v), flush=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
