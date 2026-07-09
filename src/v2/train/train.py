from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    import dominion_v2_py as dz
except ModuleNotFoundError as exc:  # pragma: no cover - gives a clearer CLI error
    raise SystemExit("dominion_v2_py not found; run with PYTHONPATH=build") from exc

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.v2.train.config import TrainConfig, add_config_args, load_config, save_config
    from src.v2.train.model import DominionNet, count_parameters, masked_policy_loss
    from src.v2.train.replay import ReplayBuffer
    from src.v2.train.selfplay import SelfPlayStats, run_self_play_generation
else:
    from .config import TrainConfig, add_config_args, load_config, save_config
    from .model import DominionNet, count_parameters, masked_policy_loss
    from .replay import ReplayBuffer
    from .selfplay import SelfPlayStats, run_self_play_generation


def select_device(name: str) -> torch.device:
    requested = name.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("mps requested but unavailable")
    return torch.device(requested)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def build_objects(config: TrainConfig, device: torch.device):
    model = DominionNet(dz.OBS_SIZE, dz.ACTION_SPACE_SIZE, config.model.hidden_sizes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )
    replay = ReplayBuffer(config.replay.capacity, dz.OBS_SIZE, dz.ACTION_SPACE_SIZE, config.seed ^ 0xA11CE)
    return model, optimizer, replay


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    batch = replay.sample(batch_size)
    obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
    policy_target = torch.as_tensor(batch.policy, dtype=torch.float32, device=device)
    value_target = torch.as_tensor(batch.value, dtype=torch.float32, device=device)
    legal_mask = torch.as_tensor(batch.legal_mask, dtype=torch.bool, device=device)

    logits, value = model(obs)
    policy_loss, entropy = masked_policy_loss(logits, legal_mask, policy_target)
    value_loss = F.mse_loss(value, value_target)
    loss = policy_loss + value_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
    }


def checkpoint_payload(
    config: TrainConfig,
    generation: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
) -> dict[str, Any]:
    return {
        "generation": generation,
        "config": config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "replay": replay.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


def save_checkpoint(
    config: TrainConfig,
    generation: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
) -> Path:
    out_dir = Path(config.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gen_{generation:04d}.pt"
    torch.save(checkpoint_payload(config, generation, model, optimizer, replay), path)
    save_config(config, out_dir / "config.json")
    return path


def load_full_checkpoint(path: str | Path, map_location: torch.device | str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(path: str | Path, device: torch.device):
    payload = load_full_checkpoint(path, device)
    cfg_dict = payload["config"]
    cfg = load_config(None)
    from src.v2.train.config import _merge_dataclass  # local to keep private helper out of public API

    _merge_dataclass(cfg, cfg_dict)
    model, optimizer, replay = build_objects(cfg, device)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    replay.load_state_dict(payload["replay"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    return cfg, int(payload["generation"]), model, optimizer, replay


def append_metrics(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    fieldnames = [
        "generation",
        "games",
        "positions",
        "policy_loss",
        "value_loss",
        "entropy",
        "games_per_hour",
        "leaves_per_sec",
        "nn_evals_per_sec",
        "inference_pct",
        "plumbing_pct",
        "wall_time",
    ]
    with out.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_training(config: TrainConfig, resume: str | None = None, profile: bool = False) -> dict[str, Any]:
    requested = config
    device = select_device(config.device)
    seed_everything(config.seed, deterministic=device.type == "cpu")
    if resume is not None:
        config, start_generation, model, optimizer, replay = load_checkpoint(resume, device)
        config.generations = requested.generations
        config.checkpoint_dir = requested.checkpoint_dir
        config.metrics_csv = requested.metrics_csv
        config.device = requested.device
        if config.device == "auto":
            config.device = device.type
    else:
        model, optimizer, replay = build_objects(config, device)
        start_generation = 0

    metrics: list[dict[str, Any]] = []
    generations = 1 if profile else max(0, config.generations - start_generation)
    print(
        json.dumps(
            {
                "device": device.type,
                "parameters": count_parameters(model),
                "obs_size": dz.OBS_SIZE,
                "action_size": dz.ACTION_SPACE_SIZE,
            },
            sort_keys=True,
        )
    )

    for generation in range(start_generation + 1, start_generation + generations + 1):
        gen_seed = config.seed + (generation * 0x9E37)
        gen_start = time.perf_counter()
        sp_stats = run_self_play_generation(model, replay, config.selfplay, gen_seed, device)

        losses = {"policy_loss": float("nan"), "value_loss": float("nan"), "entropy": float("nan")}
        steps = config.optim.train_steps_per_generation
        if len(replay) > 0 and steps > 0:
            accum = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
            for _ in range(steps):
                step_losses = train_step(model, optimizer, replay, config.optim.batch_size, device)
                for key in accum:
                    accum[key] += step_losses[key]
            losses = {key: value / steps for key, value in accum.items()}

        path = save_checkpoint(config, generation, model, optimizer, replay)
        row = {
            "generation": generation,
            "games": sp_stats.games,
            "positions": sp_stats.positions,
            "policy_loss": losses["policy_loss"],
            "value_loss": losses["value_loss"],
            "entropy": losses["entropy"],
            "games_per_hour": sp_stats.games_per_hour,
            "leaves_per_sec": sp_stats.leaves_per_sec,
            "nn_evals_per_sec": sp_stats.nn_evals_per_sec,
            "inference_pct": sp_stats.inference_pct,
            "plumbing_pct": sp_stats.plumbing_pct,
            "wall_time": time.perf_counter() - gen_start,
            "checkpoint": str(path),
        }
        append_metrics(config.metrics_csv, row)
        metrics.append(row)
        print(json.dumps(row, sort_keys=True))

    if profile and metrics:
        row = metrics[-1]
        print(
            "profile "
            f"games_per_hour={row['games_per_hour']:.2f} "
            f"leaves_per_sec={row['leaves_per_sec']:.2f} "
            f"nn_evals_per_sec={row['nn_evals_per_sec']:.2f} "
            f"inference_pct={row['inference_pct']:.2f} "
            f"plumbing_pct={row['plumbing_pct']:.2f}"
        )
    return {"config": config.to_dict(), "metrics": metrics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.device is not None:
        config.device = args.device
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
        config.metrics_csv = str(Path(args.checkpoint_dir) / "metrics.csv")
    run_training(config, resume=args.resume, profile=args.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
