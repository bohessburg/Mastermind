from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import warnings
from dataclasses import asdict
from math import cos, pi
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
    from src.v2.train.gating import (
        GateStats,
        archive_previous_best,
        gate_result,
        gating_enabled,
        initialize_best_checkpoint,
        league_checkpoint_paths,
        load_best_checkpoint,
        compact_selfplay_segments,
        plan_selfplay_segments,
        run_gate_match,
        save_best_checkpoint,
    )
    from src.v2.train.inference_server import InferenceServer, serialize_cpu_state_dict
    from src.v2.train.model import DominionNet, count_parameters, masked_policy_loss
    from src.v2.train.replay import ReplayBuffer, load_replay_state, save_replay_state
    from src.v2.train.selfplay import SelfPlayStats, run_routed_self_play_generation, run_self_play_generation
    from src.v2.train.workers import ParallelSelfPlayPool
else:
    from .config import TrainConfig, add_config_args, load_config, save_config
    from .gating import (
        GateStats,
        archive_previous_best,
        gate_result,
        gating_enabled,
        initialize_best_checkpoint,
        league_checkpoint_paths,
        load_best_checkpoint,
        compact_selfplay_segments,
        plan_selfplay_segments,
        run_gate_match,
        save_best_checkpoint,
    )
    from .inference_server import InferenceServer, serialize_cpu_state_dict
    from .model import DominionNet, count_parameters, masked_policy_loss
    from .replay import ReplayBuffer, load_replay_state, save_replay_state
    from .selfplay import SelfPlayStats, run_routed_self_play_generation, run_self_play_generation
    from .workers import ParallelSelfPlayPool


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
    best_generation: int | None = None,
) -> dict[str, Any]:
    """Return the small, inference-usable generation checkpoint payload."""
    payload = {
        "generation": generation,
        "config": config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        # RNG state is small generation metadata used for reproducible resume;
        # the potentially multi-GB replay data lives in replay_state.npz.
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if best_generation is not None:
        payload["best_generation"] = int(best_generation)
    return payload


def save_checkpoint(
    config: TrainConfig,
    generation: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    best_generation: int | None = None,
) -> Path:
    out_dir = Path(config.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gen_{generation:04d}.pt"
    torch.save(checkpoint_payload(config, generation, model, optimizer, best_generation), path)
    # Keep exactly one crash-safe replay snapshot rather than embedding it in
    # every generation checkpoint.
    save_replay_state(replay, out_dir / "replay_state.npz")
    save_config(config, out_dir / "config.json")
    return path


def load_full_checkpoint(path: str | Path, map_location: torch.device | str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_resume_path(resume: str | Path | None, checkpoint_dir: str | Path) -> str | None:
    if resume is None:
        return None
    if str(resume) != "latest":
        return str(resume)
    root = Path(checkpoint_dir)
    candidates = sorted(root.glob("gen_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"--resume latest found no gen_*.pt files in {root}")
    return str(candidates[-1])


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint_path = Path(path)
    payload = load_full_checkpoint(checkpoint_path, device)
    cfg_dict = payload["config"]
    cfg = load_config(None)
    from src.v2.train.config import _merge_dataclass  # local to keep private helper out of public API

    _merge_dataclass(cfg, cfg_dict)
    model, optimizer, replay = build_objects(cfg, device)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    replay_path = checkpoint_path.parent / "replay_state.npz"
    if replay_path.exists():
        load_replay_state(replay, replay_path)
    else:
        warnings.warn(
            f"replay state {replay_path} is missing; replay buffer was not restored and resume will use an empty buffer",
            RuntimeWarning,
            stacklevel=2,
        )
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
    if "numpy_rng_state" in payload:
        np.random.set_state(payload["numpy_rng_state"])
    if "python_rng_state" in payload:
        random.setstate(payload["python_rng_state"])
    return cfg, int(payload["generation"]), model, optimizer, replay


def learning_rate_for_generation(config: TrainConfig, generation: int) -> float:
    schedule = config.optim.lr_schedule.lower()
    if schedule == "constant":
        return config.optim.lr
    if schedule == "step":
        if config.optim.step_decay_every <= 0:
            return config.optim.lr
        steps = max(0, generation - 1) // config.optim.step_decay_every
        return max(config.optim.min_lr, config.optim.lr * (config.optim.step_decay_gamma ** steps))
    if schedule == "cosine":
        total = max(1, config.generations - 1)
        progress = min(1.0, max(0.0, (generation - 1) / total))
        span = config.optim.lr - config.optim.min_lr
        return config.optim.min_lr + (0.5 * span * (1.0 + cos(pi * progress)))
    raise ValueError(f"unknown lr_schedule: {config.optim.lr_schedule}")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


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
        "lr",
        "games_per_hour",
        "leaves_per_sec",
        "nn_evals_per_sec",
        "inference_pct",
        "plumbing_pct",
        "workers",
        "aggregate_games_per_hour",
        "server_evals_per_sec",
        "server_mean_batch_size",
        "server_batch_wait_p50_ms",
        "server_batch_wait_p99_ms",
        "wall_time",
        "eval_opponent",
        "eval_games",
        "eval_wins",
        "eval_losses",
        "eval_ties",
        "eval_truncated",
        "eval_win_pct_excl_ties",
        "eval_games_per_hour",
        "gate_result",
        "gate_win_pct",
        "best_generation",
        "league_games",
    ]
    with out.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _add_stats(total: SelfPlayStats, update: SelfPlayStats) -> None:
    total.games += update.games
    total.positions += update.positions
    total.leaves += update.leaves
    total.nn_evals += update.nn_evals
    total.wall_time += update.wall_time
    total.inference_time += update.inference_time
    total.plumbing_time += update.plumbing_time


def _run_segmented_single_pipeline(
    model_table: list[torch.nn.Module],
    segments: list[Any],
    replay: ReplayBuffer,
    config: TrainConfig,
    generation: int,
    device: torch.device,
) -> SelfPlayStats:
    """Exact segment fallback for one-worker gated test and CPU runs."""
    total = SelfPlayStats()
    base_seed = int(config.seed) + (int(generation) * 0x9E37)
    for task_index, segment in enumerate(segments):
        seat_models = (model_table[segment.seat0_model_id], model_table[segment.seat1_model_id])
        stats = run_routed_self_play_generation(
            seat_models,
            replay,
            config.selfplay,
            seed=base_seed ^ (task_index * 0x10001),
            device=device,
            target_games=segment.n_games,
        )
        _add_stats(total, stats)
    return total


def run_training(config: TrainConfig, resume: str | None = None, profile: bool = False) -> dict[str, Any]:
    requested = config
    device = select_device(config.device)
    seed_everything(config.seed, deterministic=device.type == "cpu")
    resume = resolve_resume_path(resume, requested.checkpoint_dir)
    if resume is not None:
        config, start_generation, model, optimizer, replay = load_checkpoint(resume, device)
        config.generations = requested.generations
        config.checkpoint_dir = requested.checkpoint_dir
        config.metrics_csv = requested.metrics_csv
        config.device = requested.device
        config.parallel_workers = requested.parallel_workers
        config.worker_device = requested.worker_device
        config.server_device = requested.server_device
        config.server_max_batch = requested.server_max_batch
        config.server_max_wait_ms = requested.server_max_wait_ms
        config.server_fp16 = requested.server_fp16
        config.server_response_timeout_s = requested.server_response_timeout_s
        config.server_transport = requested.server_transport
        config.server_shm_slots = requested.server_shm_slots
        config.server_poll = requested.server_poll
        config.gate_games = requested.gate_games
        config.gate_sims = requested.gate_sims
        config.gate_threshold = requested.gate_threshold
        config.league_fraction = requested.league_fraction
        config.league_pool_size = requested.league_pool_size
        config.gate_warmup_generations = requested.gate_warmup_generations
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

    inference_server: InferenceServer | None = None
    pool: ParallelSelfPlayPool | None = None
    use_gating = gating_enabled(config)
    best_model: DominionNet | None = None
    best_generation: int | None = None
    best_path: Path | None = None
    if use_gating:
        # A fresh run seeds best from the initial candidate. On resume, best.pt
        # is authoritative because rejected generation checkpoints are still
        # candidate checkpoints and must not silently become self-play policy.
        resume_best = Path(resume).parent / "best.pt" if resume is not None else None
        best_model, best_generation, best_path = initialize_best_checkpoint(
            config,
            model,
            device,
            source_path=resume_best,
            start_generation=start_generation,
        )
    try:
        if config.parallel_workers > 1 and config.worker_device.lower() == "server":
            inference_server = InferenceServer(config, config.parallel_workers)
        if config.parallel_workers > 1:
            pool = ParallelSelfPlayPool(config, inference_server)
        for generation in range(start_generation + 1, start_generation + generations + 1):
            gen_start = time.perf_counter()
            lr = learning_rate_for_generation(config, generation)
            set_optimizer_lr(optimizer, lr)
            if not use_gating:
                # Keep the legacy default path and its seed derivation intact.
                if pool is None:
                    gen_seed = config.seed + (generation * 0x9E37)
                    sp_stats = run_self_play_generation(model, replay, config.selfplay, gen_seed, device)
                    aggregate_games_per_hour = sp_stats.games_per_hour
                else:
                    if inference_server is not None:
                        # The acknowledgement is a generation barrier: workers do
                        # not submit work until the server owns this complete state.
                        inference_server.sync_weights(model, generation)
                    parallel_result = pool.generate(model, replay, generation)
                    sp_stats = parallel_result.stats
                    aggregate_games_per_hour = parallel_result.aggregate_games_per_hour
            else:
                assert best_model is not None
                assert best_generation is not None
                assert best_path is not None
                all_league_paths = league_checkpoint_paths(config)
                sampled_segments = plan_selfplay_segments(
                    config.selfplay.games_per_generation,
                    config.league_fraction,
                    len(all_league_paths),
                    config.seed ^ (generation * 0xC0FFEE),
                )
                segments, history_indices = compact_selfplay_segments(sampled_segments)
                league_paths = [all_league_paths[index] for index in history_indices]
                planned_league_games = sum(segment.n_games for segment in segments if segment.is_league)
                if planned_league_games and inference_server is not None:
                    raise ValueError("mini-league self-play requires worker_device='cpu' or 'cuda', not 'server'")
                if pool is None:
                    model_table = [best_model, *[load_best_checkpoint(config, device, path)[0] for path in league_paths]]
                    sp_stats = _run_segmented_single_pipeline(
                        model_table,
                        segments,
                        replay,
                        config,
                        generation,
                        device,
                    )
                    aggregate_games_per_hour = sp_stats.games_per_hour
                else:
                    if inference_server is not None:
                        inference_server.sync_weights(best_model, generation)
                    payloads = [] if inference_server is not None else [serialize_cpu_state_dict(best_model)]
                    if inference_server is None:
                        payloads.extend(
                            serialize_cpu_state_dict(load_best_checkpoint(config, device, path)[0])
                            for path in league_paths
                        )
                    parallel_result = pool.generate(
                        best_model,
                        replay,
                        generation,
                        segments=segments,
                        model_state_payloads=payloads,
                    )
                    sp_stats = parallel_result.stats
                    aggregate_games_per_hour = parallel_result.aggregate_games_per_hour
                    planned_league_games = parallel_result.league_games
            server_metrics = (
                inference_server.collect_metrics(generation)
                if inference_server is not None
                else {
                    "server_evals_per_sec": 0.0,
                    "server_mean_batch_size": 0.0,
                    "server_batch_wait_p50_ms": 0.0,
                    "server_batch_wait_p99_ms": 0.0,
                }
            )

            losses = {"policy_loss": float("nan"), "value_loss": float("nan"), "entropy": float("nan")}
            steps = config.optim.train_steps_per_generation
            if len(replay) > 0 and steps > 0:
                accum = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
                for _ in range(steps):
                    step_losses = train_step(model, optimizer, replay, config.optim.batch_size, device)
                    for key in accum:
                        accum[key] += step_losses[key]
                losses = {key: value / steps for key, value in accum.items()}

            gate_row: dict[str, Any] = {}
            if use_gating:
                assert best_model is not None
                assert best_generation is not None
                assert best_path is not None
                if generation <= config.gate_warmup_generations:
                    # Cold-start warmup: strict gating from a random-init best
                    # deadlocks (candidates train on random-play data and lose
                    # gate matches to uniform-prior search). Accept
                    # unconditionally until the data pool is net-guided.
                    gate_stats = GateStats(wins=0, losses=0, ties=0)
                    result = "warmup_accepted"
                else:
                    gate_stats = run_gate_match(model, best_model, config, generation, device)
                    result = gate_result(gate_stats, config.gate_threshold)
                if result in ("accepted", "warmup_accepted"):
                    archive_previous_best(config, best_path, best_generation)
                    best_model.load_state_dict(model.state_dict())
                    best_model.eval()
                    best_generation = generation
                    save_best_checkpoint(config, best_generation, best_model, best_path)
                gate_row = {
                    "gate_result": result,
                    "gate_win_pct": gate_stats.win_pct,
                    "best_generation": best_generation,
                }

            path = save_checkpoint(
                config,
                generation,
                model,
                optimizer,
                replay,
                best_generation=best_generation if use_gating else None,
            )
            eval_row: dict[str, Any] = {}
            should_eval = (
                not profile
                and config.eval.eval_every_n_generations > 0
                and (
                    generation == start_generation + 1
                    or generation % config.eval.eval_every_n_generations == 0
                )
            )
            if should_eval:
                if __package__ in (None, ""):
                    from src.v2.train.evaluate import evaluate_checkpoint
                else:
                    from .evaluate import evaluate_checkpoint

                stats = evaluate_checkpoint(
                    path,
                    opponent=config.eval.eval_opponent,
                    games=config.eval.eval_games,
                    sims=config.eval.eval_sims,
                    kingdoms=config.eval.eval_kingdoms,
                    seed=config.seed ^ (generation * 0x4556),
                    device_name=device.type,
                    n_games=config.eval.eval_n_games,
                    max_batch=config.eval.eval_max_batch,
                )
                eval_row = {
                    "eval_opponent": stats.opponent,
                    "eval_games": stats.games,
                    "eval_wins": stats.wins,
                    "eval_losses": stats.losses,
                    "eval_ties": stats.ties,
                    "eval_truncated": stats.truncated,
                    "eval_win_pct_excl_ties": stats.win_pct_excl_ties,
                    "eval_games_per_hour": stats.games_per_hour,
                }
            row = {
                "generation": generation,
                "games": sp_stats.games,
                "positions": sp_stats.positions,
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "entropy": losses["entropy"],
                "lr": lr,
                "games_per_hour": sp_stats.games_per_hour,
                "leaves_per_sec": sp_stats.leaves_per_sec,
                "nn_evals_per_sec": sp_stats.nn_evals_per_sec,
                "inference_pct": sp_stats.inference_pct,
                "plumbing_pct": sp_stats.plumbing_pct,
                "workers": config.parallel_workers,
                "aggregate_games_per_hour": aggregate_games_per_hour,
                **server_metrics,
                "wall_time": time.perf_counter() - gen_start,
                "checkpoint": str(path),
                **gate_row,
                "league_games": planned_league_games if use_gating else 0,
            }
            row.update(eval_row)
            append_metrics(config.metrics_csv, row)
            metrics.append(row)
            print(json.dumps(row, sort_keys=True))
    finally:
        if pool is not None:
            pool.close()
        if inference_server is not None:
            inference_server.close()

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
    config_path = Path(__file__).resolve().parent / "configs" / "smoke.json" if args.smoke else args.config
    config = load_config(config_path)
    if args.device is not None:
        config.device = args.device
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
        config.metrics_csv = str(Path(args.checkpoint_dir) / "metrics.csv")
    elif args.smoke:
        config.checkpoint_dir = "/tmp/dominion_v2_train_smoke"
        config.metrics_csv = "/tmp/dominion_v2_train_smoke/metrics.csv"
    if args.smoke and args.device is None:
        config.device = "cpu"
        config.generations = 1
    run_training(config, resume=args.resume, profile=args.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
