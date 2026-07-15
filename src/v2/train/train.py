from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
import time
import warnings
from dataclasses import asdict
from math import cos, isfinite, pi
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
    from src.v2.train.config import TrainConfig, add_config_args, load_config, save_config, validate_deep_slice_config
    from src.v2.train.gating import (
        GateStats,
        archive_previous_best,
        archive_self_checkpoint,
        gate_result,
        gating_enabled,
        initialize_best_checkpoint,
        league_checkpoint_paths,
        load_best_checkpoint,
        compact_selfplay_segments,
        assign_kingdom_phase_to_segments,
        effective_kingdom_phase,
        effective_league_fraction,
        effective_scripted_fractions,
        league_opponent_weights,
        plan_training_selfplay_segments,
        run_gate_match,
        save_best_checkpoint,
        seed_league_checkpoints,
    )
    from src.v2.train.inference_server import InferenceServer, serialize_cpu_state_dict
    from src.v2.train.model import build_model, count_parameters, masked_policy_loss, model_config_dict
    from src.v2.train.observation import obs_size_for_config, obs_size_for_version, obs_version_for_checkpoint
    from src.v2.train.replay import ReplayBuffer, load_replay_state, save_replay_state
    from src.v2.train.selfplay import SelfPlayStats, run_routed_self_play_generation, run_self_play_generation
    from src.v2.train.workers import ParallelSelfPlayPool
else:
    from .config import TrainConfig, add_config_args, load_config, save_config, validate_deep_slice_config
    from .gating import (
        GateStats,
        archive_previous_best,
        archive_self_checkpoint,
        gate_result,
        gating_enabled,
        initialize_best_checkpoint,
        league_checkpoint_paths,
        load_best_checkpoint,
        compact_selfplay_segments,
        assign_kingdom_phase_to_segments,
        effective_kingdom_phase,
        effective_league_fraction,
        effective_scripted_fractions,
        league_opponent_weights,
        plan_training_selfplay_segments,
        run_gate_match,
        save_best_checkpoint,
        seed_league_checkpoints,
    )
    from .inference_server import InferenceServer, serialize_cpu_state_dict
    from .model import build_model, count_parameters, masked_policy_loss, model_config_dict
    from .observation import obs_size_for_config, obs_size_for_version, obs_version_for_checkpoint
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
    obs_size = obs_size_for_config(config)
    model = build_model(config.model, obs_size, dz.ACTION_SPACE_SIZE).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )
    replay = ReplayBuffer(config.replay.capacity, obs_size, dz.ACTION_SPACE_SIZE, config.seed ^ 0xA11CE)
    return model, optimizer, replay


def validate_model_config(config: TrainConfig) -> None:
    """Validate architecture choices before a training run allocates a model."""
    model_config = model_config_dict(config.model)
    arch = model_config.get("arch", "mlp")
    if not isinstance(arch, str):
        raise ValueError("model.arch must be a string")
    if arch not in {"mlp", "card_transformer"}:
        raise ValueError(f"unknown model.arch {arch!r}; expected 'mlp' or 'card_transformer'")
    if arch == "card_transformer" and int(config.selfplay.obs_version) != 2:
        raise ValueError("model.arch='card_transformer' requires selfplay.obs_version == 2")


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
    if __package__ in (None, ""):
        from src.v2.train.config import _merge_dataclass  # local keeps the helper private
    else:
        from .config import _merge_dataclass

    _merge_dataclass(cfg, cfg_dict)
    validate_model_config(cfg)
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


def _initial_weights_config(payload: object, path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return validated metadata needed to safely warm-start a new campaign."""
    if not isinstance(payload, dict):
        raise ValueError(f"init-weights checkpoint {path} must contain an object payload")
    if not isinstance(payload.get("model"), dict):
        raise ValueError(f"init-weights checkpoint {path} is missing a model state dict")
    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"init-weights checkpoint {path} has a non-object 'config' payload")
    checkpoint_model = checkpoint_config.get("model")
    checkpoint_selfplay = checkpoint_config.get("selfplay")
    if not isinstance(checkpoint_model, dict):
        raise ValueError(f"init-weights checkpoint {path} is missing config.model metadata")
    if not isinstance(checkpoint_selfplay, dict):
        raise ValueError(f"init-weights checkpoint {path} is missing config.selfplay metadata")
    return checkpoint_model, checkpoint_selfplay


def load_initial_weights(
    path: str | Path,
    config: TrainConfig,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    """Load only compatible network weights for a fresh training campaign.

    This intentionally receives neither optimizer nor replay.  Keeping the
    boundary narrow makes it impossible for a warm start to accidentally
    inherit source campaign state as a full ``--resume`` does.
    """
    checkpoint_path = Path(path)
    payload = load_full_checkpoint(checkpoint_path, device)
    checkpoint_model, checkpoint_selfplay = _initial_weights_config(payload, checkpoint_path)

    checkpoint_arch = checkpoint_model.get("arch", "mlp")
    expected_model = model_config_dict(config.model)
    expected_arch = expected_model.get("arch", "mlp")
    if checkpoint_arch != expected_arch:
        raise ValueError(
            f"init-weights checkpoint {checkpoint_path} has model.arch {checkpoint_arch!r}, "
            f"but the current run requires {expected_arch!r}"
        )
    if checkpoint_arch == "mlp":
        checkpoint_hidden_sizes = checkpoint_model.get("hidden_sizes")
        expected_hidden_sizes = list(config.model.hidden_sizes)
        if not isinstance(checkpoint_hidden_sizes, list):
            raise ValueError(
                f"init-weights checkpoint {checkpoint_path} is missing config.model.hidden_sizes; "
                "cannot validate model architecture"
            )
        if checkpoint_hidden_sizes != expected_hidden_sizes:
            raise ValueError(
                f"init-weights checkpoint {checkpoint_path} has hidden_sizes {checkpoint_hidden_sizes}, "
                f"but the current run requires {expected_hidden_sizes}"
            )
    elif checkpoint_arch == "card_transformer":
        for key, default in (("d_model", 192), ("n_layers", 3), ("n_heads", 4), ("ffn_multiplier", 4), ("dropout", 0.0)):
            if checkpoint_model.get(key, default) != expected_model.get(key, default):
                raise ValueError(
                    f"init-weights checkpoint {checkpoint_path} has {key} "
                    f"{checkpoint_model.get(key, default)!r}, but the current run requires "
                    f"{expected_model.get(key, default)!r}"
                )
    else:
        raise ValueError(f"init-weights checkpoint {checkpoint_path} has unknown model.arch {checkpoint_arch!r}")

    stored_obs_version = checkpoint_selfplay.get("obs_version")
    if not isinstance(stored_obs_version, int) or isinstance(stored_obs_version, bool):
        raise ValueError(
            f"init-weights checkpoint {checkpoint_path} is missing a valid config.selfplay.obs_version"
        )
    required_obs_version = int(config.selfplay.obs_version)
    try:
        stored_obs_width = obs_size_for_version(stored_obs_version)
        model_obs_version = 2 if checkpoint_arch == "card_transformer" else obs_version_for_checkpoint(payload)
        model_obs_width = obs_size_for_version(model_obs_version)
        required_obs_width = obs_size_for_config(config)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"init-weights checkpoint {checkpoint_path} has an invalid observation input layout: {exc}"
        ) from exc
    if (
        stored_obs_version != required_obs_version
        or model_obs_version != required_obs_version
        or stored_obs_width != required_obs_width
        or model_obs_width != required_obs_width
    ):
        raise ValueError(
            f"init-weights checkpoint {checkpoint_path} has stored obs_version {stored_obs_version} "
            f"and model layout v{model_obs_version} (input width {model_obs_width}); current run requires "
            f"obs_version {required_obs_version} and input width {required_obs_width}"
        )

    if checkpoint_arch == "mlp":
        raw_input_scale = checkpoint_model.get("input_scale", 1.0)
        if isinstance(raw_input_scale, bool) or not isinstance(raw_input_scale, (int, float)) or not isfinite(raw_input_scale):
            raise ValueError(f"init-weights checkpoint {checkpoint_path} has an invalid config.model.input_scale")
        stored_input_scale = float(raw_input_scale)
        required_input_scale = float(config.model.input_scale)
        if stored_input_scale != required_input_scale:
            raise ValueError(
                f"init-weights checkpoint {checkpoint_path} has input_scale {stored_input_scale}, "
                f"but the current run requires {required_input_scale}"
            )

    try:
        model.load_state_dict(payload["model"])
    except (RuntimeError, TypeError, KeyError) as exc:
        raise ValueError(
            f"init-weights checkpoint {checkpoint_path} cannot be loaded with the current model architecture"
        ) from exc


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


METRICS_FIELDNAMES = [
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
        "eval_end_province",
        "eval_end_piles",
        "eval_end_trunc",
        "eval_win_pct_excl_ties",
        "eval_games_per_hour",
        "gate_result",
        "gate_win_pct",
        "best_generation",
        "league_games",
        "deep_games",
        "deep_positions",
        "kingdom_phase",
        "scripted_games",
        "scripted_wins",
        "routed_fast_path_batches",
        "routed_split_batches",
]


def _dynamic_metric_fieldnames(row: dict[str, Any]) -> list[str]:
    return sorted(
        key
        for key in row
        if (
            key.startswith("scripted_games_")
            or key.startswith("scripted_wins_")
            or key.startswith("league_games_")
            or key.startswith("league_wins_")
            or (key.startswith("sentinel_") and (key.endswith("_games") or key.endswith("_wins")))
        )
    )


def append_metrics(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dynamic_fieldnames = _dynamic_metric_fieldnames(row)
    required_fieldnames = [*METRICS_FIELDNAMES, *dynamic_fieldnames]
    if out.exists():
        with out.open(newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_fieldnames:
            fieldnames = list(existing_fieldnames)
            missing_fieldnames = [fieldname for fieldname in required_fieldnames if fieldname not in fieldnames]
            if missing_fieldnames:
                fieldnames.extend(missing_fieldnames)
                with tempfile.NamedTemporaryFile(
                    "w", newline="", dir=out.parent, delete=False
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(existing_rows)
                    writer.writerow({key: row.get(key, "") for key in fieldnames})
                    temporary_path = Path(handle.name)
                temporary_path.replace(out)
                return
            with out.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writerow({key: row.get(key, "") for key in fieldnames})
            return

    fieldnames = [*METRICS_FIELDNAMES, *dynamic_fieldnames]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
    total.routed_fast_path_batches += update.routed_fast_path_batches
    total.routed_split_batches += update.routed_split_batches
    total.scripted_games += update.scripted_games
    total.scripted_wins += update.scripted_wins
    total.deep_games += update.deep_games
    total.deep_positions += update.deep_positions
    for kind, (games, wins) in update.scripted_by_kind.items():
        previous_games, previous_wins = total.scripted_by_kind.get(kind, (0, 0))
        total.scripted_by_kind[kind] = (previous_games + games, previous_wins + wins)
    for opponent, (games, wins) in update.league_by_opponent.items():
        previous_games, previous_wins = total.league_by_opponent.get(opponent, (0, 0))
        total.league_by_opponent[opponent] = (previous_games + games, previous_wins + wins)


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
            scripted_kind=segment.scripted_kind,
            scripted_nn_player=segment.nn_player,
            kingdom_pool=segment.kingdom_pool,
            kingdom_mode=segment.kingdom_mode,
            sims_override=segment.sims_override,
            league_opponent=segment.league_opponent,
        )
        if segment.sims_override:
            stats.deep_games += stats.games
            stats.deep_positions += stats.positions
        _add_stats(total, stats)
    return total


def validated_eval_sentinels(raw_sentinels: object) -> list[tuple[str, int]]:
    """Validate the compact eval-ladder sentinel configuration."""
    if not isinstance(raw_sentinels, list):
        raise ValueError("eval_sentinels must be a list of {opponent, games} objects")
    allowed = {"bigmoney", "engine", "engine2", "engine3", "mcts"}
    sentinels: list[tuple[str, int]] = []
    seen: set[str] = set()
    for index, raw_sentinel in enumerate(raw_sentinels):
        if not isinstance(raw_sentinel, dict):
            raise ValueError(f"eval sentinel {index} must be an object")
        opponent = raw_sentinel.get("opponent")
        games = raw_sentinel.get("games")
        if not isinstance(opponent, str) or opponent not in allowed:
            raise ValueError(f"eval sentinel {index} opponent must be one of {', '.join(sorted(allowed))}")
        if not isinstance(games, int) or isinstance(games, bool) or games <= 0:
            raise ValueError(f"eval sentinel {index} games must be a positive integer")
        if opponent in seen:
            raise ValueError(f"eval_sentinels contains duplicate opponent {opponent!r}")
        seen.add(opponent)
        sentinels.append((opponent, games))
    return sentinels


def run_training(config: TrainConfig, resume: str | None = None, profile: bool = False) -> dict[str, Any]:
    requested = config
    if not isinstance(config.init_weights, str):
        raise ValueError("init_weights must be a checkpoint path string")
    if resume is not None and config.init_weights:
        raise ValueError("--resume and --init-weights cannot be used together")
    validate_model_config(config)
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
        config.server_compile = requested.server_compile
        config.server_autocast_bf16 = requested.server_autocast_bf16
        config.server_batch_buckets = requested.server_batch_buckets
        config.server_response_timeout_s = requested.server_response_timeout_s
        config.server_transport = requested.server_transport
        config.server_shm_slots = requested.server_shm_slots
        config.server_poll = requested.server_poll
        config.gate_games = requested.gate_games
        config.gate_sims = requested.gate_sims
        config.gate_threshold = requested.gate_threshold
        config.gate_temp_moves = requested.gate_temp_moves
        config.league_fraction = requested.league_fraction
        config.league_pool_size = requested.league_pool_size
        config.league_opponents_per_gen = requested.league_opponents_per_gen
        config.league_seed_checkpoints = requested.league_seed_checkpoints
        config.league_self_every = requested.league_self_every
        config.league_schedule = requested.league_schedule
        config.scripted_opponents = requested.scripted_opponents
        config.scripted_opponent_schedule = requested.scripted_opponent_schedule
        config.kingdom_curriculum = requested.kingdom_curriculum
        config.gate_warmup_generations = requested.gate_warmup_generations
        config.gate_force_accept_every = requested.gate_force_accept_every
        validate_model_config(config)
        if config.device == "auto":
            config.device = device.type
    else:
        model, optimizer, replay = build_objects(config, device)
        if config.init_weights:
            load_initial_weights(config.init_weights, config, model, device)
        start_generation = 0

    validate_deep_slice_config(config.selfplay)

    metrics: list[dict[str, Any]] = []
    generations = 1 if profile else max(0, config.generations - start_generation)
    print(
        json.dumps(
            {
                "device": device.type,
                "parameters": count_parameters(model),
                "obs_size": obs_size_for_config(config),
                "action_size": dz.ACTION_SPACE_SIZE,
            },
            sort_keys=True,
        )
    )

    inference_server: InferenceServer | None = None
    pool: ParallelSelfPlayPool | None = None
    use_gating = gating_enabled(config)
    if not isinstance(config.scripted_opponents, dict):
        raise ValueError("scripted_opponents must be an object mapping kind to fraction")
    effective_scripted_fractions(config.scripted_opponent_schedule, config.scripted_opponents, start_generation)
    effective_league_fraction(config.league_schedule, config.league_fraction, start_generation)
    if (
        not isinstance(config.league_opponents_per_gen, int)
        or isinstance(config.league_opponents_per_gen, bool)
        or config.league_opponents_per_gen < 0
    ):
        raise ValueError("league_opponents_per_gen must be a non-negative integer")
    if (
        not isinstance(config.league_self_every, int)
        or isinstance(config.league_self_every, bool)
        or config.league_self_every < 0
    ):
        raise ValueError("league_self_every must be a non-negative integer")
    effective_kingdom_phase(config.kingdom_curriculum, config.selfplay.kingdom_mode, start_generation)
    eval_sentinels = validated_eval_sentinels(config.eval.eval_sentinels)
    configured_scripted_kinds = sorted(
        set(config.scripted_opponents) | set(config.scripted_opponent_schedule)
    )
    league_configured = (
        float(config.league_fraction) > 0.0
        or bool(config.league_schedule)
        or bool(config.league_seed_checkpoints)
        or int(config.league_self_every) > 0
    )
    league_performance: dict[str, tuple[int, int]] = {}
    configured_league_opponents: set[str] = set()
    best_model: torch.nn.Module | None = None
    best_generation: int | None = None
    best_path: Path | None = None
    if league_configured:
        # External opponents must be available before generation one even for
        # ungated runs; their input pipeline is validated during this copy.
        seed_league_checkpoints(config, device)
        configured_league_opponents.update(path.name for path in league_checkpoint_paths(config))
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
            planned_league_games = 0
            effective_league = effective_league_fraction(
                config.league_schedule,
                config.league_fraction,
                generation,
            )
            effective_scripted = effective_scripted_fractions(
                config.scripted_opponent_schedule,
                config.scripted_opponents,
                generation,
            )
            effective_kingdom = effective_kingdom_phase(
                config.kingdom_curriculum,
                config.selfplay.kingdom_mode,
                generation,
            )
            use_segments = (
                use_gating
                or league_configured
                or bool(config.scripted_opponents)
                or bool(config.scripted_opponent_schedule)
                or bool(config.kingdom_curriculum)
                or float(config.selfplay.deep_slice_fraction) > 0.0
            )
            if not use_segments:
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
                active_model = best_model if use_gating else model
                assert active_model is not None
                # This deliberately remains independent of candidate gating:
                # margin-era runs use the current candidate as model zero.
                all_league_paths = league_checkpoint_paths(config)
                league_names = [path.name for path in all_league_paths]
                configured_league_opponents.update(league_names)
                sampled_segments = plan_training_selfplay_segments(
                    config.selfplay.games_per_generation,
                    effective_league,
                    len(all_league_paths),
                    effective_scripted,
                    config.seed ^ (generation * 0xC0FFEE),
                    deep_slice_fraction=config.selfplay.deep_slice_fraction,
                    deep_slice_sims=config.selfplay.deep_slice_sims,
                    sims_per_move=config.selfplay.sims_per_move,
                    league_opponent_weights=league_opponent_weights(league_names, league_performance),
                    league_opponent_names=league_names,
                    parallel_workers=config.parallel_workers,
                    league_opponents_per_gen=config.league_opponents_per_gen,
                )
                sampled_segments = assign_kingdom_phase_to_segments(
                    sampled_segments,
                    effective_kingdom,
                    config.seed ^ (generation * 0x4B1D0),
                )
                segments, history_indices = compact_selfplay_segments(sampled_segments)
                league_paths = [all_league_paths[index] for index in history_indices]
                planned_league_games = sum(segment.n_games for segment in segments if segment.is_league)
                if planned_league_games and inference_server is not None:
                    raise ValueError("mini-league self-play requires worker_device='cpu' or 'cuda', not 'server'")
                if pool is None:
                    model_table = [active_model, *[load_best_checkpoint(config, device, path)[0] for path in league_paths]]
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
                        inference_server.sync_weights(active_model, generation)
                    payloads = []
                    if inference_server is None:
                        payloads.append(
                            (
                                serialize_cpu_state_dict(active_model),
                                model_config_dict(getattr(active_model, "_dominion_model_config", config.model)),
                            )
                        )
                    if inference_server is None:
                        for path in league_paths:
                            opponent, _ = load_best_checkpoint(config, device, path)
                            payloads.append(
                                (
                                    serialize_cpu_state_dict(opponent),
                                    model_config_dict(getattr(opponent, "_dominion_model_config", config.model)),
                                )
                            )
                    parallel_result = pool.generate(
                        active_model,
                        replay,
                        generation,
                        segments=segments,
                        model_state_payloads=payloads,
                    )
                    sp_stats = parallel_result.stats
                    aggregate_games_per_hour = parallel_result.aggregate_games_per_hour
                    planned_league_games = parallel_result.league_games
            for opponent, (games, wins) in sp_stats.league_by_opponent.items():
                # Use the most recently observed per-opponent rate for the
                # following draw; unplayed opponents retain their last rate.
                league_performance[opponent] = (games, wins)
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
                    stale_generations = generation - best_generation - 1
                    if (
                        int(config.gate_force_accept_every) > 0
                        and stale_generations >= int(config.gate_force_accept_every)
                    ):
                        result = "forced_accepted"
                    else:
                        result = gate_result(gate_stats, config.gate_threshold)
                if result in ("accepted", "warmup_accepted", "forced_accepted"):
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
            if league_configured:
                self_paths = archive_self_checkpoint(config, path, generation)
                configured_league_opponents.update(path.name for path in self_paths)
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
                    "eval_end_province": stats.end_province,
                    "eval_end_piles": stats.end_piles,
                    "eval_end_trunc": stats.end_trunc,
                    "eval_win_pct_excl_ties": stats.win_pct_excl_ties,
                    "eval_games_per_hour": stats.games_per_hour,
                }
                for sentinel_index, (opponent, games) in enumerate(eval_sentinels):
                    sentinel_stats = evaluate_checkpoint(
                        path,
                        opponent=opponent,
                        games=games,
                        sims=config.eval.eval_sims,
                        kingdoms=config.eval.eval_kingdoms,
                        seed=(config.seed ^ (generation * 0x4556) ^ ((sentinel_index + 1) * 0x10001)),
                        device_name=device.type,
                        n_games=config.eval.eval_n_games,
                        max_batch=config.eval.eval_max_batch,
                    )
                    eval_row[f"sentinel_{opponent}_wins"] = sentinel_stats.wins
                    eval_row[f"sentinel_{opponent}_games"] = sentinel_stats.games
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
                "league_games": planned_league_games,
                "deep_games": sp_stats.deep_games,
                "deep_positions": sp_stats.deep_positions,
                "kingdom_phase": effective_kingdom.label,
                "scripted_games": sp_stats.scripted_games,
                "scripted_wins": sp_stats.scripted_wins,
                "routed_fast_path_batches": sp_stats.routed_fast_path_batches,
                "routed_split_batches": sp_stats.routed_split_batches,
            }
            for kind in configured_scripted_kinds:
                games, wins = sp_stats.scripted_by_kind.get(kind, (0, 0))
                row[f"scripted_games_{kind}"] = games
                row[f"scripted_wins_{kind}"] = wins
            for opponent in sorted(configured_league_opponents | set(sp_stats.league_by_opponent)):
                games, wins = sp_stats.league_by_opponent.get(opponent, (0, 0))
                row[f"league_wins_{opponent}"] = wins
                row[f"league_games_{opponent}"] = games
            if effective_scripted:
                row["scripted_opponent_fractions"] = effective_scripted
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
    if args.resume is not None and args.init_weights is not None:
        parser.error("--resume and --init-weights cannot be used together")
    if args.init_weights is not None:
        config.init_weights = args.init_weights
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
    if args.resume is not None and config.init_weights:
        parser.error("--resume and --init-weights cannot be used together")
    run_training(config, resume=args.resume, profile=args.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
