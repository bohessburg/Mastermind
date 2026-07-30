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
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F

try:
    import dominion_v2_py as dz
except ModuleNotFoundError as exc:  # pragma: no cover - gives a clearer CLI error
    raise SystemExit("dominion_v2_py not found; run with PYTHONPATH=build") from exc

from src.v2.encoder_compat import require_native_runner_encoder_compatibility

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.v2.train.config import (
        TrainConfig,
        add_config_args,
        anchor_weight_for_generation,
        load_config,
        opening_template_schedule,
        save_config,
        scheduled_opening_selfplay_config,
        validate_aux_margin_config,
        validate_determinize_config,
        validate_deep_slice_config,
        validate_forced_playouts_config,
        validate_imitation_config,
        validate_opening_template_config,
        validate_optim_config,
        validate_optimizer_kind,
        validate_temperature_config,
        validate_value_target_config,
    )
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
    from src.v2.train.card_transformer import margin_bucket_ids
    from src.v2.train.human_data import HumanBatch, HumanTupleDataset, load_human_tuples
    from src.v2.train.model import build_model, count_parameters, masked_policy_loss, model_config_dict
    from src.v2.train.observation import obs_size_for_config, obs_size_for_version, obs_version_for_checkpoint
    from src.v2.train.progress import TrainingProgress
    from src.v2.train.replay import ReplayBuffer, load_replay_state, save_replay_state
    from src.v2.train.selfplay import SelfPlayStats, run_routed_self_play_generation, run_self_play_generation
    from src.v2.train.workers import ParallelSelfPlayPool
else:
    from .config import (
        TrainConfig,
        add_config_args,
        anchor_weight_for_generation,
        load_config,
        opening_template_schedule,
        save_config,
        scheduled_opening_selfplay_config,
        validate_aux_margin_config,
        validate_determinize_config,
        validate_deep_slice_config,
        validate_forced_playouts_config,
        validate_imitation_config,
        validate_opening_template_config,
        validate_optim_config,
        validate_optimizer_kind,
        validate_temperature_config,
        validate_value_target_config,
    )
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
    from .card_transformer import margin_bucket_ids
    from .human_data import HumanBatch, HumanTupleDataset, load_human_tuples
    from .model import build_model, count_parameters, masked_policy_loss, model_config_dict
    from .observation import obs_size_for_config, obs_size_for_version, obs_version_for_checkpoint
    from .progress import TrainingProgress
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
    # Keep direct callers on the same self-describing transformer-config path
    # as run_training()/checkpoint loading.
    validate_aux_margin_config(config)
    validate_model_config(config)
    validate_optim_config(config.optim)
    obs_size = obs_size_for_config(config)
    model = build_model(config.model, obs_size, dz.ACTION_SPACE_SIZE).to(device)
    optimizer_class = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
    }[config.optim.optimizer]
    optimizer = optimizer_class(
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
    if arch == "card_transformer":
        obs_version = int(config.selfplay.obs_version)
        if obs_version not in (2, 3):
            raise ValueError("model.arch='card_transformer' requires selfplay.obs_version == 2 or 3")
        configured_version = model_config.get("obs_version")
        if configured_version is None:
            # v2 remains implicit for byte-compatible historical metadata.
            # V3 must record its tokenizer layout because it is no longer the
            # only transformer observation ABI.
            if obs_version == 3:
                config.model.obs_version = obs_version
        elif (
            not isinstance(configured_version, int)
            or isinstance(configured_version, bool)
            or configured_version != obs_version
        ):
            raise ValueError("model.obs_version must match selfplay.obs_version for card_transformer")


def hard_label_policy_loss(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    actions: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross entropy for one demonstrated legal action per row."""
    masked_logits = logits.masked_fill(~legal_mask, -1.0e9)
    per_row = F.cross_entropy(masked_logits, actions.to(dtype=torch.long), reduction="none")
    if weights is None:
        return per_row.mean()
    return (per_row * weights).mean()


def anchor_awr_weights(
    value_target: torch.Tensor,
    value_prediction: torch.Tensor | None,
    beta: float,
) -> torch.Tensor:
    """Return mean-one AWR weights for a human policy batch.

    The value prediction is detached by the caller: this reweights the policy
    objective without introducing a second, implicit value-gradient path.
    Keeping the exponent's ceiling at three limits any single demonstration's
    relative influence to a small, explicit bound.  Subtracting the maximum
    before exponentiation is algebraically cancelled by batch normalization
    and prevents underflow for very small positive beta.
    """
    if beta <= 0.0:
        # This branch deliberately does not inspect ``value_prediction``.
        # beta=0 is exactly the unweighted hard-label CE path.
        return torch.ones_like(value_target)
    if value_prediction is None:
        raise ValueError("positive anchor_awr_beta requires a value prediction")
    log_weights = torch.clamp((value_target - value_prediction) / float(beta), max=3.0)
    stable_weights = torch.exp(log_weights - log_weights.max())
    return stable_weights / stable_weights.mean()


def _human_batch_tensors(batch: HumanBatch, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(batch.obs, dtype=torch.float32, device=device),
        torch.as_tensor(batch.action, dtype=torch.long, device=device),
        torch.as_tensor(batch.legal, dtype=torch.bool, device=device),
        torch.as_tensor(batch.value, dtype=torch.float32, device=device),
    )


def human_imitation_losses(
    model: torch.nn.Module,
    batch: HumanBatch,
    device: torch.device,
    *,
    anchor_awr_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute hard-action CE and value MSE for one human tuple batch."""
    obs, actions, legal_mask, value_target = _human_batch_tensors(batch, device)
    logits, value_prediction = model(obs)
    if anchor_awr_beta > 0.0:
        awr_weights = anchor_awr_weights(
            value_target,
            value_prediction.detach(),
            anchor_awr_beta,
        )
        policy_loss = hard_label_policy_loss(logits, legal_mask, actions, weights=awr_weights)
    else:
        # Keep beta=0 on the ordinary CE reduction path, rather than merely
        # multiplying it by an all-ones tensor.
        policy_loss = hard_label_policy_loss(logits, legal_mask, actions)
    value_loss = F.mse_loss(value_prediction, value_target)
    return policy_loss, value_loss


def _human_margin_tensor(batch: HumanBatch, device: torch.device) -> torch.Tensor:
    if batch.margin is None:
        raise ValueError("auxiliary margin training requires HumanBatch.margin")
    margin = np.asarray(batch.margin, dtype=np.int16)
    if margin.shape != np.asarray(batch.value).shape:
        raise ValueError("HumanBatch.margin shape must match HumanBatch.value")
    # Materialize labels directly as int64: CrossEntropy expects long class
    # IDs and this also keeps the optional MPS path away from int16 tensors.
    return torch.as_tensor(margin, dtype=torch.long, device=device)


def _forward_with_aux(
    model: torch.nn.Module,
    obs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_with_aux = getattr(model, "forward_with_aux", None)
    if not callable(forward_with_aux):
        raise ValueError("auxiliary margin training requires a CardTokenNet with aux_margin_buckets")
    logits, value, aux_logits = forward_with_aux(obs)
    return logits, value, aux_logits


def _aux_margin_cross_entropy(
    model: torch.nn.Module,
    aux_logits: torch.Tensor,
    margins: torch.Tensor,
) -> torch.Tensor:
    buckets = getattr(model, "aux_margin_buckets", None)
    if buckets is None:
        raise ValueError("auxiliary margin training requires configured aux_margin_buckets")
    labels = margin_bucket_ids(margins, int(buckets))
    if aux_logits.ndim != 2 or aux_logits.shape != (labels.shape[0], int(buckets)):
        raise ValueError("auxiliary margin head output shape does not match configured buckets")
    return F.cross_entropy(aux_logits, labels)


def human_imitation_losses_with_aux(
    model: torch.nn.Module,
    batch: HumanBatch,
    device: torch.device,
    *,
    anchor_awr_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute one shared-context human policy/value/auxiliary objective."""

    obs, actions, legal_mask, value_target = _human_batch_tensors(batch, device)
    margins = _human_margin_tensor(batch, device)
    logits, value_prediction, aux_logits = _forward_with_aux(model, obs)
    if anchor_awr_beta > 0.0:
        awr_weights = anchor_awr_weights(
            value_target,
            value_prediction.detach(),
            anchor_awr_beta,
        )
        policy_loss = hard_label_policy_loss(logits, legal_mask, actions, weights=awr_weights)
    else:
        policy_loss = hard_label_policy_loss(logits, legal_mask, actions)
    value_loss = F.mse_loss(value_prediction, value_target)
    aux_loss = _aux_margin_cross_entropy(model, aux_logits, margins)
    return policy_loss, value_loss, aux_loss


def human_pretrain_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: HumanBatch,
    device: torch.device,
) -> dict[str, float]:
    """Run one behavior-cloning optimizer step on human tuples."""
    model.train()
    policy_loss, value_loss = human_imitation_losses(model, batch, device)
    loss = policy_loss + value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
    }


def run_human_pretrain(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterator[HumanBatch],
    *,
    steps: int,
    device: torch.device,
    pretrain_lr: float | None = None,
) -> list[dict[str, float]]:
    """Run fresh-campaign behavior cloning with a flush-safe heartbeat."""
    if steps <= 0:
        return []
    original_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if pretrain_lr is not None:
        set_optimizer_lr(optimizer, float(pretrain_lr))
    history: list[dict[str, float]] = []
    try:
        for step in range(1, int(steps) + 1):
            result = human_pretrain_step(model, optimizer, next(batches), device)
            history.append(result)
            # Long fresh runs must not appear stalled while pretraining before
            # generation one.  Fifty optimizer steps is the project's normal
            # compact heartbeat cadence.
            if step % 50 == 0 or step == int(steps):
                print(
                    "imitation pretrain "
                    f"step={step}/{int(steps)} loss={result['loss']:.6f} "
                    f"policy={result['policy_loss']:.6f} value={result['value_loss']:.6f}",
                    flush=True,
                )
    finally:
        for group, lr in zip(optimizer.param_groups, original_lrs, strict=True):
            group["lr"] = lr
    return history


def load_human_dataset_for_config(config: TrainConfig) -> HumanTupleDataset:
    """Load tuple data aligned to the active model ABI and value convention."""
    dataset = load_human_tuples(
        config.imitation.human_tuples,
        value_scheme=config.selfplay.value_target,
        margin_blend_alpha=config.selfplay.margin_blend_alpha,
        margin_scale=config.selfplay.margin_scale,
        opponent_kinds=config.imitation.opponent_kinds or None,
        seat_indices=config.imitation.seat_indices or None,
    )
    expected_obs_size = obs_size_for_config(config)
    if dataset.obs_width != expected_obs_size:
        raise ValueError(
            f"human tuples have obs width {dataset.obs_width}, but this run uses obs width {expected_obs_size}"
        )
    if dataset.action_width != int(dz.ACTION_SPACE_SIZE):
        raise ValueError(
            f"human tuples have action width {dataset.action_width}, but this run uses action width {dz.ACTION_SPACE_SIZE}"
        )
    return dataset


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    device: torch.device,
    *,
    human_batches: Iterator[HumanBatch] | None = None,
    anchor_weight: float = 0.0,
    anchor_awr_beta: float = 0.0,
    aux_margin_weight: float = 0.0,
) -> dict[str, float]:
    # Keep the established self-play-only code path literally separate. In
    # particular, inactive anchor/aux objectives must not sample human data,
    # consume an extra RNG draw, or change the legacy return dictionary.
    if anchor_weight <= 0.0 and aux_margin_weight <= 0.0:
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

    # Retain the established anchor-only path as well. This remains useful for
    # existing imitation experiments and keeps it independent of the new head.
    if aux_margin_weight <= 0.0:
        if human_batches is None:
            raise ValueError("human_batches is required when anchor_weight is positive")
        model.train()
        batch = replay.sample(batch_size)
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
        policy_target = torch.as_tensor(batch.policy, dtype=torch.float32, device=device)
        value_target = torch.as_tensor(batch.value, dtype=torch.float32, device=device)
        legal_mask = torch.as_tensor(batch.legal_mask, dtype=torch.bool, device=device)

        logits, value = model(obs)
        policy_loss, entropy = masked_policy_loss(logits, legal_mask, policy_target)
        value_loss = F.mse_loss(value, value_target)
        anchor_policy_loss, anchor_value_loss = human_imitation_losses(
            model,
            next(human_batches),
            device,
            anchor_awr_beta=anchor_awr_beta,
        )
        loss = policy_loss + value_loss + float(anchor_weight) * (anchor_policy_loss + anchor_value_loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "anchor_policy_loss": float(anchor_policy_loss.detach().cpu()),
            "anchor_value_loss": float(anchor_value_loss.detach().cpu()),
        }

    if anchor_weight > 0.0 and human_batches is None:
        raise ValueError("human_batches is required when anchor_weight is positive")
    model.train()
    batch = replay.sample(batch_size)
    obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
    policy_target = torch.as_tensor(batch.policy, dtype=torch.float32, device=device)
    value_target = torch.as_tensor(batch.value, dtype=torch.float32, device=device)
    legal_mask = torch.as_tensor(batch.legal_mask, dtype=torch.bool, device=device)
    margin = torch.as_tensor(batch.margin, dtype=torch.long, device=device)

    logits, value, aux_logits = _forward_with_aux(model, obs)
    policy_loss, entropy = masked_policy_loss(logits, legal_mask, policy_target)
    value_loss = F.mse_loss(value, value_target)
    aux_margin_loss = _aux_margin_cross_entropy(model, aux_logits, margin)
    if anchor_weight > 0.0:
        assert human_batches is not None
        anchor_policy_loss, anchor_value_loss, anchor_aux_margin_loss = human_imitation_losses_with_aux(
            model,
            next(human_batches),
            device,
            anchor_awr_beta=anchor_awr_beta,
        )
        # The anchor's auxiliary CE follows the same source weighting as its
        # policy/value losses, while the reported metric remains in CE units.
        aux_margin_loss = aux_margin_loss + float(anchor_weight) * anchor_aux_margin_loss
        loss = (
            policy_loss
            + value_loss
            + float(aux_margin_weight) * aux_margin_loss
            + float(anchor_weight) * (anchor_policy_loss + anchor_value_loss)
        )
    else:
        loss = policy_loss + value_loss + float(aux_margin_weight) * aux_margin_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    result = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "aux_margin_loss": float(aux_margin_loss.detach().cpu()),
    }
    if anchor_weight > 0.0:
        result["anchor_policy_loss"] = float(anchor_policy_loss.detach().cpu())
        result["anchor_value_loss"] = float(anchor_value_loss.detach().cpu())
    return result


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
        "encoder_generation": int(dz.ENCODER_GENERATION),
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


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    *,
    optimizer_override: str | None = None,
):
    checkpoint_path = Path(path)
    payload = load_full_checkpoint(checkpoint_path, device)
    require_native_runner_encoder_compatibility(payload, checkpoint_path)
    cfg_dict = payload["config"]
    cfg = load_config(None)
    if __package__ in (None, ""):
        from src.v2.train.config import _merge_dataclass  # local keeps the helper private
    else:
        from .config import _merge_dataclass

    _merge_dataclass(cfg, cfg_dict)
    # Checkpoints that predate the optimizer field were trained with Adam.
    # A new config file without the field is deliberately AdamW by default,
    # but a historical checkpoint must retain its coupled-L2 behavior.
    checkpoint_optim = cfg_dict.get("optim")
    if isinstance(checkpoint_optim, dict) and "optimizer" not in checkpoint_optim:
        cfg.optim.optimizer = "adam"
    validate_aux_margin_config(cfg)
    validate_model_config(cfg)
    validate_optim_config(cfg.optim)
    validate_imitation_config(cfg.imitation)
    if optimizer_override is not None:
        requested_optimizer = validate_optimizer_kind(optimizer_override)
        if requested_optimizer != cfg.optim.optimizer:
            raise ValueError(
                f"resume optimizer mismatch for checkpoint {checkpoint_path}: checkpoint config records "
                f"{cfg.optim.optimizer!r}, but the resume-time override requests {requested_optimizer!r}"
            )
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
    require_native_runner_encoder_compatibility(payload, checkpoint_path)
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
        if checkpoint_arch == "card_transformer":
            configured_model_version = checkpoint_model.get("obs_version")
            model_obs_version = 2 if configured_model_version is None else int(configured_model_version)
        else:
            model_obs_version = obs_version_for_checkpoint(payload)
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
        "aux_margin_loss",
        "anchor_policy_loss",
        "anchor_value_loss",
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
        "opening_templates_enabled",
        "opening_lambda",
        "opening_p_unconstrained",
        "mean_cards_trashed_per_game",
        "opening_template_games_t0",
        "opening_template_games_t1",
        "opening_template_games_t2",
        "opening_template_games_t3",
        "opening_template_games_t4",
        "opening_template_games_t5",
        "opening_template_games_t6",
        "opening_template_vs_unconstrained_games_t1",
        "opening_template_vs_unconstrained_games_t2",
        "opening_template_vs_unconstrained_games_t3",
        "opening_template_vs_unconstrained_games_t4",
        "opening_template_vs_unconstrained_games_t5",
        "opening_template_vs_unconstrained_games_t6",
        "opening_template_win_rate_vs_unconstrained_t1",
        "opening_template_win_rate_vs_unconstrained_t2",
        "opening_template_win_rate_vs_unconstrained_t3",
        "opening_template_win_rate_vs_unconstrained_t4",
        "opening_template_win_rate_vs_unconstrained_t5",
        "opening_template_win_rate_vs_unconstrained_t6",
        "unconstrained_buys_chapel",
        "unconstrained_buys_sentry",
        "unconstrained_buys_moneylender",
        "unconstrained_buys_village",
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
    total.cards_trashed += update.cards_trashed
    for index in range(7):
        total.opening_template_games[index] += update.opening_template_games[index]
        total.opening_template_vs_unconstrained_games[index] += (
            update.opening_template_vs_unconstrained_games[index]
        )
        total.opening_template_vs_unconstrained_wins[index] += (
            update.opening_template_vs_unconstrained_wins[index]
        )
    for index in range(4):
        total.unconstrained_buy_counts[index] += update.unconstrained_buy_counts[index]
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
    selfplay_config: Any | None = None,
) -> SelfPlayStats:
    """Exact segment fallback for one-worker gated test and CPU runs."""
    total = SelfPlayStats()
    effective_selfplay = config.selfplay if selfplay_config is None else selfplay_config
    base_seed = int(config.seed) + (int(generation) * 0x9E37)
    for task_index, segment in enumerate(segments):
        seat_models = (model_table[segment.seat0_model_id], model_table[segment.seat1_model_id])
        stats = run_routed_self_play_generation(
            seat_models,
            replay,
            effective_selfplay,
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
    allowed = {"bigmoney", "engine", "engine2", "engine3", "thinner", "mcts"}
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


def _format_evals_per_second(value: float) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:.0f}k"
    return f"{value:.0f}"


def _print_selfplay_heartbeat(snapshot: dict[str, Any]) -> None:
    """One deliberately flush-safe console line for an in-flight generation."""
    message = (
        f"gen {snapshot['generation']} selfplay "
        f"{snapshot['games_done']}/{snapshot['games_total']} games, "
        f"{snapshot['recent_games_per_hour']:.0f} games/hr"
    )
    if "server_evals_per_sec" in snapshot:
        message += f", srv {_format_evals_per_second(float(snapshot['server_evals_per_sec']))} evals/s"
    print(message, flush=True)


def run_training(config: TrainConfig, resume: str | None = None, profile: bool = False) -> dict[str, Any]:
    requested = config
    if not isinstance(config.init_weights, str):
        raise ValueError("init_weights must be a checkpoint path string")
    if resume is not None and config.init_weights:
        raise ValueError("--resume and --init-weights cannot be used together")
    validate_aux_margin_config(config)
    validate_model_config(config)
    validate_optim_config(config.optim)
    validate_imitation_config(config.imitation)
    validate_value_target_config(config.selfplay)
    validate_temperature_config(config.selfplay)
    device = select_device(config.device)
    seed_everything(config.seed, deterministic=device.type == "cpu")
    resume = resolve_resume_path(resume, requested.checkpoint_dir)
    is_fresh_run = resume is None
    if resume is not None:
        config, start_generation, model, optimizer, replay = load_checkpoint(
            resume,
            device,
            optimizer_override=requested.optim.optimizer,
        )
        config.generations = requested.generations
        config.checkpoint_dir = requested.checkpoint_dir
        config.metrics_csv = requested.metrics_csv
        config.device = requested.device
        config.parallel_workers = requested.parallel_workers
        config.worker_device = requested.worker_device
        config.server_selfplay = requested.server_selfplay
        config.server_device = requested.server_device
        config.server_max_batch = requested.server_max_batch
        config.server_coalesce_target_rows = requested.server_coalesce_target_rows
        config.server_coalesce_ms = requested.server_coalesce_ms
        config.server_max_wait_ms = requested.server_max_wait_ms
        config.server_fp16 = requested.server_fp16
        config.server_compile = requested.server_compile
        config.server_autocast_bf16 = requested.server_autocast_bf16
        config.server_batch_buckets = requested.server_batch_buckets
        config.server_response_timeout_s = requested.server_response_timeout_s
        config.server_install_timeout_s = requested.server_install_timeout_s
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
        validate_aux_margin_config(config)
        validate_model_config(config)
        validate_imitation_config(config.imitation)
        validate_value_target_config(config.selfplay)
        validate_temperature_config(config.selfplay)
        if config.device == "auto":
            config.device = device.type
    else:
        model, optimizer, replay = build_objects(config, device)
        if config.init_weights:
            load_initial_weights(config.init_weights, config, model, device)
        start_generation = 0

    human_dataset: HumanTupleDataset | None = None
    anchor_batches: Iterator[HumanBatch] | None = None

    def ensure_human_dataset() -> HumanTupleDataset:
        nonlocal human_dataset
        if human_dataset is None:
            human_dataset = load_human_dataset_for_config(config)
        return human_dataset

    # Behavior cloning happens only for a new campaign.  A resumed optimizer
    # must continue from its checkpoint rather than replaying the pretraining
    # phase, even if the checkpoint's imitation config remains enabled.
    if is_fresh_run and config.imitation.pretrain_steps > 0:
        pretrain_batches = ensure_human_dataset().minibatches(
            config.imitation.pretrain_batch_size,
            int(config.seed) ^ 0x4855_4D41_4E,
        )
        run_human_pretrain(
            model,
            optimizer,
            pretrain_batches,
            steps=config.imitation.pretrain_steps,
            device=device,
            pretrain_lr=config.imitation.pretrain_lr,
        )

    validate_determinize_config(config.selfplay)
    validate_temperature_config(config.selfplay)
    validate_deep_slice_config(config.selfplay)
    validate_forced_playouts_config(config.selfplay)
    validate_opening_template_config(config.selfplay)
    if not isinstance(config.server_selfplay, bool):
        raise ValueError("server_selfplay must be a boolean")
    shared_server_selfplay = (
        config.server_selfplay or config.worker_device.lower() == "server"
    )
    if config.server_selfplay and config.parallel_workers <= 1:
        raise ValueError("server_selfplay requires parallel_workers greater than one")

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
    progress = TrainingProgress(config.checkpoint_dir)
    use_gating = gating_enabled(config)
    if not isinstance(config.scripted_opponents, dict):
        raise ValueError("scripted_opponents must be an object mapping kind to fraction")
    effective_scripted_fractions(config.scripted_opponent_schedule, config.scripted_opponents, start_generation)
    effective_league_fraction(config.league_schedule, config.league_fraction, start_generation)
    anchor_weight_for_generation(config.imitation, start_generation)
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
        league_load_device = torch.device("cpu") if shared_server_selfplay else device
        seed_league_checkpoints(config, league_load_device)
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
        progress.start(_print_selfplay_heartbeat)
        if config.parallel_workers > 1 and shared_server_selfplay:
            inference_server = InferenceServer(config, config.parallel_workers)
        if config.parallel_workers > 1:
            pool = ParallelSelfPlayPool(config, inference_server)
        for generation in range(start_generation + 1, start_generation + generations + 1):
            gen_start = time.perf_counter()
            progress.transition(
                generation,
                "selfplay",
                games_total=config.selfplay.games_per_generation,
                server_mode=inference_server is not None,
            )

            def report_selfplay_progress(games_done: int, positions_done: int) -> None:
                server_evals = (
                    inference_server.drain_telemetry()
                    if inference_server is not None
                    else None
                )
                progress.update(
                    games_done=games_done,
                    positions_done=positions_done,
                    server_evals_per_sec=server_evals,
                )

            lr = learning_rate_for_generation(config, generation)
            set_optimizer_lr(optimizer, lr)
            generation_selfplay = scheduled_opening_selfplay_config(config.selfplay, generation)
            if config.selfplay.opening_templates_enabled:
                opening_lambda, opening_p_unconstrained = opening_template_schedule(
                    config.selfplay,
                    generation,
                )
            else:
                opening_lambda, opening_p_unconstrained = 0.0, 1.0
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
            effective_anchor_weight = anchor_weight_for_generation(config.imitation, generation)
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
                    sp_stats = run_self_play_generation(
                        model,
                        replay,
                        generation_selfplay,
                        gen_seed,
                        device,
                    )
                    aggregate_games_per_hour = sp_stats.games_per_hour
                else:
                    parallel_result = pool.generate(
                        model,
                        replay,
                        generation,
                        selfplay_config=generation_selfplay,
                        progress_callback=report_selfplay_progress,
                    )
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
                if pool is None:
                    model_table = [active_model, *[load_best_checkpoint(config, device, path)[0] for path in league_paths]]
                    sp_stats = _run_segmented_single_pipeline(
                        model_table,
                        segments,
                        replay,
                        config,
                        generation,
                        device,
                        selfplay_config=generation_selfplay,
                    )
                    aggregate_games_per_hour = sp_stats.games_per_hour
                else:
                    payloads = [
                        (
                            serialize_cpu_state_dict(active_model),
                            model_config_dict(
                                getattr(active_model, "_dominion_model_config", config.model)
                            ),
                        )
                    ]
                    opponent_device = torch.device("cpu") if inference_server is not None else device
                    for path in league_paths:
                        opponent, _ = load_best_checkpoint(config, opponent_device, path)
                        payloads.append(
                            (
                                serialize_cpu_state_dict(opponent),
                                model_config_dict(
                                    getattr(opponent, "_dominion_model_config", config.model)
                                ),
                            )
                        )
                    parallel_result = pool.generate(
                        active_model,
                        replay,
                        generation,
                        segments=segments,
                        model_state_payloads=payloads,
                        selfplay_config=generation_selfplay,
                        progress_callback=report_selfplay_progress,
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
            progress.update(
                games_done=sp_stats.games,
                positions_done=sp_stats.positions,
                server_evals_per_sec=server_metrics.get("server_evals_per_sec"),
                force=True,
            )
            progress.transition(
                generation,
                "train",
                games_total=config.selfplay.games_per_generation,
                games_done=sp_stats.games,
                positions_done=sp_stats.positions,
                recent_games_per_hour=aggregate_games_per_hour,
                server_mode=inference_server is not None,
                server_evals_per_sec=server_metrics.get("server_evals_per_sec"),
            )

            losses = {
                "policy_loss": float("nan"),
                "value_loss": float("nan"),
                "aux_margin_loss": float("nan"),
                "anchor_policy_loss": float("nan"),
                "anchor_value_loss": float("nan"),
                "entropy": float("nan"),
            }
            steps = config.optim.train_steps_per_generation
            if len(replay) > 0 and steps > 0:
                anchor_active = effective_anchor_weight > 0.0
                aux_active = float(config.aux_margin_weight) > 0.0
                if anchor_active and anchor_batches is None:
                    anchor_batches = ensure_human_dataset().minibatches(
                        config.imitation.anchor_batch_size,
                        int(config.seed) ^ 0x414E_4348_4F52,
                    )
                accum = {
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "aux_margin_loss": 0.0,
                    "anchor_policy_loss": 0.0,
                    "anchor_value_loss": 0.0,
                    "entropy": 0.0,
                }
                for _ in range(steps):
                    step_losses = train_step(
                        model,
                        optimizer,
                        replay,
                        config.optim.batch_size,
                        device,
                        human_batches=anchor_batches if anchor_active else None,
                        anchor_weight=effective_anchor_weight if anchor_active else 0.0,
                        anchor_awr_beta=config.imitation.anchor_awr_beta,
                        aux_margin_weight=config.aux_margin_weight if aux_active else 0.0,
                    )
                    for key in ("policy_loss", "value_loss", "entropy"):
                        accum[key] += step_losses[key]
                    if aux_active:
                        accum["aux_margin_loss"] += step_losses["aux_margin_loss"]
                    if anchor_active:
                        accum["anchor_policy_loss"] += step_losses["anchor_policy_loss"]
                        accum["anchor_value_loss"] += step_losses["anchor_value_loss"]
                losses = {key: value / steps for key, value in accum.items()}
                if not anchor_active:
                    # NaN is this module's existing disabled/inactive metric
                    # convention (the same value used when no train steps run).
                    losses["anchor_policy_loss"] = float("nan")
                    losses["anchor_value_loss"] = float("nan")
                if not aux_active:
                    losses["aux_margin_loss"] = float("nan")

            gate_row: dict[str, Any] = {}
            if use_gating:
                progress.transition(
                    generation,
                    "gate",
                    games_total=config.selfplay.games_per_generation,
                    games_done=sp_stats.games,
                    positions_done=sp_stats.positions,
                    recent_games_per_hour=aggregate_games_per_hour,
                    server_mode=inference_server is not None,
                    server_evals_per_sec=server_metrics.get("server_evals_per_sec"),
                )
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

            progress.transition(
                generation,
                "checkpoint",
                games_total=config.selfplay.games_per_generation,
                games_done=sp_stats.games,
                positions_done=sp_stats.positions,
                recent_games_per_hour=aggregate_games_per_hour,
                server_mode=inference_server is not None,
                server_evals_per_sec=server_metrics.get("server_evals_per_sec"),
            )
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
                progress.transition(
                    generation,
                    "eval",
                    games_total=config.selfplay.games_per_generation,
                    games_done=sp_stats.games,
                    positions_done=sp_stats.positions,
                    recent_games_per_hour=aggregate_games_per_hour,
                    server_mode=inference_server is not None,
                    server_evals_per_sec=server_metrics.get("server_evals_per_sec"),
                )
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
                    honest=config.eval.eval_honest,
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
                        honest=config.eval.eval_honest,
                    )
                    eval_row[f"sentinel_{opponent}_wins"] = sentinel_stats.wins
                    eval_row[f"sentinel_{opponent}_games"] = sentinel_stats.games
            row = {
                "generation": generation,
                "games": sp_stats.games,
                "positions": sp_stats.positions,
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "aux_margin_loss": losses["aux_margin_loss"],
                "anchor_policy_loss": losses["anchor_policy_loss"],
                "anchor_value_loss": losses["anchor_value_loss"],
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
                "opening_templates_enabled": config.selfplay.opening_templates_enabled,
                "opening_lambda": opening_lambda,
                "opening_p_unconstrained": opening_p_unconstrained,
                "mean_cards_trashed_per_game": (
                    float(sp_stats.cards_trashed) / float(sp_stats.games)
                    if sp_stats.games > 0
                    else 0.0
                ),
                "unconstrained_buys_chapel": sp_stats.unconstrained_buy_counts[0],
                "unconstrained_buys_sentry": sp_stats.unconstrained_buy_counts[1],
                "unconstrained_buys_moneylender": sp_stats.unconstrained_buy_counts[2],
                "unconstrained_buys_village": sp_stats.unconstrained_buy_counts[3],
            }
            for template_id in range(7):
                row[f"opening_template_games_t{template_id}"] = sp_stats.opening_template_games[template_id]
            for template_id in range(1, 7):
                matchup_games = sp_stats.opening_template_vs_unconstrained_games[template_id]
                matchup_wins = sp_stats.opening_template_vs_unconstrained_wins[template_id]
                row[f"opening_template_vs_unconstrained_games_t{template_id}"] = matchup_games
                row[f"opening_template_win_rate_vs_unconstrained_t{template_id}"] = (
                    float(matchup_wins) / float(matchup_games) if matchup_games > 0 else 0.0
                )
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
        progress.close()

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
