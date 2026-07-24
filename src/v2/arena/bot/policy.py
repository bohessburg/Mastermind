"""Shared local neural-network policy serving.

Both the interactive web server and arena runtime use this module so their
checkpoint reconstruction and parked-leaf NN-MCTS decisions stay identical.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import dominion_v2_py as dz


@dataclass
class NNPolicy:
    """A policy/value model and the lazily imported Torch module that runs it."""

    model: Any
    torch: Any
    obs_version: int
    device: str


class NNCheckpointError(Exception):
    """A safe, user-facing failure while preparing an NN policy."""


def load_policy(
    checkpoint_path: Path,
    *,
    obs_version: int | None = None,
    device: str = "cpu",
) -> NNPolicy:
    """Load a local training checkpoint for policy serving.

    ``obs_version`` may validate the expected checkpoint observation layout;
    leaving it unset preserves the historical checkpoint-metadata inference.
    """
    if not checkpoint_path.is_file():
        raise NNCheckpointError("neural-network checkpoint is unavailable")

    try:
        import torch
        from src.v2.train.model import build_model
    except ImportError as error:
        raise NNCheckpointError("neural-network bot requires PyTorch") from error

    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:  # pragma: no cover - older supported Torch versions
            checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]
        if not isinstance(config, dict):
            raise TypeError("checkpoint config must be a dict")
        model_config = config["model"]
        if not isinstance(model_config, dict):
            raise TypeError("checkpoint model config must be a dict")
        arch = model_config.get("arch", "mlp")

        selfplay_config = config.get("selfplay")
        if isinstance(selfplay_config, dict) and "obs_version" in selfplay_config:
            checkpoint_obs_version = int(selfplay_config["obs_version"])
            if checkpoint_obs_version not in (1, 2):
                raise ValueError("checkpoint selfplay obs_version must be 1 or 2")
        elif arch == "card_transformer":
            # A CardTokenNet is structurally v2 even if a hand-written
            # checkpoint omitted the self-play metadata.
            checkpoint_obs_version = 2
        else:
            model_state = checkpoint["model"]
            if not isinstance(model_state, dict):
                raise TypeError("checkpoint model state must be a dict")
            weight = model_state.get("trunk.0.weight")
            if weight is None:
                for name, candidate in model_state.items():
                    if str(name).startswith("trunk.") and str(name).endswith(".weight"):
                        weight = candidate
                        break
            if weight is None:
                weight = model_state.get("policy_head.weight")
            shape = getattr(weight, "shape", None)
            if shape is None or len(shape) != 2:
                raise ValueError("checkpoint model is missing its input linear layer")
            input_width = int(shape[1])
            if input_width == int(dz.OBS_SIZE_V1):
                checkpoint_obs_version = 1
            elif input_width == int(dz.OBS_SIZE_V2):
                checkpoint_obs_version = 2
            else:
                raise ValueError("checkpoint model input size is not a supported observation layout")

        if obs_version is not None:
            if obs_version not in (1, 2):
                raise ValueError("obs_version must be 1 or 2")
            if obs_version != checkpoint_obs_version:
                raise ValueError("checkpoint observation layout does not match requested obs_version")
        else:
            obs_version = checkpoint_obs_version

        obs_size = int(dz.OBS_SIZE_V1 if obs_version == 1 else dz.OBS_SIZE_V2)
        model = build_model(model_config, obs_size, int(dz.ACTION_SPACE_SIZE))
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()
    except Exception as error:
        raise NNCheckpointError("neural-network checkpoint could not be loaded") from error

    return NNPolicy(model=model, torch=torch, obs_version=obs_version, device=device)


def choose_nn_action(game: Any, seat: int, policy: NNPolicy) -> int:
    """Return the policy network's greedy legal action for one decision."""
    torch = policy.torch
    observation = torch.as_tensor(
        game.encode(seat, policy.obs_version), dtype=torch.float32, device=policy.device
    ).unsqueeze(0)
    legal_mask = torch.as_tensor(game.legal_mask(), dtype=torch.bool, device=policy.device).unsqueeze(0)
    masked_logits, _ = policy.model.evaluate(observation, legal_mask)
    return int(torch.argmax(masked_logits, dim=-1).item())


def choose_nnmcts_action(
    game: Any,
    seat: int,
    policy: NNPolicy,
    *,
    sims: int = 400,
    determinizations: int = 2,
    wall_clock_cap: float | None = None,
) -> int:
    """Run one parked-leaf NN-MCTS decision and return a legal action.

    A cap is optional so legacy callers retain their exact simulation-budget
    behavior.  When a caller supplies one, it may stop after a completed
    inference batch and falls back to the first legal action.
    """
    torch = policy.torch
    searcher = dz.DecisionSearcher(
        game,
        seat,
        {
            "sims": sims,
            "c_puct": 1.25,
            "determinizations": determinizations,
            "obs_version": policy.obs_version,
            # Collapse-trained checkpoints (c7+) never search treasure plays;
            # searching them here puts the net off-distribution (see the
            # 2026-07-11 eval-flag bug in docs/training-log.md).
            "auto_play_treasures": True,
            "prune_treasure_plays": True,
            # State-derived seeding also makes replay/undo decisions stable.
            "seed": (
                int(game.state_hash()) ^ ((seat + 1) * 0x9E3779B97F4A7C15)
            )
            & 0xFFFFFFFFFFFFFFFF,
        },
    )
    started_at = time.monotonic()
    while not searcher.done():
        obs, masks = searcher.collect_leaves()
        if obs.shape[0] == 0:
            continue
        with torch.no_grad():
            logits, values = policy.model.evaluate(
                torch.as_tensor(obs, dtype=torch.float32, device=policy.device),
                torch.as_tensor(masks, dtype=torch.bool, device=policy.device),
            )
        searcher.provide_evaluations(
            values.detach().cpu().numpy().astype(np.float32, copy=False),
            logits.detach().cpu().numpy().astype(np.float32, copy=False),
        )
        if wall_clock_cap is not None and time.monotonic() - started_at >= wall_clock_cap:
            return _first_legal_action(game)
    action = int(searcher.best_action())
    return action if _is_legal(game, action) else _first_legal_action(game)


def _is_legal(game: Any, action: int) -> bool:
    return 0 <= action < int(dz.ACTION_SPACE_SIZE) and bool(game.legal_mask()[action])


def _first_legal_action(game: Any) -> int:
    legal = np.flatnonzero(game.legal_mask())
    return int(legal[0]) if legal.size else int(dz.A_PASS)
