"""Shared local neural-network policy serving.

Both the interactive web server and arena runtime use this module so their
checkpoint reconstruction and parked-leaf NN-MCTS decisions stay identical.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

import dominion_v2_py as dz
from src.v2.encoder_compat import (
    LEGACY_ENCODER_GENERATION,
    checkpoint_encoder_generation,
    generation_mismatch_message,
)


@dataclass
class NNPolicy:
    """A policy/value model and the lazily imported Torch module that runs it."""

    model: Any
    torch: Any
    obs_version: int
    device: str
    encoder_generation: int
    obs_transform: Callable[[Any], Any] | None = None

    def transform_observations(self, observations: Any) -> Any:
        """Apply the policy's checkpoint-specific input compatibility layer."""
        return observations if self.obs_transform is None else self.obs_transform(observations)

    def evaluate(self, observations: Any, legal_masks: Any) -> tuple[Any, Any]:
        """Evaluate after applying the checkpoint-specific observation transform."""
        return self.model.evaluate(self.transform_observations(observations), legal_masks)


class NNCheckpointError(Exception):
    """A safe, user-facing failure while preparing an NN policy."""


@dataclass(frozen=True)
class _LegacyShimLayout:
    """Native encoder.h-derived positions needed to restore generation-1 inputs."""

    observation_size: int
    supply_offset: int
    supply_size: int
    pile_block_size: int
    pile_count_field: int
    pile_base_field: int
    pile_trait_field: int
    landscape_offset: int
    landscape_id_size: int
    landscape_prophecy_offset: int


def _legacy_shim_layout(obs_version: int) -> _LegacyShimLayout:
    """Read the native layout rather than duplicating encoder.h offset arithmetic."""
    raw = dz.encoder_layout(int(obs_version))
    try:
        return _LegacyShimLayout(
            observation_size=int(dz.obs_size_for(int(obs_version))),
            supply_offset=int(raw["supply_offset"]),
            supply_size=int(raw["supply_size"]),
            pile_block_size=int(raw["pile_block_size"]),
            pile_count_field=int(raw["pile_count_field"]),
            pile_base_field=int(raw["pile_base_field"]),
            pile_trait_field=int(raw["pile_trait_field"]),
            landscape_offset=int(raw["landscape_offset"]),
            landscape_id_size=int(raw["landscape_id_size"]),
            landscape_prophecy_offset=int(raw["landscape_prophecy_offset"]),
        )
    except (KeyError, TypeError, ValueError) as error:  # pragma: no cover - native ABI invariant
        raise RuntimeError("native encoder layout metadata is invalid") from error


def _validate_legacy_shim_observations(observations: Any, layout: _LegacyShimLayout) -> None:
    if getattr(observations, "ndim", 0) < 1 or int(observations.shape[-1]) != layout.observation_size:
        raise ValueError(
            "legacy encoder shim expected observations ending in width "
            f"{layout.observation_size}"
        )


def _restore_legacy_constants(observations: Any, layout: _LegacyShimLayout) -> Any:
    """Copy observations and recreate generation-1 sentinel encodings.

    The native layout identifies both the v1 offsets and the v2/v3 shared
    prefix.  Only structurally populated supply rows receive the old trait
    constant; unused rows were zero in both encoder generations.
    """
    _validate_legacy_shim_observations(observations, layout)
    supply_stop = layout.supply_offset + layout.supply_size
    landscape_ids = slice(layout.landscape_offset, layout.landscape_offset + layout.landscape_id_size)
    prophecy_index = layout.landscape_offset + layout.landscape_prophecy_offset

    if isinstance(observations, np.ndarray):
        if not np.issubdtype(observations.dtype, np.floating):
            raise TypeError("legacy encoder shim requires floating NumPy observations")
        restored = observations.copy()
        restored[..., landscape_ids] = 1.0
        restored[..., prophecy_index] = 1.0
        piles = restored[..., layout.supply_offset:supply_stop].reshape(
            *restored.shape[:-1], -1, layout.pile_block_size
        )
        populated = (piles[..., layout.pile_count_field] != 0.0) | (
            piles[..., layout.pile_base_field] != 0.0
        )
        traits = piles[..., layout.pile_trait_field]
        traits[populated] = 1.0
        return restored

    if not bool(getattr(observations, "is_floating_point", lambda: False)()):
        raise TypeError("legacy encoder shim requires floating Torch observations")
    restored = observations.clone()
    restored[..., landscape_ids] = 1.0
    restored[..., prophecy_index] = 1.0
    piles = restored[..., layout.supply_offset:supply_stop].reshape(
        *restored.shape[:-1], -1, layout.pile_block_size
    )
    populated = (piles[..., layout.pile_count_field] != 0.0) | (
        piles[..., layout.pile_base_field] != 0.0
    )
    traits = piles[..., layout.pile_trait_field]
    traits[populated] = 1.0
    return restored


def _legacy_obs_transform(obs_version: int) -> Callable[[Any], Any]:
    layout = _legacy_shim_layout(obs_version)
    return lambda observations: _restore_legacy_constants(observations, layout)


def load_policy(
    checkpoint_path: Path,
    *,
    obs_version: int | None = None,
    device: str = "cpu",
    legacy_shim: bool = False,
) -> NNPolicy:
    """Load a local training checkpoint for policy serving.

    ``obs_version`` may validate the expected checkpoint observation layout;
    leaving it unset preserves the historical checkpoint-metadata inference.
    ``legacy_shim`` opt-ins to the only known cross-generation adapter:
    generation-1 checkpoint inputs on the generation-2 sentinel-fixed engine.
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
        if not isinstance(checkpoint, dict):
            raise TypeError("checkpoint must be a dict")
        checkpoint_generation = checkpoint_encoder_generation(checkpoint)
        runtime_generation = int(dz.ENCODER_GENERATION)
        use_legacy_shim = checkpoint_generation != runtime_generation
        if use_legacy_shim:
            mismatch = generation_mismatch_message(
                checkpoint_path, checkpoint_generation, runtime_generation
            )
            if not legacy_shim:
                raise NNCheckpointError(
                    f"{mismatch}; pass legacy_shim=True to serve a generation-1 checkpoint "
                    "with restored pre-sentinel-fix constants"
                )
            if not (
                checkpoint_generation == LEGACY_ENCODER_GENERATION and runtime_generation == 2
            ):
                raise NNCheckpointError(
                    f"{mismatch}; legacy_shim only supports generation-1 checkpoints on a "
                    "generation-2 engine"
                )
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
            if checkpoint_obs_version not in (1, 2, 3):
                raise ValueError("checkpoint selfplay obs_version must be 1, 2, or 3")
        elif arch == "card_transformer":
            # Pre-v3 CardTokenNet checkpoints omitted layout metadata and are
            # structurally v2. New checkpoints record their tokenizer layout.
            configured_version = model_config.get("obs_version", 2)
            checkpoint_obs_version = 2 if configured_version is None else int(configured_version)
            if checkpoint_obs_version not in (2, 3):
                raise ValueError("card_transformer model obs_version must be 2 or 3")
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
            elif input_width == int(dz.OBS_SIZE_V3):
                checkpoint_obs_version = 3
            else:
                raise ValueError("checkpoint model input size is not a supported observation layout")

        if obs_version is not None:
            if obs_version not in (1, 2, 3):
                raise ValueError("obs_version must be 1, 2, or 3")
            if obs_version != checkpoint_obs_version:
                raise ValueError("checkpoint observation layout does not match requested obs_version")
        else:
            obs_version = checkpoint_obs_version

        obs_size = int(dz.obs_size_for(obs_version))
        model = build_model(model_config, obs_size, int(dz.ACTION_SPACE_SIZE))
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()
    except NNCheckpointError:
        raise
    except Exception as error:
        raise NNCheckpointError("neural-network checkpoint could not be loaded") from error

    return NNPolicy(
        model=model,
        torch=torch,
        obs_version=obs_version,
        device=device,
        encoder_generation=checkpoint_generation,
        obs_transform=_legacy_obs_transform(obs_version) if use_legacy_shim else None,
    )


def choose_nn_action(game: Any, seat: int, policy: NNPolicy) -> int:
    """Return the policy network's greedy legal action for one decision."""
    torch = policy.torch
    observation = torch.as_tensor(
        game.encode(seat, policy.obs_version), dtype=torch.float32, device=policy.device
    ).unsqueeze(0)
    legal_mask = torch.as_tensor(game.legal_mask(), dtype=torch.bool, device=policy.device).unsqueeze(0)
    masked_logits, _ = policy.evaluate(observation, legal_mask)
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
            logits, values = policy.evaluate(
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
