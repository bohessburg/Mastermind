"""Checkpoint compatibility checks for changes to native encoder semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

LEGACY_ENCODER_GENERATION = 1


class EncoderGenerationError(ValueError):
    """A checkpoint was produced by a semantically different encoder."""


@dataclass(frozen=True)
class LegacyConstantLayout:
    """Native encoder positions whose generation-1 sentinel was constant 1."""

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


def _runtime_encoder_generation() -> int:
    # Keep payload-only validation usable in tools that intentionally do not
    # load the optional native extension.
    import dominion_v2_py as dz

    return int(dz.ENCODER_GENERATION)


def legacy_constant_layout(obs_version: int) -> LegacyConstantLayout:
    """Read native layout metadata used to restore generation-1 sentinels."""
    import dominion_v2_py as dz

    raw = dz.encoder_layout(int(obs_version))
    try:
        return LegacyConstantLayout(
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


def restore_legacy_constants(
    observations: Any,
    layout: LegacyConstantLayout,
) -> Any:
    """Copy observations and recreate generation-1 landscape/trait constants.

    Only structurally populated supply rows receive the old trait sentinel;
    unused rows were zero under both encoder generations.
    """
    if getattr(observations, "ndim", 0) < 1 or int(observations.shape[-1]) != layout.observation_size:
        raise ValueError(
            "legacy encoder shim expected observations ending in width "
            f"{layout.observation_size}"
        )
    supply_stop = layout.supply_offset + layout.supply_size
    landscape_ids = slice(layout.landscape_offset, layout.landscape_offset + layout.landscape_id_size)
    prophecy_index = layout.landscape_offset + layout.landscape_prophecy_offset

    if isinstance(observations, np.ndarray):
        if not np.issubdtype(observations.dtype, np.floating):
            raise TypeError("legacy encoder shim requires floating NumPy observations")
        restored = observations.copy()
    else:
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


def restore_legacy_constants_for_observation_version(observations: Any, obs_version: int) -> Any:
    """Restore generation-1 constants using the native layout for ``obs_version``."""
    return restore_legacy_constants(observations, legacy_constant_layout(obs_version))


def checkpoint_encoder_generation(payload: Mapping[str, Any]) -> int:
    """Return a checkpoint's encoder generation, treating unstamped files as v1."""
    generation = payload.get("encoder_generation", LEGACY_ENCODER_GENERATION)
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise ValueError(f"checkpoint encoder_generation must be a positive integer, got {generation!r}")
    return int(generation)


def generation_mismatch_message(
    checkpoint: str | Path,
    checkpoint_generation: int,
    runtime_generation: int | None = None,
) -> str:
    """Format one clear diagnostic shared by serving and native runners."""
    runtime = _runtime_encoder_generation() if runtime_generation is None else int(runtime_generation)
    return (
        f"checkpoint {checkpoint} uses encoder generation {int(checkpoint_generation)}, "
        f"but this build uses encoder generation {runtime}"
    )


def require_native_runner_encoder_compatibility(
    payload: Mapping[str, Any],
    checkpoint: str | Path,
    *,
    legacy_shim: bool = False,
) -> int:
    """Reject generation mismatches for runners that evaluate native buffers.

    EvalRunner and SelfPlayRunner feed their native encoded batches directly to
    the model, so Python cannot install the serving shim between encoder and
    inference.  A legacy checkpoint therefore needs a generation-1 build.
    """
    checkpoint_generation = checkpoint_encoder_generation(payload)
    runtime_generation = _runtime_encoder_generation()
    if checkpoint_generation == runtime_generation:
        return checkpoint_generation
    shim_note = " even with --legacy-shim" if legacy_shim else ""
    raise EncoderGenerationError(
        f"{generation_mismatch_message(checkpoint, checkpoint_generation, runtime_generation)}; "
        f"native-runner evaluation cannot apply a Python legacy shim{shim_note}. "
        "Evaluate this checkpoint with a pre-sentinel-fix encoder-generation-1 build."
    )


def require_league_encoder_compatibility(
    payload: Mapping[str, Any],
    checkpoint: str | Path,
) -> int:
    """Allow the routed generation-1-on-generation-2 league compatibility path.

    Unlike native runners, league leaf evaluation passes through Python before
    a model sees it, so it can restore the old constants for generation-1
    ancestors. Other generation mismatches remain unsafe and are rejected.
    """
    checkpoint_generation = checkpoint_encoder_generation(payload)
    runtime_generation = _runtime_encoder_generation()
    if checkpoint_generation == runtime_generation:
        return checkpoint_generation
    if checkpoint_generation == LEGACY_ENCODER_GENERATION and runtime_generation == 2:
        return checkpoint_generation
    raise EncoderGenerationError(
        f"{generation_mismatch_message(checkpoint, checkpoint_generation, runtime_generation)}; "
        "league routing supports only generation-1 checkpoints on a generation-2 engine"
    )
