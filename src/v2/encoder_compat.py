"""Checkpoint compatibility checks for changes to native encoder semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

LEGACY_ENCODER_GENERATION = 1


class EncoderGenerationError(ValueError):
    """A checkpoint was produced by a semantically different encoder."""


def _runtime_encoder_generation() -> int:
    # Keep payload-only validation usable in tools that intentionally do not
    # load the optional native extension.
    import dominion_v2_py as dz

    return int(dz.ENCODER_GENERATION)


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
