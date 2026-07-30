"""Observation-version helpers shared by v2 training and evaluation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from src.v2.encoder_compat import (
    LEGACY_ENCODER_GENERATION,
    restore_legacy_constants_for_observation_version,
)


# Keep these literals local instead of importing the optional native binding:
# model loading and the pure downgrade helper are also used by checkpoint-only
# tools.  They mirror ``encode/encoder.h``.
OBS_SIZE_V1 = 1141
OBS_SIZE_V2 = 1717
OBS_SIZE_V3 = 1788


def _bindings() -> Any:
    # Keep this import lazy so persistence-only helpers remain usable without
    # requiring the native extension at module-import time.
    import dominion_v2_py as dz

    return dz


def obs_size_for_version(version: int) -> int:
    """Return the native fixed observation width for a stored version."""
    return int(_bindings().obs_size_for(int(version)))


def obs_size_for_config(config: Any) -> int:
    return obs_size_for_version(int(config.selfplay.obs_version))


def obs_version_for_width(width: int) -> int:
    """Return the protocol version represented by a fixed observation width."""
    sizes = {
        OBS_SIZE_V1: 1,
        OBS_SIZE_V2: 2,
        OBS_SIZE_V3: 3,
    }
    try:
        return sizes[int(width)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unsupported observation width {width!r}") from exc


def downgrade_v3_observations(observations: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Return the exact v2 encoding represented by a batch of v3 observations.

    The v3 encoder appends data after the byte-identical v2 prefix.  Its two
    metadata scalars identify the enclosing layout, so the output needs one
    output allocation for the sliced prefix and then only those scalars are
    rewritten.  The source batch is never mutated.
    """
    if isinstance(observations, np.ndarray):
        if not np.issubdtype(observations.dtype, np.floating):
            raise TypeError("v3 observations must use a floating NumPy dtype")
        if observations.ndim < 1 or observations.shape[-1] != OBS_SIZE_V3:
            raise ValueError(f"v3 observations must end in width {OBS_SIZE_V3}")
        downgraded = observations[..., :OBS_SIZE_V2].copy()
        downgraded[..., 0] = 2.0
        downgraded[..., 1] = float(OBS_SIZE_V2)
        return downgraded
    if isinstance(observations, torch.Tensor):
        if not observations.is_floating_point():
            raise TypeError("v3 observations must use a floating Torch dtype")
        if observations.ndim < 1 or observations.shape[-1] != OBS_SIZE_V3:
            raise ValueError(f"v3 observations must end in width {OBS_SIZE_V3}")
        downgraded = observations[..., :OBS_SIZE_V2].clone()
        downgraded[..., 0] = 2.0
        downgraded[..., 1] = float(OBS_SIZE_V2)
        return downgraded
    raise TypeError("v3 observations must be a NumPy array or Torch tensor")


def observations_for_model(
    observations: np.ndarray | torch.Tensor,
    source_version: int,
    model_version: int,
    encoder_generation: int = 2,
) -> np.ndarray | torch.Tensor:
    """Adapt a runner batch for one model's known observation protocol.

    There is intentionally one compatibility direction: a v3 runner can
    serve a v2 league model by exact downgrade.  V1 is not a prefix of v2 and
    no model receives an implicit upgrade.
    """
    source = int(source_version)
    target = int(model_version)
    if source == target:
        adapted = observations
    elif source == 3 and target == 2:
        adapted = downgrade_v3_observations(observations)
    else:
        raise ValueError(
            f"cannot serve obs-v{source} observations to an obs-v{target} model; "
            "only the exact v3-to-v2 downgrade is supported"
        )
    if int(encoder_generation) == LEGACY_ENCODER_GENERATION:
        # This is intentionally after a possible v3->v2 slice: each model
        # receives generation-1 constants at positions in its own input ABI.
        return restore_legacy_constants_for_observation_version(adapted, target)
    return adapted


def model_observation_version(model: Any, fallback_version: int) -> int:
    """Read a model's serialized protocol tag, with safe legacy fallbacks."""
    config = getattr(model, "_dominion_model_config", None)
    if isinstance(config, Mapping) and config.get("obs_version") is not None:
        raw_version = config["obs_version"]
    else:
        raw_version = getattr(model, "obs_version", fallback_version)
    if not isinstance(raw_version, int) or isinstance(raw_version, bool) or raw_version not in (1, 2, 3):
        raise ValueError(f"model has an invalid obs_version {raw_version!r}")
    return int(raw_version)


def model_encoder_generation(model: Any, fallback_generation: int = 2) -> int:
    """Read model routing metadata, treating live training models as generation 2."""
    raw_generation = getattr(model, "_dominion_encoder_generation", None)
    if raw_generation is None:
        config = getattr(model, "_dominion_model_config", None)
        if isinstance(config, Mapping):
            raw_generation = config.get("encoder_generation")
    if raw_generation is None:
        raw_generation = fallback_generation
    if (
        not isinstance(raw_generation, int)
        or isinstance(raw_generation, bool)
        or raw_generation < 1
    ):
        raise ValueError(f"model has an invalid encoder_generation {raw_generation!r}")
    return int(raw_generation)


def obs_version_for_checkpoint(payload: Mapping[str, Any]) -> int:
    """Infer an observation protocol version from the saved model input width.

    Checkpoint configs are advisory: legacy checkpoints do not contain an
    ``obs_version`` field. The first trunk linear layer is the authoritative
    stored input shape for every DominionNet checkpoint.
    """
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("checkpoint is missing a model state dict")

    weight = model.get("trunk.0.weight")
    if weight is None:
        for name, candidate in model.items():
            if str(name).startswith("trunk.") and str(name).endswith(".weight"):
                weight = candidate
                break
    # DominionNet also permits an empty hidden_sizes list. In that shape the
    # policy head is directly connected to the observation input.
    if weight is None:
        weight = model.get("policy_head.weight")
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("checkpoint model is missing a two-dimensional first trunk weight")
    input_size = int(shape[1])

    dz = _bindings()
    if input_size == int(dz.OBS_SIZE_V1):
        return 1
    if input_size == int(dz.OBS_SIZE_V2):
        return 2
    if input_size == int(dz.OBS_SIZE_V3):
        return 3
    raise ValueError(
        f"checkpoint model input size {input_size} is not a supported observation layout "
        f"({int(dz.OBS_SIZE_V1)}, {int(dz.OBS_SIZE_V2)}, or {int(dz.OBS_SIZE_V3)})"
    )
