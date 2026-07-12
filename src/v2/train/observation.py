"""Observation-version helpers shared by v2 training and evaluation."""

from __future__ import annotations

from typing import Any, Mapping


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
    raise ValueError(
        f"checkpoint model input size {input_size} is not a supported observation layout "
        f"({int(dz.OBS_SIZE_V1)} or {int(dz.OBS_SIZE_V2)})"
    )
