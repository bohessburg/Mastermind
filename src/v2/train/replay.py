from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ReplayBatch:
    obs: np.ndarray
    policy: np.ndarray
    value: np.ndarray
    legal_mask: np.ndarray
    margin: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, obs_size: int, action_size: int, seed: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.obs_size = int(obs_size)
        self.action_size = int(action_size)
        self.obs = np.zeros((self.capacity, self.obs_size), dtype=np.float32)
        self.policy = np.zeros((self.capacity, self.action_size), dtype=np.float32)
        self.value = np.zeros((self.capacity,), dtype=np.float32)
        self.legal_mask = np.zeros((self.capacity, self.action_size), dtype=np.bool_)
        self.margin = np.zeros((self.capacity,), dtype=np.int16)
        self.write = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        policy: np.ndarray,
        value: np.ndarray,
        legal_mask: np.ndarray,
        margin: np.ndarray | None = None,
    ) -> None:
        obs = np.asarray(obs, dtype=np.float32)
        policy = np.asarray(policy, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32)
        legal_mask = np.asarray(legal_mask, dtype=np.bool_)
        if margin is None:
            # Keep pre-aux callers source-compatible. Fresh self-play paths
            # always provide the native terminal margin explicitly.
            margin = np.zeros(value.shape, dtype=np.int16)
        else:
            margin = np.asarray(margin, dtype=np.int16)
        if obs.ndim != 2 or obs.shape[1] != self.obs_size:
            raise ValueError("obs shape mismatch")
        if policy.shape != (obs.shape[0], self.action_size):
            raise ValueError("policy shape mismatch")
        if value.shape != (obs.shape[0],):
            raise ValueError("value shape mismatch")
        if legal_mask.shape != (obs.shape[0], self.action_size):
            raise ValueError("legal_mask shape mismatch")
        if margin.shape != (obs.shape[0],):
            raise ValueError("margin shape mismatch")

        n = obs.shape[0]
        for start in range(0, n):
            idx = self.write
            self.obs[idx] = obs[start]
            self.policy[idx] = policy[start]
            self.value[idx] = value[start]
            self.legal_mask[idx] = legal_mask[start]
            self.margin[idx] = margin[start]
            self.write = (self.write + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if self.size <= 0:
            raise ValueError("cannot sample empty replay")
        indices = self.rng.integers(0, self.size, size=int(batch_size), endpoint=False)
        return ReplayBatch(
            obs=self.obs[indices].copy(),
            policy=self.policy[indices].copy(),
            value=self.value[indices].copy(),
            legal_mask=self.legal_mask[indices].copy(),
            margin=self.margin[indices].copy(),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "obs_size": self.obs_size,
            "action_size": self.action_size,
            "write": self.write,
            "size": self.size,
            "obs": self.obs[: self.size].copy(),
            "policy": self.policy[: self.size].copy(),
            "value": self.value[: self.size].copy(),
            "legal_mask": self.legal_mask[: self.size].copy(),
            "margin": self.margin[: self.size].copy(),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay capacity mismatch")
        if int(state["obs_size"]) != self.obs_size or int(state["action_size"]) != self.action_size:
            raise ValueError("replay shape mismatch")
        self.write = int(state["write"])
        self.size = int(state["size"])
        self.obs.fill(0.0)
        self.policy.fill(0.0)
        self.value.fill(0.0)
        self.legal_mask.fill(False)
        self.margin.fill(0)
        self.obs[: self.size] = state["obs"]
        self.policy[: self.size] = state["policy"]
        self.value[: self.size] = state["value"]
        self.legal_mask[: self.size] = state["legal_mask"]
        if "margin" in state:
            self.margin[: self.size] = state["margin"]
        else:
            warnings.warn(
                "replay state has no margin column; auxiliary margin training needs fresh data and has been zero-filled",
                RuntimeWarning,
                stacklevel=2,
            )
        self.rng.bit_generator.state = state["rng_state"]


def save_replay_state(replay: ReplayBuffer, path: str | Path) -> Path:
    """Atomically persist the current replay contents in a compact NPZ file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = replay.state_dict()
    metadata = {
        "version": 1,
        "capacity": state["capacity"],
        "obs_size": state["obs_size"],
        "action_size": state["action_size"],
        "write": state["write"],
        "size": state["size"],
        "rng_state": state["rng_state"],
    }
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        # Passing an already-open file avoids numpy silently appending ".npz"
        # to the temporary filename.  os.replace makes the completed file the
        # only visible version after a crash-safe write.
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps(metadata)),
                obs=state["obs"],
                policy=state["policy"],
                value=state["value"],
                legal_mask=state["legal_mask"],
                margin=state["margin"],
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_replay_state(replay: ReplayBuffer, path: str | Path) -> None:
    """Load a replay state saved by :func:`save_replay_state`."""
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if int(metadata.get("version", 0)) != 1:
            raise ValueError("unsupported replay state version")
        state = {
            "capacity": metadata["capacity"],
            "obs_size": metadata["obs_size"],
            "action_size": metadata["action_size"],
            "write": metadata["write"],
            "size": metadata["size"],
            "rng_state": metadata["rng_state"],
            # Archive-backed arrays become invalid once the context closes.
            "obs": archive["obs"].copy(),
            "policy": archive["policy"].copy(),
            "value": archive["value"].copy(),
            "legal_mask": archive["legal_mask"].copy(),
        }
        if "margin" in archive.files:
            state["margin"] = archive["margin"].copy()
        else:
            state["margin"] = np.zeros((int(metadata["size"]),), dtype=np.int16)
            warnings.warn(
                "replay state has no margin column; auxiliary margin training needs fresh data and has been zero-filled",
                RuntimeWarning,
                stacklevel=2,
            )
    replay.load_state_dict(state)
