from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ReplayBatch:
    obs: np.ndarray
    policy: np.ndarray
    value: np.ndarray
    legal_mask: np.ndarray


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
        self.write = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, obs: np.ndarray, policy: np.ndarray, value: np.ndarray, legal_mask: np.ndarray) -> None:
        obs = np.asarray(obs, dtype=np.float32)
        policy = np.asarray(policy, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32)
        legal_mask = np.asarray(legal_mask, dtype=np.bool_)
        if obs.ndim != 2 or obs.shape[1] != self.obs_size:
            raise ValueError("obs shape mismatch")
        if policy.shape != (obs.shape[0], self.action_size):
            raise ValueError("policy shape mismatch")
        if value.shape != (obs.shape[0],):
            raise ValueError("value shape mismatch")
        if legal_mask.shape != (obs.shape[0], self.action_size):
            raise ValueError("legal_mask shape mismatch")

        n = obs.shape[0]
        for start in range(0, n):
            idx = self.write
            self.obs[idx] = obs[start]
            self.policy[idx] = policy[start]
            self.value[idx] = value[start]
            self.legal_mask[idx] = legal_mask[start]
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
        self.obs[: self.size] = state["obs"]
        self.policy[: self.size] = state["policy"]
        self.value[: self.size] = state["value"]
        self.legal_mask[: self.size] = state["legal_mask"]
        self.rng.bit_generator.state = state["rng_state"]
