from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [1024, 1024, 512])


@dataclass
class SelfPlayConfig:
    n_games: int = 64
    sims_per_move: int = 64
    games_per_generation: int = 64
    max_batch: int = 512
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.30
    dirichlet_frac: float = 0.25
    temp_moves: int = 12
    kingdom_mode: str = "random"
    fixed_kingdom: list[str] = field(
        default_factory=lambda: [
            "Sentry",
            "Library",
            "Throne Room",
            "Bandit",
            "Witch",
            "Moat",
            "Village",
            "Smithy",
            "Market",
            "Remodel",
        ]
    )
    max_recorded_moves: int = 512
    max_tree_nodes: int = 4096


@dataclass
class OptimConfig:
    lr: float = 1.0e-3
    lr_schedule: str = "constant"
    min_lr: float = 1.0e-5
    step_decay_every: int = 20
    step_decay_gamma: float = 0.5
    weight_decay: float = 1.0e-4
    batch_size: int = 256
    train_steps_per_generation: int = 100


@dataclass
class ReplayConfig:
    capacity: int = 200_000


@dataclass
class EvalConfig:
    eval_every_n_generations: int = 0
    eval_games: int = 200
    eval_sims: int = 400
    eval_opponent: str = "engine"
    eval_kingdoms: str = "random"
    eval_n_games: int = 64
    eval_max_batch: int = 512


@dataclass
class TrainConfig:
    seed: int = 12345
    generations: int = 10
    device: str = "auto"
    # Parallel collection is opt-in so the legacy single-pipeline run remains
    # exactly deterministic for the default configuration.
    parallel_workers: int = 1
    worker_device: str = "cuda"
    # Used only when worker_device="server". The server owns the one model
    # inference context while workers remain CPU-only SelfPlayRunner hosts.
    server_device: str = "cuda"
    server_max_batch: int = 8192
    server_max_wait_ms: float = 2.0
    server_fp16: bool = False
    server_response_timeout_s: float = 30.0
    # Shared memory is the fast path; queue is retained for unsupported hosts.
    server_transport: str = "shm"
    server_shm_slots: int = 2
    # queue blocks on the shared request-header queue; spin polls SHM counters.
    server_poll: str = "queue"
    checkpoint_dir: str = "checkpoints"
    metrics_csv: str = "checkpoints/metrics.csv"
    model: ModelConfig = field(default_factory=ModelConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, data: dict[str, Any]) -> Any:
    for key, value in data.items():
        if key.startswith("_comment"):
            continue
        if not hasattr(instance, key):
            raise ValueError(f"unknown config key: {key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None) -> TrainConfig:
    cfg = TrainConfig()
    if path is None:
        return cfg
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("config root must be an object")
    return _merge_dataclass(cfg, data)


def save_config(config: TrainConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--smoke", action="store_true")
