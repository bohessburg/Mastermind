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
    weight_decay: float = 1.0e-4
    batch_size: int = 256
    train_steps_per_generation: int = 100


@dataclass
class ReplayConfig:
    capacity: int = 200_000


@dataclass
class TrainConfig:
    seed: int = 12345
    generations: int = 10
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    metrics_csv: str = "checkpoints/metrics.csv"
    model: ModelConfig = field(default_factory=ModelConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, data: dict[str, Any]) -> Any:
    for key, value in data.items():
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
