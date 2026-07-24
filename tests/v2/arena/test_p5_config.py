from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.v2.arena.config import ArenaConfig


def test_default_arena_config_loads_credentials_only_from_environment() -> None:
    config = ArenaConfig.load(
        "configs/arena.json",
        environ={"ARENA_USER": "operator", "ARENA_PASS": "secret"},
    )

    assert config.sims == 400
    assert config.checkpoint_path == Path(
        "checkpoints/remote/campaign15/gen_0045.pt"
    )
    assert config.arena_user == "operator"
    assert config.arena_pass == "secret"


def test_arena_config_rejects_invalid_pacing(tmp_path: Path) -> None:
    path = tmp_path / "arena.json"
    path.write_text(
        json.dumps(
            {
                "think_time": {"min_seconds": 3, "max_seconds": 2},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="maximum think time"):
        ArenaConfig.load(path, environ={})
