"""Offline human/self-play mixture smoke tests.

Run directly with:
    PYTHONPATH=build ./.venv/bin/python src/v2/train/test_offline_fit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.train import offline_fit
from src.v2.train.replay import ReplayBuffer, save_replay_state


def _human_fixture(root: Path, rows: int = 8) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    obs = np.zeros((rows, 1788), dtype=np.float32)
    obs[:, 0] = 3.0
    obs[:, 1] = 1788.0
    action = np.arange(rows, dtype=np.int32) % 3
    legal = np.ones((rows, 357), dtype=np.bool_)
    np.savez_compressed(
        root / "tuples-00000.npz",
        obs=obs,
        action=action,
        legal=legal,
        value=np.zeros((rows,), dtype=np.float32),
        margin=np.asarray([-2, -1, 0, 1, 2, 3, 4, 5][:rows], dtype=np.int16),
        winner=np.asarray([0] * rows, dtype=np.int8),
        seat_index=np.zeros((rows,), dtype=np.int16),
        game_index=np.zeros((rows,), dtype=np.int32),
        ply_index=np.arange(rows, dtype=np.int32),
        turn_number=np.arange(1, rows + 1, dtype=np.int32),
    )
    manifest = {
        "schema_version": 1,
        "obs_width": 1788,
        "action_width": 357,
        "shards": [{"path": "tuples-00000.npz", "tuples": rows}],
        "totals": {"tuples_exported": rows},
        "games": [{"index": 0, "seat_kinds": ["human", "bot:bigmoney"]}],
    }
    (root / "tuple_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _replay_fixture(path: Path, rows: int = 8) -> Path:
    replay = ReplayBuffer(capacity=rows, obs_size=1788, action_size=357, seed=717)
    obs = np.zeros((rows, 1788), dtype=np.float32)
    obs[:, 0] = 3.0
    obs[:, 1] = 1788.0
    policy = np.zeros((rows, 357), dtype=np.float32)
    policy[:, 0] = 1.0
    legal = np.ones((rows, 357), dtype=np.bool_)
    replay.add(obs, policy, np.linspace(-1.0, 1.0, rows, dtype=np.float32), legal)
    return save_replay_state(replay, path)


def test_tiny_offline_fit_mixes_half_human_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    replay_path = _replay_fixture(tmp_path / "replay_state.npz")
    human_root = _human_fixture(tmp_path / "tuples")

    assert offline_fit.main(
        [
            "--data",
            str(replay_path),
            "--human-tuples",
            str(human_root),
            "--human-fraction",
            "0.5",
            "--arch",
            "mlp",
            "--mlp-hidden-sizes",
            "8",
            "--batch-size",
            "4",
            "--steps",
            "2",
            "--log-every",
            "1",
            "--device",
            "cpu",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "train_human(" in output
    assert "train_selfplay(" in output
    assert "final comparison by source" in output


if __name__ == "__main__":  # pragma: no cover - standalone test entry point
    raise SystemExit(pytest.main([str(Path(__file__)), "-q"]))
