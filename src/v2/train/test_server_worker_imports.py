"""Regression coverage for the Torch-free server self-play worker bootstrap."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_server_worker_bootstrap_does_not_import_torch() -> None:
    """A spawned server worker must skip both trainer and local-evaluator Torch imports."""

    repository = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    inherited_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repository / "build"), inherited_pythonpath) if part
    )
    probe = """
import runpy
import sys

runpy.run_path(sys.argv[1], run_name='__mp_main__')
from src.v2.train.config import TrainConfig
from src.v2.train.workers import _server_selfplay_enabled

config = TrainConfig()
config.server_selfplay = True
assert _server_selfplay_enabled(config)
loaded = sorted(name for name in sys.modules if name == 'torch' or name.startswith('torch.'))
assert not loaded, loaded
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(repository / "src/v2/train/train.py")],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
