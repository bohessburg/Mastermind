from __future__ import annotations

from pathlib import Path

from src.v2.arena.bot.policy import NNCheckpointError, load_policy


def test_load_policy_rejects_missing_checkpoint(tmp_path: Path) -> None:
    try:
        load_policy(tmp_path / "missing-policy.pt")
    except NNCheckpointError as error:
        assert str(error) == "neural-network checkpoint is unavailable"
    else:
        raise AssertionError("missing checkpoint unexpectedly loaded")
