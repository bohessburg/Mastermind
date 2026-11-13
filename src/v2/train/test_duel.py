from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.v2.encoder_compat import EncoderGenerationError
from . import duel
from .test_train_smoke import tiny_config
from .train import build_objects, save_checkpoint


ROOT = Path(__file__).resolve().parents[3]
V2_CHECKPOINT = ROOT / "checkpoints/remote/campaign15/gen_0045.pt"
V3_CHECKPOINT = ROOT / "checkpoints/remote/campaign18/gen_0005.pt"


def test_duel_progress_line_is_flushed_at_ten_game_cadence(capsys) -> None:
    now = [10.0]
    progress = duel._DuelProgress(20, clock=lambda: now[0])

    progress.record(5, 4, 0)
    assert capsys.readouterr().out == ""
    now[0] = 12.0
    progress.record(5, 4, 1)
    first = capsys.readouterr().out
    assert first == "10/20 games, a 5W-4L-1T, 18000 games/hr\n"

    progress.record(6, 4, 1)
    assert capsys.readouterr().out == ""
    now[0] = 14.0
    progress.record(10, 8, 2)
    assert capsys.readouterr().out == "20/20 games, a 10W-8L-2T, 18000 games/hr\n"


@pytest.mark.skipif(
    not V2_CHECKPOINT.exists() or not V3_CHECKPOINT.exists(),
    reason="mixed-version remote checkpoints are not available locally",
)
def test_native_duel_refuses_legacy_checkpoints() -> None:
    """SelfPlayRunner cannot intercept its native leaves with the Python shim."""
    with pytest.raises(EncoderGenerationError, match=r"encoder generation 1.*encoder generation 2"):
        duel.duel_checkpoints(
            V3_CHECKPOINT,
            V2_CHECKPOINT,
            games=2,
            sims=16,
            kingdoms="random",
            seed=0xD0E1,
            device_name="cpu",
            n_games=1,
            max_batch=32,
            legacy_shim=True,
        )


def test_duel_cli_honest_smoke_with_tiny_checkpoints(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoints: list[Path] = []
    for index in range(2):
        config = tiny_config(tmp_path / f"model_{index}", seed=0xD0E1 + index, generations=1)
        config.model.hidden_sizes = [8]
        config.selfplay.obs_version = 2
        config.selfplay.max_tree_nodes = 256
        model, optimizer, replay = build_objects(config, torch.device("cpu"))
        checkpoints.append(save_checkpoint(config, 1, model, optimizer, replay))

    assert duel.main(
        [
            "--a",
            str(checkpoints[0]),
            "--b",
            str(checkpoints[1]),
            "--games",
            "2",
            "--sims",
            "2",
            "--kingdoms",
            "fixed",
            "--device",
            "cpu",
            "--honest",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert '"honest": true' in output
    assert "2," in output
