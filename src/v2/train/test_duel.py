from __future__ import annotations

from pathlib import Path

import pytest

from . import duel


ROOT = Path(__file__).resolve().parents[3]
V2_CHECKPOINT = ROOT / "checkpoints/remote/campaign15/gen_0045.pt"
V3_CHECKPOINT = ROOT / "checkpoints/remote/campaign18/gen_0005.pt"


@pytest.mark.skipif(
    not V2_CHECKPOINT.exists() or not V3_CHECKPOINT.exists(),
    reason="mixed-version remote checkpoints are not available locally",
)
def test_mixed_observation_checkpoint_duel_is_seeded_and_downgrades_v2(monkeypatch) -> None:
    """A real v3-v2 match must route the v2 net the 1717-wide ABI."""
    real_load_model = duel.load_model
    v2_widths: list[int] = []

    def load_with_v2_spy(checkpoint, device):
        model, config = real_load_model(checkpoint, device)
        if Path(checkpoint) == V2_CHECKPOINT:
            evaluate = model.evaluate

            def spy(observations, masks):
                v2_widths.append(int(observations.shape[-1]))
                return evaluate(observations, masks)

            model.evaluate = spy
        return model, config

    monkeypatch.setattr(duel, "load_model", load_with_v2_spy)
    first, _, _, _ = duel.duel_checkpoints(
        V3_CHECKPOINT,
        V2_CHECKPOINT,
        games=2,
        sims=16,
        kingdoms="random",
        seed=0xD0E1,
        device_name="cpu",
        n_games=1,
        max_batch=32,
    )
    second, _, _, _ = duel.duel_checkpoints(
        V3_CHECKPOINT,
        V2_CHECKPOINT,
        games=2,
        sims=16,
        kingdoms="random",
        seed=0xD0E1,
        device_name="cpu",
        n_games=1,
        max_batch=32,
    )

    assert first.games == 2
    assert first.wins_a + first.wins_b + first.ties == first.games
    assert all(width == 1717 for width in v2_widths)
    assert v2_widths
    assert (
        first.games,
        first.wins_a,
        first.wins_b,
        first.ties,
        first.truncated,
        first.end_province,
        first.end_piles,
    ) == (
        second.games,
        second.wins_a,
        second.wins_b,
        second.ties,
        second.truncated,
        second.end_province,
        second.end_piles,
    )
