"""Standalone CPU checks for static inference-server batch bucketing."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

# This file is intentionally runnable as
# PYTHONPATH=build ./.venv/bin/python tests/v2/py/test_inference_server_bucketing.py
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dominion_v2_py as dz
from src.v2.train.card_transformer import (
    ACTION_DEF_COUNT,
    MAX_PILES,
    MAX_SLOTS,
    OBS_OWN_OFFSET,
    OBS_SIZE_V2,
    OBS_SUPPLY_OFFSET,
    SUPPLY_BLOCK_SIZE,
    CardTokenNet,
)
from src.v2.train.config import TrainConfig, load_config, save_config
from src.v2.train.inference_server import _bucketed_batch_size
from src.v2.train.model import DominionNet


def _realish_observations(batch_size: int) -> torch.Tensor:
    """Build fp32 v2 rows with random counts and valid active supply DefIds."""

    generator = torch.Generator().manual_seed(20260712)
    obs = torch.zeros((batch_size, OBS_SIZE_V2), dtype=torch.float32)
    own = obs[:, OBS_OWN_OFFSET : OBS_OWN_OFFSET + (5 * MAX_SLOTS)].reshape(
        batch_size,
        5,
        MAX_SLOTS,
    )
    own[:, :, :24] = torch.randint(0, 12, (batch_size, 5, 24), generator=generator).float()

    supply = obs[
        :, OBS_SUPPLY_OFFSET : OBS_SUPPLY_OFFSET + (MAX_PILES * SUPPLY_BLOCK_SIZE)
    ].reshape(batch_size, MAX_PILES, SUPPLY_BLOCK_SIZE)
    for row in range(batch_size):
        def_ids = torch.randperm(ACTION_DEF_COUNT, generator=generator)[:24]
        supply[row, :24, 0] = torch.randint(0, 61, (24,), generator=generator).float()
        supply[row, :24, 1] = (def_ids + 1).float()
        supply[row, :24, 2] = (def_ids + 1).float()

    active_def_ids = supply[:, :24, 2]
    assert torch.all((active_def_ids >= 1) & (active_def_ids <= ACTION_DEF_COUNT))
    return obs


def _assert_zero_padding_is_row_independent(model: torch.nn.Module) -> None:
    live_count = 3
    bucket_size = 8
    obs = _realish_observations(live_count)
    generator = torch.Generator().manual_seed(7)
    legal_mask = torch.rand((live_count, dz.ACTION_SPACE_SIZE), generator=generator) > 0.2
    legal_mask[:, 0] = True

    padded_obs = torch.zeros((bucket_size, OBS_SIZE_V2), dtype=torch.float32)
    padded_mask = torch.zeros((bucket_size, dz.ACTION_SPACE_SIZE), dtype=torch.bool)
    padded_obs[:live_count] = obs
    padded_mask[:live_count] = legal_mask

    model.eval()
    with torch.no_grad():
        direct_logits, direct_values = model.evaluate(obs, legal_mask)
        bucket_logits, bucket_values = model.evaluate(padded_obs, padded_mask)

    torch.testing.assert_close(bucket_logits[:live_count], direct_logits, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(bucket_values[:live_count], direct_values, rtol=1.0e-5, atol=1.0e-6)


def test_bucket_padding_is_safe_for_both_architectures() -> None:
    torch.manual_seed(123)
    _assert_zero_padding_is_row_independent(DominionNet(OBS_SIZE_V2, dz.ACTION_SPACE_SIZE, hidden_sizes=[32]))
    _assert_zero_padding_is_row_independent(
        CardTokenNet(
            OBS_SIZE_V2,
            dz.ACTION_SPACE_SIZE,
            d_model=16,
            n_layers=1,
            n_heads=4,
            ffn_multiplier=2,
        )
    )


def test_bucket_selection() -> None:
    buckets = [128, 256, 512, 1024]
    assert _bucketed_batch_size(128, buckets) == 128
    assert _bucketed_batch_size(129, buckets) == 256
    assert _bucketed_batch_size(1025, buckets) == 1025


def test_server_config_defaults_and_round_trip() -> None:
    defaults = TrainConfig()
    assert defaults.server_compile is False
    assert defaults.server_autocast_bf16 is False
    assert defaults.server_batch_buckets is None

    with tempfile.TemporaryDirectory(prefix="dominion_server_config_") as directory:
        path = Path(directory) / "config.json"
        config = TrainConfig()
        config.server_compile = True
        config.server_autocast_bf16 = True
        config.server_batch_buckets = [128, 256, 512]
        save_config(config, path)
        restored = load_config(path)

    assert restored.server_compile is True
    assert restored.server_autocast_bf16 is True
    assert restored.server_batch_buckets == [128, 256, 512]


def main() -> None:
    test_bucket_padding_is_safe_for_both_architectures()
    test_bucket_selection()
    test_server_config_defaults_and_round_trip()
    print("test_inference_server_bucketing: PASS")


if __name__ == "__main__":
    main()
