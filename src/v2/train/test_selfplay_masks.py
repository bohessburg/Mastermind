from __future__ import annotations

import numpy as np
import pytest
import torch

import dominion_v2_py as dz

from .model import masked_policy_loss
from .replay import ReplayBuffer
from .selfplay import _records_to_replay
from .workers import _pack_records, add_packed_records


def _record_with_legal_zero_visit() -> dict[str, np.ndarray]:
    policy = np.zeros((1, dz.ACTION_SPACE_SIZE), dtype=np.float32)
    policy[0, 3] = 1.0
    legal_mask = np.zeros((1, dz.ACTION_SPACE_SIZE), dtype=np.bool_)
    legal_mask[0, 3] = True
    legal_mask[0, 7] = True
    return {
        "observations": np.zeros((1, 5), dtype=np.float32),
        "policy_targets": policy,
        "legal_mask": legal_mask,
        "values": np.zeros((1,), dtype=np.float32),
        "margins": np.asarray([7], dtype=np.int16),
    }


def test_true_legal_mask_reaches_direct_and_packed_replay_verbatim() -> None:
    record = _record_with_legal_zero_visit()
    expected = record["legal_mask"]
    expected_margin = record["margins"]

    direct = ReplayBuffer(4, 5, dz.ACTION_SPACE_SIZE, seed=1)
    assert _records_to_replay([record], direct) == (1, 1)
    np.testing.assert_array_equal(direct.legal_mask[:1], expected)
    np.testing.assert_array_equal(direct.margin[:1], expected_margin)

    packed = _pack_records([record], 5)
    packed_replay = ReplayBuffer(4, 5, dz.ACTION_SPACE_SIZE, seed=2)
    assert add_packed_records(packed_replay, packed) == (1, 1)
    np.testing.assert_array_equal(packed_replay.legal_mask[:1], expected)
    np.testing.assert_array_equal(packed_replay.margin[:1], expected_margin)


def test_missing_true_legal_mask_fails_loudly() -> None:
    record = _record_with_legal_zero_visit()
    del record["legal_mask"]
    replay = ReplayBuffer(4, 5, dz.ACTION_SPACE_SIZE, seed=3)
    with pytest.raises(ValueError, match="legal_mask"):
        _records_to_replay([record], replay)
    with pytest.raises(ValueError, match="legal_mask"):
        _pack_records([record], 5)


def test_legal_zero_visit_action_stays_in_policy_ce_denominator() -> None:
    logits = torch.zeros((1, dz.ACTION_SPACE_SIZE), dtype=torch.float32)
    target = torch.zeros((1, dz.ACTION_SPACE_SIZE), dtype=torch.float32)
    target[0, 3] = 1.0
    true_mask = torch.zeros((1, dz.ACTION_SPACE_SIZE), dtype=torch.bool)
    true_mask[0, 3] = True
    true_mask[0, 7] = True

    true_loss, _ = masked_policy_loss(logits, true_mask, target)
    old_loss, _ = masked_policy_loss(logits, target > 0.0, target)

    assert true_loss.item() == pytest.approx(float(np.log(2.0)))
    assert old_loss.item() == pytest.approx(0.0)
    assert true_loss.item() != pytest.approx(old_loss.item())
