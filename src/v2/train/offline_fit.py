"""Small offline fit harness for replay and human-imitation comparisons.

The harness reads replay snapshots directly and can mix manifest-backed human
tuples into every optimizer batch. It never touches online workers, and emits
a serving-compatible checkpoint only when explicitly asked to do so.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.v2.train.card_transformer import (
        ACTION_SPACE_SIZE,
        DEFAULT_AUX_MARGIN_BUCKETS,
        OBS_SIZE_V2,
        OBS_SIZE_V3,
        CardTokenNet,
        margin_bucket_ids,
    )
    from src.v2.train.human_data import HumanTupleDataset, load_human_tuples
    from src.v2.train.model import DominionNet, count_parameters, masked_policy_loss
else:
    from .card_transformer import (
        ACTION_SPACE_SIZE,
        DEFAULT_AUX_MARGIN_BUCKETS,
        OBS_SIZE_V2,
        OBS_SIZE_V3,
        CardTokenNet,
        margin_bucket_ids,
    )
    from .human_data import HumanTupleDataset, load_human_tuples
    from .model import DominionNet, count_parameters, masked_policy_loss


DEFAULT_REPLAY = Path("checkpoints/remote/campaign14/replay_state.npz")
SUPPORTED_OBS_WIDTHS = frozenset({OBS_SIZE_V2, OBS_SIZE_V3})
# A 250K v3 prefix expands to multiple GiB. Above this threshold, decode NPZ
# members into temporary memmaps so cheap offline fits retain only minibatches
# in RAM instead of being killed before their first optimizer step.
MEMMAP_PREFIX_THRESHOLD_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class ReplayArrays:
    obs: np.ndarray
    policy: np.ndarray
    value: np.ndarray
    legal_mask: np.ndarray
    margin: np.ndarray
    _scratch: tempfile.TemporaryDirectory[str] | None = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return int(self.obs.shape[0])


@dataclass(frozen=True)
class HumanArrays:
    obs: np.ndarray
    action: np.ndarray
    value: np.ndarray
    legal_mask: np.ndarray
    margin: np.ndarray

    def __len__(self) -> int:
        return int(self.obs.shape[0])


@dataclass(frozen=True)
class LossMetrics:
    total: float
    policy: float
    value: float
    entropy: float
    aux_margin: float = float("nan")


class _MetricAccumulator:
    def __init__(self) -> None:
        self.totals = [0.0, 0.0, 0.0, 0.0]
        self.aux_total = 0.0
        self.aux_samples = 0
        self.samples = 0

    def add(self, metrics: LossMetrics, samples: int) -> None:
        if samples <= 0:
            return
        self.totals[0] += metrics.total * samples
        self.totals[1] += metrics.policy * samples
        self.totals[2] += metrics.value * samples
        self.totals[3] += metrics.entropy * samples
        if math.isfinite(metrics.aux_margin):
            self.aux_total += metrics.aux_margin * samples
            self.aux_samples += samples
        self.samples += samples

    def finish(self) -> LossMetrics | None:
        if self.samples == 0:
            return None
        return _finish_metrics(
            *self.totals,
            self.samples,
            aux_margin=(self.aux_total / self.aux_samples if self.aux_samples else float("nan")),
        )


class _CyclingIndices:
    """A local seeded shuffled cycle used to make every mixed batch exact."""

    def __init__(self, indices: np.ndarray, seed: int):
        if len(indices) <= 0:
            raise ValueError("cannot cycle an empty index set")
        self.indices = np.asarray(indices, dtype=np.intp)
        self.rng = np.random.default_rng(int(seed))
        self.order = self.rng.permutation(self.indices)
        self.cursor = 0

    def take(self, count: int) -> np.ndarray:
        if count < 0:
            raise ValueError("batch count cannot be negative")
        if count == 0:
            return np.empty((0,), dtype=np.intp)
        pieces: list[np.ndarray] = []
        remaining = count
        while remaining:
            available = len(self.order) - self.cursor
            take = min(available, remaining)
            pieces.append(self.order[self.cursor : self.cursor + take])
            self.cursor += take
            remaining -= take
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(self.indices)
                self.cursor = 0
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)


def _read_exact(handle, byte_count: int) -> bytes:
    """Read exactly ``byte_count`` decompressed bytes from a zip member."""
    chunks: list[bytes] = []
    remaining = int(byte_count)
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            raise ValueError("replay archive ended before the requested array prefix")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_npy_prefix(
    archive: zipfile.ZipFile,
    name: str,
    rows: int,
    *,
    mmap_path: Path | None = None,
) -> np.ndarray:
    """Read a C-order row prefix, optionally spilling it to a memmap."""
    with archive.open(f"{name}.npy") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version in ((2, 0), (3, 0)):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported .npy version {version} in {name}")
        if fortran_order or not shape:
            raise ValueError(f"{name} must be a non-empty C-order row array")
        if rows > int(shape[0]):
            raise ValueError(f"requested {rows} rows from {name}, archive has {shape[0]}")
        trailing_shape = tuple(int(dim) for dim in shape[1:])
        row_bytes = int(np.prod(trailing_shape, dtype=np.int64) if trailing_shape else 1) * dtype.itemsize
        output_shape = (rows, *trailing_shape)
        if mmap_path is None:
            raw = _read_exact(handle, rows * row_bytes)
            return np.frombuffer(raw, dtype=dtype).reshape(output_shape).copy()

        output = np.lib.format.open_memmap(
            mmap_path,
            mode="w+",
            dtype=dtype,
            shape=output_shape,
        )
        # Keep each decompressed transport buffer modest even for the 1.8 GiB
        # observation member in a 250K c19 slice.
        chunk_rows = max(1, (8 * 1024 * 1024) // row_bytes)
        for start in range(0, rows, chunk_rows):
            stop = min(rows, start + chunk_rows)
            raw = _read_exact(handle, (stop - start) * row_bytes)
            output[start:stop] = np.frombuffer(raw, dtype=dtype).reshape((stop - start, *trailing_shape))
        output.flush()
        return output


def _invert_c19_margin_blend(value: np.ndarray, alpha0: float = 0.6) -> np.ndarray:
    """Recover c19's raw margins from its alpha=0.6 MarginBlend labels.

    This reproduces the bijective recorded-range invariant from
    ``bench/value_target_sweep/sweep.py``: non-ties satisfy
    ``|v| = a + (1-a) * (0.5 + 0.5 * min(|m|, 20) / 20)``.
    """

    scale = 20.0
    sign = np.sign(value)
    graded = (np.abs(value) - alpha0) / (1.0 - alpha0)
    margin = np.round((graded - 0.5) * 2.0 * scale)
    margin = np.clip(margin, 0.0, scale) * sign
    margin[value == 0.0] = 0.0
    return np.ascontiguousarray(margin, dtype=np.int16)


def load_replay_prefix(path: str | Path, max_samples: int | None) -> ReplayArrays:
    """Load a replay prefix without allocating an entire compressed archive."""
    replay_path = Path(path)
    scratch: tempfile.TemporaryDirectory[str] | None = None
    with zipfile.ZipFile(replay_path) as archive:
        with archive.open("metadata.npy") as handle:
            metadata = json.loads(str(np.load(handle, allow_pickle=False).item()))
        total_rows = int(metadata["size"])
        rows = total_rows if max_samples is None else min(total_rows, int(max_samples))
        if rows < 2:
            raise ValueError("need at least two replay samples for a train/validation split")
        use_memmap = rows * int(metadata["obs_size"]) * np.dtype(np.float32).itemsize >= MEMMAP_PREFIX_THRESHOLD_BYTES
        if use_memmap:
            scratch = tempfile.TemporaryDirectory(prefix="dominion_offline_fit_")
            scratch_root = Path(scratch.name)
        else:
            scratch_root = None
        obs = _read_npy_prefix(
            archive,
            "obs",
            rows,
            mmap_path=(scratch_root / "obs.npy" if scratch_root is not None else None),
        )
        policy = _read_npy_prefix(
            archive,
            "policy",
            rows,
            mmap_path=(scratch_root / "policy.npy" if scratch_root is not None else None),
        )
        value = _read_npy_prefix(
            archive,
            "value",
            rows,
            mmap_path=(scratch_root / "value.npy" if scratch_root is not None else None),
        ).reshape(rows)
        legal_mask = _read_npy_prefix(
            archive,
            "legal_mask",
            rows,
            mmap_path=(scratch_root / "legal_mask.npy" if scratch_root is not None else None),
        )
        if "margin.npy" in archive.namelist():
            margin = _read_npy_prefix(
                archive,
                "margin",
                rows,
                mmap_path=(scratch_root / "margin.npy" if scratch_root is not None else None),
            ).reshape(rows)
        else:
            # campaign19 predates the replay margin column. Its c19 targets
            # obey the exact alpha=0.6 inversion documented above.
            margin = _invert_c19_margin_blend(value)
            warnings.warn(
                "replay has no margin column; deriving c19 margins by exact alpha=0.6 MarginBlend inversion",
                RuntimeWarning,
                stacklevel=2,
            )

    if obs.ndim != 2 or obs.shape[0] != rows or int(obs.shape[1]) not in SUPPORTED_OBS_WIDTHS:
        allowed = ", ".join(str(width) for width in sorted(SUPPORTED_OBS_WIDTHS))
        raise ValueError(f"expected replay obs width one of {allowed}, got {obs.shape}")
    if policy.shape != (rows, ACTION_SPACE_SIZE):
        raise ValueError(f"expected policy shape [{rows}, {ACTION_SPACE_SIZE}], got {policy.shape}")
    if legal_mask.shape != (rows, ACTION_SPACE_SIZE):
        raise ValueError(f"expected legal_mask shape [{rows}, {ACTION_SPACE_SIZE}], got {legal_mask.shape}")
    if margin.shape != (rows,):
        raise ValueError(f"expected margin shape [{rows}], got {margin.shape}")
    return ReplayArrays(
        obs=obs.astype(np.float32, copy=False),
        policy=policy.astype(np.float32, copy=False),
        value=value.astype(np.float32, copy=False),
        legal_mask=legal_mask.astype(np.bool_, copy=False),
        margin=margin.astype(np.int16, copy=False),
        _scratch=scratch,
    )


def human_arrays(dataset: HumanTupleDataset) -> HumanArrays:
    """Adapt the hard-label loader output for this harness's source interface."""
    return HumanArrays(
        obs=np.ascontiguousarray(dataset.obs, dtype=np.float32),
        action=np.ascontiguousarray(dataset.action, dtype=np.int64),
        value=np.ascontiguousarray(dataset.value, dtype=np.float32),
        legal_mask=np.ascontiguousarray(dataset.legal, dtype=np.bool_),
        margin=np.ascontiguousarray(dataset.margin, dtype=np.int16),
    )


def load_human_arrays(path: str | Path) -> HumanArrays:
    """Load the default margin-blend human target from tuple shards."""
    return human_arrays(load_human_tuples(path))


def _batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _batch_tensors(data: ReplayArrays, indices: np.ndarray, device: torch.device):
    return (
        torch.as_tensor(data.obs[indices], dtype=torch.float32, device=device),
        torch.as_tensor(data.policy[indices], dtype=torch.float32, device=device),
        torch.as_tensor(data.value[indices], dtype=torch.float32, device=device),
        torch.as_tensor(data.legal_mask[indices], dtype=torch.bool, device=device),
    )


def _margin_tensor(data: ReplayArrays, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(data.margin[indices], dtype=torch.long, device=device)


def _human_batch_tensors(data: HumanArrays, indices: np.ndarray, device: torch.device):
    return (
        torch.as_tensor(data.obs[indices], dtype=torch.float32, device=device),
        torch.as_tensor(data.action[indices], dtype=torch.long, device=device),
        torch.as_tensor(data.value[indices], dtype=torch.float32, device=device),
        torch.as_tensor(data.legal_mask[indices], dtype=torch.bool, device=device),
    )


def _human_margin_tensor(data: HumanArrays, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(data.margin[indices], dtype=torch.long, device=device)


def _losses(
    logits: torch.Tensor,
    values: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    legal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The soft-policy self-play loss used by train.py."""
    policy_loss, entropy = masked_policy_loss(logits, legal_mask, policy_target)
    value_loss = F.mse_loss(values, value_target)
    return policy_loss + value_loss, policy_loss, value_loss, entropy


def _losses_with_aux(
    logits: torch.Tensor,
    values: torch.Tensor,
    aux_logits: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    legal_mask: torch.Tensor,
    margins: torch.Tensor,
    aux_margin_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total, policy_loss, value_loss, entropy = _losses(
        logits,
        values,
        policy_target,
        value_target,
        legal_mask,
    )
    aux_loss = F.cross_entropy(aux_logits, margin_bucket_ids(margins, int(aux_logits.shape[-1])))
    return total + float(aux_margin_weight) * aux_loss, policy_loss, value_loss, entropy, aux_loss


def _human_losses(
    logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    value_target: torch.Tensor,
    legal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard-label CE plus value MSE for human demonstrations."""
    masked = logits.masked_fill(~legal_mask, -1.0e9)
    policy_loss = F.cross_entropy(masked, actions, reduction="mean")
    probs = torch.softmax(masked, dim=-1)
    log_probs = torch.log_softmax(masked, dim=-1)
    entropy = -(probs * log_probs).masked_fill(~legal_mask, 0.0).sum(dim=-1).mean()
    value_loss = F.mse_loss(values, value_target)
    return policy_loss + value_loss, policy_loss, value_loss, entropy


def _human_losses_with_aux(
    logits: torch.Tensor,
    values: torch.Tensor,
    aux_logits: torch.Tensor,
    actions: torch.Tensor,
    value_target: torch.Tensor,
    legal_mask: torch.Tensor,
    margins: torch.Tensor,
    aux_margin_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total, policy_loss, value_loss, entropy = _human_losses(
        logits,
        values,
        actions,
        value_target,
        legal_mask,
    )
    aux_loss = F.cross_entropy(aux_logits, margin_bucket_ids(margins, int(aux_logits.shape[-1])))
    return total + float(aux_margin_weight) * aux_loss, policy_loss, value_loss, entropy, aux_loss


def _as_metrics(
    losses: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
) -> LossMetrics:
    return LossMetrics(*(float(metric.detach().cpu()) for metric in losses))


def _finish_metrics(
    total: float,
    policy: float,
    value: float,
    entropy: float,
    samples: int,
    *,
    aux_margin: float = float("nan"),
) -> LossMetrics:
    return LossMetrics(
        total=total / samples,
        policy=policy / samples,
        value=value / samples,
        entropy=entropy / samples,
        aux_margin=aux_margin,
    )


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: ReplayArrays,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> LossMetrics:
    """Retained single-source replay epoch helper for older comparisons."""
    model.train()
    totals = [0.0, 0.0, 0.0, 0.0]
    for batch_indices in _batches(indices, batch_size):
        obs, policy_target, value_target, legal_mask = _batch_tensors(data, batch_indices, device)
        logits, values = model(obs)
        losses = _losses(logits, values, policy_target, value_target, legal_mask)
        optimizer.zero_grad(set_to_none=True)
        losses[0].backward()
        optimizer.step()
        weight = len(batch_indices)
        for index, metric in enumerate(losses):
            totals[index] += float(metric.detach().cpu()) * weight
    return _finish_metrics(*totals, len(indices))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: ReplayArrays,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    *,
    aux_margin_weight: float = 0.0,
) -> LossMetrics:
    model.eval()
    totals = [0.0, 0.0, 0.0, 0.0]
    aux_total = 0.0
    for batch_indices in _batches(indices, batch_size):
        obs, policy_target, value_target, legal_mask = _batch_tensors(data, batch_indices, device)
        if aux_margin_weight > 0.0:
            logits, values, aux_logits = model.forward_with_aux(obs)
            losses = _losses_with_aux(
                logits,
                values,
                aux_logits,
                policy_target,
                value_target,
                legal_mask,
                _margin_tensor(data, batch_indices, device),
                aux_margin_weight,
            )
            aux_total += float(losses[4].detach().cpu()) * len(batch_indices)
            losses = losses[:4]
        else:
            losses = _losses(*model(obs), policy_target, value_target, legal_mask)
        weight = len(batch_indices)
        for index, metric in enumerate(losses):
            totals[index] += float(metric.detach().cpu()) * weight
    return _finish_metrics(
        *totals,
        len(indices),
        aux_margin=(aux_total / len(indices) if aux_margin_weight > 0.0 else float("nan")),
    )


@torch.no_grad()
def evaluate_human(
    model: torch.nn.Module,
    data: HumanArrays,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    *,
    aux_margin_weight: float = 0.0,
) -> LossMetrics:
    model.eval()
    totals = [0.0, 0.0, 0.0, 0.0]
    aux_total = 0.0
    for batch_indices in _batches(indices, batch_size):
        obs, actions, value_target, legal_mask = _human_batch_tensors(data, batch_indices, device)
        if aux_margin_weight > 0.0:
            logits, values, aux_logits = model.forward_with_aux(obs)
            losses = _human_losses_with_aux(
                logits,
                values,
                aux_logits,
                actions,
                value_target,
                legal_mask,
                _human_margin_tensor(data, batch_indices, device),
                aux_margin_weight,
            )
            aux_total += float(losses[4].detach().cpu()) * len(batch_indices)
            losses = losses[:4]
        else:
            losses = _human_losses(*model(obs), actions, value_target, legal_mask)
        weight = len(batch_indices)
        for index, metric in enumerate(losses):
            totals[index] += float(metric.detach().cpu()) * weight
    return _finish_metrics(
        *totals,
        len(indices),
        aux_margin=(aux_total / len(indices) if aux_margin_weight > 0.0 else float("nan")),
    )


def mixed_batch_counts(batch_size: int, human_fraction: float, *, has_selfplay: bool, has_human: bool) -> tuple[int, int]:
    """Return exact (self-play, human) row counts for each optimizer batch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not has_selfplay:
        return 0, batch_size
    if not has_human:
        return batch_size, 0
    if batch_size < 2:
        raise ValueError("a mixed batch requires batch_size of at least two")
    human_rows = int(round(batch_size * human_fraction))
    human_rows = max(1, min(batch_size - 1, human_rows))
    return batch_size - human_rows, human_rows


def _mixed_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    selfplay: ReplayArrays | None,
    selfplay_indices: np.ndarray,
    human: HumanArrays | None,
    human_indices: np.ndarray,
    device: torch.device,
    aux_margin_weight: float = 0.0,
) -> dict[str, tuple[LossMetrics, int]]:
    model.train()
    source_losses: dict[
        str,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int],
    ] = {}
    if selfplay is not None and len(selfplay_indices):
        obs, policy_target, value_target, legal_mask = _batch_tensors(selfplay, selfplay_indices, device)
        if aux_margin_weight > 0.0:
            logits, values, aux_logits = model.forward_with_aux(obs)
            losses = _losses_with_aux(
                logits,
                values,
                aux_logits,
                policy_target,
                value_target,
                legal_mask,
                _margin_tensor(selfplay, selfplay_indices, device),
                aux_margin_weight,
            )
        else:
            base_losses = _losses(*model(obs), policy_target, value_target, legal_mask)
            losses = (*base_losses, base_losses[0].new_tensor(float("nan")))
        source_losses["selfplay"] = (*losses, len(selfplay_indices))
    if human is not None and len(human_indices):
        obs, actions, value_target, legal_mask = _human_batch_tensors(human, human_indices, device)
        if aux_margin_weight > 0.0:
            logits, values, aux_logits = model.forward_with_aux(obs)
            losses = _human_losses_with_aux(
                logits,
                values,
                aux_logits,
                actions,
                value_target,
                legal_mask,
                _human_margin_tensor(human, human_indices, device),
                aux_margin_weight,
            )
        else:
            base_losses = _human_losses(*model(obs), actions, value_target, legal_mask)
            losses = (*base_losses, base_losses[0].new_tensor(float("nan")))
        source_losses["human"] = (*losses, len(human_indices))
    if not source_losses:
        raise ValueError("mixed step needs at least one source row")

    total_rows = sum(values[-1] for values in source_losses.values())
    combined_loss = sum(values[0] * (values[-1] / total_rows) for values in source_losses.values())
    optimizer.zero_grad(set_to_none=True)
    combined_loss.backward()
    optimizer.step()
    return {
        source: (_as_metrics(values[:5]), values[-1])
        for source, values in source_losses.items()
    }


def train_mixed_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    selfplay: ReplayArrays | None,
    selfplay_indices: np.ndarray | None,
    human: HumanArrays | None,
    human_indices: np.ndarray | None,
    selfplay_rows: int,
    human_rows: int,
    steps: int,
    device: torch.device,
    seed: int,
    aux_margin_weight: float = 0.0,
) -> dict[str, LossMetrics]:
    """Train exact source proportions for a finite number of optimizer steps."""
    if steps <= 0:
        return {}
    selfplay_cycle = (
        _CyclingIndices(selfplay_indices, seed ^ 0x5150) if selfplay is not None and selfplay_rows else None
    )
    human_cycle = _CyclingIndices(human_indices, seed ^ 0x4855) if human is not None and human_rows else None
    accumulators = {"selfplay": _MetricAccumulator(), "human": _MetricAccumulator()}
    for _ in range(steps):
        source_losses = _mixed_step(
            model,
            optimizer,
            selfplay=selfplay,
            selfplay_indices=(selfplay_cycle.take(selfplay_rows) if selfplay_cycle is not None else np.empty(0, dtype=np.intp)),
            human=human,
            human_indices=(human_cycle.take(human_rows) if human_cycle is not None else np.empty(0, dtype=np.intp)),
            device=device,
            aux_margin_weight=aux_margin_weight,
        )
        for source, (metrics, rows) in source_losses.items():
            accumulators[source].add(metrics, rows)
    return {
        source: metrics
        for source, accumulator in accumulators.items()
        if (metrics := accumulator.finish()) is not None
    }


def parse_hidden_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--mlp-hidden-sizes must be comma-separated integers") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("--mlp-hidden-sizes must contain positive integers")
    return sizes


def build_model(args: argparse.Namespace, obs_size: int = OBS_SIZE_V2) -> torch.nn.Module:
    if args.arch == "mlp":
        return DominionNet(
            obs_size,
            ACTION_SPACE_SIZE,
            hidden_sizes=args.mlp_hidden_sizes,
            input_scale=args.mlp_input_scale,
        )
    return CardTokenNet(
        obs_size,
        ACTION_SPACE_SIZE,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_multiplier=args.ffn_multiplier,
        dropout=args.dropout,
        aux_margin_buckets=(DEFAULT_AUX_MARGIN_BUCKETS if args.aux_margin_weight > 0.0 else None),
    )


def select_device(name: str) -> torch.device:
    requested = name.lower()
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("mps requested but unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    return torch.device(requested)


def _split_indices(samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if samples < 2:
        raise ValueError("need at least two samples for a train/validation split")
    shuffled = np.random.default_rng(seed).permutation(samples)
    val_count = max(1, int(round(samples * 0.05)))
    return shuffled[val_count:], shuffled[:val_count]


def _steps_per_epoch(
    selfplay_train: np.ndarray | None,
    human_train: np.ndarray | None,
    selfplay_rows: int,
    human_rows: int,
) -> int:
    counts: list[int] = []
    if selfplay_train is not None and selfplay_rows:
        counts.append(math.ceil(len(selfplay_train) / selfplay_rows))
    if human_train is not None and human_rows:
        counts.append(math.ceil(len(human_train) / human_rows))
    if not counts:
        raise ValueError("no source rows were selected for training")
    return max(counts)


def _source_validation(
    model: torch.nn.Module,
    *,
    selfplay: ReplayArrays | None,
    selfplay_indices: np.ndarray | None,
    human: HumanArrays | None,
    human_indices: np.ndarray | None,
    batch_size: int,
    device: torch.device,
    aux_margin_weight: float = 0.0,
) -> dict[str, LossMetrics]:
    results: dict[str, LossMetrics] = {}
    if selfplay is not None and selfplay_indices is not None:
        results["selfplay"] = evaluate(
            model,
            selfplay,
            selfplay_indices,
            batch_size,
            device,
            aux_margin_weight=aux_margin_weight,
        )
    if human is not None and human_indices is not None:
        results["human"] = evaluate_human(
            model,
            human,
            human_indices,
            batch_size,
            device,
            aux_margin_weight=aux_margin_weight,
        )
    return results


def _format_metrics(metrics: LossMetrics | None) -> str:
    if metrics is None:
        return "n/a"
    aux = "n/a" if not math.isfinite(metrics.aux_margin) else f"{metrics.aux_margin:.6f}"
    return f"ce={metrics.policy:.6f},value={metrics.value:.6f},aux={aux}"


def _print_trajectory(step: int, train_metrics: dict[str, LossMetrics], val_metrics: dict[str, LossMetrics]) -> None:
    print(
        f"step={step:04d} "
        f"train_human({_format_metrics(train_metrics.get('human'))}) "
        f"val_human({_format_metrics(val_metrics.get('human'))}) "
        f"train_selfplay({_format_metrics(train_metrics.get('selfplay'))}) "
        f"val_selfplay({_format_metrics(val_metrics.get('selfplay'))})",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_REPLAY, help="replay_state.npz input")
    parser.add_argument("--human-tuples", type=Path, default=Path("exports/tuples"), help="tuple shard directory")
    parser.add_argument("--human-fraction", type=float, default=0.0, help="human rows in every mixed train batch")
    parser.add_argument("--human-only", action="store_true", help="fit only hard-label human tuples; do not load replay")
    parser.add_argument("--arch", choices=("mlp", "transformer"), required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--max-samples", type=int, default=None, help="deterministic replay prefix for quick fits")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=None, help="optimizer steps; overrides --epochs")
    parser.add_argument("--log-every", type=int, default=0, help="optimizer-step trajectory cadence (zero means epoch cadence)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--mlp-hidden-sizes", type=parse_hidden_sizes, default=(1536, 1536, 768))
    parser.add_argument("--mlp-input-scale", type=float, default=16.0)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--aux-margin-weight",
        type=float,
        default=0.0,
        help="weight for CardTokenNet terminal-margin distribution CE (zero disables it)",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help="optional serving-compatible fitted checkpoint path",
    )
    parser.add_argument(
        "--encoder-generation",
        type=int,
        default=None,
        help="required with --checkpoint-out; input encoder generation to stamp",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive")
    if args.steps is not None and args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.log_every < 0:
        raise SystemExit("--log-every cannot be negative")
    if args.lr <= 0:
        raise SystemExit("--lr must be positive")
    if not 0.0 <= args.human_fraction <= 1.0:
        raise SystemExit("--human-fraction must be between zero and one")
    if not math.isfinite(args.aux_margin_weight) or args.aux_margin_weight < 0.0:
        raise SystemExit("--aux-margin-weight must be a finite non-negative number")
    if args.aux_margin_weight > 0.0 and args.arch != "transformer":
        raise SystemExit("--aux-margin-weight requires --arch transformer")
    if args.checkpoint_out is not None and (args.encoder_generation is None or args.encoder_generation <= 0):
        raise SystemExit("--checkpoint-out requires a positive --encoder-generation")

    use_human = bool(args.human_only or args.human_fraction > 0.0)
    use_selfplay = not args.human_only and args.human_fraction < 1.0
    effective_human_fraction = 1.0 if args.human_only else float(args.human_fraction)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    selfplay = load_replay_prefix(args.data, args.max_samples) if use_selfplay else None
    human = load_human_arrays(args.human_tuples) if use_human else None
    if selfplay is not None and human is not None and selfplay.obs.shape[1] != human.obs.shape[1]:
        raise SystemExit(
            f"cannot mix replay obs width {selfplay.obs.shape[1]} with human obs width {human.obs.shape[1]}"
        )
    obs_size = int(selfplay.obs.shape[1] if selfplay is not None else human.obs.shape[1])
    selfplay_train, selfplay_val = _split_indices(len(selfplay), args.seed) if selfplay is not None else (None, None)
    human_train, human_val = _split_indices(len(human), args.seed ^ 0x4855) if human is not None else (None, None)
    selfplay_rows, human_rows = mixed_batch_counts(
        args.batch_size,
        effective_human_fraction,
        has_selfplay=selfplay is not None,
        has_human=human is not None,
    )
    steps_per_epoch = _steps_per_epoch(selfplay_train, human_train, selfplay_rows, human_rows)
    total_steps = int(args.steps) if args.steps is not None else args.epochs * steps_per_epoch
    log_every = args.log_every or steps_per_epoch

    device = select_device(args.device)
    model = build_model(args, obs_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    params = count_parameters(model)
    print(
        f"offline_fit arch={args.arch} obs={obs_size} selfplay={len(selfplay) if selfplay is not None else 0} "
        f"human={len(human) if human is not None else 0} steps={total_steps} device={device} params={params}",
        flush=True,
    )

    selfplay_cycle = _CyclingIndices(selfplay_train, args.seed ^ 0x5150) if selfplay_rows else None
    human_cycle = _CyclingIndices(human_train, args.seed ^ 0x4855) if human_rows else None
    interval = {"selfplay": _MetricAccumulator(), "human": _MetricAccumulator()}
    last_train: dict[str, LossMetrics] = {}
    last_val: dict[str, LossMetrics] = {}
    for step in range(1, total_steps + 1):
        source_losses = _mixed_step(
            model,
            optimizer,
            selfplay=selfplay,
            selfplay_indices=(selfplay_cycle.take(selfplay_rows) if selfplay_cycle is not None else np.empty(0, dtype=np.intp)),
            human=human,
            human_indices=(human_cycle.take(human_rows) if human_cycle is not None else np.empty(0, dtype=np.intp)),
            device=device,
            aux_margin_weight=args.aux_margin_weight,
        )
        for source, (metrics, rows) in source_losses.items():
            interval[source].add(metrics, rows)
        if step % log_every == 0 or step == total_steps:
            last_train = {
                source: metrics
                for source, accumulator in interval.items()
                if (metrics := accumulator.finish()) is not None
            }
            last_val = _source_validation(
                model,
                selfplay=selfplay,
                selfplay_indices=selfplay_val,
                human=human,
                human_indices=human_val,
                batch_size=args.batch_size,
                device=device,
                aux_margin_weight=args.aux_margin_weight,
            )
            _print_trajectory(step, last_train, last_val)
            interval = {"selfplay": _MetricAccumulator(), "human": _MetricAccumulator()}

    final_train = _source_validation(
        model,
        selfplay=selfplay,
        selfplay_indices=selfplay_train,
        human=human,
        human_indices=human_train,
        batch_size=args.batch_size,
        device=device,
        aux_margin_weight=args.aux_margin_weight,
    )
    final_val = _source_validation(
        model,
        selfplay=selfplay,
        selfplay_indices=selfplay_val,
        human=human,
        human_indices=human_val,
        batch_size=args.batch_size,
        device=device,
        aux_margin_weight=args.aux_margin_weight,
    )
    print("final comparison by source")
    print("source       train_ce  train_value  train_aux  val_ce    val_value  val_aux")
    for source in ("human", "selfplay"):
        train_metrics = final_train.get(source)
        val_metrics = final_val.get(source)
        if train_metrics is None or val_metrics is None:
            print(f"{source:<11} n/a       n/a          n/a        n/a       n/a        n/a")
        else:
            train_aux = "n/a" if not math.isfinite(train_metrics.aux_margin) else f"{train_metrics.aux_margin:.6f}"
            val_aux = "n/a" if not math.isfinite(val_metrics.aux_margin) else f"{val_metrics.aux_margin:.6f}"
            print(
                f"{source:<11} {train_metrics.policy:>8.6f}  {train_metrics.value:>11.6f}  "
                f"{train_aux:>9}  {val_metrics.policy:>8.6f}  {val_metrics.value:>9.6f}  {val_aux:>9}",
                flush=True,
            )
    if args.checkpoint_out is not None:
        model_config: dict[str, object]
        if args.arch == "mlp":
            model_config = {
                "arch": "mlp",
                "hidden_sizes": list(args.mlp_hidden_sizes),
                "input_scale": args.mlp_input_scale,
            }
        else:
            model_config = {
                "arch": "card_transformer",
                "obs_version": 2 if obs_size == OBS_SIZE_V2 else 3,
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "ffn_multiplier": args.ffn_multiplier,
                "dropout": args.dropout,
            }
            if args.aux_margin_weight > 0.0:
                model_config["aux_margin_buckets"] = DEFAULT_AUX_MARGIN_BUCKETS
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generation": 0,
                "config": {
                    "model": model_config,
                    "selfplay": {"obs_version": 2 if obs_size == OBS_SIZE_V2 else 3},
                    "aux_margin_weight": float(args.aux_margin_weight),
                },
                "encoder_generation": int(args.encoder_generation),
                "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            },
            args.checkpoint_out,
        )
        print(f"checkpoint={args.checkpoint_out}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
