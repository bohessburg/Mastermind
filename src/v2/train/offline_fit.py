"""Small offline fit harness for comparing C14 MLP and C15 card-token models.

It intentionally consumes a replay snapshot directly rather than touching the
online trainer, workers, or checkpoint format.  With ``--max-samples`` it
streams only a deterministic prefix from the compressed NPZ; that makes a
20k-sample smoke run practical even when the source replay contains millions
of observations.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.v2.train.card_transformer import ACTION_SPACE_SIZE, OBS_SIZE_V2, CardTokenNet
    from src.v2.train.model import DominionNet, count_parameters, masked_policy_loss
else:
    from .card_transformer import ACTION_SPACE_SIZE, OBS_SIZE_V2, CardTokenNet
    from .model import DominionNet, count_parameters, masked_policy_loss


DEFAULT_REPLAY = Path("checkpoints/remote/campaign14/replay_state.npz")


@dataclass(frozen=True)
class ReplayArrays:
    obs: np.ndarray
    policy: np.ndarray
    value: np.ndarray
    legal_mask: np.ndarray

    def __len__(self) -> int:
        return int(self.obs.shape[0])


@dataclass(frozen=True)
class LossMetrics:
    total: float
    policy: float
    value: float
    entropy: float


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


def _read_npy_prefix(archive: zipfile.ZipFile, name: str, rows: int) -> np.ndarray:
    """Read a C-contiguous row prefix without inflating an entire NPZ member."""

    with archive.open(f"{name}.npy") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version in ((2, 0), (3, 0)):
            # NumPy's public API keeps the 2.0 header reader for the
            # UTF-8-bearing 3.0 format as well.
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported .npy version {version} in {name}")
        if fortran_order or not shape:
            raise ValueError(f"{name} must be a non-empty C-order row array")
        if rows > int(shape[0]):
            raise ValueError(f"requested {rows} rows from {name}, archive has {shape[0]}")
        trailing_shape = tuple(int(dim) for dim in shape[1:])
        row_bytes = int(np.prod(trailing_shape, dtype=np.int64) if trailing_shape else 1) * dtype.itemsize
        raw = _read_exact(handle, rows * row_bytes)
    return np.frombuffer(raw, dtype=dtype).reshape((rows, *trailing_shape)).copy()


def load_replay_prefix(path: str | Path, max_samples: int | None) -> ReplayArrays:
    """Load replay data, avoiding a multi-GB allocation for smoke-run prefixes."""

    replay_path = Path(path)
    with zipfile.ZipFile(replay_path) as archive:
        with archive.open("metadata.npy") as handle:
            metadata = json.loads(str(np.load(handle, allow_pickle=False).item()))
        total_rows = int(metadata["size"])
        rows = total_rows if max_samples is None else min(total_rows, int(max_samples))
        if rows < 2:
            raise ValueError("need at least two replay samples for a train/validation split")

        obs = _read_npy_prefix(archive, "obs", rows)
        policy = _read_npy_prefix(archive, "policy", rows)
        value = _read_npy_prefix(archive, "value", rows).reshape(rows)
        legal_mask = _read_npy_prefix(archive, "legal_mask", rows)

    if obs.shape != (rows, OBS_SIZE_V2):
        raise ValueError(f"expected obs shape [{rows}, {OBS_SIZE_V2}], got {obs.shape}")
    if policy.shape != (rows, ACTION_SPACE_SIZE):
        raise ValueError(f"expected policy shape [{rows}, {ACTION_SPACE_SIZE}], got {policy.shape}")
    if legal_mask.shape != (rows, ACTION_SPACE_SIZE):
        raise ValueError(f"expected legal_mask shape [{rows}, {ACTION_SPACE_SIZE}], got {legal_mask.shape}")
    return ReplayArrays(
        obs=np.ascontiguousarray(obs, dtype=np.float32),
        policy=np.ascontiguousarray(policy, dtype=np.float32),
        value=np.ascontiguousarray(value, dtype=np.float32),
        legal_mask=np.ascontiguousarray(legal_mask, dtype=np.bool_),
    )


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


def _losses(
    logits: torch.Tensor,
    values: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    legal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # This is deliberately the same policy loss and value MSE used by
    # train.py's train_step.
    policy_loss, entropy = masked_policy_loss(logits, legal_mask, policy_target)
    value_loss = F.mse_loss(values, value_target)
    return policy_loss + value_loss, policy_loss, value_loss, entropy


def _finish_metrics(total: float, policy: float, value: float, entropy: float, samples: int) -> LossMetrics:
    return LossMetrics(
        total=total / samples,
        policy=policy / samples,
        value=value / samples,
        entropy=entropy / samples,
    )


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: ReplayArrays,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> LossMetrics:
    model.train()
    totals = [0.0, 0.0, 0.0, 0.0]
    for batch_indices in _batches(indices, batch_size):
        obs, policy_target, value_target, legal_mask = _batch_tensors(data, batch_indices, device)
        logits, values = model(obs)
        loss, policy_loss, value_loss, entropy = _losses(
            logits, values, policy_target, value_target, legal_mask
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        weight = len(batch_indices)
        for index, metric in enumerate((loss, policy_loss, value_loss, entropy)):
            totals[index] += float(metric.detach().cpu()) * weight
    return _finish_metrics(*totals, len(indices))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: ReplayArrays,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> LossMetrics:
    model.eval()
    totals = [0.0, 0.0, 0.0, 0.0]
    for batch_indices in _batches(indices, batch_size):
        obs, policy_target, value_target, legal_mask = _batch_tensors(data, batch_indices, device)
        logits, values = model(obs)
        loss, policy_loss, value_loss, entropy = _losses(
            logits, values, policy_target, value_target, legal_mask
        )
        weight = len(batch_indices)
        for index, metric in enumerate((loss, policy_loss, value_loss, entropy)):
            totals[index] += float(metric.detach().cpu()) * weight
    return _finish_metrics(*totals, len(indices))


def parse_hidden_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--mlp-hidden-sizes must be comma-separated integers") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("--mlp-hidden-sizes must contain positive integers")
    return sizes


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.arch == "mlp":
        return DominionNet(
            OBS_SIZE_V2,
            ACTION_SPACE_SIZE,
            hidden_sizes=args.mlp_hidden_sizes,
            input_scale=args.mlp_input_scale,
        )
    return CardTokenNet(
        OBS_SIZE_V2,
        ACTION_SPACE_SIZE,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_multiplier=args.ffn_multiplier,
        dropout=args.dropout,
    )


def select_device(name: str) -> torch.device:
    requested = name.lower()
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("mps requested but unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    return torch.device(requested)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_REPLAY, help="replay_state.npz input")
    parser.add_argument("--arch", choices=("mlp", "transformer"), required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--max-samples", type=int, default=None, help="deterministic archive prefix for quick fits")
    parser.add_argument("--epochs", type=int, default=5)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive")
    if args.lr <= 0:
        raise SystemExit("--lr must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = load_replay_prefix(args.data, args.max_samples)
    split_rng = np.random.default_rng(args.seed)
    shuffled = split_rng.permutation(len(data))
    val_count = max(1, int(round(len(data) * 0.05)))
    val_indices = shuffled[:val_count]
    train_indices = shuffled[val_count:]
    if len(train_indices) == 0:
        raise SystemExit("not enough samples after validation split")

    device = select_device(args.device)
    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    params = count_parameters(model)
    print(
        f"offline_fit arch={args.arch} samples={len(data)} train={len(train_indices)} val={len(val_indices)} "
        f"device={device} params={params}",
        flush=True,
    )

    last_train: LossMetrics | None = None
    last_val: LossMetrics | None = None
    epoch_rng = np.random.default_rng(args.seed ^ 0xC15)
    for epoch in range(1, args.epochs + 1):
        epoch_indices = train_indices[epoch_rng.permutation(len(train_indices))]
        last_train = train_epoch(model, optimizer, data, epoch_indices, args.batch_size, device)
        last_val = evaluate(model, data, val_indices, args.batch_size, device)
        print(
            f"epoch={epoch:02d} "
            f"train(total={last_train.total:.6f}, policy={last_train.policy:.6f}, value={last_train.value:.6f}) "
            f"val(total={last_val.total:.6f}, policy={last_val.policy:.6f}, value={last_val.value:.6f})",
            flush=True,
        )

    assert last_train is not None and last_val is not None
    print("\nfinal comparison")
    print("arch         params    train_total  train_policy  train_value  val_total  val_policy  val_value")
    print(
        f"{args.arch:<12} {params:>7}  {last_train.total:>11.6f}  {last_train.policy:>12.6f}  "
        f"{last_train.value:>11.6f}  {last_val.total:>9.6f}  {last_val.policy:>10.6f}  {last_val.value:>9.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
