"""Measure policy/value ``evaluate`` throughput for the v2 network architectures."""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.v2.train.card_transformer import ACTION_SPACE_SIZE, OBS_SIZE_V2
    from src.v2.train.inference_server import compile_server_evaluator
    from src.v2.train.model import build_model, count_parameters
else:
    from .card_transformer import ACTION_SPACE_SIZE, OBS_SIZE_V2
    from .inference_server import compile_server_evaluator
    from .model import build_model, count_parameters


def parse_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--batch-sizes must be comma-separated positive integers") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("--batch-sizes must contain positive integers")
    return sizes


def parse_hidden_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--mlp-hidden-sizes must be comma-separated positive integers") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("--mlp-hidden-sizes must contain positive integers")
    return sizes


def select_device(name: str) -> torch.device:
    requested = name.lower()
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("mps requested but unavailable")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("mlp", "card_transformer"), required=True)
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=(64, 256, 1024))
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--mlp-hidden-sizes", type=parse_hidden_sizes, default=(1536, 1536, 768))
    parser.add_argument("--mlp-input-scale", type=float, default=16.0)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="run the server evaluation wrapper through torch.compile",
    )
    parser.add_argument("--bf16", action="store_true", help="run evaluations under bfloat16 autocast")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    device = select_device(args.device)
    if args.bf16 and device.type not in {"cpu", "cuda"}:
        raise SystemExit("--bf16 is supported by this benchmark only on cpu or cuda")
    model_config = {
        "arch": args.arch,
        "hidden_sizes": args.mlp_hidden_sizes,
        "input_scale": args.mlp_input_scale,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "ffn_multiplier": args.ffn_multiplier,
        "dropout": args.dropout,
    }
    model = build_model(model_config, OBS_SIZE_V2, ACTION_SPACE_SIZE).to(device)
    model.eval()
    evaluator = compile_server_evaluator(model) if args.compile else None

    print(
        f"arch={args.arch} device={device.type} params={count_parameters(model)} iters={args.iters} "
        f"compile={args.compile} bf16={args.bf16}",
        flush=True,
    )
    print("batch_size  evals/sec  ms/batch", flush=True)
    autocast_context = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16) if args.bf16 else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        for batch_size in args.batch_sizes:
            obs = torch.empty((batch_size, OBS_SIZE_V2), dtype=torch.float32, device=device).uniform_(0.0, 40.0)
            legal_mask = torch.ones((batch_size, ACTION_SPACE_SIZE), dtype=torch.bool, device=device)
            for _ in range(args.warmup):
                if evaluator is None:
                    model.evaluate(obs, legal_mask)
                else:
                    evaluator(obs, legal_mask)
            synchronize(device)

            start = time.perf_counter()
            for _ in range(args.iters):
                if evaluator is None:
                    model.evaluate(obs, legal_mask)
                else:
                    evaluator(obs, legal_mask)
            synchronize(device)
            elapsed = time.perf_counter() - start
            evals_per_second = (batch_size * args.iters) / elapsed
            milliseconds_per_batch = 1000.0 * elapsed / args.iters
            print(f"{batch_size:>10}  {evals_per_second:>9.1f}  {milliseconds_per_batch:>8.3f}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
