"""Empirically calibrate the Duchy value tripwire on sampled game states."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

import duchy_probe
from _common import write_json


def _canonical(path: str | Path) -> Path:
    return Path(path).resolve()


def _legacy_shim_targets(values: list[str]) -> tuple[bool, set[Path]]:
    """Support a global shim or one explicit ``--legacy-shim CHECKPOINT``."""
    global_shim = "__all__" in values
    targets = {_canonical(value) for value in values if value != "__all__"}
    return global_shim, targets


def calibrate(
    states_path: str | Path,
    checkpoints: list[str | Path],
    *,
    healthy_checkpoints: list[str | Path] | None = None,
    legacy_shim_all: bool = False,
    legacy_shim_targets: set[Path] | None = None,
) -> dict[str, Any]:
    """Measure all checkpoints and derive a lower three-sigma dV tripline."""
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    selected = [Path(checkpoint) for checkpoint in checkpoints]
    target_set = legacy_shim_targets or set()
    healthy_set = (
        {_canonical(checkpoint) for checkpoint in healthy_checkpoints}
        if healthy_checkpoints is not None
        else {_canonical(checkpoint) for checkpoint in selected}
    )
    available_set = {_canonical(checkpoint) for checkpoint in selected}
    unknown_shim_targets = target_set - available_set
    if unknown_shim_targets:
        names = ", ".join(str(path) for path in sorted(unknown_shim_targets))
        raise ValueError(f"legacy-shim checkpoint is not in --checkpoints: {names}")
    unknown_healthy = healthy_set - available_set
    if unknown_healthy:
        names = ", ".join(str(path) for path in sorted(unknown_healthy))
        raise ValueError(f"healthy checkpoint is not in --checkpoints: {names}")
    if not healthy_set:
        raise ValueError("healthy reference set must not be empty")

    print(f"calibrate_duchy states={states_path} checkpoints={len(selected)}", flush=True)
    rows: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(selected, start=1):
        use_legacy_shim = legacy_shim_all or _canonical(checkpoint) in target_set
        print(
            f"calibrate_duchy checkpoint={index}/{len(selected)} path={checkpoint} "
            f"legacy_shim={use_legacy_shim} start",
            flush=True,
        )
        result = duchy_probe.run(
            checkpoint,
            legacy_shim=use_legacy_shim,
            states_path=states_path,
        )
        means = result["means"]
        row = {
            "checkpoint": str(checkpoint),
            "legacy_shim": bool(use_legacy_shim),
            "obs_version": int(result["obs_version"]),
            "states": len(result["states"]),
            "dV": float(means["value_delta"]),
            "dPts": float(means["p_buy_duchy_delta_points"]),
            "dP": float(means["p_buy_duchy_delta"]),
            "healthy_reference": _canonical(checkpoint) in healthy_set,
        }
        rows.append(row)
        print(
            f"calibrate_duchy checkpoint={index}/{len(selected)} dV={row['dV']:+.6f} "
            f"dPts={row['dPts']:+.3f} done",
            flush=True,
        )

    healthy_dv = np.asarray(
        [row["dV"] for row in rows if row["healthy_reference"]], dtype=np.float64
    )
    mean = float(np.mean(healthy_dv))
    std = float(np.std(healthy_dv, ddof=0))
    threshold = float(mean - 3.0 * std)
    print(
        f"calibrate_duchy healthy_n={healthy_dv.size} dV_mean={mean:+.6f} "
        f"dV_std={std:.6f} recommended_trip_threshold={threshold:+.6f}",
        flush=True,
    )
    return {
        "probe": "duchy_calibration",
        "states_file": str(states_path),
        "measurement": {
            "counterfactual": "opponent collection Duchy count +2 (+6 VP)",
            "dV": "mean injected value minus base value",
            "dPts": "mean injected P(Buy Duchy) minus base probability, in percentage points",
        },
        "checkpoints": rows,
        "healthy_reference": {
            "checkpoints": [row["checkpoint"] for row in rows if row["healthy_reference"]],
            "n": int(healthy_dv.size),
            "dV_mean": mean,
            "dV_std": std,
            "std_ddof": 0,
            "recommended_trip_threshold": threshold,
            "rule": "dV_mean - 3 * dV_std",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--healthy-checkpoints",
        nargs="+",
        type=Path,
        help="healthy subset used for mean/std; defaults to all --checkpoints",
    )
    parser.add_argument(
        "--legacy-shim",
        nargs="?",
        const="__all__",
        action="append",
        default=[],
        metavar="CHECKPOINT",
        help=(
            "use with no value for every checkpoint, or repeat as "
            "--legacy-shim path/to/c15.pt for only that checkpoint"
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("bench/duchy_calibration.json"))
    args = parser.parse_args()
    legacy_all, legacy_targets = _legacy_shim_targets(args.legacy_shim)
    result = calibrate(
        args.states,
        args.checkpoints,
        healthy_checkpoints=args.healthy_checkpoints,
        legacy_shim_all=legacy_all,
        legacy_shim_targets=legacy_targets,
    )
    output = write_json(args.out, result)
    print(f"calibrate_duchy json={output}", flush=True)


if __name__ == "__main__":
    main()
