#!/usr/bin/env python3
"""Compare a bench run against this machine's baseline with a ±15% gate.

Usage:
    ./build/v2_bench | python3 bench/check_regression.py
    python3 bench/check_regression.py bench/current.json

The baseline (bench/baseline.json) must have been recorded on the SAME
machine (regenerate with: ./build/v2_bench > bench/baseline.json).
CI does not run this gate — runner hardware differs from the baseline.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main() -> int:
    baseline = json.loads((ROOT / "baseline.json").read_text())
    if len(sys.argv) > 1:
        current = json.loads(Path(sys.argv[1]).read_text())
    else:
        current = json.load(sys.stdin)

    failures = []
    if current["step_median_ns"] > baseline["step_median_ns"] * 1.15:
        failures.append(
            f"step_median_ns {current['step_median_ns']} > "
            f"{baseline['step_median_ns'] * 1.15:.2f}"
        )
    if current["random_games_per_sec"] < baseline["random_games_per_sec"] * 0.85:
        failures.append(
            f"random_games_per_sec {current['random_games_per_sec']} < "
            f"{baseline['random_games_per_sec'] * 0.85:.2f}"
        )

    print("baseline:", json.dumps(baseline, sort_keys=True))
    print("current: ", json.dumps(current, sort_keys=True))
    if failures:
        print("bench regression:", "; ".join(failures), file=sys.stderr)
        return 1
    print("bench OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())
