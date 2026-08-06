"""Run every standing behaviour probe for one checkpoint and merge its JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from _common import default_output, write_json
import chapel_probe
import duchy_probe
import human_record_probe
import militia_probe
import value_probe


def run(
    checkpoint: str | Path,
    *,
    exports: str | Path = "exports",
    skips: set[str] | None = None,
    legacy_shim: bool = False,
) -> dict[str, Any]:
    skipped = skips or set()
    probes: dict[str, Any] = {}
    runners: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
        ("value", lambda: value_probe.run(checkpoint, legacy_shim=legacy_shim)),
        ("chapel", lambda: chapel_probe.run(checkpoint, legacy_shim=legacy_shim)),
        ("militia", lambda: militia_probe.run(checkpoint, legacy_shim=legacy_shim)),
        ("duchy", lambda: duchy_probe.run(checkpoint, legacy_shim=legacy_shim)),
        (
            "human_record",
            lambda: human_record_probe.run(checkpoint, exports, legacy_shim=legacy_shim),
        ),
    )
    for name, runner in runners:
        probes[name] = {"skipped": True} if name in skipped else runner()
    return {
        "probe": "run_all",
        "checkpoint": str(checkpoint),
        "skipped": sorted(skipped),
        "probes": probes,
    }


def scorecard(result: dict[str, Any]) -> str:
    lines = [f"run_all checkpoint={result['checkpoint']}"]
    rendered = {
        "value": value_probe.scorecard,
        "chapel": chapel_probe.scorecard,
        "militia": militia_probe.scorecard,
        "duchy": duchy_probe.scorecard,
        "human_record": human_record_probe.scorecard,
    }
    for name, formatter in rendered.items():
        probe = result["probes"][name]
        lines.append(f"{name}: skipped" if probe.get("skipped") else formatter(probe))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--legacy-shim", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--exports", type=Path, default=Path("exports"))
    parser.add_argument("--skip-value", action="store_true")
    parser.add_argument("--skip-chapel", action="store_true")
    parser.add_argument("--skip-militia", action="store_true")
    parser.add_argument("--skip-duchy", action="store_true")
    parser.add_argument("--skip-human-record", action="store_true")
    args = parser.parse_args()
    skips = {
        name
        for name, enabled in {
            "value": args.skip_value,
            "chapel": args.skip_chapel,
            "militia": args.skip_militia,
            "duchy": args.skip_duchy,
            "human_record": args.skip_human_record,
        }.items()
        if enabled
    }
    result = run(
        args.checkpoint,
        exports=args.exports,
        skips=skips,
        legacy_shim=args.legacy_shim,
    )
    output = write_json(args.out or default_output(args.checkpoint, "run_all"), result)
    print(scorecard(result))
    print(f"json={output}")


if __name__ == "__main__":
    main()
