"""Classify seed-replay web exports without changing the source corpus.

The web server's local export directory has historically also accumulated
pytest games and unstarted lobby sessions.  This module creates a compact,
refreshable manifest so downstream consumers can select the training corpus
without moving or deleting any source file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .local import is_complete_local_export_data


MANIFEST_NAME = "corpus_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
CLASSIFICATIONS = ("real", "pytest_artifact", "stub", "incomplete")


def manifest_path(exports_root: Path | str = Path("exports")) -> Path:
    return Path(exports_root) / MANIFEST_NAME


def source_files(exports_root: Path | str = Path("exports")) -> tuple[Path, ...]:
    """Return just the local and Hetzner seed-export JSON files.

    Records, arena archives, tuple output, and the manifest itself are all
    deliberately outside this inventory.
    """
    root = Path(exports_root)
    paths: list[Path] = []
    for directory in (root, root / "hetzner"):
        if not directory.is_dir():
            continue
        paths.extend(
            path
            for path in sorted(directory.glob("*.json"))
            if path.name != MANIFEST_NAME and not path.name.endswith(".game-record.json")
        )
    return tuple(paths)


def classify_export_data(data: object) -> str:
    """Classify one decoded source export.

    Classification order is intentional: a pytest game remains quarantined
    even when it is complete, and a zero-action lobby remains a stub even
    though it cannot be a complete replay.
    """
    if not isinstance(data, Mapping):
        return "incomplete"
    seats = data.get("seats")
    seat_kinds = seats if isinstance(seats, list) else []
    if any("pytest" in str(kind).lower() for kind in seat_kinds):
        return "pytest_artifact"
    actions = data.get("actions")
    if isinstance(actions, list) and not actions:
        return "stub"
    if not is_complete_local_export_data(data):
        return "incomplete"
    return "real"


def build_manifest(exports_root: Path | str = Path("exports")) -> dict[str, Any]:
    """Scan both seed-export source directories and write their manifest."""
    root = Path(exports_root)
    files: list[dict[str, Any]] = []
    for path in source_files(root):
        stat = path.stat()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        seats = data.get("seats") if isinstance(data, Mapping) else []
        actions = data.get("actions") if isinstance(data, Mapping) else []
        relative_path = path.relative_to(root).as_posix()
        source_directory = root.name
        if path.parent == root / "hetzner":
            source_directory = f"{root.name}/hetzner"
        files.append(
            {
                "path": relative_path,
                "source_directory": source_directory,
                "classification": classify_export_data(data),
                "seat_kinds": [str(kind) for kind in seats] if isinstance(seats, list) else [],
                "action_count": len(actions) if isinstance(actions, list) else 0,
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )

    files.sort(key=lambda item: str(item["path"]))
    totals = _totals(files)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_root": root.name,
        "files": files,
        "totals": totals,
    }
    destination = manifest_path(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def manifest_is_current(
    manifest: Mapping[str, Any], exports_root: Path | str = Path("exports")
) -> bool:
    """Return whether file membership, sizes, and mtimes match the manifest."""
    root = Path(exports_root)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return False
    entries = manifest.get("files")
    if not isinstance(entries, list):
        return False
    expected: dict[str, tuple[int, int]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            return False
        path = entry.get("path")
        size = entry.get("size_bytes")
        mtime = entry.get("mtime_ns")
        if not isinstance(path, str) or not isinstance(size, int) or not isinstance(mtime, int):
            return False
        expected[path] = (size, mtime)
    actual = {
        path.relative_to(root).as_posix(): (int(path.stat().st_size), int(path.stat().st_mtime_ns))
        for path in source_files(root)
    }
    return expected == actual


def load_or_create_manifest(exports_root: Path | str = Path("exports")) -> dict[str, Any]:
    """Read a current manifest, regenerating it when it is absent or stale."""
    root = Path(exports_root)
    destination = manifest_path(root)
    try:
        loaded = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return build_manifest(root)
    if not isinstance(loaded, dict) or not manifest_is_current(loaded, root):
        return build_manifest(root)
    return loaded


def manifest_entries(
    manifest: Mapping[str, Any],
    *,
    classifications: Iterable[str] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Return manifest rows, optionally restricted to classification labels."""
    allowed = set(classifications) if classifications is not None else None
    rows = manifest.get("files", [])
    if not isinstance(rows, list):
        return ()
    return tuple(
        row
        for row in rows
        if isinstance(row, Mapping)
        and (allowed is None or row.get("classification") in allowed)
    )


def paths_for_classifications(
    exports_root: Path | str,
    classifications: Iterable[str],
) -> tuple[Path, ...]:
    """Resolve source files selected by a current manifest."""
    root = Path(exports_root)
    manifest = load_or_create_manifest(root)
    return tuple(root / str(entry["path"]) for entry in manifest_entries(manifest, classifications=classifications))


def _totals(files: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    totals: dict[str, Counter[str]] = {}
    for entry in files:
        source = str(entry["source_directory"])
        totals.setdefault(source, Counter())[str(entry["classification"])] += 1
    return {
        source: {classification: int(counter.get(classification, 0)) for classification in CLASSIFICATIONS}
        for source, counter in sorted(totals.items())
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify local Dominion web-export corpus files.")
    parser.add_argument(
        "--exports-root",
        type=Path,
        default=Path("exports"),
        help="root containing local exports and an optional hetzner/ directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_manifest(args.exports_root)
    print(manifest_path(args.exports_root))
    for source, counts in manifest["totals"].items():
        print(
            f"{source}: " + ", ".join(f"{kind}={counts[kind]}" for kind in CLASSIFICATIONS)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
