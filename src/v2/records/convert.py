"""CLI and provenance autodetection for unified game record conversion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .arena import convert_arena_archive, is_arena_game_dir
from .local import (
    convert_local_export,
    is_complete_local_export_data,
    is_local_export_data,
)
from .model import GameRecord
from .writer import write_record


def discover_sources(path: Path | str) -> tuple[Path, ...]:
    """Discover local exports or arena game directories under one path."""
    candidate = Path(path)
    if candidate.is_file():
        return (candidate,) if _is_local_export_file(candidate) else ()
    if is_arena_game_dir(candidate):
        return (candidate,)
    if not candidate.is_dir():
        return ()

    sources: list[Path] = []
    sources.extend(
        child
        for child in sorted(candidate.glob("*.json"))
        if _is_local_export_file(child)
    )
    sources.extend(
        child
        for child in sorted(candidate.rglob("*-game-*"))
        if is_arena_game_dir(child)
    )
    return tuple(dict.fromkeys(sources))


def convert_source(path: Path | str) -> GameRecord:
    """Autodetect and convert exactly one source."""
    source = Path(path)
    if is_arena_game_dir(source):
        return convert_arena_archive(source)
    if source.is_file() and _is_local_export_file(source):
        return convert_local_export(source)
    raise ValueError(f"unrecognized game source: {source}")


def output_path_for(
    source: Path,
    record: GameRecord,
    output_dir: Path | None,
) -> Path:
    """Choose a collision-resistant conventional output path."""
    if output_dir is not None:
        return output_dir / f"{record.source}-{record.game_id}.game-record.json"
    if record.source == "arena":
        return source / "game-record.json"
    return source.with_name(f"{source.stem}.game-record.json")


def convert_paths(
    paths: Sequence[Path | str],
    *,
    output_dir: Path | str | None = None,
) -> tuple[Path, ...]:
    """Convert all discovered sources with the one shared writer."""
    destination_root = Path(output_dir) if output_dir is not None else None
    sources: list[Path] = []
    for path in paths:
        sources.extend(discover_sources(path))
    unique_sources = tuple(dict.fromkeys(sources))
    if not unique_sources:
        raise ValueError("no local exports or arena game archives found")

    written: list[Path] = []
    for source in unique_sources:
        record = convert_source(source)
        destination = output_path_for(source, record, destination_root)
        written.append(write_record(record, destination))
    return tuple(written)


def emit_arena_record(archive: Path | str) -> Path:
    """Emit the conventional record next to a completed arena result."""
    source = Path(archive)
    record = convert_arena_archive(source)
    return write_record(record, source / "game-record.json")


def _is_local_export_file(path: Path) -> bool:
    if not path.is_file() or path.name.endswith(".game-record.json"):
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return is_local_export_data(value) and is_complete_local_export_data(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert local web exports or arena archives to game records."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="local export, arena game/session directory, or export root",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="write all records into this directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the converter CLI."""
    args = _parser().parse_args(argv)
    try:
        written = convert_paths(args.paths, output_dir=args.out)
    except (OSError, ValueError, KeyError) as error:
        print(f"game-record conversion failed: {error}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    print(f"converted {len(written)} game(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
