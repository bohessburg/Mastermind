#!/usr/bin/env python3
"""Merge the c21 imitation corpora into deterministic game-disjoint splits.

The output deliberately uses only the tuple arrays consumed by
``src.v2.train.human_data.load_human_tuples``.  The source manifests use
overlapping ``game_index`` domains, so indices are reassigned independently
inside each split while the original game record and source remain in the
manifest for provenance.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ARRAYS = (
    "obs",
    "action",
    "legal",
    "value",
    "margin",
    "winner",
    "seat_index",
    "game_index",
    "ply_index",
    "turn_number",
)
SOURCES = (("seeded", Path("exports/tuples")), ("dgames", Path("exports/tuples_dgames")))
OUTPUT_MARKER = ".c21_merge_corpus"


@dataclass(frozen=True)
class SourceCorpus:
    name: str
    root: Path
    manifest: dict[str, Any]
    games: dict[int, dict[str, Any]]
    active_game_indices: frozenset[int]
    tuple_count: int


class SplitShardWriter:
    """Stream one provenance-homogeneous split into loader-compatible shards."""

    def __init__(self, root: Path, source: str, shard_size: int) -> None:
        self.root = root
        self.source = source
        self.shard_size = shard_size
        self._parts: dict[str, list[np.ndarray]] = {name: [] for name in REQUIRED_ARRAYS}
        self._rows = 0
        self._number = 0
        self.entries: list[dict[str, Any]] = []

    def append(self, arrays: dict[str, np.ndarray]) -> None:
        rows = int(arrays["action"].shape[0])
        if rows == 0:
            return
        start = 0
        while start < rows:
            available = self.shard_size - self._rows
            take = min(available, rows - start)
            for name in REQUIRED_ARRAYS:
                self._parts[name].append(np.ascontiguousarray(arrays[name][start : start + take]))
            self._rows += take
            start += take
            if self._rows == self.shard_size:
                self.flush()

    def flush(self) -> None:
        if self._rows == 0:
            return
        filename = f"tuples-{self.source}-{self._number:05d}.npz"
        destination = self.root / filename
        payload = {
            name: (parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0))
            for name, parts in self._parts.items()
        }
        np.savez_compressed(destination, **payload)
        self.entries.append({"path": filename, "tuples": self._rows, "source": self.source})
        print(f"merge wrote split={self.root.name} source={self.source} shard={filename} tuples={self._rows}", flush=True)
        self._parts = {name: [] for name in REQUIRED_ARRAYS}
        self._rows = 0
        self._number += 1


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "tuple_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"tuple manifest not found: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"tuple manifest root must be an object: {path}")
    if not isinstance(manifest.get("obs_width"), int) or not isinstance(manifest.get("action_width"), int):
        raise ValueError(f"tuple manifest is missing positive widths: {path}")
    if not isinstance(manifest.get("games"), list) or not isinstance(manifest.get("shards"), list):
        raise ValueError(f"tuple manifest is missing games or shards: {path}")
    return manifest


def inspect_source(name: str, root: Path) -> SourceCorpus:
    """Validate source shard fields and return only games that have rows."""

    manifest = _load_manifest(root)
    games: dict[int, dict[str, Any]] = {}
    for raw_game in manifest["games"]:
        if not isinstance(raw_game, dict) or not isinstance(raw_game.get("index"), int):
            raise ValueError(f"source {name} has a game without an integer index")
        index = int(raw_game["index"])
        if index in games:
            raise ValueError(f"source {name} repeats game index {index}")
        games[index] = raw_game

    active_games: set[int] = set()
    tuple_count = 0
    for shard_number, entry in enumerate(manifest["shards"], start=1):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"source {name} has an invalid shard entry")
        shard_path = root / entry["path"]
        with np.load(shard_path, allow_pickle=False) as shard:
            missing = [field for field in REQUIRED_ARRAYS if field not in shard]
            if missing:
                raise ValueError(f"source {name} shard {shard_path} is missing {', '.join(missing)}")
            rows = int(shard["action"].shape[0])
            if shard["obs"].shape != (rows, int(manifest["obs_width"])):
                raise ValueError(f"source {name} shard {shard_path} has an incompatible obs shape")
            if shard["legal"].shape != (rows, int(manifest["action_width"])):
                raise ValueError(f"source {name} shard {shard_path} has an incompatible legal shape")
            for field in ("action", "value", "margin", "winner", "seat_index", "game_index", "ply_index", "turn_number"):
                if shard[field].shape != (rows,):
                    raise ValueError(f"source {name} shard {shard_path} has an incompatible {field} shape")
            game_indices = np.unique(shard["game_index"])
            unknown = [int(index) for index in game_indices if int(index) not in games]
            if unknown:
                raise ValueError(f"source {name} shard {shard_path} references unknown game indices {unknown[:5]}")
            active_games.update(int(index) for index in game_indices)
            tuple_count += rows
        print(
            f"merge inspected source={name} shard={shard_number}/{len(manifest['shards'])} tuples={tuple_count}",
            flush=True,
        )

    reported = manifest.get("totals", {}).get("tuples_exported")
    if reported is not None and int(reported) != tuple_count:
        raise ValueError(f"source {name} manifest reports {reported} tuples, found {tuple_count}")
    if not active_games:
        raise ValueError(f"source {name} has no exported games")
    return SourceCorpus(name, root, manifest, games, frozenset(active_games), tuple_count)


def _split_game_keys(corpora: list[SourceCorpus], validation_fraction: float, seed: int) -> set[tuple[str, int]]:
    game_keys = [(corpus.name, index) for corpus in corpora for index in sorted(corpus.active_game_indices)]
    validation_count = max(1, int(round(len(game_keys) * validation_fraction)))
    validation_count = min(validation_count, len(game_keys) - 1)
    if validation_count <= 0:
        raise ValueError("at least two exported games are required for a validation split")
    positions = np.random.default_rng(seed).permutation(len(game_keys))[:validation_count]
    return {game_keys[int(position)] for position in positions}


def _split_game_metadata(
    corpora: list[SourceCorpus], validation_keys: set[tuple[str, int]], split: str
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], int]]:
    selected = {
        (corpus.name, source_index)
        for corpus in corpora
        for source_index in corpus.active_game_indices
        if ((corpus.name, source_index) in validation_keys) == (split == "val")
    }
    games: list[dict[str, Any]] = []
    for output_index, (source, source_index) in enumerate(sorted(selected)):
        corpus = next(corpus for corpus in corpora if corpus.name == source)
        game = dict(corpus.games[source_index])
        game["index"] = output_index
        game["source"] = source
        game["source_index"] = source_index
        games.append(game)
    return games, {key: index for index, key in enumerate(sorted(selected))}


def _write_split(
    output_root: Path,
    split: str,
    corpora: list[SourceCorpus],
    validation_keys: set[tuple[str, int]],
    shard_size: int,
    seed: int,
    validation_fraction: float,
) -> dict[str, Any]:
    destination = output_root / split
    destination.mkdir(parents=True, exist_ok=True)
    games, remap = _split_game_metadata(corpora, validation_keys, split)
    writers = {corpus.name: SplitShardWriter(destination, corpus.name, shard_size) for corpus in corpora}
    tuple_counts: Counter[str] = Counter()

    for corpus in corpora:
        writer = writers[corpus.name]
        source_remap = {
            source_index: remap[(corpus.name, source_index)]
            for source_index in corpus.active_game_indices
            if (corpus.name, source_index) in remap
        }
        for shard_number, entry in enumerate(corpus.manifest["shards"], start=1):
            with np.load(corpus.root / entry["path"], allow_pickle=False) as shard:
                source_indices = shard["game_index"]
                selected = np.isin(source_indices, np.fromiter(source_remap, dtype=np.int32))
                if bool(selected.any()):
                    arrays = {name: np.asarray(shard[name][selected]) for name in REQUIRED_ARRAYS}
                    arrays["game_index"] = np.fromiter(
                        (source_remap[int(index)] for index in source_indices[selected]), dtype=np.int32
                    )
                    writer.append(arrays)
                    tuple_counts[corpus.name] += int(arrays["action"].shape[0])
            print(
                f"merge routing split={split} source={corpus.name} shard={shard_number}/{len(corpus.manifest['shards'])} "
                f"tuples={tuple_counts[corpus.name]}",
                flush=True,
            )
        writer.flush()

    shards = [entry for writer in writers.values() for entry in writer.entries]
    game_counts = Counter(game["source"] for game in games)
    total_tuples = int(sum(tuple_counts.values()))
    manifest = {
        "schema_version": 2,
        "obs_version": int(corpora[0].manifest.get("obs_version", 3)),
        "obs_width": int(corpora[0].manifest["obs_width"]),
        "action_width": int(corpora[0].manifest["action_width"]),
        "value_target": corpora[0].manifest.get("value_target", "margin"),
        "split": split,
        "split_seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "source_manifests": {corpus.name: str(corpus.root / "tuple_manifest.json") for corpus in corpora},
        "games": games,
        "shards": shards,
        "totals": {
            "tuples_exported": total_tuples,
            "games_processed": len(games),
            "tuples_by_source": dict(sorted(tuple_counts.items())),
            "games_by_source": dict(sorted(game_counts.items())),
        },
    }
    (destination / "tuple_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _verify_outputs(output_root: Path, train: dict[str, Any], val: dict[str, Any]) -> None:
    """Check both game-disjoint provenance and the actual imitation loader."""

    train_games = {(game["source"], game["source_index"]) for game in train["games"]}
    val_games = {(game["source"], game["source_index"]) for game in val["games"]}
    if train_games & val_games:
        raise AssertionError("a game appears in both train and validation manifests")
    for split, manifest in (("train", train), ("val", val)):
        declared_games = {int(game["index"]) for game in manifest["games"]}
        observed_games: set[int] = set()
        observed_rows = 0
        for entry in manifest["shards"]:
            if entry.get("source") not in {"seeded", "dgames"}:
                raise AssertionError(f"{split} shard is missing a valid source field")
            with np.load(output_root / split / entry["path"], allow_pickle=False) as shard:
                observed_rows += int(shard["action"].shape[0])
                observed_games.update(int(index) for index in np.unique(shard["game_index"]))
        if observed_rows != int(manifest["totals"]["tuples_exported"]):
            raise AssertionError(f"{split} manifest tuple total does not match shards")
        if observed_games != declared_games:
            raise AssertionError(f"{split} shard game indices do not match manifest games")

    # Import after writing so the actual public loader, rather than a local
    # reimplementation of its assumptions, proves the output is consumable.
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from src.v2.train.human_data import load_human_tuples

    for split, manifest in (("train", train), ("val", val)):
        dataset = load_human_tuples(output_root / split)
        if len(dataset) != int(manifest["totals"]["tuples_exported"]):
            raise AssertionError(f"imitation loader count mismatch for {split}")
        print(f"merge loader validation split={split} tuples={len(dataset)} games={len(manifest['games'])}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeded-dir", type=Path, default=Path("exports/tuples"))
    parser.add_argument("--dgames-dir", type=Path, default=Path("exports/tuples_dgames"))
    parser.add_argument("--output-dir", type=Path, default=Path("exports/tuples_all"))
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--shard-size", type=int, default=10_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be between zero and one")
    if args.shard_size <= 0:
        raise SystemExit("--shard-size must be positive")

    source_paths = (("seeded", args.seeded_dir), ("dgames", args.dgames_dir))
    corpora = [inspect_source(name, root) for name, root in source_paths]
    if len({int(corpus.manifest["obs_width"]) for corpus in corpora}) != 1:
        raise SystemExit("source corpora disagree on obs_width")
    if len({int(corpus.manifest["action_width"]) for corpus in corpora}) != 1:
        raise SystemExit("source corpora disagree on action_width")

    validation_keys = _split_game_keys(corpora, args.validation_fraction, args.seed)
    print(
        f"merge split selected validation_games={len(validation_keys)} total_games="
        f"{sum(len(corpus.active_game_indices) for corpus in corpora)} seed={args.seed}",
        flush=True,
    )
    marker = args.output_dir / OUTPUT_MARKER
    if args.output_dir.exists():
        if not marker.is_file():
            raise SystemExit(
                f"refusing to replace unmanaged output directory {args.output_dir}; choose a fresh --output-dir"
            )
        print(f"merge removing previous output={args.output_dir}", flush=True)
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / OUTPUT_MARKER).write_text("c21_merge_corpus\n", encoding="utf-8")

    train = _write_split(
        args.output_dir,
        "train",
        corpora,
        validation_keys,
        args.shard_size,
        args.seed,
        args.validation_fraction,
    )
    val = _write_split(
        args.output_dir,
        "val",
        corpora,
        validation_keys,
        args.shard_size,
        args.seed,
        args.validation_fraction,
    )
    _verify_outputs(args.output_dir, train, val)
    print(
        "merge complete "
        f"train tuples={train['totals']['tuples_exported']} games={len(train['games'])} "
        f"val tuples={val['totals']['tuples_exported']} games={len(val['games'])} "
        f"total tuples={train['totals']['tuples_exported'] + val['totals']['tuples_exported']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
