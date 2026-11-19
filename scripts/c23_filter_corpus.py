#!/usr/bin/env python3
"""Build a rating-filtered, game-disjoint imitation corpus.

The engine-generated seeded corpus is retained in full.  Rows from the
dominion.games corpus are retained only when the manifest game/player seat
joins to a finite rating level at or above ``--min-level``.  In particular,
this removes both the policy and value targets for excluded human rows rather
than merely assigning them a small policy weight.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Reuse c21's source inspection, deterministic split, shard writer, and—most
# importantly—the actual public-loader validation instead of duplicating its
# assumptions here.
from scripts.c21_merge_corpus import (  # noqa: E402
    REQUIRED_ARRAYS,
    SourceCorpus,
    SplitShardWriter,
    _split_game_keys,
    _split_game_metadata,
    _verify_outputs,
    inspect_source,
)
from src.v2.train.human_data import (  # noqa: E402
    RATING_BAND_EXPERT,
    RATING_BAND_LOW,
    RATING_BAND_RAMP,
    RATING_BAND_UNRATED,
    _manifest_rating_games,
    _rating_entry_level_and_deviation,
    rating_band_for_level,
)


OUTPUT_MARKER = ".c23_filter_corpus"
SOURCE_SEEDED = "seeded"
SOURCE_DGAMES = "dgames"

CATEGORY_KEPT_BELOW_40 = "kept_below_40"
CATEGORY_KEPT_40_TO_50 = "kept_40_to_50"
CATEGORY_KEPT_50_PLUS = "kept_50_plus"
CATEGORY_DROPPED_BELOW_MIN = "dropped_below_min_level"
CATEGORY_DROPPED_UNRATED = "dropped_unrated"


class SeatRatingFilter:
    """Resolve manifest tuple seats through the same ratings join as the loader."""

    def __init__(self, manifest: dict[str, Any], sidecar: dict[str, Any], min_level: float) -> None:
        # These private helpers are intentionally imported from human_data:
        # they validate ``games[index].id`` / ``player_ids`` and normalize the
        # sidecar's level field exactly as runtime skill weighting does.
        self._games = _manifest_rating_games(manifest)
        self._sidecar = sidecar
        self._min_level = min_level
        self._decisions: dict[tuple[int, int], tuple[bool, str]] = {}

    def decision(self, game_index: int, seat_index: int) -> tuple[bool, str]:
        """Return whether a row stays and its summary category."""

        key = (game_index, seat_index)
        cached = self._decisions.get(key)
        if cached is not None:
            return cached

        game = self._games.get(game_index)
        if game is None:
            raise ValueError(f"human tuple row references unknown game_index {game_index}")
        game_id, player_ids = game
        if player_ids is None:
            level = None
        elif not 0 <= seat_index < len(player_ids):
            raise ValueError(f"human tuple row has invalid seat_index {seat_index}")
        else:
            # This is the lookup used by human_data._load_policy_weights:
            # game id -> stringified player id -> rating entry.
            game_ratings = self._sidecar.get(game_id)
            entry = game_ratings.get(str(player_ids[seat_index])) if isinstance(game_ratings, dict) else None
            level, _deviation = _rating_entry_level_and_deviation(entry)

        if level is None or rating_band_for_level(level) == RATING_BAND_UNRATED:
            result = (False, CATEGORY_DROPPED_UNRATED)
        elif level < self._min_level:
            result = (False, CATEGORY_DROPPED_BELOW_MIN)
        elif rating_band_for_level(level) == RATING_BAND_EXPERT:
            result = (True, CATEGORY_KEPT_50_PLUS)
        elif rating_band_for_level(level) == RATING_BAND_RAMP:
            result = (True, CATEGORY_KEPT_40_TO_50)
        else:
            # This is possible only with --min-level below 40.  It keeps the
            # default summary bands unambiguous while still honoring the flag.
            assert rating_band_for_level(level) == RATING_BAND_LOW
            result = (True, CATEGORY_KEPT_BELOW_40)
        self._decisions[key] = result
        return result

    def select(self, game_indices: np.ndarray, seat_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if game_indices.shape != seat_indices.shape:
            raise ValueError("game_index and seat_index shapes differ")
        decisions = [
            self.decision(int(game_index), int(seat_index))
            for game_index, seat_index in zip(game_indices, seat_indices, strict=True)
        ]
        selected = np.fromiter((keep for keep, _category in decisions), dtype=np.bool_, count=len(decisions))
        categories = np.asarray([category for _keep, category in decisions], dtype="U32")
        return selected, categories


def _load_ratings_sidecar(path: Path) -> dict[str, Any]:
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"ratings sidecar not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"ratings sidecar is not valid JSON: {path}") from exc
    if not isinstance(sidecar, dict):
        raise ValueError("ratings sidecar root must be an object")
    return sidecar


def _inspect_filtered_dgames(
    corpus: SourceCorpus, rating_filter: SeatRatingFilter
) -> tuple[SourceCorpus, Counter[str]]:
    """Find the human rows/games that survive the seat-level rating filter."""

    categories: Counter[str] = Counter()
    kept_games: set[int] = set()
    kept_tuples = 0
    observed_tuples = 0
    for shard_number, entry in enumerate(corpus.manifest["shards"], start=1):
        with np.load(corpus.root / entry["path"], allow_pickle=False) as shard:
            selected, row_categories = rating_filter.select(shard["game_index"], shard["seat_index"])
            categories.update(row_categories.tolist())
            kept_games.update(int(index) for index in np.unique(shard["game_index"][selected]))
            kept_tuples += int(selected.sum())
            observed_tuples += int(shard["action"].shape[0])
        print(
            f"filter inspected dgames shard={shard_number}/{len(corpus.manifest['shards'])} "
            f"kept_tuples={kept_tuples} kept_games={len(kept_games)}",
            flush=True,
        )

    if observed_tuples != corpus.tuple_count:
        raise AssertionError("dgames tuple count changed while filtering")
    if kept_tuples != sum(categories[category] for category in (CATEGORY_KEPT_BELOW_40, CATEGORY_KEPT_40_TO_50, CATEGORY_KEPT_50_PLUS)):
        raise AssertionError("rating category counts do not match kept tuple count")
    return (
        SourceCorpus(
            name=corpus.name,
            root=corpus.root,
            manifest=corpus.manifest,
            games=corpus.games,
            active_game_indices=frozenset(kept_games),
            tuple_count=kept_tuples,
        ),
        categories,
    )


def _write_filtered_split(
    output_root: Path,
    split: str,
    corpora: list[SourceCorpus],
    validation_keys: set[tuple[str, int]],
    rating_filter: SeatRatingFilter,
    shard_size: int,
    seed: int,
    validation_fraction: float,
    min_level: float,
    ratings_path: Path,
) -> dict[str, Any]:
    """Route one game-disjoint split, dropping ineligible dgames rows."""

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
                if corpus.name == SOURCE_DGAMES and bool(selected.any()):
                    rating_selected, _categories = rating_filter.select(
                        source_indices[selected], shard["seat_index"][selected]
                    )
                    selected_indices = np.flatnonzero(selected)
                    selected.fill(False)
                    selected[selected_indices[rating_selected]] = True
                if bool(selected.any()):
                    arrays = {name: np.asarray(shard[name][selected]) for name in REQUIRED_ARRAYS}
                    arrays["game_index"] = np.fromiter(
                        (source_remap[int(index)] for index in source_indices[selected]), dtype=np.int32
                    )
                    writer.append(arrays)
                    tuple_counts[corpus.name] += int(arrays["action"].shape[0])
            print(
                f"filter routing split={split} source={corpus.name} "
                f"shard={shard_number}/{len(corpus.manifest['shards'])} tuples={tuple_counts[corpus.name]}",
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
        "rating_filter": {
            "min_level": float(min_level),
            "ratings_sidecar": str(ratings_path),
            "dgames_policy": "keep only rows whose acting seat has a known level at or above min_level",
            "seeded_policy": "included unfiltered",
        },
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-level", type=float, default=40.0)
    parser.add_argument("--out", "--output-dir", dest="out", type=Path, default=Path("exports/tuples_all_40plus"))
    parser.add_argument(
        "--ratings", type=Path, default=Path("data/dominion_games/ratings/game_ratings.json")
    )
    parser.add_argument("--seeded-dir", type=Path, default=Path("exports/tuples"))
    parser.add_argument("--dgames-dir", type=Path, default=Path("exports/tuples_dgames"))
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--shard-size", type=int, default=10_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.min_level):
        raise SystemExit("--min-level must be finite")
    if not 0.0 < args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be between zero and one")
    if args.shard_size <= 0:
        raise SystemExit("--shard-size must be positive")

    seeded = inspect_source(SOURCE_SEEDED, args.seeded_dir)
    dgames = inspect_source(SOURCE_DGAMES, args.dgames_dir)
    if int(seeded.manifest["obs_width"]) != int(dgames.manifest["obs_width"]):
        raise SystemExit("source corpora disagree on obs_width")
    if int(seeded.manifest["action_width"]) != int(dgames.manifest["action_width"]):
        raise SystemExit("source corpora disagree on action_width")

    rating_filter = SeatRatingFilter(dgames.manifest, _load_ratings_sidecar(args.ratings), args.min_level)
    filtered_dgames, categories = _inspect_filtered_dgames(dgames, rating_filter)
    corpora = [seeded, filtered_dgames]

    validation_keys = _split_game_keys(corpora, args.validation_fraction, args.seed)
    print(
        f"filter split selected validation_games={len(validation_keys)} total_games="
        f"{sum(len(corpus.active_game_indices) for corpus in corpora)} seed={args.seed}",
        flush=True,
    )

    marker = args.out / OUTPUT_MARKER
    if args.out.exists():
        if not marker.is_file():
            raise SystemExit(
                f"refusing to replace unmanaged output directory {args.out}; choose a fresh --out"
            )
        print(f"filter removing previous output={args.out}", flush=True)
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    marker.write_text("c23_filter_corpus\n", encoding="utf-8")

    train = _write_filtered_split(
        args.out,
        "train",
        corpora,
        validation_keys,
        rating_filter,
        args.shard_size,
        args.seed,
        args.validation_fraction,
        args.min_level,
        args.ratings,
    )
    val = _write_filtered_split(
        args.out,
        "val",
        corpora,
        validation_keys,
        rating_filter,
        args.shard_size,
        args.seed,
        args.validation_fraction,
        args.min_level,
        args.ratings,
    )

    # This is c21's verification function: it verifies game disjointness,
    # manifest/shard consistency, and loads both splits through load_human_tuples.
    _verify_outputs(args.out, train, val)

    level_label = f"{args.min_level:g}"
    human_kept = filtered_dgames.tuple_count
    human_dropped = categories[CATEGORY_DROPPED_BELOW_MIN] + categories[CATEGORY_DROPPED_UNRATED]
    print(
        "filter summary games "
        f"kept={len(filtered_dgames.active_game_indices)} "
        f"dropped={len(dgames.active_game_indices) - len(filtered_dgames.active_game_indices)} "
        f"seeded_unfiltered={len(seeded.active_game_indices)}",
        flush=True,
    )
    print(
        "filter summary tuples "
        f"kept-40-to-50={categories[CATEGORY_KEPT_40_TO_50]} "
        f"kept-50-plus={categories[CATEGORY_KEPT_50_PLUS]} "
        f"dropped-below-{level_label}={categories[CATEGORY_DROPPED_BELOW_MIN]} "
        f"dropped-unrated={categories[CATEGORY_DROPPED_UNRATED]} "
        f"seeded-unfiltered={seeded.tuple_count} "
        f"human-kept={human_kept} human-dropped={human_dropped}",
        flush=True,
    )
    if categories[CATEGORY_KEPT_BELOW_40]:
        print(
            f"filter summary tuples kept-below-40={categories[CATEGORY_KEPT_BELOW_40]} "
            "(enabled by --min-level below 40)",
            flush=True,
        )
    print(
        "filter complete "
        f"train tuples={train['totals']['tuples_exported']} games={len(train['games'])} "
        f"val tuples={val['totals']['tuples_exported']} games={len(val['games'])} "
        f"total tuples={train['totals']['tuples_exported'] + val['totals']['tuples_exported']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
