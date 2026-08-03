"""Join append-only dominion.games ratings to captured-game manifests.

Ratings are intentionally kept outside tuple shards because they drift over
time.  This script writes a replaceable sidecar whose values are selected from
the append-only observations store nearest to each game's capture timestamp.

Usage::

    ./.venv/bin/python scripts/dgames_ratings_join.py
    ./.venv/bin/python scripts/dgames_ratings_join.py \
        --raw-root data/dominion_games/raw \
        --observations data/dominion_games/ratings/observations.jsonl
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterable, Mapping


RATING_TYPE_NAMES = (
    "RATINGS_2P",
    "RATINGS_3P",
    "RATINGS_2P_BLITZ",
    "RATINGS_3P_BLITZ",
)
DEFAULT_RAW_ROOT = Path("data/dominion_games/raw")
DEFAULT_RATINGS_ROOT = Path("data/dominion_games/ratings")


@dataclass(frozen=True)
class RatingObservation:
    """One validated append-only leaderboard observation."""

    player_id: int
    observed_at_utc: str
    rank: int
    rating: float
    trend: float
    rating_type: str

    @property
    def observed_at(self) -> datetime:
        return parse_utc_timestamp(self.observed_at_utc, field="observed_at_utc")


@dataclass(frozen=True)
class CoverageReport:
    """Honest two-player resolution accounting for the game-ratings sidecar."""

    games_total: int
    both: int
    one: int
    neither: int
    manifests_without_player_ids: int
    games_without_rating_type: int
    games_with_non_two_player_metadata: int

    def as_json(self) -> dict[str, object]:
        denominator = self.games_total

        def bucket(games: int) -> dict[str, float | int]:
            return {
                "games": games,
                "fraction": 0.0 if denominator == 0 else games / denominator,
            }

        return {
            "games_total": denominator,
            "both_players_rated": bucket(self.both),
            "one_player_rated": bucket(self.one),
            "neither_player_rated": bucket(self.neither),
            "manifests_without_player_ids": self.manifests_without_player_ids,
            "games_without_rating_type": self.games_without_rating_type,
            "games_with_non_two_player_metadata": self.games_with_non_two_player_metadata,
        }


def parse_utc_timestamp(value: object, *, field: str) -> datetime:
    """Parse an ISO-8601 timestamp as UTC without accepting a naive clock."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty UTC timestamp string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"invalid {field} timestamp {value!r}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _required_int(document: Mapping[str, object], field: str, *, context: str) -> int:
    value = document.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} has invalid {field}")
    return value


def _required_number(document: Mapping[str, object], field: str, *, context: str) -> float:
    value = document.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{context} has invalid {field}")
    return float(value)


def parse_observation_document(document: Mapping[str, object], *, context: str) -> RatingObservation:
    """Validate the fields that make a rating record usable for a time join."""

    rating_type = document.get("rating_type")
    if not isinstance(rating_type, str) or rating_type not in RATING_TYPE_NAMES:
        raise ValueError(f"{context} has unsupported rating_type {rating_type!r}")
    observed_at_utc = document.get("observed_at_utc")
    parse_utc_timestamp(observed_at_utc, field=f"{context}.observed_at_utc")
    return RatingObservation(
        player_id=_required_int(document, "player_id", context=context),
        observed_at_utc=str(observed_at_utc),
        rank=_required_int(document, "rank", context=context),
        rating=_required_number(document, "rating", context=context),
        trend=_required_number(document, "trend", context=context),
        rating_type=rating_type,
    )


def load_observations(path: Path | str) -> tuple[RatingObservation, ...]:
    """Load all JSONL records; malformed rows fail rather than disappearing."""

    source = Path(path)
    if not source.exists():
        return ()
    observations: list[RatingObservation] = []
    with source.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from error
            if not isinstance(document, dict):
                raise ValueError(f"{source}:{line_number}: observation is not an object")
            observations.append(
                parse_observation_document(document, context=f"{source}:{line_number}")
            )
    return tuple(observations)


def nearest_observation(
    capture_time: datetime,
    candidates: Iterable[RatingObservation],
) -> RatingObservation | None:
    """Return the closest timestamp, preferring the earlier observation on a tie."""

    materialized = tuple(candidates)
    if not materialized:
        return None
    return min(
        materialized,
        key=lambda observation: (
            abs((observation.observed_at - capture_time).total_seconds()),
            observation.observed_at,
        ),
    )


def _index_observations(
    observations: Iterable[RatingObservation],
) -> dict[tuple[int, str], tuple[RatingObservation, ...]]:
    indexed: dict[tuple[int, str], list[RatingObservation]] = {}
    for observation in observations:
        indexed.setdefault((observation.player_id, observation.rating_type), []).append(
            observation
        )
    return {
        key: tuple(sorted(values, key=lambda observation: observation.observed_at))
        for key, values in indexed.items()
    }


def _manifest_players(manifest: Mapping[str, object]) -> tuple[int, ...]:
    """Read player ids from current and forward-compatible manifest locations."""

    candidates: object | None = None
    outcome = manifest.get("outcome")
    if isinstance(outcome, Mapping):
        game_result = outcome.get("game_result")
        if isinstance(game_result, Mapping):
            candidates = game_result.get("players")
    if candidates is None:
        candidates = manifest.get("players")
    if candidates is None:
        candidates = manifest.get("player_ids")
    if not isinstance(candidates, list):
        return ()

    player_ids: list[int] = []
    for player in candidates:
        if isinstance(player, Mapping):
            value = player.get("player_id")
        else:
            value = player
        if not isinstance(value, int) or isinstance(value, bool):
            return ()
        if value not in player_ids:
            player_ids.append(value)
    return tuple(player_ids)


def _manifest_rating_type(manifest: Mapping[str, object]) -> str | None:
    """Map the collector's GameResult rating ordinal to the leaderboard key."""

    direct = manifest.get("rating_type")
    if isinstance(direct, str) and direct in RATING_TYPE_NAMES:
        return direct
    outcome = manifest.get("outcome")
    if not isinstance(outcome, Mapping):
        return None
    game_result = outcome.get("game_result")
    if not isinstance(game_result, Mapping):
        return None
    ordinal = game_result.get("rating_type_ordinal")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        return None
    return RATING_TYPE_NAMES[ordinal] if 0 <= ordinal < len(RATING_TYPE_NAMES) else None


def _sidecar_entry(
    observation: RatingObservation,
    *,
    capture_time: datetime,
) -> dict[str, object]:
    return {
        "rating": observation.rating,
        "rank": observation.rank,
        "trend": observation.trend,
        "rating_type": observation.rating_type,
        "observed_at_utc": observation.observed_at_utc,
        "staleness_seconds": abs((observation.observed_at - capture_time).total_seconds()),
    }


def join_manifests(
    manifests: Iterable[tuple[str, Mapping[str, object]]],
    observations: Iterable[RatingObservation],
) -> tuple[dict[str, dict[str, dict[str, object] | None]], CoverageReport]:
    """Join all supplied game manifests without assigning a synthetic rating.

    A missing type or player observation produces an explicit ``null`` sidecar
    value.  The join deliberately does not fall back across rating types:
    a blitz rating is not a standard 2-player rating.
    """

    indexed = _index_observations(observations)
    sidecar: dict[str, dict[str, dict[str, object] | None]] = {}
    total = both = one = neither = 0
    missing_player_ids = missing_rating_type = non_two_player = 0

    for game_id, manifest in manifests:
        total += 1
        capture_time = parse_utc_timestamp(
            manifest.get("capture_time_utc"),
            field=f"game {game_id}.capture_time_utc",
        )
        player_ids = _manifest_players(manifest)
        if not player_ids:
            missing_player_ids += 1
        if len(player_ids) != 2:
            non_two_player += 1
        rating_type = _manifest_rating_type(manifest)
        if rating_type is None:
            missing_rating_type += 1

        joined_players: dict[str, dict[str, object] | None] = {}
        resolved = 0
        for player_id in player_ids:
            observation = (
                None
                if rating_type is None
                else nearest_observation(capture_time, indexed.get((player_id, rating_type), ()))
            )
            if observation is None:
                joined_players[str(player_id)] = None
            else:
                joined_players[str(player_id)] = _sidecar_entry(
                    observation,
                    capture_time=capture_time,
                )
                resolved += 1
        sidecar[str(game_id)] = joined_players

        # Captures are normally exactly two-player.  Retain the requested
        # both/one/neither accounting even for malformed/nonstandard metadata
        # rather than silently dropping those games from its denominator.
        if resolved >= 2:
            both += 1
        elif resolved == 1:
            one += 1
        else:
            neither += 1

    return sidecar, CoverageReport(
        games_total=total,
        both=both,
        one=one,
        neither=neither,
        manifests_without_player_ids=missing_player_ids,
        games_without_rating_type=missing_rating_type,
        games_with_non_two_player_metadata=non_two_player,
    )


def load_manifests(raw_root: Path | str) -> tuple[tuple[str, dict[str, object]], ...]:
    """Load every collector manifest, including incomplete captures for honest coverage."""

    root = Path(raw_root)
    manifests: list[tuple[str, dict[str, object]]] = []
    for path in sorted(root.glob("*.manifest.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read manifest {path}: {error}") from error
        if not isinstance(document, dict):
            raise ValueError(f"manifest {path} is not an object")
        game_id = document.get("game_id", path.name.removesuffix(".manifest.json"))
        if isinstance(game_id, bool) or not isinstance(game_id, (int, str)):
            raise ValueError(f"manifest {path} has invalid game_id")
        manifests.append((str(game_id), document))
    return tuple(manifests)


def write_sidecar(
    path: Path | str,
    sidecar: Mapping[str, Mapping[str, Mapping[str, object] | None]],
) -> Path:
    """Atomically replace the derived sidecar; observations themselves remain append-only."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(sidecar, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--observations",
        type=Path,
        default=DEFAULT_RATINGS_ROOT / "observations.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RATINGS_ROOT / "game_ratings.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifests = load_manifests(args.raw_root)
    observations = load_observations(args.observations)
    sidecar, coverage = join_manifests(manifests, observations)
    destination = write_sidecar(args.output, sidecar)
    report = coverage.as_json()
    print(
        f"ratings join: {report['games_total']} game(s), {len(observations)} observation(s)",
        flush=True,
    )
    for key in ("both_players_rated", "one_player_rated", "neither_player_rated"):
        bucket = report[key]
        assert isinstance(bucket, dict)
        print(
            f"coverage {key}: {bucket['games']}/{report['games_total']} "
            f"({float(bucket['fraction']):.1%})",
            flush=True,
        )
    if report["manifests_without_player_ids"] or report["games_without_rating_type"]:
        print(
            "coverage caveats: "
            f"missing-player-ids={report['manifests_without_player_ids']}; "
            f"missing-rating-type={report['games_without_rating_type']}",
            flush=True,
        )
    print(f"sidecar: {destination.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
