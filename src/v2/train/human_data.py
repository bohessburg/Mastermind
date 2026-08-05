"""Loading and deterministic minibatching for replay-verified human tuples.

The tuple exporter intentionally stores hard actions and raw terminal margins.
Keeping the raw margin as the source of truth here means an imitation run can
use the same value-target convention as the self-play run that consumes it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MARGIN_BLEND_ALPHA = 0.6
DEFAULT_MARGIN_SCALE = 20.0
VALUE_SCHEMES = frozenset({"margin_blend", "margin", "plain_margin", "outcome", "winloss"})
DEFAULT_RATINGS_SIDECAR = Path("data/dominion_games/ratings/game_ratings.json")

# c22 signed policy-teacher curve. The joined sidecar's native Glicko
# deviation tops out at 0.499 (p95 0.336), so normalize against its natural
# 0.5 ceiling rather than the UI-converted deviation (roughly 7.5x larger).
POLICY_WEIGHT_EPSILON = 0.02
POLICY_WEIGHT_RAMP_START = 40.0
POLICY_WEIGHT_RAMP_END = 50.0
POLICY_WEIGHT_EXPERT = 3.0
RATING_DEVIATION_THRESHOLD = 0.5

RATING_BAND_UNRATED = "unrated_or_missing"
RATING_BAND_LOW = "level_below_40"
RATING_BAND_RAMP = "level_40_to_50"
RATING_BAND_EXPERT = "level_50_plus"


@dataclass(frozen=True)
class HumanBatch:
    """A hard-label imitation batch.

    ``action`` deliberately remains an integer action index rather than a
    one-hot policy.  Human records have one demonstrated action, and retaining
    that representation avoids allocating a dense [batch, 357] target merely
    to compute cross entropy.
    """

    obs: np.ndarray
    action: np.ndarray
    legal: np.ndarray
    value: np.ndarray
    # Optional only for compatibility with older direct HumanBatch fixtures;
    # manifest-backed batches always carry the raw terminal margin.
    margin: np.ndarray | None = None
    # None retains the original unweighted CE reduction exactly. Loader-backed
    # c22 skill-weighted batches carry one positive scalar per demonstration.
    policy_weight: np.ndarray | None = None

    def __iter__(self):
        """Allow ``obs, action, legal, value = next(iterator)`` callers."""
        yield self.obs
        yield self.action
        yield self.legal
        yield self.value


@dataclass(frozen=True)
class HumanTupleDataset:
    """In-memory human tuple arrays after manifest-driven filtering."""

    obs: np.ndarray
    action: np.ndarray
    legal: np.ndarray
    value: np.ndarray
    margin: np.ndarray
    winner: np.ndarray
    seat_index: np.ndarray
    game_index: np.ndarray
    ply_index: np.ndarray
    turn_number: np.ndarray
    policy_weight: np.ndarray
    rating_band: np.ndarray
    skill_weighting: bool
    manifest: dict[str, Any]

    def __len__(self) -> int:
        return int(self.action.shape[0])

    @property
    def obs_width(self) -> int:
        return int(self.obs.shape[1])

    @property
    def action_width(self) -> int:
        return int(self.legal.shape[1])

    def minibatches(self, batch_size: int, seed: int) -> "HumanBatchIterator":
        """Return a seeded iterator that reshuffles and cycles forever."""

        return HumanBatchIterator(self, batch_size=batch_size, seed=seed)

    def iter_minibatches(self, batch_size: int, seed: int) -> "HumanBatchIterator":
        """Alias retained for callers that prefer an explicit iterator verb."""

        return self.minibatches(batch_size=batch_size, seed=seed)


class HumanBatchIterator(Iterator[HumanBatch]):
    """A deterministic, non-exhausting shuffled iterator over human tuples."""

    def __init__(self, dataset: HumanTupleDataset, *, batch_size: int, seed: int):
        if len(dataset) <= 0:
            raise ValueError("cannot iterate an empty human tuple dataset")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(int(seed))
        self._order = self.rng.permutation(len(dataset))
        self._cursor = 0

    def __iter__(self) -> "HumanBatchIterator":
        return self

    def __next__(self) -> HumanBatch:
        # A batch may cross an epoch boundary (and may be larger than the
        # corpus), so collect one or more contiguous permutation slices.
        pieces: list[np.ndarray] = []
        remaining = self.batch_size
        while remaining:
            available = len(self._order) - self._cursor
            take = min(remaining, available)
            pieces.append(self._order[self._cursor : self._cursor + take])
            self._cursor += take
            remaining -= take
            if self._cursor == len(self._order):
                self._order = self.rng.permutation(len(self.dataset))
                self._cursor = 0
        indices = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        return HumanBatch(
            obs=np.ascontiguousarray(self.dataset.obs[indices]),
            action=np.ascontiguousarray(self.dataset.action[indices]),
            legal=np.ascontiguousarray(self.dataset.legal[indices]),
            value=np.ascontiguousarray(self.dataset.value[indices]),
            margin=np.ascontiguousarray(self.dataset.margin[indices]),
            policy_weight=(
                np.ascontiguousarray(self.dataset.policy_weight[indices])
                if self.dataset.skill_weighting
                else None
            ),
        )


def recompute_value_targets(
    margins: np.ndarray | Iterable[int],
    *,
    scheme: str = "margin_blend",
    alpha: float = DEFAULT_MARGIN_BLEND_ALPHA,
    scale: float = DEFAULT_MARGIN_SCALE,
) -> np.ndarray:
    """Return value targets derived from signed terminal margins.

    ``margin`` follows the native self-play target: a non-tie is
    ``sign(margin) * (0.5 + 0.5 * clipped_abs_margin / scale)``.  The
    ``margin_blend`` variant interpolates that result with win/loss by
    ``alpha``.  This exactly matches ``selfplay_margin_blend_value`` in the
    native runner, including its explicit zero target for ties.
    """

    normalized_scheme = str(scheme).lower()
    if normalized_scheme == "plain_margin":
        normalized_scheme = "margin"
    if normalized_scheme not in VALUE_SCHEMES:
        allowed = ", ".join(sorted(VALUE_SCHEMES))
        raise ValueError(f"unknown human value target {scheme!r}; expected one of {allowed}")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)):
        raise ValueError("margin scale must be a finite positive number")
    normalized_scale = float(scale)
    if normalized_scale <= 0.0:
        raise ValueError("margin scale must be a finite positive number")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)):
        raise ValueError("margin blend alpha must be a finite number between zero and one")
    normalized_alpha = float(alpha)
    if not 0.0 <= normalized_alpha <= 1.0:
        raise ValueError("margin blend alpha must be a finite number between zero and one")

    margin = np.asarray(margins, dtype=np.float32)
    sign = np.sign(margin)
    if normalized_scheme in {"outcome", "winloss"}:
        return sign.astype(np.float32, copy=False)

    graded = np.minimum(np.abs(margin), normalized_scale) / normalized_scale
    margin_value = sign * (0.5 + 0.5 * graded)
    # np.sign already maps a zero margin to exactly zero.  Keep the explicit
    # result rather than relying on any stored winner field: the raw margin is
    # what lets callers recompute every supported convention consistently.
    if normalized_scheme == "margin":
        return margin_value.astype(np.float32, copy=False)
    blended = sign * (normalized_alpha + (1.0 - normalized_alpha) * (0.5 + 0.5 * graded))
    return blended.astype(np.float32, copy=False)


def load_human_tuples(
    tuple_dir: str | Path = Path("exports/tuples"),
    *,
    value_scheme: str = "margin_blend",
    margin_blend_alpha: float = DEFAULT_MARGIN_BLEND_ALPHA,
    margin_scale: float = DEFAULT_MARGIN_SCALE,
    opponent_kinds: Iterable[str] | str | None = None,
    seat_indices: Iterable[int] | int | None = None,
    opponent_kind: str | None = None,
    seat_index: int | None = None,
    skill_weighting: bool = False,
    ratings_sidecar: str | Path = DEFAULT_RATINGS_SIDECAR,
) -> HumanTupleDataset:
    """Load manifest-declared tuple shards and optionally filter their rows.

    ``opponent_kinds`` matches the acting row's *other* seats.  Both exporter
    spellings (``"bot:bigmoney"``) and short spellings (``"bigmoney"``) are
    accepted.  ``seat_indices`` filters by the row's acting seat index.

    When ``skill_weighting`` is enabled, ratings are joined at load time using
    ``games[index].id`` / ``games[index].player_ids`` and each row's
    ``game_index`` / ``seat_index``. This keeps ratings replaceable without
    reconverting tuple shards. The value target remains unweighted.
    """

    if opponent_kind is not None:
        if opponent_kinds is not None:
            raise ValueError("pass only one of opponent_kind and opponent_kinds")
        opponent_kinds = opponent_kind
    if seat_index is not None:
        if seat_indices is not None:
            raise ValueError("pass only one of seat_index and seat_indices")
        seat_indices = seat_index
    if not isinstance(skill_weighting, bool):
        raise ValueError("skill_weighting must be a boolean")

    root = Path(tuple_dir)
    manifest_path = root / "tuple_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"human tuple manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"human tuple manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("human tuple manifest root must be an object")

    obs_width = _positive_manifest_int(manifest, "obs_width")
    action_width = _positive_manifest_int(manifest, "action_width")
    shard_entries = manifest.get("shards")
    if not isinstance(shard_entries, list) or not shard_entries:
        raise ValueError("human tuple manifest must contain a non-empty shards list")

    arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "obs",
            "action",
            "legal",
            "margin",
            "winner",
            "seat_index",
            "game_index",
            "ply_index",
            "turn_number",
        )
    }
    for entry in shard_entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("human tuple manifest shards must contain path objects")
        shard_path = root / entry["path"]
        _append_shard_arrays(arrays, shard_path, obs_width=obs_width, action_width=action_width)

    joined = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
    row_count = int(joined["action"].shape[0])
    expected_rows = manifest.get("totals", {}).get("tuples_exported") if isinstance(manifest.get("totals"), dict) else None
    if expected_rows is not None and int(expected_rows) != row_count:
        raise ValueError(
            f"human tuple manifest reports {expected_rows} rows but shards contain {row_count}"
        )

    selected = np.ones((row_count,), dtype=np.bool_)
    requested_seats = _normalize_seat_indices(seat_indices)
    if requested_seats is not None:
        selected &= np.isin(joined["seat_index"], np.asarray(sorted(requested_seats), dtype=np.int16))
    requested_opponents = _normalize_opponent_kinds(opponent_kinds)
    if requested_opponents is not None:
        game_opponents = _manifest_game_opponents(manifest)
        matches = np.zeros((row_count,), dtype=np.bool_)
        for row, (game_index, seat_index) in enumerate(
            zip(joined["game_index"], joined["seat_index"], strict=True)
        ):
            opponent_labels = game_opponents.get(int(game_index))
            if opponent_labels is None:
                raise ValueError(f"human tuple row references unknown game_index {int(game_index)}")
            if int(seat_index) < 0 or int(seat_index) >= len(opponent_labels):
                raise ValueError(f"human tuple row has invalid seat_index {int(seat_index)}")
            matches[row] = any(
                _kind_matches(label, requested_opponents)
                for actor_seat, label in enumerate(opponent_labels)
                if actor_seat != int(seat_index)
            )
        selected &= matches

    if not bool(selected.any()):
        raise ValueError("human tuple filters selected no rows")
    filtered = {name: np.ascontiguousarray(values[selected]) for name, values in joined.items()}
    actions = filtered["action"]
    legal = filtered["legal"]
    if np.any(actions < 0) or np.any(actions >= action_width):
        raise ValueError("human tuple action index is outside the action width")
    action_rows = np.arange(actions.shape[0], dtype=np.intp)
    if not np.all(legal[action_rows, actions.astype(np.intp, copy=False)]):
        raise ValueError("human tuple contains a demonstrated action that is not legal")

    value = recompute_value_targets(
        filtered["margin"],
        scheme=value_scheme,
        alpha=margin_blend_alpha,
        scale=margin_scale,
    )
    policy_weight = np.ones((actions.shape[0],), dtype=np.float32)
    rating_band = np.full((actions.shape[0],), "unweighted", dtype="U18")
    if skill_weighting:
        policy_weight, rating_band = _load_policy_weights(
            manifest,
            filtered["game_index"],
            filtered["seat_index"],
            ratings_sidecar,
        )
    return HumanTupleDataset(
        obs=filtered["obs"].astype(np.float32, copy=False),
        action=filtered["action"].astype(np.int64, copy=False),
        legal=filtered["legal"].astype(np.bool_, copy=False),
        value=np.ascontiguousarray(value, dtype=np.float32),
        margin=filtered["margin"].astype(np.int16, copy=False),
        winner=filtered["winner"].astype(np.int8, copy=False),
        seat_index=filtered["seat_index"].astype(np.int16, copy=False),
        game_index=filtered["game_index"].astype(np.int32, copy=False),
        ply_index=filtered["ply_index"].astype(np.int32, copy=False),
        turn_number=filtered["turn_number"].astype(np.int32, copy=False),
        policy_weight=policy_weight,
        rating_band=rating_band,
        skill_weighting=skill_weighting,
        manifest=manifest,
    )


def rating_band_for_level(level: float | None) -> str:
    """Classify a displayed leaderboard level for c22 audit accounting."""

    if level is None or not math.isfinite(level):
        return RATING_BAND_UNRATED
    if level < POLICY_WEIGHT_RAMP_START:
        return RATING_BAND_LOW
    if level < POLICY_WEIGHT_RAMP_END:
        return RATING_BAND_RAMP
    return RATING_BAND_EXPERT


def policy_weight_for_rating(level: float | None, deviation: float | None = None) -> float:
    """Return the signed c22 policy-only teaching weight for one player."""

    band = rating_band_for_level(level)
    if band == RATING_BAND_UNRATED:
        # Missing ratings are deliberately the low/value-only band, not an
        # invitation to invent a confidence discount below the signed epsilon.
        return POLICY_WEIGHT_EPSILON
    assert level is not None
    if band == RATING_BAND_LOW:
        weight = POLICY_WEIGHT_EPSILON
    elif band == RATING_BAND_RAMP:
        progress = (level - POLICY_WEIGHT_RAMP_START) / (
            POLICY_WEIGHT_RAMP_END - POLICY_WEIGHT_RAMP_START
        )
        weight = POLICY_WEIGHT_EPSILON + progress * (1.0 - POLICY_WEIGHT_EPSILON)
    else:
        weight = POLICY_WEIGHT_EXPERT

    if deviation is None or not math.isfinite(deviation):
        return float(weight)
    confidence = max(0.5, 1.0 - max(0.0, deviation) / RATING_DEVIATION_THRESHOLD)
    return float(weight * confidence)


def _load_policy_weights(
    manifest: dict[str, Any],
    game_indices: np.ndarray,
    seat_indices: np.ndarray,
    ratings_sidecar: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    ratings_path = Path(ratings_sidecar)
    try:
        sidecar = json.loads(ratings_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"ratings sidecar not found: {ratings_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"ratings sidecar is not valid JSON: {ratings_path}") from exc
    if not isinstance(sidecar, dict):
        raise ValueError("ratings sidecar root must be an object")

    games = _manifest_rating_games(manifest)
    weights = np.empty((game_indices.shape[0],), dtype=np.float32)
    bands = np.empty((game_indices.shape[0],), dtype="U18")
    assignments: dict[tuple[int, int], tuple[float, str]] = {}
    for row, (game_index, seat_index) in enumerate(zip(game_indices, seat_indices, strict=True)):
        key = (int(game_index), int(seat_index))
        assignment = assignments.get(key)
        if assignment is None:
            game = games.get(key[0])
            if game is None:
                raise ValueError(f"human tuple row references unknown game_index {key[0]}")
            game_id, player_ids = game
            if player_ids is None:
                level = deviation = None
            elif not 0 <= key[1] < len(player_ids):
                raise ValueError(f"human tuple row has invalid seat_index {key[1]}")
            else:
                game_ratings = sidecar.get(game_id)
                entry = game_ratings.get(str(player_ids[key[1]])) if isinstance(game_ratings, dict) else None
                level, deviation = _rating_entry_level_and_deviation(entry)
            assignment = (policy_weight_for_rating(level, deviation), rating_band_for_level(level))
            assignments[key] = assignment
        weights[row], bands[row] = assignment
    return np.ascontiguousarray(weights), np.ascontiguousarray(bands)


def _manifest_rating_games(manifest: dict[str, Any]) -> dict[int, tuple[str, tuple[int, ...] | None]]:
    games = manifest.get("games")
    if not isinstance(games, list):
        raise ValueError("human tuple manifest games must be a list for skill weighting")
    result: dict[int, tuple[str, tuple[int, ...] | None]] = {}
    for raw_game in games:
        if not isinstance(raw_game, dict):
            raise ValueError("human tuple manifest games must contain objects")
        index = raw_game.get("index")
        game_id = raw_game.get("id")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("human tuple manifest game index must be a non-negative integer")
        if not isinstance(game_id, (int, str)) or isinstance(game_id, bool) or not str(game_id):
            raise ValueError("human tuple manifest game id must be a non-empty string or integer")
        raw_player_ids = raw_game.get("player_ids")
        player_ids = (
            tuple(raw_player_ids)
            if isinstance(raw_player_ids, list)
            and all(isinstance(player_id, int) and not isinstance(player_id, bool) for player_id in raw_player_ids)
            else None
        )
        if index in result:
            raise ValueError(f"human tuple manifest repeats game index {index}")
        result[index] = (str(game_id), player_ids)
    return result


def _rating_entry_level_and_deviation(entry: object) -> tuple[float | None, float | None]:
    if not isinstance(entry, dict):
        return None, None
    raw_level = entry.get("level", entry.get("rating"))
    raw_deviation = entry.get("deviation")
    level = _finite_optional_number(raw_level)
    deviation = _finite_optional_number(raw_deviation)
    return level, deviation


def _finite_optional_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    return normalized if math.isfinite(normalized) else None


def _positive_manifest_int(manifest: dict[str, Any], name: str) -> int:
    value = manifest.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"human tuple manifest {name} must be a positive integer")
    return int(value)


def _append_shard_arrays(
    destination: dict[str, list[np.ndarray]],
    shard_path: Path,
    *,
    obs_width: int,
    action_width: int,
) -> None:
    try:
        with np.load(shard_path, allow_pickle=False) as shard:
            required = tuple(destination)
            missing = [name for name in required if name not in shard]
            if missing:
                raise ValueError(f"human tuple shard {shard_path} is missing {', '.join(missing)}")
            rows = int(shard["action"].shape[0])
            if shard["action"].shape != (rows,):
                raise ValueError(f"human tuple shard {shard_path} has an invalid action shape")
            if shard["obs"].shape != (rows, obs_width):
                raise ValueError(f"human tuple shard {shard_path} has an invalid obs shape")
            if shard["legal"].shape != (rows, action_width):
                raise ValueError(f"human tuple shard {shard_path} has an invalid legal shape")
            if "value" not in shard or shard["value"].shape != (rows,):
                raise ValueError(f"human tuple shard {shard_path} has an invalid value shape")
            for name in ("margin", "winner", "seat_index", "game_index", "ply_index", "turn_number"):
                if shard[name].shape != (rows,):
                    raise ValueError(f"human tuple shard {shard_path} has an invalid {name} shape")
            for name in required:
                destination[name].append(np.asarray(shard[name]).copy())
    except OSError as exc:
        raise FileNotFoundError(f"human tuple shard not found: {shard_path}") from exc


def _normalize_seat_indices(values: Iterable[int] | int | None) -> set[int] | None:
    if values is None:
        return None
    raw_values = (values,) if isinstance(values, int) and not isinstance(values, bool) else values
    if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Iterable):
        raise ValueError("seat_indices must be an integer or iterable of integers")
    result: set[int] = set()
    for value in raw_values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("seat_indices must contain non-negative integers")
        result.add(int(value))
    return result


def _normalize_opponent_kinds(values: Iterable[str] | str | None) -> set[str] | None:
    if values is None:
        return None
    raw_values = (values,) if isinstance(values, str) else values
    if not isinstance(raw_values, Iterable):
        raise ValueError("opponent_kinds must be a string or iterable of strings")
    result: set[str] = set()
    for value in raw_values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("opponent_kinds must contain non-empty strings")
        result.add(value.strip())
    return result


def _manifest_game_opponents(manifest: dict[str, Any]) -> dict[int, tuple[str, ...]]:
    games = manifest.get("games")
    if not isinstance(games, list):
        raise ValueError("human tuple manifest games must be a list")
    result: dict[int, tuple[str, ...]] = {}
    for raw_game in games:
        if not isinstance(raw_game, dict):
            raise ValueError("human tuple manifest games must contain objects")
        index = raw_game.get("index")
        seat_kinds = raw_game.get("seat_kinds")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("human tuple manifest game index must be a non-negative integer")
        if not isinstance(seat_kinds, list) or not seat_kinds or not all(isinstance(kind, str) for kind in seat_kinds):
            raise ValueError("human tuple manifest game seat_kinds must be a non-empty string list")
        if index in result:
            raise ValueError(f"human tuple manifest repeats game index {index}")
        result[index] = tuple(seat_kinds)
    return result


def _kind_matches(label: str, requested: set[str]) -> bool:
    normalized = label.removeprefix("bot:")
    return label in requested or normalized in requested
