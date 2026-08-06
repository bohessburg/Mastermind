"""Export replay-verified imitation-learning tuples from real web exports."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import dominion_v2_py as dz

from .corpus import load_or_create_manifest, manifest_entries
from .local import final_state_hash_matches, is_complete_local_export_data


OBS_VERSION = 3
OBS_WIDTH = 1788
ACTION_WIDTH = 357
DEFAULT_ALPHA = 0.6
DEFAULT_SCALE = 20.0


class ReplayVerificationError(ValueError):
    """A source export did not exactly replay under the active engine."""


@dataclass(frozen=True)
class PendingTuple:
    obs: np.ndarray
    legal: np.ndarray
    action: int
    seat_index: int
    ply_index: int
    turn_number: int


@dataclass(frozen=True)
class ReplayResult:
    game_id: str
    source_path: Path
    seat_kinds: tuple[str, ...]
    rows: tuple[PendingTuple, ...]
    per_seat_decisions: tuple[int, ...]
    scores: tuple[int, ...]
    winner: int | None
    truncated: bool
    actual_hash: str
    selected: bool


@dataclass(frozen=True)
class TupleExportResult:
    manifest_path: Path
    shard_paths: tuple[Path, ...]
    games_processed: int
    tuples_exported: int
    decision_counts_by_actor_kind: dict[str, int]
    decision_counts_by_opponent_kind: dict[str, int]
    failures: tuple[str, ...]


def margin_blend_value(margin: int, *, alpha: float = DEFAULT_ALPHA, scale: float = DEFAULT_SCALE) -> float:
    """Mirror the self-play MarginBlend target for a signed final margin."""
    if margin == 0:
        return 0.0
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    sign = 1.0 if margin > 0 else -1.0
    graded = min(abs(float(margin)), scale) / scale
    return sign * (alpha + (1.0 - alpha) * (0.5 + 0.5 * graded))


def replay_export(
    data: Mapping[str, Any],
    *,
    source_path: Path,
    game_id: str,
    include_bot_seats: bool = False,
    opponents: set[str] | None = None,
) -> ReplayResult:
    """Replay an export and capture selected pre-action obs-v3 positions.

    The rows intentionally wait in memory until the terminal score is known;
    final targets are then attached from the perspective of each acting seat.
    """
    if not is_complete_local_export_data(data):
        raise ReplayVerificationError(f"{source_path}: not a complete local export")
    seats_raw = data.get("seats")
    kingdom_raw = data.get("kingdom")
    actions_raw = data.get("actions")
    if not isinstance(seats_raw, list) or not isinstance(kingdom_raw, list) or not isinstance(actions_raw, list):
        raise ReplayVerificationError(f"{source_path}: malformed local export")
    seat_kinds = tuple(str(kind) for kind in seats_raw)
    normalized_opponents = opponents or None
    game_selected = (
        normalized_opponents is None
        or _matches_opponents(seat_kinds, normalized_opponents)
    )

    game = dz.new_game(
        dz.Setup(players=len(seat_kinds), kingdom=[int(value) for value in kingdom_raw]),
        int(data["seed"]),
    )
    rows: list[PendingTuple] = []
    per_seat_decisions = [0 for _ in seat_kinds]
    for ply_index, raw_action in enumerate(actions_raw):
        action = int(raw_action)
        decision = dict(game.current_decision())
        actor = int(decision["player"])
        mask = np.asarray(game.legal_mask(), dtype=np.bool_)
        if action < 0 or action >= mask.size or not bool(mask[action]):
            raise ReplayVerificationError(
                f"{source_path}: illegal action at ply {ply_index}: {action}"
            )
        if actor < 0 or actor >= len(seat_kinds):
            raise ReplayVerificationError(f"{source_path}: invalid actor {actor} at ply {ply_index}")
        per_seat_decisions[actor] += 1
        if game_selected and _include_seat(
            seat_kinds[actor],
            include_bot_seats=include_bot_seats,
            opponents=normalized_opponents,
        ):
            obs = np.asarray(game.encode(actor, OBS_VERSION), dtype=np.float32)
            if obs.shape != (OBS_WIDTH,):
                raise ReplayVerificationError(
                    f"{source_path}: obs-v{OBS_VERSION} width {obs.shape}, expected ({OBS_WIDTH},)"
                )
            if mask.shape != (ACTION_WIDTH,):
                raise ReplayVerificationError(
                    f"{source_path}: legal mask width {mask.shape}, expected ({ACTION_WIDTH},)"
                )
            rows.append(
                PendingTuple(
                    obs=obs.copy(),
                    legal=mask.copy(),
                    action=action,
                    seat_index=actor,
                    ply_index=ply_index,
                    turn_number=int(game.turn()) // len(seat_kinds) + 1,
                )
            )
        game.step(action)

    if not game.game_over():
        raise ReplayVerificationError(f"{source_path}: replay did not reach game over")
    actual_hash = _format_hash(game.state_hash())
    if (
        data.get("final_state_hash") is None
        and data.get(f"final_state_hash_gen{int(dz.ENCODER_GENERATION)}") is None
    ):
        raise ReplayVerificationError(f"{source_path}: missing final_state_hash")
    if not final_state_hash_matches(data, game.state_hash()):
        raise ReplayVerificationError(
            f"{source_path}: hash mismatch: replay {actual_hash} matches no recorded generation"
        )
    scores = tuple(int(game.score(seat)) for seat in range(len(seat_kinds)))
    truncated = bool(game.truncated())
    highest = max(scores)
    leaders = [seat for seat, score in enumerate(scores) if score == highest]
    winner = leaders[0] if len(leaders) == 1 else None
    return ReplayResult(
        game_id=game_id,
        source_path=source_path,
        seat_kinds=seat_kinds,
        rows=tuple(rows),
        per_seat_decisions=tuple(per_seat_decisions),
        scores=scores,
        winner=winner,
        truncated=truncated,
        actual_hash=actual_hash,
        selected=game_selected,
    )


def export_real_corpus(
    exports_root: Path | str = Path("exports"),
    *,
    output_dir: Path | str | None = None,
    include_bot_seats: bool = False,
    opponents: Iterable[str] | None = None,
    alpha: float = DEFAULT_ALPHA,
    scale: float = DEFAULT_SCALE,
    shard_size: int = 10_000,
) -> TupleExportResult:
    """Replay every manifest-classified real export and write tuple shards."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    # Validate once even if there are no selected games.
    margin_blend_value(1, alpha=alpha, scale=scale)
    root = Path(exports_root)
    destination = Path(output_dir) if output_dir is not None else root / "tuples"
    destination.mkdir(parents=True, exist_ok=True)
    selected_opponents = _parse_opponents(opponents)
    corpus_manifest = load_or_create_manifest(root)
    sources = [
        root / str(entry["path"])
        for entry in manifest_entries(corpus_manifest, classifications=("real",))
    ]

    accumulator = _ShardAccumulator(destination, shard_size)
    games: list[dict[str, Any]] = []
    failures: list[str] = []
    actor_counts: Counter[str] = Counter()
    opponent_counts: Counter[str] = Counter()
    for source in sources:
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
            replayed = replay_export(
                data,
                source_path=source,
                game_id=source.stem,
                include_bot_seats=include_bot_seats,
                opponents=selected_opponents,
            )
        except (OSError, json.JSONDecodeError, ReplayVerificationError, TypeError, ValueError) as error:
            failures.append(f"{source}: {error}")
            continue
        if not replayed.selected:
            # The --opponents selection excluded the entire game.
            continue
        game_index = len(games)
        per_seat_exported = [0 for _ in replayed.seat_kinds]
        for row in replayed.rows:
            margin = _seat_margin(replayed.scores, row.seat_index)
            value = (
                0.0
                if replayed.truncated or replayed.winner is None
                else margin_blend_value(margin, alpha=alpha, scale=scale)
            )
            accumulator.append(
                row,
                value=value,
                margin=margin,
                winner=-1 if replayed.winner is None else replayed.winner,
                game_index=game_index,
            )
            per_seat_exported[row.seat_index] += 1
            actor_kind = replayed.seat_kinds[row.seat_index]
            actor_counts[actor_kind] += 1
            for opponent_kind in _opponent_kinds(replayed.seat_kinds, row.seat_index):
                opponent_counts[opponent_kind] += 1
        games.append(
            {
                "index": game_index,
                "id": replayed.game_id,
                "source_path": _relative_source_path(root, source),
                "seat_kinds": list(replayed.seat_kinds),
                "decision_counts": list(replayed.per_seat_decisions),
                "exported_decision_counts": per_seat_exported,
                "outcome": {
                    "winner": replayed.winner,
                    "scores": list(replayed.scores),
                    "margins": [
                        _seat_margin(replayed.scores, seat)
                        for seat in range(len(replayed.scores))
                    ],
                    "truncated": replayed.truncated,
                    "final_state_hash": replayed.actual_hash,
                },
            }
        )

    shard_paths = accumulator.finish()
    manifest = {
        "schema_version": 1,
        "obs_version": OBS_VERSION,
        "obs_width": OBS_WIDTH,
        "action_width": ACTION_WIDTH,
        "value_target": {"name": "margin_blend", "alpha": alpha, "scale": scale},
        "filters": {
            "include_bot_seats": include_bot_seats,
            "opponents": sorted(selected_opponents) if selected_opponents else None,
        },
        "source_manifest": str(root / "corpus_manifest.json"),
        "games": games,
        "shards": [
            {"path": path.name, "tuples": _shard_tuple_count(path)} for path in shard_paths
        ],
        "totals": {
            "games_processed": len(games),
            "tuples_exported": accumulator.total_rows,
            "decision_counts_by_actor_kind": dict(sorted(actor_counts.items())),
            "decision_counts_by_opponent_kind": dict(sorted(opponent_counts.items())),
            "failed_replays": len(failures),
        },
        "failures": failures,
    }
    result_manifest = destination / "tuple_manifest.json"
    result_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TupleExportResult(
        manifest_path=result_manifest,
        shard_paths=tuple(shard_paths),
        games_processed=len(games),
        tuples_exported=accumulator.total_rows,
        decision_counts_by_actor_kind=dict(sorted(actor_counts.items())),
        decision_counts_by_opponent_kind=dict(sorted(opponent_counts.items())),
        failures=tuple(failures),
    )


class _ShardAccumulator:
    def __init__(self, output_dir: Path, shard_size: int) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.rows: list[tuple[PendingTuple, float, int, int, int]] = []
        self.paths: list[Path] = []
        self.total_rows = 0

    def append(
        self,
        row: PendingTuple,
        *,
        value: float,
        margin: int,
        winner: int,
        game_index: int,
    ) -> None:
        self.rows.append((row, value, margin, winner, game_index))
        self.total_rows += 1
        if len(self.rows) >= self.shard_size:
            self._flush()

    def finish(self) -> tuple[Path, ...]:
        if self.rows:
            self._flush()
        return tuple(self.paths)

    def _flush(self) -> None:
        index = len(self.paths)
        destination = self.output_dir / f"tuples-{index:05d}.npz"
        rows = self.rows
        np.savez_compressed(
            destination,
            obs=np.stack([row.obs for row, *_ in rows]).astype(np.float32, copy=False),
            action=np.asarray([row.action for row, *_ in rows], dtype=np.int32),
            legal=np.stack([row.legal for row, *_ in rows]).astype(np.bool_, copy=False),
            value=np.asarray([value for _, value, *_ in rows], dtype=np.float32),
            margin=np.asarray([margin for _, _, margin, _, _ in rows], dtype=np.int16),
            winner=np.asarray([winner for _, _, _, winner, _ in rows], dtype=np.int8),
            seat_index=np.asarray([row.seat_index for row, *_ in rows], dtype=np.int16),
            game_index=np.asarray([game_index for *_, game_index in rows], dtype=np.int32),
            ply_index=np.asarray([row.ply_index for row, *_ in rows], dtype=np.int32),
            turn_number=np.asarray([row.turn_number for row, *_ in rows], dtype=np.int32),
        )
        self.paths.append(destination)
        self.rows = []


def _include_seat(kind: str, *, include_bot_seats: bool, opponents: set[str] | None) -> bool:
    if kind == "human":
        return True
    return include_bot_seats and (opponents is None or _normalized_kind(kind) in opponents)


def _matches_opponents(seat_kinds: Sequence[str], opponents: set[str]) -> bool:
    return any(kind != "human" and _normalized_kind(kind) in opponents for kind in seat_kinds)


def _parse_opponents(opponents: Iterable[str] | None) -> set[str] | None:
    if opponents is None:
        return None
    values: set[str] = set()
    for value in opponents:
        values.update(piece.strip().removeprefix("bot:") for piece in str(value).split(",") if piece.strip())
    return values or None


def _normalized_kind(kind: str) -> str:
    if kind.startswith("bot:"):
        return kind.split(":", 2)[1]
    return kind.removeprefix("bot:")


def _opponent_kinds(seat_kinds: Sequence[str], actor: int) -> tuple[str, ...]:
    return tuple(kind for seat, kind in enumerate(seat_kinds) if seat != actor)


def _seat_margin(scores: Sequence[int], actor: int) -> int:
    """Return actor score minus its strongest opponent (exact for 2-player games)."""
    opponents = [score for seat, score in enumerate(scores) if seat != actor]
    return int(scores[actor]) - max(opponents) if opponents else 0


def _format_hash(value: int) -> str:
    return f"0x{int(value):016x}"


def _relative_source_path(root: Path, source: Path) -> str:
    try:
        return source.relative_to(root).as_posix()
    except ValueError:
        return str(source)


def _shard_tuple_count(path: Path) -> int:
    with np.load(path) as shard:
        return int(shard["action"].shape[0])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export replay-verified imitation tuples from real web exports.")
    parser.add_argument("--exports-root", type=Path, default=Path("exports"))
    parser.add_argument("--out", type=Path, help="output directory (default: exports/tuples)")
    parser.add_argument("--include-bot-seats", action="store_true", help="emit actions made by bot seats too")
    parser.add_argument(
        "--opponents",
        action="append",
        help="comma-separated bot kinds to retain, e.g. nnmcts,bigmoney (repeatable)",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--shard-size", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = export_real_corpus(
            args.exports_root,
            output_dir=args.out,
            include_bot_seats=args.include_bot_seats,
            opponents=args.opponents,
            alpha=args.alpha,
            scale=args.scale,
            shard_size=args.shard_size,
        )
    except (OSError, ValueError) as error:
        print(f"tuple export failed: {error}", file=sys.stderr)
        return 2
    print(result.manifest_path)
    print(
        f"processed {result.games_processed} game(s), exported {result.tuples_exported} tuple(s)",
        file=sys.stderr,
    )
    for failure in result.failures:
        print(f"replay failure: {failure}", file=sys.stderr)
    return 2 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
