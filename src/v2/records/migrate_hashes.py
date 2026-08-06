"""Append current-generation state hashes to legacy local web exports.

The command deliberately uses the existing corpus manifest as its migration
inventory. It replays every non-pytest, non-stub candidate but writes only
legal, game-over replays; failures leave their files byte-for-byte untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import dominion_v2_py as dz

from .corpus import build_manifest, manifest_path
from .local import is_local_export_data


VERIFICATION_FIELD = "final_state_hash_verification"
SKIPPED_CLASSIFICATIONS = frozenset({"pytest_artifact", "stub"})


@dataclass(frozen=True)
class MigrationFailure:
    path: Path
    reason: str


@dataclass(frozen=True)
class MigrationReport:
    migrated: tuple[Path, ...]
    skipped: tuple[Path, ...]
    failed: tuple[MigrationFailure, ...]


def _current_generation() -> int:
    # State-hash and encoder generation currently coincide at the
    # landscape-sentinel repair, so the engine's existing marker is canonical.
    return int(dz.ENCODER_GENERATION)


def _gen_field() -> str:
    return f"final_state_hash_gen{_current_generation()}"


def _format_hash(value: int) -> str:
    return f"0x{int(value):016x}"


def _verification_record() -> dict[str, Any]:
    return {
        "checked_generations": [1, _current_generation()],
        "matching_generations": [_current_generation()],
    }


def _load_manifest_entries(exports_root: Path) -> tuple[Mapping[str, Any], ...]:
    """Read the existing manifest without refreshing its pre-migration labels."""
    source = manifest_path(exports_root)
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read existing corpus manifest {source}: {error}") from error
    entries = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"invalid corpus manifest {source}: files must be a list")
    valid_entries = tuple(entry for entry in entries if isinstance(entry, Mapping))
    if len(valid_entries) != len(entries):
        raise ValueError(f"invalid corpus manifest {source}: entries must be objects")
    return valid_entries


def _candidate_paths(
    exports_root: Path,
) -> tuple[tuple[tuple[Path, str], ...], tuple[Path, ...]]:
    """Return manifest-selected replay candidates and quarantined source files."""
    candidates: list[tuple[Path, str]] = []
    skipped: list[Path] = []
    for entry in _load_manifest_entries(exports_root):
        relative = entry.get("path")
        classification = entry.get("classification")
        if not isinstance(relative, str) or not isinstance(classification, str):
            raise ValueError("invalid corpus manifest entry: path and classification are required")
        path = exports_root / relative
        try:
            path.relative_to(exports_root)
        except ValueError as error:
            raise ValueError(f"manifest path escapes export root: {relative!r}") from error
        if classification in SKIPPED_CLASSIFICATIONS:
            skipped.append(path)
        else:
            candidates.append((path, classification))
    return tuple(candidates), tuple(skipped)


def _replay_hash(data: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Replay once and return ``(hash, error)`` without mutating source data."""
    if not is_local_export_data(data):
        return None, "not a local web export"
    try:
        seats = data["seats"]
        kingdom = data["kingdom"]
        actions = data["actions"]
        game = dz.new_game(
            dz.Setup(players=len(seats), kingdom=[int(value) for value in kingdom]),
            int(data["seed"]),
        )
        for index, raw_action in enumerate(actions):
            action = int(raw_action)
            mask = game.legal_mask()
            if action < 0 or action >= len(mask) or not bool(mask[action]):
                return None, f"illegal action at index {index}: {action}"
            game.step(action)
        if not game.game_over():
            return None, "replay did not reach game_over"
        return _format_hash(game.state_hash()), None
    except (KeyError, TypeError, ValueError) as error:
        return None, f"replay setup failed: {error}"


def _append_migration_fields(raw: bytes, state_hash: str) -> bytes:
    """Insert fields before the final JSON object brace, preserving all prior bytes."""
    end = len(raw)
    while end and raw[end - 1] in b" \t\r\n":
        end -= 1
    if end == 0 or raw[end - 1 : end] != b"}":
        raise ValueError("source JSON is not an object ending in '}'")
    fields = {
        _gen_field(): state_hash,
        VERIFICATION_FIELD: _verification_record(),
    }
    appended = json.dumps(fields, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    # Strip both object braces because the source object already has members.
    return raw[: end - 1] + b"," + appended[1:-1] + raw[end - 1 :]


def _migrate_one(path: Path, *, dry_run: bool) -> tuple[bool, str | None]:
    """Migrate one candidate, returning ``(migrated, failure_reason)``."""
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return False, f"cannot decode JSON: {error}"
    if not isinstance(data, Mapping):
        return False, "JSON root is not an object"

    actual_hash, error = _replay_hash(data)
    if error is not None:
        return False, error
    assert actual_hash is not None

    generation_field = _gen_field()
    existing = data.get(generation_field)
    if existing is not None:
        if str(existing).lower() == actual_hash:
            return False, None
        return False, f"existing {generation_field} conflicts with replay: {existing}"
    if VERIFICATION_FIELD in data:
        return False, f"existing {VERIFICATION_FIELD} without {generation_field}"

    migrated = _append_migration_fields(raw, actual_hash)
    if not dry_run:
        path.write_bytes(migrated)
    return True, None


def migrate_exports(
    exports_root: Path | str = Path("exports"),
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> MigrationReport:
    """Replay manifest-selected legacy exports and append verified gen2 hashes."""
    root = Path(exports_root)
    candidates, quarantined = _candidate_paths(root)
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        candidates = candidates[:limit]

    migrated: list[Path] = []
    skipped: list[Path] = list(quarantined)
    failed: list[MigrationFailure] = []
    for path, classification in candidates:
        if classification == "incomplete":
            # The existing manifest was generated before this hash-generation
            # change. Its incomplete label distinguishes already-corrupt
            # legacy hashes from valid gen1 hashes that now need a gen2 value.
            _, error = _migrate_one(path, dry_run=True)
            if error is None:
                error = (
                    "manifest marks replay incomplete: legacy final_state_hash "
                    "failed pre-migration verification"
                )
            failed.append(MigrationFailure(path=path, reason=error))
            continue
        changed, error = _migrate_one(path, dry_run=dry_run)
        if error is not None:
            failed.append(MigrationFailure(path=path, reason=error))
        elif changed:
            migrated.append(path)
        else:
            skipped.append(path)

    if not dry_run and migrated:
        # Refresh only after selection: before migration, the legacy hashes must
        # not cause historical real exports to be relabeled as incomplete.
        build_manifest(root)
    return MigrationReport(tuple(migrated), tuple(skipped), tuple(failed))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append replay-verified current-generation hashes to legacy exports."
    )
    parser.add_argument(
        "--exports-root",
        type=Path,
        default=Path("exports"),
        help="root containing corpus_manifest.json and optional hetzner/ exports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="replay and report results without changing exports or the manifest",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="process at most this many non-artifact/non-stub manifest candidates",
    )
    return parser


def _print_report(report: MigrationReport, *, dry_run: bool) -> None:
    verb = "would migrate" if dry_run else "migrated"
    for path in report.migrated:
        print(f"{verb}: {path}")
    for failure in report.failed:
        print(f"failed: {failure.path}: {failure.reason}", file=sys.stderr)
    print(
        f"migration report: migrated={len(report.migrated)} "
        f"skipped={len(report.skipped)} failed={len(report.failed)}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = migrate_exports(
            args.exports_root,
            dry_run=bool(args.dry_run),
            limit=args.limit,
        )
    except ValueError as error:
        print(f"hash migration failed: {error}", file=sys.stderr)
        return 2
    _print_report(report, dry_run=bool(args.dry_run))
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
