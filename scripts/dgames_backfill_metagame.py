"""Backfill card pool level and player ratings into existing capture manifests.

Reads each raw capture only until its METAGAME_INFO log entry, then writes the
same fields the collector now records (see ``_metagame_manifest_fields``).
Manifests of captures still in progress are left to the collector.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dgames_collect import _atomic_json_write, _metagame_manifest_fields  # noqa: E402
from src.v2.arena.protocol.parser import ArenaParser  # noqa: E402
from src.v2.arena.protocol.recording import decode_record_binary  # noqa: E402

FIELDS = ("card_pool_level", "rated_game", "player_ratings", "top_rating")


def _read_metagame(raw_path: Path, game_id: int) -> ArenaParser:
    parser = ArenaParser()
    with gzip.open(raw_path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("kind") != "binary":
                continue
            frame = decode_record_binary(record, source=str(raw_path))
            if frame is None:
                continue
            try:
                parser.parse_frame(frame)
            except Exception:
                continue
            if game_id in parser.metagame_by_game:
                break
    return parser


def _names_by_id(manifest: dict) -> dict[int, str]:
    players = ((manifest.get("outcome") or {}).get("game_result") or {}).get("players") or []
    return {int(p["player_id"]): str(p.get("player_name")) for p in players if "player_id" in p}


def main(argv: list[str] | None = None) -> int:
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--raw-root", type=Path, required=True)
    args.add_argument("--force", action="store_true", help="rewrite manifests that already have the fields")
    opts = args.parse_args(argv)
    updated = skipped = missing_raw = no_info = 0
    paths = sorted(opts.raw_root.glob("*.manifest.json"))
    for index, manifest_path in enumerate(paths, start=1):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("incomplete_reason") == "capture in progress" or (
            not opts.force and all(field in manifest for field in FIELDS)
        ):
            skipped += 1
            continue
        game_id = int(manifest["game_id"])
        raw_path = manifest_path.with_name(f"{game_id}.jsonl.gz")
        if not raw_path.exists():
            missing_raw += 1
            continue
        parser = _read_metagame(raw_path, game_id)
        names = _names_by_id(manifest) or dict(parser.player_names_by_id)
        fields = _metagame_manifest_fields(parser, game_id, names)
        if fields["card_pool_level"] is None:
            no_info += 1
        manifest.update(fields)
        _atomic_json_write(manifest_path, manifest)
        updated += 1
        if index % 1000 == 0:
            print(f"backfill: {index}/{len(paths)} manifests scanned, {updated} updated", flush=True)
    print(
        f"backfill done: {len(paths)} manifests, updated={updated}, skipped={skipped}, "
        f"no_metagame_info={no_info}, missing_raw={missing_raw}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
