"""Shared JSON serialization for unified game records."""

from __future__ import annotations

import json
from pathlib import Path

from .model import GameRecord, validate_record


def write_record(record: GameRecord, path: Path | str) -> Path:
    """Validate and atomically write one pretty, deterministic JSON record."""
    validate_record(record)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
