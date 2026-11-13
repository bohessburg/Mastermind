"""Human-readable arena event dump.

Run with:
    python -m src.v2.arena.protocol.dump [frames.jsonl]
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

from .events import GameEvent, UnknownFrame
from .recording import newest_recording, parse_recording


def format_event(event: GameEvent) -> str:
    timestamp = (
        f"{event.timestamp_ms:013d}" if event.timestamp_ms is not None else " " * 13
    )
    values = []
    for event_field in fields(event):
        if event_field.name == "timestamp_ms":
            continue
        value = getattr(event, event_field.name)
        if isinstance(event, UnknownFrame) and event_field.name == "raw":
            value = f"<{len(value)} bytes>"
        values.append(f"{event_field.name}={value!r}")
    return f"{timestamp} {type(event).__name__} " + " ".join(values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump Dominion arena protocol events")
    parser.add_argument("recording", nargs="?", type=Path)
    parser.add_argument(
        "--hide-unknown",
        action="store_true",
        help="omit unsupported message types from the event listing",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    recording = args.recording or newest_recording()
    if recording is None:
        raise SystemExit("no arena-recordings/*/frames.jsonl found")
    result = parse_recording(recording)
    for event in result.events:
        if args.hide_unknown and isinstance(event, UnknownFrame):
            continue
        print(format_event(event))
    print(
        f"# sessions={len(result.sessions)} "
        f"coverage={result.stats.decode_coverage:.2%} "
        f"unknown={result.unknown_msg_types}"
    )


if __name__ == "__main__":
    main()
