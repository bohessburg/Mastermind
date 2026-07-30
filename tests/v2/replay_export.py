from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import dominion_v2_py as dz

from src.v2.records.local import final_state_hash_matches


def format_hash(value: int) -> str:
    return f"0x{int(value):016x}"


def replay_export_data(data: dict[str, Any]) -> int:
    seats = list(data["seats"])
    setup = dz.Setup(players=len(seats), kingdom=list(data["kingdom"]))
    game = dz.new_game(setup, int(data["seed"]))
    for index, action in enumerate(data["actions"]):
        action_int = int(action)
        mask = game.legal_mask()
        if action_int < 0 or action_int >= len(mask) or not bool(mask[action_int]):
            raise AssertionError(f"illegal action at export index {index}: {action_int}")
        game.step(action_int)
    return int(game.state_hash())


def verify_export_data(data: dict[str, Any]) -> int:
    first = replay_export_data(data)
    second = replay_export_data(data)
    if first != second:
        raise AssertionError(f"nondeterministic replay: {format_hash(first)} != {format_hash(second)}")
    if not final_state_hash_matches(data, first):
        raise AssertionError(
            f"hash mismatch: replay {format_hash(first)} matches no recorded generation"
        )
    return first


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: replay_export.py EXPORT.json", file=sys.stderr)
        return 2
    with Path(argv[1]).open("r", encoding="utf-8") as file:
        data = json.load(file)
    print(format_hash(verify_export_data(data)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
