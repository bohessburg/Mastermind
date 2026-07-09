from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import dominion_v2_py as dz


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
    expected = data.get("final_state_hash")
    if expected is not None and format_hash(first) != str(expected).lower():
        raise AssertionError(f"hash mismatch: replay {format_hash(first)} != export {expected}")
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
