from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFS_PATH = Path(__file__).resolve().parents[1] / "client-data" / "defs.gen.json"


@lru_cache(maxsize=1)
def load_defs() -> dict[str, Any]:
    with DEFS_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    data["by_id"] = {int(card["id"]): card for card in data["defs"]}
    data["by_name"] = {card["name"]: card for card in data["defs"]}
    return data


def def_by_id(def_id: int) -> dict[str, Any]:
    return load_defs()["by_id"][int(def_id)]


def def_id(name: str) -> int:
    return int(load_defs()["by_name"][name]["id"])


def def_name(def_id: int | None) -> str:
    if def_id is None:
        return ""
    return str(def_by_id(int(def_id))["name"])
