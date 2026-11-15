#!/usr/bin/env python3
"""Write a BC training split of the level-1 corpus that excludes bc_a's validation games.

The level-1 corpus (exports/tuples_dgames_level1) contains every game of the
old 40+ corpus, including its validation games. Dropping exactly those games
lets a BC trained here be validated on exports/tuples_all_40plus/val, the set
bc_a was validated on, so the two accuracies compare directly.

  python scripts/bc_split_level1.py [--source DIR] [--val DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path("exports/tuples_dgames_level1"))
    parser.add_argument("--val", type=Path, default=Path("exports/tuples_all_40plus/val"))
    parser.add_argument("--out", type=Path, default=Path("exports/tuples_dgames_level1_bc/train"))
    args = parser.parse_args()
    manifest = json.loads((args.source / "tuple_manifest.json").read_text())
    val_ids = {str(game["id"]) for game in json.loads((args.val / "tuple_manifest.json").read_text())["games"]}
    held_indices = np.asarray(sorted(int(g["index"]) for g in manifest["games"] if str(g["id"]) in val_ids), dtype=np.int64)
    args.out.mkdir(parents=True, exist_ok=True)
    shards, kept_rows, dropped_rows = [], 0, 0
    for entry in manifest["shards"]:
        with np.load(args.source / entry["path"]) as shard:
            arrays = {name: shard[name] for name in shard.files}
        keep = ~np.isin(arrays["game_index"].astype(np.int64), held_indices)
        dropped_rows += int((~keep).sum())
        if not keep.any():
            continue
        np.savez_compressed(args.out / entry["path"], **{name: values[keep] for name, values in arrays.items()})
        shards.append({**entry, "tuples": int(keep.sum())})
        kept_rows += int(keep.sum())
    out_manifest = dict(manifest)
    out_manifest["shards"] = shards
    out_manifest["totals"] = {**manifest.get("totals", {}), "tuples_exported": kept_rows}
    out_manifest["bc_split"] = {
        "source": str(args.source), "excluded_val_manifest": str(args.val),
        "excluded_games": int(held_indices.size), "excluded_rows": dropped_rows,
    }
    (args.out / "tuple_manifest.json").write_text(json.dumps(out_manifest) + "\n")
    print(json.dumps({"kept_rows": kept_rows, "dropped_rows": dropped_rows,
                      "excluded_games": int(held_indices.size), "val_games": len(val_ids)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
