"""Merge scripts/worlds_scripted.py slices: score per arm and paired difference vs 1x128."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/worlds3")
arms: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
for f in sorted(root.glob("*.json")):
    d = json.loads(f.read_text())
    arm = f"{d['worlds']}x{d['sims'] // d['worlds']}"
    arms[(d["opponent"], arm)].update({int(k): v for k, v in d["per_game"].items()})

order = ["1x128", "2x128", "4x128", "8x128", "1x256", "1x512", "1x1024"]
for opp in ("engine3", "bigmoney"):
    base = arms.get((opp, "1x128"), {})
    for arm in order:
        g = arms.get((opp, arm))
        if not g:
            continue
        n = len(g)
        s = sum(g.values()) / n
        row = {"opponent": opp, "arm": arm, "games": n, "score": round(s, 4),
               "se": round(math.sqrt(max(s * (1 - s), 1e-9) / n), 4)}
        common = sorted(set(g) & set(base))
        if arm != "1x128" and common:
            diffs = [g[i] - base[i] for i in common]
            m = sum(diffs) / len(diffs)
            var = sum((x - m) ** 2 for x in diffs) / max(len(diffs) - 1, 1)
            se = math.sqrt(var / len(diffs))
            row.update({"diff_vs_1x128": round(m, 4), "diff_se": round(se, 4),
                        "z": round(m / se, 2) if se > 0 else None})
        print(json.dumps(row))
