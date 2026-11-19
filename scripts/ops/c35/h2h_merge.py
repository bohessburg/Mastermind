"""Merge c35 head-to-head results (sharded tags _s0.._s3 combined), paired SE per board."""
import glob, json, math, re, sys
from collections import defaultdict
root = sys.argv[1]
groups = defaultdict(list)
for f in sorted(glob.glob(f"{root}/*.json")):
    d = json.load(open(f))
    groups[re.sub(r"_s\d+$", "", d["tag"])].append(d)
for tag, parts in sorted(groups.items()):
    s0 = [x for d in parts for x in d["a_scores_seat0"]]
    s1 = [x for d in parts for x in d["a_scores_seat1"]]
    w = sum(d["wins"] for d in parts); l = sum(d["losses"] for d in parts); t = sum(d["ties"] for d in parts)
    n = len(s0)
    pair = [(a + b) / 2 for a, b in zip(s0, s1)]
    m = sum(pair) / n
    se = math.sqrt(sum((x - m) ** 2 for x in pair) / (n - 1) / n)
    elo = 400 * math.log10(m / (1 - m)) if 0 < m < 1 else float("nan")
    print(f"{tag:22s} {w}-{l}-{t}  score {m:.3f} +- {se:.3f}  elo {elo:+.0f}  shards {len(parts)}  wall {max(d['wall_s'] for d in parts):.0f}s")
