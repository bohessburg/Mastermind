"""Value head vs actual arena outcome, by the bot's deck size (positions from scripts/arena_positions.py)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))
import dominion_v2_py as dz  # noqa: E402
from src.v2.arena.bot.policy import load_policy  # noqa: E402
from thin_worlds import BUCKETS, bucket_of, int_keys  # noqa: E402

policy = load_policy(Path(sys.argv[2]), device="mps")
torch = policy.torch
rows, obs, masks = [], [], []
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r["outcome"] not in ("win", "loss"):
        continue
    g = dz.game_from_snapshot(int_keys(r["snapshot"]))
    obs.append(g.encode(int(r["our_seat"]), policy.obs_version))
    masks.append(g.legal_mask())
    rows.append((bucket_of(int(r["our_owned"])), r["outcome"] == "win", int(r["turn"] or 0), r["game_id"]))
values = []
with torch.no_grad():
    for s in range(0, len(obs), 2048):
        _, v = policy.evaluate(torch.as_tensor(np.stack(obs[s:s + 2048]), dtype=torch.float32, device=policy.device),
                               torch.as_tensor(np.stack(masks[s:s + 2048]), dtype=torch.bool, device=policy.device))
        values.extend(v.detach().cpu().numpy().reshape(-1).tolist())
values = np.asarray(values)
print(f"{len(values)} positions; value range {values.min():.3f}..{values.max():.3f}, median {np.median(values):.3f}")
t = defaultdict(list)
for (b, won, turn, gid), v in zip(rows, values):
    t[b].append((v, won, gid))
print("bucket  n      games  win%   meanV(won) meanV(lost)  lost-but-V>0.6  AUC")
for lo, _ in BUCKETS:
    b = bucket_of(lo)
    x = t.get(b)
    if not x:
        continue
    v = np.array([a for a, _, _ in x]); w = np.array([c for _, c, _ in x])
    pos, neg = v[w], v[~w]
    # AUC: chance a won-game position has higher value than a lost-game one
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / max(len(pos) * len(neg), 1)
    games = len({g for _, _, g in x})
    print(f"{b:7} {len(x):6} {games:5}  {w.mean():5.1%}  {pos.mean():9.3f}  {neg.mean():10.3f}  "
          f"{(neg > 0.6).mean():13.1%}  {auc:.3f}")
