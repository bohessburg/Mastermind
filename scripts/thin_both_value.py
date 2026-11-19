import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.append('.'); sys.path.append('scripts')
import dominion_v2_py as dz
from src.v2.arena.bot.policy import load_policy
from thin_worlds import int_keys
policy = load_policy(Path('checkpoints/remote/campaign33/gen_0051.pt'), device='mps'); torch = policy.torch
rows, obs, masks = [], [], []
for line in open('bench/thin/positions.jsonl'):
    r = json.loads(line)
    if r['outcome'] not in ('win', 'loss') or not r['turn']: continue
    g = dz.game_from_snapshot(int_keys(r['snapshot']))
    obs.append(g.encode(int(r['our_seat']), policy.obs_version)); masks.append(g.legal_mask())
    rows.append(r)
vals = []
with torch.no_grad():
    for s in range(0, len(obs), 2048):
        _, v = policy.evaluate(torch.as_tensor(np.stack(obs[s:s+2048]), dtype=torch.float32, device=policy.device),
                               torch.as_tensor(np.stack(masks[s:s+2048]), dtype=torch.bool, device=policy.device))
        vals.extend(v.cpu().numpy().reshape(-1).tolist())
t = defaultdict(list)
for r, v in zip(rows, vals):
    turn = int(r['turn']); tb = '1-8' if turn <= 8 else '9-12' if turn <= 12 else '13-16' if turn <= 16 else '17+'
    kind = 'both<=9' if r['our_owned'] <= 9 and r['opp_owned'] <= 9 else 'us<=9 only' if r['our_owned'] <= 9 else 'opp<=9 only' if r['opp_owned'] <= 9 else 'neither'
    t[(tb, kind)].append((v, r['outcome'] == 'win', r['game_id']))
print('turns  kind         n     games  win%   meanV  predicted-win%  lost&V>0.6 (of lost)')
for tb in ('1-8', '9-12', '13-16', '17+'):
    for kind in ('both<=9', 'us<=9 only', 'opp<=9 only', 'neither'):
        x = t.get((tb, kind))
        if not x or len(x) < 50: continue
        v = np.array([a for a, _, _ in x]); w = np.array([b for _, b, _ in x])
        lost = v[~w]
        print(f"{tb:6} {kind:11} {len(x):6} {len({g for *_, g in x}):5}  {w.mean():5.1%}  {v.mean():6.3f}  {(v > 0).mean():13.1%}  {(lost > 0.6).mean() if len(lost) else 0:8.1%}")
