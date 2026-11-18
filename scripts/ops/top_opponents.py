import json, bisect, collections
from datetime import datetime
def ts(s): return datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
games=[json.loads(l) for l in open('exports/arena/record.jsonl') if l.strip()]
names={g['opponent'] for g in games}
obs=collections.defaultdict(list)
polls=set()
for l in open('data/dominion_games/ratings/observations.jsonl'):
    if '"RATINGS_2P"' not in l: continue
    r=json.loads(l)
    polls.add(r['observed_at_utc'][:13])
    if r['player_name'] in names:
        obs[r['player_name']].append((ts(r['observed_at_utc']),r['rating'],r['converted_skill'],r['converted_deviation'],r['rank'],r['game_count']))
print('games',len(games),'distinct opponents',len(names),'with a 2P leaderboard row',len(obs),'poll hours',len(polls))
rows=[]
for g in games:
    o=obs.get(g['opponent'])
    if not o: continue
    o.sort(); t=ts(g['timestamp'])
    best=min(o,key=lambda x:abs(x[0]-t))
    rows.append((best[1],g,best,abs(best[0]-t)/86400))
print('games vs rated-listed opponents',len(rows))
rows.sort(key=lambda x:-x[0])
for lvl,g,b,gap in rows[:12]:
    print(f"{lvl:6.2f} mu {b[2]:.2f} phi {b[3]:.2f} rank {b[4]:5d} games {b[5]:6d} | {g['opponent']:<22} {g['result']:<5} {g['scores']} {g['timestamp'][:16]} game {g['game_id']} obs gap {gap:.1f}d")
w=[r for r in rows if r[1]['result']=='win']
print('--- highest beaten')
for lvl,g,b,gap in w[:5]:
    print(f"{lvl:6.2f} mu {b[2]:.2f} rank {b[4]:5d} | {g['opponent']:<22} {g['scores']} {g['timestamp'][:16]} game {g['game_id']} obs gap {gap:.1f}d")
