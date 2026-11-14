"""Print the last N i_claudius leaderboard observations with their percentile among RATINGS_2P rows."""
import json, sys
n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
path = 'data/dominion_games/ratings/observations.jsonl'
mine = []
with open(path) as f:
    for line in f:
        if '"i_claudius"' in line:
            mine.append(json.loads(line))
mine = [d for d in mine if d['rating_type'] == 'RATINGS_2P'][-n:]
stamps = {d['observed_at_utc'] for d in mine}
ratings = {s: [] for s in stamps}
by_type = {s: {} for s in stamps}
with open(path) as f:
    for line in f:
        i = line.find('"observed_at_utc":"')
        s = line[i + 19:i + 19 + 27]
        if s in stamps:
            d = json.loads(line)
            by_type[s][d['rating_type']] = by_type[s].get(d['rating_type'], 0) + 1
            if d['rating_type'] == 'RATINGS_2P':
                ratings[s].append(d['rating'])
for d in mine:
    s = d['observed_at_utc']
    total = len(ratings[s])
    below = sum(1 for r in ratings[s] if r < d['rating'])
    print('RATING', s[:19], 'level %.2f' % d['rating'], 'mu %.2f' % d['converted_skill'], 'phi %.2f' % d['converted_deviation'],
          'rank %d of %d' % (d['rank'], total), 'percentile %.1f' % (100.0 * below / total), 'top %.1f%%' % (100.0 * d['rank'] / total),
          'games', d['game_count'], 'types', by_type[s])
