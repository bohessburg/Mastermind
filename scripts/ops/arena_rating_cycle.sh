#!/bin/bash
# usage: arena_rating_cycle.sh TARGET_ROWS
# Wait for the arena to be up, then for the ledger to reach TARGET_ROWS; SIGINT the driver between
# games, poll the leaderboard once and print the i_claudius rating. The caller relaunches the arena.
# exit 0 = rating printed, 2 = arena would not stop (no poll), 4 = arena exited on its own (no poll),
# 6 = poll failed, 3 = arena never came up.
S="$(cd "$(dirname "$0")" && pwd)"
cd /Users/paisho/Projects/Mastermind || exit 9
export PYTHONUNBUFFERED=1
TARGET=$1
rows() { wc -l < exports/arena/record.jsonl | tr -d ' '; }
wait_up() {
  PID=""
  for i in $(seq 1 900); do
    PID=$(pgrep -f "bin/python -m src.v2.arena.main --config" | head -n 1)
    [ -n "$PID" ] && grep -q "live archive" "$S/arena_c29.log" 2>/dev/null && return 0
    PID=""
    [ $((i % 60)) -eq 0 ] && echo "$(date -u +%H:%M:%S) waiting for the arena to come up"
    sleep 2
  done
  return 1
}
wait_up || { echo "arena never came up in 30 minutes"; tail -n 5 "$S/arena_c29.log" | cut -c1-200; exit 3; }
echo "$(date -u +%H:%M:%S) arena pid $PID up, ledger rows $(rows), target $TARGET"
n=0
while true; do
  r=$(rows)
  [ "$r" -ge "$TARGET" ] && break
  if ! kill -0 "$PID" 2>/dev/null; then
    sleep 6
    if pgrep -f "arena_retry.sh" >/dev/null; then
      echo "$(date -u +%H:%M:%S) arena attempt ended, retry loop still alive; waiting for the next attempt"
      wait_up || { echo "arena never came back in 30 minutes"; exit 3; }
      echo "$(date -u +%H:%M:%S) arena pid $PID up, ledger rows $(rows), target $TARGET"
      continue
    fi
    echo "$(date -u +%H:%M:%S) arena process EXITED on its own, ledger rows $r"
    tail -n 12 "$S/arena_c29.log" | cut -c1-300
    exit 4
  fi
  n=$((n+1)); [ $((n % 300)) -eq 0 ] && echo "$(date -u +%H:%M:%S) running, ledger rows $r"
  sleep 2
done
echo "$(date -u +%H:%M:%S) ledger reached $(rows) rows; sending SIGINT between games"
tail -n 1 exports/arena/record.jsonl | cut -c1-300
kill -INT "$PID" 2>/dev/null
gone=0
for i in $(seq 1 60); do
  if ! kill -0 "$PID" 2>/dev/null && ! pgrep -f "caffeinate -is env PYTHONPATH=build" >/dev/null; then gone=1; break; fi
  sleep 2
done
[ "$gone" -eq 1 ] || { echo "$(date -u +%H:%M:%S) arena still alive 120 s after SIGINT; not polling"; exit 2; }
echo "$(date -u +%H:%M:%S) arena exited; ledger rows $(rows); polling the leaderboard once"
./.venv/bin/python scripts/dgames_ratings.py --once > "$S/ratings_poll.log" 2>&1
rc=$?
grep -i "leaderboard\|outcome\|abort\|error\|fail\|heartbeat" "$S/ratings_poll.log" | tail -n 8 | cut -c1-200
echo "$(date -u +%H:%M:%S) poller exit $rc"
grep '"i_claudius"' data/dominion_games/ratings/observations.jsonl | tail -n 2 | ./.venv/bin/python -I -c "
import sys,json
for l in sys.stdin:
    d=json.loads(l)
    print('RATING', d['observed_at_utc'][:19], d['rating_type'], 'level %.2f'%d['rating'], 'mu %.2f'%d['converted_skill'], 'phi %.2f'%d['converted_deviation'], 'glicko2 mu %.4f phi %.4f vol %.4f'%(d['skill'],d['deviation'],d['volatility']), 'rank', d['rank'], 'games', d['game_count'])
"
[ "$rc" -eq 0 ] || exit 6
exit 0
