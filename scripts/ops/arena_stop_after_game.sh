#!/bin/bash
# Wait for the current arena game to finish (new ledger row), then SIGINT the driver so it
# shuts the browser down cleanly between games. Heartbeat every minute.
cd /Users/paisho/Projects/Mastermind
PID=$(pgrep -f "bin/python -m src.v2.arena.main --config" | head -n 1)
[ -z "$PID" ] && { echo "arena not running"; exit 0; }
start=$(wc -l < exports/arena/record.jsonl | tr -d ' ')
echo "$(date -u +%H:%M:%S) arena pid $PID, ledger rows $start, waiting for the game to end"
for i in $(seq 1 1500); do
  kill -0 "$PID" 2>/dev/null || { echo "$(date -u +%H:%M:%S) arena exited on its own"; exit 0; }
  rows=$(wc -l < exports/arena/record.jsonl | tr -d ' ')
  [ "$rows" -gt "$start" ] && break
  [ $((i % 30)) -eq 0 ] && echo "$(date -u +%H:%M:%S) game still in progress"
  sleep 2
done
rows=$(wc -l < exports/arena/record.jsonl | tr -d ' ')
[ "$rows" -gt "$start" ] || { echo "no game finished in 50 minutes; not stopping"; exit 1; }
echo "$(date -u +%H:%M:%S) game ended (rows $rows); sending SIGINT"
tail -n 1 exports/arena/record.jsonl | cut -c1-300
kill -INT "$PID"
for i in $(seq 1 45); do kill -0 "$PID" 2>/dev/null || { echo "$(date -u +%H:%M:%S) arena exited"; exit 0; }; sleep 2; done
echo "$(date -u +%H:%M:%S) arena still alive 90 s after SIGINT"; exit 2
