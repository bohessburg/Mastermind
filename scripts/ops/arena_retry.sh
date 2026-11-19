#!/bin/bash
# arena_retry.sh -- launch the arena; while it exits 5 within 3 minutes (retained live table), wait 10 min and retry (max 12 tries).
S="$(cd "$(dirname "$0")" && pwd)"
cd /Users/paisho/Projects/Mastermind || exit 9
for i in $(seq 1 12); do
  if pgrep -x caffeinate > /dev/null && pgrep -f "caffeinate -is env PYTHONPATH=build" > /dev/null; then echo "arena already running"; exit 8; fi
  start=$(date +%s)
  echo "$(date -u +%H:%M:%S) attempt $i: launching"
  PYTHONUNBUFFERED=1 ./scripts/arena_overnight.sh --config configs/arena_c29.json > "$S/arena_c29.log" 2>&1
  rc=$?
  dur=$(( $(date +%s) - start ))
  echo "$(date -u +%H:%M:%S) attempt $i: exit $rc after ${dur}s"
  if [ "$rc" -ne 5 ] || [ "$dur" -gt 180 ]; then exit "$rc"; fi
  cp "$S/arena_c29.log" "$S/arena_c29_retry_$i.log"
  for m in 1 2; do sleep 300; echo "$(date -u +%H:%M:%S) waiting for the retained game to end ($((m*5))/10 min)"; done
done
echo "still exit 5 after 12 attempts"; exit 5
