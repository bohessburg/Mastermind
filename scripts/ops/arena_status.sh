#!/bin/bash
# arena_status.sh [WAIT_SECONDS] -- print whether the arena is up, ledger rows, newest run dir and log tail.
# Lives in a file so that the calling shell's command text never contains the arena pgrep pattern
# (arena_retry.sh refuses to launch, exit 8, while any process command line matches it).
S="$(cd "$(dirname "$0")" && pwd)"
cd /Users/paisho/Projects/Mastermind || exit 9
[ -n "$1" ] && sleep "$1"
date -u +%H:%M:%S
if pgrep -f "caffeinate -is env PYTHONPATH=build" >/dev/null; then echo "arena UP"; else echo "arena DOWN"; fi
echo "helpers alive: $(pgrep -f 'arena_retry.sh|arena_rating_cycle.sh' | wc -l | tr -d ' ')"
echo "ledger rows: $(wc -l < exports/arena/record.jsonl | tr -d ' ')"
d=$(ls -t exports/arena | grep -v record | head -n 1); echo "newest run: $d"; ls "exports/arena/$d" | head -n 4
tail -n 3 "$S/arena_c29.log" 2>/dev/null | cut -c1-200
