#!/bin/bash
# After an arena exit: archive the log, preflight, relaunch on the configured checkpoint.
# usage: arena_relaunch.sh ARCHIVE_NAME
S="$(cd "$(dirname "$0")" && pwd)"
cd /Users/paisho/Projects/Mastermind || exit 9
export PYTHONUNBUFFERED=1
pgrep -f "caffeinate -is env PYTHONPATH=build" >/dev/null && { echo "arena still running; not relaunching"; exit 1; }
[ -e "$S/$1" ] && { echo "$1 exists; pick another archive name"; exit 1; }
cp "$S/arena_c29.log" "$S/$1"
echo "$(date -u +%H:%M:%S) archived $1: $(grep 'arena record' "$S/$1" | tail -n 1)"
grep -n "checkpoint_path" configs/arena_c29.json
OMP_NUM_THREADS=1 PYTHONPATH=build ./.venv/bin/python -m pytest tests/v2/arena/ -q --deselect tests/v2/arena/test_p5_config.py::test_default_arena_config_loads_credentials_only_from_environment 2>&1 | tail -n 1
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "arena tests failed; not relaunching"; exit 1; }
PYTHONPATH=build ./.venv/bin/python -m src.v2.arena.main --config configs/arena_c29.json --dry-run 2>&1 | tail -n 1
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "dry run failed; not relaunching"; exit 1; }
echo "$(date -u +%H:%M:%S) preflight green; ledger rows $(wc -l < exports/arena/record.jsonl | tr -d ' '); relaunching"
exec "$S/arena_retry.sh"
