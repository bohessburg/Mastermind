#!/bin/bash
# Honest re-baseline chain (task #2): three 200-game runs @400 sims / K=2.
# Detached runner; progress in per-run .log files, sentinel BASELINES_DONE at end.
cd /Users/paisho/Projects/Mastermind || exit 1
export PYTHONPATH=build PYTHONUNBUFFERED=1
OUT=bench/honest_eval
PY=./.venv/bin/python

echo "start $(date)" > "$OUT/chain_status.txt"

$PY -m src.v2.train.honest_eval \
  --a checkpoints/remote/campaign15/gen_0045.pt --opponent engine3 \
  --games 200 --sims 400 --determinizations 2 --seed 20260728 \
  --out "$OUT/c15g45_vs_engine3_400.json" > "$OUT/c15g45_vs_engine3_400.log" 2>&1
echo "run1 exit $? $(date)" >> "$OUT/chain_status.txt"

$PY -m src.v2.train.honest_eval \
  --a checkpoints/remote/campaign19/gen_0040.pt --opponent engine3 \
  --games 200 --sims 400 --determinizations 2 --seed 20260728 \
  --out "$OUT/c19g40_vs_engine3_400.json" > "$OUT/c19g40_vs_engine3_400.log" 2>&1
echo "run2 exit $? $(date)" >> "$OUT/chain_status.txt"

$PY -m src.v2.train.honest_eval \
  --a checkpoints/remote/campaign19/gen_0040.pt --b checkpoints/remote/campaign15/gen_0045.pt \
  --games 200 --sims 400 --determinizations 2 --seed 20260728 \
  --out "$OUT/c19g40_vs_c15g45_400.json" > "$OUT/c19g40_vs_c15g45_400.log" 2>&1
echo "run3 exit $? $(date)" >> "$OUT/chain_status.txt"

echo "done $(date)" >> "$OUT/chain_status.txt"
touch "$OUT/BASELINES_DONE"
