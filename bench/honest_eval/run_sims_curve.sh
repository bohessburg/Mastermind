#!/bin/bash
# Honest sims curve (task #3): c15 gen_0045 vs engine3 at 200/800/1600 sims
# (400-point = baseline run, same seed/boards) + c19 gen_0040 at 1600.
cd /Users/paisho/Projects/Mastermind || exit 1
export PYTHONPATH=build PYTHONUNBUFFERED=1
OUT=bench/honest_eval
PY=./.venv/bin/python

echo "curve start $(date)" > "$OUT/curve_status.txt"

for SIMS in 200 800 1600; do
  $PY -m src.v2.train.honest_eval \
    --a checkpoints/remote/campaign15/gen_0045.pt --opponent engine3 \
    --games 200 --sims "$SIMS" --determinizations 2 --seed 20260728 \
    --out "$OUT/c15g45_vs_engine3_${SIMS}.json" > "$OUT/c15g45_vs_engine3_${SIMS}.log" 2>&1
  echo "c15 sims=$SIMS exit $? $(date)" >> "$OUT/curve_status.txt"
done

$PY -m src.v2.train.honest_eval \
  --a checkpoints/remote/campaign19/gen_0040.pt --opponent engine3 \
  --games 200 --sims 1600 --determinizations 2 --seed 20260728 \
  --out "$OUT/c19g40_vs_engine3_1600.json" > "$OUT/c19g40_vs_engine3_1600.log" 2>&1
echo "c19 sims=1600 exit $? $(date)" >> "$OUT/curve_status.txt"

echo "curve done $(date)" >> "$OUT/curve_status.txt"
touch "$OUT/CURVE_DONE"
