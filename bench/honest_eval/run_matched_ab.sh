#!/bin/bash
# Matched A/B: evaluate.py c15 gen_0045 vs engine3, same seed/scheduler,
# honest OFF vs ON, 200 games each. Isolates the determinize-mode effect
# from kingdom/seed variance (the #9 validation confound).
cd /Users/paisho/Projects/Mastermind || exit 1
export PYTHONPATH=build PYTHONUNBUFFERED=1
OUT=bench/honest_eval
PY=./.venv/bin/python
CKPT=checkpoints/remote/campaign15/gen_0045.pt

echo "ab start $(date)" > "$OUT/ab_status.txt"

$PY -m src.v2.train.evaluate --checkpoint "$CKPT" --opponent engine3 \
  --games 200 --sims 400 --kingdoms random --seed 777001 \
  > "$OUT/ab_clairvoyant.log" 2>&1
echo "clairvoyant exit $? $(date)" >> "$OUT/ab_status.txt"

$PY -m src.v2.train.evaluate --checkpoint "$CKPT" --opponent engine3 \
  --games 200 --sims 400 --kingdoms random --seed 777001 --honest \
  > "$OUT/ab_honest.log" 2>&1
echo "honest exit $? $(date)" >> "$OUT/ab_status.txt"

echo "ab done $(date)" >> "$OUT/ab_status.txt"
touch "$OUT/AB_DONE"
