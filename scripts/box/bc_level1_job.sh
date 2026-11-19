#!/bin/bash
# BC on the level-1 corpus, two seeds, against bc_a (2026-11-15). Resumable.
#   1. split: level-1 minus bc_a's 492 validation games
#   2. BC (bc_a recipe) seeds A and B, validated on exports/tuples_all_40plus/val
#   3. raw policy vs engine3, 2000 games each: bc_a, bc_f100, A, B
#   4. head-to-head with search vs bc_a, 800 games each (c32 gate settings)
cd /root/Mastermind || exit 1
export PYTHONPATH=build:. PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT=bench/bc_level1; mkdir -p $OUT
log() { echo "$(date -u +%H:%M:%S) $*"; }
[ -f exports/tuples_dgames_level1_bc/train/tuple_manifest.json ] || { log split; $PY scripts/bc_split_level1.py || exit 2; }
for s in A:20261108 B:20261109; do
  n=${s%%:*}; seed=${s##*:}
  [ -f checkpoints/bc_level1_$n/bc_f100.pt ] && continue
  log "BC seed $n ($seed)"
  $PY scripts/bc_learning_curve.py --arms f100 --train-tuples exports/tuples_dgames_level1_bc/train \
    --val-tuples exports/tuples_all_40plus/val --model-config configs/run_c25.json --device cuda --seed $seed \
    --output-dir checkpoints/bc_level1_$n --report-dir $OUT/bc_$n || exit 3
done
for c in bca:checkpoints/campaign25/bc_a.pt f100:checkpoints/bc_curve/bc_f100.pt A:checkpoints/bc_level1_A/bc_f100.pt B:checkpoints/bc_level1_B/bc_f100.pt; do
  n=${c%%:*}; ck=${c#*:}
  grep -q "^engine3,2000" $OUT/raw_engine3_$n.log 2>/dev/null && continue
  log "raw engine3 $n"
  $PY -m src.v2.train.evaluate --checkpoint $ck --opponent engine3 --games 2000 --sims 2 --honest --kingdoms random --device cuda > $OUT/raw_engine3_$n.log 2>&1 || exit 4
  tail -1 $OUT/raw_engine3_$n.log
done
log "head-to-head vs bc_a"
$PY scripts/box/bc_h2h.py || exit 5
log "ALL DONE"
