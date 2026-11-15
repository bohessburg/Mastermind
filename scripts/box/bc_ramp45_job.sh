#!/bin/bash
# Level-1 BC with the policy-weight ramp starting at 45 instead of 40 (two seeds),
# scored like bc_level1_job.sh: raw policy vs engine3 (2000 games) and 800-game
# head-to-head with search vs bc_a. Run from the checkout given as $1 (default
# /root/gatetest); waits for bc_level1_job to finish first. Resumable.
REPO=${1:-/root/gatetest}
cd "$REPO" || exit 1
export PYTHONPATH=build:. PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT=bench/bc_level1
log() { echo "$(date -u +%H:%M:%S) $*"; }
while pgrep -f "bc_level1_job_gatetest|bc_learning_curve.py --arms f100 --train-tuples exports/tuples_dgames_level1_bc/train --val-tuples exports/tuples_all_40plus/val --model-config configs/run_c25.json --device cuda --seed 2026110[89] " >/dev/null; do sleep 30; done
if [ -f /root/ship/bcwork.bundle ]; then git fetch -q /root/ship/bcwork.bundle v2-phase1 && git merge -q --ff-only FETCH_HEAD; fi
log "repo at $(git log --oneline -1)"
for s in A45:20261108 B45:20261109; do
  n=${s%%:*}; seed=${s##*:}
  mkdir -p /root/Mastermind/checkpoints/bc_level1_$n; ln -sfn /root/Mastermind/checkpoints/bc_level1_$n checkpoints/bc_level1_$n
  [ -f checkpoints/bc_level1_$n/bc_f100.pt ] && continue
  log "BC ramp-45 seed $n ($seed)"
  $PY scripts/bc_learning_curve.py --arms f100 --train-tuples exports/tuples_dgames_level1_bc/train \
    --val-tuples exports/tuples_all_40plus/val --model-config configs/run_c25.json --device cuda --seed $seed \
    --ramp-start 45 --output-dir checkpoints/bc_level1_$n --report-dir $OUT/bc_$n || exit 3
done
for n in A45 B45; do
  grep -q "^engine3,2000" $OUT/raw_engine3_$n.log 2>/dev/null && continue
  log "raw engine3 $n"
  $PY -m src.v2.train.evaluate --checkpoint checkpoints/bc_level1_$n/bc_f100.pt --opponent engine3 --games 2000 \
    --sims 2 --honest --kingdoms random --device cuda > $OUT/raw_engine3_$n.log 2>&1 || exit 4
  tail -1 $OUT/raw_engine3_$n.log
done
log "head-to-head vs bc_a"
$PY scripts/box/bc_h2h.py A45 B45 || exit 5
log "ALL DONE"
