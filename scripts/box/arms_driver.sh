#!/bin/bash
# c32 train-steps arms (2026-11-15): run each arm for 6 generations, then the
# 2000-game ladders against c31 gen 27. Resumable: a restart (supervisor,
# reboot) skips finished arms and resumes a part-done one from its checkpoint.
#   supervisor program dominion-arms; log /root/Mastermind/arms_driver.log
cd /root/Mastermind || exit 1
INIT=checkpoints/remote/campaign31/gen_0027.pt
GENS=6
log() { echo "$(date -u +%H:%M:%S) $*"; }
rows() { [ -f "$1" ] && echo $(( $(wc -l < "$1") - 1 )) || echo 0; }
while pgrep -f "configs/run_c33_trial.json" >/dev/null; do sleep 30; done
for n in s160 s240 s160h; do
  M=checkpoints/campaign32_$n/metrics.csv
  if [ "$(rows $M)" -ge $GENS ]; then log "arm $n already has $(rows $M) generations"; continue; fi
  log "arm $n: starting at $(rows $M) generations"
  bash scripts/box/resume.sh configs/run_c32_$n.json $INIT >> arms_$n.log 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null && [ "$(rows $M)" -lt $GENS ]; do sleep 20; done
  if [ "$(rows $M)" -lt $GENS ]; then log "arm $n: trainer exited early at $(rows $M) generations"; exit 4; fi
  log "arm $n: $GENS generations done; stopping the trainer"
  kill -INT $P 2>/dev/null; sleep 30; kill -TERM $P 2>/dev/null; sleep 10
  for pid in $(pgrep -f "dominion-selfplay|multiprocessing.spawn|src.v2.train.train" || true); do kill -9 "$pid" 2>/dev/null; done
  sleep 5
done
log "ladders"
PYTHONPATH=build:. PYTHONUNBUFFERED=1 .venv/bin/python scripts/box/ladder.py || { log "ladder failed"; exit 5; }
log "ALL DONE"
