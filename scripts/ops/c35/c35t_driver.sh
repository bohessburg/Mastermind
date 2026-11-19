#!/bin/bash
# c35 truncate continued from gen 6 to gen 60 or a 10-rejection plateau; resumes after a crash.
cd /root/Mastermind
for try in 1 2 3 4 5 6; do
  echo "$(date -u +%FT%TZ) c35t driver: start (try $try)"
  bash scripts/box/resume.sh configs/run_c35_truncate_long.json checkpoints/remote/bc_level1_R40B/bc_f100.pt
  rc=$?
  echo "$(date -u +%FT%TZ) c35t driver: exited rc=$rc"
  [ $rc -eq 0 ] && break
  sleep 20
done
echo "$(date -u +%FT%TZ) C35T DONE"
