#!/bin/bash
# Run exactly one generation of configs/run_c31_mac.json on this Mac, then stop the trainer.
# Heartbeat every 5 minutes; memory logged once a minute to checkpoints/campaign31_mac/mem.log.
S="$(cd "$(dirname "$0")" && pwd)"
cd /Users/paisho/Projects/Mastermind || exit 1
D=checkpoints/campaign31_mac; LOG=$D/train.log
PAT="src.v2.train.train --config configs/run_c31_mac.json"
alive() { pgrep -f "$PAT" >/dev/null; }
rows() { [ -f $D/metrics.csv ] && echo $(( $(wc -l < $D/metrics.csv) - 1 )) || echo 0; }
bash "$S/mac_train.sh" > "$S/mac_train_stdout.log" 2>&1 &
sleep 20
alive || { echo "trainer did not start"; cat "$S/mac_train_stdout.log"; tail -n 20 "$LOG"; exit 3; }
bash "$S/mac_mem.sh" &
MEM=$!
echo "$(date -u +%H:%M:%S) trainer up"
rc=0
for i in $(seq 1 2160); do
  sleep 10
  [ "$(rows)" -ge 1 ] && break
  alive || { echo "$(date -u +%H:%M:%S) trainer exited before finishing generation 1"; rc=4; break; }
  if [ $((i % 30)) -eq 0 ]; then
    echo "$(date -u +%H:%M:%S) $(grep -v "inference server" "$LOG" | tail -n 1 | cut -c10-150) | $(tail -n 1 $D/mem.log 2>/dev/null | sed 's/.*| //') | free $(vm_stat | awk '/Pages free/ {gsub("\\.","",$3); printf "%.1f", $3*16384/1073741824}')GB"
  fi
done
[ "$(rows)" -ge 1 ] || [ $rc -ne 0 ] || { echo "no generation in 6 hours"; rc=5; }
echo "$(date -u +%H:%M:%S) rows=$(rows); stopping the trainer"
pkill -INT -f "$PAT"; sleep 15; alive && { pkill -TERM -f "$PAT"; sleep 10; }
alive && pkill -KILL -f "$PAT"
kill $MEM 2>/dev/null
echo --- phases; grep -v "selfplay [0-9]*/\|inference server" "$LOG" | tail -n 40 | cut -c1-600
echo --- metrics; cat $D/metrics.csv 2>/dev/null | cut -c1-1500
exit $rc
