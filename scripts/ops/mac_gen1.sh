#!/bin/bash
# Run GENS generations (default 1) of MAC_CONFIG (default configs/run_c31_mac.json) on this Mac, then stop the trainer.
# MAC_RESUME=1 resumes from the newest checkpoint in MAC_DIR; GENS counts new metrics rows.
# Heartbeat every 5 minutes; memory logged once a minute to checkpoints/campaign31_mac/mem.log.
# Battery: below PAUSE_BELOW% (default 50) with generations still to go, the
# next generation holds (trainer PAUSE file) until RESUME_AT% (default 80);
# below 12% the trainer is stopped.
S="$(cd "$(dirname "$0")" && pwd)"
cd /Users/paisho/Projects/Mastermind || exit 1
export MAC_CONFIG=${MAC_CONFIG:-configs/run_c31_mac.json} MAC_DIR=${MAC_DIR:-checkpoints/campaign31_mac}
D=$MAC_DIR; LOG=$D/train.log
PAT="src.v2.train.train --config $MAC_CONFIG"
alive() { pgrep -f "$PAT" >/dev/null; }
batt() { pmset -g batt | grep -o '[0-9]*%' | head -n 1 | tr -d %; }
rm -f $D/PAUSE
rows0() { [ -f $D/metrics.csv ] && echo $(( $(wc -l < $D/metrics.csv) - 1 )) || echo 0; }
START=$(rows0)
rows() { echo $(( $(rows0) - START )); }
bash "$S/mac_train.sh" ${MAC_RESUME:+resume} > "$S/mac_train_stdout.log" 2>&1 &
sleep 20
alive || { echo "trainer did not start"; cat "$S/mac_train_stdout.log"; tail -n 20 "$LOG"; exit 3; }
bash "$S/mac_mem.sh" &
MEM=$!
echo "$(date -u +%H:%M:%S) trainer up"
rc=0
for i in $(seq 1 4320); do
  sleep 10
  [ "$(rows)" -ge "${GENS:-1}" ] && break
  b=$(batt)
  if [ "$b" -lt "${STOP_BELOW:-12}" ]; then echo "$(date -u +%H:%M:%S) battery ${b}% - stopping the trainer"; rc=6; break; fi
  if [ ! -f $D/PAUSE ] && [ "$(rows)" -lt $(( ${GENS:-1} - 1 )) ] && [ "$b" -lt "${PAUSE_BELOW:-50}" ]; then
    touch $D/PAUSE; echo "$(date -u +%H:%M:%S) battery ${b}% - next generation will wait for ${RESUME_AT:-80}%"
  fi
  if [ -f $D/PAUSE ] && [ "$b" -ge "${RESUME_AT:-80}" ]; then
    rm -f $D/PAUSE; echo "$(date -u +%H:%M:%S) battery ${b}% - resuming"
  fi
  alive || { echo "$(date -u +%H:%M:%S) trainer exited before finishing ${GENS:-1} generation(s)"; rc=4; break; }
  if [ $((i % 30)) -eq 0 ]; then
    echo "$(date -u +%H:%M:%S) $(grep -v "inference server" "$LOG" | tail -n 1 | cut -c10-150) | $(tail -n 1 $D/mem.log 2>/dev/null | sed 's/.*| //') | batt $(batt)% | free $(vm_stat | awk '/Pages free/ {gsub("\\.","",$3); printf "%.1f", $3*16384/1073741824}')GB"
  fi
done
[ "$(rows)" -ge "${GENS:-1}" ] || [ $rc -ne 0 ] || { echo "generations unfinished after 12 hours"; rc=5; }
echo "$(date -u +%H:%M:%S) rows=$(rows); stopping the trainer"
pkill -INT -f "$PAT"; sleep 15; alive && { pkill -TERM -f "$PAT"; sleep 10; }
alive && pkill -KILL -f "$PAT"
kill $MEM 2>/dev/null
rm -f $D/PAUSE
echo --- phases; grep -v "selfplay [0-9]*/\|inference server" "$LOG" | tail -n 40 | cut -c1-600
echo --- metrics; tail -n $(( ${GENS:-1} + 1 )) $D/metrics.csv 2>/dev/null | cut -c1-1500
exit $rc
