#!/bin/bash
# Log the Mac trainer's memory once a minute: per-process footprint (dirty incl. compressed), system compressor.
cd /Users/paisho/Projects/Mastermind || exit 1
CFG=${MAC_CONFIG:-configs/run_c31_mac.json}
OUT=${MAC_DIR:-checkpoints/campaign31_mac}/mem.log
while true; do
  T=$(pgrep -f "python -m src.v2.train.train --config $CFG" | head -n 1)
  [ -z "$T" ] && { sleep 20; T=$(pgrep -f "python -m src.v2.train.train --config $CFG" | head -n 1); [ -z "$T" ] && { echo "$(date -u +%H:%M:%S) trainer gone" >> $OUT; exit 0; }; }
  line="$(date -u +%H:%M:%S)"
  for p in $T $(pgrep -P $T); do
    f=$(footprint $p 2>/dev/null | grep "Footprint:" | sed 's/.*Footprint: \([0-9.]* [A-Z]*\).*/\1/' | tr -d ' ')
    [ -n "$f" ] && line="$line $f"
  done
  comp=$(vm_stat | awk '/occupied by compressor/ {gsub("\\.","",$5); printf "%.1f", $5*16384/1073741824}')
  echo "$line | compressor ${comp}GB" >> $OUT
  sleep 60
done
