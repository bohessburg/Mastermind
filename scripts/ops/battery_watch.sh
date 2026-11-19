#!/bin/bash
# Exit (and so notify) when the battery falls below 20%. Heartbeat every 30 minutes.
i=0
while true; do
  line=$(pmset -g batt | tail -n 1)
  pct=$(echo "$line" | grep -o '[0-9]*%' | head -n 1 | tr -d '%')
  src=$(pmset -g batt | head -n 1 | sed "s/.*'\(.*\)'.*/\1/")
  if [ -n "$pct" ] && [ "$pct" -lt 20 ]; then echo "$(date -u +%H:%M:%S) BATTERY LOW ${pct}% on $src"; exit 0; fi
  [ $((i % 30)) -eq 0 ] && echo "$(date -u +%H:%M:%S) battery ${pct}% on $src"
  i=$((i+1)); sleep 60
done
