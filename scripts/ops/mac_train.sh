#!/bin/bash
# Launch Mac training (default configs/run_c31_mac.json; MAC_CONFIG/MAC_DIR/MAC_INIT override) with a UTC-timestamped log.
# usage: mac_train.sh            (fresh start from c31 gen 27 weights; refuses if checkpoints exist)
#        mac_train.sh resume     (resume from the newest checkpoint in checkpoints/campaign31_mac)
set -o pipefail
cd /Users/paisho/Projects/Mastermind || exit 1
CFG=${MAC_CONFIG:-configs/run_c31_mac.json}
D=${MAC_DIR:-checkpoints/campaign31_mac}
LOG=$D/train.log
mkdir -p "$D"
if pgrep -f "src.v2.train.train --config $CFG" >/dev/null; then echo "trainer already running"; exit 3; fi
if [ "$1" = "resume" ]; then
  CK=$(ls "$D"/gen_*.pt 2>/dev/null | sort | tail -n 1)
  [ -z "$CK" ] && { echo "no checkpoint to resume"; exit 4; }
  ARGS="--resume $CK"
else
  ls "$D"/gen_*.pt >/dev/null 2>&1 && { echo "checkpoints exist; use resume"; exit 5; }
  ARGS="--init-weights ${MAC_INIT:-checkpoints/remote/campaign31/gen_0027.pt}"
fi
echo "$(date -u +%H:%M:%S) LAUNCH $ARGS" >> "$LOG"
PYTHONPATH=build:. PYTHONUNBUFFERED=1 caffeinate -s -i ./.venv/bin/python -m src.v2.train.train --config $CFG $ARGS 2>&1 \
  | while IFS= read -r line; do printf '%s %s\n' "$(date -u +%H:%M:%S)" "$line"; done >> "$LOG"
rc=${PIPESTATUS[0]}
echo "$(date -u +%H:%M:%S) EXIT $rc" >> "$LOG"
echo "trainer exit $rc"; tail -n 15 "$LOG"
exit $rc
