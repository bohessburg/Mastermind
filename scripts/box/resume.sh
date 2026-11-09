#!/usr/bin/env bash
# Supervisor entrypoint: reap orphaned workers, then resume the latest
# generation checkpoint, or start from INIT on first launch.
#   scripts/box/resume.sh <config.json> <init_weights.pt> [extra train args]
# Workers retitle to dominion-selfplay-N, so pattern-kills on src.v2.train
# miss them (c21 OOM incident) -- reap by spawn_main / title too.
set -uo pipefail
CONFIG=$1; INIT=$2; shift 2
cd /root/Mastermind
for pid in $(pgrep -f "dominion-selfplay|multiprocessing.spawn|src.v2.train.train" || true); do
  [ "$pid" != "$$" ] && kill -9 "$pid" 2>/dev/null
done
sleep 2
CKPT_DIR=$(.venv/bin/python -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint_dir'])" "$CONFIG")
LATEST=$(ls "$CKPT_DIR"/gen_*.pt 2>/dev/null | sort | tail -1)
export PYTHONPATH=build:. PYTHONUNBUFFERED=1
if [ -n "$LATEST" ]; then
  echo "resume.sh: resuming $LATEST"
  exec .venv/bin/python -m src.v2.train.train --config "$CONFIG" --resume "$LATEST" "$@"
fi
echo "resume.sh: fresh start from $INIT"
exec .venv/bin/python -m src.v2.train.train --config "$CONFIG" --init-weights "$INIT" "$@"
