#!/usr/bin/env bash
# Box side: every STEP-th generation of a campaign plays a paired match against
# a fixed anchor checkpoint, so progress is measured against something that
# does not move when the gate promotes. One match at a time; exits when the
# campaign's STOPPED marker appears and nothing is left to play.
#   scripts/box/ladder_watch.sh <config.json> <anchor.pt> <out_dir> [step=5] [pairs=200] [sims=128]
set -uo pipefail
CONFIG=$1; ANCHOR=$2; OUT=$3; STEP=${4:-5}; PAIRS=${5:-200}; SIMS=${6:-128}
cd /root/Mastermind
export PYTHONPATH=build:. PYTHONUNBUFFERED=1
CKPT_DIR=$(.venv/bin/python -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint_dir'])" "$CONFIG")
SEED=$(.venv/bin/python -c "import json,sys;print(json.load(open(sys.argv[1]))['seed'])" "$CONFIG")
mkdir -p "$OUT"
echo "ladder_watch: $CKPT_DIR every $STEP generations vs $ANCHOR ($PAIRS pairs, $SIMS sims)"
while true; do
  played=0
  for ckpt in "$CKPT_DIR"/gen_*.pt; do
    [ -e "$ckpt" ] || continue
    gen=$((10#$(basename "$ckpt" .pt | sed 's/gen_//')))
    [ $((gen % STEP)) -eq 0 ] || continue
    tag=$(printf "gen%04d_vs_anchor" "$gen")
    [ -e "$OUT/$tag.json" ] && continue
    # The trainer writes checkpoints in place; let the file settle.
    sleep 20
    echo "ladder_watch: $(date -u +%H:%M:%S) playing $tag"
    .venv/bin/python scripts/c29_search_ladder.py --tag "$tag" --a "$ckpt" --b "$ANCHOR" \
      --a-sims "$SIMS" --b-sims "$SIMS" --pairs "$PAIRS" --determinize per_decision \
      --config "$CONFIG" --seed "$SEED" --device cuda --out "$OUT" \
      || { echo "ladder_watch: $tag failed"; echo '{"failed": true}' > "$OUT/$tag.json"; }
    played=1
  done
  if [ "$played" -eq 0 ] && [ -e "$CKPT_DIR/STOPPED" ]; then
    echo "ladder_watch: campaign stopped, nothing left to play"
    exit 0
  fi
  sleep 60
done
