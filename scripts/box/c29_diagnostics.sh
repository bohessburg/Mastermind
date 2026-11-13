#!/usr/bin/env bash
# c29 drawing-board diagnostics: search ladder, corrected gates, decision
# regret, leaf parity. Runs the matchups JOBS at a time on one GPU and prints a status
# line per job every minute. Finished tags are skipped, so it is safe to
# re-run after an interruption.
#
#   scripts/box/c29_diagnostics.sh [ladder|gates|regret|measure|parity|all]
#
# Environment:
#   JOBS   parallel processes            (default 6; use 2-3 on a laptop)
#   PAIRS  boards per match, x2 games    (default 200)
#   CHUNK  concurrent games per process  (default 100)
#   BC     baseline checkpoint           (default checkpoints/campaign25/bc_a.pt)
#   CANDS  candidate checkpoints for the corrected gates
#          (default checkpoints/campaign28/gen_000[1-6].pt)
#   PY     python interpreter            (default .venv/bin/python)
#   LADDER_OUT, REGRET_OUT  result directories (default bench/c29_ladder,
#          bench/c29_regret). Point them elsewhere to re-run tags whose
#          results already exist, e.g. after a search change.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=build:. PYTHONUNBUFFERED=1
WHAT=${1:-all}
JOBS=${JOBS:-6}; PAIRS=${PAIRS:-200}; CHUNK=${CHUNK:-100}
BC=${BC:-checkpoints/campaign25/bc_a.pt}
CANDS=${CANDS:-$(ls checkpoints/campaign28/gen_000[1-6].pt 2>/dev/null | tr '\n' ' ')}
PY=${PY:-.venv/bin/python}
LADDER_OUT=${LADDER_OUT:-bench/c29_ladder}; REGRET_OUT=${REGRET_OUT:-bench/c29_regret}
mkdir -p "$LADDER_OUT/logs" "$REGRET_OUT/logs"
# One small script per job: BSD xargs cannot substitute long command lines.
QUEUE=$(mktemp -d)
enqueue() { printf '%s\n' "$1" > "$QUEUE/$(printf '%03d' "$(ls "$QUEUE" | wc -l)").sh"; }

ladder() { # tag, then c29_search_ladder.py arguments
  local tag=$1; shift
  [ -f "$LADDER_OUT/$tag.json" ] && return
  enqueue "$PY scripts/c29_search_ladder.py --tag $tag --pairs $PAIRS --chunk $CHUNK --out $LADDER_OUT $* > $LADDER_OUT/logs/$tag.log 2>&1"
}
parity() { # tag, then c29_leaf_parity.py arguments
  local tag=$1; shift
  [ -f "$LADDER_OUT/leaf_parity_$tag.json" ] && return
  enqueue "$PY scripts/c29_leaf_parity.py --tag $tag --ckpt $BC --out $LADDER_OUT $* > $LADDER_OUT/logs/leaf_parity_$tag.log 2>&1"
}
regret() { # tag, then c29_decision_regret.py arguments
  local tag=$1; shift
  [ -f "$REGRET_OUT/$tag.json" ] && return
  enqueue "$PY scripts/c29_decision_regret.py --tag $tag --out $REGRET_OUT $* > $REGRET_OUT/logs/$tag.log 2>&1"
}

if [ "$WHAT" = gates ] || [ "$WHAT" = all ]; then
  # Corrected gate for every candidate: clairvoyant (the training regime)
  # first for all of them, then honest search.
  for ckpt in $CANDS; do
    name=$(basename "$(dirname "$ckpt")")_$(basename "$ckpt" .pt)
    ladder "gate_${name}"        --a "$ckpt" --b "$BC"
  done
  for ckpt in $CANDS; do
    name=$(basename "$(dirname "$ckpt")")_$(basename "$ckpt" .pt)
    ladder "gate_${name}_honest" --a "$ckpt" --b "$BC" --determinize per_decision
  done
fi
if [ "$WHAT" = ladder ] || [ "$WHAT" = all ]; then
  # Does search add strength over the raw policy, and does more search add more?
  for mode in off per_decision; do
    sfx=$([ "$mode" = off ] && echo "" || echo "_honest")
    ladder "bc_128_vs_raw$sfx"   --a "$BC" --a-sims 128 --b-sims raw --determinize $mode
    ladder "bc_512_vs_raw$sfx"   --a "$BC" --a-sims 512 --b-sims raw --determinize $mode
    ladder "bc_512_vs_128$sfx"   --a "$BC" --a-sims 512 --b-sims 128 --determinize $mode
    # Does the value head help that search?
    ladder "bc_128_vs_zerov$sfx" --a "$BC" --a-sims 128 --b-sims 128 --b-zero-value --determinize $mode
    ladder "bc_512_vs_zerov$sfx" --a "$BC" --a-sims 512 --b-sims 512 --b-zero-value --determinize $mode
  done
  # Is the honest searcher weaker than the clairvoyant one at equal budget?
  # (Cannot be asked in one match: the information model is per game.)
fi
if [ "$WHAT" = measure ] || [ "$WHAT" = all ]; then
  # Measure the measurement: the old gate conditions, one at a time, on the
  # same boards as the corrected gate of the first candidate.
  first=$(echo $CANDS | awk '{print $1}')
  if [ -n "$first" ]; then
    name=$(basename "$(dirname "$first")")_$(basename "$first" .pt)
    ladder "meas_${name}_legacy"     --a "$first" --b "$BC" --legacy-gate
    ladder "meas_${name}_crossroute" --a "$first" --b "$BC" --cross-route
    ladder "meas_${name}_reuse"      --a "$first" --b "$BC" --tree-reuse
    ladder "meas_${name}_openingT"   --a "$first" --b "$BC" --opening-temp
    ladder "meas_bc_self"            --a "$BC" --b "$BC"
  fi
fi
if [ "$WHAT" = parity ] || [ "$WHAT" = all ]; then
  # Do several leaves in flight search like one? Root visits on the same
  # positions against sequential search, at each budget self-play uses:
  # the gate (128), normal games (512) and the deep slice (4096). The second
  # leaf count of each pair is what tail mode would use if it were uncapped.
  parity bca_128  --sims 128  --leaves 4 8   --games 32 --max-turns 30
  parity bca_512  --sims 512  --leaves 8 16  --games 32 --max-turns 24
  parity bca_4096 --sims 4096 --leaves 32 64 --games 16 --max-turns 20
fi
if [ "$WHAT" = regret ] || [ "$WHAT" = all ]; then
  # Is the search's move better than the prior's move, by honest playouts?
  for sims in 128 512; do
    regret "bc_s${sims}"        --model "$BC" --sims $sims --games 128 --positions 600 --overrides-only
    regret "bc_s${sims}_honest" --model "$BC" --sims $sims --games 128 --positions 600 --overrides-only --determinize per_decision
  done
  regret "bc_s512_all" --model "$BC" --sims 512 --games 128 --positions 600
fi

total=$(ls "$QUEUE" | wc -l | tr -d ' ')
echo "c29_diagnostics: $total jobs queued ($WHAT), $JOBS at a time, pairs=$PAIRS chunk=$CHUNK"
[ "$total" = 0 ] && { rm -rf "$QUEUE"; exit 0; }
ls "$QUEUE"/*.sh | xargs -P "$JOBS" -n 1 bash &
runner=$!
while kill -0 "$runner" 2>/dev/null; do
  sleep 60
  echo "== $(date +%H:%M) done $(ls $LADDER_OUT/*.json $REGRET_OUT/*.json 2>/dev/null | wc -l | tr -d ' ') json"
  # Last line of every log still being written (finished jobs end in DONE).
  for log in "$LADDER_OUT"/logs/*.log "$REGRET_OUT"/logs/*.log; do
    [ -f "$log" ] || continue
    [ "$(basename "$log")" = _gates_run.log ] && continue
    grep -q "DONE" "$log" || tail -n 1 "$log"
  done
done
wait "$runner"
rm -rf "$QUEUE"
echo "c29_diagnostics: finished"
grep -h "DONE" "$LADDER_OUT"/logs/*.log "$REGRET_OUT"/logs/*.log 2>/dev/null
