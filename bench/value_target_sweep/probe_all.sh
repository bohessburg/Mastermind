#!/bin/bash
cd /Users/paisho/Projects/Mastermind || exit 1
export PYTHONPATH=build PYTHONUNBUFFERED=1
PY=./.venv/bin/python
OUT=bench/value_target_sweep
for a in 0.6 0.4 0.2 0.0; do
  for s in 1 2 3; do
    $PY scripts/probes/value_probe.py --checkpoint "$OUT/fit2_a${a}_s${s}.pt" --legacy-shim --out "$OUT/vp2s_a${a}_s${s}.json" > /dev/null 2>&1
    echo "vp a=$a s=$s exit $?"
    $PY scripts/probes/duchy_probe.py --checkpoint "$OUT/fit2_a${a}_s${s}.pt" --legacy-shim --out "$OUT/dp2s_a${a}_s${s}.json" > /dev/null 2>&1
    echo "dp a=$a s=$s exit $?"
  done
done
$PY - <<'EOF'
import json, statistics as st
from pathlib import Path
OUT = Path("bench/value_target_sweep")
rows = []
for a in ["0.6", "0.4", "0.2", "0.0"]:
    ms, es, js, gaps, dps, dvs = [], [], [], [], [], []
    for s in "123":
        vp = json.loads((OUT / f"vp2s_a{a}_s{s}.json").read_text())
        dp = json.loads((OUT / f"dp2s_a{a}_s{s}.json").read_text())
        m, e, j = vp["money_mean"], vp["engine_mean"], vp["junk_mean"]
        ms.append(m); es.append(e); js.append(j); gaps.append(m - e)
        dps.append(dp["duchy_prob_delta_points"]); dvs.append(dp["value_delta"])
    def fmt(x): return f"{st.mean(x):+.3f}±{st.pstdev(x):.3f}"
    rows.append((a, fmt(ms), fmt(es), fmt(js), fmt(gaps), fmt(dps), fmt(dvs)))
print(f"{'alpha':>5} {'money':>14} {'engine':>14} {'junk':>14} {'gap(m-e)':>14} {'duchyΔP':>14} {'duchyΔV':>14}")
for r in rows:
    print(f"{r[0]:>5} {r[1]:>14} {r[2]:>14} {r[3]:>14} {r[4]:>14} {r[5]:>14} {r[6]:>14}")
EOF
echo "PROBE_ALL_DONE"
