#!/bin/bash
cd /Users/paisho/Projects/Mastermind || exit 1
export PYTHONPATH=build PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p bench/regime_ab/arm_off bench/regime_ab/arm_det
echo "ab2 start $(date)" > bench/regime_ab/status.txt
./.venv/bin/python -m src.v2.train.train --config bench/regime_ab/ab_off.json > bench/regime_ab/arm_off.log 2>&1
echo "arm_off exit $? $(date)" >> bench/regime_ab/status.txt
./.venv/bin/python -m src.v2.train.train --config bench/regime_ab/ab_det.json > bench/regime_ab/arm_det.log 2>&1
echo "arm_det exit $? $(date)" >> bench/regime_ab/status.txt
echo "ab2 done $(date)" >> bench/regime_ab/status.txt
touch bench/regime_ab/AB2_DONE
