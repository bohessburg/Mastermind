#!/bin/bash
cd /Users/paisho/Projects/Mastermind
B="ssh -o BatchMode=yes -o ConnectTimeout=20 -p 40973 root@85.165.238.33"
echo "$(date -u +%T) waiting for resample gen_0001.pt"
until $B 'grep -q "\"generation\":2,\"phase\":\"selfplay\"" /root/Mastermind/checkpoints/campaign35_resample/progress.json && test -f /root/Mastermind/checkpoints/campaign35_resample/gen_0001.pt' 2>/dev/null; do sleep 10; done
echo "$(date -u +%T) gen 1 checkpointed, gen 2 self-play started; stopping trainer"
$B 'supervisorctl stop dominion-c35' 2>&1 | grep -v "^Welcome\|^Have fun\|^AI agents"
scp -q -P 40973 src/v2/train/config.py src/v2/train/train.py src/v2/train/gating.py src/v2/train/diagnostics.py src/v2/train/test_gate_workers.py root@85.165.238.33:/root/Mastermind/src/v2/train/
scp -q -P 40973 configs/run_c35_resample.json root@85.165.238.33:/root/Mastermind/configs/
$B 'cd /root/Mastermind; md5sum src/v2/train/gating.py src/v2/train/diagnostics.py | cut -c1-12; PYTHONPATH=build:. .venv/bin/python -m pytest -x -q src/v2/train/test_gate_workers.py src/v2/train/test_gating.py src/v2/train/test_diagnostics.py 2>&1 | tail -2' 2>&1 | grep -v "^Welcome\|^Have fun\|^AI agents"
md5 -q src/v2/train/gating.py src/v2/train/diagnostics.py | cut -c1-12
if $B 'cd /root/Mastermind; PYTHONPATH=build:. .venv/bin/python -m pytest -x -q src/v2/train/test_gate_workers.py 2>&1 | tail -1 | grep -q passed' 2>/dev/null; then
  $B 'supervisorctl start dominion-c35' 2>&1 | grep -v "^Welcome\|^Have fun\|^AI agents"
  echo "$(date -u +%T) DEPLOYED and restarted"
else
  echo "$(date -u +%T) TESTS FAILED ON BOX; trainer left stopped"
fi
