#!/bin/bash
cd /Users/paisho/Projects/Mastermind
echo "$(date -u +%T) waiting for control gen 12 self-play to start"
until ssh -o BatchMode=yes -o ConnectTimeout=20 -p 40973 root@85.165.238.33 'grep -q "\"generation\":12,\"phase\":\"selfplay\"" /root/Mastermind/checkpoints/campaign35_control/progress.json && test -f /root/Mastermind/checkpoints/campaign35_control/gen_0011.pt' 2>/dev/null; do sleep 10; done
echo "$(date -u +%T) gen 11 checkpointed; stopping trainer"
ssh -o BatchMode=yes -p 40973 root@85.165.238.33 'supervisorctl stop dominion-c35c' 2>&1 | grep -v "^Welcome\|^Have fun\|^AI agents"
scp -q -P 40973 src/v2/train/config.py src/v2/train/train.py src/v2/train/gating.py src/v2/train/test_gate_workers.py root@85.165.238.33:/root/Mastermind/src/v2/train/
scp -q -P 40973 configs/run_c35_control_long.json root@85.165.238.33:/root/Mastermind/configs/
ssh -o BatchMode=yes -p 40973 root@85.165.238.33 'cd /root/Mastermind; md5sum src/v2/train/gating.py src/v2/train/config.py | cut -c1-12' 2>&1 | grep -v "^Welcome\|^Have fun\|^AI agents"
md5 -q src/v2/train/gating.py src/v2/train/config.py | cut -c1-12
if ssh -o BatchMode=yes -p 40973 root@85.165.238.33 'cd /root/Mastermind; PYTHONPATH=build:. .venv/bin/python -m pytest -x -q src/v2/train/test_gate_workers.py src/v2/train/test_gating.py 2>&1 | tail -1 | tee /dev/stderr | grep -q passed' 2>&1; then
  ssh -o BatchMode=yes -p 40973 root@85.165.238.33 'supervisorctl start dominion-c35c' 2>&1 | grep -v "^Welcome\|^Have fun\|^AI agents"
  echo "$(date -u +%T) DEPLOYED and restarted"
else
  echo "$(date -u +%T) TESTS FAILED ON BOX; trainer left stopped"
fi
