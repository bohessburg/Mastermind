#!/bin/bash
# Emits one line per phase transition / gate / failure for c35; state persists across re-arms.
cd /Users/paisho/Projects/Mastermind
ST=/private/tmp/claude-501/-Users-paisho-Projects-Mastermind/6ca4328b-95ee-4551-ad0e-626e4a1135cd/scratchpad/c35c_mon_state
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -p 40973 root@85.165.238.33"
fails=0
while true; do
  out=$($SSH 'cd /root/Mastermind; for a in control; do f=checkpoints/campaign35_$a/progress.json; [ -f $f ] && echo "P $a $(cat $f)"; done; grep -E "\"gate_result\"|exited rc=|C35C DONE|Traceback|Error|Killed" supervisor-c35c.log | sed "s/^/L /"; supervisorctl status dominion-c35c | sed "s/^/S /"' 2>/dev/null)
  if [ -z "$out" ]; then fails=$((fails+1)); [ $fails -eq 5 ] && echo "c35: box unreachable for 5 min"; sleep 60; continue; fi
  fails=0
  printf "%s\n" "$out" | python3 /private/tmp/claude-501/-Users-paisho-Projects-Mastermind/6ca4328b-95ee-4551-ad0e-626e4a1135cd/scratchpad/c35c_monitor.py "$ST"
  if [ $? -eq 3 ]; then
    for arm in control; do
      mkdir -p checkpoints/remote/campaign35_$arm
      rsync -a -e "ssh -o BatchMode=yes -p 40973" root@85.165.238.33:/root/Mastermind/checkpoints/campaign35_$arm/ checkpoints/remote/campaign35_$arm/ >/dev/null 2>&1
    done
  fi
  printf "%s\n" "$out" | grep -q "C35C DONE" && exit 0
  sleep 60
done
