#!/usr/bin/env bash
# Local side: ship committed HEAD (git bundle) + corpus + inits to a box.
#   scripts/box/ship.sh <port> <host> [extra repo-relative paths...]
# Uses the hetzner_dgames key; retries because vast uplinks drop.
set -euo pipefail
PORT=$1; HOST=$2; shift 2
cd "$(git rev-parse --show-toplevel)"
BUNDLE=$(mktemp -d)/repo.bundle
git bundle create "$BUNDLE" "$(git branch --show-current)" >/dev/null 2>&1
K="ssh -i $HOME/.ssh/hetzner_dgames -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -p $PORT"
PATHS=(exports/tuples_all_40plus data/dominion_games/ratings/game_ratings.json
       checkpoints/remote/campaign15/gen_0045.pt "$@")
for i in $(seq 1 20); do
  $K root@"$HOST" "mkdir -p /root/ship" && \
  rsync -a --partial -e "$K" "$BUNDLE" scripts/box/bootstrap.sh root@"$HOST":/root/ship/ && \
  rsync -aR --partial -e "$K" "${PATHS[@]}" root@"$HOST":/root/ship/ && { echo "SHIPPED (try $i)"; exit 0; }
  echo "retry $i"; sleep 10
done
exit 1
