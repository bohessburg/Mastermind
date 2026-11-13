# c24 Ops Cheatsheet (box 42049)

Copy-paste commands for checking on c24 yourself. All remote commands go
through this SSH base (add `-L 8080:localhost:8080` if you want the tunnel):

```sh
BOX="ssh -p 42049 root@76.66.207.254"
```

Run that line first in your shell, then paste any block below as-is.

## 1. Quick status (is it alive, what gen, how far along)

```sh
$BOX 'supervisorctl status dominion-c24; echo; cat /root/Mastermind/checkpoints/campaign24/progress.json'
```

- `RUNNING` + a recent `updated_unix` = healthy.
- `generation` / `games_done` / `recent_games_per_hour` tell you pace
  (~1000 games/hr, ~1.2 h per gen is normal).

## 2. GPU / load sanity

```sh
$BOX 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; echo; uptime'
```

GPU 0 = training/inference server. GPUs 1–3 are free for duels/evals/vibes.

## 3. Training metrics + sentinel results (READ BY HEADER, never by column position)

```sh
$BOX 'cd /root/Mastermind && ./.venv/bin/python - <<EOF
import csv
rows = list(csv.DictReader(open("checkpoints/campaign24/metrics.csv")))
for r in rows[-6:]:
    print(f"gen {r[\"generation\"]:>3}  entropy {float(r[\"entropy\"]):.2f}  "
          f"pos/game {float(r[\"positions_per_game\"] or 0):.0f}  "
          f"sentinel {r.get(\"eval_opponent\",\"\") or \"-\"} "
          f"{r.get(\"eval_wins\",\"\") or \"\"}/{r.get(\"eval_games\",\"\") or \"\"}"
EOF'
```

If a column name errors, list the real header first:

```sh
$BOX 'head -1 /root/Mastermind/checkpoints/campaign24/metrics.csv | tr "," "\n" | nl'
```

The sentinel (100 games vs engine3, 50 per seat) runs automatically every
gen — you never need to start it. It shows up in the metrics row when the
gen finishes. Milestone bars: beat c23 pointwise (28.3 / 31.6 / 26.7 / 24.1),
no two consecutive declines, 35.6 by g15–20.

## 4. Supervisor log tail (if something looks stuck)

```sh
$BOX 'tail -30 /root/Mastermind/supervisor-c24.log'
```

Restart (only if genuinely wedged — wait for a metrics row after a gen
boundary before judging):

```sh
$BOX 'supervisorctl restart dominion-c24'
```

## 5. Start a champ duel on a free GPU (200 games vs c15 gen_0045)

As of 2026-11 every duel uses the CANONICAL eval boards: omit `--seed`
(default 20261108) and keep the default `--seating paired`, so each kingdom is
played twice with seats swapped and every generation/campaign is measured on
the same 100 boards. The summary prints a Wilson 95% CI, Elo, and the paired
CI. For go/no-go decisions prefer `python -m src.v2.train.sprt` (stops early
once decisive), and rate everything jointly with
`python -m src.v2.train.ladder bench/`.

```sh
$BOX 'cd /root/Mastermind && CUDA_VISIBLE_DEVICES=1 nice -n 15 env PYTHONPATH=build PYTHONUNBUFFERED=1 nohup ./.venv/bin/python -m src.v2.train.honest_eval \
  --a checkpoints/campaign24/gen_0015.pt \
  --b checkpoints/remote/campaign15/gen_0045.pt \
  --games 200 --sims 400 --determinizations 2 --sims-b 400 --determinizations-b 2 \
  --workers 24 --device cuda --legacy-shim \
  --out bench/c24_g15_vs_champ.json > duel_c24_g15.log 2>&1 & echo STARTED'
```

## 6. Check a running duel / read its result

```sh
$BOX 'tail -3 /root/Mastermind/duel_c24_g15.log'
```

`progress: 90/200 30-57-3 ...` = wins-losses-draws for the `--a` side (c24).
When done:

```sh
$BOX 'cat /root/Mastermind/bench/c24_g15_vs_champ.json | ./.venv/bin/python -m json.tool | head -20' 2>/dev/null || echo "not finished yet"
```

## 7. Vibe check (one full game per board, on a free GPU)

```sh
$BOX 'cd /root/Mastermind && CUDA_VISIBLE_DEVICES=3 env PYTHONPATH=build PYTHONUNBUFFERED=1 ./.venv/bin/python vibe_c24.py sentry checkpoints/campaign24/gen_0015.pt'
$BOX 'cd /root/Mastermind && CUDA_VISIBLE_DEVICES=3 env PYTHONPATH=build PYTHONUNBUFFERED=1 ./.venv/bin/python vibe_c24.py chapel checkpoints/campaign24/gen_0015.pt'
```

Prints turn-by-turn plays/buys/trashes and the final score. Note: the chapel
board pool is a reconstructed baseline (original was lost) — compare chapel
vibes only against other c24-era chapel vibes.

## 8. Download latest weights to your machine

Run from your laptop (not inside $BOX). Latest gen:

```sh
LATEST=$(ssh -p 42049 root@76.66.207.254 'ls /root/Mastermind/checkpoints/campaign24/gen_*.pt | sort | tail -1')
echo "$LATEST"
scp -P 42049 root@76.66.207.254:"$LATEST" /Users/paisho/Projects/Mastermind/checkpoints/campaign24/
```

Also grab metrics for local plotting:

```sh
scp -P 42049 root@76.66.207.254:/root/Mastermind/checkpoints/campaign24/metrics.csv /Users/paisho/Projects/Mastermind/checkpoints/campaign24/
```

## 9. Probe suite (runs locally, needs the downloaded checkpoint)

```sh
cd /Users/paisho/Projects/Mastermind
PYTHONPATH=build ./.venv/bin/python scripts/probes/run_all.py --checkpoint checkpoints/campaign24/gen_0015.pt
```

Reads: value probe (want engine > money > junk), chapel attraction, militia
inversion (known-broken this campaign — expect `inversion_present`).

## 10. Kill the campaign (only on an explicit cut decision)

```sh
$BOX 'supervisorctl stop dominion-c24'
```

The supervisor conf has stopasgroup/killasgroup, so this takes the worker
fleet down with it — no orphan reaping needed.
