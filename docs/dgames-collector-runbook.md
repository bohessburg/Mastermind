# dominion.games Collector Runbook

How to run the scraper yourself. Written 2026-08-26 after the first
1.5M-position campaign.

Everything below assumes you are in `/Users/paisho/Projects/Mastermind`
and that `./.venv` is the project venv.

## TL;DR

```sh
# start (5 accounts, base-set filter on)
./.venv/bin/python scripts/dgames_fleet.py \
  --accounts-file data/dominion_games/fleet.accounts.json \
  "--collector-arg=--target-min-age-seconds=420" \
  "--collector-arg=--target-max-age-seconds=840"

# stop  -- Ctrl-C, or from another shell:
pkill -INT -f 'python.*dgames_fleet.py'

# convert captures -> training tuples (minutes at 13K games)
./.venv/bin/python scripts/dgames_convert.py

# refresh skill ratings
./.venv/bin/python scripts/dgames_ratings.py --once
./.venv/bin/python scripts/dgames_ratings_join.py
```

**Always stop with SIGINT, never `kill -9` / `pkill` without `-INT`.**
An abrupt kill can invalidate a browser profile's stored session and
forces a manual re-login.

## One-time setup

1. **Accounts.** Create dominion.games accounts (the collector used 5).
   Put credentials in `.env` at the repo root, indexed by id:
   ```
   DGAMES_USER_1=...
   DGAMES_PASS_1=...
   ...through _5
   ```
   `.env` is gitignored and chmod 600. Keep it that way. The unsuffixed
   `DGAMES_USER` / `DGAMES_PASS` pair is a separate account used by the
   ratings poller — see "Ratings" below.

2. **Accounts file** at `data/dominion_games/fleet.accounts.json`
   (gitignored, deliberately credential-free):
   ```json
   {"accounts": [{"id": "1"}, {"id": "2"}, {"id": "3"}, {"id": "4"}, {"id": "5"}]}
   ```
   Browser profiles are auto-created at `dgames-profiles/account-<id>`.

3. **Playwright/Chromium** must be installed:
   ```sh
   pip install -r src/v2/arena/requirements.txt
   ./.venv/bin/python -m playwright install chromium
   ```

That's it. The collector logs in with credentials on first run and
self-heals thereafter — you should not need `scripts/dgames_login.py`
unless self-heal is unavailable.

## Running

Start it (see TL;DR). It runs until you stop it, or pass
`--run-seconds N` for a bounded run.

Redirect output somewhere you can tail:

```sh
nohup ./.venv/bin/python scripts/dgames_fleet.py \
  --accounts-file data/dominion_games/fleet.accounts.json \
  "--collector-arg=--target-min-age-seconds=420" \
  "--collector-arg=--target-max-age-seconds=840" \
  > /tmp/fleet.log 2>&1 &
```

**Note the `--collector-arg=--flag=value` form.** Argparse rejects
`--collector-arg --flag` because the value starts with `--`; the
equals form is required.

### Checking health

```sh
# fleet-level
grep 'fleet heartbeat' /tmp/fleet.log | tail -1

# real throughput from the manifests (more honest than the fleet's
# lifetime average, which lags reality)
./.venv/bin/python - <<'PY'
import json, glob
from datetime import datetime, timezone, timedelta
now = datetime.now(timezone.utc); ts = []
for m in glob.glob('data/dominion_games/raw/*.manifest.json'):
    d = json.load(open(m))
    if d['capture_status'] != 'complete':
        continue
    ts.append(datetime.fromisoformat(d['completed_at_utc'].replace('Z', '+00:00')))
print('complete: %d   last 1h: %d/hr' % (
    len(ts), sum(1 for t in ts if t > now - timedelta(hours=1))))
PY
```

A healthy heartbeat looks like:
```
games/hour=60  completed=500  claim-contention=8%  non-base-skip=90%
accounts=1:running,2:running,3:running,4:running,5:running
```

| field | healthy | meaning |
|---|---|---|
| `games/hour` | 40-80 | swings with time of day; 25-30 at 4am is normal |
| `claim-contention` | 5-12% | accounts reaching for the same table; rises with account count |
| `non-base-skip` | ~90% | only ~10-18% of live games are base-set — expected, not a fault |
| `accounts` | all `running` | `stale`/`down` for one account is usually self-healing |

## Stopping

```sh
pkill -INT -f 'python.*dgames_fleet.py'
```

Give it up to ~90s: SIGINT stops each account at a safe boundary, marks
any in-flight capture incomplete, and writes a summary. Verify:

```sh
pgrep -fc 'ms-playwright'   # want 0
```

## Converting to training tuples

```sh
./.venv/bin/python scripts/dgames_convert.py
```

Rebuilds `exports/tuples_dgames/` **from raw every run** — it is not
incremental, so it takes several minutes at 13K games. Run it in the
background and do not run two at once (they write the same directory).

Healthy output:
```
emitted 1581527 tuple(s) from 12732 game(s); quarantined 951;
final-deck match rate 100.0%; divergence 0.0%
```

**The gate that matters is `final-deck match rate 100.0% / divergence
0.0%`.** Every emitted game's derived deck composition must exactly
equal the final deck histogram the game itself reported. If divergence
is ever non-zero, stop and investigate — do not train on it.

Expect **~93-97% of complete captures to convert**. The quarantined
remainder is mostly genuinely incomplete captures. Raw frames are
always preserved, so quarantined games are recoverable if the converter
improves — the corpus has been reprocessed from raw several times to
recover exactly this way.

## Ratings (skill weighting)

The corpus is all-comers; ratings let you weight by skill.

```sh
./.venv/bin/python scripts/dgames_ratings.py --once       # poll leaderboard
./.venv/bin/python scripts/dgames_ratings_join.py         # join to games
```

- The poller needs its **own account** (the unsuffixed `DGAMES_USER` /
  `DGAMES_PASS`). One session per account, so it will evict a collector
  if it shares one.
- Observations are **append-only** at
  `data/dominion_games/ratings/observations.jsonl` — a time series, so
  games can be matched to ratings near when they were played.
- The join writes `data/dominion_games/ratings/game_ratings.json`, a
  sidecar **decoupled from the npz files**. Re-tune the weighting curve
  without reconverting anything.
- Coverage runs ~85% of games with both players rated.

Join it to tuples via each row's `player_id` plus the manifest's
`game_index_to_game_id` map.

## Tuning

| flag | script default | campaign used | notes |
|---|---|---|---|
| `--target-min-age-seconds` | 240 (4m) | **420 (7m)** | join band lower bound |
| `--target-max-age-seconds` | 720 (12m) | **840 (14m)** | join band upper bound |
| `--no-base-filter` | off | off | capture ALL games, not just base-set |
| `--fleet-join-rate-per-minute` | 30.0 | 30.0 | fleet-wide politeness cap |

**The script defaults are NOT the tuned values.** Bare defaults give you
the 4-12m band, which measured worse. The 7-14m band in the TL;DR is an
explicit override — pass it, or you silently get the slower setting.

**Join band.** A game cannot be joined after it ends, so the band's
lower bound structurally excludes shorter games. Measured game lengths:
rated 2P median 9.2 min, blitz 7.1, unrated 11.7. The 7-14m band
excludes ~20% of games (~90% of blitz) but nearly doubled throughput
versus 4-12m, with zero join timeouts. Jack accepted that bias
deliberately — see `docs/research-agenda.md`.

**`--no-base-filter`** raises the keep rate ~5.5x by capturing every
expansion. Raw frames are kept regardless, so those games become usable
whenever the engine supports their cards. Use it once base-set coverage
is sufficient.

**Account count.** ~2.2 GB RAM per account (Chromium is 98% of the
footprint). 5 accounts ≈ 11 GB. Supply-side ceiling is ~90-130 base
completions/hour site-wide; at 5 accounts you are already taking
~70% of it, so more accounts yield less than linearly.

## Troubleshooting

**All accounts fail with `ERR_NAME_NOT_RESOLVED` / `ERR_INTERNET_DISCONNECTED`**
Network dropped. On macOS a poisoned DNS cache looks like this — `dig`
resolves but `curl` does not. Fix:
```sh
sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder
```
Then restart the fleet.

**Page loads time out but the site is fast**
System load. Chromium starves above ~load 30 and page loads exceed the
30s timeout. Check `sysctl -n vm.loadavg`. Stop the fleet or the
competing workload; it recovers on its own once load drops.

**One account goes `down` repeatedly, others fine**
Usually a transient DNS blip tripping that account's 5-consecutive-
failure limit. The supervisor restarts it. No action needed.

**`SESSION EXPIRED: run scripts/dgames_login.py`**
Credentialed self-heal is unavailable (missing `.env` entry). Either
fix the credentials or log in by hand:
```sh
./.venv/bin/python scripts/dgames_login.py
```

**Throughput collapses to a few games/hour**
Check load first, then `non-base-skip` — if it is >93%, the lobby
simply has fewer base games right now. Supply varies a lot by hour.

## What's on disk

| path | contents | gitignored |
|---|---|---|
| `data/dominion_games/raw/` | `<game_id>.jsonl.gz` + `.manifest.json` per game | yes |
| `exports/tuples_dgames/` | npz shards + `tuple_manifest.json` | yes |
| `data/dominion_games/ratings/` | observations + join sidecar | yes |
| `data/dominion_games/coord/` | fleet claims, lobby cache, rate limit | yes |
| `dgames-profiles/` | per-account Chromium profiles | yes |
| `.env` | credentials | yes |

**Raw frames are ground truth and can never be re-fetched.** Never
delete `data/dominion_games/raw/`. ~1 MB per game (13 GB at 13K games).
Every converter improvement so far has been applied retroactively by
reprocessing raw.

## Reference

- Protocol: `data/dominion_games/recon/RECON.md`
- Corpus sizing, skill weighting, join-band analysis:
  `docs/research-agenda.md`
- Original mission brief: `docs/dominion-games-scraper-handoff.md`
