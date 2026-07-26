# Arena: running the dominion.games driver

How to run the trained checkpoint against people on dominion.games, what the
process does on its own, and how to read what it leaves behind.

Architecture is in `docs/arena-plan.md`; build history and per-incident notes
are in `docs/arena-progress.md`. This file is the operator's guide.

## What it is

A headful Chromium session (Playwright) runs the normal dominion.games web
client, logged in as the bot account. The driver:

- reads the client's own WebSocket feed to follow the game exactly,
- decides with `DecisionSearcher` + the trained checkpoint,
- answers questions by sending the client's own `ANSWER_QUESTION` protocol
  message on that same socket,
- clicks the UI only for lobby buttons and modals.

The account is openly identified as a bot and the experiment is run with
permission. The driver never sends chat and never replies to chat.

## Prerequisites

1. Engine bindings built (Release, with Python):

   ```sh
   PYBIND11_DIR="$(./.venv/bin/python -m pybind11 --cmakedir)"
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -Dpybind11_DIR="$PYBIND11_DIR"
   cmake --build build
   ```

2. A checkpoint at the path in `configs/arena.json` (default
   `checkpoints/remote/campaign15/gen_0045.pt`).

3. **A logged-in browser profile.** Credentials are never read from the repo.
   The persistent profile lives in `arena-profile/` (gitignored). If it is
   missing or logged out, run any session once and log in by hand in the
   window that opens; the profile keeps the session afterwards.

4. Base-set-only games. The card pool is set through the site's own game
   filter on the account; the driver additionally refuses to play a kingdom
   containing cards the engine does not implement (it resigns and requeues).

## Running

Preflight — always green before a live run:

```sh
OMP_NUM_THREADS=1 PYTHONPATH=build ./.venv/bin/python -m pytest tests/v2/arena/
PYTHONPATH=build ./.venv/bin/python -m src.v2.arena.main --config configs/arena.json --dry-run
```

The dry run replays the reference recording through the whole live pipeline
without opening a browser. If it fails, do not start a live session.

Supervised session (a person watching):

```sh
PYTHONPATH=build ./.venv/bin/python -m src.v2.arena.main --config configs/arena.json
```

Unattended session (keeps the Mac awake, unlimited games):

```sh
./scripts/arena_overnight.sh
```

Both open a visible browser window. Watch it if you like; do not click in it
while the driver is playing.

Stop with Ctrl-C. The driver closes the browser cleanly so the profile is not
left locked.

## What it does by itself

Once started it needs no clicks:

1. Clicks **Start search** for automatch.
2. On a match, automatch drops straight into the game screen and the client
   asks a start-confirmation question in the feed (what the UI shows as a
   "Start game" prompt) — the driver answers it.
3. Plays the game: treasures are auto-played before buys, attacks and
   reactions handled, mid-card decisions (Library, Sentry, Throne Room…)
   stepped through a shadow engine state kept in sync with the server.
4. Denies every opponent undo request (granting would rewind state the
   tracker cannot follow).
5. Claims the win if the opponent's clock expires (after a 30 s grace).
6. On game end: dismisses "The game has ended", clicks **Leave Table**,
   searches again.

## Configuration (`configs/arena.json`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `checkpoint` | campaign15 gen_0045 | model to serve |
| `sims` | 400 | MCTS simulations per decision (sweep is flat 200–1000; do not raise) |
| `determinizations` | 2 | hidden-information re-deals per decision |
| `think_time_min/max_seconds` | small | pacing so play is polite, not instant |
| `actuation_mode` | `protocol` | how answers are sent; `clicks` is the legacy DOM fallback |
| `lobby.max_games_per_session` | `0` | 0 = unlimited until Ctrl-C |
| `lobby.*_timeout_seconds` | various | per-state lobby bounds |
| `stall_watchdog_seconds` | 120 | abort if we owe a move and nothing happens |
| `undo.auto_deny` | `true` | only supported policy |
| `timeout.claim_grace_seconds` | 30 | wait before claiming an expired clock |

## Exit codes

The process exits with a machine-readable code and writes `exit.json` into
the run directory. An outer supervisor restarts on any non-zero code.

| Code | Meaning | Usual response |
| ---: | --- | --- |
| 0 | Clean stop (game cap or Ctrl-C) | nothing |
| 3 | Stall watchdog / idle watchdog | restart |
| 4 | Divergence or actuation failure | read the report, then restart |
| 5 | Lobby error (blocked or unrecognized screen) | read the captured DOM, then restart |
| 1 | Unexpected crash | read the traceback |

Restarting with a fresh browser has reliably cleared stuck site-side states.

## Where the output goes

```
exports/arena/
  record.jsonl                     # one line per completed game, all sessions
  <session-ts>/
    session.json                   # which checkpoint/obs/sims played this session
    frames.jsonl                   # raw feed for the whole session
    exit.json                      # why the process stopped
    dom-*.html                     # DOM captures at failures/waits
    divergence-*.png, stall-*.png  # screenshots at aborts
    <ts>-game-<id>/
      frames.jsonl                 # raw feed for this game (replayable)
      events.jsonl                 # normalized events
      decisions.jsonl              # every question and our answer
      result.json                  # outcome, VP scores, placings, opponent
      divergence.json / stall.json # present only if this game aborted
```

Running record:

```sh
python3 -c "
import json; from collections import Counter
r=[json.loads(l) for l in open('exports/arena/record.jsonl')]
c=Counter(x['result'] for x in r); print(c, len(r), 'games')"
```

Human-readable decode of any feed:

```sh
PYTHONPATH=build ./.venv/bin/python -m src.v2.arena.protocol.dump \
  exports/arena/<session>/<game-dir>/frames.jsonl
```

## When something breaks

The driver is built to stop loudly rather than play on in an unknown state.
Every abort leaves enough evidence to diagnose offline:

1. Read `exit.json` for the class of failure.
2. Read `divergence.json` / `stall.json` — they carry the failing question,
   the intended action, the resolved click targets, and a tracker summary.
3. Look at the screenshot and any `dom-*.html` capture.
4. Decode `frames.jsonl` around the reported frame index.
5. Reproduce offline by replaying that archive in a test before fixing —
   every past incident in `tests/v2/arena/` was pinned this way, and those
   archives are cited by path, so don't delete referenced runs.

Two rules learned the hard way:

- **Never guess a selector.** If it is not in a recorded DOM capture, capture
  it first. Unknown blocking modals are snapshotted automatically.
- **"Process is running" does not mean "a game is happening."** Check the
  feed's last write time and the latest `TurnStart`, not just `pgrep`.

## Known gaps

- Only automatch is automated; hosted tables use a different (supported but
  less exercised) Ready control.
- The lobby's card-pool filter is set on the account, not by the driver.
- No post-hoc review tooling yet: archives are complete enough to rebuild any
  decision and re-run a deeper search over it, but nothing does that yet.
