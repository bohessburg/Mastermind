# Arena Stream — Progress Log

Running log for the "arena" stream: the best trained checkpoint (campaign 15
gen 45, card-token transformer) playing live base-set games against people on
dominion.games. The account is openly identified as a bot and the experiment
is run with permission. Architecture lives in `docs/arena-plan.md`; this file
tracks what is built and what is next.

## Context

dominion.games runs a standard web client that talks to the game server over
a public WebSocket. The same feed is what the site's own client uses to render
a game, and the site supports attaching to a running game's feed to spectate.
Our integration simply speaks that same client protocol: a browser session
(driven by Playwright) runs the normal client and uses its connection to handle
the lobby, we read the game feed the client already receives, and we play our
moves through the normal client UI. This is ordinary client interoperability —
reading a feed the client is designed to expose and clicking the same buttons a
person would, using the site's own client the whole way through.

## Status by phase

- **P0 — Recording harness: complete (2026-07-24).**
  `src/v2/arena/recon/record.py` + `src/v2/arena/browser/ws_hook.js`. A
  Playwright recorder with a persistent login profile that saves the game
  feed (both directions) as JSONL alongside periodic screenshots and DOM
  snapshots, for use as offline fixtures. Clean shutdown on Ctrl-C. Covered
  by an offline test using a local echo server (`tests/v2/arena/
  test_recon_record.py`) — no network required.

- **P1 — Feed decoder + event parser: complete (2026-07-24).**
  `src/v2/arena/protocol/` (`frames.py`, `cards.py`, `events.py`,
  `parser.py`, `recording.py`, `dump.py`). Turns the raw client feed into a
  clean, normalized stream of `GameEvent` objects (game start, turn start,
  play/buy/gain/trash/discard/draw, reveals, shuffles, resource updates,
  pending decisions, decision-resolved, game end, chat). Verified against a
  real 3-game recording:
  - **100% coverage** of game-relevant messages (21,203 of 21,203).
  - All three games decode with correct kingdoms, player names, and seat
    assignment, including a mid-game reconnect.
  - The client's full-state message is decoded directly, giving exact card,
    zone, and counter state at game start and on reconnect.
  - The public/private information boundary is handled correctly out of the
    box: our own hand decodes to exact cards, while the opponent's draws
    decode as counts only — which is precisely the information model the
    engine's search already expects.
  - Card identities map cleanly onto engine card definitions via a shared
    symbol scheme (e.g. feed `THRONE_ROOM` ↔ display "Throne Room" ↔ engine
    `def_id("Throne Room")`).
  - 16/16 tests pass (`tests/v2/arena/`).

  Full protocol notes are in the "P1 protocol findings" section of
  `docs/arena-plan.md`.

- **P2 — Public-information tracker: complete (2026-07-24).**
  `src/v2/arena/shadow/tracker.py`. Folds the P1 event stream into live
  public state: supply piles, per-seat zones (our hand exact; opponent hand
  as a count plus the derivable hand∪deck composition; discard composition;
  in-play; set-aside), the trash, resources, turn/phase, and the current
  pending-decision context, exposed as a frozen `TrackerSnapshot` — the input
  P3 turns into an engine `GameState`. Zones are tracked as known-card
  multisets plus an anonymous count, with per-seat ownership totals; any
  disagreement with a server-reported count raises a loud `TrackerError`
  (the divergence tripwire). Supporting parser extensions: the full-state
  message now also emits a `FullState` event carrying per-zone contents and
  counter values (previously discarded), and all movement events carry
  normalized from/to zone kinds; `Topdeck`, `ZoneTransfer`, `PileUpdate`,
  and `PileReorder` events were added.
  Verified golden-style over the 3-game recording
  (`tests/v2/arena/test_shadow_tracker.py`): all 16,883 fixture events fold
  with non-negative counts and per-card conservation asserted after every
  event; all four full-states reconcile exactly (including the mid-game
  reconnect, handled as a replacement resync); 300+ decisions cross-checked
  offered-cards ⊆ tracked hand. 18/18 tests pass (`tests/v2/arena/`).

## Remaining phases (from the plan)

- **P3 — StateBuilder (engine + bindings).** The one piece of C++ work:
  construct an engine `GameState` from a P2 snapshot, plus deck-order patching
  and an invariant check. This is what lets the trained model reason about a
  live game.
- **P4 — Decision service.** Factor the model-serving loop out of the web
  server into a shared module reused by both the server and the arena runtime.
- **P5 — Actuator + supervised game loop.** Map an engine action to the
  matching click in the normal client UI; play one full game with a person
  watching.
- **P6 — Lobby loop + supervisor + deploy.** Unattended base-set sessions
  with archiving and recovery.

## How to reproduce P1 results

```sh
# from repo root, with the venv and engine bindings built
./.venv/bin/python -m pytest tests/v2/arena/
# human-readable decode of a recording:
./.venv/bin/python -m src.v2.arena.protocol.dump arena-recordings/<run>/frames.jsonl
```

## Notes / carryovers

- Recordings and the browser profile live under `arena-recordings/` and
  `arena-profile/` (both gitignored). The current 3-game recording is the
  reference fixture for P1–P2 tests.
- Card and pile identities in the feed are numeric; the mapping to names is
  generated from the client's published card table (documented in
  `cards.py`) and regenerated when the client version changes.
- Work through P1 is currently uncommitted on branch `v2-phase1`.
