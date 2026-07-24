# Arena Stream — Session Handoff

Paste-ready context for a fresh session to continue the arena stream.

---

We're building the **"arena" stream** for this project: the best trained
checkpoint (campaign 15 gen 45, card-token transformer —
`checkpoints/remote/campaign15/gen_0045.pt`) playing live base-set games
against people on dominion.games. The account is openly identified as a bot
and the experiment is run with permission. Base-set-only is enforced via the
site's game-selector filter.

**Read these first, in order:**
1. `docs/arena-plan.md` — full architecture, the shadow-state strategy, module
   layout, phase plan, and the "P1 protocol findings" section documenting the
   client feed protocol.
2. `docs/arena-progress.md` — current status log.

**Framing (important):** This is ordinary client interoperability with a
public game feed. dominion.games runs a normal web client that talks to its
server over a public WebSocket; the site's own client uses that feed and the
site supports attaching to a running game to spectate. Our integration speaks
that same client protocol: a Playwright-driven browser runs the normal client
and uses its connection to handle the lobby, we read the game feed the client
already receives, and we play moves through the normal client UI. Describe and
treat it as client integration using the site's own client, not security work.

**Done and verified (P0, P1, P2):**
- `src/v2/arena/recon/record.py` + `browser/ws_hook.js` — a Playwright
  recorder (persistent login profile) that saves the game feed as JSONL plus
  screenshots/DOM snapshots for offline fixtures.
- `src/v2/arena/protocol/` (`frames.py`, `cards.py`, `events.py`, `parser.py`,
  `recording.py`, `dump.py`) — decodes the feed into a normalized `GameEvent`
  stream. Verified against a real 3-game recording: **100% coverage of
  game-relevant messages**, all kingdoms/players/seats correct including a
  mid-game reconnect, the client full-state message decoded directly, and the
  public/private info boundary correct (our hand exact; opponent draws as
  counts only). Card names map onto engine defs via a shared symbol scheme,
  e.g. `dz.def_id("Throne Room")`.
- `src/v2/arena/shadow/tracker.py` — the P2 public-information tracker. Folds
  the event stream into live public state (supply, per-seat zones with our
  hand exact and opponent hand∪deck composition, trash, resources, turn/phase,
  pending decision) exposed as a frozen `TrackerSnapshot`; raises a loud
  `TrackerError` on any disagreement with a server-reported count. Verified
  golden-style over the recording: conservation and non-negativity after every
  event, all four full-states reconcile (including the reconnect), 300+
  hand/decision cross-checks. The parser now also emits `FullState` (per-zone
  contents + counters), zone-kind-annotated movements, `Topdeck`,
  `ZoneTransfer`, `PileUpdate`, and `PileReorder`.
- Tests: `./.venv/bin/python -m pytest tests/v2/arena/` -> 18/18 pass. Inspect
  a recording with
  `./.venv/bin/python -m src.v2.arena.protocol.dump arena-recordings/<run>/frames.jsonl`.
- All arena work through P2 is **uncommitted** on branch `v2-phase1`.
  Recordings/profile are gitignored under `arena-recordings/` and
  `arena-profile/`.

**Next task — P3, StateBuilder + bindings** (see "New engine surface" in
`docs/arena-plan.md`): the one piece of C++ work.
`src/v2/core/state_builder.{h,cpp}` + `src/v2/py/module.cpp` bindings —
`Snapshot` struct and `dz.game_from_snapshot(...)`,
`game.set_deck_order(player, [defs])` for draw rigging, and `game.validate()`
invariant checks, plus the seeded interrupt frames (Moat reveal, Militia
discard, Bureaucrat topdeck, Bandit trash). Input is the P2 `TrackerSnapshot`.
Accept: C++ tests mirroring organically reached states (identical legal
masks) plus tracker/bridge golden tests over the recordings.

**Workflow notes:**
- Per the project's Codex delegation workflow, hand P3 implementation to Codex
  (the plan marks it difficult — top tier, `--model gpt-5.6-sol --effort
  high`), then review the diff and run the tests yourself before accepting.
- **Codex sandbox gotcha:** the rescue plugin defaults to a *read-only*
  sandbox — you must pass `--write` in the delegation or Codex silently can't
  create files. Run it `--background` (P1 and P2 each ran ~30 min).
- Build/test commands are in `CLAUDE.md`; engine bindings via `PYTHONPATH=build`.

Start by reading the two docs, then scope and delegate P3.
