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

**Done and verified (P0–P6; the bot plays live unattended):**
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
- `src/v2/core/state_builder.{h,cpp}` + `module.cpp` bindings + `shadow/
  bridge.py` — the P3 StateBuilder. `dz.game_from_snapshot(dict)` builds a
  native `GameState` from a `TrackerSnapshot` (our hand exact, opponent
  hand∪deck re-dealt for `determinize()`, seeded interrupt frames for Moat/
  Militia/Bureaucrat/Bandit), `game.set_deck_order(player, defs)` rigs draws
  (draw-order API; must be a permutation), `game.validate()` enforces
  base-set card conservation. Golden over the recording: 930 our-turn
  decision points build valid games; 529 actually-taken plays/buys confirmed
  in the legal mask; mid-card questions (337) are explicitly excluded — they
  need P5's rigged stepping.
- `src/v2/arena/bot/policy.py` — the P4 shared serving path (checkpoint load
  + NN / NN-MCTS decision loop), imported by the web server. Wall-clock cap
  degrades to first-legal (binding limitation), so config it generously.
- Tests: 144/144 CTest, 20/20 arena pytest, 29/29 server pytest, py smoke.
  Server pytest needs `OMP_NUM_THREADS=1` on the Mac (pre-existing torch/OpenMP
  import deadlock, see progress doc). Inspect a recording with
  `./.venv/bin/python -m src.v2.arena.protocol.dump arena-recordings/<run>/frames.jsonl`.
- Committed on `v2-phase1`: P0–P2 (`a2780de`), P4 (`a450bd3`), P3 next commit.
  Recordings/profile are gitignored under `arena-recordings/` and
  `arena-profile/`.

**P5 and P6 are also done, and the bot has played real games.** `actuate/`
(answer mapper + Playwright and Mock actuators + divergence verification),
`fsm/game.py` (the in-game loop), `fsm/lobby.py` (full unattended lobby
cycle), `browser/session.py` (live session + in-process feed pump),
`config.py`, `archive.py`, and `scripts/arena_overnight.sh`.

**Game answers go out on the wire, not as clicks.** After repeated
canvas-click failures, `actuate/protocol.py` sends the client's own
`ANSWER_QUESTION` message (msg 37) — all 932 recorded human answers
re-encode byte-identically. Clicking is now used only for lobby buttons and
modals. `actuation_mode: "clicks"` remains as a fallback.

**First unattended overnight run (2026-07-24/25):** 21 completed games,
record 6W–14L–1T, appended to `exports/arena/record.jsonl`. Eight distinct
failure classes were found live and fixed, each pinned by a regression test
built from its own archive — see "First live results" in
`docs/arena-progress.md` for the list.

**To run it: `docs/arena-usage.md`** (prerequisites, preflight, exit-code
contract, archive layout, and how to diagnose an abort).

**Possible next tasks** (none blocking):
- **Loss review tooling.** Every game archive can rebuild the exact engine
  state at each of our decisions; nothing yet re-runs a deeper search over
  them to flag likely blunders. Suggested: `python -m src.v2.arena.review
  <game-dir>` annotating each decision with a high-sim search preference.
- **Aggregate stats** over `record.jsonl` (win rate by kingdom, opening buys
  vs outcome).
- **Strength work** — the bot currently loses more than it wins against live
  humans; that is a training/search question, not a driver question.
- Hosted-table flow is supported but lightly exercised (automatch is the
  tested path).

**Workflow notes:**
- Per the project's Codex delegation workflow, hand implementation work to
  Codex, then review the diff and run the tests yourself before accepting.
- **Codex sandbox gotcha:** the rescue plugin defaults to a *read-only*
  sandbox — pass `--write` in the delegation or Codex silently can't create
  files. Run long tasks `--background` (most ran 15–35 min). Two gotchas seen
  repeatedly: the forwarder sometimes launches a DUPLICATE job after a
  timeout (check `status`, cancel the copy with zero applied changes), and
  long threads occasionally stall silently — check the job log's last
  timestamp before assuming it is working.
- Local gotcha: run the web-server pytest with `OMP_NUM_THREADS=1` (known
  torch/OpenMP import deadlock on this Mac, predates the arena stream).
- Build/test commands are in `CLAUDE.md`; engine bindings via `PYTHONPATH=build`.
- **Never guess a DOM selector**; capture it. **Never trust `pgrep` as proof
  a game is happening**; check feed freshness.

Start with `docs/arena-usage.md` if you want to run it, or
`docs/arena-progress.md` if you want the build history.
