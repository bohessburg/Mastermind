# Web UI Handoff

Scope for a session working ONLY on the web playtest harness
(`src/v2/web/`). The engine, bindings, and training stack are owned by
another session — do not modify `src/v2/core|cards|mcts|encode|py|train`,
`tests/v2`, or CMake targets without coordinating.

## Architecture (do not violate)

- **The server is authoritative.** Client sends `{type:"act", action:<id>}`;
  server validates every action against the C++ legal mask before stepping.
  The client contains ZERO game logic — it renders server messages only.
- **Privacy is enforced server-side per seat**: own hand visible; opponent
  hand only as count; deck order never leaves the server; discard TOP is
  public, contents are not. The shared log must never name a card only one
  player is entitled to know (Sentry keeps, Militia non-top discards).
  Private detail goes through per-seat log delivery. There are pytest
  regression tests for all of this — extend them for anything new.
- Server: FastAPI, `src/v2/web/server/` (`main.py` sessions/WS/bots/undo,
  `observer.py` log formatting + prompts, `defs.py` def-table access).
  Card display data is generated at build time into
  `src/v2/web/client-data/defs.gen.json` (via `v2_dump_defs`).
- Client: Vite/React/TS, `src/v2/web/client/` (`protocol.ts` wire types,
  `state.ts` reducers/helpers — put logic here for testability, `App.tsx`
  components, `styles.css`).

## Run / test

```sh
# server (needs built bindings at build/)
PYTHONPATH=build ./.venv/bin/python -m uvicorn src.v2.web.server.main:app --port 8000
# client dev (proxies /api,/ws to :8000)
cd src/v2/web/client && npm run dev
# suites (ALL must stay green before commit)
PYTHONPATH=build ./.venv/bin/python -m pytest src/v2/web/server -q   # 20
cd src/v2/web/client && npm test && npm run build                    # 8 + build
```

### Neural-network bot seat

Run a local training checkpoint on CPU with `bot:nn`.  The default checkpoint
path is `checkpoints/remote/campaign1/gen_0025.pt`; set
`DOMINION_NN_CHECKPOINT` to select a different default:

```sh
DOMINION_NN_CHECKPOINT=checkpoints/remote/campaign1/gen_0025.pt \
  PYTHONPATH=build ./.venv/bin/python -m uvicorn src.v2.web.server.main:app --port 8000

curl -sS -X POST http://127.0.0.1:8000/api/session \
  -H 'content-type: application/json' \
  -d '{"seats":["human","bot:nn"]}'
```

Open `http://127.0.0.1:8000` and use the returned human seat token in **Join
existing game**.  `bot:nn:/path/to/checkpoint.pt` overrides the environment
variable for that seat.  The current create-game form only offers its built-in
bot choice, so custom NN seats are created through this API request.

**Caveat:** `bot:nn` is the raw policy network only. It does not use MCTS or
any other search at play time.

Production mode: `npm run build`, FastAPI serves `dist/` at `/`.
Server restart required after Python changes; hard-refresh after client
build. The engine module import comes from `build/` — if bindings are
stale, rebuild `dominion_v2_py` (see CLAUDE.md), but coordinate first.

## Wire protocol

Types in `client/src/protocol.ts` mirror the server exactly. Additive
fields only — never break existing consumers. Recent additions:
`decision.select_zone?: "hand"|"supply"|"discard"|"set_aside"` (which
zone's tiles are clickable for A_SELECT decisions),
`is_basic_treasure` in defs.gen.json (play-all-treasures button).

## Feature state (2026-07-09)

Done: session create/join (human-vs-bot, human-vs-human), full table UI,
decision panel with multi-select, click-to-play/buy/select on tiles with
zone disambiguation, play-all-basic-treasures, card popovers, trash-content
popover, verbose privacy-correct logs (per-hit Bandit incl. Throne replays,
Sentry summaries + acting-seat private lines), undo (human-vs-bot,
replay-prefix), game export (`GET /api/session/{id}/export`) which doubles
as golden-replay authoring + the fixture format
(`fixtures_throne_bandit_replay.json` test), random kingdom option,
no-scroll 100dvh layout with internal scroll regions.

Known gaps / candidate work:
1. The no-scroll layout is verified by CSS reasoning, NOT rendered pixels
   (sandbox had no browser). Verify at 1440×900 and 1280×800 across
   action/buy/multi-select/reaction/gameover states; fix any body scroll.
2. W.8 human playtest checklist (IMPLEMENTATION_PLAN §4b) is with the
   owner; bugs from playtests land here. Export files from suspicious
   games become regression fixtures (see test_server.py fixture test).
3. Reconnect UX: WS reconnect exists; session-resume polish untested.
4. No mobile/touch pass; desktop-first is fine for now.
5. Undo mid-multi-select rewinds a full human decision — acceptable,
   but re-verify after any decision-flow change.

## Conventions

- Owner's workflow: delegate non-trivial implementation to Codex
  (`codex:codex-rescue` subagent, `--resume` continues its session),
  review the actual diff yourself, iterate. Small glue edits directly.
- Commits: under 5 words, no body, no co-author lines.
- 2-player is the primary configuration.
- Cards removed in 2nd edition are out of scope (IMPLEMENTATION_PLAN
  conventions).
- The engine/bindings are trusted ground truth: if the UI shows something
  weird, replay the exported game before suspecting the engine — every
  log/display bug so far has been in the observer/UI layers.
