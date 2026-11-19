# dominion.games Live-Scrape Handoff

Session brief for building the live-game collector + converter. Written
2026-07-31 by the c20 campaign session. Self-contained: read this, then
the Repo Pointers table, and you can start without other context.

## Mission

Record live human games from dominion.games via its public spectator
websocket, convert qualifying games into DominionZero training tuples,
and grow the human corpus from today's ~12.7K tuples (~108 games) toward
100K+ games over time. This is the highest-leverage data project in the
program: the c20 campaign's central finding is that the value head dies
where engine wins are absent from outcome data, and human games are the
richest source of engine-win outcomes. At 100K+ games, imitation stops
being a small anchor and becomes an AlphaGo-class supervised phase (c21
design input).

## What was granted

- Jack contacted the dominion.games administrators (2026-07-31). Full
  database access was declined, but **permission to scrape live games
  via the websocket was granted**.
- Every game is publicly spectatable by design ("intended so that anyone
  can spectate an ongoing game").
- **Game IDs are a simple increasing integer** — enumerable discovery.
- Open item: get the training-use permission in writing / confirm terms
  of service cover model training, not just spectating. Flag to Jack;
  do not block collection on it, he owns this.
- Be a good citizen regardless of permission: bounded concurrent
  sockets (start ≤10), exponential backoff on errors, no hammering
  historical IDs, identifiable User-Agent if the protocol allows.

## Milestone 0 — protocol recon (do this first, everything depends on it)

We know nothing about their wire protocol. Capture it empirically:
spectate a live game in a browser with devtools open (Network → WS →
Frames), save the full frame log for several games start-to-finish.
Then answer, in a written recon note:

1. **Hand visibility (RISK #1, decides everything).** Does the spectator
   stream carry each player's actual hand/draw contents, or only the
   public log? Many clients send full state and filter in the UI —
   inspect raw frames, not the rendered page. Three outcomes:
   - Full info-set per seat present → full policy+value tuples, green
     light for the complete plan below.
   - Hidden but drawn-card identities appear when played/revealed →
     fallback tier: **buy-decision imitation only** (at buy time the
     played treasures and board are public; Thompson's Dominion DQN got
     money→engine conversion from buy decisions alone). Still valuable.
   - Also check: can a **finished** game be loaded/replayed by ID with
     more information than live spectating showed? Their replay/undo
     system implies complete server-side decision records; a
     completed-game endpoint may reveal per-seat views.
2. **Auth**: does the websocket require a logged-in session/cookie, or
   is it truly anonymous? If a session is needed, use one dedicated
   account (ask Jack to make one), never parallel account farming.
3. **Discovery**: how do you learn current live game IDs? Lobby/list
   endpoint, or probe the increasing integer near the newest ID?
   Measure global throughput by sampling the newest ID twice a day
   apart (ΔID ≈ games/day).
4. **Metadata before commitment**: can you see the kingdom, player
   count, and player ratings/levels *before* subscribing to the full
   stream? If yes, pre-filter and only record qualifying games.
5. **Card identifiers**: names or numeric IDs on the wire? Get the
   full vocabulary for the mapping table.
6. **End-of-game**: how are completion, resignation, and timeout
   signaled? Is final score in the stream?

## What qualifies (the filter)

- 2 players, game reached a real conclusion.
- Kingdom = 10 cards, ALL within our 26-card 2E base roster: Artisan,
  Bandit, Bureaucrat, Cellar, Chapel, Council Room, Festival, Gardens,
  Harbinger, Laboratory, Library, Market, Merchant, Militia, Mine,
  Moat, Moneylender, Poacher, Remodel, Sentry, Smithy, Throne Room,
  Vassal, Village, Witch, Workshop. Basics only otherwise (Copper,
  Silver, Gold, Estate, Duchy, Province, Curse) — no Colony/Platinum/
  Potion, no events/landmarks/ways/projects, no Shelters.
- Likely-favorable base rate: free-tier dominion.games accounts play
  base-only, so base games should be plentiful — verify by sampling.
- Record skill metadata (ratings/levels) if visible; do NOT filter on
  skill at collection time — filter at training time. Storage is cheap,
  recollection is impossible.
- Resignations: keep the game; policy decision at conversion time is
  win/loss sign only, no margin target (margin is undefined — the aux
  margin head and pure-margin value target need real final scores, so
  resigned games contribute sign-only value or policy-only tuples).

## Architecture (proposed, adjust to recon findings)

```
collector daemon (24/7)          offline, on demand
┌──────────────────────┐   ┌─────────────────────────────┐
│ discover live IDs    │   │ filter (base-2E, 2p, done)  │
│ subscribe ≤N sockets │ → │ convert → our record schema │
│ append raw frames    │   │ golden-verify vs engine     │
│ *.jsonl.gz per game  │   │ emit training tuples (npz)  │
└──────────────────────┘   └─────────────────────────────┘
```

- **Raw first, always.** Store the untouched frame stream per game
  (`data/dominion_games/raw/<game_id>.jsonl.gz` + a manifest with
  capture date, kingdom, players, ratings, outcome). Conversion will
  be rewritten as schema understanding improves; raw is ground truth
  and can never be re-fetched. Never convert-and-discard.
- Collector host: Jack's Mac or the Hetzner VPS (138.199.222.149).
  VPS pros: always-on, already public. VPS rule: **no repo source on
  that box** (standing order after the source wipe) — deploy the
  collector as a self-contained script or container, like the game
  server. The collector is tiny (a websocket client + gzip writer);
  a single-file script with no repo imports is the cleanest fit.
- Conversion runs locally in the repo where the engine bindings live.

## Conversion: scraped stream → training tuples

Target is the existing tuple pipeline, NOT a new one. Today's flow:
web-server session exports (`exports/*.json`, schema: `{seed, kingdom
(def ids), seats, actions (flat id list), obs_version,
final_state_hash}`) are exactly replayed with the seed by
`src/v2/train/human_data.py`, which calls the engine at each replayed
ply and writes `exports/tuples/tuples-*.npz` + `tuple_manifest.json`.
Scraped games have **no seed**, so exact seeded replay is impossible.
Two candidate strategies — prototype B first, it self-verifies:

- **Strategy B (forced-deal replay, preferred):** replay the scraped
  action sequence through our engine with drawn/revealed card
  identities forced to match the observed log instead of RNG. Needs an
  engine hook (a scripted-dealer mode on the shuffle/draw path). Gives
  full-fidelity `GameState` at every ply → `encode()` → tuples, and
  legality checking at every step is the golden verify for free. Check
  whether fuzz/golden-replay infra already has a deal-injection hook
  before writing one.
- **Strategy A (info-set snapshot):** build a state per decision from
  the observed info-set (obs is a pure function of hand + public info +
  deck-as-multiset; deck ORDER doesn't matter — established finding,
  see `scripts/probes/human_record_probe.py` and the obs analysis in
  docs/training-log.md). Needs a state-injection constructor in the
  bindings; more code, no free legality check. Fallback if forced-deal
  turns out invasive.
- **Golden verification (non-negotiable):** every recorded action must
  be legal in our engine at its reconstructed state, and the terminal
  score/supply must match the stream's end state. Divergence = rules
  bug on one side or reconstruction bug — quarantine the game, log it,
  never silently drop or force-fit. Track the divergence rate; if
  >~2% investigate before scaling up.
- **Action mapping:** their card vocabulary → our DefIds
  (`src.v2.web.server.defs.load_defs()` / `def_id(name)`), their
  decision structure → our flat action ids (buys are `206+def`,
  action plays `1+def` — see `bench/buy_stats.py` for the decode).
  Watch ordered sub-decisions (Sentry keep/trash/discard/topdeck
  order, Library set-asides, Throne Room targets): their event
  granularity may not be 1:1 with our choice frames.
- Keep scraped tuples in a **separate store** from the seeded-export
  corpus (e.g. `exports/tuples_dgames/`) with a source tag — training
  configs choose the mix; provenance must survive.

## Acceptance criteria

- M0: recon note answering the six questions, with saved raw captures
  of ≥3 complete games checked into `data/dominion_games/recon/`.
- M1: collector runs 24h unattended, ≥95% of subscribed games captured
  to completion, restart-safe (survives disconnects, resumes discovery,
  no duplicate IDs), with heartbeat logging (games live / captured /
  dropped) — long-running scripts must emit progress, PYTHONUNBUFFERED,
  no opaque multi-hour silence.
- M2: converter turns ≥1 real scraped base-set game into tuples with
  clean golden verification end-to-end.
- M3: filter + conversion batch over a week of capture; report yield
  (games/day scraped, % base-2E, % surviving verification, tuples/game)
  so we can project time-to-100K.

## Boundaries (active systems this session must not touch)

- c20 training is LIVE on box 2 (ssh -p 30839 root@211.21.106.81) —
  do not touch that box at all.
- The Hetzner VPS game server (deploy-dominion-1/-caddy-1 containers,
  https://paisho.duckdns.org) is serving the public champ — if the
  collector lands there, it must not modify the existing containers,
  compose file, or ports 80/443/8000, and no repo source on the box.
- Jack's arena terminal sessions are his; never monitor or kill them.
- Don't modify the existing seeded-export pipeline (`human_data.py`
  replay path) — extend alongside it.

## Repo pointers (read in this order)

| Path | Why |
|---|---|
| `docs/session-handoff.md` | overall project state, c20 context |
| `src/v2/train/human_data.py` | existing export→tuple pipeline to extend (tuple_dir `exports/tuples`) |
| `exports/*.json` | current export schema (the seeded flavor) |
| `src/v2/web/server/defs.py` | card name ↔ DefId mapping source |
| `bench/buy_stats.py` | action-id decode conventions (buy 206+def, play 1+def) |
| `scripts/probes/human_record_probe.py` | info-set/obs reconstruction principle in practice |
| `docs/implemented-cards.md` | exact supported roster for the filter |
| `docs/dominion_rules_reference.md` | rules reference when streams disagree with our engine |
| `configs/run_c20.json` `imitation` block | how tuples are consumed (BC pretrain + anchor) |
| `IMPLEMENTATION_PLAN.md` task #30 | prior plan this supersedes (DB dump → live scrape) |

## Notes for the implementer

- Store everything as append-only; the collector must be crash-safe
  mid-game (partial captures marked incomplete, resumed or discarded).
- Timestamps in UTC everywhere; game_id is the primary key.
- Expect to build the card-name mapping table once during recon and
  golden-test it (all 33 base cards round-trip).
- When in doubt about a rules interaction in conversion, the engine is
  authoritative for tuples; the quarantine bucket is authoritative for
  honesty about what didn't convert.
