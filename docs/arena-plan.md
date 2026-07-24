# Arena: DominionZero on dominion.games

Architecture for the "arena" stream: the best trained checkpoint (campaign 15
gen 45, card-token transformer) playing live base-set games against humans on
dominion.games. The account is disclosed as a bot and permission for the
experiment has been granted; games are restricted to base set via the site's
game-selector filter.

Status: P0–P4 complete and tested (2026-07-24): recording harness, feed
decoder/parser (100% decode coverage over a 3-game recording), public-info
tracker (golden-reconciled against every full-state), StateBuilder + bindings
+ bridge (930 recorded decision points build valid games; 529 taken actions
confirmed legal), and the shared decision service. Next: P5 (actuator +
supervised game loop). See "P1 protocol findings" below. Recordings land
under `arena-recordings/` (gitignored).

## Design decisions (settled)

1. **One browser, one connection.** A Playwright-driven Chromium session runs
   the normal web client and uses its single logged-in WebSocket connection.
   We reuse that one client connection rather than opening a second one,
   because the client already maintains the session (login, keepalives,
   message ordering) and a second connection would just duplicate and conflict
   with it.
2. **Read the client's own feed.** A small script registered before page load
   observes the frames on the client's own WebSocket and forwards them to
   Python. Game state comes from this decoded feed rather than from the
   rendered page.
3. **Read the feed, click the UI.** Our moves are made as normal clicks in the
   client UI; the feed side is read-only. Sending protocol messages directly
   is a possible later convenience, taken only if the recordings show clean,
   stateless message shapes.
4. **No algorithm changes.** `DecisionSearcher` + `determinize()` already
   plays from public info + own hand (opponent hand∪deck is re-dealt
   randomly, counts preserved). The bot needs a shadow `GameState` that is
   correct in exactly that information — nothing more.
5. **Kingdom gate.** At game start the parsed kingdom is checked against the
   def table. Unknown card → post a polite chat message, resign, requeue.
   (Base-set filter should make this rare; the gate makes it safe.)

## Shadow-state strategy

The engine cannot replay a remote game via `new_game` + `step()` because its
RNG draws/shuffles will not match reality. Three regimes keep the shadow
state exact:

- **Turn-boundary resync.** At the start of each of our turns, build a fresh
  `GameState` from the tracker snapshot (supply piles, our exact hand + deck
  composition + discard, opponent zone counts + hand∪deck composition,
  in-play, resources, turn/phase). Clean state, empty frame stack. Any drift
  accumulated while observing the opponent's turn is wiped.
- **Rigged stepping within our turn.** Our own actions are stepped through
  the real interpreter so mid-resolution frame stacks (Throne Room chains,
  Library, Vassal, Cellar draws...) stay exact. Draw outcomes are forced by
  patching our deck order to match observed reality *before* stepping: click
  first, observe the server's actual outcome frames, arrange the shadow deck
  top, then `step()`. The interim (unobserved) deck order never matters
  because every draw is rigged just-in-time.
- **Seeded interrupt frames on opponent turns.** Opponent actions are never
  stepped. When the opponent's turn demands a decision from us, build a
  snapshot with one seeded pending frame. The base-set interrupt surface is
  small and enumerable:
  - Moat reveal (reaction window to any attack)
  - Militia: discard down to 3
  - Bureaucrat: choose/reveal Victory card to topdeck
  - Bandit: choose which revealed Treasure to trash (when two qualify)

Every decision then flows: tracker snapshot → shadow `Game` →
`DecisionSearcher` (existing determinized search) → action id → actuator.

Divergence detection: after each of our actions, the next decoded frames must
match the intended action; the tracker also cross-checks zone counts against
every count the server reports. Any mismatch aborts the game loudly
(screenshot + frame log), never plays on.

## New engine surface (C++ + bindings)

The only engine work in the stream. In `src/v2/core/state_builder.{h,cpp}`,
exposed through `src/v2/py/module.cpp`:

1. `Snapshot` struct + `dz.game_from_snapshot(snapshot)` — construct a
   `GameState` from: supply pile counts, per-player zones (our hand exact;
   opponent hand as count + combined hand∪deck composition), discard
   compositions, in-play, resources (actions/buys/coins), turn number, phase,
   current player, and an optional seeded interrupt frame (enum of the four
   cases above + attacker/defender seats).
2. Zone patching for draw rigging — `game.set_deck_order(player, [defs])`
   (full deck contents in order; must be a permutation of the current deck
   composition or it throws).
3. Invariant check — `game.validate()` (card conservation per pile,
   zone-count consistency) callable after any build/patch; used in tests and
   as a runtime tripwire.

`GameState` is POD and memcpy-cloneable, which is what makes snapshot
construction tractable. Builder states must be indistinguishable from
organically reached states as far as the legal mask and interpreter are
concerned.

## Module layout (Python, new)

```
src/v2/arena/
  main.py            # orchestrator entrypoint (asyncio), config load
  config.py          # ArenaConfig (see Config below)
  browser/
    session.py       # Playwright lifecycle, persistent profile, observer install
    ws_hook.js       # registers the client-feed observer via add_init_script
  protocol/
    frames.py        # raw frame envelope decode (format fixed by recordings)
    events.py        # normalized GameEvent dataclasses
    parser.py        # frames -> GameEvent stream
  shadow/
    tracker.py       # public-info tracker: zones, counts, kingdom, decisions
    bridge.py        # tracker snapshot -> dz.game_from_snapshot / rigged step
  bot/
    policy.py        # checkpoint load + DecisionSearcher loop (factored out
                     # of web/server/main.py, shared by both)
  actuate/
    clicks.py        # action id -> DOM gesture map (uses actions.h regions)
    verify.py        # post-action frame verification
  fsm/
    lobby.py         # login, base-set search/automatch, join, requeue
    game.py          # in-game loop: wait decision -> snapshot -> search -> act
    supervisor.py    # health monitor, recovery (reload+reconnect), alerts
  archive.py         # per-game record: frames.jsonl, events, results summary
recon/
  record.py          # phase-0 harness: manual play, log frames + screenshots
```

`bot/policy.py` is extracted from `_choose_nnmcts_action` /
`_load_nn_policy` in `src/v2/web/server/main.py`; the server imports it so
there is one serving path.

## Testing

- **Parser:** recorded `frames.jsonl` fixtures → expected event streams
  (table-driven; new fixture per client-update breakage).
- **Tracker + bridge (golden-style):** for each recorded game, at every one
  of our decision points, build the shadow state and assert (a) the action
  actually taken is in `legal_mask()`, (b) `validate()` passes, (c) tracked
  zone counts match every server-reported count.
- **StateBuilder (C++, `tests/v2/`):** build snapshots mirroring organically
  reached states (via seeded self-play) and assert identical legal masks and
  `state_hash`-relevant behavior; interrupt-frame seeding covered per case.
- **Live:** supervised single games (operator watching the real UI, which
  keeps rendering) before any unattended run.

## Config

`configs/arena.json`:

- checkpoint path (default `checkpoints/remote/campaign15/gen_0045.pt`),
  obs version, sims (default 400 — sweep shows flat 200–1000, do not raise),
  determinizations, per-decision wall-clock cap
- think-time pacing (min/max delay so play speed is polite, not instant)
- lobby: search filter settings, rated/unrated, requeue policy, max games
  per session
- chat: game-start announcement text (bot disclosure)
- account login via env (`ARENA_USER` / `ARENA_PASS`), never in the repo

## Deployment

- **Dev:** headful Chromium on the Mac, operator watching.
- **Prod:** Hetzner box (already hosts the checkpoint): `xvfb-run` + x11vnc
  for remote eyeballs, systemd unit, archives to `exports/arena/`,
  jsonl logs. CPU torch inference is fine for one concurrent game.

## Phases

Implementation is delegated to Codex per the standard workflow; the recorded
games need a human on the account.

- **P0 — recording harness** (routine): `recon/record.py` + `ws_hook.js`.
  Accept: operator logs in manually, plays; harness emits timestamped
  bidirectional `frames.jsonl` + periodic screenshots. Then: 3–5 manual
  games recorded, including at least one Militia/Moat interaction and one
  game-end → lobby → new-game cycle.
- **P1 — feed decode + parser** (difficulty set by frame format):
  `protocol/` with fixture tests over the recordings. Accept: every recorded
  game parses to a complete event stream with zero unknown-frame gaps in
  game-relevant messages.
- **P2 — tracker** : `shadow/tracker.py`. Accept: golden-style count checks
  pass over all recordings.
- **P3 — StateBuilder + bindings** (difficult — top-tier delegation):
  engine work above. Accept: C++ tests + tracker/bridge golden tests pass.
- **P4 — decision service extraction** (routine): `bot/policy.py`; server
  tests still pass.
- **P5 — actuator + supervised game loop**: `actuate/`, `fsm/game.py`.
  Accept: one full supervised game played end-to-end with zero divergence
  aborts.
- **P6 — lobby FSM + supervisor + deploy**: unattended multi-game sessions
  on Hetzner with archiving and recovery.

P1–P2 and P3 can proceed in parallel once recordings exist; P4 is
independent and can go first.

## Open questions (non-blocking)

- Frame format (JSON vs binary) — decided by the P0 recordings; determines P1
  effort and whether sending messages directly ever replaces DOM clicks.
- Automatch vs hosting a table with the base filter — whichever the lobby
  recordings show is more reliably automatable.
- Reconnect semantics after a mid-game browser reload — the recordings should
  capture one deliberate reload to see the resync messages.

## P1 protocol findings

Confirmed against web client bundle 2.2.8 and
`arena-recordings/20260724T142103.096991Z/frames.jsonl`:

- Binary integers are big-endian. Protocol `int` values occupy four bytes;
  signed values use two's-complement. Strings are a byte-length `u32` followed
  by UTF-8. Arrays begin with an element-count `u32`. Booleans are one byte.
- Inbound WS bodies are `[sequence:u32][msgType:u32][payload]`. Sequence
  numbers are monotonic from zero for each connection and reset after the
  recorded reconnect. Outbound bodies are `[msgType:u32][payload]` and have no
  client sequence field. Four zero-length inbound binary recorder records are
  valid no-ops.
- Card IDs are insertion ordinals of the bundle's `CardNames` object, not
  engine definition IDs: `Back=0`, `Curse=1`, `Copper=2`, `Artisan=8`, and
  `Workshop=33`. Bundle 2.2.8 has 881 slots. Thirteen reserved symbols have the
  shared display name `unused`, so the generated table retains both unique
  symbols and display names.
- Server message 32 (`gameEventInfo`) begins with a subtype `u32`. Observed
  layouts are:
  - 0 CardMove:
    `[fromZone][toZone][cardIds:int[]][visibleIds:int[]][movementType]`
    `[animationClass]`
  - 1 CounterChange: `[counterIndex][newValue]`
  - 2 PileUpdate: `[zoneIndex][topCardId]`
  - 3 TurnDescription:
    `[ownerSeat][turnNumber][turnType][controllerSeat]`
  - 4 Shuffle: `[ownerSeat][includeDiscard:boolean]`
  - 8 ZoneCreated: `[zoneIndex][ownerSeat][zoneType][createdByCardId]`
  - 19 PileReorder: `[zoneIndex][cardIds:int[]]`
- Server message 33 (`gameLogInfo`) is
  `[startIndex][entryCount][entries...]`. Each entry starts with an entry type.
  Type 0 is `[logName][depth][arguments:LogArgument[]]`; type 1 is
  `[decisionIndex][playerSeat][answers:int[]][autoPlayed:boolean]`.
  `LogArgument` is a tagged union. The recording exercises card frequencies,
  player, number, turn-description, cost, and metagame-info arguments; all
  11,868 message-33 frames consume exactly.
- Server message 37 (`questionAsked`) is
  `[questionIndex][questionClass][classPayload]`. Choice questions contain a
  question description, min/max, a tagged offered-element array, decline
  button, accumulated answers, and affected cards. Complex questions contain
  recursively tagged subquestions. The recording exercises card, buyable-card,
  game-button, and card-mode choices. Client message 37
  (`ANSWER_QUESTION`) is
  `[questionIndex][answerCount][answers...][autoPlayed:boolean]`; all 933
  observed questions and 932 recorded client answers parse exactly.
- Server message 38 (`fullGameState`) supplies the game ID, player IDs,
  card-name ordinals and instance counts, zones/owners, counters, and current
  state. The first seven non-Back card ordinals are the base piles; the
  remaining ten ordinals identify the selected base kingdom. Login-success
  message 2 supplies our player ID, and table-details message 10 supplies
  player names, allowing `GameStart(kingdom, players, our_seat)` normalization.
- Other decoded supporting traffic is server 0 (chat), 14 (game finished), 34
  (ticker log entry), 36 (pong), 41 (timer update), and the empty tournament
  overview seen at 47; client 0 is outbound chat and client 44 is the empty
  heartbeat. On the supplied recording, structural coverage for inbound
  32/33 plus outbound 37 is 21,203/21,203 (100%).

Remaining unknowns are lobby/account/control messages that do not affect
the normalized game stream. In this fixture their `(direction, msgType)` keys
are inbound 1, 5, 6, 7, 16, 19, 20, 21, 23, 29 and outbound 1, 6, 11, 12, 22,
24, 27. Their raw payloads and sequence metadata are emitted as
`UnknownFrame`, rather than discarded. Game-finished message 14 is currently
normalized as an end marker; detailed score/table-result fields remain for a
later parser extension.
