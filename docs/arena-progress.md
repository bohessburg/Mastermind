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

- **P3 — StateBuilder + bindings: complete (2026-07-24).**
  `src/v2/core/state_builder.{h,cpp}` constructs a clean or single-interrupt
  native `GameState` from tracker public state, with exact local hand,
  re-dealt opponent hand∪deck composition, public zones/resources, supply,
  trash, turn, and phase. The four base-set interrupt frames (Moat, Militia,
  Bureaucrat, Bandit) seed the existing interpreter layouts directly.
  `game.set_deck_order()` rigs full decks in next-draw-first order and rejects
  non-permutations; `game.validate()` checks structural invariants and
  base-set per-card conservation without changing the POD `GameState` layout
  or byte hashes. `src/v2/arena/shadow/bridge.py` maps tracker display names
  and phases to the native dict binding, including deterministic resolution
  of reconnect-era anonymous public-zone slots from known ownership.
  C++ tests cover 100+ sampled organic decision states, same-action behavior,
  all interrupts, rigged draws, invalid snapshots, and deliberate corruption.
  The real-recording golden builds 930 local-turn decision points and checks
  305 action plays, 130 buys, and 94 treasure plays against native legal
  masks, with 337 internal questions and 64 pass/unmapped phase answers
  counted explicitly. The full 144-test CTest suite and the bridge golden pass.

- **P4 — Decision service: complete (2026-07-24).**
  `src/v2/arena/bot/policy.py`: the checkpoint loader and NN / parked-leaf
  NN-MCTS decision functions factored out of the web server, parameterized by
  sims, determinizations, device, and an optional per-decision wall-clock
  cap. The server imports it, so there is one serving path. Note: a mid-search
  wall-clock timeout falls back to the first legal action (the binding's
  `best_action()` is only meaningful once the search completes), so the arena
  config should treat the cap as a generous safety valve, with pacing done via
  think-time delay. Server suite passes unchanged.

## Remaining phases (from the plan)

- **P5 — Actuator + supervised game loop: offline complete (2026-07-24).**
  `src/v2/arena/actuate/` supplies a pure engine-action → client-answer
  mapper, a DOM/canvas Playwright actuator, an offline mock actuator, and
  post-action verification. `src/v2/arena/fsm/game.py` is the supervised
  asyncio loop: tracker tripwire, kingdom gate, clean phase-boundary/native
  resync, seeded opponent interrupts, policy/provider boundary, polite
  think-time delay, observed-draw rigging, native stepping, and loud
  screenshot-capable divergence abort. `config.py`, `configs/arena.json`, and
  `archive.py` provide operator settings, environment-only credentials, and
  per-game JSONL/result records.

  Offline acceptance against the reference recording:
  - All **932/932 recorded answers** are producible by the pure mapper.
    **152** questions have multiple equivalent encodings (duplicate physical
    card copies, selection order, or manual-single-treasure vs autoplay);
    these are enumerated rather than guessed.
  - All three recorded games replay to `GameEnd` with **zero divergence
    aborts**. The mock actuator submits 932 live questions: three pre-turn
    start handshakes plus 929 engine decisions. One of the 933 pending
    questions is superseded by `GameEnd` without an answer. Every engine
    decision validates its native shadow, and 259 observed-outcome steps use
    deck rigging/reconstruction.
  - A deliberately corrupted `DecisionResolved` aborts immediately and stops
    the actuator.

  Python binding workaround: `set_deck_order()` can order only the current
  deck, not discard just before an interpreter-internal shuffle. The loop
  therefore retains the clean snapshot/action history for the current effect
  segment; after observing such a shuffle it rebuilds with the known
  deck+discard pool in observed source order and replays the native actions.
  No C++ or binding changes were needed. The remaining P5 step is the human
  operator's single supervised live game, which will verify the documented
  client-2.2.8 canvas selectors.

  The supervised live runner is now wired: `browser/session.py` launches the
  persistent headful profile, installs the same WebSocket hook before client
  navigation, archives every raw frame, and streams records through the shared
  decoder/parser into the game loop without a disk round-trip. `main.py`
  supports the operator-run live command and a queue-backed fixture dry run.
  Lobby setup remains deliberately manual for this phase. The outstanding P5
  acceptance step is still the operator-observed live game.

- **First supervised live-game follow-up (2026-07-24).**
  The first live game exposed two integration bugs and was stopped by the
  divergence tripwire. The archived turn-one buy question still had three
  Coppers in hand. `DecisionSearcher(auto_play_treasures=True)` correctly
  returned its first forced canonical Copper play without requesting an NN
  evaluation, but the arena provider treated that collapse step as the final
  client decision. The provider now mirrors organic self-play: on a buy
  question it clones the shadow, plays every offered treasure in ascending
  definition order, asks the policy for a buy/pass only on that collapsed
  state, submits the client's autoplay control, and defers the selected
  buy/pass until the feed supplies the treasure-free follow-up question. This
  keeps the checkpoint's evaluated root at the training distribution (played
  treasures and credited coins) while retaining manual-treasure encodings in
  recorded replay.

  The same question also exposed a DOM-region collision. The old actuator
  searched every visible card stack by display name, so the supply Copper was
  the first match for an intended hand Copper. Client 2.2.8's saved DOM maps
  the protocol regions as follows:

  - `0:<card>`: landscape supply-pile child of `div.card-stacks`, with a
    visible pile counter.
  - `1:0:<card>`: portrait local-hand child of `div.card-stacks`, with the
    client hand z-index range 2000–2999.
  - `2:AUTOPLAY_TREASURES`: the wide (at least 3.5:1) canvas under
    `div.game-buttons`; the separate rightmost canvas is the decline/end-phase
    control.
  - Effect choices: local-hand and supply questions reuse those regions;
    revealed-card overlays use portrait display stacks at z-index 10000+.

  Targets are now resolved by region plus exact identity, and actuation stops
  if that pair does not identify exactly one physical element. Finally,
  resolution answers are checked as soon as each `DecisionResolved` event
  arrives. Previously the loop buffered all outcome events and called the
  verifier only at the next `PendingDecision`; question 3's first mismatch was
  therefore detected at question 5, making the report look one turn late.
  Repeated/batched resolution copies are each checked immediately and cannot
  delay or weaken the first check.

- **Second supervised live-game follow-up (2026-07-24).**
  The second run reached turn 9 with 17 verified autoplay/buy decisions, then
  stopped at the first opponent-turn hand prompt: Militia question 55 offered
  bare card names and the mapper produced `(2, 0, 1)`, but the actuator treated
  every unknown bare-name question as a high-z-index revealed-card display.
  The first planned click was therefore Gold in the display region; the live
  Gold was a local-hand stack at z-index 2000–2999, so resolution found zero
  targets and aborted before clicking any card.

  Bare-name hand prompts now explicitly include Militia, Chapel, Cellar,
  Poacher, Remodel trash, Artisan topdeck, the Moat reaction window, and the
  other base-set hand selectors. Duplicate offered copies remain separate
  protocol indices but deliberately produce repeated clicks on one collapsed
  physical stack. Card-body clicks use the recorded stack center, outside the
  distinct lower-right `.all-button`, so “All” cannot turn one intended click
  into an unintended bulk selection.

  Multi-select effects now always finish with the recorded wide primary canvas
  (`Confirm Discarding` / `Confirm Trashing`), including when the mapped answer
  has already reached the prompt's terminal count. Before that final click the
  actuator reads the recorded `<selection-cross>` marker and selected-stack
  counter, and requires the selected card multiset to exactly match the mapped
  labels. A mismatch waits briefly for rendering and then aborts without
  confirming. The same primary-canvas mapping handles empty decline/skip
  submissions such as declining a Moat reaction; selecting Moat itself maps to
  its hand stack and auto-submits.

  Actuation failures now report the offered tuple, the complete resolved target
  plan, the failing step, `not-found` versus `click-error`, and the observed
  target/button facts. Offline regressions cover the saved duplicate-selection
  DOM (three selected Silvers in one counted stack), confirmation versus Undo
  canvas geometry, ambiguity rejection, three repeated Copper clicks on one
  logical stack, the exact Militia plan, Moat reveal/decline plans, and replay
  of the live-game-2 archive through question 55.

- **Third supervised live-game follow-up (2026-07-24).**
  The third run reached turn 9 and correctly resolved 22 decisions before the
  Library per-drawn-Action mode prompt exposed a batching boundary error.
  The native Library interpreter presents the same player/kind/source
  signature for each successive card, but the client exposes each choice as a
  separate `CHOOSE_MODE` question with exactly one answer. The provider now
  treats `CHOOSE_MODE` and non-complex exact-one prompts as one-action client
  questions before applying its engine-signature batching heuristic. The next
  identical prompt is therefore resynced from its observed outcome and planned
  independently.

  The action mapper now rejects an action plan that produces the wrong fixed
  answer arity, so an over-batched mode plan cannot reach the browser. Offline
  regression coverage replays the archived third game through Library
  questions 64 and 65 with the live provider on both prompts, verifies the
  repeated-signature unit case, and rejects the formerly emitted two-answer
  mapping. The tracker also normalizes Library's transient `zone-type-24` as
  player set-aside state, which lets the replay continue from question 64 to
  65. Provider regressions retain batching for Militia, Cellar, Chapel, and
  Sentry trash/discard/topdeck; the existing reference and game-2 replay
  goldens cover their client mappings.

- **Fifth supervised live-game follow-up (2026-07-24).**
  Game 4 (id 181363348) completed end-to-end. Game 5 (id 181363699) then
  exposed a false-positive verifier abort at action question 92: two identical
  Witches appeared as separate offered indices but one collapsed browser stack.
  The click submitted index 0 while the client resolved index 1; both encodings
  map to the same engine Witch play and the observed `Play`/`Attack` events
  confirmed that outcome. `IntendedAction` now retains the complete set of
  mapper-enumerated encodings for the submitted gesture, and every immediate
  and settled resolution check accepts only a member of that set. The expected
  downstream `Play`/`Buy` event verification remains unchanged. Offline
  regression coverage replays the archived game with the recorded submitted
  index and continues through its available terminal frames; unit coverage
  retains aborts for an out-of-set answer and a mismatched question index.

- **Lobby automation and chat silence (2026-07-24).**
  `src/v2/arena/fsm/lobby.py` now runs the supervised inter-game flow in the
  existing headful session: homepage → search → matched table/direct game →
  game loop → game-over → dismiss end-game modal → leave table → requeue. It
  has explicit HOMEPAGE, SEARCHING, TABLE_WAITING, IN_GAME, GAME_OVER, and
  LEAVING states; all state timeouts and the per-session limit are
  config-driven (`0` means requeue until Ctrl-C). A live protocol pump stays
  alive across games while each game is handed to a fresh loop/tracker and
  receives its own archive.

  Every lobby selector is tied to the reference capture:

  - `button.automatch-button[ng-click="$ctrl.automatch.searchNow()"]` is
    the **Start search** button in `dom-34066.html` and `dom-4341034.html`.
  - `score-table-buttons button.lobby-button[ng-click="$ctrl.readyClick()"]`
    is the recorded hosted-table start control (labelled **Ready** by client
    2.2.8) in `dom-530325.html`. This remains specific to hosted-table flow;
    automatch does not depend on it.
  - `game-area` is the in-play board signal, present in mid-game
    `dom-3978828.html` and `dom-2848087.html` but absent from the table-waiting
    `dom-530325.html`. Automatch goes directly from search to this board; the
    apparent **Start game** modal there is an in-feed question, not a lobby
    control. That table snapshot does contain `game-chat`, so chat is only a
    secondary diagnostic signal.
  - `score-table-buttons button.lobby-button[ng-click="$ctrl.leave()"]` is
    **Leave Table** in `dom-530325.html`.
  - `game-ended-notification modal-window button.lobby-button[ng-click="$ctrl.ok()"]`
    is **Ok** in the blocking game-ended modal in `dom-4326014.html`. Decoded
    inbound message 14 (`GameEnd`) is the authoritative game-over signal; the
    modal is dismissed and confirmed gone before the recorded `Leave Table`
    score-page control is clicked. If that modal is absent after its short
    config-driven wait, the FSM proceeds only when `Leave Table` is already
    actionable; otherwise it raises a loud `LobbyError`, retains the browser
    for the operator, and never guess-clicks.

  The bot now has no chat-send path: the game-start announcement/config was
  removed and the kingdom gate resigns silently.  Inbound `Chat` events remain
  decoded and archived as before.

  While SEARCHING or TABLE_WAITING, the live session archives a labeled DOM
  snapshot at state entry. A terminal missing-control resolution captures a
  second labeled snapshot before raising its `LobbyError`; these use
  `page.content()` only and do not touch the WebSocket/event feed.

- **Homepage cold-start hardening (2026-07-25).**
  The 20260725T012508.958883Z lobby failure was a logged-in, connected client
  whose automatch button had not yet rendered, rather than a wrong-page or
  disconnected state. The HOMEPAGE timeout is now 90 seconds. At its halfway
  point the FSM performs one reload only if there is no visible modal content
  and no rendered reconnecting/connecting/loading component; their persistent
  empty Angular shell tags do not count. A rendered reconnecting/loading state
  is allowed to recover without a reload, while rendered `reconnecting-failed`
  triggers that single reload immediately. The original terminal DOM capture
  and loud `LobbyError` remain intact.

  The reload retains WebSocket capture without a session change: the
  `__arenaFrame` exposed function belongs to the persistent browser context,
  and the context-level `add_init_script` reruns `ws_hook.js` in the new
  document before its page scripts create sockets. Its callbacks therefore
  continue to append to the same `frames.jsonl` mirror and frame queue. P0's
  recorder uses that same context-level hook/binding arrangement and tracks
  navigation on its existing page; the streaming parser treats the new socket
  `open` as a reconnect/session reset while preserving parser state.

- **Automatch start-handshake fix (2026-07-24).**
  The stalled game 181367084 archive shows `GameStart`, then local
  `question-411` (`CHOOSE_MODE`, exactly one `card-mode-*`, exactly one
  required answer), then the opponent's resolution and `GameEnd`, with no
  `TurnStart`. Working and reference captures resolve both seats' corresponding
  questions with `(0,)` before turn 1. The game loop now recognizes the local
  prompt from that full structure while no turn exists; the literal
  `question-411` localization key is corroborating evidence, not a dependency.
  It submits option zero directly without building or searching a native
  shadow game, records the decision, and verifies the ensuing
  `DecisionResolved` immediately.

  Actuation first uses the existing `CHOOSE_MODE` canvas path. A single visible
  mode canvas is clicked directly; if other game canvases are visible, exactly
  one wide primary canvas is accepted. Failure captures a labeled live DOM
  snapshot and raises `ActuationError` with the selector, visible button facts,
  and snapshot result. The live entrypoint supplies the session DOM callback.

  Any other local `PendingDecision` before turn context now produces a
  screenshot-backed divergence abort instead of the former silent `continue`.
  Lobby automatch likewise hands off as soon as `game-area` appears and never
  waits for a text **Start game** lobby button; only the recorded hosted-table
  **Ready** path remains. The three reference games contain one local start
  handshake each, so replay actuation increases from 929 to 932 gestures while
  the 932/932 paired recorded-answer mappings remain reconciled (the one
  terminal unanswered Sentry prompt is still not actuated).

- **Multi-step gesture re-render fix (2026-07-24).**
  Two later live games reached Militia's discard prompt with the correct
  Estate/Gold + Copper + Copper + Confirm plan, but stopped on the second
  Copper. After each selection the client re-renders `div.card-stacks`; the
  remaining unselected Copper stack is briefly non-clickable and may be a new
  DOM node. The actuator now retains the ordered logical target plan for error
  reporting while resolving each physical card/button immediately before its
  click. Every resolution polls for up to three seconds by default for the
  live predicate: an unselected, clickable region-qualified card stack, or a
  present, enabled button canvas.

  Selection clicks in confirmable multi-card prompts are additionally checked
  one at a time. The actuator requires the selected-copy count for that
  identity to equal the number of preceding logical clicks, clicks the current
  unselected sibling, then waits for the count to advance by exactly one. A
  no-effect click is retried once with a fresh DOM resolution; a second
  no-effect or an over-selection aborts with the original structured
  question/plan/step diagnostics. The exact duplicate accounting comes from
  recorded `dom-4326014.html`: selected cards are separate direct stack
  siblings carrying `<selection-cross>`, and one selected Silver sibling's
  visible numeric counter encodes all three selected Silvers while the
  remaining unselected Silver is another sibling. The existing exact
  selected-multiset check remains the final gate before Confirm.

  Offline fake-page regressions cover the one-poll non-clickable animation,
  Estate + Copper + Copper + Confirm completion, a one-retry no-effect
  failure, and replacement of the Copper node between clicks. Both archived
  Militia aborts still replay to their original correct four-step logical
  plans.

- **Occluded duplicate-hand-stack follow-up (2026-07-25).**
  Game 181368037 reproduced the second-Copper Militia failure after the
  per-step polling fix. The operator screenshot established that the state was
  permanent occlusion rather than animation: the selected Copper is popped
  upward and rotated across the top of the remaining Copper stack, whose lower
  card body stays visible.

  The recorded selected-state DOM provides the exact geometry. In
  `dom-4326014.html`, the unselected Silver hand sibling is
  `111.803px × 176.4px` at
  `translateX(466.197px) translateY(536.4px) rotateZ(0deg)`, while the
  selected three-Silver sibling is the same size at
  `translateX(455.017px) translateY(492.3px) rotateZ(-10deg)`—44.1px upward
  and 11.18px left. `dom-3631985.html` records the same layout for Market:
  unselected at `(399.496px, 536.4px)` and selected at
  `(388.315px, 492.3px) rotateZ(-10deg)`. The remaining stack's distinct
  `.all-button` occupies only the lower-right rectangle
  `left:79.1071px; top:150.94px; width:30.0762px; height:21.3444px`.

  Hand-stack resolution now hit-tests center, bottom-center, bottom-left, and
  two lower-third points with `document.elementFromPoint()`. A point is usable
  only when the hit is the target stack or an ordinary descendant; the
  distinct `.all-button` is explicitly excluded. Center remains first for the
  normal path, while bottom-center is the first occlusion escape because the
  recorded 44.1px upward pop leaves the card bottom exposed and it avoids the
  lower-right All control. The chosen verified point is passed to Playwright
  as a target-relative position. If Playwright still reports interception,
  the hand-only fallback issues mouse down/up at the already hit-tested
  viewport coordinate. Failure diagnostics list every point and the topmost
  element, stack identity, and selected state found there.

  Fake-page regressions permanently cover the remaining Copper's center while
  leaving its bottom exposed, exercise the coordinate-dispatch fallback, and
  verify Estate + Copper + Copper + Confirm completes. A fully covered variant
  fails at step 3 with all point/cover identities in the structured error.

- **Undo auto-deny and bounded tracker recovery (2026-07-25).**
  The failed game 181368463 in
  `exports/arena/20260725T002942.738597Z/` established that granting an
  opponent undo rewinds server state while the arena tracker is normally
  forward-only. Server message 35 is now decoded as three big-endian 32-bit
  fields, `[metagame-kind][player-seat][decision-index]`. Client bundle 2.2.8
  defines kind 0 as an undo request, 1 as a timeout offer, 2 as undo denied,
  and 3 as undo cancelled. The three game-181368463 frames are request
  `(0, 0, 21)`, cancellation `(3, 0, 21)`, and request `(0, 0, 39)`;
  the latter is followed by the recorded outbound grant and rewound
  FullState. Message 35 no longer remains unknown in that archive.

  `undo.auto_deny` is true in `configs/arena.json`, defaults true, and rejects
  false: granting is not a supported configuration. Bundle 2.2.8 supplies the
  exact negative control
  `undo-request modal-window button.lobby-button[ng-click="$ctrl.decline()"]`.
  Recorded DOM snapshots contain the undo component shells but no rendered
  request, so the actuator also has a guarded generic-button fallback confined
  to `undo-request modal-window`. It accepts exactly one visible button whose
  `ng-click` handler or label is negative and never selects a grant/accept
  handler. Ambiguity, absence, or click failure captures a labeled DOM
  snapshot, logs a loud warning, makes no click, and leaves the request to
  expire without aborting the game. A live modal monitor similarly snapshots
  each unrecognized modal fingerprint once per visible occurrence.

  A non-replacement mid-game FullState mismatch still raises `TrackerError` by
  default. The sole exception is a mismatch within both 30 seconds and 64
  normalized events of an unresolved undo request; a decoded denial or
  cancellation closes that window immediately. That FullState uses the
  existing reconnect replacement machinery as authoritative state, clears any
  pending question, and emits resync evidence. The game loop then invalidates
  the native shadow, turn reconstruction history, pending actuation, and
  provider multi-question state before the next decision. It logs at critical
  severity and writes an explicit `UndoResync` record into the per-game
  archive. The original failed archive now replays through game end; the
  identical FullState with undo signals removed still trips the original
  zone-count mismatch.

- **Overnight mode (2026-07-24).**
  The live process is now restart-safe for an external supervisor. The game
  loop runs a configurable stall watchdog (`stall_watchdog_seconds`, default
  120). It is armed only while the local client owes progress: either the
  current turn belongs to our seat, or a local `PendingDecision` remains
  unresolved (including opponent-turn attacks and the pre-game start
  handshake). Ordinary opponent-turn silence does not arm it. A timeout
  captures a screenshot and DOM snapshot and writes `stall.json` in the game
  archive with the frame index, last game-relevant event timestamp, elapsed
  silence, tracker summary, and pending-question detail.

  Server message 14 now emits a result-bearing `GameResult` end event instead
  of a bare `GameEnd`. Client bundle 2.2.8 defines its layout as:

  ```
  GameFinished =
    TableDetails
    GameResult {
      tableId: long
      gameId: long
      ratingType: optional enum
      emptyPiles: CardName[]
      playerResults: PlayerResult[]
      autoContinue: boolean
    }
    continueAllowed: boolean
    matchCompleted: boolean

  PlayerResult =
    playerId: int
    rank: int
    score: Score { totalPoints: int, usedTurns: int, parts: ScorePart[] }
    cardNames: CardFrequency[]
    resignIndex: int
    resignationType: optional enum
  ```

  The parser consumes the result tail exactly, maps player IDs back to seat
  order, and reports per-seat VP, placings, the unique winning seat, or a tie.
  Rank is authoritative: archive
  `exports/arena/20260724T220259.308785Z/` decodes games as:

  - 181364095: seats 0–1 scored 33–44, ranks 2–1 (our win).
  - 181364271: 25–39, ranks 2–1 (our loss).
  - 181364518: 27–28, ranks 2–1 (our loss).
  - 181364739: 3–3, ranks 2–1 (our win by the server result).
  - 181364786: 3–3, ranks 1–2 (our loss by the server result).

  All five VP tuples match the final per-seat `points` counters in the feed.
  Archive `exports/arena/20260724T214053.394532Z/` additionally decodes
  completed game 181363348 as 45–30, ranks 1–2 (our seat 1 lost); its following
  game 181363699 was the documented actuation abort and is not added to the
  completed-game ledger. If a future message-14 payload is unfamiliar, the
  parser logs an error, emits an unknown result while preserving the end
  marker, and never lets ledger bookkeeping crash the session.

  Each completed game's `result.json` includes our seat, opponent, outcome
  (`win`, `loss`, `tie`, or `unknown`), scores, placings, winning seat, and tie
  flag. Completion also appends one compact JSON object to the cross-session
  `exports/arena/record.jsonl`; the session prints its running W-L-T tally.

  Process exit codes are a supervisor-facing contract:

  | Code | Meaning |
  | ---: | --- |
  | 0 | Clean end: configured game cap reached, or Ctrl-C |
  | 1 | Unexpected crash/startup failure |
  | 3 | Stall-watchdog abort |
  | 4 | Tracker, bridge, verification, mapping, or actuation divergence |
  | 5 | Lobby FSM error/timeout |

  Every non-zero live path first writes one-line `exit.json` in the run
  directory (`exit_code`, `exit_reason`, `game_id`, and `archive_dir`) and then
  reaches `ArenaSession.stop()`, which closes the persistent browser context
  and Playwright before process exit. Divergence and lobby paths no longer
  wait indefinitely with the profile locked.

  `configs/arena.json` now sets `max_games_per_session` to `0`, retaining the
  established meaning “unlimited until Ctrl-C.” `scripts/arena_overnight.sh`
  launches the process under `caffeinate -is`; restart looping remains the
  responsibility of the outer orchestrator.

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
- Committed on branch `v2-phase1`: P0–P2 (`a2780de`), P4 (`a450bd3`), P3
  follows.
- The web-server pytest suite intermittently deadlocks on this Mac inside
  torch's OpenMP threads at import (predates the arena stream; hung runs from
  July 10/14 observed). Workaround: `OMP_NUM_THREADS=1` when running
  `src/v2/web/server/test_server.py`.
