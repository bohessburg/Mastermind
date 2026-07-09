# DominionZero v2: Ground-Up Refactor Plan

Goal: rebuild the engine so that (1) new cards are cheap to implement and test,
(2) MCTS/AlphaZero training is a first-class citizen, and (3) the hot path
(step, clone, legal-move enumeration) is allocation-free and cache-friendly.

**Scope commitment: the architecture must support every card in every
expansion, including all landscape types** — Events, Landmarks, Projects,
Ways, Traits, Allies, Prophecies, Boons/Hexes, Artifacts, and States — plus
the mechanics they ride on: Durations, Reserve/Tavern mat, Exile, Coffers,
Villagers, Favors, Debt, Potion costs, VP tokens, Night phase, extra turns,
split/mixed piles, non-supply piles (Horses, Loots, Spoils), and Heirlooms.
Implementation is phased by mechanic (§7), but **no data structure ships that
would have to be replaced to get there.** Every "simplification" below was
checked against the full card pool first.

The v1 engine stays in-tree during the rewrite and is used as a differential-
testing oracle for the base set. Nothing is deleted until v2 beats it on
correctness and speed.

---

## 0. The three decisions that drive everything

### D1. Invert control flow: the engine is a state machine, not a game loop

v1 runs games imperatively (`GameRunner::run()` calls into agents via
`DecisionFn` callbacks). Mid-card state lives on the C++ call stack, so state
cannot be cloned at a decision point — which is exactly where MCTS needs to
clone.

v2 exposes the RL-standard interface and nothing else:

```cpp
struct Game {
    static GameState new_game(const Setup& setup, uint64_t seed);
    // Setup = kingdom piles + landscapes (events/ways/prophecy/...) + options

    // Who must act, and what may they do. Never allocates.
    DecisionRequest current_decision(const GameState& s);   // player, choice kind, source
    int legal_actions(const GameState& s, ActionMask& out); // bitmask over global action space

    // Advance by exactly one decision. Returns terminal flag.
    bool step(GameState& s, Action a);

    // Cloning is trivial because GameState is a flat value type (see D2).
    // GameState b = a;   <-- this is the entire clone API
};
```

All "agents" (bots, humans, GUI, MCTS) become drivers that loop
`while (!terminal) { a = pick(legal); step(s, a); }`. GameRunner, DecisionFn,
and the agent-callback plumbing are deleted, not ported.

**How card effects suspend:** an explicit **effect stack** inside GameState
(not coroutines — coroutine frames are heap-allocated and not memcpy-able).
Each in-flight effect is a small POD frame:

```cpp
struct EffectFrame {          // fixed-size, memcpy-safe
    uint16_t source;          // card/landscape def whose effect is running
    uint8_t  pc;              // step index within that def's effect program
    uint8_t  player;          // whose effect
    uint8_t  flags;           // via-Way, via-Throne (multiplier), duration-pending, ...
    int16_t  data[8];         // scratch: counts, revealed defs, choices so far
};
```

`step()` runs an interpreter loop: execute effect steps until one needs a
player decision, then return with `current_decision` populated. Throne Room /
King's Court re-push the target's frame with a multiplier; attacks push one
frame per opponent (reaction window first).

Full-expansion additions designed in from day one:

- **Frames can outlive the turn.** Durations (Seaside onward) are frames
  parked in a per-player `pending[]` list keyed to a wake-up window
  ("start of my next turn", "start of each of your turns", "when you next
  gain", Hireling's "for the rest of the game"). Cleanup skips cards with
  live frames. This one mechanism covers Durations, Reserve tavern-mat
  wake-ups (Coin of the Realm, Teacher), Archive/Crypt set-asides, and
  Prophecy/Project standing effects.
- **A trigger bus, not ad-hoc hooks.** Every state transition (`gain`,
  `trash`, `discard`, `shuffle`, `play`, `phase_start/end`, `turn_start/end`,
  `would_gain` for movers like Watchtower/Trader) publishes to a registry of
  subscriptions derived from what's in play / bought Projects / active
  Prophecy / Traits on piles. When several triggers fire at once, the owner
  orders them — that ordering is itself a `DecisionRequest` (the rules
  require this; Dominion is full of "resolve in any order" windows).
- **A turn scheduler, not a modulo counter.** Outpost, Mission, Voyage,
  Possession, and Fleet grant extra/modified turns with constraints; the
  state holds a small queue of `(player, turn_kind)` entries instead of
  `current = (current+1) % n`.

### D2. GameState is a flat, fixed-size value type (zero heap, memcpy clone)

v1 carries `std::string` names per card instance, `unordered_map<string,...>`
mats, `std::function` members, per-player mt19937s. v2:

- **Cards are `uint16_t` def ids** (the full pool is ~750 cards + ~200
  landscape defs; uint8 is not enough). No heap identity. Where a card must
  temporarily *be* something else — Band of Misfits, Overlord, Inheritance,
  Necromancer, Ways — the zone entry carries `(def, behaves_as)`; where a
  pile-wide modifier applies — Traits, Inheritance, Ferry/Training tokens —
  it lives on the **pile**, not the copies.
- **Zones are counts where order can't matter, ordered arrays where it can.**
  Hand, discard-as-multiset… no: discard *tops* are visible and Harbinger-
  class effects browse it, but discard order is never hidden information —
  so: hand, in-play-counts, Exile, most mats = `uint16_t counts[]`; deck =
  ordered `uint16_t deck[]`; discard = ordered array too (cheap, and makes
  "top of discard" and Counting-House-style queries exact).
- **Supply piles are ordered stacks of def ids**, not counts. Split piles
  (Sauna/Avanto, Catapult/Rocks), Knights, Ruins, and Castles are mixed,
  ordered, and top-visible; a uniform pile is just a stack where every entry
  matches. Pile struct also carries: Trait id, embargo/token slots
  (per-player Adventures tokens: +Card/+Action/+Buy/+Coin/−Cost/Trash),
  and gain counters (Obelisk-class Landmarks).
- **Per-player resource block:** actions, buys, coins, **Coffers, Villagers,
  Favors, Debt, Potion coins, VP tokens**, plus journey-token orientation and
  −Card/−Coin tokens. All flat ints.
- **Global landscape block:** selected Events (buyable any time in Buy
  phase), Landmarks (scoring hooks), Projects (per-player bought bitset),
  Ways (one active), active Prophecy + sun-token count (Rising Sun),
  Boon/Hex decks as ordered arrays with discard piles (they shuffle),
  Artifact holders (player index per artifact), States per player.
- **One RNG for the whole game**, seeded at `new_game`: xoshiro256++.
  (seed, action history) fully determines a game — also the debugging story.
- **No strings, no std::function, no vectors anywhere in GameState.**
  With full-expansion zones, `sizeof(GameState)` lands around 8–16KB rather
  than v1-base's 2–4KB — still a single memcpy, still arena-friendly.
  Kingdom-dependent capacity (which mats exist, boon decks, etc.) is bounded
  at compile time; unused blocks are dead weight we accept for copyability.

Names, card text, and logging live in the static `CardDef` table and a
separate optional observer layer (see §5) — the sim core never touches them.

**Costs are a triple.** `Cost {coins, potion, debt}` everywhere, with
`effective_cost(state, def)` applying Bridge/Highway/Ferry-token/Quarry-class
reductions. v1's raw `card->cost` comparisons are banned from day one.

### D3. Cards are data + tiny effect programs, not free-form lambdas

The v1 bug class ("decide() returns indices, card treated it as values" —
Sentry/Mine/Bureaucrat) exists because every card hand-rolls its own decision
plumbing. v2 makes the common cases declarative and the plumbing impossible
to get wrong:

```cpp
// A large fraction of the pool is vanilla bonuses — one line, zero code:
DEF(Village,    cost(3), action(), plus_cards(1), plus_actions(2));
DEF(Laboratory, cost(5), action(), plus_cards(2), plus_actions(1));

// The rest compose a small vocabulary of effect ops with typed choice steps:
DEF(Cellar, cost(2), action(), plus_actions(1),
    Choose(FromHand, Any, UpTo::All, Then::Discard),   // engine owns index math
    DrawPerChosen());

DEF(Mine, cost(5), action(),
    Choose(FromHand, IsTreasure, Exactly(0,1), Then::Trash),
    GainToHand(IsTreasure, CostUpToTrashedPlus(3)));

DEF(Militia, cost(4), action(), plus_coins(2),
    Attack(EachOpponent, DiscardDownTo(3)));

// Landscapes use the SAME def table and the SAME interpreter:
DEF(Expedition, event(), cost(3), NextHand(+2));                 // Event
DEF(WayOfTheOx, way(),   plus_actions(2));                       // Way
DEF(Citadel,    project(), cost(8), OnFirstActionEachTurn(ReplayIt()));
DEF(Obelisk,    landmark(), ScoreHook(PerCardInChosenPile(2)));
```

- Effect ops are enum-tagged POD instructions executed by the D1 interpreter.
  A choice op *always* receives validated selections resolved by the engine —
  card code never sees raw indices, so the Sentry/Mine/Bureaucrat bug class
  cannot be written.
- Every `DecisionRequest` automatically carries `source` (card *or*
  landscape def) + choice kind + the op that asked — NN and UI get full
  context for free.
- **Ways are alternative effect programs** attached at play time: playing an
  Action offers "play as printed" or "play via Way X"; the frame's `flags`
  records which program runs. Prophecies/Projects/Traits/Landmarks register
  trigger-bus subscriptions or score hooks — they need no new machinery
  beyond D1's bus.
- **Escape hatch:** genuinely weird cards (Library, Sentry, Black Market,
  Possession) implement `custom_step(GameState&, EffectFrame&) ->
  NeedDecision|Done` in the def table. Same interpreter contract, still
  cloneable because all their state lives in the frame. Budget expectation
  from surveying the pool: ~70% pure DSL, ~25% DSL + one custom op,
  ~5% fully custom.
- The DSL grows **only** when a second card needs the same op — vocabulary
  is driven by the pool, not speculation. Ops like `DiscardDownTo` are
  implemented once, tested once, reused by every card that needs them.

---

## 1. Global action space (fixed, NN-ready)

One global enumeration, sized at compile time over the full def table,
stable across games and kingdoms:

```
[0]                pass / done with phase / decline
[play  d]          play card def d (as printed)
[way   d]          play current Action via Way d          (Ways in setup)
[buy   p]          buy from pile p / buy Event e / buy Project e
[select d]         select card def d (hand/supply/reveal/discard choices)
[option k]         option k: yes/no, keep/discard/trash, trigger-order slots,
                   boon/hex option picks, token placements
[call   d]         call Reserve card d from tavern mat
[spend]            Coffers / Villager / Favor / debt-repay micro-actions
```

- `legal_actions()` writes a bitmask (`std::bitset<ACTION_SPACE>`); the NN
  policy head has one logit per slot and masks illegal moves. Full-pool size
  is a few thousand slots — comparable to chess (~4.7K) and fine for a
  policy head.
- Because hands are count arrays, "select a card from hand" is a choice over
  *defs present*, not hand positions — smaller space, and positional
  symmetry collapses transpositions in the tree for free. Ordered-zone
  choices (deck reorders, mixed-pile picks) select by position via `option k`.
- Multi-select choices (Cellar, Chapel, Militia) are decomposed into repeated
  single selections + `pass`, so every decision is exactly one action.
  Trigger-ordering windows (D1) are just another `option` decision.

## 2. Observation encoder

A separate module (not inside the sim core):

```cpp
void encode(const GameState& s, int perspective_player, float* out);  // fixed size
```

Channels: own hand/deck/discard/in-play/Exile/mat counts per def, opponent
public info (deck size, discard top, VP tokens, in-play, mats, Debt/Coffers/
Villagers/Favors), supply counts + pile tops + Traits/tokens per pile,
landscape state (Events available, Projects bought, active Prophecy + suns,
Way), turn/phase/resources scalars, and the current `DecisionRequest`
(source def one-hot + choice kind). Hidden information (opponent hand, own
deck order) is *excluded*; determinization for imperfect-info MCTS is a
training-loop concern, and the seeded RNG + count-based zones make sampling
determinizations trivial.

## 3. Testing strategy (rebuilt with the engine, not after)

1. **Table-driven card specs.** One declarative fixture per card:
   given zones/counts → script of actions → expected zones/counts. Specs are
   data; adding a card's tests means adding rows, not writing plumbing.
   Target: no card merges without a spec. Landscape defs use the same format.
2. **Card-conservation invariant** checked after *every* step in debug builds
   (total cards across all zones + supply + trash + set-asides constant,
   modulo explicit creators like Horses/Loot): the assert that would have
   caught v1's Sentry duplication instantly. Compiled out in Release.
3. **Property/fuzz harness:** N seeds × random-agent games over random
   kingdoms *including landscapes*, asserting invariants each step (counts,
   non-negative resources, Debt rules, phase legality, effect stack sanity,
   no orphaned duration frames). Runs under ASan/UBSan in CI.
4. **Mechanic conformance suites:** per-mechanic test packs written from the
   rulebook + wiki rulings (Duration×Throne, Way×on-play triggers,
   cost-reduction stacking, trigger-ordering windows, extra-turn limits).
   These are the expansion-readiness gate, not per-card tests alone.
5. **Differential testing vs v1** for the base set (fix v1's Sentry/Mine/
   Bureaucrat first so the oracle is truthful) + **golden replays**:
   (seed, action list) → expected final state hash, forever.
6. **Determinism test that is real:** same seed + same actions ⇒ identical
   state hash, twice. (v1's version of this test checks nothing.)

## 4. Performance plan

Budgets (Release, M-series laptop, 2-player kingdoms):
- `step()`: **zero heap allocations**, < 200ns typical decision.
- Clone: memcpy of 8–16KB, < 1µs.
- Random-agent full games, base kingdoms: **> 100K games/sec** single-threaded
  (v1: 12K). Landscape-heavy kingdoms will be slower; the bench suite tracks
  both.
- Legal-action mask: < 100ns base, < 500ns with landscape checks.

How: no strings/allocs in the sim core; counts live in L1; single xoshiro
RNG; interpreter is a switch over enum ops (no std::function dispatch);
trigger bus is a fixed subscription table rebuilt only when in-play/Projects
change. Benchmarks measure what MCTS pays — step, clone, and mask latency,
plus games/sec — one binary, in CI with regression thresholds.
CMake: default Release with warning, `-DENABLE_SANITIZERS`, LTO, identical
warning flags for tests and engine.

## 5. Layering

```
src/v2/
  core/     GameState (POD), effect interpreter, trigger bus, turn scheduler,
            step/legal/clone, RNG          ← no I/O, no strings
  cards/    CardDef table, effect DSL, custom_step impls,
            one file per expansion (incl. its landscapes)
  encode/   observation encoder, action-space mapping
  drivers/  bots (ported BM/Heuristic/EngineBot), interactive TUI
  web/      browser playtest harness: game server (seats, sessions, per-seat
            hidden-info filtering) + text-only SPA client — replaces the
            raylib GUI entirely (human-vs-bot and human-vs-human)
  observe/  optional event observer: turns step transitions into readable logs
  bench/    step/clone/mask/games-per-sec benchmarks
tests/v2/   card specs, mechanic conformance packs, invariants, fuzz,
            differential-vs-v1, golden replays
```

GUI/interactive modes consume the same `step()` API as MCTS. The observer
layer reconstructs "P1 plays Smithy, draws 3" from state diffs + the effect
stack; the core never formats a string.

The web client is deliberately dominion.games-shaped but text-only: no card
art, just name/cost/types/text rendered as styled tiles. Because every
decision already arrives as an enumerated `DecisionRequest` with legal
actions, the UI is a thin renderer — it never contains game logic, so
playtesting the UI *is* playtesting the engine and action space.

## 6. Known monsters (named early so they don't ambush us)

- **Possession** — you make decisions for another player's turn with gains
  redirected: handled by the turn scheduler (`turn_kind = possessed`) +
  a gain-redirect flag; decisions already carry `player` explicitly.
- **Black Market** — a hidden ordered side deck of otherwise-out-of-kingdom
  cards; fits the non-supply-pile slot but inflates the def table in play.
  Custom_step + its own zone; deliberately scheduled last.
- **Stash** — you choose its position during a shuffle: shuffle becomes a
  decision point when Stash is present (rare-path custom op).
- **Inheritance / Band of Misfits / Overlord** — covered by `behaves_as`
  (D2); the conformance pack for "card identity vs ability source" gates
  their expansion.
- **Trigger-order windows** — the single most pervasive full-pool
  correctness risk; built into the bus from phase 1, not retrofitted.

## 7. Migration sequence (each phase lands green on CI)

| Phase | Deliverable | Exit criterion |
|-------|------------|----------------|
| 1 | Core skeleton: GameState POD (full-scope layout: uint16 defs, Cost triple, resource block, ordered piles), phases, treasures/buy/cleanup, vanilla DSL, step/legal/clone, seeded RNG, trigger bus + turn scheduler stubs | Big-Money games run; determinism + conservation tests pass; bench reports step/clone/games-sec |
| 2 | Choice ops + effect stack: Choose/Gain/Trash/Discard ops, attack+reaction window, engine-invoked triggers | All 26 v1 level-1 cards on v2; specs + differential-vs-v1 green |
| 3 | Custom-step escape hatch (Library, Sentry, Throne chains); fuzz under sanitizers | Full base set; 10M fuzz steps clean; ≥ 50K games/sec |
| 4 | RL surface: observation encoder, action mask API, pybind11 bindings, batched self-play runner | Python runs seeded self-play at target throughput |
| W | **Web playtest harness** (runs parallel with 4–5): WebSocket game server over the step API, text-only browser client, human-vs-bot and human-vs-human | A human completes full base-set games in the browser; bugs filed from playtests |
| 5 | Port bots + TUI driver; delete v1 + GameRunner/DecisionFn **+ raylib GUI (replaced by web client)**; rewrite how-to-implement-cards as a DSL cookbook | v1 removed; benchmarks published |
| 6 | MCTS scaffold: arena-allocated tree, PUCT, determinization hook — NN pluggable | MCTS-random beats EngineBot at 1K sims/move |
| 7 | **Durations + mats** (Seaside/Adventures core): persistent frames, tavern mat, Reserve calls, extra-turn scheduler for real | Seaside conformance pack green |
| 8 | **Token economies** (Prosperity/Renaissance/Guilds): VP tokens, Coffers, Villagers, Projects, Artifacts, Debt + Potion costs (Empires/Alchemy) | Mechanic packs green; encoder extended |
| 9 | **Landscape wave 1**: Events, Landmarks, Ways (Menagerie), Exile | Random-kingdom fuzz incl. landscapes clean |
| 10 | **Landscape wave 2**: Boons/Hexes (Nocturne, Night phase), Allies/Favors, Traits, Prophecies (Rising Sun) | Full landscape fuzz; encoder covers all landscape state |
| 11 | Remaining expansions card-by-card, hardest monsters (§6) last | Full pool implemented, conformance suites green |

Rules for the migration: no new cards on v1 (its level-2 file doesn't compile
anyway); v1 bug fixes only to keep the differential oracle truthful. Phases
1–6 make the engine *trainable*; 7–11 grow the pool without ever changing
the state layout, action space shape, or step API — that stability is the
whole point of designing for the full pool up front.

## 8. What we deliberately drop from v1

- Per-instance card ids and name-string tables (replaced by uint16 def ids +
  `behaves_as` where identity genuinely diverges).
- `DecisionFn`, `GameRunner`, ActionOption label strings.
- `TurnFlag` card-specific entries (Merchant/Sentry state moves into effect
  frames; Merchant becomes a trigger-bus subscription).
- Per-player mt19937s and `std::random_device` seeding.
- Hardcoded Merchant logic in the runner, duplicate stress harnesses
  (one bench binary with flags), dead on_* hooks (replaced by the engine-
  invoked trigger bus).
- Raw `card->cost` integer comparisons (replaced by the Cost triple +
  `effective_cost`).
