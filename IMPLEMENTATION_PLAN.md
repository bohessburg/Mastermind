# DominionZero v2: Detailed Implementation Plan

Companion to `REFACTOR_PLAN.md` (the architecture rationale). This document is
the build order: concrete types, module contracts, task breakdown per phase,
and acceptance criteria. Tasks are sized S (<½ day), M (½–2 days), L (2–5 days).

Conventions used throughout:
- All v2 code lives under `src/v2/` and `tests/v2/`; v1 is untouched except
  the three oracle bug fixes (Phase 2, task 2.8).
- Every task lands as its own commit/PR with tests; CI (build + tests +
  sanitizer job + bench smoke) must stay green.
- No task may introduce heap allocation into `core/` hot paths — enforced by
  the alloc-counter test (task 1.9).
- Cards removed in a set's 2nd edition (base: Adventurer, Chancellor, Feast,
  Spy, Thief, Woodcutter; Intrigue: Coppersmith, Great Hall, Saboteur, Scout,
  Secret Chamber, Tribute) are out of scope for v2 — do not implement them.

---

## 1. Core data model (`src/v2/core/`)

### 1.1 Fundamental types — `types.h`

```cpp
using DefId  = uint16_t;                 // index into global def table (~1000 entries)
using Slot   = uint8_t;                  // index into THIS game's def-slot table
using PlayerId = uint8_t;

constexpr int MAX_PLAYERS      = 4;      // training targets 2; engine supports 4
constexpr int MAX_SLOTS        = 64;     // distinct defs in one game (kingdom 10 +
                                         // basics 10 + mixed-pile members + non-supply
                                         // families: Knights=10, Loot=15, Prizes=5...)
constexpr int MAX_PILES        = 24;     // supply piles incl. basics
constexpr int MAX_NONSUPPLY    = 8;      // Horse/Loot/Spoils/Madman/... piles in play
constexpr int MAX_DECK_CARDS   = 160;    // per-player ordered zone caps (assert-guarded)
constexpr int MAX_EFFECT_DEPTH = 24;
constexpr int MAX_PENDING      = 24;     // parked duration/reserve frames per player
constexpr int MAX_LANDSCAPES   = 4;      // events/ways/projects/landmarks per game

struct Cost {
    int8_t  coins  = 0;
    int8_t  potion = 0;
    int16_t debt   = 0;
    // comparisons: "costs up to" is componentwise; "costs exactly" is equality.
    // NEVER compare raw ints — all comparisons go through these helpers.
    bool fits_within(const Cost& budget) const;
    bool operator==(const Cost&) const = default;
};
```

**Slot mapping.** A game touches ≤ MAX_SLOTS distinct defs. `GameState` holds
`DefId slot_to_def[MAX_SLOTS]` + a per-game inverse built at setup
(`Slot slot_of(DefId)` — binary search over a sorted copy, or a flat 2KB
table in the *setup* object, not the state). All zone count arrays are
indexed by Slot, keeping the state small and the encoder a linear scan.
Anything crossing the API boundary (actions, observations) speaks DefId.

### 1.2 Zones and player state — `state.h`

```cpp
struct OrderedZone {                     // deck, discard, boon/hex decks
    uint8_t  size = 0;
    Slot     cards[MAX_DECK_CARDS];      // [size-1] = top for deck
};

struct InPlayEntry {                     // in-play needs identity, not just counts:
    Slot     slot;                       //   printed card
    Slot     behaves_as;                 //   != slot for BoM/Overlord/Inheritance/Necromancer
    uint8_t  flags;                      //   via_way | duration_live | throne_held | ...
};

struct PlayerState {
    // Unordered zones: counts per slot
    uint8_t hand[MAX_SLOTS];
    uint8_t exile[MAX_SLOTS];
    uint8_t tavern[MAX_SLOTS];           // Reserve mat
    uint8_t island_mat[MAX_SLOTS];       // Island / Native Village / Archive set-asides
    // Ordered zones
    OrderedZone deck;
    OrderedZone discard;                 // ordered: top is public info, Harbinger browses it
    // In-play with identity
    uint8_t     in_play_size = 0;
    InPlayEntry in_play[24];
    // Parked frames: durations, reserve wake-ups, "at start of next turn"
    uint8_t     pending_size = 0;
    EffectFrame pending[MAX_PENDING];
    // Resources (persistent across turns unless noted)
    uint8_t coffers = 0, villagers = 0, favors = 0;
    uint8_t vp_tokens_lo = 0; uint8_t vp_tokens_hi = 0;   // uint16 split keeps struct packed
    uint8_t debt = 0;
    uint8_t journey_up : 1, minus_card : 1, minus_coin : 1, flags_pad : 5;
};
```

```cpp
struct Pile {
    Slot     base;                       // uniform contents
    uint8_t  count;                      // remaining copies (uniform part)
    uint8_t  mixed_len = 0;              // >0 → ordered mixed pile (Knights/Castles/split)
    Slot     mixed[12];                  // top = [mixed_len-1]
    uint8_t  trait = NO_LANDSCAPE;       // Trait attached to this pile
    uint8_t  embargo = 0;
    uint8_t  gain_counter = 0;           // Landmark bookkeeping (Obelisk/Aqueduct…)
    uint8_t  adv_tokens[MAX_PLAYERS];    // Adventures token bitmask per player
};
```

```cpp
struct GameState {                       // TRIVIALLY COPYABLE — static_assert enforced
    // --- identity/setup ---
    uint8_t  num_players;
    DefId    slot_to_def[MAX_SLOTS];
    uint8_t  num_slots;
    // --- board ---
    Pile     piles[MAX_PILES];           uint8_t num_piles;
    Pile     nonsupply[MAX_NONSUPPLY];   uint8_t num_nonsupply;   // Horse/Loot/Spoils…
    uint8_t  trash[MAX_SLOTS];           // counts (Rogue/Graverobber/Necromancer read it)
    PlayerState players[MAX_PLAYERS];
    // --- landscapes ---
    uint8_t  events[MAX_LANDSCAPES];     // landscape ids (separate table from cards)
    uint8_t  ways[MAX_LANDSCAPES];
    uint8_t  landmarks[MAX_LANDSCAPES];
    uint8_t  projects[MAX_LANDSCAPES];
    uint8_t  project_bought[MAX_LANDSCAPES];   // bitset per project, bit = player
    uint8_t  prophecy = NO_LANDSCAPE;    uint8_t sun_tokens = 0;
    uint8_t  artifact_holder[NUM_ARTIFACTS];   // player index or NONE
    OrderedZone boons, boons_discard, hexes, hexes_discard;
    // --- turn machinery ---
    uint8_t  phase;                      // Action/Buy(+Treasure merged)/Night/Cleanup/Over
    uint8_t  actions, buys;
    int16_t  coins;  uint8_t potion_coins;
    TurnQueue turn_queue;                // ring buffer of (player, turn_kind)
    uint16_t turn_counter;
    uint8_t  truncated : 1;              // hit MAX_TURNS — exposed in result for RL
    // --- suspended effects ---
    uint8_t     effect_depth = 0;
    EffectFrame effect_stack[MAX_EFFECT_DEPTH];
    PendingDecision decision;            // what step() is waiting for (see §3)
    // --- rng ---
    Xoshiro256pp rng;                    // 32 bytes, seeded at new_game
};
static_assert(std::is_trivially_copyable_v<GameState>);
```

Target `sizeof(GameState)` ≈ 8–16KB. Task 1.9 adds a static_assert ceiling
(`sizeof(GameState) <= 16384`) so growth is a conscious decision.

### 1.3 The def tables — `defs.h`

Two static tables built once at startup (constexpr where possible):

```cpp
struct CardDef {
    const char* name;                    // NEVER read by core/ — observer/UI only
    Cost        cost;
    uint16_t    types;                   // bitmask incl. Duration/Night/Reserve/Command/…
    int8_t      vp;                      // static VP; dynamic via score_hook
    int8_t      coin_value;              // basic treasures fast path
    EffectSpan  on_play;                 // span into the global instruction array
    EffectSpan  on_gain, on_trash, on_discard, on_reveal;   // trigger programs
    uint32_t    trigger_mask;            // which TriggerKinds this def listens to
    CustomStepFn custom;                 // nullptr for pure-DSL cards
    ScoreHookFn  score_hook;             // Gardens, Landmarks share the signature
};
struct LandscapeDef { /* name, kind(Event/Way/…), Cost, EffectSpan, trigger_mask, hooks */ };
```

Card registration is a static table per expansion file
(`cards/base.cpp`, `cards/seaside.cpp`, …), built with the `DEF(...)` macro
family from `REFACTOR_PLAN.md` §D3. No runtime registry mutation after init.

---

## 2. Effect interpreter (`core/interp.{h,cpp}`)

### 2.1 Instruction set

```cpp
enum class Op : uint8_t {
    // resources
    PlusCards, PlusActions, PlusBuys, PlusCoins, PlusCoffers, PlusVillagers,
    PlusFavors, PlusVP, TakeDebt, RepayDebtFree,
    // card movement (no decision)
    DrawTo,               // Library-class: draw until hand == arg (with set-aside variant → custom)
    GainSpecific,         // gain def arg to zone arg2 (Silver to discard, Horse, Loot…)
    TrashSelf, DiscardSelf, TopdeckSelf, ExileSelf, ReturnToPile,
    // choices — the ONLY ops that suspend
    Choose,               // zone, filter, min..max, Then::{Discard,Trash,Topdeck,Exile,
                          //   Reveal,SetAside,PutInHand,Play,Keep}
    ChooseGain,           // filter over supply/nonsupply, destination zone
    ChooseOption,         // k labeled options (yes/no, modes like Courtier/Pawn)
    ChooseOrder,          // reorder n known cards (deck put-backs)
    // control
    Attack,               // push per-opponent sub-frame with reaction window
    EachOtherPlayer,      // non-attack (Council Room draw)
    Repeat,               // Throne multiplier — re-run target program
    PerChosen,            // multiply next op by |last selection| (Cellar draw)
    IfElse,               // predicate id → branch to pc offsets
    EmitTrigger,          // publish to bus (rarely needed explicitly)
    CallCustom,           // transfer to def->custom
    End
};
struct Instr { Op op; uint8_t a, b, c; int16_t arg; };
```

Vocabulary discipline: an op is added **only when a second card needs it**
(tracked in a "DSL wishlist" doc during card porting). Everything else is
`CallCustom`.

### 2.2 Filters and predicates

`Choose`/`ChooseGain` reference a `Filter` value (POD): zone selector +
type-mask requirement + cost constraint relative to frame data
(`CostUpTo(frame.data[k] + n)` covers Remodel/Mine/Expand…). Filters are
evaluated by the engine to produce the option set; **card code never
enumerates options itself.**

### 2.3 Interpreter contract

```cpp
enum class RunResult { NeedDecision, Continue, FrameDone };
RunResult interp_run(GameState& s);      // executes until suspend or stack empty
void      interp_resume(GameState& s, Action a);   // decode + apply, then interp_run
```

- Suspension: `Choose*` ops fill `s.decision` (see §3) and return.
- `EffectFrame.data[]` layout is per-op documented; `pc` advances only after
  an op fully resolves (multi-select choices re-enter the same op until
  min/max satisfied or `pass`).
- `custom_step` functions get the same contract:
  `RunResult f(GameState&, EffectFrame&)` — they read/write only the state
  and their frame, so cloning is transparently safe.

### 2.4 Trigger bus — `core/triggers.{h,cpp}`

```cpp
enum class TriggerKind : uint8_t { OnGain, WouldGain, OnTrash, OnDiscard, OnShuffle,
    OnPlayAction, OnPlayTreasure, StartOfTurn, EndOfTurn, StartOfBuy, EndOfBuy,
    OnFirstPlay /*Merchant-class*/, OnCall, OnExile, ... };

struct Subscription { uint8_t owner; uint16_t source; uint8_t kind_of_source; };
struct TriggerTable {                    // rebuilt only when in-play/projects/prophecy change
    Subscription subs[64]; uint8_t count;
    uint8_t dirty : 1;
};
void emit(GameState& s, TriggerKind k, TriggerPayload p);
```

`emit` collects matching subscriptions; if more than one belongs to the same
player for the same window, it pushes a `ChooseOrder` decision (the rules
require player-ordered resolution) before pushing the trigger frames.
`WouldGain` runs before the gain finalizes so Watchtower/Trader can redirect
— the gain executor consults the payload's (possibly rewritten) destination.

**All gains/trashes/discards go through exactly one function each**
(`do_gain`, `do_trash`, `do_discard` in `core/moves.cpp`). These are the only
places triggers fire from and the only places zone counts change for those
verbs. v1's "dead hooks" failure mode is structurally impossible.

### 2.5 Turn scheduler — `core/turns.{h,cpp}`

Ring buffer of `(player, TurnKind)` where
`TurnKind ∈ {Normal, Outpost, Mission, Voyage, Possession, FleetFinal}`.
Cleanup pops the queue; if empty, pushes `(next_player, Normal)`. Extra-turn
rules ("no two consecutive extra turns") are enforced at enqueue time.
Game-end (Province/3-pile/prophecy conditions) is evaluated at the single
end-of-turn checkpoint; `truncated` flag set if `turn_counter` hits the cap.

---

## 3. Decisions and the action space (`core/actions.{h,cpp}`)

### 3.1 PendingDecision

```cpp
struct PendingDecision {
    PlayerId player;
    uint8_t  kind;            // PhaseAction | PhaseBuy | PhaseNight | Choose | ChooseGain
                              // | ChooseOption | ChooseOrder | ReactWindow | OrderTriggers
    uint16_t source;          // def or landscape asking (0 = phase itself)
    uint8_t  min_left, max_left;   // remaining picks for multi-select
    // option payload for ChooseOption/ChooseOrder (labels resolved by observer layer)
};
```

### 3.2 Global action id layout (compile-time offsets)

```
A_PASS            = 0
A_PLAY(d)         = 1                  + d          // as printed
A_WAY(w, d)  → encoded as A_PLAY_VIA_WAY_BASE + w*NUM_DEFS + d? NO —
                  ways in a game ≤ MAX_LANDSCAPES, so: A_WAY_BASE + way_slot*NUM_DEFS + d
A_BUY_PILE(d)     = A_BUY_BASE         + d          // by def of pile top
A_BUY_EVENT(e)    = A_EVENT_BASE       + e          // landscape table index
A_SELECT(d)       = A_SELECT_BASE      + d          // choose def d (hand/supply/reveal/discard/trash)
A_OPTION(k)       = A_OPTION_BASE      + k          // k < 16: modes, yes/no, order slots
A_CALL(d)         = A_CALL_BASE        + d          // Reserve call from tavern
A_SPEND_COFFERS / A_SPEND_VILLAGER / A_REPAY_DEBT / A_SPEND_FAVOR(op)
ACTION_SPACE_SIZE = A_END              // ≈ 3–5K slots; fixed for the NN forever
```

`ActionMask = std::bitset<ACTION_SPACE_SIZE>`. Two functions own the mapping:

```cpp
int  legal_actions(const GameState&, ActionMask& out);   // returns popcount
void apply_action(GameState&, Action a);                 // = step() body
```

Positional choices (ordered piles, deck reorders) use `A_OPTION(k)` slots —
the NN sees "position k", context comes from the encoder's decision block.

### 3.3 step() skeleton

```cpp
bool Game::step(GameState& s, Action a) {
    assert(legal(s, a));                       // debug builds
    dispatch(s, a);                            // phase move or interp_resume
    while (true) {
        RunResult r = interp_run(s);           // drains effect stack + triggers
        if (r == RunResult::NeedDecision) return false;
        if (advance_turn_machinery(s)) continue;   // phase/turn transitions may push frames
        break;                                     // stable: either new decision or game over
    }
    return s.phase == Phase::Over;
}
```

Auto-advance rule: if a decision would have exactly one legal action **and**
it is `A_PASS`, the engine takes it itself (configurable off for UI drivers).
This keeps MCTS trees free of forced nodes.

---

## 4. Encoder & bindings (`src/v2/encode/`, `src/v2/py/`)

- `encode(state, player, float* out)`: fixed layout documented in
  `encode/layout.md`, versioned (`OBS_VERSION`). Sections: per-slot zone
  counts (own), public opponent block, pile block (count/top/trait/tokens),
  landscape block, resource scalars, decision block (kind one-hot + source
  def embedding index + min/max).
- `py/module.cpp` (pybind11): `new_game(setup, seed)`, `legal_mask() -> numpy
  bool[A]`, `step(a)`, `encode() -> numpy float32[OBS]`, `clone()`, plus
  `BatchRunner(n_games)` that steps N games and yields stacked observations
  with the GIL released — the batched-inference surface for self-play.

---

## 4b. Web playtest harness (`src/v2/web/`)

Replaces the raylib GUI entirely. Purpose: a browser UI, dominion.games-shaped
but **text-only** (no card art — name, cost incl. potion/debt, types, and full
card text on styled tiles), usable for human-vs-bot and human-vs-human
playtests of the real engine.

### Architecture

```
web/server/   Python (FastAPI + uvicorn + websockets) on top of the Phase-4
              pybind11 bindings. Owns: sessions, seats, bot drivers,
              per-seat hidden-info filtering, action validation (re-checks
              the legal mask; the client is untrusted).
web/client/   Vite + React + TypeScript SPA. Stateless renderer of server
              messages; contains ZERO game logic.
```

Rationale for a Python server: it reuses the bindings that Phase 4 builds
anyway, playtest throughput is human-speed (no C++ server needed), and bot
seats / future NN agents plug in as Python callables over the action mask.

**The server is authoritative.** Per seat it sends only public info + that
player's own hand (deck order and opponent hands never leave the server —
same visibility contract as the observation encoder, reusing its filter).
Card display data (name/text/cost/types) is exported once from the def
table as generated JSON (`web/client/src/defs.gen.json`, task W.2) so the
client needs no round-trip per card.

### Wire protocol (JSON over one WebSocket per seat)

```jsonc
// server → client
{"type":"table",   "seats":[...], "kingdom":[defIds], "landscapes":[...]}
{"type":"state",   "view":{piles, myHand, myPlayArea, opponents:[{handCount,
                    deckCount, discardTop, inPlay, vp?}], trashTop, resources,
                    phase, turn}}                       // full redraw, sent on change
{"type":"decision","seat":0, "kind":"Choose", "source":{"def":123},
                    "prompt":"Militia: discard down to 3",
                    "options":[{"action":417,"label":"Discard Estate","def":45}],
                    "min":1,"max":1}
{"type":"log",     "lines":["P1 plays Militia", ...]}   // from the observer layer
{"type":"gameover","scores":[...], "winner":0, "truncated":false}

// client → server
{"type":"act", "action":417}
{"type":"undo_request"}        // human-vs-bot only: rewind via (seed, action log) replay
```

`prompt` and `label` strings come from the observer layer server-side — the
client never derives text from game state. Undo is free because a game *is*
(seed, action list): rewind = replay a prefix (v2 determinism makes this a
5-line feature; it was impossible in v1).

### UI layout (single screen, dominion.games-shaped)

- **Top:** supply grid — one tile per pile: name, cost badge (coin/potion/
  debt), remaining count, type-colored border (Action/Treasure/Victory/
  Curse/Attack/Reaction/Duration/Night); embargo/trait/token chips.
  Landscape row (Events/Ways/Projects/Landmarks/Prophecy + suns) when present.
- **Middle:** opponent strip (hand count, deck count, discard top, in-play,
  resources) · shared: trash top, current-turn resource bar
  (actions/buys/coins/potion/debt/coffers/villagers/favors/VP-tokens).
- **Bottom:** my in-play row, my hand (tiles), my mats (Exile/Tavern/Island)
  as collapsible trays.
- **Right rail:** game log (observer lines) + decision panel: the prompt and
  one button per legal option; **only legal actions are ever clickable**;
  multi-select shows picked-so-far with a Done button mapped to `A_PASS`.
- Hover/tap any tile anywhere → full card text popover. That popover is the
  whole "graphics" story.

### Phase W tasks (parallel with Phases 4–5; needs Phase 3's base set + 4.2 bindings)

| # | Task | Size |
|---|------|------|
| W.1 | Server skeleton: session create/join (URL token per seat), WebSocket lifecycle, seat config (human/bot per seat), authoritative act-validation loop | M |
| W.2 | Def-table JSON export (name/cost/types/text per DefId) generated at build time; observer-layer prompt/label endpoint | S |
| W.3 | Client scaffold (Vite/React/TS), protocol types, table/state rendering: supply grid, hand, in-play, opponent strip, resource bar | L |
| W.4 | Decision panel: option buttons from `decision` messages, multi-select flow, reaction-window and trigger-order prompts | M |
| W.5 | Game log rail + card-text popovers + type-colored tiles | S |
| W.6 | Bot seats (BM/EngineBot/random via action mask) + "thinking" pacing delay; human-vs-human via second seat token | M |
| W.7 | Undo (replay-prefix) for human-vs-bot; export game as (seed, action log) JSON — doubles as a golden-replay authoring tool | M |
| W.8 | Playtest checklist run: 10 full human games across kingdoms, every base-set decision type exercised; file bugs | M |

**Exit:** a human can play complete base-set games in a browser against a bot
and against another human; exported (seed, action log) files replay
identically in the test harness. Later mechanic phases (7–11) extend the UI
with the zones/chips already stubbed in W.3's layout (mats, landscape row,
token chips) — no structural rework.

---

## 5. Testing infrastructure (`tests/v2/`)

### 5.1 Card spec harness — `spec_harness.{h,cpp}` (build first, task 2.1)

Declarative fixtures; Catch2 underneath:

```cpp
CARD_SPEC("Mine trashes chosen treasure, gains to hand") {
    given().hand("Copper", "Estate", "Mine").deck("Silver");
    play("Mine");
    choose("Copper");                       // select-by-def, like the real action space
    gain("Silver");
    expect().hand_has("Silver").hand_lacks("Copper")
            .trash_has("Copper").coins(0);
    expect_conservation();
}
```

`choose()/gain()` translate names → DefId → action ids through the real
`legal_actions` path, so specs exercise the action space, not internals.

### 5.2 Invariants — `invariants.{h,cpp}`

`check_invariants(state, baseline)` asserted after **every** step in debug:
card conservation (modulo creators: Horses/Loot/Spoils tracked via a created-
counter), non-negative resources, debt rules (can't buy while in debt),
hand/deck sizes within caps, effect stack sanity (depth, players valid),
pile counts vs initial, no live frame referencing a dead source.

### 5.3 Suites

| Suite | Contents | Gate for |
|---|---|---|
| specs/ | per-card spec files, one per expansion | every card PR |
| conformance/ | per-mechanic packs from rulebook+wiki: duration×throne, way×trigger, cost-reduction stacking, trigger ordering, would-gain movers, extra turns | each mechanic phase |
| fuzz/ | seeded random-agent games, random kingdoms(+landscapes), invariants each step; ASan/UBSan job in CI | phases 3+ |
| differential/ | v1-vs-v2 base set: same kingdom + scripted decisions → same outcome | phase 2–3 |
| golden/ | (seed, action list) → state hash; regenerated only deliberately | forever |
| determinism | same seed+actions ⇒ identical hash, twice; clone-equivalence: step(clone) == step(original) | phase 1 |

### 5.4 Bench — `src/v2/bench/bench.cpp` (one binary, flags)

Measures: `step` ns (median/p99 over recorded decision stream), clone ns,
`legal_actions` ns, random-agent games/sec, BM-agent games/sec. Emits JSON;
CI compares against `bench/baseline.json` with ±15% regression gate.

---

## 6. Build & CI

- `CMakeLists.txt`: new targets `dominion_v2` (lib), `v2_tests`, `v2_bench`,
  `v2_py` (optional, pybind11 via FetchContent).
  Default `CMAKE_BUILD_TYPE=Release` with warning; `-DENABLE_SANITIZERS=ON`
  option (ASan+UBSan); LTO for Release; `-Wall -Wextra -Wpedantic
  -Wconversion` on **all** v2 targets including tests.
- GitHub Actions: {Release tests, Debug+sanitizer tests+fuzz-short, bench
  smoke} × {macOS, Linux}. Docker image from the `docker` branch work reused
  for the Linux job.
- `.gitignore`: `Testing/`, `build*/`.

---

## 7. Phase-by-phase task breakdown

### Phase 1 — Core skeleton (engine plays Big-Money-only games)

| # | Task | Size |
|---|------|------|
| 1.1 | `types.h`, `Cost`, xoshiro RNG + tests (distribution smoke, state size) | S |
| 1.2 | `state.h`: all structs above, static_asserts (trivially-copyable, size ceiling) | M |
| 1.3 | Def tables + `DEF` macro for vanilla ops; basic cards (Copper→Province, Curse, Potion, Platinum, Colony) | M |
| 1.4 | Setup: kingdom→slots, pile init (uniform), starting decks, shuffle | M |
| 1.5 | Phase machine + turn scheduler (Normal turns only) + cleanup + game-end + `truncated` | M |
| 1.6 | `legal_actions`/`apply_action` for phase decisions (play treasure, buy, pass); auto-advance rule | M |
| 1.7 | Interpreter loop for resource ops only (`PlusCards`… via vanilla DSL) | M |
| 1.8 | Determinism + clone-equivalence + conservation tests; random & BM driver bots | M |
| 1.9 | Alloc-counter test (override global new in test build; zero allocs across 1K games); size ceiling assert | S |
| 1.10 | Bench binary + baseline.json + CI wiring (incl. sanitizer job) | M |

**Exit:** seeded BM-vs-BM games run end-to-end; determinism/conservation/alloc
tests green; bench reports step/clone/mask/games-sec; CI green on both OSes.

### Phase 2 — Choice ops, attacks, triggers (all v1 level-1 cards)

| # | Task | Size |
|---|------|------|
| 2.1 | Spec harness (`CARD_SPEC`) | M |
| 2.2 | `Choose` op + Then:: executors + multi-select re-entry + `A_SELECT` mapping | L |
| 2.3 | `ChooseGain`, `ChooseOption`, `ChooseOrder`, `IfElse`, `PerChosen`, `Repeat` | L |
| 2.4 | `do_gain/do_trash/do_discard` choke points + trigger bus core + `OnFirstPlay` (Merchant) | L |
| 2.5 | `Attack` + reaction window (Moat) + per-opponent frames | M |
| 2.6 | Port 20 simple level-1 cards as DSL + specs (Cellar, Chapel, Village, Smithy, Militia, Witch, Mine, Remodel, Workshop, Bureaucrat, …) | L |
| 2.7 | Throne Room via `Repeat`; Vassal, Harbinger, Poacher, Merchant, Moneylender + specs | M |
| 2.8 | Fix v1 Sentry/Mine/Bureaucrat (oracle truthfulness) — 3 one-liners + v1 tests | S |
| 2.9 | Differential runner v1-vs-v2 (scripted-decision adapter) over base set | M |

**Exit:** all 26 level-1 cards on v2 with specs; differential green; trigger-
ordering window covered by a conformance test even though base set rarely hits it.

### Phase 3 — Custom steps, fuzz hardening (full base set)

| # | Task | Size |
|---|------|------|
| 3.1 | `custom_step` contract + Library (draw-to-7 with set-aside) + Sentry (uses ChooseOrder) | M |
| 3.2 | Remaining base-set cards + specs; Gardens via `score_hook` | M |
| 3.3 | Fuzz harness (random kingdoms, invariant checks) + CI sanitizer fuzz job (10M steps) | M |
| 3.4 | Golden replays (5 seeds); bench re-baseline; perf pass to ≥50K games/sec (profile: expected wins = branchless mask build, trigger-table caching) | L |

**Exit:** full base set; 10M fuzz steps clean under ASan/UBSan; ≥50K
games/sec random-agent; golden replays locked.

> **3.4 outcome (2026-07):** perf pass landed 15.6K→38.8K random-agent
> games/sec (BM 80K, step 42ns, mask 26ns). The 50K random-agent target was
> not met: profiling shows no remaining local hotspot — random games simply
> average ~440 decisions. Closing the gap would require coarsening the
> decision decomposition (changes the NN action space), rejected. Accepted
> deviation; revisit only if self-play throughput proves insufficient.

### Phase 4 — RL surface

| # | Task | Size |
|---|------|------|
| 4.1 | Encoder + `layout.md` + round-trip tests (encode(clone)==encode(orig)) | M |
| 4.2 | pybind11 module (game, mask, step, encode, clone) + wheel build in CI | M |
| 4.3 | `BatchRunner` (vectorized games, GIL-released stepping) + Python smoke test (random self-play throughput) | M |
| 4.4 | Imperfect-info determinizer: sample hidden zones consistent with public info + own knowledge (for MCTS later) | M |

**Exit:** Python drives seeded batch self-play; observation layout versioned.

### Phase 5 — Cutover

| # | Task | Size |
|---|------|------|
| 5.1 | Port BM/Heuristic/EngineBot decision logic onto action-mask interface | M |
| 5.2 | Port interactive TUI to `step()` + observer layer (string rendering lives here) | M |
| 5.3 | Delete v1 (`game/`, `engine/`, old harnesses, old tests) **and the raylib GUI (`src/gui/`, replaced by the Phase-W web client)**; keep golden/differential artifacts | M |
| 5.4 | Rewrite `docs/how-to-implement-cards.md` as DSL cookbook (op reference, filter reference, spec template, custom_step guide) | M |

**Exit:** single engine in tree; docs match the real API (v1's doc-drift
root cause addressed).

### Phase 6 — MCTS scaffold

| # | Task | Size |
|---|------|------|
| 6.1 | Arena-allocated tree (nodes reference GameState copies in a slab), PUCT select/expand/backprop | L |
| 6.2 | Determinized rollouts using 4.4; virtual-loss hooks for future batching | M |
| 6.3 | Eval harness: MCTS(1K sims, random rollouts) vs EngineBot, seat-swapped, 1K games | M |

**Exit:** MCTS beats EngineBot; per-move sim throughput reported by bench.

> **6.3 outcome (2026-07):** uniform-random rollouts are empirically
> uninformative in Dominion — MCTS(1K sims) lost ~200/200 to EngineBot AND
> BigMoney with a flat 100→3000-sim scaling curve, while beating RandomBot
> 40/0 with 37/40 truncations (search learns treasure-hoarding because
> greening never converts under random continuations; predicted by README).
> Gate revised: rollouts use a fast scripted policy (BigMoney-class), the
> standard card-game MCTS remedy; the NN value head later replaces rollouts
> entirely.
>
> **Heuristic-rollout rerun:** the scripted rollout fixed the finish-game
> pathology (RandomBot truncations 37/40 → 0/40) and made MCTS competitive
> with BigMoney (91/82/27 at 1K sims), but the Phase 6 EngineBot gate still
> failed (47/141/12 at 1K sims, random kingdoms). Remaining gap is strategy
> quality/action valuation, not tree correctness or terminal greening.
>
> **Action-playing rollout rerun:** one targeted rollout upgrade made playouts
> play Action cards before buying. Heuristic+actions vs EngineBot over 200
> random-kingdom seat-swapped games: K=8 64/128/8, K=4 83/109/8, K=2
> 98/94/8 (51.0% excluding ties). EngineLike rollouts, which add a trimmed
> EngineBot-style buy, cleared the gate decisively: K=8 110/77/13, K=4
> 110/73/17, K=2 123/64/13 (65.8% excluding ties). MctsConfig now defaults to
> EngineLike rollouts with K=2. Best-config vs BigMoney was 173/22/5.

### Phase T — Base-set NN training (inserted 2026-07; runs before Phase 7)

Goal: AlphaZero-style training on the base set to ≥ parity with EngineBot.
Hardware target: local M1 Max (measured: 1.2M NN evals/sec MPS @ batch 4096
on a 2.9M-param MLP; engine cost negligible). Bottleneck is batching
efficiency, not FLOPs.

| # | Task | Size |
|---|------|------|
| T.1 | Batched-leaf NN-MCTS: C++ `SelfPlayRunner` (N parallel games, per-game Mcts with virtual loss; `collect_leaves()` → stacked obs, `provide_evaluations(policy, value)` resumes search; emits finished-game records: per-decision obs, visit-count policy targets, outcome) + pybind surface | L |
| T.2 | Python training loop `src/v2/train/`: PyTorch MLP (masked policy over ACTION_SPACE + tanh value), MPS; replay buffer; self-play → train → checkpoint cycle from a config file; deterministic seeding | M |
| T.3 | Eval ladder: periodic checkpoint eval vs EngineBot/BigMoney (NN-MCTS at eval sims), win-rate tracking; gate = ≥50% vs EngineBot excl. ties, ≥200 random-kingdom seat-swapped games | M |
| T.4 | Training container: `Dockerfile.train` (CUDA base, builds dominion_v2_py, torch, entrypoint = training config), `--device auto` (cuda/mps/cpu), `--smoke` validation mode (~2 min), resume-from-checkpoint, artifacts (checkpoints/replay/metrics CSV) on a mounted volume | M |
| T.5 | Remote-run orchestration (`scripts/infra/`): provider tooling (vast.ai/RunPod CLI wrappers) to (a) search offers by GPU/vCPU/price and provision, (b) bootstrap the training image on the box, (c) launch a run and stream logs locally, (d) periodic + final checkpoint/metrics sync back to local `checkpoints/`, (e) status (GPU util, games/hr, latest eval), (f) teardown with artifact-sync guarantee + idle-cost guard. Orchestration stays LOCAL — the box is a disposable worker. API keys via env/keychain only, never in repo. | L |

**Exit:** a trained checkpoint beats EngineBot; the run is reproducible from
config + seed; a full run can execute on a rented GPU box end-to-end (provision
→ train → sync → teardown) driven from the local session.

**Status (2026-07-24): exit criteria MET; program paused between phases.**
T.1-T.4 shipped and battle-tested over campaigns 1-17 on rented vast.ai GPU
boxes (T.5 is manual-console + scripts rather than CLI wrappers — accepted).
Flagship: campaign 15 gen_0045, a card-token transformer that beats
EngineBot v3 (~56-62% at 400 sims) and took the first game off a human.
Full history in `docs/training-log.md`; resume state and next scoped moves
in `docs/session-handoff.md`.

### Phase T2 — Campaign roadmap c18–c21 (decided 2026-07-25)

Premise (from the human-games analysis, `docs/human-games-analysis.md`):
gen_0045 is a big-money bot that loses to thin engines; the failure is
plan-level exploration, not value accuracy. c17 is ABANDONED (its verdict no
longer matters). Standing policy from here: **no scripted opponents in the
training pool** — scripts are demoted to instruments (sentinels/eval probes)
only; c16 proved scripted pressure is inert as curriculum, and any script
caps the ecology at its author's understanding. Diversity is injected into
self-play instead. One lever per campaign; levers accumulate.

| Campaign | Lever | Notes |
|---|---|---|
| c18 | **Forced-opening / archetype-seeded self-play**, on obs-v3 | Both seats neural. Force/bias the first N turns' buys from diverse archetype templates (Chapel-thin, Village/Smithy/Lab, trasher-first, money control), then release the net; anneal forcing over the campaign. Rides on obs-v3, which requires from-scratch anyway. obs-v3 scope: (a) global trash section + tokenizer trash features; (b) **decision-semantics encoding** — select-semantic one-hot on the decision block (keep/discard/trash/topdeck/gain, from the DSL `Then::` payload + `DiscardDownTo`=keep) plus tokenizer-side embedding of the decision-source card, fixing the probe-confirmed Militia keep-inversion (policy points at junk on all `Choose` frames; see human-games-analysis). Pool: self-play + ancestor league of past *neural* checkpoints only. |
| c19 | **Network scale-up ~4x**: d192/3L/4H (1.5M) → d320/5L/8H (~6M) | Matches the measured 3-5x compiled-inference headroom (368-439K evals/s vs pipeline demand). From-scratch (widths change). Launch gate = c15 gate-2 protocol on-box: compiled+bf16 bench, accept if self-play games/hr ≥ ~70% of c18's. Stretch d384/6L (~10.6M, ~7x) only if the gate clears with margin. |
| pre-c20 gate | **Architecture audit + literature review** | Before c20 launches: thorough audit of CardTokenNet architecture and topology (token structure, head layout, depth/width balance, value-head design, tokenizer feature conditioning) plus a literature review of comparable set-structured / card-game / AlphaZero-family work (set transformers, pointer networks, OpenAI Five/AlphaStar entity encoders, published Dominion/deckbuilder AI). Output: written findings + any architecture changes to fold into c20 alongside its sampling-window lever. (Jack, 2026-07-27.) |
| c20 | **Extended high-temperature sampling window** | Dominion's "opening" is every buy decision all game; the early-move temperature cutoff is mis-transplanted from Go/chess. Keep τ=1 sampling on buy decisions much deeper (schedule TBD at launch). Warm-starts from c19. |
| c21+ | **Neural exploiter league** (AlphaStar-style) | Exploiter agents trained specifically to beat the current main agent, seeded from diverse forced-opening starts, feeding the main agent's opponent pool. Removes the scripted ceiling permanently. Largest infra lift; hold until c18-c20 reads are in. |

Instruments for every campaign: engine3 sentinel (money-mirror strength,
near-saturated), the "thinner" scripted sentinel (EVAL ONLY; Chapel-engine
exploiter profile), and the 113-competitive-human-game record set as a
"does it punish money" regression probe. Calibration caveat (2026-07-25):
c15 gen_0045 beats thinner 73-74% @400 sims across two builds (Chapel-money
and Chapel-engine variants), and thinner scores WORSE on Chapel boards than
its own average in scripted duels — script-quality piloting makes thinning
a net liability, so the scripted sentinel is a weak lower bound on the
thin-engine threat, not a faithful proxy for the humans who beat the bot
83% with the same archetype. Treat its reads as directional only; the
human-record probe and post-campaign arena runs are the true thin-engine
instruments until the c21 neural exploiters exist (which are the principled
fix for exactly this piloting gap). c18 mid-campaign health check: fraction of replay-buffer
games containing ≥3 trashes, and unforced trasher/village buy rates —
if these aren't moving by ~gen 15, the forcing schedule is wrong. Also
re-run the Militia keep-probe (headless rebuilt hands, `militia_probe.py`
pattern) against c18 checkpoints: the keep-inversion should disappear
once select semantics are encoded; Militia/Bureaucrat kingdom win rates
are the arena-side confirmation.

### Phase T3 — The c20 restructure (2026-07-28/30; supersedes T2's c20/c21 rows)

Premise (value-head engine probe + pre-c20 audit, `docs/training-log.md`
2026-07-28/30 entries): the money attractor is multi-layer — outcome-data
monoculture, saturated value-target geometry (±[0.8,1.0] only), search that
strangles unvisited lines (legal-mask bug + no forced exploration),
exploration that dies by turn ~5, and a clairvoyant training/eval regime
(perfect info + known future draws everywhere except web/arena). One-lever
campaigns kept failing because every layer independently favors money. T3
fixes the layers together; c21 (neural exploiter league) remains deferred.

Build order (all landed 2026-07-29/30 unless noted; each reviewed + full
suites green; results in the training log):

| # | Item | Status |
|---|------|--------|
| T3.1 | Honest-eval harness (`honest_eval.py`, serving-identical DecisionSearcher) + honest flagship re-baselines + honest sims curve | DONE — c15 44.9 / c19 47.7 vs engine3 @400/K2; honest depth flat 200–1600 |
| T3.2 | True legal-mask recording (kills the `policy>0` reconstruction bug) | DONE |
| T3.3 | Landscape-sentinel memset repair + encoder-generation guard + legacy shim (`--legacy-shim`; pre-fix checkpoints must never run bare on post-fix builds) | DONE — shim reproduces c15 probe means exactly |
| T3.4 | Determinized self-play (`selfplay.determinize: off\|per_decision\|per_turn`) + honest eval/duel modes (`--honest`) | DONE — regime A/B: ~7–11% cost, identical trajectories |
| T3.5 | Corpus quarantine + human-tuple exporter (111 real games → 12,734 human-seat tuples w/ raw margins) | DONE |
| T3.6 | Imitation stack: BC pretrain + persistent anchor (floored, never annealed to zero) + optional AWR + `offline_fit --human-tuples` | DONE |
| T3.7 | Probe suite as standing instruments (`scripts/probes/`, run_all) | DONE — new finding: Militia keep-inversion persists in c18/c19 |
| T3.8 | Value-target geometry sweep (α ∈ {.6,.4,.2,0}, 3 seeds, exact margin inversion) | DONE — α=0.0 uniquely seed-stable, engine-parity pricing; Duchy probe = full-scale non-regression gate |
| T3.9 | Forced playouts + policy-target pruning (KataGo) | DONE — config-gated, default on for c20 |
| T3.10 | Per-seat decision-kind temperature schedule (`temp_mode: per_seat_buy`, buys τ=1 through turn ~14) | DONE |
| T3.11 | Duel-validated engine curriculum pools (`configs/kingdom_pools_c20.json`; sentry_engine 94.5%, thin_engine 88.4% vs BM; draw-engine board REJECTED at 51.3%) | DONE |
| T3.12 | Aux margin-distribution value head (KataGo decomposed targets) | in flight |
| T3.13 | AdamW default (config-surfaced, legacy-safe resume) | DONE |
| T3.14 | Serving-stack search parity (K=2 costs ~3.6 pts; ~6 pts residual vs EvalRunner suspected in within-decision batching; free deploy strength) | OPEN |
| T3.15 | Data collection: Jack local sessions (gold), server auto-export on gameover + VPS redeploy, arena value-only tuple pipeline (blocked on regime flip) | OPEN |
| T3.16 | `run_c20.json` assembly + pre-registration (Jack-gated; template/SIL disposition memo in task #20) | OPEN |

Measurement doctrine after T3.1/T3.4: track BOTH bars — serving-harness
honest (deployed config; c15 44.9 / c19 47.7 / duel 45.6) and
EvalRunner-honest (~54). Clairvoyance itself is worth ~0 vs engine3
(matched A/B); the old eval-vs-serving gap was stack quality, not
information. The c19-era 1600-sim regression was a clairvoyant-regime
artifact; honest depth is neutral and the value head is the binding
ceiling.

### Phase 7 — Durations & mats (Seaside/Adventures core mechanics)

| # | Task | Size |
|---|------|------|
| 7.1 | Pending-frame lifecycle: park at resolution, wake at window, cleanup skip, Throne-held durations (`throne_held` flag keeps the Throne out too) | L |
| 7.2 | Duration conformance pack (wiki rulings: Throne×Duration, discard timing) | M |
| 7.3 | Seaside card wave + specs (Fishing Village, Wharf, Caravan, Lighthouse—attack immunity via trigger, Outpost—extra turns for real, Island—mat, Native Village) | L |
| 7.4 | Reserve/tavern: `A_CALL` actions, wake windows (Coin of the Realm, Ratcatcher, Royal Carriage×Duration) | L |
| 7.5 | Adventures tokens on piles (+1 Card/Action/…, −Cost, Trash) wired into `effective_cost` and play resolution | M |

### Phase 8 — Token economies & alternate costs

| # | Task | Size |
|---|------|------|
| 8.1 | Coffers/Villagers micro-actions (`A_SPEND_*`, legal any time appropriate) + Guilds wave | M |
| 8.2 | VP tokens + Prosperity wave (Monument, Bishop, Goons — stacking trigger test) | M |
| 8.3 | Debt: buy-with-debt, repay action, can't-buy-while-owing; Empires wave 1 | M |
| 8.4 | Potion costs end-to-end + Alchemy wave | M |
| 8.5 | Projects (per-player standing triggers) + Renaissance wave + Artifacts (holder transfer) | L |

### Phase 9 — Landscape wave 1

| # | Task | Size |
|---|------|------|
| 9.1 | Events: `A_BUY_EVENT`, once-per-turn flags, Menagerie/Adventures/Empires event packs | L |
| 9.2 | Landmarks: score hooks + gain counters (Obelisk, Aqueduct, Tomb triggers) | M |
| 9.3 | Ways: `A_WAY` actions, frame flag, way×on-play-trigger conformance (Way of the Mouse: set-aside card!) | L |
| 9.4 | Exile zone + Menagerie card wave (gain-to-exile, discard-from-exile trigger) | M |
| 9.5 | Fuzz with landscapes in kingdom sampling | S |

### Phase 10 — Landscape wave 2

| # | Task | Size |
|---|------|------|
| 10.1 | Night phase (legal-action wiring, phase machine already has the slot) + Nocturne wave (Heirlooms in setup, spirits nonsupply) | L |
| 10.2 | Boons/Hexes: shared ordered decks + discards, receive-timing conformance | L |
| 10.3 | Allies + Favors (spend windows per Ally) + Liaisons | L |
| 10.4 | Traits (pile modifiers → trigger subscriptions) | M |
| 10.5 | Prophecies + sun tokens (Rising Sun) — global state flip machinery | M |

### Phase 11 — Long tail

Remaining expansions card-by-card (Intrigue/Cornucopia/Hinterlands/Dark Ages —
Ruins/Shelters/Knights mixed piles; Plunder — Loot deck; Empires split piles),
then the monsters in order: Inheritance → Band of Misfits/Overlord
(`behaves_as` conformance pack) → Stash (shuffle decision) → Possession
(possessed TurnKind + gain-redirect) → Black Market (side deck, def-table
pressure) last. Each card: DSL-first, spec required, fuzz kingdom pool grows
automatically from the def table.

---

## 8. Sequencing notes & risk register

- **Critical path to training** is Phases 1→4 (≈ 6–9 weeks of focused work).
  Phases 5–6 can overlap with early NN work; 7–11 grow the pool while
  training on base-set kingdoms already runs.
- **Phase W (web harness) runs parallel with 4–5** and is worth starting as
  soon as 4.2's bindings exist: human playtests are the best correctness
  probe for decision flows that bot fuzzing exercises blindly (prompts,
  multi-select, reaction windows), and W.7's exported games feed the golden-
  replay suite.
- **Biggest technical risk:** trigger-ordering windows interacting with the
  effect stack (nested `emit` during a trigger's own resolution). Mitigation:
  the bus enqueues, never recurses — frames only pushed by the interpreter
  loop; conformance pack written *before* the first triggered card (2.4).
- **Second risk:** action-space size vs NN head. Locked constants
  (`ACTION_SPACE_SIZE`, `OBS_VERSION`) from Phase 4 on; any change bumps the
  version and invalidates checkpoints — treat as breaking.
- **Perf risk:** trigger-table rebuilds on every in-play change. Mitigation:
  dirty-flag + rebuild lazily on first emit; most cards have empty
  trigger_mask so the table stays tiny in base kingdoms.
- **Scope control:** DSL ops added only on second use; `custom_step` is not
  a code smell, it's the pressure valve that keeps the DSL small.
