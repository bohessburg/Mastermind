# How To Implement Cards

This is the v2 card cookbook. The v2 engine is table-driven: card definitions
live in `src/v2/cards/base.cpp`, public card metadata lives in
`src/v2/core/defs.h`, and the interpreter in `src/v2/core/interp.cpp` executes
small POD instructions. Card code must keep `GameState` memcpy-cloneable and
must not allocate in core paths.

`REFACTOR_PLAN.md` and `IMPLEMENTATION_PLAN.md` remain the architecture source
of truth. This file is the practical checklist for adding cards.

## Files To Touch

| File | Purpose |
|---|---|
| `src/v2/core/defs.h` | Def ids, type bits, `Op`, `Filter`, `Then`, `CardDef` |
| `src/v2/cards/base.cpp` | Static card table, effect programs, filters, score hooks, custom steps |
| `src/v2/observe/card_text.cpp` | Web/observer card text only, never core logic |
| `tests/v2/test_specs_base.cpp` | CARD_SPEC coverage for card behavior |
| `tests/v2/fuzz_harness.cpp` | Add the card to fuzz kingdoms when it is playable |
| `tests/v2/golden/` | Regenerate only when a deliberate behavior change affects golden games |

## Definition Pattern

Every card has a `DefId` in `defs.h` and one `CardDef` entry in
`kBaseCards`. The table order must match the ids exactly.

Use the simple macros for cards with no hooks:

```cpp
DOMINION_V2_DEF(Copper, (Cost{0, 0, 0}), TYPE_TREASURE, 0, 1)
DOMINION_V2_DEF_EFFECT(Village, (Cost{3, 0, 0}), TYPE_ACTION, 0, 0, SPAN_VILLAGE)
```

Use an explicit `CardDef` when the card has a multi-word name, trigger,
`custom` function, or `score_hook`:

```cpp
CardDef{
    "Throne Room",
    Cost{4, 0, 0},
    TYPE_ACTION,
    0,
    0,
    SPAN_THRONE_ROOM,
    NO_EFFECT,
    NO_EFFECT,
    NO_EFFECT,
    NO_EFFECT,
    0U,
    nullptr,
    nullptr,
}
```

The fields are:

| Field | Meaning |
|---|---|
| `name` | Stable display/test name |
| `cost` | `Cost{coins, potion, debt}`; comparisons must use `fits_within` |
| `types` | Bitmask of `TYPE_ACTION`, `TYPE_TREASURE`, `TYPE_VICTORY`, etc. |
| `vp` | Static VP for Victory/Curse cards |
| `coin_value` | Treasure fast path value |
| `on_play` | Effect span in `kEffectInstrs` |
| `on_gain`, `on_trash`, `on_discard`, `on_reveal` | Trigger spans |
| `trigger_mask` | OR of `trigger_mask(TriggerKind::...)` |
| `custom` | `RunResult (*)(GameState&, EffectFrame&)` for stateful cards |
| `score_hook` | Per-card scoring hook, e.g. Gardens |

## Effect Programs

Effect programs are `constexpr Instr kEffectInstrs[]`. A span is an offset and
length:

```cpp
constexpr EffectSpan SPAN_MARKET{62, 5};

Instr{Op::PlusCards, 0, 0, 0, 1},
Instr{Op::PlusActions, 0, 0, 0, 1},
Instr{Op::PlusBuys, 0, 0, 0, 1},
Instr{Op::PlusCoins, 0, 0, 0, 1},
Instr{Op::End, 0, 0, 0, 0},
```

`Instr` fields are intentionally small:

```cpp
struct Instr {
    Op op;
    uint8_t a;
    uint8_t b;
    uint8_t c;
    int16_t arg;
};
```

Prefer the DSL for anything expressible as resources, choices, gains, attacks,
or small control flow. Add a new op only after the behavior is likely to be
reused by a second real card. Use `custom_step` for genuinely stateful flows
such as Library and Sentry.

## Op Reference

Resource ops:

| Op | Args | Semantics |
|---|---|---|
| `PlusCards` | `arg=count` | Draw cards for `frame.player`, reshuffling with game RNG |
| `PlusActions` | `arg=count` | Add actions, saturating uint8 resources |
| `PlusBuys` | `arg=count` | Add buys |
| `PlusCoins` | `arg=count` | Add coins |
| `PlusCoffers` | reserved | Declared for later expansions |
| `PlusVillagers` | reserved | Declared for later expansions |
| `PlusFavors` | reserved | Declared for later expansions |
| `PlusVP` | reserved | Declared for later VP-token cards |
| `TakeDebt` | reserved | Declared for debt cards |
| `RepayDebtFree` | reserved | Declared for debt cards |
| `DrawTo` | reserved | Declared for draw-to-N patterns |

Movement and gain ops:

| Op | Args | Semantics |
|---|---|---|
| `GainSpecific` | `a=GainDestination`, `arg=DefId` | Gain a specific supply card if present |
| `GainCurse` | `a=GainDestination` | Gain Curse if the pile is nonempty |
| `TrashSelf` | none | Trash the source card from in-play via `do_trash` |
| `DiscardSelf` | reserved | Declared for later self-discard effects |
| `TopdeckSelf` | reserved | Declared for later self-topdeck effects |
| `ExileSelf` | reserved | Declared for later exile effects |
| `ReturnToPile` | reserved | Declared for later return effects |
| `DiscardDeckTop` | none | Discard top deck card, recording the revealed def |
| `PlayLastFromDiscard` | none | Play the def recorded by `DiscardDeckTop` from discard |
| `PlayChosenRepeated` | `arg=times` | Play the last chosen card from hand repeated `times` |
| `BanditAttack` | none | Bandit reveal/trash/discard payload |

Choice ops:

| Op | Args | Semantics |
|---|---|---|
| `Choose` | `a=FilterId`, `b=min`, `c=max`, `arg=Then` | Suspend for `A_SELECT(def)` or `A_PASS` once the clamped minimum is met |
| `ChooseGain` | `a=FilterId`, `b=min`, `c=max`, `arg=GainDestination` | Choose matching supply pile top and gain it |
| `ChooseOption` | `a=count` | Suspend for `A_OPTION(k)` in `[0, count)` |
| `ChooseOrder` | `a=count` | Repeated options for ordering known cards |

Attack and multi-player ops:

| Op | Args | Semantics |
|---|---|---|
| `Attack` | `a=absolute payload offset` | Push one attack frame per opponent in turn order, with reaction windows |
| `DiscardDownTo` | `arg=target hand size` | Opponent chooses cards to keep; the rest are discarded |
| `EachOtherPlayer` | `a=absolute payload offset` | Push a non-attack frame per opponent |

Control ops:

| Op | Args | Semantics |
|---|---|---|
| `Repeat` | `a=target pc`, `b=count` | Re-run a program range, keeping its counter separate from choice data |
| `PerChosen` | none | Multiply the next simple resource op by `last selection count` |
| `IfElse` | `a=PredicateId`, `b=true pc delta`, `c=false pc delta`, `arg=predicate arg` | Branch within the current span |
| `EmitTrigger` | reserved | Declared for later explicit trigger effects |
| `CallCustom` | reserved | Declared for later hybrid DSL/custom cards |
| `End` | none | Complete the current frame |

Predicates currently available: `AlwaysFalse`, `AlwaysTrue`, `ChosenAny`,
`LastOptionEqualsArg`, `CoinsAtLeastArg`, and `LastChosenIsAction`.

## Filters

Filters are static POD rows in `kFilters`:

```cpp
Filter{
    ZoneSelector::Supply,
    TYPE_TREASURE,
    CostLimitKind::LastChosenPlus,
    Cost{},
    3,
    ANY_DEF,
    ANY_DEF,
}
```

Fields:

| Field | Meaning |
|---|---|
| `zone` | `Hand`, `Supply`, or `Discard` |
| `type_mask` | Required card types; zero means any type |
| `cost_kind` | `None`, `Fixed`, or `LastChosenPlus` |
| `max_cost` | Fixed componentwise cost limit |
| `coin_delta` | Coin delta for cost-relative gains |
| `exact_def` | Match only this def, or `ANY_DEF` |
| `exclude_def` | Reject this def, or `ANY_DEF` |

Choice availability is clamped to the currently available matching cards. If a
mandatory choice has no legal selection, the op completes instead of deadlocking.

`Then` executors for `Choose` are:

| Then | Effect |
|---|---|
| `Discard` | `do_discard(..., MoveZone::Hand/Discard)` |
| `Trash` | `do_trash(...)` |
| `Topdeck` | Move to deck top |
| `Exile` | Reserved |
| `Reveal` | Reserved/no-op until a card needs persistent reveal state |
| `SetAside` | Reserved for future shared set-aside flows |
| `PutInHand` | Gain/return to hand where supported |
| `Play` | Used by Throne Room to play a selected Action |
| `Keep` | Used by keep/reorder flows |

## Examples

Vanilla card:

```cpp
constexpr EffectSpan SPAN_VILLAGE{7, 3};

Instr{Op::PlusCards, 0, 0, 0, 1},
Instr{Op::PlusActions, 0, 0, 0, 2},
Instr{Op::End, 0, 0, 0, 0},
```

Choice card:

```cpp
// Chapel: trash up to 4 cards from hand.
Instr{Op::Choose, FILTER_HAND_ANY, 0, 4, static_cast<int16_t>(Then::Trash)},
Instr{Op::End, 0, 0, 0, 0},
```

Cost-relative gain:

```cpp
// Remodel: trash one card, then gain up to +2 coins to discard.
Instr{Op::Choose, FILTER_HAND_ANY, 1, 1, static_cast<int16_t>(Then::Trash)},
Instr{Op::IfElse, static_cast<uint8_t>(PredicateId::ChosenAny), 1, 2, 0},
Instr{Op::ChooseGain, FILTER_SUPPLY_LAST_PLUS_2, 1, 1,
      static_cast<int16_t>(GainDestination::Discard)},
Instr{Op::End, 0, 0, 0, 0},
```

Attack:

```cpp
// Militia: +2 coins, then each opponent discards down to 3.
Instr{Op::PlusCoins, 0, 0, 0, 2},
Instr{Op::Attack, 36, 0, 0, 0},
Instr{Op::End, 0, 0, 0, 0},
Instr{Op::DiscardDownTo, 0, 0, 0, 3},
Instr{Op::End, 0, 0, 0, 0},
```

Trigger:

```cpp
CardDef{
    "Merchant",
    Cost{3, 0, 0},
    TYPE_ACTION,
    0,
    0,
    SPAN_MERCHANT,
    SPAN_MERCHANT_ON_FIRST_PLAY,
    NO_EFFECT,
    NO_EFFECT,
    NO_EFFECT,
    trigger_mask(TriggerKind::OnFirstPlay),
    nullptr,
    nullptr,
}
```

The trigger bus rebuilds subscriptions from in-play cards and enqueues trigger
frames. It must never call the interpreter recursively.

Score hook:

```cpp
int16_t gardens_score(const GameState& state, PlayerId player, DefId) noexcept;

CardDef{
    "Gardens",
    Cost{4, 0, 0},
    TYPE_VICTORY,
    0,
    0,
    NO_EFFECT,
    NO_EFFECT,
    NO_EFFECT,
    NO_EFFECT,
    NO_EFFECT,
    0U,
    nullptr,
    gardens_score,
}
```

Score hooks are called by `score()` over all owned zones.

## Custom Step Guide

Custom steps are for cards with stateful, card-by-card suspension that would
make the DSL harder to audit than the card text. Current examples are Library
and Sentry.

Contract:

- Signature: `RunResult fn(GameState&, EffectFrame&) noexcept`.
- Return `NeedDecision` only after filling `state.decision`.
- Return `Continue` when the interpreter should call the same frame again.
- Return `FrameDone` when the frame is complete.
- Store all progress in `EffectFrame` fields or existing POD zones.
- Do not allocate and do not hold pointers into `effect_stack` across a
  decision.
- Any visible temporary cards must be counted by invariants. Prefer
  `PlayerState::set_aside` for set-aside cards; Bandit-style reveal slots are
  acceptable only when the invariant helper accounts for them.
- Trash/discard/gain through `do_trash`, `do_discard`, and `do_gain`.

Frame data discipline:

- Document every `frame.data[]` slot used by a complex helper near the helper.
- Keep choice-owned slots separate from control-owned slots. `Repeat` has its
  own counter slot and must not share with active choice bookkeeping.
- Reset temporary slots before finishing the frame if they represent cards
  outside normal zones.

## Spec Harness

Every card needs one or more `CARD_SPEC` tests. The harness drives the real
legal action and `Game::step` path; helper calls fail loudly if the scripted
action is illegal.

```cpp
CARD_SPEC("Remodel trashes then gains up to plus two") {
    given().hand("Remodel", "Estate");

    play("Remodel");
    choose("Estate");
    gain("Duchy");

    expect().trash_has("Estate").discard_has("Duchy");
    expect_conservation();
}
```

Useful helpers:

| Helper | Purpose |
|---|---|
| `given().hand(...)`, `.deck(...)`, `.discard(...)` | Set player 0 zones |
| `given().player_hand(p, ...)` | Set another player's zone |
| `play(name)` | Sends `A_PLAY(def)` |
| `choose(name)` / `gain(name)` | Sends `A_SELECT(def)` |
| `option(k)` | Sends `A_OPTION(k)` |
| `pass()` | Sends `A_PASS` |
| `empty_supply(name)` | Empty one pile for edge cases |
| `expect().hand_has(name)` etc. | Assert zone/resource results |
| `expect_conservation()` | Assert total card count baseline |

Add edge cases for empty filters, exact/min/max choices, gain destination, clone
equivalence at suspensions, and card conservation whenever a card sets aside or
reveals cards.

## Porting Checklist

1. Read the rules text and add/confirm observer text.
2. Write the CARD_SPEC first, including conservation.
3. Decide whether the card is DSL, DSL plus hook, or custom.
4. Add the `DefId`, span/filter rows, and `CardDef`.
5. Route all gains, trashes, and discards through move choke points.
6. Add trigger masks or score hooks when required.
7. Add the card to fuzz/conservation kingdom lists.
8. Run Release `v2_tests`.
9. Regenerate goldens only when the behavior change is intended and reviewed.

