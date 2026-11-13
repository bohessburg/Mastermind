# Implemented Cards

The v2 engine is the only runtime engine. The 2nd-edition base kingdom is
complete: 26/26 kingdom cards are registered, tested with CARD_SPEC coverage,
included in fuzz kingdoms, and available to the web/TUI/Python surfaces.

Second-edition removed cards such as Woodcutter and Feast are intentionally out
of scope for the current base-set milestone.

## Basic Cards

| Card | Cost | Type | Runtime notes |
|---|---:|---|---|
| Copper | 0 | Treasure | `coin_value=1` fast path |
| Silver | 3 | Treasure | `coin_value=2`; Merchant trigger target |
| Gold | 6 | Treasure | `coin_value=3` |
| Platinum | 9 | Treasure | Colony setup option |
| Potion | 4 | Treasure | Potion budget support |
| Estate | 2 | Victory | 1 VP |
| Duchy | 5 | Victory | 3 VP |
| Province | 8 | Victory | Game-end pile |
| Colony | 11 | Victory | Colony setup option and game-end pile |
| Curse | 0 | Curse | -1 VP |

## Kingdom Cards

| Card | Cost | Types | Implementation |
|---|---:|---|---|
| Artisan | 6 | Action | DSL choice/gain/topdeck |
| Bandit | 5 | Action, Attack | DSL attack with Bandit payload |
| Bureaucrat | 4 | Action, Attack | DSL gain plus opponent choice |
| Cellar | 2 | Action | DSL discard and per-chosen draw |
| Chapel | 2 | Action | DSL trash up to 4 |
| Council Room | 5 | Action | DSL draw/buy plus each-other-player draw |
| Festival | 5 | Action | DSL resources |
| Gardens | 4 | Victory | `score_hook` |
| Harbinger | 3 | Action | DSL discard-zone choice |
| Laboratory | 5 | Action | DSL resources |
| Library | 5 | Action | `custom_step` draw-to-7 with set-aside choices |
| Market | 5 | Action | DSL resources |
| Merchant | 3 | Action | DSL resources plus trigger subscription |
| Militia | 4 | Action, Attack | DSL attack discard-down-to |
| Mine | 5 | Action | DSL trash treasure and gain to hand |
| Moat | 2 | Action, Reaction | DSL draw plus reaction immunity |
| Moneylender | 4 | Action | DSL optional Copper trash |
| Poacher | 4 | Action | DSL resources plus empty-pile discard |
| Remodel | 4 | Action | DSL trash and cost-relative gain |
| Sentry | 5 | Action | `custom_step` look/dispose/order |
| Smithy | 4 | Action | DSL draw |
| Throne Room | 4 | Action | DSL choose Action and repeated play |
| Vassal | 3 | Action | DSL discard deck top and optional play |
| Village | 3 | Action | DSL draw/actions |
| Witch | 5 | Action, Attack | DSL draw plus Curse attack |
| Workshop | 3 | Action | DSL gain up to 4 |

## Test-Only Definitions

The v2 def table also contains a few zero-cost test definitions used to exercise
interpreter control flow and trigger ordering. They are not part of the playable
kingdom roster exposed by setup helpers or documentation.

