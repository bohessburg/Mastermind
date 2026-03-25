# DominionZero: Implementation Progress

Cross-reference: [Full planning doc](../.claude/plans/inherited-leaping-pinwheel.md)

## Status Summary

**75 tests passing. Zero warnings. All 26 base kingdom cards implemented. Full game loop working. ~675 games/sec.**

---

## Plan Steps: Completed

### Step 1: Build System ✅
- `CMakeLists.txt` with `dominion_engine` library, 4 executables, Catch2 v3 via FetchContent
- **Deviation from plan:** Added 3 extra executables beyond what the plan specified:
  - `dominion_play` — human vs heuristic bot (random kingdom)
  - `dominion_2p` — local 2-player hot-seat
  - `dominion_zero` — benchmark suite

### Step 2: Game Layer Enhancements ✅
Everything from the plan implemented:
- `Card` struct extended with `on_react`, `vp_fn`, `on_gain`, `on_trash`, `on_duration` ability slots
- `ChoiceType` extended: `PLAY_CARD`, `YES_NO`, `REVEAL`, `MULTI_FATE`, `ORDER`, `SELECT_FROM_DISCARD`
- `Player` new methods: `peek_deck`, `remove_deck_top`, `trash_from_hand_return`, `find_in_hand`, `all_cards`, `remove_from_discard`, `remove_from_discard_by_index`, `topdeck_from_hand`, `discard_size`
- `GameState` new methods: `resolve_attack`, `play_card_from_hand`, `play_card_effect`, `gainable_piles` (two overloads), `effective_cost`, `total_cards_owned`, `set_phase`, turn flags (`get_turn_flag`/`set_turn_flag`)
- `Phase` enum: added `TREASURE` and `NIGHT`
- `calculate_scores`: supports dynamic VP via `vp_fn`
- `winner`: implements tiebreaker (fewer turns wins) via `turns_taken_` tracking

### Step 3: Test Infrastructure ✅
- `TestGame` helper class with `set_hand`, `set_deck`, `set_discard`, `set_seed`, `setup_supply`
- `scripted_decisions`, `minimal_decisions`, `greedy_decisions` decision factories
- `play_card` convenience method, `hand_names`/`deck_names`/`discard_names`/`in_play_names` accessors
- Basic tests for Player (8), Supply (4), GameState (7)

### Step 4: Wave 1 Kingdom Cards ✅
Smithy, Village, Laboratory, Festival, Market — all with tests including edge cases (partial draw, empty deck shuffle).

### Step 5: Wave 2 Kingdom Cards ✅
Cellar, Chapel, Workshop, Moneylender, Poacher, Vassal, Harbinger — all with multi-section tests covering different decision paths.

### Step 6: Wave 3 Kingdom Cards ✅
Remodel, Mine, Artisan, Sentry, Library, Bureaucrat — all with tests including empty-hand edge cases, multi-step decision sequences.

### Step 7: Wave 4 Kingdom Cards ✅
Moat, Militia, Bandit, Witch, Council Room, Throne Room, Gardens — all with tests. Extensive interaction tests:
- Throne Room + Smithy/Village/Festival/Militia/Witch/Moat/Lab/Cellar/Moneylender/Chapel
- Throne Room + Throne Room (double TR)
- Throne Room + Throne Room + Throne Room (triple TR)
- Throne Room with no actions in hand
- Throne Room picking non-first action in hand (regression for index bug)
- Moat blocking Militia/Witch/Bandit/Bureaucrat
- Moat decline to reveal
- Council Room NOT blockable by Moat (not an Attack)
- Militia with 0/3/5 card hands
- Bandit with Copper-only/no-treasure reveals
- Witch with empty Curse pile

### Step 8: Action Space + Game Runner ✅
- `build_action_decision`, `build_treasure_decision`, `build_buy_decision` — all implemented with "Play all Treasures" shortcut
- `GameRunner` with full turn loop (action → treasure → buy → cleanup)
- `Agent` interface with `RandomAgent`, `BigMoneyAgent`, `HeuristicAgent`
- `GameObserver` callback for narrating game events
- **Engine-level validation:** `make_decision_fn` validates choice count and index bounds, re-asks on violation
- **DecisionFn bridge:** correctly translates hand indices to card IDs, handles YES_NO/MULTI_FATE/GAIN display labels
- `Rules` namespace: `can_play_action`, `can_play_treasure`, `can_buy`, `playable_actions`, `playable_treasures`, `buyable_piles`
- Action space tests (4)
- Invariant tests: full games with BigMoney and Random agents

### Step 9: Wave 5 + Integration ✅
- Merchant: turn-flag-based Silver bonus (`merchant_count`/`merchant_silver_triggered`)
- Merchant bonus applied during treasure-playing in GameRunner
- Full integration: games run to completion with all 26 kingdom cards

### Step 10: Benchmark ✅
- `main.cpp` runs 4 matchups (BigMoney vs BigMoney, Heuristic vs BigMoney, BigMoney vs Heuristic, BigMoney vs Random) each for 10 seconds
- Reports games/sec, win rates, avg scores, avg turns
- Results: ~675 games/sec (BigMoney vs BigMoney), Heuristic beats BigMoney ~60%
- **Not yet at 1000 games/sec target** — optimization pass needed later

---

## Beyond the Plan: Additional Work

### HeuristicAgent (not in original plan)
A smarter bot that plays actions by priority and buys by cost with priority tiebreakers:
- **Action priority:** Villages first → cantrips → Throne Room → terminals (Witch > Council Room > Smithy > Militia > ...)
- **Buy priority:** Most expensive card, ties broken by: Province > Gold > Witch > Lab > Festival > ... > Silver > Village > ... > Chapel > Moat > Cellar
- **Sub-decision heuristics:** Trash curses/estates/coppers, discard victory cards first, always reveal Moat, gain most expensive option
- Late-game awareness: buys Duchy when ≤5 Provinces, Estate when ≤2

### Interactive Play (not in original plan)
- `dominion_play`: Human vs HeuristicAgent with random 10-of-26 kingdom
- `dominion_2p`: Local 2-player hot-seat mode
- Full game state display before every decision (supply, trash, both hands, decks, discard counts, in-play)
- Observer-based narration of bot turns and card effects

### Bugs Found and Fixed
1. **Throne Room index bug:** `chosen[0]` was used as hand index directly instead of `action_indices[chosen[0]]`. Only triggered when the Throned action wasn't the first card in hand.
2. **Sub-decision routing:** `make_decision_fn` originally routed all sub-decisions to the current turn's agent, not the affected player's agent. Militia's discard prompt went to the attacker instead of the defender.
3. **YES_NO display:** Options `{0, 1}` were displayed as card names (card IDs 0 and 1 = supply Coppers) instead of "Yes"/"No".
4. **GAIN display:** Supply pile options showed "Option 0", "Option 1" instead of card names. Fixed by passing top card IDs instead of arbitrary indices.
5. **Multi-select enforcement:** Neither the engine nor the UI validated that players provided the required number of choices. Militia's "discard exactly 2" accepted 1. Fixed at engine level (re-ask on violation) and UI level (prompt with count).

---

## Plan Steps: Not Yet Started

### Future: AI Layer (Plan Section 12)
- `GameEnvironment` RL wrapper (`reset`/`step`/`get_observation`/`clone`/`determinize`)
- Observation tensor encoding (~200 floats)
- Information Set MCTS with neural network evaluation
- Self-play training loop with `GameRecord` data collection

---

## File Inventory

```
src/
  main.cpp                          Benchmark suite (4 matchups, 10s each)
  interactive.cpp                   Human vs HeuristicAgent (random kingdom)
  local_multiplayer.cpp             Local 2-player hot-seat
  game/
    card.h/.cpp                     Card struct + CardRegistry (extended with 5 new ability slots)
    player.h/.cpp                   Player zones + 8 new methods
    supply.h/.cpp                   Supply piles (unchanged from original)
    game_state.h/.cpp               GameState (extended: attack resolution, turn flags, tiebreaker)
    cards/
      base_cards.h/.cpp             7 base supply cards + setup functions
      base_kingdom.h/.cpp           All 26 Base Set kingdom cards
  engine/
    action_space.h/.cpp             DecisionType, DecisionPoint, build_*_decision()
    rules.h/.cpp                    can_play_*, playable_*, buyable_*
    game_runner.h/.cpp              GameRunner, Agent, RandomAgent, BigMoneyAgent, HeuristicAgent

tests/
  test_helpers.h/.cpp               TestGame helper class
  test_player.cpp                   8 tests
  test_supply.cpp                   4 tests
  test_game_state.cpp               7 tests
  test_cards_base.cpp               31 tests (all 26 cards + edge cases)
  test_card_interactions.cpp        15 tests (TR combos, Moat blocking, Council Room)
  test_attack_reactions.cpp         6 tests
  test_action_space.cpp             4 tests
  test_invariants.cpp               3 tests
```

**Total: 75 tests, 4 executables, ~3500 lines of engine code**

---

## How to Build and Run

```bash
cd /Users/paisho/Projects/Mastermind
cmake -B build
cmake --build build

# Run tests
cd build && ctest --output-on-failure

# Benchmarks
./build/dominion_zero

# Play vs bot (random kingdom)
./build/dominion_play

# 2-player local
./build/dominion_2p
```
