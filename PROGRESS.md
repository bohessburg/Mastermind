# DominionZero: Implementation Progress

Cross-reference: [Full planning doc](../.claude/plans/inherited-leaping-pinwheel.md) | [Benchmarks](docs/benchmarks.md)

## Status Summary

**75 tests passing. Zero warnings. All 26 base kingdom cards implemented. 3 bot agents. 12,000 games/sec (Release). Engine bot beats BigMoney 56%.**

---

## Plan Steps: Completed

### Step 1: Build System ✅
- `CMakeLists.txt` with `dominion_engine` library, Catch2 v3 via FetchContent
- 4 executables: `dominion_zero` (benchmarks), `dominion_play` (human vs bot), `dominion_2p` (local 2-player), `dominion_tests`
- **Important:** Use `cmake -B build -DCMAKE_BUILD_TYPE=Release` for benchmarks (8-16x faster than Debug)

### Step 2: Game Layer Enhancements ✅
- `Card` struct: `on_react`, `vp_fn`, `on_gain`, `on_trash`, `on_duration` ability slots
- `ChoiceType` extended: `PLAY_CARD`, `YES_NO`, `REVEAL`, `MULTI_FATE`, `ORDER`, `SELECT_FROM_DISCARD`
- `Player` new methods: `peek_deck`, `remove_deck_top`, `trash_from_hand_return`, `find_in_hand`, `all_cards`, `remove_from_discard`, `remove_from_discard_by_index`, `topdeck_from_hand`, `discard_size`
- `GameState` new methods: `resolve_attack`, `play_card_from_hand`, `play_card_effect`, `gainable_piles`, `effective_cost`, `total_cards_owned`, `set_phase`, turn flags
- `Phase` enum: added `TREASURE` and `NIGHT`
- Dynamic VP scoring (`vp_fn`), tiebreaker by turn count
- **Performance optimization:** `card_def()` uses vector index (not hash map), `Supply` has index-based API and cached Province pile index, `gainable_piles()` iterates `piles()` directly instead of allocating

### Step 3: Test Infrastructure ✅
- `TestGame` helper: `set_hand/deck/discard`, `scripted_decisions`, `minimal_decisions`, `greedy_decisions`
- 75 tests across 9 test files covering all cards, interactions, edge cases

### Steps 4-7: All 26 Base Kingdom Cards ✅
All cards implemented with tests:
- **Wave 1 (simple):** Smithy, Village, Laboratory, Festival, Market
- **Wave 2 (choices):** Cellar, Chapel, Workshop, Moneylender, Poacher, Vassal, Harbinger
- **Wave 3 (complex):** Remodel, Mine, Artisan, Sentry, Library, Bureaucrat
- **Wave 4 (attacks/reactions/recursion):** Moat, Militia, Bandit, Witch, Council Room, Throne Room, Gardens
- **Wave 5 (triggered):** Merchant

Throne Room tested extensively: TR+Smithy, TR+Village, TR+Festival, TR+Militia, TR+Witch, TR+Moat, TR+Lab, TR+Cellar, TR+Moneylender, TR+Chapel, TR+TR (double), TR+TR+TR (triple).

### Step 8: Action Space + Game Runner ✅
- `build_action_decision`, `build_treasure_decision`, `build_buy_decision`
- `GameRunner` with full turn loop, `GameObserver` callback, turn/decision caps (80 turns, 5000 decisions)
- Engine-level validation: `make_decision_fn` validates choice count/bounds, re-asks on violation
- `DecisionFn` bridge: translates hand indices to card IDs, proper labels for YES_NO/MULTI_FATE/GAIN/ORDER

### Step 9: Wave 5 + Integration ✅
- Merchant turn-flag-based Silver bonus
- Full games run to completion with all 26 kingdom cards

### Step 10: Benchmark ✅
- 9 matchups × 5000 games each, random 10-of-26 kingdoms
- Reports total/completed/timed-out games, win rates, avg scores, avg turns, games/sec
- 45,000 games complete in ~6.5 seconds (Release)

---

## Beyond the Plan

### Three Bot Agents
1. **BigMoneyAgent** — Pure money strategy based on wiki.dominionstrategy.com "Big Money optimized" rules. Builds treasure to 16 money value, then greens with duchy/estate dancing.
2. **HeuristicAgent** — Plays all actions by priority (villages → cantrips → terminals), buys most expensive card with diminishing returns on duplicates (`(1+copies)^2` penalty).
3. **EngineBot** — Kingdom-aware strategy selection:
   - **FULL_ENGINE**: Chapel + Village + terminal draw → build engine, then green
   - **BM_PLUS_X**: Good terminal but no village → BigMoney + 1-2 key actions (wins 63% vs BM)
   - **PURE_BM**: Nothing useful → falls back to BigMoney

### Bot Rankings (seat-adjusted, random kingdoms)
| Bot A vs Bot B | A win% | B win% |
|----------------|--------|--------|
| Engine vs BigMoney | 56% | 41% |
| Engine vs Heuristic | 69% | 28% |
| BigMoney vs Heuristic | 57% | 40% |

### Interactive Play
- `dominion_play`: Human vs EngineBot with random 10-of-26 kingdom
- `dominion_2p`: Local 2-player hot-seat
- Full game state display (supply, trash, hands, decks, discard, in-play) before every decision
- Optional choices show "Skip" option, Sentry shows card names for fate/order decisions

### Performance Optimization
- `card_def()`: single vector index instead of two hash map lookups
- `card_names_`: `vector<string>` instead of `unordered_map<int, string>`
- `card_defs_`: cached `const Card*` per card_id
- `Supply`: index-based API (`piles()`, `pile_at()`, `top_card_index()`), cached Province pile index
- `build_buy_decision`: iterates `piles()` directly, no `all_pile_names()` allocation
- Result: **6,000 → 12,000 games/sec** (2x from string optimization), **340 → 12,000** (35x including Debug→Release)

### Bugs Found and Fixed
1. **Throne Room index bug:** `chosen[0]` used as hand index instead of `action_indices[chosen[0]]`
2. **Sub-decision routing:** All sub-decisions routed to current turn's agent instead of affected player's agent
3. **YES_NO display:** Options shown as card names instead of "Yes"/"No"
4. **GAIN display:** Supply options showed "Option 0" instead of card names
5. **Multi-select enforcement:** Engine didn't validate choice count, Militia accepted 1 discard instead of 2
6. **Sentry context:** MULTI_FATE and ORDER prompts didn't show which card was being decided on

---

## Plan Steps: Not Yet Started

### Future: AI Layer (Plan Section 12)
- `GameEnvironment` RL wrapper (`reset`/`step`/`get_observation`/`clone`/`determinize`)
- Observation tensor encoding (~200 floats)
- Information Set MCTS with neural network evaluation
- Self-play training loop with `GameRecord` data collection
- State clone + step interface (needed for MCTS tree search)

---

## File Inventory

```
src/
  main.cpp                          Benchmark suite (9 matchups × 5000 games)
  interactive.cpp                   Human vs EngineBot (random kingdom)
  local_multiplayer.cpp             Local 2-player hot-seat
  game/
    card.h/.cpp                     Card struct + CardRegistry
    player.h/.cpp                   Player zones + card movement
    supply.h/.cpp                   Supply piles (string + index-based API)
    game_state.h/.cpp               GameState (vector-indexed card lookups, turn flags)
    cards/
      base_cards.h/.cpp             7 base supply cards + setup
      base_kingdom.h/.cpp           All 26 Base Set kingdom cards
  engine/
    action_space.h/.cpp             DecisionType, DecisionPoint, build_*_decision()
    rules.h/.cpp                    Validation helpers
    game_runner.h/.cpp              GameRunner, Agent, BigMoneyAgent, HeuristicAgent, EngineBot

tests/
  test_helpers.h/.cpp               TestGame helper class
  test_player.cpp                   8 tests
  test_supply.cpp                   4 tests
  test_game_state.cpp               7 tests
  test_cards_base.cpp               31 tests
  test_card_interactions.cpp        15 tests
  test_attack_reactions.cpp         6 tests
  test_action_space.cpp             4 tests
  test_invariants.cpp               3 tests (+ 1 determinism test)

docs/
  dominion_rules_reference.md       Distilled Dominion rules
  benchmarks.md                     Performance data and bot rankings
```

**Total: 75 tests, 4 executables, 3 bot agents, ~4500 lines of engine code**

---

## How to Build and Run

```bash
# Build (Release for performance)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run tests
cd build && ctest --output-on-failure

# Benchmarks (45K games in ~6.5s)
./build/dominion_zero

# Play vs Engine Bot (random kingdom)
./build/dominion_play

# 2-player local
./build/dominion_2p
```
