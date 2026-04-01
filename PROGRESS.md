# DominionZero: Implementation Progress

Cross-reference: [Plan](PLAN.md) | [Benchmarks](docs/benchmarks.md) | [Card List](docs/implemented-cards.md)

## Status Summary

**75 tests passing. Zero warnings. 45 cards total (7 base + 26 Level 1 + 12 Level 2 planned). 5 agents. 12,000 games/sec (Release). EngineBot beats BigMoney 56%.**

---

## Completed

### Build System
- CMake with `dominion_engine` library, Catch2 v3 via FetchContent
- 6 build targets: `dominion_zero` (benchmarks), `dominion_play` (human vs bot), `dominion_2p` (local 2-player), `dominion_stress` (stress testing), `dominion_gui` (Raylib, conditional), `dominion_tests`
- Use `cmake -B build -DCMAKE_BUILD_TYPE=Release` for benchmarks (35x faster than Debug)

### Game Layer
- `Card` struct with full ability slots: `on_play`, `on_react`, `vp_fn`, `on_gain`, `on_trash`, `on_duration`
- `ChoiceType` enum: DISCARD, TRASH, GAIN, TOPDECK, EXILE, PLAY_CARD, YES_NO, REVEAL, MULTI_FATE, ORDER, SELECT_FROM_DISCARD
- `Player` with complete zone management (hand, deck, discard, in_play, set_aside, mats) and operations (peek_deck, remove_deck_top, trash_from_hand_return, find_in_hand, all_cards, topdeck_from_hand, etc.)
- `GameState` with attack resolution, play_card_from_hand/play_card_effect, gainable_piles, effective_cost, dynamic VP scoring, tiebreaker by turn count, turn flags, actions_played tracking
- `Phase` enum: ACTION, TREASURE, BUY, NIGHT, CLEANUP, GAME_OVER
- Vector-indexed card lookups (not hash map) for performance

### Engine Layer
- `DecisionPoint` with `build_action_decision`, `build_treasure_decision`, `build_buy_decision`
- `GameRunner` with full turn loop, `GameObserver` callback, safety caps (80 turns, 5000 decisions)
- `DecisionFn` bridge with validation, proper labels for all choice types
- 5 agents: RandomAgent, BetterRandomAgent, BigMoneyAgent, HeuristicAgent, EngineBot

### All 26 Base Set Kingdom Cards (Level 1)
- **Simple:** Smithy, Village, Laboratory, Festival, Market
- **Choices:** Cellar, Chapel, Workshop, Moneylender, Poacher, Vassal, Harbinger
- **Complex:** Remodel, Mine, Artisan, Sentry, Library, Bureaucrat
- **Attacks/Reactions:** Moat, Militia, Bandit, Witch, Council Room, Throne Room, Gardens
- **Triggered:** Merchant (Silver bonus via turn flags)
- Throne Room tested extensively: TR+Smithy, TR+Village, TR+Festival, TR+Militia, TR+Witch, TR+Moat, TR+Lab, TR+Cellar, TR+Moneylender, TR+Chapel, TR+TR (double), TR+TR+TR (triple)

### Testing (75 tests across 9 compiled test files)
- test_player.cpp: 8 tests
- test_supply.cpp: 4 tests
- test_game_state.cpp: 7 tests
- test_cards_base.cpp: 29 tests
- test_card_interactions.cpp: 14 tests
- test_attack_reactions.cpp: 6 tests
- test_action_space.cpp: 4 tests
- test_invariants.cpp: 3 tests (includes determinism test)

### Benchmarks
- 9 matchups x 5000 games each, random 10-of-26 kingdoms
- Reports win rates, avg scores, avg turns, games/sec
- 45,000 games in ~6.5 seconds (Release)

### Interactive Play
- Terminal: Human vs EngineBot, local 2-player hot-seat
- GUI: Raylib-based visual client with card rendering and game state display
- Full state display before every decision, card names in prompts

### Performance Optimization
- `card_def()`: vector index instead of hash map (2 lookups eliminated per call)
- `card_names_`: `vector<string>` instead of `unordered_map<int, string>`
- Supply: index-based API, cached Province pile index
- `build_buy_decision`: iterates piles directly, no allocation
- Result: 340 (Debug) -> 12,000 (Release) games/sec

### Bugs Found and Fixed
1. Throne Room index bug: `chosen[0]` used as hand index instead of `action_indices[chosen[0]]`
2. Sub-decision routing: routed to current turn's agent instead of affected player's agent
3. YES_NO display: showed card names instead of "Yes"/"No"
4. GAIN display: showed "Option 0" instead of card names
5. Multi-select enforcement: engine didn't validate choice count (Militia accepted 1 discard instead of 2)
6. Sentry context: MULTI_FATE and ORDER prompts didn't show which card was being decided on

---

## In Progress

### Level 2 Cards
- 12 cards planned: Worker's Village, Courtyard, Hamlet, Lookout, Swindler, Scholar, Storeroom, Conspirator, Menagerie, Oasis, King's Court, Courier, Sentinel
- Test file exists (`test_cards_level2.cpp`, 9 tests) but not yet compiled into test suite
- Cards documented in `docs/implemented-cards.md` but not yet registered in CardRegistry

---

## Not Started

### Python API
- Branch `python-API` exists but no Python code yet
- Goal: Python bindings for the C++ engine to enable ML training

### AI Layer
- GameEnvironment RL wrapper (reset/step/get_observation/clone/determinize)
- Observation tensor encoding (~200 floats)
- Information Set MCTS with neural network evaluation
- Self-play training loop
- Action space deduplication for MCTS branching factor reduction

---

## File Inventory

```
src/
  main.cpp                          Benchmark suite (9 matchups x 5000 games)
  interactive.cpp                   Human vs EngineBot (random kingdom)
  local_multiplayer.cpp             Local 2-player hot-seat
  stress_test.cpp                   Stress testing utility
  game/
    card.h/.cpp                     Card struct + CardRegistry
    player.h/.cpp                   Player zones + card movement
    supply.h/.cpp                   Supply piles (index-based API)
    game_state.h/.cpp               GameState (turn lifecycle, scoring, attacks)
    cards/
      base_cards.h/.cpp             7 base supply cards + setup
      level_1_cards.h/.cpp          All 26 Base Set kingdom cards
  engine/
    action_space.h/.cpp             DecisionType, DecisionPoint, build_*_decision()
    rules.h/.cpp                    Validation helpers
    agents.h/.cpp                   5 agent implementations
    game_runner.h/.cpp              GameRunner, Agent interface, GameObserver
  gui/
    gui_main.cpp                    Raylib GUI client
    gui_agent.cpp/.h                GUI agent for interactive play
  ui/
    terminal_helpers.h              Terminal output helpers

tests/
  test_helpers.h/.cpp               TestGame helper class
  test_player.cpp                   8 tests
  test_supply.cpp                   4 tests
  test_game_state.cpp               7 tests
  test_cards_base.cpp               29 tests
  test_card_interactions.cpp        14 tests
  test_attack_reactions.cpp         6 tests
  test_action_space.cpp             4 tests
  test_invariants.cpp               3 tests
  test_cards_level2.cpp             9 tests (not yet compiled)

docs/
  dominion_rules_reference.md       Distilled Dominion rules
  benchmarks.md                     Performance data and bot rankings
  bot_strategies.md                 Agent strategy details
  implemented-cards.md              Card table (45 cards)
  how-to-implement-cards.md         Card implementation guide
```

**Total: 75 compiled tests, 6 build targets, 5 agents, ~5800 lines of engine code**

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

# Play vs Engine Bot
./build/dominion_play

# 2-player local
./build/dominion_2p

# Stress test
./build/dominion_stress
```
