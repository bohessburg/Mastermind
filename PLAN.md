# DominionZero Engine: Implementation Plan

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Game Layer](#3-game-layer)
4. [Engine Layer](#4-engine-layer)
5. [Card Implementation](#5-card-implementation)
6. [Testing Strategy](#6-testing-strategy)
7. [Bot Agents](#7-bot-agents)
8. [Interactive Play & GUI](#8-interactive-play--gui)
9. [Future: AI Layer](#9-future-ai-layer)
10. [Future: Performance](#10-future-performance)

---

## 1. Project Overview

**Goal:** Build a fast, correct, modular Dominion game engine in C++ as the foundation for AlphaZero-style self-play reinforcement learning.

**Three priorities (in order):**
1. **Correctness** — All Dominion rules handled properly, including edge cases
2. **Speed** — Target >1000 games/second for self-play training (currently 12,000/sec)
3. **AI-friendliness** — Clean action space, enumerable legal moves, clonable state

---

## 2. Architecture

### Three-Layer Design

```
┌───────────────────────────────────────────────┐
│  AI Layer (future)                            │
│  GameEnvironment, MCTS, Observation encoding  │
├───────────────────────────────────────────────┤
│  Engine Layer                                 │
│  GameRunner, ActionSpace, Agents              │
├───────────────────────────────────────────────┤
│  Game Layer                                   │
│  GameState, Player, Supply, Card, Registry    │
└───────────────────────────────────────────────┘
```

### Card Effect Model: Callbacks

Cards define effects as C++ lambdas (`on_play`, `on_react`, `vp_fn`, `on_gain`, `on_trash`, `on_duration`). This is the most natural to write and fast to implement. For AI/MCTS clonability, state is cloned before card resolution and replayed with different `DecisionFn` responses.

### Key Design Decisions
- **Card IDs:** Integer IDs assigned at creation, vector-indexed lookups (not hash map)
- **Supply:** Index-based API with cached Province pile index for fast game-over checks
- **DecisionFn:** Callback-based player choices; engine validates choice count/bounds and re-asks on violation
- **Attack resolution:** Iterates opponents in turn order, checks reaction cards, blocks if applicable

---

## 3. Game Layer

### Components
- **Card** (`card.h/cpp`): Card struct with type bitmask, cost, ability slots (`on_play`, `on_react`, `vp_fn`, `on_gain`, `on_trash`, `on_duration`), CardRegistry for definitions
- **Player** (`player.h/cpp`): Hand/deck/discard/in_play/set_aside/mats zones, card movement, draw/shuffle, peek/remove deck operations
- **Supply** (`supply.h/cpp`): Pile management, game-end detection (Province empty or 3+ piles empty), index-based API
- **GameState** (`game_state.h/cpp`): Turn lifecycle, phase management (ACTION/TREASURE/BUY/NIGHT/CLEANUP/GAME_OVER), scoring with dynamic VP, tiebreaker by turn count, attack resolution, turn flags

### Dominion Rules Coverage
- Full turn structure: Action phase, Treasure phase, Buy phase, Cleanup
- "Do as much as you can" rule
- Throne Room edge cases (TR on nothing, TR on self-trashing card, nested TR)
- Reaction timing (resolve before attack)
- Game end at end of turn, not mid-turn
- Gaining vs buying distinction
- Shuffle-on-demand (not proactive)

---

## 4. Engine Layer

### Action Space (`action_space.h/cpp`)
- `DecisionPoint` structure with player ID, decision type, legal options, min/max choices
- `build_action_decision()`: Actions in hand + "End Actions"
- `build_treasure_decision()`: Treasures in hand + "Play All" + "Stop"
- `build_buy_decision()`: Affordable piles + "End Buys"

### Game Runner (`game_runner.h/cpp`)
- Full turn loop with `GameObserver` callback
- Safety caps: 80 turns, 5000 decisions per game
- `DecisionFn` bridge: translates hand indices to card IDs, proper labels for all choice types

### Agents (`agents.h/cpp`)
- Abstract `Agent` interface: given a `DecisionPoint` and `GameState`, return chosen option indices
- Five implementations: RandomAgent, BetterRandomAgent, BigMoneyAgent, HeuristicAgent, EngineBot

---

## 5. Card Implementation

### Card Levels
Cards are organized by implementation complexity:

- **Base Supply Cards (7):** Copper, Silver, Gold, Estate, Duchy, Province, Curse
- **Level 1 — Base Set Kingdom Cards (26):** All 26 kingdom cards from the original Dominion base set, implemented in waves from simple (+Cards/+Actions) through complex (attacks, reactions, recursion, triggered abilities)
- **Level 2 — Cross-Set Cards (12 planned):** Worker's Village, Courtyard, Hamlet, Lookout, Swindler, Scholar, Storeroom, Conspirator, Menagerie, Oasis, King's Court, Courier, Sentinel

### Implementation Pattern
Each card is a registered `Card` struct with lambda-based ability slots. Cards that need player decisions use the `DecisionFn` callback. Cards are registered in `CardRegistry` and referenced by name for supply setup.

---

## 6. Testing Strategy

### Test Infrastructure
- `TestGame` helper class: `set_hand/deck/discard`, `scripted_decisions`, `minimal_decisions`, `greedy_decisions`
- Catch2 v3 via FetchContent, CTest integration with 10-second timeout

### Test Coverage Areas
- Player zone manipulation and card movement
- Supply pile management and game-end detection
- Game state initialization, scoring, turn advancement
- Individual card behavior (all 26 Level 1 cards)
- Multi-card interactions and Throne Room variants (TR+Smithy, TR+TR, TR+TR+TR, etc.)
- Attack/reaction mechanics (Moat blocking, Militia, Bandit, Witch)
- Action space decision building
- Game invariants and determinism (same seed = same result)

---

## 7. Bot Agents

### BigMoneyAgent
Pure money strategy based on wiki.dominionstrategy.com "Big Money optimized." Builds treasure to 16 money value, then greens with duchy/estate dancing.

### HeuristicAgent
Plays all actions by priority (villages -> cantrips -> terminals), buys most expensive card with diminishing returns on duplicates (`(1+copies)^2` penalty).

### EngineBot
Kingdom-aware strategy selection:
- **FULL_ENGINE:** Chapel + Village + terminal draw -> build engine, then green
- **BM_PLUS_X:** Good terminal but no village -> BigMoney + 1-2 key actions
- **PURE_BM:** Nothing useful -> falls back to BigMoney

### Bot Rankings (seat-adjusted, random kingdoms)
| Bot A vs Bot B | A win% | B win% |
|----------------|--------|--------|
| Engine vs BigMoney | 56% | 41% |
| Engine vs Heuristic | 69% | 28% |
| BigMoney vs Heuristic | 57% | 40% |

---

## 8. Interactive Play & GUI

### Terminal Modes
- **dominion_play:** Human vs EngineBot with random 10-of-26 kingdom
- **dominion_2p:** Local 2-player hot-seat
- Full game state display (supply, trash, hands, decks, discard, in-play) before every decision
- Optional choices show "Skip" option, Sentry shows card names for fate/order decisions

### GUI (Raylib)
- Visual card rendering, game state display, supply piles
- GUI agent for interactive play
- Conditional build (requires raylib via pkg-config)

---

## 9. Future: AI Layer

Not yet implemented. Planned components:

### GameEnvironment (RL wrapper)
- `reset()` / `step(action_index)` / `get_observation()` / `clone()` / `determinize(player_id, rng)`
- Observation tensor: per-card-name counts across zones, scalar features (actions/buys/coins/turn/phase), decision context, legal action mask (~200 floats)

### Python API
- Python bindings for the C++ engine (branch exists, not yet implemented)
- Enable training with Python ML frameworks (PyTorch, etc.)

### MCTS
- Information Set MCTS: clone env, determinize hidden info, run standard MCTS with neural network evaluation, aggregate across determinizations
- Action space deduplication: collapse identical actions by card name to reduce branching factor (e.g., 4 Coppers = 1 "Play Copper" option)
- Sub-decision strategy: top-level play decisions searched by MCTS, sub-decisions (TR target, Chapel trash, Militia discard) use heuristics

### Neural Network
- Network outputs distribution over MAX_ACTION_OPTIONS=64 slots, masked by legal_action_mask
- Self-play training loop with GameRecord data collection

---

## 10. Future: Performance

Issues to address before AlphaZero training at scale:

- **String-based turn flags:** Replace `unordered_map<string, int>` with enum keys or flat array
- **Conditional tracking counters:** Counters like `actions_played_` only needed when specific cards (Conspirator) are in the kingdom; could use tracking bitfield
- **Reveal / public information system:** Per-turn event log for imperfect-information training
- **String-based card names:** Consider integer card type IDs or interned strings for supply/registry lookups in millions of simulations
