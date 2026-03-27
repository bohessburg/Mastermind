# DominionZero Engine: Complete Implementation Plan

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Existing Codebase Reference](#2-existing-codebase-reference)
3. [Dominion Rules Summary for Engine Design](#3-dominion-rules-summary-for-engine-design)
4. [Architecture Decisions and Rationale](#4-architecture-decisions-and-rationale)
5. [Game Layer Enhancements](#5-game-layer-enhancements)
6. [Engine Layer Design](#6-engine-layer-design)
7. [Card Implementation Framework](#7-card-implementation-framework)
8. [Base Kingdom Card Implementations](#8-base-kingdom-card-implementations)
9. [Testing Strategy](#9-testing-strategy)
10. [Build System](#10-build-system)
11. [Implementation Sequence](#11-implementation-sequence)
12. [Future: AI Layer](#12-future-ai-layer)
13. [Edge Cases and Gotchas](#13-edge-cases-and-gotchas)

---

## 1. Project Overview

**Goal:** Build a fast, correct, modular Dominion game engine in C++ as the foundation for AlphaZero-style self-play reinforcement learning.

**Project location:** `/Users/paisho/Projects/Mastermind`

**Rules reference:** `Dominion_CompleteRules_v11.1.pdf` (188 pages, v11.1 July 2025 by Nick Knutsen). Use `pdftotext Dominion_CompleteRules_v11.1.pdf - | sed -n 'START,ENDp'` to search. Key sections:
- Ch II (p12-16): Turn structure, phases, game end
- Ch III (p17-26): Card abilities, resolving effects, timing rules
- Ch VI (p39-45): Summaries of all triggers, timing, common mistakes
- Ch VII (p46-179): Per-card rulings

**Three priorities (in order):**
1. **Correctness** — All Dominion rules handled properly, including edge cases
2. **Speed** — Target >1000 games/second for eventual self-play training
3. **AI-friendliness** — Clean action space, enumerable legal moves, clonable state

---

## 2. Existing Codebase Reference

### Directory Structure
```
/Users/paisho/Projects/Mastermind/
├── CMakeLists.txt                    (EMPTY — needs build system)
├── README.md                         (Project overview, research directions)
├── Dominion_CompleteRules_v11.1.pdf  (188-page rules reference)
├── src/
│   ├── main.cpp                      (EMPTY)
│   ├── game/
│   │   ├── card.h                    (Card definition, CardType bitmask, DecisionFn, CardRegistry)
│   │   ├── card.cpp                  (CardRegistry implementation — static unordered_map)
│   │   ├── player.h                  (Player: hand/deck/discard/in_play/set_aside/mats zones)
│   │   ├── player.cpp                (Card movement, draw, shuffle, cleanup)
│   │   ├── supply.h                  (SupplyPile struct, Supply class)
│   │   ├── supply.cpp                (Pile management, game-end detection)
│   │   ├── game_state.h              (GameState: phases, turn state, card ID system)
│   │   ├── game_state.cpp            (Turn lifecycle, scoring, gain helpers)
│   │   └── cards/
│   │       ├── base_cards.h          (BaseCards namespace: register_all, setup_supply, setup_starting_decks)
│   │       ├── base_cards.cpp        (Copper/Silver/Gold/Estate/Duchy/Province/Curse definitions + setup)
│   │       ├── kingdom_cards.h       (EMPTY)
│   │       └── kingdom_cards.cpp     (EMPTY)
│   └── engine/
│       ├── action_space.h            (EMPTY)
│       ├── action_space.cpp          (EMPTY)
│       ├── rules.h                   (EMPTY)
│       ├── rules.cpp                 (EMPTY)
│       ├── game_runner.h             (EMPTY)
│       └── game_runner.cpp           (EMPTY)
└── tests/
    ├── test_action_space.cpp         (EMPTY)
    ├── test_cards.cpp                (EMPTY)
    └── test_game_state.cpp           (EMPTY)
```

### File-by-File Analysis

#### `src/game/card.h` (86 lines)
- `CardType` enum (bitmask, `uint16_t`): None, Action, Treasure, Victory, Curse, Attack, Reaction, Duration, Night, Command
- Bitwise operators `|` and `&` for combining types
- `has_type(types, flag)` helper
- `ChoiceType` enum: DISCARD, TRASH, GAIN, TOPDECK, EXILE
- `DecisionFn` type: `std::function<std::vector<int>(int player_id, ChoiceType, const std::vector<int>& options, int min_choices, int max_choices)>`
- `Card` struct: name, cost, types, text, victory_points, coin_value, tags (unordered_set<string>), on_play (std::function<void(GameState&, int, DecisionFn)>)
- Helper methods: is_action(), is_treasure(), is_victory(), etc., has_tag()
- `CardRegistry` namespace: register_card(), get(), all_names()

**Key limitation:** Only one ability slot (`on_play`). No reaction, when-gain, when-trash, dynamic VP, or duration slots.

#### `src/game/card.cpp` (32 lines)
- Static `std::unordered_map<std::string, Card>` as the registry
- `register_card` stores by name, `get` returns `const Card*`, `all_names` returns all keys

#### `src/game/player.h` (58 lines)
- Private: id_, hand_, deck_ (draw from back), discard_, in_play_, set_aside_ (map<string,vec>), mats_ (map<string,vec>), rng_ (mt19937)
- Read accessors: get_id, hand_size, deck_size, get_hand/deck/discard/in_play, get_set_aside/mats/set_aside_by/mat
- Card movement: draw_cards(count), discard_from_hand(idx), discard_hand(), trash_from_hand(idx), add_to_hand(id), add_to_discard(id), add_to_deck_top(id), play_from_hand(idx), cleanup()
- Set-aside: set_aside(id, source), remove_from_set_aside(source, idx)
- Mats: add_to_mat(id, name), remove_from_mat(name, idx)
- shuffle_discard_into_deck(), set_rng(mt19937)

**Key limitation:** `trash_from_hand` doesn't return the card_id (just erases). No peek_deck or remove_deck_top operations.

#### `src/game/player.cpp` (122 lines)
- Constructor seeds rng from random_device
- `draw_cards`: loops, shuffles discard into deck when empty, pops from back of deck_
- `discard_from_hand`: pushes to discard_, erases from hand_ by index
- `trash_from_hand`: only erases from hand_ — **caller must add to trash**
- `play_from_hand`: pushes to in_play_, erases from hand_
- `cleanup`: discard_hand(), move in_play_ to discard_, draw_cards(5)
- `shuffle_discard_into_deck`: appends discard to deck, clears discard, std::shuffle with rng_

#### `src/game/supply.h` (50 lines)
- `SupplyPile` struct: pile_name, card_ids (vector<int>, gain from back), starting_count
- `Supply` class: add_pile, gain_from (returns card_id or -1), top_card, count, is_pile_empty, available_piles, all_pile_names, empty_pile_count, is_game_over, get_pile (mutable/const)
- Private: piles_ vector, pile_index_ map (name→index)

#### `src/game/supply.cpp` (77 lines)
- `is_game_over`: Province empty OR 3+ piles empty
- `gain_from`: pops from back of card_ids vector

#### `src/game/game_state.h` (92 lines)
- `Phase` enum: ACTION, BUY, CLEANUP, GAME_OVER
- `GameState` class private: players_ (vector), current_player_, supply_, trash_ (vector<int>), phase_, actions_, buys_, coins_, turn_number_, next_card_id_, card_names_ (unordered_map<int,string>)
- Card ID system: create_card(name)→id, card_name(id)→string, card_def(id)→Card*
- Player access: num_players, get_player, current_player_id
- Supply access: get_supply (mutable/const)
- Turn state: current_phase, actions, buys, coins, turn_number, add_actions/buys/coins
- Game actions: gain_card(pid, pile) → to discard, gain_card_to_hand, gain_card_to_deck_top, trash_card(card_id), get_trash
- Lifecycle: start_game, start_turn, advance_phase
- End: is_game_over, calculate_scores→vec<int>, winner→pid or -1

#### `src/game/game_state.cpp` (205 lines)
- Constructor: creates `num_players` Player objects, sets phase=ACTION, actions=1, buys=1, coins=0, turn=1
- `create_card`: increments next_card_id_, stores mapping
- `gain_card`: calls supply_.gain_from, then player.add_to_discard
- `advance_phase`: ACTION→BUY→CLEANUP (runs player.cleanup(), advances current_player_, increments turn, checks game over)→GAME_OVER
- `calculate_scores`: iterates all cards in all zones (hand+deck+discard+in_play+set_aside+mats), sums victory_points
- `winner`: finds max score, returns -1 if tie

**Key limitation:** `calculate_scores` only uses static `victory_points`, no dynamic VP (Gardens). No `resolve_attack`. No `play_card_from_hand` or `play_card_effect`. Phase enum missing TREASURE/NIGHT sub-phases. `advance_phase` only handles the simple case.

#### `src/game/cards/base_cards.h` (19 lines)
```cpp
namespace BaseCards {
    void register_all();
    void setup_supply(GameState& state, const std::vector<std::string>& kingdom_cards);
    void setup_starting_decks(GameState& state);
}
```

#### `src/game/cards/base_cards.cpp` (164 lines)
- `register_all`: Registers Copper (cost 0, +1 coin), Silver (cost 3, +2), Gold (cost 6, +3), Estate (cost 2, 1VP), Duchy (cost 5, 3VP), Province (cost 8, 6VP), Curse (cost 0, -1VP)
- Treasures have `on_play` that calls `state.add_coins(N)`
- Victory/Curse have `on_play = nullptr`
- `setup_supply`: Copper=60-(7*num_players), Silver=40, Gold=30, Victory=8 for 2p/12 for 3+, Curse=(num_players-1)*10, Kingdom=10 each (victory kingdoms get victory_count)
- `setup_starting_decks`: Each player gets 7 Copper + 3 Estate in discard, shuffles, draws 5

---

## 3. Dominion Rules Summary for Engine Design

### Turn Structure (Ch II, p12-13)

A turn has three mandatory phases (plus optional Night if Night cards exist):

```
START OF TURN
  → Resolve start-of-turn triggers (Duration effects, etc.)

ACTION PHASE
  → Pool starts at 1 Action
  → Player may play Action cards from hand (costs 1 Action each)
  → Cards may grant +Actions, +Cards, +Coins, +Buys
  → Player may also spend Villager tokens for +1 Action each

BUY PHASE (Part 1: Treasures)
  → Player may play any number of Treasure cards from hand (free)
  → Treasures add Coins to money pool
  → At any time, player may spend Coffers tokens for +Coins

BUY PHASE (Part 2: Purchases)
  → Pool starts at 1 Buy
  → Player may buy cards from Supply (costs 1 Buy + coins ≤ money pool)
  → When bought, card is gained to discard pile
  → Player may also buy Events/Projects instead of cards
  → CANNOT play more Treasures after first buy

NIGHT PHASE (only if Night cards in game)
  → Player may play Night cards from hand (free, any number)

CLEAN-UP PHASE
  → Discard all cards in play (one by one — order can matter for triggers)
  → Discard all cards in hand (all at once)
  → Draw 5 new cards
  → Duration cards that have future effects stay in play (skip discard)

NEXT PLAYER'S TURN
```

### Game End Conditions (Ch II, p15)

Game ends at the end of a turn when EITHER:
- Province pile is empty, OR
- 3 or more Supply piles are empty (4+ for 5-6 players)
- If Colonies in game: also ends when Colony pile empty

Scoring: Sum VP from all owned cards (hand+deck+discard+in_play+set_aside+mats) + VP tokens. **Tiebreaker:** Player with fewer turns wins. If same turns, shared victory.

### Playing a Card (Ch II, p15)

1. Announce the card
2. Place it in play area (face up)
3. Resolve its effects top-to-bottom
4. Card stays in play until Clean-up

### Card Abilities and Triggered Abilities (Ch III, p17-26)

**Ability types on a single card** (separated by dividing lines):
- **Play ability** (topmost): resolved when card is played
- **Below-the-line abilities**: triggered at specific times (when-gain, when-trash, when-discard, reaction, etc.)

**Throne Room rule**: Only the play ability (above dividing line) is resolved multiple times.

**Key triggers** (Ch III p21-22, Ch VI p39-42):
- **before-play**: Reactions to Attacks (Moat, etc.), Adventures tokens
- **after-play**: Royal Carriage, Citadel, etc.
- **when-buy**: After paying, before gaining (mostly removed in 2022 errata)
- **when-gain**: After card reaches destination (discard/hand/deck)
- **when-discard**: After card(s) discarded
- **when-trash**: After card trashed
- **start-of-turn**: Duration effects, ongoing abilities

**Resolution order for concurrent triggers** (Ch III p22-24):
- Multiple players: resolve in turn order starting with current player
- Same player, multiple triggers: player chooses order
- Triggered abilities can interrupt mid-resolution (resolve before continuing)

### Reactions (Ch III, p20-21)

- Trigger when another player plays an Attack card
- Resolved BEFORE the Attack's play ability
- Each opponent gets a chance, in turn order
- Players get additional chances after other players resolve Reactions
- Revealing Moat blocks the attack for that player
- A Reaction can be revealed multiple times to the same attack

### Attack Pattern (common across many cards)

```
Attacker plays Attack card
→ For each other player (turn order):
   → Check for Reaction cards in hand
   → Player may reveal Reactions (e.g., Moat)
   → If not blocked: resolve attack effect for that player
```

### "Do as much as you can" Rule (Ch III, p18)

If you can't fully complete an instruction, do as much as possible. E.g., draw 3 but only 2 left in deck+discard → draw 2.

### "If you do/did" Contingency (Ch III, p19)

Effects contingent on a prior effect only happen if the prior effect succeeded. E.g., Trading Post: "trash 2 cards from hand. If you did, gain a Silver."

### Duration Cards (Ch IV, p27)

- Stay in play past Clean-up if they have unresolved future effects
- At start of next turn, resolve their future effects, then they get cleaned up normally
- Throne Room on a Duration: both TR and Duration stay in play

### Information Visibility (Ch II, p14)

**Public to all:**
- All cards in trash
- Number of cards in all Supply/non-Supply piles
- All cards in any player's play area
- All cards set aside face up
- Number of cards in each player's hand
- Top card only of each discard pile
- All tokens

**Public to owning player only:**
- Cards in own hand
- Cards set aside face down
- Number of cards in own deck

**Cannot normally look through:**
- Own discard pile (unless ability allows)
- Any deck

---

## 4. Architecture Decisions and Rationale

### Three-Layer Design

```
┌───────────────────────────────────────────────┐
│  AI Layer (future)                            │
│  GameEnvironment, MCTS, Observation encoding  │
├───────────────────────────────────────────────┤
│  Engine Layer (to build)                      │
│  GameRunner, ActionSpace, attack resolution   │
├───────────────────────────────────────────────┤
│  Game Layer (mostly exists)                   │
│  GameState, Player, Supply, Card, Registry    │
└───────────────────────────────────────────────┘
```

### Card Effect Model: Callbacks (not Effect Queue)

**Decision:** Keep the existing callback-based `on_play` approach. Cards define their effects as C++ lambdas.

**Alternatives considered:**
1. **Effect Queue / Command Pattern** — Cards produce a list of `Effect` data objects, engine processes them. Fully serializable/clonable mid-resolution. But: requires building a complex DSL to express all card effects, massive upfront work for 500+ cards, hard to express cards like Library (conditional loops).
2. **C++20 Coroutines** — Cards `co_yield` at decision points. Clean code, but coroutine frames can't be portably cloned (compiler-specific layout), and adds compiler requirements.
3. **Hybrid** — Some cards use callbacks, others use effect queue. Worst of both worlds: inconsistent patterns, boundary bugs.

**Why callbacks win for now:**
- Most natural to write: card code reads like the rules text
- The `DecisionFn` callback already handles player choices
- Fast to implement for 500+ cards
- For AI/MCTS, clonability is handled at the Engine layer via the GameRunner (see below)

**Future migration path:** If MCTS needs mid-card-resolution cloning, refactor hot-path cards to an effect queue system. The Engine layer abstracts this from the AI layer.

### AI Clonability Strategy

MCTS needs to clone state and explore different choices. With callbacks, we can't clone mid-`on_play` execution (stack frames aren't copyable). Two approaches:

**Approach A (recommended for v1): Clone before card resolution, replay with different choices.** The GameRunner clones GameState before playing a card. To explore a different sub-decision (e.g., different discard choice), replay the card from the clone with a different `DecisionFn` response. Card effects are computationally trivial, so replay overhead is negligible.

**Approach B (for later): Decompose decisions into sequential steps at the Engine layer.** The Engine exposes each sub-decision as a separate step. "Discard 2 of 5" becomes two sequential single-card decisions. The AI sees: step 1 = pick card to discard (5 options), step 2 = pick another (4 options). State is clonable between steps because no callback is in flight.

---

## 5. Game Layer Enhancements

### 5.1 card.h Modifications

#### New ChoiceType values
```cpp
enum class ChoiceType {
    // Existing
    DISCARD,
    TRASH,
    GAIN,
    TOPDECK,
    EXILE,
    // New
    PLAY_CARD,         // Choose a card to play (Throne Room)
    YES_NO,            // Binary choice (reveal Moat? set aside with Library?)
    REVEAL,            // Reveal from hand (Bureaucrat defense)
    MULTI_FATE,        // Per-card trash/discard/keep (Sentry)
    ORDER,             // Choose ordering (topdeck order for Sentry put-back)
    SELECT_FROM_DISCARD, // Choose card from own discard (Harbinger)
};
```

#### New ability slots on Card struct
```cpp
struct Card {
    // Existing fields unchanged
    std::string name;
    int cost;
    CardType types;
    std::string text;
    int victory_points;
    int coin_value;
    std::unordered_set<std::string> tags;
    std::function<void(GameState&, int, DecisionFn)> on_play;

    // NEW: Reaction ability
    // Called when an Attack targets this player and they have this card in hand.
    // Returns true if the attack is blocked for this player.
    // If nullptr, card has no reaction ability.
    std::function<bool(GameState&, int player_id, int attacker_id, DecisionFn)> on_react;

    // NEW: Dynamic VP calculation (Gardens = cards_owned / 10)
    // If set, called during scoring instead of using static victory_points.
    // Receives const GameState& and player_id.
    std::function<int(const GameState&, int player_id)> vp_fn;

    // NEW: When-gain trigger
    // Called when this card is gained by a player.
    std::function<void(GameState&, int player_id, DecisionFn)> on_gain;

    // NEW: When-trash trigger
    // Called when this card is trashed.
    std::function<void(GameState&, int player_id, DecisionFn)> on_trash;

    // NEW: Duration next-turn effect
    // Called at start of the owning player's next turn if card is still in play.
    std::function<void(GameState&, int player_id, DecisionFn)> on_duration;

    // Existing helpers unchanged
    bool is_action() const   { return has_type(types, CardType::Action); }
    // ... etc
};
```

### 5.2 player.h Additions

```cpp
// Peek at top N cards of deck without removing them.
// Shuffles discard into deck if needed.
// Returns card_ids from top (index 0 = topmost).
std::vector<int> peek_deck(int count);

// Remove and return the top card from deck (-1 if empty after shuffle attempt).
int remove_deck_top();

// Trash from hand and return the card_id (the existing trash_from_hand doesn't return it).
// The caller is still responsible for adding to the trash pile.
int trash_from_hand_return(int hand_index);

// Find the index of a card_id in hand (-1 if not found).
int find_in_hand(int card_id) const;

// Get all card_ids in all zones (for scoring, total card count).
std::vector<int> all_cards() const;
```

#### Implementation notes for player.cpp:

```cpp
std::vector<int> Player::peek_deck(int count) {
    // If not enough in deck, shuffle discard in first
    if ((int)deck_.size() < count) {
        if (!discard_.empty()) {
            shuffle_discard_into_deck();
        }
    }
    int actual = std::min(count, (int)deck_.size());
    std::vector<int> result;
    result.reserve(actual);
    // deck_ draws from the back, so "top" = back
    for (int i = 0; i < actual; i++) {
        result.push_back(deck_[deck_.size() - 1 - i]);
    }
    return result;
}

int Player::remove_deck_top() {
    if (deck_.empty()) {
        if (discard_.empty()) return -1;
        shuffle_discard_into_deck();
    }
    if (deck_.empty()) return -1;
    int card_id = deck_.back();
    deck_.pop_back();
    return card_id;
}

int Player::trash_from_hand_return(int hand_index) {
    int card_id = hand_[hand_index];
    hand_.erase(hand_.begin() + hand_index);
    return card_id;
}

int Player::find_in_hand(int card_id) const {
    for (int i = 0; i < (int)hand_.size(); i++) {
        if (hand_[i] == card_id) return i;
    }
    return -1;
}

std::vector<int> Player::all_cards() const {
    std::vector<int> result;
    auto append = [&](const std::vector<int>& v) {
        result.insert(result.end(), v.begin(), v.end());
    };
    append(hand_);
    append(deck_);
    append(discard_);
    append(in_play_);
    for (const auto& [_, cards] : set_aside_) append(cards);
    for (const auto& [_, cards] : mats_) append(cards);
    return result;
}
```

### 5.3 game_state.h Additions

```cpp
// --- New Phase values ---
enum class Phase {
    ACTION,
    TREASURE,    // NEW: Playing treasures (Buy phase part 1)
    BUY,         // Now means: buying cards (Buy phase part 2)
    NIGHT,       // NEW: Night phase
    CLEANUP,
    GAME_OVER
};

// --- New methods on GameState ---

// Attack resolution: iterates over all non-attacker players in turn order.
// For each, opens a reaction window. If not blocked, calls attack_effect.
// The decide callback is passed through for both reactions and attack effects.
void resolve_attack(
    int attacker_id,
    std::function<void(GameState&, int target_id, DecisionFn)> attack_effect,
    DecisionFn decide);

// Play a card's on_play effect by card_id (for Throne Room — card already in play).
// Does NOT move the card. Just calls card_def(card_id)->on_play.
void play_card_effect(int card_id, int player_id, DecisionFn decide);

// Play a card from hand: moves it from hand to in_play, then calls on_play.
// Returns the card_id that was played.
int play_card_from_hand(int player_id, int hand_index, DecisionFn decide);

// Get all supply piles whose top card costs <= max_cost and pile is non-empty.
std::vector<std::string> gainable_piles(int max_cost) const;

// Get all supply piles whose top card costs <= max_cost, is non-empty,
// and has the specified type.
std::vector<std::string> gainable_piles(int max_cost, CardType required_type) const;

// Get the cost of a card by name (for now just card->cost; later supports reductions).
int effective_cost(const std::string& card_name) const;

// Count total cards owned by a player (all zones).
int total_cards_owned(int player_id) const;
```

#### Implementation notes for resolve_attack:

```cpp
void GameState::resolve_attack(
    int attacker_id,
    std::function<void(GameState&, int target_id, DecisionFn)> attack_effect,
    DecisionFn decide)
{
    for (int i = 1; i < num_players(); i++) {
        int target_id = (attacker_id + i) % num_players();
        Player& target = get_player(target_id);
        bool blocked = false;

        // Check for Reaction cards in target's hand
        const auto& hand = target.get_hand();
        for (int h = 0; h < (int)hand.size(); h++) {
            const Card* card = card_def(hand[h]);
            if (card && card->is_reaction() && card->on_react) {
                // Offer the reaction
                if (card->on_react(*this, target_id, attacker_id, decide)) {
                    blocked = true;
                    break;  // Moat-style: one block is enough
                }
            }
        }

        if (!blocked) {
            attack_effect(*this, target_id, decide);
        }
    }
}
```

**Note:** This is a simplified v1 that only handles the common case (one reaction check per opponent, first block wins). The full Dominion rules allow revealing multiple Reactions in sequence, re-checking after other players react, etc. This can be enhanced later.

#### Implementation for play_card_from_hand:

```cpp
int GameState::play_card_from_hand(int player_id, int hand_index, DecisionFn decide) {
    Player& player = get_player(player_id);
    int card_id = player.get_hand()[hand_index];
    player.play_from_hand(hand_index);  // moves to in_play

    const Card* card = card_def(card_id);
    if (card && card->on_play) {
        card->on_play(*this, player_id, decide);
    }
    return card_id;
}
```

### 5.4 Fix calculate_scores for dynamic VP

In `game_state.cpp`, modify the scoring loop:

```cpp
for (int card_id : all_cards) {
    const Card* card = card_def(card_id);
    if (card) {
        if (card->vp_fn) {
            scores[p] += card->vp_fn(*this, p);
        } else {
            scores[p] += card->victory_points;
        }
    }
}
```

### 5.5 Fix winner() tiebreaker

Current implementation returns -1 for any tie. Dominion tiebreaker: player with fewer turns wins. Need to track turns per player:

```cpp
// Add to GameState private:
std::vector<int> turns_taken_;  // per player

// In advance_phase CLEANUP case, before advancing current_player_:
turns_taken_[current_player_]++;

// In winner():
// If tied on score, compare turns_taken_[i] — fewer turns wins
// If still tied, shared victory (-1)
```

---

## 6. Engine Layer Design

### 6.1 Action Space (`src/engine/action_space.h`)

```cpp
#pragma once

#include "../game/game_state.h"
#include <vector>
#include <string>

// What kind of top-level or sub-decision the player faces
enum class DecisionType : uint8_t {
    PLAY_ACTION,          // Choose action from hand to play (or pass to buy phase)
    PLAY_TREASURE,        // Choose treasure from hand to play (or stop)
    PLAY_ALL_TREASURES,   // Shortcut: play all basic treasures at once
    BUY_CARD,             // Choose supply pile to buy from (or pass)
    CHOOSE_CARDS_IN_HAND, // Sub-decision: select card(s) from hand
    CHOOSE_SUPPLY_PILE,   // Sub-decision: pick a supply pile to gain from
    CHOOSE_OPTION,        // Sub-decision: pick from explicit numbered options
    REACT,                // Whether to reveal/use a reaction card
    PLAY_NIGHT,           // Choose night card from hand to play (or pass)
};

struct ActionOption {
    int local_id;         // 0-based index within this decision
    int card_id;          // Physical card ID if relevant (-1 otherwise)
    std::string card_name;// Card name if relevant
    std::string label;    // Human-readable: "Play Smithy", "Buy Province", "Pass"
    bool is_pass;         // True for pass/stop/skip option
};

struct DecisionPoint {
    int player_id;
    DecisionType type;
    std::vector<ActionOption> options;  // Legal options
    std::string source_card;           // What card caused this (empty for phase decisions)
    ChoiceType sub_choice_type;        // For CHOOSE_CARDS_IN_HAND: discard/trash/etc.
    int min_choices;
    int max_choices;
};

// Build the appropriate decision point for the current game phase
DecisionPoint build_phase_decision(const GameState& state);

// Build decision for action phase: actions in hand + "End Actions"
DecisionPoint build_action_decision(const GameState& state);

// Build decision for treasure phase: treasures in hand + "Play All" + "Stop"
DecisionPoint build_treasure_decision(const GameState& state);

// Build decision for buy phase: affordable piles + "End Buys"
DecisionPoint build_buy_decision(const GameState& state);
```

### 6.2 Game Runner (`src/engine/game_runner.h`)

```cpp
#pragma once

#include "action_space.h"
#include "../game/game_state.h"
#include "../game/cards/base_cards.h"

#include <memory>
#include <vector>
#include <string>

// Abstract agent interface: given a decision point and game state, return a choice
class Agent {
public:
    virtual ~Agent() = default;

    // Given a decision point, return indices of chosen options.
    // For single-select decisions, return a vector with one element.
    // For pass/skip, return the index of the pass option.
    virtual std::vector<int> decide(
        const DecisionPoint& dp,
        const GameState& state) = 0;
};

// Random agent: picks uniformly from legal options
class RandomAgent : public Agent {
public:
    RandomAgent(uint64_t seed = 42);
    std::vector<int> decide(const DecisionPoint& dp, const GameState& state) override;
private:
    std::mt19937 rng_;
};

// Big Money agent: buy Province > Gold > Silver, play all treasures, skip actions
class BigMoneyAgent : public Agent {
public:
    std::vector<int> decide(const DecisionPoint& dp, const GameState& state) override;
};

struct GameResult {
    std::vector<int> scores;
    int winner;           // player_id or -1 for tie
    int total_turns;      // total turns across all players
    int total_decisions;  // total decision points encountered
};

class GameRunner {
public:
    GameRunner(int num_players, const std::vector<std::string>& kingdom_cards);

    // Run a complete game. Agents vector must have num_players entries.
    GameResult run(std::vector<Agent*> agents);

    // Access state (for debugging/testing)
    const GameState& state() const;

private:
    GameState state_;
    int num_players_;
    std::vector<std::string> kingdom_cards_;

    // Create a DecisionFn that delegates to an Agent.
    // This bridges the callback world (card on_play) to the Agent interface.
    DecisionFn make_decision_fn(Agent* agent);

    // Run one complete turn for the current player
    void run_turn(Agent* agent);

    // Sub-phases
    void run_action_phase(Agent* agent);
    void run_treasure_phase(Agent* agent);
    void run_buy_phase(Agent* agent);
    void run_cleanup_phase();
};
```

#### GameRunner implementation sketch:

```cpp
GameResult GameRunner::run(std::vector<Agent*> agents) {
    // Setup
    BaseCards::register_all();
    // Register kingdom cards would go here
    BaseCards::setup_supply(state_, kingdom_cards_);
    BaseCards::setup_starting_decks(state_);
    state_.start_game();

    int total_decisions = 0;

    while (!state_.is_game_over()) {
        int pid = state_.current_player_id();
        run_turn(agents[pid]);
    }

    return {
        state_.calculate_scores(),
        state_.winner(),
        state_.turn_number(),
        total_decisions
    };
}

void GameRunner::run_turn(Agent* agent) {
    state_.start_turn();            // phase = ACTION, actions=1, buys=1, coins=0
    run_action_phase(agent);        // play action cards until out of actions or pass
    run_treasure_phase(agent);      // play treasure cards
    run_buy_phase(agent);           // buy cards
    run_cleanup_phase();            // discard + draw 5
    // advance_phase to handle next player transition + game end check
}

void GameRunner::run_action_phase(Agent* agent) {
    while (state_.actions() > 0 && state_.current_phase() == Phase::ACTION) {
        DecisionPoint dp = build_action_decision(state_);

        // If no action cards in hand, auto-pass
        if (dp.options.size() <= 1) break; // only "End Actions" option

        auto choice = agent->decide(dp, state_);
        int chosen = choice[0];

        if (dp.options[chosen].is_pass) break;

        // Play the chosen action
        int card_id = dp.options[chosen].card_id;
        Player& player = state_.get_player(state_.current_player_id());
        int hand_idx = player.find_in_hand(card_id);

        state_.add_actions(-1);  // costs 1 action
        DecisionFn decide = make_decision_fn(agent);
        state_.play_card_from_hand(state_.current_player_id(), hand_idx, decide);
    }
    // Transition to treasure phase
    // state_ phase management...
}

DecisionFn GameRunner::make_decision_fn(Agent* agent) {
    return [this, agent](int player_id, ChoiceType choice_type,
                          const std::vector<int>& options,
                          int min_choices, int max_choices) -> std::vector<int> {
        // Build a DecisionPoint from the callback parameters
        DecisionPoint dp;
        dp.player_id = player_id;
        dp.type = DecisionType::CHOOSE_CARDS_IN_HAND; // or derive from choice_type
        dp.sub_choice_type = choice_type;
        dp.min_choices = min_choices;
        dp.max_choices = max_choices;

        for (int i = 0; i < (int)options.size(); i++) {
            ActionOption opt;
            opt.local_id = i;
            opt.card_id = options[i]; // might be hand indices, might be card_ids
            opt.is_pass = false;
            dp.options.push_back(opt);
        }

        return agent->decide(dp, state_);
    };
}
```

### 6.3 Rules (`src/engine/rules.h`)

This file handles validation and rule enforcement:

```cpp
#pragma once
#include "../game/game_state.h"

namespace Rules {
    // Can this player play this card from hand in the current phase?
    bool can_play_card(const GameState& state, int player_id, int card_id);

    // Can this player buy from this supply pile?
    bool can_buy(const GameState& state, int player_id, const std::string& pile_name);

    // Get all cards in hand that can be played in current phase
    std::vector<int> playable_cards(const GameState& state, int player_id);

    // Get all supply piles that can be bought
    std::vector<std::string> buyable_piles(const GameState& state, int player_id);
}
```

---

## 7. Card Implementation Framework

### Pattern for each card

Every kingdom card follows this template:

```cpp
CardRegistry::register_card({
    .name = "CardName",
    .cost = N,
    .types = CardType::Action,  // | CardType::Attack, etc.
    .text = "Card text here",
    .victory_points = 0,
    .coin_value = 0,
    .tags = {},
    .on_play = [](GameState& state, int pid, DecisionFn decide) {
        // Card effect implementation
    },
    .on_react = nullptr,  // or reaction lambda
    .vp_fn = nullptr,     // or dynamic VP lambda
    .on_gain = nullptr,
    .on_trash = nullptr,
    .on_duration = nullptr,
});
```

### Common patterns in card implementations

#### Pattern: Simple stat boosts
```cpp
// Village: +1 Card, +2 Actions
.on_play = [](GameState& state, int pid, DecisionFn) {
    state.get_player(pid).draw_cards(1);
    state.add_actions(2);
}
```

#### Pattern: Player chooses cards from hand (discard/trash)
```cpp
// Chapel: Trash up to 4 cards from your hand
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    Player& player = state.get_player(pid);
    if (player.hand_size() == 0) return;

    std::vector<int> options;
    for (int i = 0; i < player.hand_size(); i++) {
        options.push_back(i);
    }

    auto chosen = decide(pid, ChoiceType::TRASH, options, 0, std::min(4, player.hand_size()));

    // IMPORTANT: sort descending before erasing by index
    std::sort(chosen.begin(), chosen.end(), std::greater<int>());
    for (int idx : chosen) {
        int card_id = player.trash_from_hand_return(idx);
        state.trash_card(card_id);
    }
}
```

#### Pattern: Gain a card from supply with cost constraint
```cpp
// Workshop: Gain a card costing up to 4
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    auto piles = state.gainable_piles(4);
    if (piles.empty()) return;

    std::vector<int> options;
    for (int i = 0; i < (int)piles.size(); i++) {
        options.push_back(i);
    }

    auto chosen = decide(pid, ChoiceType::GAIN, options, 1, 1);
    if (!chosen.empty()) {
        state.gain_card(pid, piles[chosen[0]]);
    }
}
```

#### Pattern: Attack with opponent discard
```cpp
// Militia: +2 Coins. Each other player discards down to 3.
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    state.add_coins(2);

    state.resolve_attack(pid, [](GameState& st, int target, DecisionFn dec) {
        Player& p = st.get_player(target);
        while (p.hand_size() > 3) {
            std::vector<int> options;
            for (int i = 0; i < p.hand_size(); i++) {
                options.push_back(i);
            }
            int must_discard = p.hand_size() - 3;
            auto chosen = dec(target, ChoiceType::DISCARD, options, must_discard, must_discard);
            std::sort(chosen.begin(), chosen.end(), std::greater<int>());
            for (int idx : chosen) {
                p.discard_from_hand(idx);
            }
        }
    }, decide);
}
```

#### Pattern: Reaction to block attack
```cpp
// Moat: Reaction — reveal to block
.on_react = [](GameState& state, int pid, int attacker_id, DecisionFn decide) -> bool {
    // 0 = don't reveal, 1 = reveal (block)
    auto chosen = decide(pid, ChoiceType::YES_NO, {0, 1}, 1, 1);
    return !chosen.empty() && chosen[0] == 1;
}
```

#### Pattern: Throne Room (recursive card playing)
```cpp
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    Player& player = state.get_player(pid);

    // Find action cards in hand
    std::vector<int> action_indices;
    for (int i = 0; i < player.hand_size(); i++) {
        const Card* card = state.card_def(player.get_hand()[i]);
        if (card && card->is_action()) {
            action_indices.push_back(i);
        }
    }

    if (action_indices.empty()) return;

    auto chosen = decide(pid, ChoiceType::PLAY_CARD, action_indices, 0, 1);
    if (chosen.empty()) return;

    int hand_idx = chosen[0];
    int card_id = player.get_hand()[hand_idx];
    player.play_from_hand(hand_idx);  // move to in_play

    const Card* target = state.card_def(card_id);
    if (target && target->on_play) {
        target->on_play(state, pid, decide);  // first play
        target->on_play(state, pid, decide);  // second play
    }
}
```

#### Pattern: Dynamic VP
```cpp
// Gardens: Worth 1 VP per 10 cards you have (rounded down)
.vp_fn = [](const GameState& state, int pid) -> int {
    return state.total_cards_owned(pid) / 10;
}
```

---

## 8. Base Kingdom Card Implementations

All 26 Base Game (2nd Edition) kingdom cards, organized by implementation wave.

### Wave 1: Simple Effects (no decisions needed)

**Smithy** (cost 4, Action): +3 Cards
```cpp
.on_play = [](GameState& state, int pid, DecisionFn) {
    state.get_player(pid).draw_cards(3);
}
```

**Village** (cost 3, Action): +1 Card, +2 Actions
```cpp
.on_play = [](GameState& state, int pid, DecisionFn) {
    state.get_player(pid).draw_cards(1);
    state.add_actions(2);
}
```

**Laboratory** (cost 5, Action): +2 Cards, +1 Action
```cpp
.on_play = [](GameState& state, int pid, DecisionFn) {
    state.get_player(pid).draw_cards(2);
    state.add_actions(1);
}
```

**Festival** (cost 5, Action): +2 Actions, +1 Buy, +2 Coins
```cpp
.on_play = [](GameState& state, int pid, DecisionFn) {
    state.add_actions(2);
    state.add_buys(1);
    state.add_coins(2);
}
```

**Market** (cost 5, Action): +1 Card, +1 Action, +1 Buy, +1 Coin
```cpp
.on_play = [](GameState& state, int pid, DecisionFn) {
    state.get_player(pid).draw_cards(1);
    state.add_actions(1);
    state.add_buys(1);
    state.add_coins(1);
}
```

### Wave 2: Simple Choices

**Cellar** (cost 2, Action): +1 Action. Discard any number, draw that many.
```cpp
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    state.add_actions(1);
    Player& player = state.get_player(pid);
    if (player.hand_size() == 0) return;

    std::vector<int> options;
    for (int i = 0; i < player.hand_size(); i++) options.push_back(i);

    auto chosen = decide(pid, ChoiceType::DISCARD, options, 0, player.hand_size());
    std::sort(chosen.begin(), chosen.end(), std::greater<int>());
    int count = (int)chosen.size();
    for (int idx : chosen) player.discard_from_hand(idx);
    player.draw_cards(count);
}
```

**Chapel** (cost 2, Action): Trash up to 4 cards from your hand.
```cpp
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    Player& player = state.get_player(pid);
    if (player.hand_size() == 0) return;

    std::vector<int> options;
    for (int i = 0; i < player.hand_size(); i++) options.push_back(i);

    auto chosen = decide(pid, ChoiceType::TRASH, options, 0, std::min(4, player.hand_size()));
    std::sort(chosen.begin(), chosen.end(), std::greater<int>());
    for (int idx : chosen) {
        int cid = player.trash_from_hand_return(idx);
        state.trash_card(cid);
    }
}
```

**Workshop** (cost 3, Action): Gain a card costing up to 4.
```cpp
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    auto piles = state.gainable_piles(4);
    if (piles.empty()) return;

    std::vector<int> options;
    for (int i = 0; i < (int)piles.size(); i++) options.push_back(i);

    auto chosen = decide(pid, ChoiceType::GAIN, options, 1, 1);
    if (!chosen.empty()) state.gain_card(pid, piles[chosen[0]]);
}
```

**Moneylender** (cost 4, Action): You may trash a Copper from your hand for +3 Coins.
```cpp
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    Player& player = state.get_player(pid);
    // Find Coppers in hand
    int copper_idx = -1;
    for (int i = 0; i < player.hand_size(); i++) {
        if (state.card_name(player.get_hand()[i]) == "Copper") {
            copper_idx = i;
            break;
        }
    }
    if (copper_idx == -1) return;

    // "You may" — offer yes/no choice
    auto chosen = decide(pid, ChoiceType::YES_NO, {0, 1}, 1, 1);
    if (!chosen.empty() && chosen[0] == 1) {
        int cid = player.trash_from_hand_return(copper_idx);
        state.trash_card(cid);
        state.add_coins(3);
    }
}
```

**Poacher** (cost 4, Action): +1 Card, +1 Action, +1 Coin. Discard a card per empty Supply pile.
```cpp
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    state.get_player(pid).draw_cards(1);
    state.add_actions(1);
    state.add_coins(1);

    int empty_piles = state.get_supply().empty_pile_count();
    Player& player = state.get_player(pid);
    int to_discard = std::min(empty_piles, player.hand_size());

    if (to_discard > 0) {
        std::vector<int> options;
        for (int i = 0; i < player.hand_size(); i++) options.push_back(i);

        auto chosen = decide(pid, ChoiceType::DISCARD, options, to_discard, to_discard);
        std::sort(chosen.begin(), chosen.end(), std::greater<int>());
        for (int idx : chosen) player.discard_from_hand(idx);
    }
}
```

**Vassal** (cost 3, Action): +2 Coins. Discard the top card of your deck. If it's an Action, you may play it.
```cpp
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    state.add_coins(2);
    Player& player = state.get_player(pid);

    int card_id = player.remove_deck_top();
    if (card_id == -1) return;

    player.add_to_discard(card_id);

    const Card* card = state.card_def(card_id);
    if (card && card->is_action()) {
        auto chosen = decide(pid, ChoiceType::YES_NO, {0, 1}, 1, 1);
        if (!chosen.empty() && chosen[0] == 1) {
            // Move from discard to in_play, then execute
            // Find it in discard (it's the last card added)
            auto& discard = player.get_discard();
            // Need to remove from discard — add a helper or handle inline
            // For now: remove last card from discard, add to in_play
            // This requires a new Player method or direct manipulation
            player.add_to_hand(card_id);  // temp: move to hand
            // Find index in hand
            int hidx = player.find_in_hand(card_id);
            player.play_from_hand(hidx);  // move to in_play
            // But we need to remove from discard first!
            // BETTER APPROACH: add remove_from_discard(card_id) to Player
            // For implementation: need Player::remove_from_discard(int card_id)
            if (card->on_play) {
                card->on_play(state, pid, decide);
            }
        }
    }
}
```

**NOTE:** Vassal reveals a need for `Player::remove_from_discard(int card_id)` — add this to the Player additions list.

**Harbinger** (cost 3, Action): +1 Card, +1 Action. Look through your discard pile. You may put a card from it onto your deck.
```cpp
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    state.get_player(pid).draw_cards(1);
    state.add_actions(1);

    Player& player = state.get_player(pid);
    const auto& discard = player.get_discard();
    if (discard.empty()) return;

    // Options: indices into discard pile, or "skip"
    std::vector<int> options;
    for (int i = 0; i < (int)discard.size(); i++) options.push_back(i);

    // min=0 means they can choose nothing (skip)
    auto chosen = decide(pid, ChoiceType::TOPDECK, options, 0, 1);
    if (!chosen.empty()) {
        int discard_idx = chosen[0];
        int card_id = discard[discard_idx];
        // Need Player::remove_from_discard_by_index(int idx)
        player.add_to_deck_top(card_id);
    }
}
```

**NOTE:** Harbinger needs `Player::remove_from_discard_by_index(int idx)`.

### Wave 3: Complex Multi-Step Choices

**Remodel** (cost 4, Action): Trash a card from your hand. Gain a card costing up to 2 more than it.
```cpp
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    Player& player = state.get_player(pid);
    if (player.hand_size() == 0) return;

    // Choose card to trash
    std::vector<int> trash_options;
    for (int i = 0; i < player.hand_size(); i++) trash_options.push_back(i);

    auto trash_chosen = decide(pid, ChoiceType::TRASH, trash_options, 1, 1);
    if (trash_chosen.empty()) return;

    int hand_idx = trash_chosen[0];
    int trashed_id = player.trash_from_hand_return(hand_idx);
    const Card* trashed = state.card_def(trashed_id);
    state.trash_card(trashed_id);

    if (!trashed) return;
    int max_cost = trashed->cost + 2;

    auto piles = state.gainable_piles(max_cost);
    if (piles.empty()) return;

    std::vector<int> gain_options;
    for (int i = 0; i < (int)piles.size(); i++) gain_options.push_back(i);

    auto gain_chosen = decide(pid, ChoiceType::GAIN, gain_options, 1, 1);
    if (!gain_chosen.empty()) {
        state.gain_card(pid, piles[gain_chosen[0]]);
    }
}
```

**Mine** (cost 5, Action): You may trash a Treasure from your hand. Gain a Treasure to your hand costing up to 3 more.
```cpp
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    Player& player = state.get_player(pid);

    // Find treasures in hand
    std::vector<int> treasure_indices;
    for (int i = 0; i < player.hand_size(); i++) {
        const Card* c = state.card_def(player.get_hand()[i]);
        if (c && c->is_treasure()) treasure_indices.push_back(i);
    }
    if (treasure_indices.empty()) return;

    auto trash_chosen = decide(pid, ChoiceType::TRASH, treasure_indices, 0, 1);
    if (trash_chosen.empty()) return;

    int hand_idx = trash_chosen[0];
    int trashed_id = player.trash_from_hand_return(hand_idx);
    const Card* trashed = state.card_def(trashed_id);
    state.trash_card(trashed_id);

    if (!trashed) return;
    int max_cost = trashed->cost + 3;

    auto piles = state.gainable_piles(max_cost, CardType::Treasure);
    if (piles.empty()) return;

    std::vector<int> gain_options;
    for (int i = 0; i < (int)piles.size(); i++) gain_options.push_back(i);

    auto gain_chosen = decide(pid, ChoiceType::GAIN, gain_options, 1, 1);
    if (!gain_chosen.empty()) {
        state.gain_card_to_hand(pid, piles[gain_chosen[0]]);
    }
}
```

**Artisan** (cost 6, Action): Gain a card to your hand costing up to 5. Put a card from your hand onto your deck.
```cpp
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    // Gain to hand
    auto piles = state.gainable_piles(5);
    if (!piles.empty()) {
        std::vector<int> gain_opts;
        for (int i = 0; i < (int)piles.size(); i++) gain_opts.push_back(i);
        auto chosen = decide(pid, ChoiceType::GAIN, gain_opts, 1, 1);
        if (!chosen.empty()) state.gain_card_to_hand(pid, piles[chosen[0]]);
    }

    // Topdeck a card from hand
    Player& player = state.get_player(pid);
    if (player.hand_size() > 0) {
        std::vector<int> topdeck_opts;
        for (int i = 0; i < player.hand_size(); i++) topdeck_opts.push_back(i);
        auto chosen = decide(pid, ChoiceType::TOPDECK, topdeck_opts, 1, 1);
        if (!chosen.empty()) {
            int cid = player.get_hand()[chosen[0]];
            // Remove from hand, add to deck top
            player.discard_from_hand(chosen[0]); // Wrong! This discards, not topdecks
            // Need: player removes from hand without discarding
            // Use: remove card from hand manually, add to deck top
            // Better: add Player::topdeck_from_hand(int hand_index)
        }
    }
}
```

**NOTE:** Need `Player::topdeck_from_hand(int hand_index)` — removes from hand, adds to deck top.

**Sentry** (cost 5, Action): +1 Card, +1 Action. Look at the top 2 cards of your deck. Trash and/or discard any number of them. Put the rest back on top in any order.
```cpp
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    state.get_player(pid).draw_cards(1);
    state.add_actions(1);

    Player& player = state.get_player(pid);
    auto top_cards = player.peek_deck(2);
    // Remove them from deck temporarily
    for (int i = 0; i < (int)top_cards.size(); i++) player.remove_deck_top();

    // For each card, decide: 0=keep, 1=discard, 2=trash
    for (int card_id : top_cards) {
        auto chosen = decide(pid, ChoiceType::MULTI_FATE, {0, 1, 2}, 1, 1);
        int fate = chosen.empty() ? 0 : chosen[0];
        if (fate == 1) {
            player.add_to_discard(card_id);
        } else if (fate == 2) {
            state.trash_card(card_id);
        }
        // fate == 0: keep for putting back
    }

    // Collect kept cards
    // (This needs rethinking — we need to track which were kept)
    // Implementation: collect kept cards in a vector, then if >1, ask for order
    // Simplified approach: decide order for all kept cards
}
```

**NOTE:** Sentry's implementation needs careful tracking of "kept" cards and an ordering decision. The `MULTI_FATE` ChoiceType will need the decide callback to pass card_id context.

**Library** (cost 5, Action): Draw until you have 7 cards in hand, skipping any Action cards you choose to; set those aside, discarding them afterwards.
```cpp
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    Player& player = state.get_player(pid);
    std::vector<int> set_aside;

    while (player.hand_size() < 7) {
        int card_id = player.remove_deck_top();
        if (card_id == -1) break; // deck + discard exhausted

        const Card* card = state.card_def(card_id);
        if (card && card->is_action()) {
            // "you may set aside" — offer choice
            auto chosen = decide(pid, ChoiceType::YES_NO, {0, 1}, 1, 1);
            if (!chosen.empty() && chosen[0] == 1) {
                set_aside.push_back(card_id);
                continue;
            }
        }
        player.add_to_hand(card_id);
    }

    // Discard all set-aside cards
    for (int cid : set_aside) {
        player.add_to_discard(cid);
    }
}
```

**Bureaucrat** (cost 4, Action-Attack): Gain a Silver onto your deck. Each other player reveals a Victory card from their hand and puts it onto their deck (or reveals a hand with no Victory cards).
```cpp
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    state.gain_card_to_deck_top(pid, "Silver");

    state.resolve_attack(pid, [](GameState& st, int target, DecisionFn dec) {
        Player& p = st.get_player(target);

        // Find Victory cards in hand
        std::vector<int> victory_indices;
        for (int i = 0; i < p.hand_size(); i++) {
            const Card* c = st.card_def(p.get_hand()[i]);
            if (c && c->is_victory()) victory_indices.push_back(i);
        }

        if (victory_indices.empty()) {
            // Reveal hand (no Victory cards) — informational, no game effect
            return;
        }

        if (victory_indices.size() == 1) {
            // Only one option, auto-topdeck
            p.topdeck_from_hand(victory_indices[0]);
        } else {
            auto chosen = dec(target, ChoiceType::REVEAL, victory_indices, 1, 1);
            if (!chosen.empty()) {
                p.topdeck_from_hand(chosen[0]);
            }
        }
    }, decide);
}
```

### Wave 4: Attacks, Reactions, Recursion

**Moat** (cost 2, Action-Reaction): +2 Cards. // When another player plays an Attack, you may reveal this from your hand to be unaffected.
```cpp
.on_play = [](GameState& state, int pid, DecisionFn) {
    state.get_player(pid).draw_cards(2);
},
.on_react = [](GameState& state, int pid, int attacker_id, DecisionFn decide) -> bool {
    auto chosen = decide(pid, ChoiceType::YES_NO, {0, 1}, 1, 1);
    return !chosen.empty() && chosen[0] == 1;
}
```

**Militia** (cost 4, Action-Attack): +2 Coins. Each other player discards down to 3 cards in hand.
```cpp
// (see pattern in section 7 above)
```

**Bandit** (cost 5, Action-Attack): Gain a Gold. Each other player reveals the top 2 cards of their deck, trashes a revealed Treasure other than Copper, and discards the rest.
```cpp
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    state.gain_card(pid, "Gold");  // gain to discard

    state.resolve_attack(pid, [](GameState& st, int target, DecisionFn dec) {
        Player& p = st.get_player(target);
        auto revealed = p.peek_deck(2);
        // Remove from deck
        for (size_t i = 0; i < revealed.size(); i++) p.remove_deck_top();

        // Find trashable treasures (non-Copper)
        std::vector<int> trashable;
        for (int i = 0; i < (int)revealed.size(); i++) {
            const Card* c = st.card_def(revealed[i]);
            if (c && c->is_treasure() && c->name != "Copper") {
                trashable.push_back(i);
            }
        }

        int trashed_idx = -1;
        if (trashable.size() == 1) {
            trashed_idx = trashable[0];
        } else if (trashable.size() > 1) {
            auto chosen = dec(target, ChoiceType::TRASH, trashable, 1, 1);
            if (!chosen.empty()) trashed_idx = chosen[0];
        }

        for (int i = 0; i < (int)revealed.size(); i++) {
            if (i == trashed_idx) {
                st.trash_card(revealed[i]);
            } else {
                p.add_to_discard(revealed[i]);
            }
        }
    }, decide);
}
```

**Witch** (cost 5, Action-Attack): +2 Cards. Each other player gains a Curse.
```cpp
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    state.get_player(pid).draw_cards(2);

    state.resolve_attack(pid, [](GameState& st, int target, DecisionFn) {
        st.gain_card(target, "Curse");
    }, decide);
}
```

**Council Room** (cost 5, Action): +4 Cards, +1 Buy. Each other player draws a card.
```cpp
.on_play = [](GameState& state, int pid, DecisionFn) {
    state.get_player(pid).draw_cards(4);
    state.add_buys(1);

    for (int i = 0; i < state.num_players(); i++) {
        if (i != pid) {
            state.get_player(i).draw_cards(1);
        }
    }
}
```

**Throne Room** (cost 4, Action): You may play an Action card from your hand twice.
```cpp
// (see pattern in section 7 above)
```

**Gardens** (cost 4, Victory): Worth 1 VP per 10 cards you have (round down).
```cpp
.types = CardType::Victory,
.victory_points = 0,
.vp_fn = [](const GameState& state, int pid) -> int {
    return state.total_cards_owned(pid) / 10;
}
```

### Wave 5: Triggered Ability Pattern

**Merchant** (cost 3, Action): +1 Card, +1 Action. The first time you play a Silver this turn, +1 Coin.

This card is unique in the Base set because it sets up a conditional trigger for a future event within the same turn. Implementation options:

**Option A (simplest): Track via GameState turn-level state**
Add a `std::unordered_map<std::string, int>` for per-turn flags on GameState. Merchant sets a flag. When Silver is played, check for the flag.

**Option B: Modify Treasure playing to check for Merchant triggers**
The GameRunner's treasure-playing loop checks if Merchant is in play and a Silver hasn't been played yet this turn.

**Recommended: Option B** — check during treasure resolution in the GameRunner.

```cpp
// In Merchant's on_play:
.on_play = [](GameState& state, int pid, DecisionFn) {
    state.get_player(pid).draw_cards(1);
    state.add_actions(1);
    // The "+1 Coin on first Silver" is tracked by the engine
    // Mark that a Merchant is active this turn
    state.set_turn_flag("merchant_active", state.get_turn_flag("merchant_active") + 1);
}

// In Silver's play (or in GameRunner's treasure resolution):
// After resolving Silver's on_play, check:
// if state.get_turn_flag("merchant_active") > 0 && !state.get_turn_flag("merchant_silver_triggered"):
//     state.add_coins(1 * merchant_count_not_yet_triggered)
//     state.set_turn_flag("merchant_silver_triggered", true)
```

**NOTE:** This requires adding turn-level flag tracking to GameState:
```cpp
// game_state.h
std::unordered_map<std::string, int> turn_flags_;
int get_turn_flag(const std::string& key) const;
void set_turn_flag(const std::string& key, int value);
// Clear all flags in start_turn()
```

### Additional Player methods needed (discovered during card implementation)

```cpp
// Remove a card from discard by card_id. Returns true if found.
bool remove_from_discard(int card_id);

// Remove a card from discard by index. Returns the card_id.
int remove_from_discard_by_index(int idx);

// Move card from hand to deck top (for Artisan, Bureaucrat).
void topdeck_from_hand(int hand_index);
```

Implementations:
```cpp
bool Player::remove_from_discard(int card_id) {
    auto it = std::find(discard_.begin(), discard_.end(), card_id);
    if (it == discard_.end()) return false;
    discard_.erase(it);
    return true;
}

int Player::remove_from_discard_by_index(int idx) {
    int card_id = discard_[idx];
    discard_.erase(discard_.begin() + idx);
    return card_id;
}

void Player::topdeck_from_hand(int hand_index) {
    int card_id = hand_[hand_index];
    hand_.erase(hand_.begin() + hand_index);
    deck_.push_back(card_id);  // top of deck = back of vector
}
```

---

## 9. Testing Strategy

### Framework: Catch2 v3

Fetch via CMake FetchContent. Uses `TEST_CASE` with `SECTION` blocks, `REQUIRE`/`CHECK` assertions.

### Test Helper Class

```cpp
// tests/test_helpers.h
#pragma once

#include "game/game_state.h"
#include "game/card.h"
#include "game/cards/base_cards.h"
#include <queue>
#include <vector>
#include <string>

class TestGame {
public:
    // Creates a game with N players. Calls BaseCards::register_all().
    // Does NOT call setup_supply or setup_starting_decks.
    explicit TestGame(int num_players = 2);

    // Set a player's hand to exactly these cards (clears existing hand).
    // Creates new card IDs via state_.create_card().
    void set_hand(int pid, const std::vector<std::string>& card_names);

    // Set a player's deck. Last element = top of deck (drawn first).
    void set_deck(int pid, const std::vector<std::string>& card_names);

    // Set a player's discard pile.
    void set_discard(int pid, const std::vector<std::string>& card_names);

    // Set all players' RNG to a fixed seed (deterministic shuffles).
    void set_seed(uint64_t seed);

    // Setup a standard supply with the given kingdom cards.
    void setup_supply(const std::vector<std::string>& kingdom_cards = {});

    // Create a DecisionFn that returns pre-scripted choices in order.
    // Each element of `choices` is the return value for one decide() call.
    // If exhausted, falls back to returning minimal valid choice.
    static DecisionFn scripted_decisions(std::vector<std::vector<int>> choices);

    // DecisionFn that always picks the minimum number of options (often 0 = skip).
    static DecisionFn minimal_decisions();

    // DecisionFn that always picks the maximum number of first-available options.
    static DecisionFn greedy_decisions();

    // Play a card from player's hand by name. Finds it, calls play_card_from_hand.
    void play_card(int pid, const std::string& card_name, DecisionFn decide);

    // Access state
    GameState& state();
    const GameState& state() const;

    // Helper: get card names in a player's hand
    std::vector<std::string> hand_names(int pid) const;
    std::vector<std::string> deck_names(int pid) const;
    std::vector<std::string> discard_names(int pid) const;

private:
    GameState state_;

    // Clear a player's zone and populate with new card IDs
    void populate_zone(int pid, const std::vector<std::string>& names,
                       void (Player::*add_fn)(int));
};
```

### Implementation of scripted_decisions

```cpp
DecisionFn TestGame::scripted_decisions(std::vector<std::vector<int>> choices) {
    auto q = std::make_shared<std::queue<std::vector<int>>>();
    for (auto& c : choices) q->push(std::move(c));

    return [q](int player_id, ChoiceType type,
               const std::vector<int>& options,
               int min_choices, int max_choices) -> std::vector<int> {
        if (!q->empty()) {
            auto result = q->front();
            q->pop();
            return result;
        }
        // Fallback: return minimum valid choice
        std::vector<int> result;
        for (int i = 0; i < min_choices && i < (int)options.size(); i++) {
            result.push_back(options[i]);
        }
        return result;
    };
}
```

### Test Organization

```
tests/
    test_helpers.h              -- TestGame class
    test_helpers.cpp            -- TestGame implementation
    test_player.cpp             -- Player zone operations, draw, shuffle
    test_supply.cpp             -- Supply pile management, game end detection
    test_game_state.cpp         -- Turn lifecycle, phase transitions, scoring
    test_cards_base.cpp         -- All 26 Base kingdom cards (one TEST_CASE each)
    test_card_interactions.cpp  -- Multi-card combos (TR+Smithy, TR+Militia, etc.)
    test_attack_reactions.cpp   -- Attack + Moat blocking, edge cases
    test_action_space.cpp       -- Legal action enumeration per phase
    test_invariants.cpp         -- Property-based: card conservation, score consistency
    test_perf.cpp               -- Benchmark: games/second (separate executable)
```

### Example Tests

```cpp
// test_cards_base.cpp

TEST_CASE("Smithy draws 3 cards", "[cards][base]") {
    TestGame game;
    game.set_hand(0, {"Smithy", "Copper", "Copper"});
    game.set_deck(0, {"Silver", "Gold", "Estate", "Duchy", "Province"});

    game.play_card(0, "Smithy", TestGame::minimal_decisions());

    auto hand = game.hand_names(0);
    // Smithy moved to in_play. Hand = Copper, Copper + 3 drawn (Province, Duchy, Estate)
    // (deck draws from back, peek returns top-first)
    REQUIRE(hand.size() == 5);
    CHECK(game.state().get_player(0).deck_size() == 2);
}

TEST_CASE("Cellar discard and draw", "[cards][base]") {
    TestGame game;
    game.set_hand(0, {"Cellar", "Estate", "Estate", "Copper", "Silver"});
    game.set_deck(0, {"Gold", "Gold", "Gold", "Gold", "Gold"});

    SECTION("discard 2, draw 2") {
        // Indices after Cellar is played from hand:
        // Remaining hand: Estate(0), Estate(1), Copper(2), Silver(3)
        // Wait — Cellar's +1 Action happens first, then the discard.
        // But the hand indices passed to decide refer to the hand AFTER Cellar left.
        // Choose indices 0 and 1 (two Estates)
        auto decide = TestGame::scripted_decisions({{0, 1}});
        game.play_card(0, "Cellar", decide);

        auto hand = game.hand_names(0);
        // Should have: Copper, Silver, Gold, Gold (4 cards)
        REQUIRE(hand.size() == 4);
        CHECK(game.state().actions() >= 1); // +1 Action from Cellar
    }

    SECTION("discard 0, draw 0") {
        auto decide = TestGame::scripted_decisions({{}});
        game.play_card(0, "Cellar", decide);

        auto hand = game.hand_names(0);
        REQUIRE(hand.size() == 4); // just remaining after Cellar played
    }
}

TEST_CASE("Militia attack with discard", "[cards][base][attack]") {
    TestGame game;
    game.set_hand(0, {"Militia"});
    game.set_hand(1, {"Estate", "Estate", "Copper", "Silver", "Gold"});

    SECTION("opponent discards to 3") {
        // Opponent must discard 2 cards. Choose indices 0, 1 (Estates).
        auto decide = TestGame::scripted_decisions({{0, 1}});
        game.play_card(0, "Militia", decide);

        CHECK(game.state().coins() == 2);
        CHECK(game.state().get_player(1).hand_size() == 3);
    }
}

TEST_CASE("Moat blocks Militia", "[cards][base][reaction]") {
    TestGame game;
    game.set_hand(0, {"Militia"});
    game.set_hand(1, {"Moat", "Estate", "Estate", "Copper", "Silver"});

    // Decision sequence: Moat reaction (YES=1), then no discard needed
    auto decide = TestGame::scripted_decisions({{1}});
    game.play_card(0, "Militia", decide);

    CHECK(game.state().get_player(1).hand_size() == 5); // unaffected
    CHECK(game.state().coins() == 2); // attacker still gets +2
}

TEST_CASE("Throne Room + Smithy draws 6", "[cards][interaction]") {
    TestGame game;
    game.set_hand(0, {"Throne Room", "Smithy", "Copper"});
    game.set_deck(0, {"Gold","Gold","Gold","Gold","Gold","Gold","Gold"});

    // Choose to Throne the Smithy (after TR leaves hand, Smithy is index 0)
    auto decide = TestGame::scripted_decisions({{0}});
    game.play_card(0, "Throne Room", decide);

    // Copper + 6 drawn = 7 cards in hand
    CHECK(game.state().get_player(0).hand_size() == 7);
}

TEST_CASE("Gardens VP scales with deck size", "[cards][base]") {
    TestGame game;
    game.set_hand(0, {"Gardens"});
    // 19 more cards = 20 total
    std::vector<std::string> deck(19, "Copper");
    game.set_deck(0, deck);

    auto scores = game.state().calculate_scores();
    CHECK(scores[0] == 2);  // 20 / 10 = 2 VP
}

// --- Invariant tests ---

TEST_CASE("Card conservation: total cards never change", "[invariant]") {
    // Run random games, after each action verify:
    // sum of all cards in all zones (all players + supply + trash) == total created
}
```

---

## 10. Build System

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.16)
project(DominionZero VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Core engine library ---
add_library(dominion_engine
    src/game/card.cpp
    src/game/player.cpp
    src/game/supply.cpp
    src/game/game_state.cpp
    src/game/cards/base_cards.cpp
    src/game/cards/base_kingdom.cpp
    src/engine/action_space.cpp
    src/engine/rules.cpp
    src/engine/game_runner.cpp
)
target_include_directories(dominion_engine PUBLIC src)
target_compile_options(dominion_engine PRIVATE -Wall -Wextra -Wpedantic)

# --- Main executable ---
add_executable(dominion_zero src/main.cpp)
target_link_libraries(dominion_zero PRIVATE dominion_engine)

# --- Tests ---
option(BUILD_TESTS "Build test suite" ON)
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(dominion_tests
        tests/test_helpers.cpp
        tests/test_player.cpp
        tests/test_supply.cpp
        tests/test_game_state.cpp
        tests/test_cards_base.cpp
        tests/test_card_interactions.cpp
        tests/test_attack_reactions.cpp
        tests/test_action_space.cpp
        tests/test_invariants.cpp
    )
    target_link_libraries(dominion_tests PRIVATE dominion_engine Catch2::Catch2WithMain)
    target_compile_options(dominion_tests PRIVATE -Wall -Wextra)

    include(CTest)
    include(Catch)
    catch_discover_tests(dominion_tests)

    # Performance benchmark (separate, not in CTest)
    add_executable(dominion_bench tests/test_perf.cpp)
    target_link_libraries(dominion_bench PRIVATE dominion_engine Catch2::Catch2WithMain)
endif()
```

---

## 11. Implementation Sequence

### Step 1: Build System + Compilation Check
- Write `CMakeLists.txt`
- Verify existing code compiles
- Files: `CMakeLists.txt`

### Step 2: Game Layer Enhancements
- Extend `Card` struct with new ability slots and ChoiceType values
- Add Player methods: `peek_deck`, `remove_deck_top`, `trash_from_hand_return`, `find_in_hand`, `all_cards`, `remove_from_discard`, `remove_from_discard_by_index`, `topdeck_from_hand`
- Add GameState methods: `resolve_attack`, `play_card_from_hand`, `play_card_effect`, `gainable_piles`, `effective_cost`, `total_cards_owned`, turn flags
- Add Phase::TREASURE, Phase::NIGHT
- Fix `calculate_scores` for dynamic VP
- Fix `winner` for tiebreaker
- Files: `card.h`, `player.h/.cpp`, `game_state.h/.cpp`

### Step 3: Test Infrastructure
- Write `test_helpers.h/.cpp` with TestGame class
- Write basic tests for existing functionality (Player zones, Supply, GameState lifecycle)
- Verify tests compile and pass
- Files: `tests/test_helpers.h/.cpp`, `tests/test_player.cpp`, `tests/test_supply.cpp`, `tests/test_game_state.cpp`

### Step 4: Wave 1 Kingdom Cards (simple effects)
- Smithy, Village, Laboratory, Festival, Market
- Add `base_kingdom.h/.cpp`
- Write tests for each card
- Files: `src/game/cards/base_kingdom.h/.cpp`, `tests/test_cards_base.cpp`

### Step 5: Wave 2 Kingdom Cards (simple choices)
- Cellar, Chapel, Workshop, Moneylender, Poacher, Vassal, Harbinger
- Write tests for each
- Files: same as step 4

### Step 6: Wave 3 Kingdom Cards (complex multi-step)
- Remodel, Mine, Artisan, Sentry, Library, Bureaucrat
- Write tests
- Files: same as step 4

### Step 7: Wave 4 Kingdom Cards (attacks, reactions, recursion)
- Moat, Militia, Bandit, Witch, Council Room, Throne Room, Gardens
- Write interaction tests (TR+Smithy, TR+Militia, Moat+Militia, etc.)
- Files: same as step 4, plus `tests/test_card_interactions.cpp`, `tests/test_attack_reactions.cpp`

### Step 8: Action Space + Game Runner
- Implement `build_action_decision`, `build_treasure_decision`, `build_buy_decision`
- Implement `GameRunner` with turn loop
- Implement `RandomAgent` and `BigMoneyAgent`
- Write action space tests
- Files: `src/engine/action_space.h/.cpp`, `src/engine/game_runner.h/.cpp`, `tests/test_action_space.cpp`

### Step 9: Wave 5 + Integration
- Merchant (turn-flag-based triggered ability)
- Full integration test: complete game from setup to scoring
- Files: `base_kingdom.cpp`

### Step 10: Benchmark + Main
- Write `main.cpp` that runs sample games
- Performance benchmark: 10K random games, measure games/sec
- Target: >1000 games/sec for 2-player base-only games
- Files: `src/main.cpp`, `tests/test_perf.cpp`

---

## 12. Future: AI Layer

Not part of the initial build but documented for architectural awareness.

### GameEnvironment (RL wrapper)
```cpp
class GameEnvironment {
    DecisionPoint reset();                    // Start new game, return first decision
    StepResult step(int action_index);        // Apply action, advance to next decision
    Observation get_observation() const;       // What current player sees
    std::unique_ptr<GameEnvironment> clone();  // Deep copy for MCTS
    void determinize(int player_id, mt19937&); // Randomize hidden info for IS-MCTS
};
```

### Observation Tensor Layout
Per-card-name counts (hand, discard, in_play, supply, trash, costs), per-opponent visible state, scalar features (actions/buys/coins/turn/phase/provinces_left/empty_piles), decision context, legal action mask. ~200 floats for a typical 2-player game.

### MCTS
Information Set MCTS: clone env, determinize hidden info, run standard MCTS with neural network evaluation, aggregate across determinizations.

### Action Space for NN
Each `DecisionPoint` has at most ~64 legal options. Network outputs distribution over `MAX_ACTION_OPTIONS=64` slots, masked by `legal_action_mask`. Sequential decomposition keeps each individual decision small.

### Action Space Deduplication
Critical optimization for MCTS: collapse identical actions by card name. 4 Coppers in hand should be 1 "Play Copper" option, not 4 separate options that lead to identical game states. Applies to:
- **Single-select** (play action, play treasure): deduplicate by card name, pick any matching card ID when executing
- **Multi-select** (discard/trash N cards): deduplicate into distinct card-name combinations. "Discard 2 from Copper×3, Estate×1" has 2 distinct choices: "2 Coppers" or "1 Copper + 1 Estate", not C(4,2)=6 index combinations
- **Buy phase**: already deduplicated (one option per supply pile)

This reduces MCTS branching factor significantly in the common case (hands full of Coppers/Estates).

### MCTS Sub-Decision Strategy
For Throne Room / King's Court type cards, sub-decisions happen mid-card-resolution where state can't be cloned (callback on the call stack). Strategy:
- **Top-level play decisions** (which card to play this turn): searched by MCTS
- **TR/KC sub-decisions** (which action to throne): use greedy "maximize action plays" heuristic — solvable without search by sorting on action yield. Policy network can override at the top level ("should I play TR at all?")
- **Other sub-decisions** (Chapel trash, Militia discard): use heuristic (trash junk, discard victory cards)
- **Fallback**: clone before card resolution, replay with different `DecisionFn` responses. Each replay costs microseconds, viable for 800 MCTS simulations per move

---

## 13. Edge Cases and Gotchas

### From the Rules PDF

1. **"Do as much as you can" rule:** If told to draw 3 but only 2 available (after shuffle), draw 2. Don't error.

2. **Throne Room on nothing:** "You may play an Action" — if no Actions in hand, TR does nothing. Not an error.

3. **Throne Room on a card that trashes itself:** The card is played from hand once (moved to in_play). First play executes. If it trashes itself during first play, second play still happens (resolve the play ability again, but the card is now in trash — "do as much as you can" applies to each effect).

4. **Empty Province pile check:** Game-end check happens at the end of a turn (after Clean-up), not immediately when Province pile empties mid-turn.

5. **Discard index ordering:** When discarding/trashing multiple cards by hand index, ALWAYS sort indices descending before erasing. Erasing index 2 shifts index 3→2, so if you then erase "index 3" you get the wrong card.

6. **Reaction timing:** Reactions resolve BEFORE the Attack. Moat blocks before Militia's discard-down-to-3 ever happens.

7. **Council Room is NOT an Attack:** "Each other player draws a card" is not an attack effect. No reaction window. Players can't block it with Moat.

8. **Witch IS an Attack:** "Each other player gains a Curse" is preceded by a reaction window.

9. **Throne Room stays in play with Duration cards:** If TR plays a Duration, both TR and the Duration stay in play until the Duration's future effects resolve. (Not relevant for base game, but design for it.)

10. **Gaining vs Buying:** Buying a card results in gaining it, but not all gains are buys. When-buy and when-gain are different triggers. Workshop "gains" (no buy trigger). Buying Silver "buys" then "gains" (both triggers).

11. **shuffle timing:** Don't shuffle discard into deck until you NEED a card from deck and it's empty. Not proactively.

12. **Gardens counts ALL cards:** Hand + deck + discard + in_play + set_aside + mats. During scoring, gather everything.

13. **Tiebreaker:** Fewer turns wins. If same turns, shared victory. The "turns" count is per-player (not total game turns).

14. **Treasure playing order:** Can matter with special treasures (not in base game). Still, design to allow order choices.

15. **Can't play Treasures after first Buy:** Once you buy a card in Buy Phase Part 2, you cannot go back to playing Treasures. The engine must enforce this.

### Implementation Pitfalls

1. **Card ID vs hand index confusion:** The `DecisionFn` receives `options` that may be hand indices (0,1,2...) or card_ids depending on context. Be consistent. Recommendation: for CHOOSE_CARDS_IN_HAND, always pass hand indices. For CHOOSE_SUPPLY_PILE, pass pile indices into a pile list.

2. **`card_names_` is an unordered_map<int,string>:** For cloning, this is the most expensive part. Consider switching to `std::vector<std::string>` indexed by card_id (since IDs are sequential) for faster cloning.

3. **Static CardRegistry:** The global registry is fine for now but means card definitions are shared across all games. If future games need per-game card mutations (Capitalism changes types, etc.), the registry will need to become per-game.

4. **DecisionFn captures:** When lambdas passed as `DecisionFn` capture references/pointers, be careful about object lifetimes, especially with `resolve_attack` which creates nested lambdas.

5. **Turn count tracking:** Current code increments `turn_number_` globally. For tiebreaker, need per-player turn tracking. `turn_number_` currently means "total half-turns across all players" — make sure the semantics are clear.

---

## Future Performance Concerns

Issues to address before AlphaZero training at scale. Not blocking now, but worth tracking.

### String-based turn flags
`set_turn_flag` / `get_turn_flag` uses `std::unordered_map<std::string, int>` — string hashing on every call. Options:
- **Enum keys** with `std::unordered_map<TurnFlag, int>`
- **Flat array** with `constexpr int` indices — fastest, zero overhead
- **Eliminate entirely** — encode context into the `DecisionFn` signature or options vector instead of stashing in shared state

### Conditional tracking counters
Counters like `actions_played_` are only needed when specific cards are in the kingdom (e.g., Conspirator). Options:
- **Tracking bitfield** — each card declares what counters it needs at registration, game setup ORs them together, increment is guarded by a single bitwise AND
- **Do nothing** — integer increments are single instructions, unlikely to matter vs allocations and string ops. Profile first.

### Reveal / public information system
Currently no mechanism for agents to know what opponents have revealed. For imperfect-information training this matters. Options:
- **Per-turn event log** on GameState — `std::vector<GameEvent>` with type (REVEAL, GAIN, TRASH), player, and card IDs. Agents query for public info.
- **Retrofit later** — add `state.log_reveal(pid, card_ids)` calls into existing lambdas. No card logic changes needed.

### String-based card names in supply/registry
`CardRegistry::get(name)` and supply pile lookups use string keys throughout. For millions of simulations this adds up. Options:
- **Integer card type IDs** — assign each card definition a numeric ID at registration, use that for all lookups
- **Interned strings** — guarantee pointer equality so comparisons are O(1)
