# How to Implement New Cards in DominionZero

This guide explains how to add new Dominion cards to the engine, following the patterns established in `level_1_cards.cpp` and `level_2_cards.cpp`.

---

## Architecture Overview

Cards are defined as `Card` structs and registered into a global `CardRegistry`. Each card is a data object with lambda callbacks for its effects — there are no card subclasses. The system lives in these files:

| File | Purpose |
|------|---------|
| `src/game/card.h` | `Card` struct, `CardType` enum, `ChoiceType` enum, `DecisionFn` type, `CardRegistry` namespace |
| `src/game/cards/level_N_cards.h` | Declares `LevelNCards::register_all()` |
| `src/game/cards/level_N_cards.cpp` | Card definitions with lambdas |
| `src/game/card_defs.h/.cpp` | Cached IDs for base cards (Copper, Silver, etc.) |
| `src/main.cpp` (and other entry points) | Calls `LevelNCards::register_all()` at startup |

---

## Step-by-Step: Adding a New Card

### 1. Choose the right level file

Levels correspond to implementation complexity. Pick the level file that matches your card's mechanics, or the level you're currently working on (e.g. `level_3_cards.cpp`).

### 2. Add the card registration

Inside the `register_all()` function of your level file, add a `CardRegistry::register_card({...})` call using designated initializers:

```cpp
CardRegistry::register_card({
    .name = "My Card",          // Unique name, used for lookups
    .cost = 4,                  // Buy cost in coins
    .types = CardType::Action,  // Bitmask of types (see below)
    .text = "+2 Cards",         // Flavor/rules text
    .victory_points = 0,        // Static VP (0 for non-Victory cards)
    .coin_value = 0,            // Coin value when played as Treasure (0 for non-Treasures)
    .tags = {},                 // Optional tags: {"Knight"}, {"Looter"}, etc.
    .on_play = [](GameState& state, int pid, DecisionFn decide) {
        // Card effect goes here
    },
    // Optional callbacks (omit or set to nullptr if unused):
    // .on_react = nullptr,
    // .vp_fn = nullptr,
    // .on_gain = nullptr,
    // .on_trash = nullptr,
    // .on_duration = nullptr,
});
```

### 3. Wire up registration (if adding a new level file)

If you created a new `level_N_cards.cpp`, you need to:

1. Create the header `level_N_cards.h`:
   ```cpp
   #pragma once
   namespace LevelNCards {
       void register_all();
   }
   ```

2. Call `LevelNCards::register_all()` in every entry point:
   - `src/main.cpp`
   - `src/stress_test.cpp`
   - `src/interactive.cpp`
   - `src/local_multiplayer.cpp`
   - `src/gui/gui_main.cpp`
   - `tests/test_helpers.cpp`

3. Add the `.cpp` file to your build system (CMakeLists.txt or Makefile).

---

## CardType Bitmask

Types are combined with `|` for multi-type cards:

```cpp
CardType::Action                         // Smithy, Village
CardType::Action | CardType::Attack      // Militia, Witch
CardType::Action | CardType::Reaction    // Moat
CardType::Victory                        // Gardens
CardType::Treasure                       // Copper, Silver, Gold
CardType::Curse                          // Curse
CardType::Duration                       // Duration cards (next-turn effects)
CardType::Night                          // Night phase cards
CardType::Command                        // Command cards
```

Check types with: `has_type(card->types, CardType::Action)` or `card->is_action()`.

---

## The Decision System

When a card needs player input, it calls the `DecisionFn` callback:

```cpp
auto chosen = decide(
    pid,                    // Which player decides
    ChoiceType::DISCARD,    // What kind of choice
    options,                // Vector of valid option indices
    min_choices,            // Minimum selections required
    max_choices             // Maximum selections allowed
);
// Returns: vector<int> of indices the player selected from `options`
```

### ChoiceType Reference

| ChoiceType | When to use | `options` contains |
|------------|-------------|-------------------|
| `DISCARD` | Discard cards from hand | Hand indices (0, 1, 2...) |
| `TRASH` | Trash cards from hand | Hand indices (or card_ids for peek-trash) |
| `GAIN` | Choose a card to gain from supply | Top card_ids of gainable piles |
| `TOPDECK` | Put a card on top of deck | Hand indices |
| `PLAY_CARD` | Choose a card to play (Throne Room) | Hand indices of valid targets |
| `YES_NO` | Binary choice | `{0, 1}` (0=no, 1=yes) |
| `REVEAL` | Reveal a card from hand | Hand indices of valid cards |
| `MULTI_FATE` | Per-card trash/discard/keep (Sentry) | `{0, 1, 2}` (keep/discard/trash) |
| `ORDER` | Choose ordering for topdeck | card_ids to order |
| `SELECT_FROM_DISCARD` | Pick from discard pile | Discard pile indices |
| `EXILE` | Exile a card | Context-dependent |

---

## Common GameState & Player Methods

### Modifying Turn Resources
```cpp
state.add_actions(2);     // +2 Actions
state.add_buys(1);        // +1 Buy
state.add_coins(3);       // +3 Coins
```

### Drawing Cards
```cpp
Player& player = state.get_player(pid);
player.draw_cards(3);     // Draw 3 cards (reshuffles discard if needed)
```

### Gaining Cards
```cpp
state.gain_card(pid, "Silver");              // Gain to discard pile
state.gain_card_to_hand(pid, "Gold");        // Gain to hand
state.gain_card_to_deck_top(pid, "Silver");  // Gain to top of deck
```

### Trashing Cards
```cpp
int card_id = player.trash_from_hand_return(hand_index);  // Remove from hand, get card_id
state.trash_card(card_id);                                 // Add to trash pile
```

### Discarding
```cpp
player.discard_from_hand(hand_index);  // Move from hand to discard
player.discard_hand();                 // Discard entire hand
```

### Deck Manipulation
```cpp
player.peek_deck(3);              // Look at top 3 (non-destructive)
player.remove_deck_top();         // Remove and return top card (-1 if empty)
player.add_to_deck_top(card_id);  // Put card on top of deck
player.topdeck_from_hand(idx);    // Move hand card to deck top
```

### Other Movement
```cpp
player.add_to_hand(card_id);                       // Add card to hand
player.add_to_discard(card_id);                    // Add card to discard
player.remove_from_discard(card_id);               // Remove by card_id
player.remove_from_discard_by_index(idx);          // Remove by index
player.find_in_hand(card_id);                      // Get hand index of card_id
player.play_from_hand(hand_index);                 // Move to in_play zone
```

### Card Info Lookups
```cpp
state.card_name(card_id);            // "Smithy"
state.card_def(card_id);             // const Card* pointer
CardRegistry::get("Smithy");         // const Card* by name
state.total_cards_owned(pid);        // Total cards across all zones
```

### Supply Queries
```cpp
state.gainable_piles(max_cost);                     // All piles with cost <= max_cost
state.gainable_piles(max_cost, CardType::Treasure);  // Filtered by type
state.gainable_piles_exact(cost);                    // Exact cost match
state.get_supply().top_card("Smithy");               // card_id of top card in pile
state.get_supply().empty_pile_count();                // Number of empty piles
```

### Turn Flags (for deferred/tracked effects)
```cpp
state.set_turn_flag("merchant_count", 2);   // Set a per-turn counter
state.get_turn_flag("merchant_count");       // Read it (0 if unset)
```

### Logging
```cpp
state.log("    Vassal reveals " + state.card_name(card_id));
```

---

## Implementation Patterns by Complexity

### Pattern 1: Simple Stat Boost (no decisions)

Cards like **Smithy**, **Village**, **Festival**, **Market**.

```cpp
// Village: +1 Card, +2 Actions
.on_play = [](GameState& state, int pid, DecisionFn) {
    state.get_player(pid).draw_cards(1);
    state.add_actions(2);
},
```

Key points:
- Use `DecisionFn` (no parameter name) when the callback is unused.
- Use `int /*pid*/` when pid is unused (e.g., Festival only modifies state-level resources).

### Pattern 2: Discard-from-Hand Choice

Cards like **Cellar**, **Poacher**, **Storeroom**, **Hamlet**.

```cpp
// Cellar: +1 Action. Discard any number, draw that many.
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    state.add_actions(1);
    Player& player = state.get_player(pid);
    if (player.hand_size() == 0) return;

    // Build options = [0, 1, 2, ... hand_size-1]
    std::vector<int> options;
    for (int i = 0; i < player.hand_size(); i++) options.push_back(i);

    auto chosen = decide(pid, ChoiceType::DISCARD, options, 0, player.hand_size());

    // CRITICAL: Sort indices in descending order before removing
    std::sort(chosen.begin(), chosen.end(), std::greater<int>());
    int count = static_cast<int>(chosen.size());
    for (int idx : chosen) player.discard_from_hand(idx);
    player.draw_cards(count);
},
```

**Critical rule**: When removing multiple cards from hand by index, always sort indices in **descending order** first. Removing index 3 before index 5 would shift index 5, causing wrong removals.

### Pattern 3: Trash-from-Hand

Cards like **Chapel**, **Remodel** (first half), **Trading Post**.

```cpp
// Chapel: Trash up to 4 cards from your hand.
auto chosen = decide(pid, ChoiceType::TRASH, options, 0, max_trash);
std::sort(chosen.begin(), chosen.end(), std::greater<int>());
for (int idx : chosen) {
    int cid = player.trash_from_hand_return(idx);
    state.trash_card(cid);
}
```

The two-step trash: `trash_from_hand_return()` removes from hand and returns the card_id, then `trash_card()` adds it to the global trash pile.

### Pattern 4: Gain from Supply

Cards like **Workshop**, **Artisan** (first half), **Armory**.

```cpp
// Workshop: Gain a card costing up to 4.
auto piles = state.gainable_piles(4);
if (piles.empty()) return;

std::vector<int> options;
for (const auto& p : piles) options.push_back(state.get_supply().top_card(p));

auto chosen = decide(pid, ChoiceType::GAIN, options, 1, 1);
if (!chosen.empty()) {
    state.gain_card(pid, piles[chosen[0]]);
}
```

The pattern: get gainable pile names -> build options from top cards -> player chooses -> index back into pile names to gain.

### Pattern 5: Trash-then-Gain (Remodel family)

Cards like **Remodel**, **Mine**, **Expand**, **Upgrade**, **Altar**.

```cpp
// Remodel: Trash a card. Gain one costing up to 2 more.
// Step 1: Choose and trash
auto trash_chosen = decide(pid, ChoiceType::TRASH, trash_options, 1, 1);
int hand_idx = trash_chosen[0];
int trashed_id = player.trash_from_hand_return(hand_idx);
const Card* trashed = state.card_def(trashed_id);
state.trash_card(trashed_id);

// Step 2: Gain based on trashed card's cost
int max_cost = trashed->cost + 2;
auto piles = state.gainable_piles(max_cost);
// ... standard gain pattern ...
```

Variants:
- **Mine**: Filters hand for Treasures only, gains Treasure to hand: `gainable_piles(max_cost, CardType::Treasure)` + `gain_card_to_hand()`
- **Upgrade**: Exact cost: `gainable_piles_exact(trashed->cost + 1)`
- **Altar**: Fixed gain cap: `gainable_piles(5)` regardless of trashed cost

### Pattern 6: Yes/No Optional Effect

Cards like **Moneylender**, **Vassal** (play action from discard).

```cpp
// Moneylender: May trash a Copper for +3 Coins.
auto chosen = decide(pid, ChoiceType::YES_NO, {0, 1}, 1, 1);
if (!chosen.empty() && chosen[0] == 1) {
    // Player said yes — do the thing
    int cid = player.trash_from_hand_return(copper_idx);
    state.trash_card(cid);
    state.add_coins(3);
}
```

### Pattern 7: Peek-and-Process (Sentry/Lookout family)

Cards that look at top N cards and process each one.

```cpp
// General pattern: peek, remove from deck, process, put back
auto top_cards = player.peek_deck(N);
int num_peeked = static_cast<int>(top_cards.size());
for (int i = 0; i < num_peeked; i++) player.remove_deck_top();

// Process each card (trash/discard/keep)...

// Put back remaining cards (if >1, ask for order)
if (keep_cards.size() > 1) {
    // Use ChoiceType::ORDER to let player pick ordering
    // Add in reverse order (last picked = deepest, first picked = top)
}
```

**Lookout** variant (trash one, discard one, keep one):
```cpp
// Trash one
auto trash_chosen = decide(pid, ChoiceType::TRASH, top_cards, 1, 1);
int trashed_id = top_cards[trash_chosen[0]];
state.trash_card(trashed_id);
top_cards.erase(std::find(top_cards.begin(), top_cards.end(), trashed_id));

// Discard one (if 2+ remain)
auto disc_chosen = decide(pid, ChoiceType::DISCARD, top_cards, 1, 1);
int discarded_id = top_cards[disc_chosen[0]];
player.add_to_discard(discarded_id);
top_cards.erase(std::find(top_cards.begin(), top_cards.end(), discarded_id));

// Put remaining card back
player.add_to_deck_top(top_cards[0]);
```

### Pattern 8: Attack Cards

Cards that affect other players. Use `state.resolve_attack()` which handles Reaction checks (e.g., Moat) automatically.

```cpp
// Witch: +2 Cards. Each other player gains a Curse.
.types = CardType::Action | CardType::Attack,
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    state.get_player(pid).draw_cards(2);

    state.resolve_attack(pid, [](GameState& st, int target, DecisionFn) {
        st.gain_card(target, "Curse");
    }, decide);
},
```

The attack lambda receives each non-blocked target. The lambda signature is always:
```cpp
[](GameState& st, int target, DecisionFn dec) { ... }
```

**Important**: When the *attacker* makes choices for the target (e.g., Swindler), use `dec(pid, ...)` not `dec(target, ...)`. Capture `pid` in the lambda:

```cpp
state.resolve_attack(pid, [pid](GameState& st, int target, DecisionFn dec) {
    // Attacker chooses what the target gains
    auto chosen = dec(pid, ChoiceType::GAIN, gain_opts, 1, 1);
    st.gain_card(target, piles[chosen[0]]);
}, decide);
```

### Pattern 9: Reaction Cards

Cards with `on_react` that can block attacks.

```cpp
.types = CardType::Action | CardType::Reaction,
.on_play = [](GameState& state, int pid, DecisionFn) {
    state.get_player(pid).draw_cards(2);
},
.on_react = [](GameState& /*state*/, int pid, int /*attacker_id*/, DecisionFn decide) -> bool {
    auto chosen = decide(pid, ChoiceType::YES_NO, {0, 1}, 1, 1);
    return !chosen.empty() && chosen[0] == 1;  // true = block the attack
},
```

### Pattern 10: Dynamic Victory Points

Cards like **Gardens** where VP depends on game state.

```cpp
.types = CardType::Victory,
.on_play = nullptr,
.on_react = nullptr,
.vp_fn = [](const GameState& state, int pid) -> int {
    return state.total_cards_owned(pid) / 10;
},
```

When `vp_fn` is set, it overrides `victory_points` during scoring.

### Pattern 11: Card Replay (Throne Room family)

Cards that play another card multiple times.

```cpp
// Throne Room: Play an Action from hand twice
.on_play = [](GameState& state, int pid, DecisionFn decide) {
    Player& player = state.get_player(pid);

    // Find Actions in hand
    std::vector<int> action_indices;
    for (int i = 0; i < player.hand_size(); i++) {
        const Card* card = state.card_def(player.get_hand()[i]);
        if (card && card->is_action()) action_indices.push_back(i);
    }
    if (action_indices.empty()) return;

    auto chosen = decide(pid, ChoiceType::PLAY_CARD, action_indices, 0, 1);
    if (chosen.empty()) return;

    int hand_idx = action_indices[chosen[0]];
    int card_id = player.get_hand()[hand_idx];
    player.play_from_hand(hand_idx);

    // Play effect twice (card is already in play zone, not in hand)
    state.play_card_effect(card_id, pid, decide);
    state.play_card_effect(card_id, pid, decide);
},
```

### Pattern 12: Select from Discard

Cards like **Harbinger**, **Mountain Village**, **Scavenger**.

```cpp
// Harbinger: Put a card from discard on top of deck
const auto& discard = player.get_discard();
if (discard.empty()) return;

std::vector<int> options;
for (int i = 0; i < static_cast<int>(discard.size()); i++) options.push_back(i);

auto chosen = decide(pid, ChoiceType::SELECT_FROM_DISCARD, options, 0, 1);
if (!chosen.empty()) {
    int discard_idx = chosen[0];
    int card_id = discard[discard_idx];
    player.remove_from_discard_by_index(discard_idx);
    player.add_to_deck_top(card_id);
}
```

### Pattern 13: Turn Flags (Deferred/Tracked Bonuses)

Cards like **Merchant** that set up a conditional bonus for later in the turn.

```cpp
// Merchant: First Silver played this turn gives +1 Coin
.on_play = [](GameState& state, int pid, DecisionFn) {
    state.get_player(pid).draw_cards(1);
    state.add_actions(1);
    state.set_turn_flag("merchant_count",
        state.get_turn_flag("merchant_count") + 1);
},
```

The actual bonus application happens in the game engine's treasure-playing logic, not in the card itself.

### Pattern 14: Reveal-and-Check Hand

Cards like **Shanty Town**, **Magnate** that check hand composition.

```cpp
// Shanty Town: +2 Actions. If no Actions in hand, +2 Cards.
.on_play = [](GameState& state, int pid, DecisionFn) {
    state.add_actions(2);
    Player& player = state.get_player(pid);
    bool has_action = false;
    for (int i = 0; i < player.hand_size(); i++) {
        const Card* c = state.card_def(player.get_hand()[i]);
        if (c && c->is_action()) { has_action = true; break; }
    }
    if (!has_action) player.draw_cards(2);
},
```

### Pattern 15: Reveal-from-Deck Loop

Cards like **Sage**, **Hunting Party** that reveal cards one at a time until a condition is met.

```cpp
// Sage: Reveal until you find a card costing >= 3, put it in hand, discard the rest.
.on_play = [](GameState& state, int pid, DecisionFn) {
    state.add_actions(1);
    Player& player = state.get_player(pid);
    std::vector<int> revealed;
    while (true) {
        int card_id = player.remove_deck_top();
        if (card_id == -1) break;
        const Card* c = state.card_def(card_id);
        if (c && c->cost >= 3) {
            player.add_to_hand(card_id);
            break;
        }
        revealed.push_back(card_id);
    }
    for (int cid : revealed) player.add_to_discard(cid);
},
```

### Pattern 16: Self-Trashing from Play

Cards like **Treasure Map**, **Tragic Hero** that trash themselves as part of their effect.

```cpp
// Treasure Map: Trash this and a Treasure Map from hand. If both trashed, gain 4 Golds to deck.
.on_play = [](GameState& state, int pid, DecisionFn) {
    Player& player = state.get_player(pid);

    // Find this card in in_play and remove it
    // Find a copy in hand and trash it
    // If both trashed successfully, gain 4 Golds to deck top
    for (int i = 0; i < 4; i++) state.gain_card_to_deck_top(pid, "Gold");
},
```

---

## Filtering Hand by Type

A recurring need is building options from only certain card types in hand:

```cpp
std::vector<int> treasure_indices;
for (int i = 0; i < player.hand_size(); i++) {
    const Card* c = state.card_def(player.get_hand()[i]);
    if (c && c->is_treasure()) treasure_indices.push_back(i);
}
```

This same pattern works with `is_action()`, `is_victory()`, etc.

---

## Non-Attack Multi-Player Effects

For positive effects on other players (like Council Room's "+1 Card to each other player"), just loop manually — don't use `resolve_attack()`:

```cpp
for (int i = 0; i < state.num_players(); i++) {
    if (i != pid) {
        state.get_player(i).draw_cards(1);
    }
}
```

---

## Checklist for New Cards

1. Pick the right level file (or create a new one)
2. Add `CardRegistry::register_card({...})` inside `register_all()`
3. Set all required fields: `name`, `cost`, `types`, `text`, `victory_points`, `coin_value`, `tags`
4. Implement `on_play` lambda (and optionally `on_react`, `vp_fn`, `on_gain`, `on_trash`, `on_duration`)
5. If new level file: create header, wire up `register_all()` in all entry points, add to build system
6. Always sort hand indices in descending order before multi-removal
7. Always check for empty hand / empty piles before presenting choices
8. Use `resolve_attack()` for Attack cards (handles Moat/Reaction checks)
9. Use `CardType::Action | CardType::Attack` for Attack types (never just `CardType::Attack`)
10. Test with the stress test runner to catch crashes
