# Tensor Encoding Options for AlphaZero Training

Research reference for observation and action tensor encoding approaches.
Each approach is designed to be implemented as a C++ encoder that writes directly
into a numpy buffer (zero-copy via pybind11).

---

## Game Dimensions

| Property | Current | Future (all expansions) |
|---|---|---|
| Unique card types | 33 | 600+ |
| Supply piles per game | 17 | ~20-25 |
| Max hand size | ~20 (mid-turn) | same |
| Hand size after cleanup | 5 | same |
| Max players | 4 | same (typically 2 for training) |
| Turn state vars | actions, buys, coins, phase, turn_number | same |
| Card properties | cost, 9 type flags, VP, coin_value | same + more complex effects |
| Decision types | 9 | more with expansions |
| Max options per decision | ~18 | ~30 with expansions |

---

## Observation Encoding

### Approach 1: Card-Count Vector

Represent each zone as a histogram over card types.

**Layout (2-player, ~230 floats):**
```
my_hand_counts[33]        # count of each card type in my hand
my_deck_counts[33]        # count of each card type in my deck (known to me)
my_discard_counts[33]     # count of each card type in my discard
my_in_play_counts[33]     # count of each card type I've played this turn
supply_counts[17]         # cards remaining in each supply pile
trash_counts[33]          # count of each card type in trash
opp_hand_size[1]          # opponent hand size (hidden composition)
opp_deck_size[1]          # opponent deck size
opp_discard_size[1]       # opponent discard size
actions[1]                # remaining actions
buys[1]                   # remaining buys
coins[1]                  # current coins
phase[6]                  # one-hot phase encoding
turn_number[1]            # normalized 0-1 (divide by 80)
empty_pile_count[1]       # for game-end awareness
```

**Implementation:**
```
encode_observation(const GameState& state, int player_id, float* buffer) {
    // Zero the buffer
    // For each card in player's hand, increment buffer[card_def_id]
    // For each card in player's deck, increment buffer[33 + card_def_id]
    // ... etc for each zone
    // Write scalar features at the end
}
```

**Pros:**
- Fixed-size regardless of hand/deck size
- Very fast to encode (~1us)
- Simple to implement and debug

**Cons:**
- Loses ordering information (irrelevant for most decisions)
- Fixed dimension tied to card count -- adding cards changes tensor shape
- Each card type is just an index, no semantic information encoded

**Scaling to 600+ cards:** Tensor grows linearly (600 per zone x 6 zones = 3600 floats).
Workable but sparse -- most entries will be zero since only ~17 card types appear per game.

---

### Approach 2: Per-Card Feature Matrix

Each card type gets a row of features combined with zone counts.

**Layout (33 x 15 = 495 floats):**
```
For each card type i (0..32):
  row[i] = [
    cost,                    # normalized (0-8) / 8
    is_action,               # 0 or 1
    is_treasure,             # 0 or 1
    is_victory,              # 0 or 1
    is_curse,                # 0 or 1
    is_attack,               # 0 or 1
    is_reaction,             # 0 or 1
    coin_value / 3.0,        # normalized
    victory_points / 6.0,    # normalized
    count_in_hand,           # 0-10
    count_in_deck,           # 0-30
    count_in_discard,        # 0-30
    count_in_supply,         # 0-60
    count_in_trash,          # 0-30
    in_kingdom,              # 1 if this card is in the kingdom, 0 otherwise
  ]

# Plus global features appended (same as Approach 1): ~12 floats
```

**Implementation:**
```
encode_observation(const GameState& state, int player_id, float* buffer) {
    // For each registered card def_id:
    //   Write static properties (cost, types) -- can be cached once at game start
    //   Count occurrences in each zone by iterating player zones
    // Append global scalars
}
```

**Pros:**
- Card properties are explicit -- network can learn "expensive actions are worth buying"
- 2D structure works with CNNs (treat as 33-channel 1D signal) or transformers
- Network sees card semantics, not just arbitrary indices

**Cons:**
- Static properties are redundant across timesteps (always the same for a given card)
- Still tied to a fixed card roster

**Scaling to 600+ cards:** 600 x 15 = 9000 floats. Still feasible but wasteful since
only ~33 card types appear in any single game. Better approach: only include rows for
cards actually in the game's kingdom (max ~20-25 rows), with a learned embedding
replacing the static features.

---

### Approach 3: Flat Feature Vector

Concatenate all features into a single 1D vector with a fixed semantic layout.

**Layout (~200 floats):**
```
# === My card counts by zone (4 zones x 33 types = 132) ===
my_hand[33]
my_deck[33]
my_discard[33]
my_in_play[33]

# === Supply state (17 piles x 2 features = 34) ===
supply_count[17]          # cards remaining (normalized by starting count)
supply_empty[17]          # binary: is pile empty?

# === Opponent observable state (per opponent, here 1 opp = 5) ===
opp_hand_size[1]
opp_deck_size[1]
opp_discard_size[1]
opp_in_play_counts[33]   # opponent's played cards are visible

# === Trash (33) ===
trash_counts[33]

# === Turn state (11) ===
actions[1]
buys[1]
coins[1]
phase_onehot[6]
turn_number[1]            # normalized
empty_piles[1]

# Total: ~250 floats
```

**Implementation:**
```
encode_observation(const GameState& state, int player_id, float* buffer) {
    int offset = 0;
    auto write_zone_counts = [&](const vector<int>& cards) {
        for (int cid : cards)
            buffer[offset + state.card_def_id(cid)] += 1.0f;
        offset += NUM_CARD_TYPES;
    };
    write_zone_counts(player.get_hand());
    write_zone_counts(player.get_deck());
    // ... etc
}
```

**Pros:**
- Dead simple, one memset + one pass over each zone
- Fastest to encode
- Direct MLP input, no reshaping needed
- Proven effective for card games (Hanabi, Poker AI)

**Cons:**
- No spatial structure (but MLPs don't need it)
- Monolithic -- harder to extend than matrix approach
- Fixed to card roster size

**Scaling to 600+ cards:** Same issue as Approach 1 -- grows linearly, mostly sparse.
Practical mitigation: use a "game-local card index" (0..K where K = number of card
types in this game, typically ~20-25) instead of global def_id. Map between local
and global at game setup.

---

### Approach 4: Set/Sequence Encoding (Transformer-Ready)

Represent each card as a token with features. Variable-length sequences for each zone.

**Layout:**
```
hand_tokens[max_hand, D]     # each card in hand is a D-dim token
deck_summary[D]              # aggregated deck representation (count-weighted)
supply_tokens[17, D]         # each supply pile is a token
global_features[G]           # scalar game state

Where each token has:
  D = embedding_dim + static_features
    = [learned_embedding(def_id), cost, types(9), coin_value, vp, count_in_zone]
```

**Implementation:**
```
// Pre-compute: embedding table of shape (num_card_types, embedding_dim)
// At encode time:
//   For each card in hand: write embedding + features as a row
//   Pad to max_hand with zeros + a padding mask
//   For deck/discard: aggregate embeddings weighted by count
//   For supply: one token per pile with count feature
```

**Pros:**
- Most expressive -- network sees individual cards, not just counts
- Variable-length input handled naturally by attention
- Learned embeddings scale to 600+ cards without changing architecture
- Same embedding used across zones -- "Smithy in hand" and "Smithy in supply" share representation
- New cards only need a new embedding row, no architecture change

**Cons:**
- Requires a transformer/set-transformer architecture (more complex, slower inference)
- Learned embeddings need training data for each card (cold start for rare cards)
- Padding/masking adds implementation complexity
- Slower encoding than flat vectors

**Scaling to 600+ cards:** This is the natural choice for large card pools.
The embedding table grows (600 x D), but the per-game tensor stays small because
only ~20-25 card types appear. New cards get new embeddings via fine-tuning or
meta-learning.

---

## Action Space Encoding

### Option A: Fixed-Size Masked Output

Single policy head of fixed size. Mask invalid actions with -inf before softmax.

**Layout:**
```
policy_logits[MAX_OPTIONS]    # e.g., MAX_OPTIONS = 50
mask[MAX_OPTIONS]             # 1 for valid, 0 for invalid

# At inference:
logits[~mask] = -inf
probs = softmax(logits)
```

**Implementation:**
```
encode_action_mask(const DecisionPoint& dp, float* mask) {
    memset(mask, 0, MAX_OPTIONS * sizeof(float));
    for (int i = 0; i < dp.options.size(); i++)
        mask[i] = 1.0f;
}

// After network forward pass:
decode_action(int chosen_index, const DecisionPoint& dp) -> vector<int> {
    return {chosen_index};  // or multiple for multi-select
}
```

**Pros:**
- Simplest to implement
- Standard in RL (MuZero, OpenSpiel)
- Works with any network architecture

**Cons:**
- Output neuron 3 means "Smithy" in one decision and "Silver" in another -- 
  the network must learn context-dependent meanings
- Max size must accommodate worst case
- Multi-select decisions (discard 3 of 7) need special handling:
  either auto-regressive (pick one at a time) or combinatorial output

**Multi-select handling options:**
1. **Auto-regressive**: For "discard 3 cards", make 3 sequential decisions
   (remove chosen card from options each time). Simple, works with this approach.
2. **Independent binary**: Output a probability per option, sample independently,
   reject if count violates min/max. Fast but rejection sampling is wasteful.
3. **Combinatorial**: Enumerate all valid subsets. Intractable for large hands.

**Recommendation:** Auto-regressive. The SteppableGame already supports this --
break multi-select into sequential single-select calls in a wrapper layer.

---

### Option B: Per-Decision-Type Heads

Separate policy heads for each decision type. Route based on DecisionType.

**Layout:**
```
action_head[MAX_HAND_ACTIONS + 1]     # play action or pass
treasure_head[MAX_HAND_TREASURES + 2] # play treasure, play all, or pass
buy_head[MAX_SUPPLY_PILES + 1]        # buy from pile or pass
discard_head[MAX_HAND + 1]            # discard card or done
trash_head[MAX_HAND + 1]              # trash card or done
yes_no_head[2]                        # binary
gain_head[MAX_SUPPLY_PILES]           # gain from pile
multi_fate_head[3]                    # keep / discard / trash
```

**Implementation:**
```
// Network has shared backbone, then branches:
switch (dp.type) {
    case PLAY_ACTION:  logits = action_head(features); break;
    case BUY_CARD:     logits = buy_head(features); break;
    case CHOOSE_CARDS_IN_HAND:
        switch (dp.sub_choice_type) {
            case DISCARD: logits = discard_head(features); break;
            case TRASH:   logits = trash_head(features); break;
            // ...
        }
}
```

**Pros:**
- Each head has clear semantics -- output 3 always means the same thing within a head
- Easier for the network to learn (less context-dependence)
- Can have different head sizes (buy head = 18, yes/no head = 2)

**Cons:**
- Many heads to maintain (9 decision types x 11 sub-choice types = up to 20 heads)
- Adding new decision types requires new heads
- Shared learning between similar decisions is lost (e.g., "gain from supply" appears
  in BUY, GAIN sub-choice, and Workshop effect -- three different heads)

---

### Option C: Card-Centric Policy with Option Embeddings

Score each option by encoding it as a feature vector, then dot-product with a
learned query vector from the game state.

**Layout:**
```
# Encode each available option as a feature vector:
option_features[num_options, F]
  F = [card_def_embedding(D), decision_type_onehot(9), sub_choice_onehot(11),
       is_pass(1), is_play_all(1), pile_index_if_buy(1)]

# Network produces a "query" vector from the observation:
query = backbone(observation)  # shape (F,)

# Score each option:
scores[i] = dot(query, option_features[i])
probs = softmax(scores[:num_options])
```

**Implementation:**
```
encode_options(const DecisionPoint& dp, const GameState& state, float* buffer) {
    for (int i = 0; i < dp.options.size(); i++) {
        int offset = i * OPTION_FEATURE_DIM;
        auto& opt = dp.options[i];
        
        // Write card embedding (looked up by def_id)
        if (opt.def_id >= 0)
            copy_embedding(opt.def_id, &buffer[offset]);
        
        // Write decision context
        buffer[offset + D + dp.type] = 1.0f;  // decision type one-hot
        buffer[offset + D + 9 + dp.sub_choice_type] = 1.0f;
        buffer[offset + D + 20] = opt.is_pass ? 1.0f : 0.0f;
        buffer[offset + D + 21] = opt.is_play_all ? 1.0f : 0.0f;
    }
}
```

**Pros:**
- "Smithy" has the same representation whether you're playing it, buying it, or trashing it
- Naturally handles variable option counts (no fixed max, no padding)
- Scales to 600+ cards -- new cards just need embeddings, not new output neurons
- Shared card embeddings between observation and action encoding
- Most aligned with the future card-embedding direction

**Cons:**
- More complex architecture (dot-product attention over options)
- Slower per-decision (encode all options, compute dot products)
- Card embeddings must be learned -- cold start problem for new cards
- Multi-select still needs auto-regressive or independent sampling

---

## Comparison Matrix

| | Count Vector (Obs 1) | Feature Matrix (Obs 2) | Flat Vector (Obs 3) | Set Encoding (Obs 4) |
|---|---|---|---|---|
| Tensor size | ~230 | ~500 | ~250 | variable |
| Encode speed | very fast | fast | very fast | moderate |
| Card semantics | none | explicit | none | learned |
| Scales to 600+ | sparse | sparse | sparse | natural |
| Architecture | MLP | MLP/CNN | MLP | Transformer |

| | Masked Output (Act A) | Per-Type Heads (Act B) | Card-Centric (Act C) |
|---|---|---|---|
| Complexity | low | medium | high |
| Semantic clarity | low | high | high |
| Scales to 600+ | poorly | moderately | naturally |
| Multi-select | auto-regressive | auto-regressive | auto-regressive |

---

## Recommended Research Path

1. **Start simple**: Obs 3 (flat vector) + Act A (masked output). Get training working.
2. **Add card features**: Obs 2 (feature matrix) + Act A. See if explicit card properties help.
3. **Move to embeddings**: Obs 4 (set encoding) + Act C (card-centric). This is the
   architecture that will scale to 600+ cards and is most aligned with the AlphaZero
   approach of learning representations end-to-end.

Each step builds on the previous one -- the C++ encoder can support all three by
having separate `encode_*` functions, selectable at runtime.
