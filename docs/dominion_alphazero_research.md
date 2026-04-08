# Superhuman Dominion AI: Research & Engineering Considerations

## Overview

Building an AlphaZero-class AI for Dominion (with all expansions) is one of the most
interesting unsolved problems at the intersection of game AI and combinatorial complexity.
This document summarizes the key challenges, architectural considerations, and a realistic
path forward.

---

## Why AlphaZero Doesn't Directly Apply

AlphaZero works by combining self-play with Monte Carlo Tree Search (MCTS) guided by a
neural network that learns value (who's winning) and policy (what to do). It achieves
superhuman performance in **perfect information, two-player, zero-sum games** with fixed
state and action spaces.

Dominion violates nearly all of those assumptions.

---

## Core Challenges

### 1. Combinatorial Kingdom Space
With ~300+ Kingdom cards across all expansions and 10 chosen per game, the number of
possible Kingdoms is on the order of C(300, 10) — in the **trillions**. A single model
must generalize across all of them, unlike AlphaZero which trained on one fixed game.
This makes Dominion not one game but effectively millions of related games.

### 2. Hidden Information & Stochasticity
- Players cannot see opponents' hands
- Deck order is unknown (shuffle randomness)
- Many cards involve random reveals (Militia, Spy, etc.)

Standard MCTS assumes full state observability. Handling hidden information requires
fundamentally different algorithms:
- **Perfect Information Monte Carlo (PIMC)**: sample possible worlds and plan within them
- **Counterfactual Regret Minimization (CFR)**: the technique behind superhuman poker AI
- **Neural Fictitious Self-Play (NFSP)**: a hybrid approach combining RL and supervised learning

### 3. Enormous & Variable Action Spaces
Legal actions on any given turn can be massive and deeply contextual. Cards like
Throne Room, Overlord, Band of Misfits, and Inheritance dynamically change what other
cards do, creating a shifting action space that is difficult to represent as a fixed
neural network output layer.

### 4. Variable-Length, Multi-Phase Turns
A single turn in Dominion — especially with engine builds involving Villages and draw
cards — can involve hundreds of sub-decisions. Unlike Chess (one move per turn), you
need to handle **arbitrary decision trees within a single turn**, potentially requiring
a sequential decision model on top of the main policy network.

### 5. Multi-Player Game Theory
AlphaZero is designed for two-player zero-sum games. Dominion is commonly played with
2–4 players, where:
- Nash equilibria aren't guaranteed to be unique or stable
- Kingmaking dynamics emerge (a losing player can determine who wins)
- Optimal strategy shifts significantly based on player count

Multi-player game theory is dramatically harder than two-player, and much less mature
in the AI literature.

### 6. State Representation Complexity
Encoding the game state into a neural network input requires capturing:
- Your deck composition (known and unknown portions)
- Your hand and discard pile
- Opponents' deck sizes, discard tops, and known cards
- All Supply piles and their counts
- Expansion-specific tokens and mats: Coffers, Villagers, Exile, Journey tokens
- Duration cards in play, Tavern mat cards, Prince targets, Inherited cards
- The current phase of the turn and available actions

This is **not a fixed-size tensor** — it varies wildly across expansions, making a
clean, generalizable representation a significant research problem in itself.

---

## State Representation: The Transformer Approach

Transformers are likely the best architectural fit for Dominion's state representation.

### Cards as Tokens
Each card is treated as a **token with a learned embedding**, exactly like words in a
sentence. A hand of 5 cards becomes a sequence of 5 token embeddings. This naturally
handles variable-length inputs without padding hacks.

### Self-Attention Captures Card Synergies
Self-attention allows every card to "look at" every other card and learn how much to
weight it. The model learns organically that:
- Village attends strongly to draw cards
- Militia attends strongly to opponent hand size
- Throne Room attends strongly to the best action card in hand

These synergies are the heart of Dominion strategy — and transformers learn relational
structure like this without any hardcoding.

### Cross-Attention Between Game Zones
Separate encoders can be used for different game zones:
- Hand encoder
- Deck/discard encoder (probabilistic, since order is unknown)
- Supply encoder
- Opponent state encoder

Cross-attention then allows them to interact: your hand attends to the Supply to
evaluate what's buyable; your deck encoder attends to your hand to understand what
engines are forming.

### Kingdom-Conditional Generalization
With the Kingdom cards always present in the input as tokens, the policy and value heads
are implicitly conditioned on which game is being played. A model trained on thousands
of different Kingdoms could potentially **zero-shot generalize** to unseen Kingdoms —
similar to how a language model generalizes to new sentences.

### Rough Architecture

```
[Hand tokens]    ──┐
[Supply tokens]  ──┤
[Deck tokens]    ──┼── Transformer Encoder ── Pooled State Vector ──┬── Value Head
[Discard tokens] ──┤                                                  └── Policy Head
[Opponent info]  ──┘
```

Each token is a card embedding + zone encoding (hand, supply, discard, etc.), and
transformer layers let them interact via attention before pooling into a single vector
for the output heads.

### Remaining Architectural Challenges
- **Deck uncertainty**: Deck composition is known but order is not. The deck encoder
  should use a **multiset embedding** rather than a sequence to reflect this.
- **Within-turn sequential decisions**: A second transformer or recurrent layer may be
  needed to handle the chain of sub-decisions within a single turn.
- **Action masking**: The policy head output must be masked to only legal actions, which
  vary dynamically based on card interactions.

---

## Engineering Considerations

### Game Simulator
A fast, bit-accurate game engine is the foundation of everything. It needs to:
- Correctly implement every card interaction across all expansions (300+ cards, many
  with complex chaining behavior)
- Run at very high speed (millions of games per hour) to support self-play
- Support programmatic action selection for MCTS rollouts
- Be rigorously tested — a single wrong card interaction corrupts all training data

Estimated effort: **3–12 months** for a full-expansion accurate simulator.

### Training Infrastructure
- Requires significant GPU compute — likely millions of GPU-hours for a superhuman model
- Self-play loop must efficiently parallelize game simulation and model inference
- Replay buffer management, ELO tracking, and periodic evaluation against baselines
  are all necessary infrastructure

### Reward Signal Design
Dominion's win condition (most Victory Points) is clear, but shaping intermediate
rewards is tricky. Potential considerations:
- Pure win/loss signal is sparse — games are long and delayed rewards slow learning
- Intermediate rewards (Province buys, deck efficiency) may speed early learning but
  risk teaching locally optimal but globally suboptimal strategies

### Evaluation Framework
Without a known superhuman benchmark, you need:
- A strong heuristic baseline (e.g., Big Money, greedy VP strategies)
- A rating system (ELO or TrueSkill) across model versions
- Diversity of test Kingdoms to detect overfitting to specific card combinations
- Human expert evaluation for qualitative assessment

---

## Recommended Phased Approach

**Phase 1 — Proof of Concept (Base game, 2 players)**
- Implement a clean, fast simulator for the Base set only
- Adapt PIMC-MCTS for hidden information
- Train a basic neural network (MLP or small transformer) on self-play
- Validate that learning occurs and beats heuristic baselines

**Phase 2 — Architecture Research**
- Design the full transformer-based state representation
- Experiment with multiset embeddings for deck uncertainty
- Benchmark different hidden-info algorithms (PIMC vs. CFR vs. NFSP)
- Expand to Intrigue and one or two other small expansions

**Phase 3 — Scale & Generalize**
- Add all expansions incrementally
- Introduce meta-learning or curriculum learning across Kingdoms
- Scale compute and evaluate zero-shot generalization to unseen Kingdoms
- Extend to 3–4 player games

---

## Rough Effort Estimate

| Component | Estimated Effort |
|---|---|
| Full-expansion game simulator | 3–12 months |
| State representation design & research | 6–18 months |
| Hidden info algorithm adaptation | 6–12 months |
| Neural network architecture | 3–6 months |
| Training infrastructure & compute | Ongoing |
| Evaluation, iteration, tuning | 12–24 months |
| **Total (serious team)** | **~3–6 years** |

Claude Code and modern tooling can compress the pure engineering work significantly,
but research bottlenecks (algorithm selection, representation design, training dynamics)
and compute requirements remain largely unchanged.

---

## Related Work & Inspirations

- **AlphaZero** (DeepMind) — MCTS + neural network self-play for perfect information games
- **AlphaStar** (DeepMind) — transformer-based state encoding for StarCraft II
- **Libratus / Pluribus** (CMU) — CFR-based superhuman poker AI; most relevant for hidden info
- **Neural Fictitious Self-Play** — handles multi-player, imperfect info via approximate best response
- **Geronimoo's Dominion Simulator** — existing heuristic-based Dominion AI, useful as a baseline
- **Set Transformer** (Lee et al.) — attention-based architecture for set-structured inputs,
  directly applicable to multiset deck embeddings

---

## Conclusion

A superhuman Dominion AI covering all expansions would be a **genuine research
contribution** — not just an engineering project. The combination of hidden information,
combinatorial Kingdom diversity, multi-player dynamics, and variable action spaces puts
it beyond the reach of existing game AI frameworks without meaningful algorithmic
innovation. The transformer architecture offers the most promising path to a
generalizable state representation, but it must be paired with appropriate hidden-info
planning algorithms and massive self-play compute to reach superhuman levels.
