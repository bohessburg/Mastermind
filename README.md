# DominionZero

An AlphaZero-style AI for the card game [Dominion](https://wiki.dominionstrategy.com/).

## Goal

Train a self-play reinforcement learning agent that learns to play Dominion at a high level, using techniques inspired by [AlphaZero](https://arxiv.org/abs/1712.01815). This requires a fast, correct, and modular game engine as the foundation.

## Why Dominion is hard

AlphaZero was designed for perfect-information, fixed-action-space games like Chess and Go. Dominion breaks those assumptions in almost every way:

- **Massive branching factor.** A single turn can involve chaining multiple actions, each with their own sub-decisions (discard 2 of 5 cards, gain a card costing up to 4, etc.). The decision tree is drastically wider and more variable than chess or Go.
- **Hidden information.** Players don't know the contents of their own deck, opponents' hands, or the order of upcoming draws. The model must reason under uncertainty. Randomness is not necessarily a problem for Monte-carlo simulations because they are random by nature, but often your strategy in a given turn greatly depends on the hand you draw.
- **Combinatorial card interactions.** With 500+ unique kingdom cards across expansions, the number of possible game setups is enormous. Each card introduces new mechanics, and cards interact in unpredictable ways. Training a policy network on all of these combinations is clearly intractable, so there needs to be some clever solution to the problem generalizing the strategy of a game which often reveals never before considered interactions and brand new strategic insights in new games.
- **Variable action spaces.** The set of legal actions changes drastically from turn to turn and phase to phase — buying, playing actions, reacting to attacks, and making forced decisions mid-resolution are all structurally different. Not to mention entirely new mechanics are introduced in expansions, and sometimes these mechanics interact with unexpected edge cases. This makes implementing the base game logic alone hard enough.
- **Player interaction.** Attack cards, reactions, and shared supply piles create strategic dependencies between players that a single-agent model can't fully capture on its own.

## Research directions

This project will involve significant experimentation across several axes:

### Card embeddings
How should cards be represented as input to the neural network? Options include learned embeddings, hand-crafted feature vectors (cost, types, tags), or embeddings derived from card text. The representation needs to generalize across different kingdom setups the model hasn't trained on, but also needs to capture card interactions and not just individual card effects. The classic example is the interactions between cards that increase your hand size (draw cards), and cards that increase your available actions (villages). These two card types aren't usually very useful on their own, and require each other to be maximally effective.

### Phase-specific policy networks
A single monolithic policy network may struggle with Dominion's structurally different decision types. Possible approaches:
- Separate policy heads (or entirely separate networks) for action phase, buy phase, and mid-resolution choices (discard, trash, gain, etc.)
- A shared trunk with phase-conditioned output layers
- Framing each sub-decision as its own mini-game

### Modeling player interaction
Should opponent state be part of the policy network's input? Options range from:
- Ignoring opponents entirely (solo optimization). Some kingdoms have limited player interaction, and the game relies mostly on who is most efficient with building their deck overall.
- Including observable opponent state (discard piles, cards in play, VP tokens)
- Modeling opponents as part of the environment with opponent modeling / belief tracking

### Handling hidden information
AlphaZero's MCTS assumes perfect information. Adapting it for Dominion may require:
- Information set MCTS (sampling possible hidden states)
- Learned belief models over deck composition
- Regularization to prevent the model from "hallucinating" knowledge it shouldn't have

## Project structure

```
src/
├── game/                  # Game state and rules
│   ├── card.h/cpp         # Card types, registry, decision callbacks
│   ├── player.h/cpp       # Player state (hand, deck, discard, mats)
│   ├── game_state.h/cpp   # Full game state, turn tracking, supply
│   ├── supply.h/cpp       # Supply pile management
│   └── cards/
│       ├── base_cards.h/cpp      # Copper, Silver, Gold, Estate, etc.
│       └── kingdom_cards.h/cpp   # Kingdom card definitions
├── engine/                # Game execution
│   ├── game_runner.h/cpp  # Turn loop and game lifecycle
│   ├── action_space.h/cpp # Legal action enumeration for the agent
│   └── rules.h/cpp        # Rule enforcement
tests/
├── test_cards.cpp
├── test_game_state.cpp
└── test_action_space.cpp
```

## Current status

**Implemented:**
- Card data model with bitmask types, tags, and `on_play` callbacks
- Card registry for global card lookup
- Player state management (hand, deck, discard, in-play, set-aside zones, mats)

**Next up:**
- Build system (CMake)
- Game state and supply piles
- Base card definitions (Copper, Silver, Gold, Estate, Duchy, Province, Curse)
- Turn structure and game loop
- First kingdom cards
- Action space generation

## Building

*Build system not yet configured. CMake setup coming soon.*

## License

TBD
