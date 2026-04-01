# DominionZero

An AlphaZero-style AI for the card game [Dominion](https://wiki.dominionstrategy.com/).

## Goal

Train a self-play reinforcement learning agent that learns to play Dominion at a high level, using techniques inspired by [AlphaZero](https://arxiv.org/abs/1712.01815). This requires a fast, correct, and modular game engine as the foundation. 

## Why Dominion is hard

AlphaZero was designed for perfect-information, fixed-action-space games like Chess and Go. Dominion breaks those assumptions in almost every way:

- **Massive branching factor.** A single turn can involve chaining multiple actions, each with their own sub-decisions (discard 2 of 5 cards, gain a card costing up to 4, etc.). The decision tree is drastically wider and more variable than chess or Go.
- **Hidden information.** Players don't know the contents of their own deck, opponents' hands, or the order of upcoming draws. The model must reason under uncertainty. Randomness is not necessarily a problem for Monte-carlo simulations because they are random by nature, but often your strategy in a given turn greatly depends on the hand you draw.
- **Combinatorial card interactions.** With 500+ unique kingdom cards across expansions, the number of possible game setups is enormous. Each card introduces new mechanics, and cards interact in unpredictable ways. Training a policy network on all of these combinations is clearly intractable, so there needs to be some clever solution to the problem generalizing the strategy of a game which often reveals never before considered interactions and brand new strategic insights in new games.
- **Variable action spaces.** The set of legal actions changes drastically from turn to turn and phase to phase: buying, playing actions, reacting to attacks, and making forced decisions mid-resolution are all structurally different. Not to mention entirely new mechanics are introduced in expansions, and sometimes these mechanics interact with unexpected edge cases. This makes implementing the base game logic alone hard enough.
- **Player interaction.** Attack cards, reactions, and shared supply piles create strategic dependencies between players that a single-agent model can't fully capture on its own.
- **Actions don't always advance game state** One of the biggest hurdles to implementing Monte Carlo Simulations is that random actions in Dominion will usually never lead to actually reaching an end-game state. Therefore some heuristics may have to be hard-coded for random playouts, which raises the challenge of performance/accuracy trade-offs.

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
- Including observable opponent state (discard piles, cards in play, VP tokens, revealed cards)

### Handling hidden information
AlphaZero's MCTS assumes perfect information. Adapting it for Dominion may require:
- Information set MCTS (sampling possible hidden states)
- Learned belief models over deck composition
- Regularization to prevent the model from "hallucinating" knowledge it shouldn't have

### How much domain knowledge to include
Though the original idea behind AlphaZero was to include zero domain knowledge other than the game rules, this might be intractible for Dominion for a number of reasons:
- Highly specific card interactions: Part of what makes Dominion so interesting is that you will frequently encounter brand new situations you never considered before, with highly specific combos available. Sometimes these combos become a monolithic strategy, meaning not adopting them results in a near guaranteed loss. A neural network generalized over 600+ cards will struggle with finding these 1/1,000,000 situations without help.
- Because turns consist of potentially dozens of individual decisions, some decisions are "obvious", in the sense that there is no reason to consider other options, and they follow naturally from understanding the basic mechanics of the game. For example, if you have a village and smithy in hand, there is almost never a reason to not play Village first. Finding clever ways to prune these decisions could save a lot of searching time for MCTS.

### Should all cards be included in training?
There are specific cards in Dominion that could greatly increase the size of the MCTS search tree, and ignoring them to prune this tree could outweigh the benefit of considering all possible ways to play it. For example:
- Cards that say "this turn, you may ..." will result in a decision being prompted every time a card is gained, played etc, essentially multiplying the number of decisions per turn.
- Cards that say "in games using this ..." change the rules slightly when they are present, and accounting for every possible case of these being present could add a lot of overhead
