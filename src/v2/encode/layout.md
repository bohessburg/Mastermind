# DominionZero v2 Observation Layout

`OBS_VERSION = 1`

`OBS_SIZE = 1141` float32 values. All values are raw scalar counts/ids unless
noted. Def-like ids use `def_id + 1`; `0` means none/empty. This keeps Copper
(`DEF_COPPER = 0`) distinguishable from no card.

Hidden information is intentionally excluded:

- Opponent hand composition is not encoded; only hand count is public.
- Own deck order is not encoded; only own deck composition counts are encoded.
- Opponent deck order and composition are not encoded; only deck size is public.

## Offsets

| Section | Offset | Size | Contents |
|---|---:|---:|---|
| Meta | 0 | 4 | Version and fixed shape metadata |
| Own zones | 4 | 320 | Perspective player's private/public zone counts |
| Opponents | 324 | 225 | Public info for up to `MAX_PLAYERS - 1` opponents |
| Supply | 549 | 528 | Public pile state for up to `MAX_PILES` piles |
| Landscapes | 1077 | 27 | Public landscape/project/artifact state |
| Resources | 1104 | 12 | Current resources and perspective-player tokens |
| Turn | 1116 | 10 | Phase/current-player/turn scalars |
| Decision | 1126 | 15 | Decision kind/source/min/max |

## Meta, offset 0

| Relative | Field |
|---:|---|
| 0 | `OBS_VERSION` |
| 1 | `OBS_SIZE` |
| 2 | Perspective player id |
| 3 | `state.num_slots` |

## Own zones, offset 4

Each zone is `MAX_SLOTS = 64` floats indexed by `Slot`.

| Relative | Size | Field |
|---:|---:|---|
| 0 | 64 | Hand counts |
| 64 | 64 | Deck composition counts, order excluded |
| 128 | 64 | Discard composition counts |
| 192 | 64 | In-play composition counts |
| 256 | 64 | Set-aside composition counts |

## Opponent blocks, offset 324

There are `MAX_PLAYERS - 1 = 3` blocks of 75 floats. Block 0 is the next player
after the perspective player, so 2-player games keep the only opponent dense and
first. Unused blocks are all zero.

Block offset: `324 + block_index * 75`.

| Relative | Size | Field |
|---:|---:|---|
| 0 | 1 | Present flag |
| 1 | 1 | Player id |
| 2 | 1 | Deck size |
| 3 | 1 | Discard size |
| 4 | 1 | Discard top def id + 1, or 0 |
| 5 | 1 | Hand count only, contents hidden |
| 6 | 64 | In-play composition counts |
| 70 | 1 | VP tokens |
| 71 | 1 | Debt |
| 72 | 1 | Coffers |
| 73 | 1 | Villagers |
| 74 | 1 | Favors |

## Supply blocks, offset 549

There are `MAX_PILES = 48` blocks of 11 floats. Unused blocks are all zero.

Block offset: `549 + pile_index * 11`.

| Relative | Field |
|---:|---|
| 0 | Remaining card count |
| 1 | Current top def id + 1, or 0 when empty |
| 2 | Base pile def id + 1, or 0 |
| 3 | Mixed pile length |
| 4 | Trait id + 1, or 0 |
| 5 | Embargo token count |
| 6 | Gain counter |
| 7 | Advantage token count for player 0 |
| 8 | Advantage token count for player 1 |
| 9 | Advantage token count for player 2 |
| 10 | Advantage token count for player 3 |

## Landscapes, offset 1077

| Relative | Size | Field |
|---:|---:|---|
| 0 | 4 | Event ids + 1 |
| 4 | 4 | Way ids + 1 |
| 8 | 4 | Landmark ids + 1 |
| 12 | 4 | Project ids + 1 |
| 16 | 4 | Project bought flags |
| 20 | 1 | Active prophecy id + 1, or 0 |
| 21 | 1 | Sun token count |
| 22 | 5 | Artifact holder player id + 1, or 0 |

## Resources, offset 1104

| Relative | Field |
|---:|---|
| 0 | Current actions |
| 1 | Current buys |
| 2 | Current coins |
| 3 | Current potion coins |
| 4 | Perspective-player debt |
| 5 | Perspective-player coffers |
| 6 | Perspective-player villagers |
| 7 | Perspective-player favors |
| 8 | Perspective-player VP tokens |
| 9 | Perspective-player journey flag |
| 10 | Perspective-player minus-card flag |
| 11 | Perspective-player minus-coin flag |

## Turn, offset 1116

| Relative | Size | Field |
|---:|---:|---|
| 0 | 5 | Phase one-hot in `Phase` enum order |
| 5 | 1 | Current player id |
| 6 | 1 | Current player is perspective player |
| 7 | 1 | Turn counter |
| 8 | 1 | Truncated flag |
| 9 | 1 | Effect stack depth |

## Decision, offset 1126

| Relative | Size | Field |
|---:|---:|---|
| 0 | 10 | Decision kind one-hot in `DecisionKind` enum order |
| 10 | 1 | Decision player id |
| 11 | 1 | Decision player is perspective player |
| 12 | 1 | Source def id + 1, or 0 for no decision |
| 13 | 1 | `min_left` |
| 14 | 1 | `max_left` |
