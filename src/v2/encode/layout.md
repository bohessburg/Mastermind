# DominionZero v2 Observation Layout

Observations are versioned fixed-size `float32` arrays. Def-like ids use
`def_id + 1`; `0` means none/empty. This keeps Copper (`DEF_COPPER = 0`)
distinguishable from no card.

## v1 (legacy)

`OBS_VERSION = 1`

`OBS_SIZE = 1141` float32 values. All values are raw scalar counts/ids unless
noted. Def-like ids use `def_id + 1`; `0` means none/empty. This keeps Copper
(`DEF_COPPER = 0`) distinguishable from no card.

Hidden information is intentionally excluded:

- Opponent hand composition is not encoded; only hand count is public.
- Own deck order is not encoded; only own deck composition counts are encoded.
- Opponent deck order and composition are not encoded; only deck size is public.

### Offsets

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

### Meta, offset 0

| Relative | Field |
|---:|---|
| 0 | `OBS_VERSION` |
| 1 | `OBS_SIZE` |
| 2 | Perspective player id |
| 3 | `state.num_slots` |

### Own zones, offset 4

Each zone is `MAX_SLOTS = 64` floats indexed by `Slot`.

| Relative | Size | Field |
|---:|---:|---|
| 0 | 64 | Hand counts |
| 64 | 64 | Deck composition counts, order excluded |
| 128 | 64 | Discard composition counts |
| 192 | 64 | In-play composition counts |
| 256 | 64 | Set-aside composition counts |

### Opponent blocks, offset 324

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

### Supply blocks, offset 549

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

### Landscapes, offset 1077

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

### Resources, offset 1104

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

### Turn, offset 1116

| Relative | Size | Field |
|---:|---:|---|
| 0 | 5 | Phase one-hot in `Phase` enum order |
| 5 | 1 | Current player id |
| 6 | 1 | Current player is perspective player |
| 7 | 1 | Turn counter |
| 8 | 1 | Truncated flag |
| 9 | 1 | Effect stack depth |

### Decision, offset 1126

| Relative | Size | Field |
|---:|---:|---|
| 0 | 10 | Decision kind one-hot in `DecisionKind` enum order |
| 10 | 1 | Decision player id |
| 11 | 1 | Decision player is perspective player |
| 12 | 1 | Source def id + 1, or 0 for no decision |
| 13 | 1 | `min_left` |
| 14 | 1 | `max_left` |

## v2

`OBS_VERSION = 2`

`OBS_SIZE_V2 = 1717` float32 values. The own-side sections are unchanged from
v1. V2 adds the public, perfect-memory information set for every opponent:
their aggregate collection composition and public discard/set-aside
compositions, each indexed by `Slot`.

Opponent deck and hand *order* remain hidden. Their aggregate composition is
public-by-memory because gains and trashing are public events. Unused opponent
blocks remain all zero.

### Global offsets

| Section | Offset | Size | Contents |
|---|---:|---:|---|
| Meta | 0 | 4 | Version and fixed shape metadata |
| Own zones | 4 | 320 | Unchanged perspective-player zone counts |
| Opponents | 324 | 801 | Three public-memory opponent blocks of 267 floats |
| Supply | 1125 | 528 | Public pile state for up to `MAX_PILES` piles |
| Landscapes | 1653 | 27 | Public landscape/project/artifact state |
| Resources | 1680 | 12 | Current resources and perspective-player tokens |
| Turn | 1692 | 10 | Phase/current-player/turn scalars |
| Decision | 1702 | 15 | Decision kind/source/min/max |

### Meta, offset 0

| Relative | Size | Field |
|---:|---:|---|
| 0 | 1 | `OBS_VERSION` (`2`) |
| 1 | 1 | `OBS_SIZE_V2` (`1717`) |
| 2 | 1 | Perspective player id |
| 3 | 1 | `state.num_slots` |

### Own zones, offset 4

Each zone is `MAX_SLOTS = 64` floats indexed by `Slot`.

| Relative | Size | Field |
|---:|---:|---|
| 0 | 64 | Hand counts |
| 64 | 64 | Deck composition counts, order excluded |
| 128 | 64 | Discard composition counts |
| 192 | 64 | In-play composition counts |
| 256 | 64 | Set-aside composition counts |

### Opponent blocks, offset 324

There are `MAX_PLAYERS - 1 = 3` blocks of 267 floats. Block 0 is the next
player after the perspective player. The first 75 values are the unchanged v1
opponent prefix; the three new 64-slot fields are appended so that this prefix
remains layout-continuous inside a v2 block.

Block offset: `324 + block_index * 267`.

| Relative | Size | Field |
|---:|---:|---|
| 0 | 1 | Present flag |
| 1 | 1 | Player id |
| 2 | 1 | Deck size |
| 3 | 1 | Discard size |
| 4 | 1 | Discard top def id + 1, or 0 (retained alongside discard composition) |
| 5 | 1 | Hand count |
| 6 | 64 | In-play composition counts |
| 70 | 1 | VP tokens |
| 71 | 1 | Debt |
| 72 | 1 | Coffers |
| 73 | 1 | Villagers |
| 74 | 1 | Favors |
| 75 | 64 | Collection composition: deck + hand + discard + in-play + set-aside |
| 139 | 64 | Discard composition |
| 203 | 64 | Set-aside composition |

### Supply blocks, offset 1125

There are `MAX_PILES = 48` blocks of 11 floats. Unused blocks are all zero.

Block offset: `1125 + pile_index * 11`.

| Relative | Size | Field |
|---:|---:|---|
| 0 | 1 | Remaining card count |
| 1 | 1 | Current top def id + 1, or 0 when empty |
| 2 | 1 | Base pile def id + 1, or 0 |
| 3 | 1 | Mixed pile length |
| 4 | 1 | Trait id + 1, or 0 |
| 5 | 1 | Embargo token count |
| 6 | 1 | Gain counter |
| 7 | 1 | Advantage token count for player 0 |
| 8 | 1 | Advantage token count for player 1 |
| 9 | 1 | Advantage token count for player 2 |
| 10 | 1 | Advantage token count for player 3 |

### Landscapes, offset 1653

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

### Resources, offset 1680

| Relative | Size | Field |
|---:|---:|---|
| 0 | 1 | Current actions |
| 1 | 1 | Current buys |
| 2 | 1 | Current coins |
| 3 | 1 | Current potion coins |
| 4 | 1 | Perspective-player debt |
| 5 | 1 | Perspective-player coffers |
| 6 | 1 | Perspective-player villagers |
| 7 | 1 | Perspective-player favors |
| 8 | 1 | Perspective-player VP tokens |
| 9 | 1 | Perspective-player journey flag |
| 10 | 1 | Perspective-player minus-card flag |
| 11 | 1 | Perspective-player minus-coin flag |

### Turn, offset 1692

| Relative | Size | Field |
|---:|---:|---|
| 0 | 5 | Phase one-hot in `Phase` enum order |
| 5 | 1 | Current player id |
| 6 | 1 | Current player is perspective player |
| 7 | 1 | Turn counter |
| 8 | 1 | Truncated flag |
| 9 | 1 | Effect stack depth |

### Decision, offset 1702

| Relative | Size | Field |
|---:|---:|---|
| 0 | 10 | Decision kind one-hot in `DecisionKind` enum order |
| 10 | 1 | Decision player id |
| 11 | 1 | Decision player is perspective player |
| 12 | 1 | Source def id + 1, or 0 for no decision |
| 13 | 1 | `min_left` |
| 14 | 1 | `max_left` |

## v3

`OBS_VERSION = 3`

`OBS_SIZE_V3 = 1788` float32 values. V3 is byte-identical to the complete v2
layout through the Decision section, except Meta values 0 and 1 carry the v3
version and shape. It appends the public global trash-pile composition indexed
by `Slot`, followed by the current selection semantic.

### Global offsets

| Section | Offset | Size | Contents |
|---|---:|---:|---|
| Meta through Decision | 0 | 1717 | Unchanged v2 sections (with v3 Meta version/shape) |
| Trash composition | 1717 | 64 | Global trash counts indexed by `Slot` |
| Select semantic | 1781 | 7 | Current selection's semantic one-hot |

### Meta, offset 0

| Relative | Size | Field |
|---:|---:|---|
| 0 | 1 | `OBS_VERSION` (`3`) |
| 1 | 1 | `OBS_SIZE_V3` (`1788`) |
| 2 | 1 | Perspective player id |
| 3 | 1 | `state.num_slots` |

### Trash composition, offset 1717

There are `MAX_SLOTS = 64` raw count values. Relative index `slot` is the
number of cards for that exact `Slot` in the global trash pile. Counts are
global and independent of perspective; unused slots are zero.

### Select semantic, offset 1781

`SelectSemantic` is a seven-way one-hot vector. It describes what happens to
the card selected by the current decision, independent of the source card id.
The vector is all-zero only for an invalid internal enum value; ordinary
non-selection decisions use the `None` entry.

| Relative | Field |
|---:|---|
| 0 | None |
| 1 | Keep (for example Militia's discard-down-to choice) |
| 2 | Discard |
| 3 | Trash |
| 4 | Topdeck |
| 5 | Gain |
| 6 | Other |
