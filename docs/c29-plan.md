# C29 Plan: Back to the Drawing Board

Written 2026-11-10, after c25, c26 and c28 all failed their gates against the
human-pretrained net `bc_a`. Read with the 2026-11-10 entry at the tail of
`docs/training-log.md`, which holds the numbers.

## Why stop and re-plan

Three self-play campaigns were rejected by the gate, and the ablations run on
top of that gate said "self-play policy targets make the net worse". Every one
of those conclusions came from one instrument: `run_gate_match` at 128 sims.
Before spending another box on training, two things had to be checked:

1. Is the gate measuring what we think it measures?
2. Is search an improvement operator on this net at all? If 512-sim search
   does not beat the raw policy, no amount of self-play can help.

The first check already failed. The second is what the next box is for.

## What was wrong with the gate

`run_gate_match` copies the self-play config and plays candidate versus best
through `play_routed_games`. Three properties of that path make it a poor
measure of "which net is stronger":

| # | Confound | Where | Status |
|---|---|---|---|
| 1 | **Cross-routed leaves.** A leaf is scored by the net of the player deciding *at the leaf*, not by the net of the player doing the search. Each side's tree is a mixture of both nets, and a move by A is backed up through opponent nodes priced by B's value head. | `SelfPlayRunner::leaf_players()` routing | Confirmed in code and by a C++ test |
| 2 | **Shared reused tree.** With `tree_reuse` on, the subtree one side built is adopted by the other side at its next decision. | `tree_reuse` in the copied config | Confirmed in code |
| 3 | **Sampled openings.** Under `temp_mode = per_seat_buy` the schedule ignores `gate_temp_moves = 0`: buys in each seat's first 6 turns, the first 10 action plies and the first 6 effect plies are sampled at T = 1 from visit counts. A net with a flatter policy makes more random early buys, so the gate penalises flatness itself, not playing strength. | `selfplay_temperature_for` | Confirmed in code and by a Python test |
| 4 | First-N-finished selection: the match keeps the first N games to finish, which favours short games. | `play_routed_games` | Confirmed in code; expected to be minor |

Confound 3 matters most for how the c26/c28 ablations were read: self-play
policy targets are flatter than the BC prior (entropy 0.51 -> 1.1), and the
gate sampled from exactly that flatness for the first six turns.

## The corrected measurement

Implemented in `src/v2/train/diagnostics.py` and `scripts/c29_search_ladder.py`:

- Greedy play: every move is the most-visited root child (`temp_mode = legacy`,
  `temp_moves = 0`, `temp_final = 0`), no Dirichlet noise.
- Independent search: the whole tree of a decision is scored by the net of the
  seat making that decision (`SelfPlayConfig::independent_search`).
- Fresh tree per decision (`tree_reuse` off).
- Paired boards: a slot manifest pins `game_index`, so each kingdom and shuffle
  stream is played twice, once with each net in seat 0. Every board is played
  exactly once per orientation, so there is no first-finished selection.
- Statistics are computed over pairs (paired SE, 95% CI, Elo).
- Per-seat budgets (`seat_sims`), including `raw` (2 sims = the prior argmax,
  checked at 100% agreement), so "search vs no search" is one match.

A match of a net against a separately loaded copy of itself must score exactly
0.500 with zero paired variance. That self-check is part of the launcher.

## Results so far

c28 gen1 against `bc_a`, the same 32 boards in both seat orientations (64
games), 128 sims, local Mac:

| Conditions | gen1 score | 95% CI | Elo |
|---|---|---|---|
| Corrected protocol, clairvoyant search | **0.859** | [0.769, 0.950] | +314 |
| Corrected protocol, honest search | **0.812** | [0.719, 0.906] | +255 |
| Corrected + sampled openings only | 0.766 | [0.658, 0.873] | +206 |
| Corrected + cross-routed leaves only | **0.266** | [0.160, 0.371] | -177 |
| Full legacy conditions | 0.438 | [0.315, 0.560] | -44 |
| Old gate as recorded in c28 (200 games) | 0.464 | | |
| Self-check: `bc_a` vs a second copy of `bc_a` | 0.500 exactly | | 0 |

What this establishes:

- The tool reproduces the old gate's verdict, and the corrected conditions
  reverse it on the same boards.
- Cross-routing alone is enough to reverse it. After a buy the turn passes, so
  the leaves that tell one buy from another are mostly opponent decisions, and
  under cross-routing those are priced by the opponent's value head. The old
  gate came close to swapping the two value heads. gen1's edge is its value
  head, and the gate gave it to `bc_a`.
- Every c28 gate rejected, so `best` stayed `bc_a` and the loop never
  iterated: all six generations were trained on `bc_a`'s self-play.
- The c26/c28 ablation readings ("self-play policy targets cause the
  regression", "self-play value is neutral", "sharpening recovers 8-10
  points") used the same gate and have to be re-measured before anyone builds
  on them.

Corrected gates for all six c28 generations (100 boards, 200 games each):

| gen | Old gate | Corrected, clairvoyant | Corrected, honest |
|---|---|---|---|
| 1 | 46.4% | 0.825 | 0.843 |
| 2 | 41.1% | 0.840 | 0.882 |
| 3 | 27.7% | 0.897 | 0.907 |
| 4 | 30.3% | 0.910 | 0.922 |
| 5 | 28.7% | 0.865 | 0.890 |
| 6 | 33.2% | 0.938 | 0.950 |

All six beat `bc_a` head to head, by more each generation, while the old gate
fell. Step 0's gate part is done; the third-party check is what remains.

What does not fit yet:

- Against engine3 (honest search, 400 sims, scripted opponent) the c28 nets
  are not better than `bc_a`: gens 1-6 pooled 41% over 1100 games, `bc_a` 47%
  over 200. A net that beats its parent 4:1 but beats a third party no more
  often may have learned its parent's games rather than the game. gen1's value
  head was fit to `bc_a` self-play, so this is a live possibility, and it is
  the main thing step 0 has to settle.

## The plan

### Step 0: re-gate what already exists (local, no box needed)

Corrected gates for c28 gen 1-6 against `bc_a`, clairvoyant and honest
(`--determinize per_decision`), 200 pairs each. This says whether any existing
checkpoint is already better than `bc_a`, and whether the c28 loop was
improving or degrading from generation to generation.

    JOBS=2 CHUNK=100 PAIRS=100 scripts/box/c29_diagnostics.sh gates

Then the third-party check: `bc_a` and the strongest c28 checkpoint against
engine3 with the same seed and at least 400 games each
(`python -m src.v2.train.evaluate --checkpoint ... --opponent engine3 --honest --games 400`), and
a corrected match of gen6 against gen1 to see whether later generations kept
improving while training on stale `bc_a` data.

### Step 1: search ladder (is search an improvement operator?)

All on `bc_a`, corrected protocol, clairvoyant and honest:

| Match | Question |
|---|---|
| 128 sims vs raw policy | Does search add strength at the gate budget? |
| 512 sims vs raw policy | Does it at the self-play budget? |
| 512 sims vs 128 sims | Does more search add more? |
| 128 vs 128 with the value output zeroed | Does the value head help search, or does search just follow the prior? |
| 512 vs 512 with the value output zeroed | Same at the self-play budget |

    scripts/box/c29_diagnostics.sh ladder

### Step 2: decision regret (is the search's move actually better?)

`scripts/c29_decision_regret.py` takes positions from greedy search games,
keeps those where the search move differs from the prior move, and plays each
candidate out many times with the policy for both seats, using common random
numbers. Playouts never consult the value head. It reports:

- search move minus prior move, in win-probability points, by honest
  (determinized) playouts, split into early / mid / late game;
- the same in the true world, which gives the part of the gain that comes from
  seeing hidden information (clairvoyant search peeking);
- split-half headroom: how much better the best candidate is than the search
  move and than the prior move;
- how well the value head tracks the playout value.

    scripts/box/c29_diagnostics.sh regret

### Step 3: decide what c29 trains

| Finding | Meaning | Next move |
|---|---|---|
| Search beats raw clearly, and candidates beat `bc_a` in the corrected gate | The loop was working and the gate was hiding it | c29 = c28 recipe with the corrected gate inside the trainer; promote the best existing checkpoint first |
| Search beats raw, candidates do not beat `bc_a` | Targets are good, learning from them is not | Work on the loss and the targets (sharpening in the live loop, policy weight, replay mix) |
| Search does not beat raw, or the zero-value net searches as well as the real one | There is no improvement operator to distil | Fix the search or the value head before any training: more sims, value calibration, playout-backed values |
| Clairvoyant gains vanish under honest search | Self-play is learning to exploit hidden information | Generate self-play with determinized search |
| Regret shows overrides are no better than the prior | Visit targets carry noise, not signal | Target pruning / sharpening, or train only on decisions with a clear visit margin |

### Step 4: change the trainer's gate

Whatever step 3 says, the trainer should stop gating with the legacy path.
`run_gate_match` should use `run_paired_match` (greedy, independent search,
paired boards) and accept on the lower end of the paired confidence interval
rather than on a raw 55% of 200 games. This is not implemented yet; the legacy
path is kept so old numbers stay reproducible (`--legacy-gate`).

## Costs

Local Mac (mps): one process plays about 550 games/hour at 128 sims, so a
200-pair match is about 45 minutes; two or three processes fit side by side.
The 512-sim ladder rungs and the regret runs want the rented GPU.
