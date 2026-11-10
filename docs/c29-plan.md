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

The first check already failed. The second was run locally the same day and
turned up something worse than either: the search itself was broken.

## Correction: the search was broken from c25 on

Found 2026-11-10, after the sections below were first written. Numbers are in
the "Drawing board, part 2" entry of `docs/training-log.md`.

`Mcts::select_child` subtracted a flat 1.0 from a child's score for every leaf
in flight below it. Values live in [-1, 1], so with several leaves in flight
per search (`leaves_in_flight = 8`, added for throughput in the c25 prep,
commit `530ffcc`) every batch was spread over different root children and
root visits came out close to uniform over the top 4 (128 sims) or 8 (512
sims) moves, whatever the prior and the value said. The move played was close
to a random pick among the top few.

- Affected: all c25-c28 self-play, their gates, and every 128-sim match in
  "Results so far" below. Not affected: engine3 evals (`EvalRunner`, one leaf
  per game), c24 and earlier, and `bc_a` itself (behaviour cloning).
- Fixed in `src/v2/mcts/tree.cpp`: a leaf in flight counts as one lost visit,
  so the penalty fades as real visits arrive. Regression test in
  `tests/v2/test_mcts.cpp`.
- With the fix, 128-sim search beats `bc_a`'s raw policy 0.580 (sequential
  search: 0.590); before, it scored 0.155.

Read "Results so far" with that in mind: the gate confounds are real in the
code, but the scores there were all produced by the broken search.

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

## Results so far (measured with the broken search)

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

The third-party check (done 2026-11-10, local):

| Net | vs engine3 (200 games, 400 sims, honest, same seed) | Corrected gate vs `bc_a` |
|---|---|---|
| `bc_a` | 50.5% | n/a |
| c28 gen6 | 38.0% | 0.950 |

Corrected gen6 vs gen1: 0.540 [0.471, 0.609], not separable from even.

At the time this read as a non-transitive gain. It was adaptation to the
broken search: six generations trained on games where the mover picks almost
at random among its top moves learned priors and values that survive that,
and `bc_a` had not. With the search fixed, gen6 vs `bc_a` is 0.570
[0.472, 0.668] (sequential search: 0.580), not separated from even, and gen6
is still 12.5 points worse against engine3. No c28 checkpoint replaces `bc_a`.

## The plan

### Step 0: re-gate what already exists (done 2026-11-10, local)

Corrected gates for c28 gen 1-6 against `bc_a`, clairvoyant and honest, 100
boards each; then `bc_a` and gen6 against engine3 on the same seed (200 games
each) and a corrected gen6 vs gen1 match. Results are above. To reproduce:

    JOBS=2 CHUNK=100 PAIRS=100 scripts/box/c29_diagnostics.sh gates
    python -m src.v2.train.evaluate --checkpoint <ckpt> --opponent engine3 --honest --games 200 --seed 20261110

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

Run locally 2026-11-10 with the fixed search (honest, 50 boards x 2 seats):

| Match | Score | 95% CI | Elo |
|---|---|---|---|
| 128 sims vs raw policy | 0.580 | [0.498, 0.662] | +56 |
| 128 sims vs raw policy, sequential search | 0.590 | [0.492, 0.688] | +63 |
| 512 sims vs raw policy | 0.620 | [0.531, 0.709] | +85 |
| 128 vs 128 with the value output zeroed | 0.590 | [0.501, 0.679] | +63 |

Search is an improvement operator on `bc_a` and the value head contributes to
it. The gain is modest and the intervals are wide. Still to run, on a GPU:
512 vs 128, 512 vs zero-value 512, clairvoyant variants, and 8 leaves in
flight against sequential search at 512 sims (only 4 in flight at 128 sims
has been checked against sequential).

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
| Candidates beat `bc_a` in the corrected gate but not a third party (observed, but with the broken search) | Self-play from a frozen parent teaches beating the parent | Gate on head-to-head AND a third-party term (engine3 plus a pool of past nets); let the loop promote so the opponent moves; mix past nets into self-play opponents |
| Search beats raw, candidates do not beat `bc_a` | Targets are good, learning from them is not | Work on the loss and the targets (sharpening in the live loop, policy weight, replay mix) |
| Search does not beat raw, or the zero-value net searches as well as the real one | There is no improvement operator to distil | Fix the search or the value head before any training: more sims, value calibration, playout-backed values |
| Clairvoyant gains vanish under honest search | Self-play is learning to exploit hidden information | Generate self-play with determinized search |
| Regret shows overrides are no better than the prior | Visit targets carry noise, not signal | Target pruning / sharpening, or train only on decisions with a clear visit margin |

Outcome (2026-11-10): none of the rows applied as written, because the
search that produced the targets was broken. With it fixed, search beats raw
and the value head helps, so there is an improvement operator to distil, and
it has never been distilled: c25-c28 trained on flattened targets.

c29 is therefore the c28 recipe run for the first time with a working search:

1. Start from `bc_a`. Do not reuse c25-c28 replay data or checkpoints.
2. Finish the ladder on the box (previous section), then the decision-regret
   run, before generating self-play.
3. Gate with the corrected protocol plus an engine3 term (step 4), so the
   loop can promote.
4. Watch in the first generation: policy-target entropy should stay near the
   BC prior's 0.5 instead of rising to 1.1; games should run about 41 turns
   with over 2 Provinces per seat, not 65 turns and 1.1.

Input features for game progress (Provinces left as its own scalar, piles to
game end, money density) are cheap to add to the derived block but are not
the fix: the raw policy already buys Province 16% -> 87% of the time as the
game goes on and 92-95% once two Provinces are gone, and it reads that from
the pile counts. Revisit only if a fixed-search campaign still greens late.

### Step 4: change the trainer's gate

Whatever step 3 says, the trainer should stop gating with the legacy path.
`run_gate_match` should use `run_paired_match` (greedy, independent search,
paired boards) and accept on the lower end of the paired confidence interval
rather than on a raw 55% of 200 games, and it should also require no loss
against engine3 on paired seeds. This is not implemented yet; the legacy
path is kept so old numbers stay reproducible (`--legacy-gate`).

## Costs

Local Mac (mps): one process plays about 550 games/hour at 128 sims, so a
200-pair match is about 45 minutes; two or three processes fit side by side.
The 512-sim ladder rungs and the regret runs want the rented GPU.
