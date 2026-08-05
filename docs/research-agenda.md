# Research Agenda

Standing tracker for future research directions beyond the current
campaign. Items graduate to IMPLEMENTATION_PLAN.md tasks when a
campaign or build commits to them. Updated 2026-08-02.

## GPU-native engine + search (the "everything on GPU" moonshot)

Run environment steps AND the MCTS tree on accelerator, PGX/mctx
style: thousands of games as one batched tensor state, env stepping
as tensor ops, search trees as on-device arrays, NN evals fused —
no CPU fleet, no serving hop. Prior art: PGX (JAX board-game envs),
DeepMind mctx (batched MCTS), podracer-style AlphaZero loops.

Why v2 is unusually well-positioned:
- `GameState` is POD/fixed-size/memcpy-cloneable — already a tensor row.
- Cards are DSL op-sequences (data, not code) — a vectorized DSL
  interpreter is GPU-VM execution, bucketed by opcode with masks; 400
  expansion cards = 400 op-programs in a table.
- The trigger bus enqueues frames without recursion — flat per-game
  frame queues map to arrays.
- Golden replays provide the correctness harness: same seeds through
  C++ and GPU engines, diff state hashes ply by ply.

Payoff: for the all-expansions campaign, a handful of H200s replaces
the 8-EPYC CPU fleet — est. 10-50x cheaper per game. Cost: research-
grade build (weeks-months); `custom_step` stateful cards are the
ugliest corner; the GPU tree (mctx-style) is the bigger half.

PREREQUISITE MEASUREMENT (cheap, do first): profile one selfplay
worker to split time between engine stepping / obs encoding / tree
bookkeeping / eval-wait. If engine steps are a small fraction, a GPU
engine alone buys little and the GPU tree is load-bearing. An
afternoon of work; decides moonshot vs. right-architecture.

## Human corpus scaling (the AlphaGo ingredient)

Corpus tiers (tuples): 56K (2026-08-02, scraper session) = upgraded
BC+anchor drop-in; ~500K (~3-5K games) = genuine SL pretrain phase
with held-out human validation; ~2M+ (~15-20K games) = AlphaGo-class
SL as the primary opening teacher (c21 design pillar). Track
attack-response frame count separately (~10K+ attack games needed for
the Militia fix) — converter should tag frame types in the manifest.
dominion.games plays all expansions: the corpus generalizes to the
full-game future. See docs/dominion-games-scraper-handoff.md.

## Skill-weighted corpus (rating-weighted policy, unweighted value)

Decided 2026-08-02. Scraped games are all-comers, so imitation targets
vary in quality. WEIGHT rather than filter: hard-filtering to top-decile
play would gut per-card coverage (Bureaucrat is already the limiting
card at 67 buys in 642 games; a decile cut leaves ~7).

Key asymmetry — **weight the POLICY target by rating, not the VALUE
target.** A weak player's chosen action is a poor imitation target, but
their game's OUTCOME is objective ground truth, and real engine-win
outcomes are exactly the c20 gap this corpus exists to fill.
Down-weighting outcomes by skill discards the signal we built the
pipeline for. Caveat: weak games visit states strong players never
reach, so the value head sees a broader, partly off-distribution state
space — argued to be good for calibration breadth, but be deliberate.

Implementation is a load-time join, not a reconversion: tuple rows
already carry `player_id` and `seat_index`, and ratings live in a
decoupled sidecar (`data/dominion_games/ratings/`), so the weighting
curve can be re-tuned without touching the corpus.

CURVE SIGNED (Jack, 2026-08-03) — coverage measured at 99% of games
(777/784 in the current sidecar), so the scheme is fully fed:
- level < 40: policy weight ~0 (epsilon) — these positions teach the
  VALUE head only (outcomes are ground truth at any skill).
- level 40-50: monotonic ramp from epsilon toward full weight.
- level 50+: HEAVILY over-weighted in policy targets — expert tier,
  the primary policy teachers.
- Unrated (rare at 99% coverage): treat as the low band (value-only).
- Value target weight: UNWEIGHTED by skill throughout (the asymmetry
  above stands).
ACCOUNTING CONSEQUENCE for the c22 gate: the 1M-tuple trigger and the
per-card coverage floors must be counted in POLICY-EFFECTIVE tuples
(weighted), not raw — a corpus dominated by sub-40 games could hit 1M
raw while starving the policy head. The pre-launch audit reports both
raw and effective counts per card.

Tooling exists but has never run live (needs its own account — one
session per account, so it would evict the collector):
`scripts/dgames_ratings.py` (leaderboard poller, append-only time
series) and `scripts/dgames_ratings_join.py` (nearest-in-time join).
Protocol re-derived in `data/dominion_games/recon/RECON.md`. NOTE: no
per-player rating lookup exists in the protocol — coverage depends
entirely on leaderboard depth, which is UNMEASURED.

## Scraper join-age band and game-length distribution (measured 2026-08-03)

True game lengths reconstructed (join age from collector logs + capture
duration), n=957. **Games are shorter than assumed: median 9.2 min, not ~12.**

Split by time control — three distinct populations, all ~19 turns, so the
difference is purely clock speed, not play:
| type | n | mean | median | p10 | p90 |
|---|---|---|---|---|---|
| RATINGS_2P | 834 | 9.6 | 9.2 | 6.1 | 13.4 |
| RATINGS_2P_BLITZ | 49 | 7.2 | 7.1 | 5.1 | 9.2 |
| unrated (no rating type) | 73 | 15.3 | 11.7 | 7.8 | 24.6 |

JOIN BAND: moved 4-12m -> **7-14m** (2026-08-03), measured 25-27 -> ~49
games/hour with ZERO join timeouts. A game cannot be joined after it ends, so
the band structurally excludes games shorter than its lower bound: at 7-14m
that is ~20% of all games, ~90% of blitz, ~15% of rated.

DECISION (Jack): accept that bias. Blitz = time-pressured = worse decisions,
so under-weighting it likely IMPROVES corpus quality for a bot meant to play
well. And short RATED games were measured to be qualitatively ordinary, not a
distinct category — resigned 26.7% vs 23.3%, |margin| 10.0 vs 11.4, turns 17.0
vs 19.4 — so excluding them costs volume, not representativeness.

FUTURE REFINEMENT (not built): per-time-control bands chosen from
`TableDetails` rules BEFORE committing (time settings are a table rule), plus
distribution-shaped non-uniform join-age sampling instead of a uniform band.
Roughly 4-8m blitz / 6-13m rated / 8-20m unrated. Strictly better than any
single global band.

## Per-card / per-pair coverage as the corpus sizing metric

Measured 2026-08-02 over 642 scraped games. Kingdom selection is
random, so games-present is uniform (214-288 per base card) and is NOT
the binding constraint. POSITIVE EXAMPLES are: buys per game-present
ranges 0.26 (Bureaucrat) to 4.9 (Festival), an 18x spread. Size the
corpus for the RAREST card or the model plays Festival well and
guesses at Bureaucrat.

At a ~1,000-positive-examples-per-card floor:
| rarest-card buys | total games |
|---|---|
| 500 | ~4,800 |
| 1,000 | ~9,600 |
| 2,000 | ~19,200 |

So **~10K games for base-set mastery**. A card sits in 40% of base
kingdoms but only 2% of 500-card kingdoms, so full-pool single-card
coverage needs **~20x more, ~200K games**; pairwise interactions
(C(500,2) = 125K pairs) are far beyond that.

Also measured: 642 games gave 640 DISTINCT kingdoms, 99.7% seen
exactly once. Dominion is a generalization problem, not a memorization
one — C(26,10) = 5.3M base kingdoms, C(500,10) = 2.46e20.

## The keep/discard discrimination problem (Militia inversion family)

Root cause established (g40-g50 probes): the VALUE HEAD itself prices
Estate-in-hand above Silver-in-hand — a learned "holding green =
winning" correlation (score-snapshot bias family), which mirrors
cannot correct because both sides share it (symmetric blindness; no
counterfactual outcome data). Canonical regression case:
militia probe `copper_copper_estate_silver`. Candidate fixes ranked:
1. Human imitation on discard/trash frames (needs corpus scale, above).
2. Deep-slice targeted search (2000+ sims) on attack-response and
   trash choice frames -> sharper policy targets from own games.
   Machinery exists; frame-type targeting hook does not. NOT built —
   Jack explicitly deferred (2026-08-02) pending a better idea.
3. Asymmetric scripted exposure (engine3 discards correctly) — fast
   hack, violates no-scripted-opponents rule, time-box if ever used.

## c21 design inputs (accumulated)

- Curriculum pools must be 12-14 cards sampling 10 (a 10-card "pool"
  is ONE fixed kingdom; 50% share overfits by ~20 gens — c20 g20 dip).
- Human-SL init (see corpus tiers) replacing/augmenting BC pretrain.
- Training sims 512+ (sharper targets; c20 used 256; AZ used 800).
- Distributed selfplay: 3-4 CPU boxes into one trainer/GPU — workers
  are already torch-free clients; moderate build.
- MODEL SIZE DECIDED (Jack, 2026-08-02): c21 is 2M-parameter class.
  Keep CardTokenNet shape, shrink width/depth (d160/3L ballpark);
  exact dims tuned by the offline BC ablation on the human corpus
  (held-out human val + probes + throughput column). Rationale:
  capacity not binding on base set (c15 existence proof); ~3.3x
  faster in every CPU-inference context (measured: c15 911 games/hr
  vs c19-class 274 in the honest harness) -> faster evals, duels,
  deployment sims; better data-efficiency at 56K-500K corpus scale.
  Corrected cost model: training-loop throughput is dominated by the
  honest-regime machinery (no tree reuse under per-turn
  determinization; c19 ran 18-20K games/hr clairvoyant WITH the big
  net) — the small net does NOT refund that; it refunds CPU-side
  inference everywhere else.
- SIL verdict pending (c20 g55 bar >= 42%): if cleared, SIL weight
  becomes a standing c21 knob.
- Value-probe instrument must be recalibrated per-campaign from real
  self-play states (synthetic fixed states go OOD — c20 g15 lesson).
- Duchy-gate thresholds were calibrated on money-era nets; an
  engine-shaped net legitimately prices injected dead cards more
  negatively (~-0.14 observed steady-state). Recalibrate.

## All-expansions endgame sizing (very speculative, 2026-08-02)

~400 kingdom cards; ~10^19 kingdoms -> generalization via card-token
architecture (already right shape). Est. 10-50M games; CPU path = ~8
dual-EPYC nodes x 6-8 weeks feeding 2-4 H200s (~$10-20K compute);
GPU-native engine path (above) collapses this to a few H200s. Engine
must grow ~400 cards first (DSL cookbook designed for it). Human
corpus at millions of all-expansion games is the biggest single lever.

## Standing open tasks (tracked in the task list)

- #25 serving-stack search parity (deployed strength +6-10pts
  hypothesis; serving is no longer the training ceiling post-sharding,
  which strengthens the K=1 deploy candidate).
- #24 arena bot-seat value tuples.
- #13 human-data collection push (superseded in scale by the scraper,
  still relevant for exact-replay seeded games).

## c22 design decisions (accumulating, 2026-08-03)

- TRIGGER: 1M human tuples (~8-10K games; per-card coverage floor met).
- From-scratch relaunch, d192/4L class, genuine SL pretrain phase with
  held-out validation (by game), then honest selfplay per c21 recipe.
- NO KINGDOM CURRICULUM (Jack, 2026-08-03): 100% natural random
  kingdoms. The curriculum was a prosthetic for the c20 outcome-data
  monoculture; 1M human games supply engine-win outcomes on natural
  board distribution at the source. Costs removed: fixed-board
  overfit vector, phase wobble, train/eval distribution mismatch.
  DIAGNOSIS REFINED (Jack, 2026-08-03): the c21 chapel-board collapse
  was a CORPUS COVERAGE GAP, not evidence a curriculum is inherently
  needed — Chapel is an expert's card, under-and-badly-represented in
  547 all-comers games; the BC prior was shallow because the teaching
  was. Therefore: PRE-LAUNCH CORPUS AUDIT AS A LAUNCH GATE — per-card
  positive-example counts vs the coverage floors (trashers and
  attack-response cards especially); under-floor cards mean targeted
  collection before launch, not scaffolding after. Rating-weighted
  policy targets amplify the well-played exemplars of expert cards.
  The pool machinery is retained in the repo as a last-resort lever
  but is NOT part of the c22 pre-registration.
- Carry-forward from c21 evidence: flat anchor (no step-down
  schedules), league capped at 15%, 80-turn selfplay cap, sampled-
  state probe calibration per-campaign, milestone battery every 5
  gens (duel + dual probes + skill-aware retention val + vibe pair).
- Retention val metric must be skill-weighted (ratings sidecar) to
  separate "forgetting" from "surpassing" — the c21 g15 lesson.

ADDENDUM (Jack, 2026-08-05): ALL fixed-kingdom machinery removed from
c22 training — kingdom_curriculum empty, no pools, no phases; 100%
uniform random kingdoms everywhere (selfplay, league, eval). Draft:
configs/run_c22_draft.json.
