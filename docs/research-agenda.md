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
