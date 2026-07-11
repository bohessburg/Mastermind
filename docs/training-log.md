# Training Log

Running record of NN training work (Phase T). Newest entries last.
Engine/bot benchmarks: see `docs/benchmarks.md`. Architecture: `REFACTOR_PLAN.md`.

## MCTS scaffold trials (Phase 6, 2026-07-09)

All trials: 1K sims/move, root-sampled determinizations, uniform priors,
200 seat-swapped games vs EngineBot on random kingdoms unless noted.

| Rollout policy | K | vs EngineBot (excl. ties) | Note |
|---|---|---|---|
| Uniform random | 8 | 0% | 37/40 truncations vs RandomBot: search learns treasure-hoarding — random rollouts can't convert greening into wins |
| BigMoney (no action plays) | 8 | 23.5% | parity with BigMoney itself; engines dead weight in playouts |
| Heuristic + action-playing | 8/4/2 | 33.3 / 43.2 / 51.0% | rollouts piloting engines makes action buys visible |
| EngineLike | 8/4/2 | 58.8 / 60.1 / **65.8%** | gate passed; default = EngineLike K=2 |

Lesson: scaffold-MCTS strength is bounded by rollout quality (sims-scaling
curves flat at every tier). Fewer determinizations → deeper trees won at this
budget. NN value head replaces rollouts entirely.

## Throughput engineering (RTX 5090 box, 24-vCPU quota, 2026-07-09)

2.9M-param MLP (1141→1024→1024→512, policy 357 + value). Self-play cost =
NN inference; engine steps negligible (42ns).

| Config | Games/hr | Bottleneck identified |
|---|---|---|
| M1 Max single pipeline (128 sims) | ~4.5–5.9K | 50/50 inference/plumbing |
| Box single pipeline | 4,486 | single-thread leaf collection (88%) |
| 8 workers, per-worker CUDA | 12,826 | GPU context contention beyond 8 |
| 24 workers, per-worker CUDA | 4,729 | contention collapse |
| 20 workers, CPU inference | 7,084 | EPYC per-core inference too slow |
| Inference server, queue transport | 3,601 | pickle IPC (~1.5MB/request) |
| Inference server, shm transport | 4,650 | server Python loop ~4ms/batch; queue header RTT 375µs–1.2ms |
| **Campaign config: 8 workers × 128 games, per-worker CUDA** | **62K+** | large concurrent-game count batches well; see campaign1 |

Server-loop optimization (single drain queue + spin-poll on shm counters)
committed but CUDA-unverified — bench with
`inference_server.py --bench-server --transport shm --poll spin` on a GPU box.

## Campaign 1 (in progress, 2026-07-09)

Config `campaign1.json` (on-box): 100 gens × 1,024 games, 128 sims, 8 CUDA
workers, replay 2M positions (run-1 whipsaw fix), lr 2e-4 cosine decay,
eval 200 games vs EngineBot @400 sims every 5 gens.

Run 1 (local M1, cut at gen 5): pipeline shakedown. Found value-loss whipsaw
0.68→0.087→0.53 from undersized replay buffer; 0/100 vs EngineBot at gen 5
(expected).

Campaign 1 outcome (CLOSED at gen 45, 2026-07-09): learning proven, then
strategy cycling. Eval vs EngineBot: 0% (gens 1-10) → 5.5% (15) → 15% (20)
→ **30.3% (25, peak — best.pt banked)** → 26.9% (30) → 16.6% (35) → 17.0%
(40) → 16.9% (45). Value loss kept improving (0.65→0.39) while external
eval fell: the net converged to a self-play equilibrium the fixed opponent
punishes — classic naive-self-play cycling, invisible without the fixed-
opponent ladder. ~46K games, ~$2 of box time. Artifacts:
checkpoints/remote/campaign1/ (local sync, 54 files).

Campaign 2 (launched 2026-07-09, running): same base config + the fixes —
AlphaGo-style gating (candidate must beat best.pt at >=55% over 60
seat-swapped games to become the data generator; rejected candidates keep
training) and mini-league (20% of self-play games vs a pool of the last 8
accepted bests). Config: configs/run_gated.json → campaign2.json on-box.
Hypothesis under test: gating pins the data-generating policy to monotonic
external strength and prevents the gen-25 regression pattern.

Campaign 2 addendum (2026-07-09): first launch hit a GATING DEADLOCK at cold
start — with best.pt = random init, candidates train exclusively on random-play
data, learn confidently-wrong priors, and lose gate matches to the random
net's uniform-prior search (gate 33% gen 1 → 0% gen 5; verified NOT a scoring
inversion via a known-strength gate match: campaign-1 gen-25 vs random = 20/0).
Fix: gate_warmup_generations=8 — unconditional acceptance while the data pool
bootstraps, strict gating after. Campaign relaunched.

Campaign 2 outcome (CLOSED at gen 20, 2026-07-09): OVER-DAMPED. After the
warmup fix, gating (0.55 over 60 temp-0 games) admitted exactly one candidate
(gen 12 at 1.00); gens 13-20 all rejected (scores 0.0-0.5); best lineage
frozen at 12; value loss fell to 0.056 = candidates memorizing the frozen
data distribution; eval 0/200 through gen 20 (campaign 1 had 15% here).
Conclusion: c1 (no gate) under-damped, c2 (strict gate) over-damped.
Artifacts: checkpoints/remote/campaign2/ (local, 917MB).

Campaign 3 (launched 2026-07-09 ~midnight, running): threading the needle —
gate 150 games @ threshold 0.53 with gate_temp_moves 8 (sampled-temperature
gate matches kill the 0-or-1 score artifact), gate_force_accept_every 10
(anti-freeze valve), league_fraction 0.25 seeded with campaign-1 gen_0025
as a standing external anchor (league/seed_0.pt confirmed loaded), plus the
same-model inference fast path. Config: configs/run_gated_v3.json →
campaign3.json on-box, seed 20260711. Success bar: track c1's curve through
gen 25 (5.5%@15/15%@20/30.3%@25) then HOLD past 25-40 where c1 collapsed.

Campaign 3 mid-course adjustment (gen 35, 2026-07-10 ~2am): evals 6%(10) →
9%(15) → 23%(20) → 26.3%(25) → 19.9%(30) → 20.0%(35). Beat c1's curve to
gen 25, dipped, then STABILIZED where c1 free-fell (c1: 16.6% and falling at
35). Diagnosis: 0.53@150 gate admitted noise promotions during the 25-30
window (lineage advanced every gen); the ratchet re-engaged by 33. Applied
the single overnight adjustment: gate_threshold 0.53→0.55 patched into the
gen-35 checkpoint payload (resume reads config from checkpoint) and resumed.
Also: /root/box_ctl.sh on the box now handles stop/patch/resume without the
pkill-self-match footgun.

Resume stall + fix (2026-07-10 ~3am): resuming with --resume alone silently
runs ZERO generations — train.py overrides config.generations (and
parallel_workers etc.) from the REQUESTED config, which is the dataclass
default (10) when --config is omitted; 10 - 35 done = 0 iterations, clean
exit 0, orphan wrapper in do_wait. Fix: always resume with BOTH
--config <campaign>.json --resume <ckpt> (box_resume.sh). gate_threshold is
NOT in the requested-override list so checkpoint patches to it survive
resume. Follow-up for the backlog: make bare --resume either inherit the
checkpoint's generations or fail loudly.

Campaign 3 outcome (CLOSED at gen 45, 2026-07-10 morning): best of the three
runs, still failed the hold. Full arc: 6%(10) → 9%(15) → 23%(20) → 26.3%(25)
→ 19.9%(30) → 20.0%(35, threshold 0.53→0.55 mid-course) → 14.7%(40, lineage
frozen at 33) → 9.1%(45, decline locked in after the gen-43 forced-accept
promoted an already-degraded candidate). Valve fired as designed but during
decay it entrenches decay. Peak checkpoint ~gen 25-27 (26.3%). Artifacts:
checkpoints/remote/campaign3/ (local, 1.9GB).

Three-campaign synthesis:
- c1 (no gate): fastest learning, peak 30.3%, then cycling collapse.
- c2 (strict gate, temp-0): frozen lineage, memorization, never left 0%.
- c3 (graded gate + league + valve): fastest EARLY curve (league seeding
  worth ~5 gens), best stability at the dip, but the gate threshold window
  between noise-promotion (0.53) and freeze (0.55) is razor-thin, and the
  valve back-fires during decay.
- Constant across ALL runs: strength erodes after ~gen 25-30 regardless of
  gating regime. The common factors — 2.9M-param net capacity, 128-sim
  search depth for data generation, and outcome-only value targets — are
  now the prime suspects, not lineage management. Candidate next steps
  (for discussion, NOT auto-run): more sims for data gen (256-400),
  bigger/wider net, eval-selected best (gate vs ENGINEBOT instead of
  self-relative), value target = score differential not just win/loss.

Pile-out forensics (2026-07-10, eval-ladder v2): 100 games, c1 gen-25 + 400
sims vs EngineBot, random kingdoms: 24W/72L/4T (25%, consistent with 30.3%
+-noise). Game endings: **66% three-pile, 34% Province, 0 truncated.**
EngineBot never races piles, so pile endings are overwhelmingly NN-steered.
Jack's hypothesis confirmed: the trained system's EngineBot wins ride
substantially on the 3-pile ending EngineBot cannot see (no pile awareness
in its buy chart). Implications: (1) single-opponent eval numbers overstate
general strength; MCTS-scaffold opponent added to ladder for a harder
reference; (2) pile-out was learned honestly from self-play (games between
money-heavy nets end by piles) — it is a real strategy, just a local
optimum; (3) win-conditioned ending split would sharpen this further
(current columns count all games).

Campaign 4 (c1 recipe + 256 sims + [1536,1536,768] net, launched 2026-07-10):
evals 0(1) → 0(5) → 2.5%(10) → **35.2%(15) — new all-time best, above c1's
full-campaign peak at gen 15.** Throughput ~18-20K games/hr (bigger net
batches efficiently; ~1.9x params nearly free on the 5090).
BUT gen-15 forensics (local, 100 games): 38% win rate with **81% pile-out
endings** (c1 gen-25 was 66%). Capacity+search accelerated learning WITHIN
the pile-out local optimum, not out of it. Key readouts for rest of run:
end_province trend, and eval vs the scaffold-MCTS opponent (defends piles).
Caveat now on record: the 50% EngineBot gate could be passed by a perfected
pile-racer; treat any gate crossing as provisional pending forensics +
scaffold-opponent confirmation.

Campaign 4 closeout (2026-07-10): gen 20 eval 45/150/5 = 23.1% — down 12pts
from the gen-15 peak (35.2%). Fastest rise and fastest slide of any run.
Cut at gen ~22 for the c5 cutover. Artifacts: checkpoints/remote/campaign4/.

Human validation of the value-bubble diagnosis: Jack played c4 gen-15 at
FULL power (policy+value+400 sims, nnmcts seat) and won easily; it played
duchy/big-money and made no attempt to contest his Province line. Search
cannot escape the value head's worldview — it optimizes toward what the
evaluator rewards, so the pile/money bubble lives in the VALUE FUNCTION,
not the policy or the lineage. This is the root-cause finding of the whole
campaign series.

Second human confirmation (2026-07-10, exports/C8L3ogdtDnlhHqdN.json,
replay-verified by state hash): lazy human money+Chapel vs c4 gen-15 full
power, 48-29 human, ended on Provinces T37. Bot bought all 8 Duchies
(T5-T25), zero Gold/Province in 37 turns, Estates at the end. Its ceiling
(~29 pts) is structurally ~20 short of an uncontested Province line (48);
it only wins if piles end the game first — true in duchy-mirror self-play,
false vs any Province buyer. The exploit in one game.

Scaffold matchup (2026-07-10, local, 60 games, both sides 400 sims):
c4 gen-15 vs scaffold-MCTS (EngineLike K=2) = 0-60. Endings: 58 province,
2 piles — the scaffold defends piles and plays Provinces, so the duchy bot
never gets its ending and loses every completed Province game. Standings
vs EngineBot (scaffold 65.8%, c4 gen-15 35.2%) invert to 0% head-to-head:
single-opponent eval numbers are not transitive strength. Adds the third
leg of evidence (self-play forensics, human playtests, scaffold matchup).
Note: ~30 games/hr single-threaded local; keep counts small.

Campaign 5 (launched 2026-07-10, running): c4 recipe + scripted_opponents
{"bigmoney": 0.20} — 20% of each generation's games vs scripted BigMoney,
records from the NN seat only, seat-swapped. Purpose: put Province-regime
punishment into the value head's training data (self-play mirrors never
contain it). NOT pure self-play anymore (80/20) but still fully
self-LEARNED — BigMoney is opposition, not supervision (no imitation
targets). Mirrors the AlphaStar-league lesson: discrete-strategic-regime
games need opponent diversity that pure mirrors can't provide. Purity is
recoverable later: anneal fraction to 0 once the regime is internalized, or
population-based self-play. Key metric: per-generation scripted_wins/
scripted_games ("vs_bm thermometer") — should climb from ~0; eval
end_province fraction should rise with it. New instruments this cycle:
DecisionSearcher pybind (single-decision NN-MCTS), bot:nnmcts web seat,
eval --opponent mcts (scaffold; NOTE: very slow, both sides search — use
small game counts), end-condition forensics columns.

c5 gens 1-20: 0/200 vs engine and 0/205 vs BigMoney at every eval (c4 had
69/200 at gen 15). Scripted-pipeline plumbing audited clean (records carry
winner + scripted_nn_player; scripted seats never recorded; value targets
per recorded player). Behavioral probe of gen-20 (policy argmax vs BM,
local): buys ONLY Copper (25-29/game), value head pinned -0.9..-1.0
through entire games — same probe shows c4 gen-15 playing estate/silver/
duchy with a live value range. Diagnosis: pessimism collapse / learned
helplessness. 20% unwinnable-from-the-start BigMoney games feed pure -1
into the value head from gen 1; value flattens to "always losing", search
gradients vanish (all leaves equally bad), policy degenerates, self-play
data quality craters, collapse self-reinforces. The medicine was right,
the dose was wrong: opponent injection needs a curriculum (start ~5% or
introduce after the net can win SOME games; anneal up), not 20% from
cold start. c5 cut at gen ~21.

Campaign 6 (launched 2026-07-10 ~12:20, running): c5 recipe with the
BigMoney curriculum — new scripted_opponent_schedule config (breakpoint
list per opponent, linear interpolation; commit 1c6db6e): pure self-play
through gen 10, 5% BigMoney at gen 11, ramping linearly to 20% by gen 25.
Seed 20260714, checkpoints/campaign6 on box. Rationale: let the net learn
money->points->wins from mirrors first so BM games carry gradient instead
of the uniform -1 flood that collapsed c5. Schedule is patchable at
resume (config+resume both required). New standing instrument: behavioral
probe (argmax buy distribution + value-head range vs BM) on synced
checkpoints ~gen 15-20 — eval columns alone can hide a collapse.

Campaign 6 closeout (2026-07-10): cut at gen ~27. Curriculum worked as
dosed (fractions tracked schedule exactly; value head stayed healthy,
context-aware, BM-pessimistic without pinning) but every eval gens 1-25
was 0/200 vs engine and 0/N vs BigMoney. Diagnosis: improvement loop
stalled at SEARCH, not learning — policy_loss == entropy every gen (net
fully fits its targets; the targets are flat), and a 400-sim search-driven
probe at gen 20 still bought only Coppers. 256 sims over ~30 flat-prior
actions cannot resolve small value differences; c4 only escaped via the
huge (and rotten) pile-race value gradients. Artifacts:
checkpoints/campaign6 on box + gens 10/14/15/20 synced locally.

Campaign 7 (launched 2026-07-10 ~13:30, running): c6 config + treasure
search collapse (commit 537ce98): auto_play_treasures (root treasure plays
forced, unsearched, unrecorded — in selfplay/eval/DecisionSearcher) +
prune_treasure_plays (in-tree: treasures forced before buys, canonical
non-decreasing def order — one path per treasure multiset). Both were
default-off; enabled in campaign7.json. Seed 20260715,
checkpoints/campaign7. Intent: concentrate all sims on real decisions to
un-stall search-target sharpening; treasure plays were ~half of all
searched decisions. Caveat on record: both features assume base-set
treasure commutativity — revisit for Storyteller/Grand Market-class cards.

Campaign 7 closeout (2026-07-10): cut at gen ~16. Treasure collapse
delivered: earliest liftoff of any run (10/188 vs engine at GEN 5; c4
needed 10, c1 needed 15), throughput ~27K games/hr (vs 15-21K), and the
first scripted BigMoney wins ever recorded (3/365 across gens 11-15; c5+c6
total was 0). But the 5% ramp arriving at gen 11 coincided with eval decay
10 -> 4 -> 0 by gen 15 — not a value collapse (vloss stable 0.55, no
pinning), rather BM-loss dilution hitting a net whose policy targets were
still flat (entropy == ploss throughout, both creeping up). Lesson: even
5% at gen 11 is too much too early for a net at ~2-5% engine strength.
Artifacts: checkpoints/campaign7 on box.

Campaign 8 (launched 2026-07-10 ~15:00, running): c7 config, gentler
curriculum — schedule [[14,0.0],[15,0.01],[34,0.20]]: 1% at gen 15,
+1%/gen to 20% at gen 34. Seed 20260716, checkpoints/campaign8. Gates
(shifted for the longer ramp): gen 25 expect >=15-20% vs engine with
vs_bm alive at the ~10% dose; gens 35-45 graveyard hold test at full 20%;
gen 55 parity-trajectory check (>=35-40%). Kill-switch unchanged:
value-head pinning on the behavioral probe.

Campaign 8 mid-run (gen 35, 2026-07-10 late): STAGE-1 CONVERGENCE. First
engine wins (3/197) and vs_bm 35/205 = 17.1% at the full 20% dose (7/113
at 25, 12/164 at 30). Probe trajectory of the money curve, gen 15->35:
Gold at $6-7 went 2% -> 6% -> 11-25% -> 32-57% (now top pick); Province
at $8 19% -> 88->74-81% (locked); Duchy at $6-7 65% -> 5-18%; early $5
Duchy 58% -> 45% (last holdout). The duchy attractor dissolved in
dose-response fashion with no value-head damage (vloss ~0.59 throughout).
Slow-ramp curriculum (1%/gen from gen 15) validated against c7's failed
5%-at-gen-11 jump. Expected ceiling unchanged: this arc converges to
~BigMoney-class play (~21% vs engine); c9 dual ladder (BM gen 6 + engine
gen 11, 1%/gen each, cap 20%; configs/run_c9.json, commit 3112d16) is the
stage-2 vehicle.

Campaign 8 closeout (2026-07-10 night): cut at gen ~47. Full arc delivered
stage 1 and mapped its ceiling: vs_bm 0 -> 6.2%(25) -> 7.3%(30) ->
17.1%(35, peak) -> 14.6%(40) -> 10.2%(45); engine wins flickered (3/197 at
35) but never held. Money curve fully assembled by gen 35 (Gold top pick
at $6-7, Province locked at $8, duchy attractor dissolved); past-peak
erosion after 40 was gentle, no value damage. Conclusion: money-class play
is a ~15-17% vs_bm / ~1.5% engine ceiling — the gate needs engines.
Artifacts: checkpoints/campaign8 on box; gens 5/15/20/25/35 + metrics
synced locally (gen 35 = stage-1 baseline checkpoint).

Campaign 9 (launched 2026-07-10 night, running): dual-opponent ladder —
c8 recipe (collapse flags, 1%/gen ramps) with BigMoney entering gen 6 and
ENGINE entering gen 11, each to a 20% cap (BM full at 25, engine at 30);
60% self-play steady state. Seed 20260717, checkpoints/campaign9, config
run_c9.json (commit 3112d16). Purpose: stage 2 — engine-class opposition
injects "engines beat money" evidence self-play and BM cannot provide.
Key metrics: per-gen scripted engine wins (direct gate progress read),
vs_bm (should recap c8's curve faster), eval vs EngineBot, and probes for
action-card buy weights (Village/Smithy/Witch etc.) — the stage-2 tell.

Campaign 9 closeout (2026-07-11 early): cut at gen 89. Eval peak 9.1% vs
engine at gen 30 — which coincided EXACTLY with peak action-card mass in
the policy (20-40% of buy distribution: Moat/Throne/Library/Village).
Money reinforcement then reconquered: by gen 60 the money spine was the
best of any run (Gold 50% at $6, Province 83% at $8) but action mass
halved and eval slid to 1.5-5%; gens 60-89 froze. Lesson: engine
opposition PULLS actions into the policy, but self-play + beaten-BM data
push them back out — exploration needs protecting, not just seeding.
Instrumentation gap found: aggregate scripted counter can't split BM vs
engine wins (pool ~50% while eval vs BM was 0/200 — resolved for the
future by per-kind counters, commit f4436fa; discriminator test abandoned
as moot). Artifacts: checkpoints/campaign9 on box; gens 20/30/60/67/85
synced locally.

Campaign 10 (launched 2026-07-11 ~06:00, running): protect the action
window. c9 base + sims 256->512 (deeper search to validate action lines),
temp_moves 12->20 (longer exploration), BigMoney anneals DOWN 20%->5%
over gens 40-60 (beaten BM is pure money-reinforcement), engine cap
raised to 25% (full at gen 35). Seed 20260718, checkpoints/campaign10,
config run_c10.json. First campaign with per-kind scripted counters
(scripted_wins_bigmoney / scripted_wins_engine columns). Expected
throughput ~half of c9 (512 sims): ~25-30K games/hr.

EVAL BUG DISCOVERY + PARITY GATE CROSSED (2026-07-11 early morning):
per-kind counters exposed impossible engine-seat training wins (43% at
c10 gen 20 vs 1.5% eval). Codex root-caused it: evaluate_checkpoint (and
the in-training eval + CLI) DROPPED the treasure-collapse flags — every
eval since c7 ran collapse-trained checkpoints in the wrong decision
space (net forced to search treasure plays it never trains on). Training
counters were honest all along; evals were crippled. Fix: eval inherits
collapse flags from the checkpoint (commit 131aca1; regression test).
CORRECTED EVALS (200 games, 400 sims): c8 gen35 = 80.0% vs EngineBot
(not 1.5%); c9 gen30 = 55.9%; c9 gen60 = 87.7%; c10 gen20 = 81.0%.
Narrative corrections: c8's "money ceiling," c9's "decline," and c10's
"slow eval" were all measurement artifacts. THE >=50% PARITY GATE WAS
FIRST CROSSED BY CAMPAIGN 8 and is now exceeded by ~38 points. Gate
crossing remains provisional pending corrected scaffold matchup (old
scaffold results incl. c9g30 0/20 used the broken path too — rerun in
flight) and human playtest. c10 resumed from gen 23 with fixed eval code;
its in-run eval numbers are now trustworthy.

Scaffold confirmation (2026-07-11, corrected flags): c9 gen-60 vs
scaffold-MCTS (EngineLike K=2, 400 sims both sides) = 8W-11L-1T (42.1%),
all 20 games Province-ended. First checkpoint ever to take games off the
scaffold (every prior attempt: 0 wins). Combined with 87.7% vs EngineBot:
PARITY GATE CONFIRMED CROSSED — pending only human playtest. Next
frontier: beat the scaffold outright, then superhuman play.

c10 gen-90 vs scaffold, PROPER SAMPLE (2026-07-11, 200 games via 10
parallel box workers, corrected flags): 50W-137L-13T = 26.7% excl ties.
All parts Province-dominated endings. The scaffold still outranks our
best net head-to-head despite gen-90's 93% vs EngineBot — the searcher
adapts in-game; the chart bot doesn't. (c9g60's earlier 42% was a 19-game
small sample; treat 26.7%@200 as the real baseline.) Scaffold = the next
ladder rung; candidate c11 ideas: scaffold seats as a third scripted
opponent tier, league play, more sims, larger net.

Campaign 11 (launched 2026-07-11 midday, running): c10 recipe with
SCAFFOLD seats replacing engine seats (commit 8d98fba) — the adaptive
rollout-MCTS that beats our best net 73/27 becomes the teacher. Schedule:
BM unchanged (1% @6 -> 20% @25 -> anneal to 5% by 60); scaffold 1% @11 ->
25% cap @35. scaffold_sims 400 = identical to the yardstick, so
scripted_wins_scaffold is a direct read against the 26.7% baseline.
FULL-FIDELITY option chosen: opponent leaves inside the NN's search tree
also resolve via scaffold search — gens will slow to ~45-60 min at full
dose (campaign ~24-30 hrs). Seed 20260719, checkpoints/campaign11.
