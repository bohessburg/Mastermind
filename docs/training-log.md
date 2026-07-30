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

c11 mid-course fix (2026-07-11 afternoon): full-fidelity in-tree scaffold
resolution stalled worker batches — gen 11 at the 1% dose ran 36+ min
(GPU 16%) vs ~5 min baseline; the synchronous in-tree scaffold searches
block each worker's whole 128-game batch, so cost scales with BATCH
CONTACT, not dose. Flipped to the chart-model option (commit pending in
log; selfplay resolve_scripted_tree_leaf uses Engine chart for Scaffold
lookahead; drive_scripted keeps full 400-sim scaffold for actual moves;
EvalRunner untouched). Resumed from gen_0010 with same config/seed.
Retro-eval context: honest c9-gen20 = 39.4% vs engine, so c11's 38.2% at
gen 10 is recipe-consistent (512 sims halves the generations), not seed
magic; pile-heavy flavor (71.5%) still pending the scaffold column's
verdict.

Campaign 11 closeout + Campaign 12 launch (2026-07-11 night): c11 cut at
gen ~13 after its gen-10 pile-rusher beat the OLD scaffold 8/10 via the
eval runner (9/10 pile endings) — the scaffold's EngineLike rollouts
ignored the pile clock; the whole scripted ladder shared the blind spot.
c12 ships four fixes (commits 8d98fba..7b0c77d): (1) pile-aware rollouts
(race when ahead / deny when behind — pile-outs stay a legitimate,
now-contested strategy); (2) rollout step cap 4096->1024 with margin-sign
cutoff scoring (pile-aware rollouts run long; cost bound restored); (3)
MARGIN VALUE TARGETS, sign-preserving hybrid v=sign*(0.5+0.5*|m|/20) —
wins never train below +0.5, crushes teach more than squeakers; (4)
nested Throne Room rules fix (TR->TR->X played the child 4x and reused
the target; interp.cpp multiplicative-fold bug found by Jack in live
play, survived 10M-step fuzzing). Certification matchup vs the c11-gen10
rusher never completed — games appear to stall to truncation with a
defender that denies the third pile, itself weak evidence the exploit is
contained; launched on Jack's call without it. Seed 20260720,
checkpoints/campaign12, config run_c12.json.

C12 CUTOVER (2026-07-11 late night, resumed from gen 20): deployed in one
restart — (1) ENGINEBOT V2 (pile-clock-aware chart, commit 766abaa):
*** EVAL BASELINE BREAK: all "vs EngineBot" numbers from gen ~21 onward
are against the pile-aware v2 bot and are NOT comparable to earlier rows
(gen-20 was 80.9% vs v1). Expect a drop; that drop is honesty. *** (2)
async scripted offload + per-thread scaffold scratches (bc1a43d);
(3) scaffold cost knobs in campaign12.json: sims 400->192 (Phase-6
flat-curve evidence), K 2->1, opening sims 128 until pile clock arms,
2 scripted threads; (4) observation v2 committed (42bc3d3, 1717 floats,
perfect-memory opponent view) but NOT active for c12 (its net has v1
inputs; obs_version stays 1) — first v2-sighted net comes with c13.
Gen-20 pre-cutover probe on record: point-greedy margin signature
(Estate 47% at $2, Duchy 46% at $5), Province 83% at $8, action mass
14-27% utility-flavored.

Campaign 12 closeout (2026-07-11 late): cut at gen 26 on Jack's call.
Final honest numbers: 81.3% vs EngineBot v2 at gen 25 (77.7% at 21 —
climbing against the pile-aware yardstick), 47.7% vs BigMoney
deterministic (plateaued: 47.5% at gen 20 — the blindness ceiling),
scaffold training seats dominated. Margin value targets validated as the
project's biggest single advance: 15.8% at gen ONE, parity-grade by gen
10, 81% by gen 20-25. Artifacts: gens 20/25/26 + metrics synced locally.
Rationale for the cut: obs-v2 changes input size, so c12's lineage cannot
carry forward — every further gen refined a net whose weights die with
the campaign, while its binding constraint (cannot see the opponent) was
already proven by the vs-BM plateau.

Campaign 13 (launched 2026-07-11 night, running): c12 recipe + OBS V2 —
first sighted net (1717-float observations: opponent collection/discard/
set-aside composition = perfect-memory information set). Net 6.45M params
(input growth). Seed 20260721, checkpoints/campaign13, run_c13.json.
Headline metrics: vs BigMoney deterministic (c12 baseline 47.7% — sight
should unlock matchup recognition and convert the tie/narrow-loss mass),
vs EngineBot v2 (c12 baseline 81.3%), and behavioral probes for
ADAPTATION: does its buy line change when the opponent's collection says
race vs engine? That question has never been askable before.

c13 false start x2 + ROOT CAUSE (2026-07-12 early): obs-v2 launches had
value_loss pinned at ~1.26 (no learning). NOT corruption — data verified
clean, contradiction-free, width-validated (hardening commit b2a4db6).
Actual cause: INPUT CONDITIONING. The encoder emits raw counts/ids (up to
OBS_SIZE itself as a literal feature); v1's distribution sat just inside
the stable region for lr 2e-4 — every prior campaign trained at the edge
of this cliff — and v2's ~600 extra count features tipped gradient norms
over it (offline: v2 pairs diverge at ANY lr raw, fit to MSE 0.11 with
inputs/10 — BETTER than v1's 0.14; the sighted data is more learnable
once digestible). Fix: DominionNet input_scale divisor (default 1.0 =
legacy-identical; checkpoint-persisted; commit pending in log), 16.0 in
run_c13.json. Backlog: principled encoder pass (log1p counts, embed id
fields, drop the size-constant feature). c13 relaunched fresh (third
launch) with obs v2 + input_scale 16.

c13 clean run (launch #4, 2026-07-12 ~03:30, running): after the two
conditioning false starts AND a double-trainer incident (launch #2 left
alive while #3 started — 18 stale processes killed; stop-before-launch
discipline now mandatory), the sighted+conditioned campaign is finally
clean: obs v2 (1717), input_scale 16, margin targets, pile-aware
opponents, async offload, single writer verified. Gen 1: 12.7% vs
EngineBot v2 (25/197), vloss 0.288 — healthiest gen-1 ever. GEN 5:
55.1% (108/196) vs the PILE-AWARE EngineBot — parity-class at generation
five, steepest opening on the hardest yardstick. Endings still rush-heavy
(163 piles) pre-opponent-arrival (BM gen 6, scaffold gen 11).
OVERNIGHT PLAN (Jack's instruction): run to gen 25+; if stalled by 25
(eval flat/declining across ~3 evals, or vloss pathology, or process
death), cut, bank best weights, design c14 with necessary changes, and
launch it. Reference bars: c12 hit 81.3% (v2 bot) at gen 25 and was
plateaued at 47.7% vs BigMoney — c13's vs-BM number is the headline
metric for whether SIGHT breaks the racing wall.

C13 GEN-25 MILESTONE — THE BIGMONEY WALL FALLS (2026-07-12 ~05:30):
overnight decision point passed at full speed. Eval 86.0% vs EngineBot v2
(oscillating 81-87 since gen 10); first Province-majority ending mix
(124/76). THE HEADLINE: deterministic BigMoney eval = 113W-65L-22T =
63.5% — c12's 47.7% plateau (the blindness ceiling, flat across 5 gens)
cleared by 16 points at gen 25. The obs-v2 thesis is CONFIRMED: opponent
visibility -> matchup adaptation -> the pure-racer wall breaks, with no
trade-off against general strength (bm training counter arc: 0 -> 16% ->
30% -> 32.5% -> 28.8% at rising dose; c12 lifetime max ~18%). gen_0025
banked locally. Run continues per the overnight contract (stall clause
nowhere near triggering).

THE SCAFFOLD RETIRES (2026-07-12 morning): c13 gen-40 vs the PILE-AWARE
scaffold (400 sims, K=2), 120 games via 6 box workers: 117W-3L = 97.5%,
endings overwhelmingly pile-outs — the net out-races a competent
clock-defending searcher at its own game. For scale: c10-gen90 managed
26.7% vs the WEAKER pre-fix scaffold; every campaign through c9 scored
zero. With EngineBot v2 at ~85-89% and BigMoney at 63.5%, every scripted
opponent in the project is now decisively beaten. The measuring ladder
above the net is now: humans. c13 continues (gen ~62, no decay, vloss
0.227 still descending); remaining program instruments: paired-context
adaptation probe (needs slot_to_def pybind helper), human playtests with
a v2-aware web loader (input_scale + obs_version plumbing — flagged for
web session).

Campaign 13 closeout (2026-07-12, cut at gen 96 on Jack's call — plateau
confirmed, no decay ever): the sighted campaign ends as the strongest and
healthiest run of the project. Final ladder: ~85-89% vs EngineBot v2
(stable band from gen 15 to the end, vloss 0.34 -> 0.23), 71.0% vs
BigMoney deterministic at gen 65 (blind ceiling was 47.7%; nine campaigns
scored ZERO), 97.5% vs the pile-aware scaffold (117-3). Flagship
checkpoint: gen_0065 (89.2% engine / 71.0% BM). Banked locally: gens
25/40/65/96 + metrics. Program state: every scripted opponent decisively
beaten; remaining yardstick is human play. Open instruments: v2-aware web
loader (obs_version + input_scale — web session), paired-context
adaptation probe (needs slot_to_def pybind helper), and the superhuman
agenda (league self-play, card-structured architecture, expanded pool).

Human playtests vs c13 gen-65 (2026-07-12 morning, Jack, full-power
nnmcts @400 sims, v2-aware loader verified — session timing + default
checkpoint resolution confirm gen-65 played; NOTE export "obs_version"
field records the format constant, not the policy version — cosmetic fix
pending):
- Game 1 (no Moat in kingdom): Jack 42-28 via Witch/Sentry/Lab engine (8
  Witch plays). Bot: disciplined money+VP, T11 first Province, zero
  engine, drowned in Curses.
- Game 2 (Moat AND Sentry available): Jack 39-32 via 6 Witches + 4
  Militias from T2. Bot bought ONE Moat at T21 (17 turns late — reactive
  margin-drift, not threat modeling) and never bought curse-trashing
  Sentry.
Diagnosis reaffirmed: the net plays refined money (beats BM 71% — no BM
clone does that; its edge is margin discipline + pile-clock calculus) but
has NO engine repertoire and NO attack response, because no training
opponent ever demonstrated competent engines or attack campaigns —
sight enables recognition, not repertoire; data poverty is the binding
constraint. Jack remains champion.

Search depth analysis (2026-07-12): web/eval 400 sims over K=2
determinizations (2x200 trees), training 512; leaves = value head (no
rollouts). Buy nodes carry 8-18 LEGAL children (treasure collapse removed
treasure-play decisions, not buy width); prior concentration (perplexity
~3) makes EFFECTIVE branching ~3-6; principal-variation depth ~6-15
plies; with ~6 plies per game-turn (both players, action+buy+effect
decisions) that is only ~2-4 TURNS of true lookahead. Engine payoffs sit
25-50 plies out — categorically beyond any raw-sims budget; depth beyond
the tree must be amortized into the value head (the AlphaZero lesson).
Costs identified: root Dirichlet noise taxes ALL legal children (up to
18) one ply deep; every decision builds its tree FROM SCRATCH; K=2 halves
the budget.

C14 SEARCH PACKAGE (approved: tree reuse + top-k expansion):
1. TREE REUSE across decisions — carry the chosen child's subtree to the
   next decision (2-3x effective sims free; interacts with
   determinization resampling and the async scripted offload — the
   delicate chunk).
2. TOP-K EXPANSION — expand/noise only the top-k (~8) prior children per
   node; hardens PUCT concentration, caps the root-noise tax, converts
   width to depth.
Also queued for c14 design: K=1 for data gen (knob exists), deep-slice
data generation (small fraction of games at 8-16x sims to bootstrap the
value head on long plans), engine-dominant curriculum kingdoms, Jack's
exports as seed/league material, card-structured architecture (slot
embeddings) for cross-kingdom synergy generalization.

CAMPAIGN 14 DESIGN (agreed 2026-07-12):
- WARM-START from c13 gen_0065 (same architecture; resume-style weight
  load into a fresh campaign dir; fresh replay buffer; lr schedule
  restarts). Rationale: the isolate-variables era is over; every gen
  should buy NEW capability, and a warm start makes the ancestor league
  automatically strength-adjacent (no cold-start helplessness).
- OPPONENT POOL: scripted opponents REMOVED from training entirely -> BM
  + scaffold move to the EVAL LADDER as sentinels (regression alarms,
  zero data anchoring). Training pool = mirrors + LEAGUE of past selves:
  the banked c13 spread (gens 10-96, 12 checkpoints, all v2/scaled).
  League sampling must be strength-matched / dose-ramped (AlphaStar
  lesson; c5 learned-helplessness precedent). Older-campaign checkpoints
  (v1 obs) excluded — per-seat encode plumbing not worth it.
- KNOWN LIMIT of league-only: the family is a money-style monoculture —
  league diversifies strength, not strategy class. Repertoire growth must
  come from:
- CURRICULUM KINGDOMS (new feature): a kingdom-pool schedule — phases of
  engine-forcing boards (Village/Smithy/Lab/Market/Festival-dense, cheap
  trashing) where money demonstrably loses, mixed back to random kingdoms
  on a schedule, so the mirror equilibrium itself learns engines.
- DEEP-SLICE DATA GEN (new feature): a small fraction of games at 8-16x
  sims so search can occasionally REACH engine payoffs and bootstrap the
  value head (depth amortization flywheel).
- SEARCH: tree reuse (hash-gated, K=1) + top-k expansion AS AMENDED after
  Jack's lock-in observation: full width + noise at the ROOT (targets and
  exploration preserved), top-k only in-tree, plus one random wildcard
  child per expansion (epsilon-exploration; prevents self-reinforcing
  prior collapse — the top-k-everywhere variant would harden the money
  attractor).
Build order: (A) tree-reuse/top-k chunk in flight + amendment on landing;
(B) curriculum kingdom schedule; (C) deep-slice sims; (D) league ramp /
strength-matched sampling over explicit checkpoint list + eval-ladder BM/
scaffold sentinels; (E) run_c14.json + warm-start verification. Launch
when a new box is provisioned.

C14 BUILD COMPLETE (2026-07-12): all five chunks landed and green (111
py + full C++ suites, incl. an all-features integration smoke): (A) tree
reuse (hash-gated, K=1) + top-k expansion with root exemption + wildcard
child (600ec61, 2e42283); (B) kingdom curriculum with segment-exact pool
mixtures (fa78314); (C) deep-slice data gen (9ed633c); (D) ungated league
with loss-weighted opponent sampling over explicit checkpoint lists,
league_schedule ramp, obs-version guards, eval sentinels (edb2db5); (E)
--init-weights warm start (weights-only; fresh optimizer/replay/
schedules) + run_c14.json (this commit): warm-start c13 gen-65, 12-seed
c13 league, engine-forcing curriculum (60% pool gens 1-15, 30% to 40,
random after), 5% deep slice @4096 sims, no scripted training opponents,
BM sentinel eval. READY TO LAUNCH on next box provision. Hardware rec:
1x RTX 5090, 16-24 vCPU, >=192GB cgroup RAM (v2 replay ~18GB + 8 workers
~10GB+), 40GB disk, ~$0.30-0.40/hr; ~8h to gen 100.

Campaign 14 LAUNCHED (2026-07-12 evening, new box ssh -p 10229
root@198.2.214.6 — RTX 5090 32GB, 183GB cgroup, 32GB disk): warm-start
from c13 gen-65, 12-seed ancestor league (lean weights-only checkpoints;
NOTE league loader needs payload["generation"] — patched on-box after the
strip removed it), engine-forcing kingdom curriculum, 5% deep slice
@4096, tree reuse + top-k(8, root-exempt+wildcard), no scripted training
opponents, BM sentinel evals. Bootstrap notes for the record: fresh-box
cmake needs -DBUILD_TESTS=OFF (Catch2 GitHub fetch blocked) and the
tarball must include tests/ (v2_fuzz sources) — handoff doc updated
mentally, launch scripts on box (/root/launch14.sh, stop_train.sh with
worker-tree kill). GPU 95%, single writer verified.

C14 THROUGHPUT SAGA RESOLVED (2026-07-13 early): generations were running
~2K games/hr (vs ~15K expected). Eliminated in order: deep slice (real
cost, disabled — needs cross-game batched deep runner before returning),
league model-splits (real, capped via league_opponents_per_gen=3),
tree reuse (innocent as toggled, but fixed properly anyway with
VISIT-TARGET semantics, 26fbb17: inherited root visits count toward
sims_per_move, min_new_sims=64 noise floor), top-k (innocent), the box
(PCIe x16, 31-core quota, 434K evals/s microbench — exonerated). ROOT
CAUSE: segment-sequential runner phases collapsed GPU batch occupancy to
~7-10 leaves/batch. FIX: SLOT-MANIFEST RUNNER (slot manifest runner
commit): one runner per worker, ALL games concurrent, per-slot model
pairing/kingdom pool/sims; leaves tagged by model id, python batches per
model per pass; 3.49x measured batch fatness. RESULT: gen 8 on the full
restored recipe (8 workers, reuse+top-k+league+curriculum) = 10,004
games/hr (5x the low) AND the best eval of the campaign: 182W-15L-3T =
92.4% vs EngineBot v2. Lesson for the log: on latency-bound GPUs,
features that fragment batches (by phase, by model, by sims tier) cost
10x more than their game counts suggest; slot-level heterogeneity is the
architecture that makes feature mixing free.

C14 RUN HISTORY (2026-07-13): post-throughput-fix, c14 ran gens 8-38 at
8.5-10K games/hr on the full recipe. Evals vs EngineBot v2 oscillated
85-92% (gen 15 dip 84.8% coinciding with curriculum phase flip; gen 20
89.7% with campaign-best BM sentinel 72/100 and vloss 0.251). BOX
MIGRATION mid-run: box 2 (port 10169) proved flaky (one spontaneous
restart wiped the container disk earlier in the campaign); migrated at
gen 20 to a fresh RTX 5090 box (port 10913, verified different physical
machine). gen_0020.pt pulled home (rsync --partial after two scp
truncation corruptions — scp is banned for >10MB on this link), pushed
to the new box with 12 lean c13 league seeds, resumed with --resume
(config init_weights must be popped — mutual exclusion). COST: the 860MB
replay buffer was not migrated, so gens 21+ trained from a fresh buffer:
gen 21 dipped to 81.5%, recovered to 89.7% by gen 30; BM sentinel dropped
to 59-60 and only partially recovered (64 at gen 35 vs 72 at gen 20).

C14 REPERTOIRE PROBE (2026-07-13): c14-gen20 vs c13-gen65 policy buy
preferences on an identical engine-forcing kingdom (Village/Smithy/Lab/
Market/Festival/Cellar/Chapel/Moat/Council Room/Throne Room). c14 moved
substantial probability mass from treasure to actions at the engine-
critical price points: action mass $3: 21.3%->35.5%, $4: 31.9%->45.6%
(Throne Room 6.5%->11.7%), $5: 27.3%->44.6% (Council Room 3.4%->6.5%).
BUT the argmax buy remained money at every coin level (Silver at $3-5,
Gold at $6-7). Verdict: the curriculum measurably softens the money
prior but has not produced a committed engine line — mid-transition at
best, and eval metrics were saturated/flat gens 20-38.

C14 CUT (2026-07-13, gen 38, Jack's call): metrics plateaued (85-92%
band, BM sentinel flat ~60-64), probe shows softening but no repertoire
crossover. Flagship candidates: gen_0020 (89.7% eval, BM sentinel 72 —
best sentinel) and gen_0030 (89.7% eval post-recovery). Artifacts pulled
home: gen_0030/0035/0038 + metrics.csv + replay_state.npz (134MB, kept
as OFFLINE VALIDATION DATA for the c15 architecture work). Conclusion
carried forward: c13->c14 exhausted what data mixture + search tuning
buys through a flat MLP; the remaining repertoire gap (no engine
commitment, no attack response) is attributed to the ARCHITECTURE.

C15 DESIGN — CARD-TOKEN TRANSFORMER (2026-07-13, approved by Jack):
replace the flat MLP [1536,1536,768] over the 1717-float obs with a
set-transformer over card tokens, built as a pure model-side change (the
C++ encoder and obs v2 layout are UNTOUCHED; tokenization is a gather/
reshape of the existing vector inside forward()).
- TOKENS: one per card DefId present in the game (~17 for 2p: 10 kingdom
  + 7 basics) + 1 global token. Card token = learned DefId embedding +
  per-card scalars (own counts by zone log1p, opponent perfect-memory
  counts, supply remaining, cost, type flags). Global token = phase,
  coins/actions/buys, VP, turn, pile clock, decision kind.
- BODY: 2-4 pre-LN transformer layers, d_model 128-256, 4 heads (~1.5-2M
  params). Rationale: attention computes card-card relations directly
  (Witch<->Moat, Village<->terminals) — the interaction structure a flat
  MLP must discover as high-order feature conjunctions and demonstrably
  hasn't.
- POLICY: pointer head — card-indexed actions (play/buy/select X) scored
  from card X's output token per action family; non-card actions from the
  global token; assembled into the flat ACTION_SPACE_SIZE logit vector so
  the evaluate(obs, mask) interface and all C++/worker plumbing are
  unchanged. Kills the input_scale=16 hack (principled log1p featurization).
- VALUE: from the global token. Margin targets unchanged.
- GATES before any campaign: (1) offline fit on c14 replay_state.npz must
  beat the MLP on held-out value+policy loss (reuse the input-conditioning
  offline harness); (2) python-side eval throughput bench >= 70% of MLP
  at realistic batch sizes, else shrink d_model/layers.
- MIGRATION: from-scratch c15 (no warm start across architectures), but
  the league loader reconstructs opponents from their own payload configs,
  so c13/c14 checkpoints seed the ancestor league from gen 1. Recipe
  otherwise carried over: margin targets, obs v2, league, curriculum,
  tree reuse + guarded top-k, slot-manifest runner.

C15 GATE 1 PASSED (2026-07-13): offline fit on 200K c14 replay positions
(same split/losses, 3 epochs, lr 1e-3): CardTokenNet (1.52M params,
d192/3L/4H) vs DominionNet (6.45M): val value MSE 0.202 vs 0.360 (-44%),
val policy CE 1.152 vs 1.162, val total 1.354 vs 1.522 — transformer
still improving at cutoff while the MLP had flattened. Correctness test
validates tokenizer counts and pointer scatter against live engine state
(tests/v2/py/test_card_transformer.py). Prototype delivered by Codex:
src/v2/train/card_transformer.py + offline_fit.py. Remaining gate 2:
eval throughput >=70% of MLP on a CUDA box (bench_eval.py, in progress
with the train.py/league/web factory integration).

C15 LAUNCHED (2026-07-13, box 3 ssh -p 10913): CardTokenNet from scratch
(1,515,302 params, d192/3L/4H confirmed in launch banner) on the full c14
recipe. Gate 2 passed on-box: compiled fullgraph + bf16 + bucketed
inference = 364K evals/s @ bs256 / 438K @ bs1024 on the 5090 (~3-5x
headroom over pipeline demand; eager was 69K and kernel-launch-bound).
Integration shipped by Codex and reviewed: build_model() factory at every
construction site (train/gating/workers/inference server/web), legacy
arch-less checkpoints load as before (verified against real c14 gen_0020),
server_compile/server_autocast_bf16/server_batch_buckets config knobs
with per-bucket startup warmup, mixed-arch league worker table covered by
tests (9 py tests green in pytest AND standalone modes). League kept per
Jack's decision as cold-start collapse insurance (c1-c5 precedent), 11
mixed-arch seeds (8 c13 + c14 gen_0020/0030/0038), removable mid-campaign
for a causal read once early gens look healthy. GPU 90% at launch,
1 trainer + 8 workers.

BENCHMARK REFRAME (2026-07-13, Jack): "EngineBot" has never been a real
engine player — it is bigmoney with extra logic (action purchases + pile
clock), so every eval ceiling in this log (the 85-92% saturation, the
"beat EngineBot" milestones) was measured against a money bot, and no
part of the training loop ever contained genuine engine pressure: the
league seeds are themselves nets trained against this benchmark, so
seeding distilled anti-bigmoney play, not engines. This is the missing
explanation for why engine repertoire never emerged even with the
engine-forcing curriculum (opportunity without punishment). A separate
session/agent is building a proper kingdom-aware engine bot that
exploits engine-friendly kingdoms (DO NOT touch that code from this
session). PLAN: c15 runs as designed (architecture experiment); when the
new engine bot lands it becomes the eval benchmark AND likely a training
opponent, and we scale back to a simpler c13-era recipe against it to
see how much of the c14 machinery a real adversary replaces.

C15 ENGINE3 CALIBRATION (2026-07-13, gen 19, 200 games @400 sims, random
kingdoms, on-box isolated build): first honest strength reads vs the new
kingdom-aware EngineBotV3 benchmark. c15 gen_0019 (transformer, from
scratch, 19 gens): 94W-101L-5T = 48.2% excl ties. c14 gen_0020 (MLP
flagship, warm-started lineage of 85+ gens): 37.4%. c13 gen_0065: 34.5%
(other session's measurement). The 19-generation transformer beats the
entire MLP lineage by ~11-14 points against real engine pressure and
plays near-parity with v3 itself. Ending forensics: c15 games end by
piles 42/200 vs c14's 12/200 — the transformer engages the pile game
substantially more. Chart-bot evals remain saturated/uninformative;
engine3 is now the benchmark of record per Jack.

C15 CALIBRATION ADDENDUM — engine2 column (2026-07-13, same protocol):
c15 gen_0019: 43.7% (83W-107L-10T, piles 34/200). c14 gen_0020: 35.8%
(67W-120L-13T, piles 9/200). With c13 gen_0065 at 35.4% (engine2) /
34.5% (engine3): both MLP flagships sit ~34-37% against either real
engine bot; the gen-19 transformer is +8 vs engine2 and +11 vs engine3.

C15 EMERGENT ATTACK RESPONSE VERIFIED (2026-07-13, gen 20): Jack observed
anecdotal tit-for-tat attack behavior in local play. Probe (8 scripted
games on a Witch/Militia/Moat kingdom; inject 2 Witches + 2 Militias into
the opponent-collection obs fields, diff raw policy buy probs): c15-gen20
attack-buy mass jumps $4: 16.0%->29.4%, $5: 25.7%->39.1%, $6-7: ~+6pts
when the opponent owns attacks; Moat response flat (retaliation, not
turtling); $8 stays on Province either way. c13-gen65 on identical
protocol: <1pt movement everywhere — the MLP never learned to read the
same perfect-memory fields. First verified emergent repertoire behavior
attributable to the card-token architecture, present in the raw policy
before search.

C15 TIT-FOR-TAT STUDY (2026-07-13, gen 20 + sweep; 213 buy states, obs
counterfactual edits, policy-only): (1) IN-KIND retaliation — opponent
Witches raise own Witch buys (18.7->29.1% @$5) leaving Militia flat;
opponent Militias do the reverse; opponent MOATS *discount* own Witch
(18.7->16.9%), raise own Moat, and RAISE value (+0.76->+0.85). (2) Dose
response: policy saturates at 2 enemy attacks (~39% attack mass @$5);
value slides monotonically +0.76 / +0.24 / -0.03 / -0.28 at dose
0/1/2/4 — one enemy attack card cuts assessed position by ~2/3. (3)
Emergence sweep (gens 5/10/15/20, shared states): conditional response
fully formed by GEN 5 (learned from mirror self-play; league seeds
barely attack); later gens sharpen baseline aggression (9.7->15.4% @$4)
rather than create the response. c13-gen65 identical protocol: <1pt.
Probe script: titfortat_probe.py (session tmp; recreate from log if
needed).

C15 GEN-30 MILESTONES (2026-07-13): chart eval 93.9% (185-12-3), BM
sentinel 82/100 (trajectory 58/65/64/69/75/82 — +10 past the c14-era
ceiling of 72, still climbing, no oscillation), vloss 0.053 (best).
ENGINE3 RE-CALIBRATION: gen_0030 100W-94L-6T = 51.6% excl ties @400
sims — the NN is now ahead of EngineBotV3, the strongest scripted
player, up from 48.2% at gen 19 (MLP flagships: 34-37%). First time any
DominionZero net has beaten the best available scripted opponent.

C15 SCORE-SNAPSHOT BIAS PROBE (2026-07-13, gen 30; Jack's theory from
play: "buys Duchies to compensate when given curses"): 42 mid-game buy
states, counterfactual obs edits. CONFIRMED at policy level: pure score
deficit (opp +2 Duchies, own deck untouched) quadruples mid-game Duchy
buys (1.2->4.9%); search amplifies further in real play. Twist: own +3
curses -> reaches for ACTIONS (53.7->58.7%), not Duchies — the
point-chasing is score-driven, not junk-driven. Deeper finding: value
head over-prices snapshots — own +3 Estates (=+3 VP + junk) reads -0.78
vs baseline -0.37, nearly as bad as +3 curses (-0.91): junk aversion and
deficit aversion both directionally right but magnitude-miscalibrated.
Same root as the 1600-sim regression (deeper search leans on value and
amplifies distortion; 31.0% vs 48.2% @400 at gen 20). c16 candidate fix:
temper margin targets (blend with win/loss or anneal margin weight) so
mid-game deficits price as recoverable. Probe pattern: counterfactual
obs edits, see also TIT-FOR-TAT STUDY entry.

HUMAN BENCHMARK, STATED PLAINLY (2026-07-13, Jack): Jack has NEVER lost
a game to any checkpoint, any architecture, any campaign. Closest ever:
c15 gen-25, 29-27 (engine board, decided by endgame Duchy/Estates; see
exports/rESH3EQclFbz2wN-.json). c15 gen-30 lost 48-16 to a committed
Village/Witch/Sentry curse engine (exports/Ds7pDyZgtoXRZ74D.json).
Scripted milestones (engine3 51.6%) establish strongest-artificial-
player status only. Human W-L vs checkpoints is now a tracked metric of
record; first bot win over Jack at full attention is the project's real
milestone.

C15 GEN-40 CALIBRATION (2026-07-14 early): 106W-90L-4T = 54.1% vs
engine3 @400 sims. Curve: 48.2 (g19) / 51.6 (g30) / 51.8 (g35) / 54.1
(g40) — the g30-35 flat window was a pause, not a plateau; stagnation
call retracted pending gen-45 read (decision rule: >=55% at g45 -> c15
continues; ~52% -> cut and launch c16). All pre-random-curriculum data;
flip at gen 41. c16 fully staged meanwhile: engine3 selfplay opponent
landed (136/136 C++ tests, per-kind metrics verified in worker smoke),
run_c16.json validated (40% engine3, no curriculum/league/deep-slice,
engine3 sentinel 200g, warm-start placeholder pending best-checkpoint
choice).

C15 GEN-45 CALIBRATION + BOX INCIDENT (2026-07-14 early): gen 45 vs
engine3 @400 sims: 125W-74L-1T = 62.8% — +8.7 pts in five gens; the
random-kingdom flip (gen 41) sharply accelerated honest strength.
Curve: 48.2/51.6/51.8/54.1/62.8 at gens 19/30/35/40/45. Decision rule
satisfied (>=55%): c15 CONTINUES; c16 stays staged. Chart eval 93.0%,
BM sentinel 76, vloss 0.031 (steepest drop of campaign, post-flip).
INCIDENT: immediately after the 400-sim duel completed, box 3 dropped
("connection closed by remote host", then connection refused) during
the 1600-sim duel — matches the vast restart signature (disk wipe +
port reassignment). EXPOSURE: gens 35/40/45 checkpoints were never
pulled home; best local checkpoint is gen_0030. Awaiting box
reappearance / Jack's vast console for new port. 1600-sim maturity
check still owed.

BOX INCIDENT CORRECTION (2026-07-14): box 3 did NOT restart — uptime 35
days, trainer PIDs continuous (29h), disk intact. The drop was a
transient vast proxy/network outage (~25 min of refused connections).
Gens 35/40/45 pulled home and verified during the scare (good hygiene
regardless). Training never stopped; no resume needed. Lesson: verify
uptime + process etimes before assuming the restart-wipe scenario;
resume15.sh staged but correctly NOT run (a blind resume would have
double-launched against a live trainer).

C15 SEARCH-DEPTH CHECK #2 (2026-07-14, gen 45): 1600 sims = 42.4% vs
engine3 (n=100) against 62.8% @400 — the deep-search regression
persists at nearly the same relative magnitude as gen 20 (31.0 vs
48.2). Value loss improving 0.067->0.031 has NOT fixed search
amplification of value miscalibration; accuracy and calibration are
diverging axes. Strengthens the tempered-margin-target hypothesis for
c16. Standing policy: all evals and web play at 400 sims.

C15 GEN-50 (2026-07-14): chart eval 95.9% (188-8-4, record), vloss 0.023
(0.040->0.031->0.023 post-flip freefall), BM sentinel 72. Engine3: 113W-
84L-3T = 57.4% — down from gen 45's 62.8 (n=200, sigma ~3.5: a ~1.5-
sigma move; curve 48.2/51.6/51.8/54.1/62.8/57.4 at g19/30/35/40/45/50).
Read: gen-45 was likely partly a high outlier; underlying level ~mid-to-
high 50s and still trending up vs the 51-54 band of g30-40. Throughput
note: random-phase gens run ~1,250 games/hr wall (thinking-denser games),
down from ~2,100 mid-campaign; no starvation signature.

C15 CUT / C16 LAUNCHED (2026-07-14 evening): gen-55 engine3 read 56.2%
confirmed the flatten (62.8 g45 was an outlier; 57.4 g50 / 56.2 g55 =
two consecutive flat checks; vloss halved 0.031->0.017 across the same
span without strength gains — accuracy/calibration divergence again).
c15 FINAL: 55 gens, best-measured engine3 62.8% (g45), underlying
~56-57%; flagships gen_0045/0050/0055 + full eval-gen archive local.
C16 LAUNCHED on box 3: warm start from c15 gen_0055, 40% EngineBotV3
scripted training opponent, 100% random kingdoms, no curriculum/league/
deep-slice, engine3 sentinel (200g) + BM sentinel (100g) every eval.
Banner verified (1,515,302 params, obs 1717), 1 trainer + 8 workers,
GPU 90%, box build rebuilt with engine3 selfplay C++ before launch.
The experiment: does direct engine pressure convert vloss gains into
honest strength where pure self-play stopped doing so.

SCORE-SNAPSHOT BIAS: FIELD CONFIRMATION (2026-07-14, Jack vs c15
gen_0045, exports/hjtaHmmk80wVPdN2.json, 35-25 Jack): trailing 5-12
through 7-18 mid-game while being cursed, the bot bought Duchy/Estate/
Duchy/Duchy on turns 19-27 instead of Sentry (in kingdom, never bought,
would have trashed curses). Early game showed good instincts (t5 Witch
retaliation, t11 Province while ahead 8-1). Exactly matches the gen-30
probe: value head prices current differential as realized outcome.
Fix ladder: (1) c16 may organically punish panic-greening via v3
losses; (2) c17 lever: tempered margin targets (blend win/loss,
early-position margin discount). Jack's human record remains unbeaten.

FIRST BOT WIN OVER JACK (2026-07-14, exports/T004mAgjh84I6YKt.json):
c15 gen_0045 @400 sims def. Jack 41-40 (Jack notes he played sloppily;
asterisk stands, milestone stands). Bandit/Moat board: bot showed
in-kind Bandit retaliation (6 plays) + double Moat defense, then won
the endgame with a SIX-DUCHY close — the score-snapshot instinct
applied at the correct time (tight race, late game) rather than
mid-game panic. Sharpens the bias diagnosis: timing miscalibration,
not a wrong instinct. Human record now 1 loss; first machine win in
project history.

C16 VERDICT + C17 LAUNCHED (2026-07-15): c16 engine3 sentinel through 25
gens: 50.5/53.5/53.5/50.0/50.0/55.5 — statistically flat at ~52, ending
where c15 rested. CONCLUSION: direct engine pressure (40% v3 opponent)
does NOT convert value accuracy into honest strength; the value-target
calibration is the last suspect standing. c16 cut at gen 30 (gens 25/29/
30 banked home). C17 LAUNCHED: identical recipe + value_target
"margin_blend" alpha 0.6 (wins train toward [0.8,1.0]; margin influence
compressed 2.5x; formula and tests in Codex delegation, 137/137 ctest).
Warm start c16 gen_0030. One variable changed; instruments for the
hypothesis: Duchy-injection probe delta, 1600-vs-400 duel gap, engine3
sentinel slope. NOTE process slip: c16 watcher silently unarmed gens
15-25 (claimed re-armed without calling Monitor) — caught at gen 29;
evals recovered from metrics.csv, nothing lost but the lesson stands.

DEEP-SEARCH REGRESSION ROOT-CAUSED (2026-07-15, c17 gen 15 instruments):
(1) Duchy-injection probe: margin_blend ELIMINATED the score-snapshot
bias — P(Duchy) delta under +6 opp VP: +0.2pts (was +3.7 on old target),
value delta -0.001 (was -0.21). Cleanest cause->fix->verify loop of the
project. (2) BUT 1600-sim duel still regressed (37.0% vs ~49.5% @400) —
bias wasn't the search killer. (3) c_puct sweep @1600: 1.25 -> 37.0%,
2.0 -> 46.5%, 3.0 -> 34.0% — inverted-U; the regression is mostly
EXPLORATION MISCALIBRATION: fixed c_puct tuned at 400 sims over-exploits
at 1600. Fix delegated: AlphaZero visit-scaled PUCT schedule
(c_init + log((N+c_base+1)/c_base)), opt-in config, tunable c_puct_base.
Ops note: box-3 network now severs any SSH session >~2 min; all long
box jobs must run detached (setsid nohup + short-connection polling).

VISIT-SCALED PUCT: NEGATIVE RESULT (2026-07-15): AlphaZero schedule
(implemented+verified, 139/139 ctest; opt-in, default off) tested at
c17 gen-15 vs engine3: 1600 sims visit_scaled(base=1400, eff~2.0 at
root) = 34.2% — performs like fixed-1.25 (37.0), NOT like fixed-2.0
(46.5). Diagnosis: the formula scales by per-node visits, so interior
low-N nodes still explore at ~init; the sweep's gain came from raising
exploration UNIFORMLY — deep low-visit nodes are the over-trusting
ones. 400-sim control 52.6% (schedule near-neutral at 400, as
predicted). STANDING CONCLUSIONS: best deep config (fixed 2.0) only
reaches PARITY with 400 sims; operating point everywhere remains 400
sims / fixed 1.25; deep-slice stays parked; depth is a value-
discrimination research problem, not a tuning problem. Code kept as
config option. First gpt-5.6-sol delegation (terra at capacity) —
inherited a capacity-crashed partial diff, fixed real gaps, delivered
green.

BOX 3 DOWN (2026-07-15): network degraded all day (SSH sessions severed
after ~2 min), then full unreachability >1 hour on port 10913. c17 was
at ~gen 20 when contact was lost. EXPOSURE: minimal — banked locally
through c16 gen_0030 and c17 gen_0015; at most ~5 c17 gens (~1h) lost
if the disk wiped. Six-hour reconnect patrol armed. If the instance is
dead/reassigned, next steps need Jack's vast console: new port or new
instance; re-bootstrap is scripted (source tarball + config + gen_0015
push + launch17.sh).

PHASE CLOSED — BOX LAPSED (2026-07-15, Jack): box 3 ran out of vast
credit (explains the day-long network degradation and final outage).
Program paused by choice; no data of consequence lost (c17 gens 16-~20
only). FLAGSHIP OF RECORD: c15 gen_0045 — best single engine3 read
(62.8%; honest band ~56-57), and the checkpoint that took the FIRST
GAME off Jack (41-40). Local archive: c13 (12 ckpts incl gen_0065),
c14 (0001/0020/0025/0030/0035/0038 + replay 1.26M), c15
(0005-0055 eval gens), c16 (0025/0029/0030), c17 (0015). All code
merged on v2-phase1: card-token transformer + factory + compiled
bucketed inference, EngineBotV3 selfplay/eval/web integration,
margin_blend targets (bias-fix VERIFIED by probe), visit-scaled PUCT
(implemented, negative result, default off). Open threads for next
provision: c17 strength verdict (sentinel was flat through gen 15 but
buffer barely converted), depth-as-research-problem, obs-v3 backlog
(trash composition, log1p pass, property-informed embeddings).

C15 GEN-45 SIMS SWEEP (2026-07-20, overnight local CPU, 500 games per
point, random kingdoms, engine3): win%-excl-ties by sims —
200: 51.9 | 400: 56.1 | 600: 53.5 | 800: 55.6 | 1000: 52.3 |
1200: 47.5 | 1400: 41.7 | 1600: 39.7. One sigma ≈ ±2.2. READ: flat
plateau 200-1000 (~52-56), then a smooth monotonic decline setting in
past 1000 — not a cliff. 1600 read (39.7) replicates the training-box
depth check (42.4), on 500 games. 400 sims confirmed as the operating
point (web deploy default). Companion probes (2026-07-19, n=100):
policy-only (sims=1) scored 4.0% vs engine3 — nearly all playing
strength lives in search; policy distillation gap is a c18+ lever.
Pile-ending rate stable ~22% across the whole sweep. Context: this is
the deployed web checkpoint (Hetzner box, 400 sims).

C15 GEN-45 vs REAL HUMANS ON DOMINION.GAMES (2026-07-24/25, first live
test): **44W-64L-4T over 112 completed games** — 39.3% raw win rate,
41.1% counting ties as half. Base-set only, unrated automatch, 400
sims / 2 determinizations (the deployed operating point), 86 distinct
opponents, ~14 hours of unattended play. This is the first read of any
checkpoint against live human opposition rather than engine3 or Jack.

READ: the flagship checkpoint is competitive but below average against
the site's unrated automatch pool. Two caveats before treating 39% as
the number: (a) opponent strength is unmeasured — no ratings were
collected, and the unrated pool is not a fixed reference like engine3;
(b) 61 of 112 games (54%) were decided by 6 VP or less, so the true
gap is narrower than the win column suggests. 20 opponents played us
more than once.

Bot losses are NOT driver artifacts: every game in the record ran to a
server-reported result, and the ~10 live failures found during the run
were all integration bugs (click/protocol/lobby), fixed and pinned with
regression tests before the long clean stretches. See
`docs/arena-progress.md` for the failure classes and `docs/arena-usage.md`
to reproduce. Full per-game archives (raw feed, every decision, VP
scores) are under `exports/arena/`, plus unified analysis records via
`python -m src.v2.records.convert` — enough to rebuild any decision and
re-search it at higher sims, which is the obvious next diagnostic for
where the 400-sim policy actually goes wrong against humans.

C18 CAMPAIGN (2026-07-25/26): from-scratch CardTokenNet on obs-v3
(1788: v2 + global trash section + select-semantic one-hot + tokenizer
source-card embedding), archetype-seeded self-play (6 opening templates,
root-prior lambda-mix annealed 0.6->0, trash-selection guard), league of
v2 ancestors served via exact v3->v2 obs downgrade, NO scripted opponents
in the training pool (Phase T2 policy). New vast 5090 box (192 cores),
16 workers, ~2,300 games/hr mature, 25 gens in ~14h wall incl. one
container-restart resume (gen 4 replayed; zero data loss). ~$7 spent.

RESULTS — fastest strength ramp in project history:
- engine3 eval: 1.0 / 14.9 / 25.5 / 36.7 / 45.7 / 45.4 at g1/5/10/15/20/25.
  Matched the entire MLP era (34-37) by g15 and c15's g19 read (~48) by g20.
- thinner sentinel (NEW, Chapel-engine exploiter script; weak lower bound
  — flagship c15 beats it 73-74%): 8.5 -> 58.5% by g20. The thin-engine
  axis climbed fastest of all.
- bigmoney: 7 -> 68%. vloss 0.349 -> 0.031 (margin_blend scale).
- CHAMPION DUELS (new duel.py, clean 400-sim seat-swapped 200g vs c15
  gen_0045 via obs downgrade): 24.7 (g5) / 27.1 (g10) / 39.0 (g15) /
  39.3 (g20) / 40.2 (g25). Surge g10->15, then FLAT ~39-40 for ten gens
  while sentinels kept climbing.

CUT at gen 25 per pre-registered rule (duel < 45%). c18 FLAGSHIP:
gen_0025 (best duel + bigmoney + lowest vloss; g20 statistically tied).

ROOT CAUSE OF THE CAP — CHAPEL PROBE (policy-only, 240 fixed turn-1..4
buy nodes on Chapel kingdoms, attack vs no-attack split, gens 5-25 +
champ): P(buy Chapel) unforced declines MONOTONICALLY 0.117 -> 0.070
(no-attack) and 0.093 -> 0.056 (attack) as lambda anneals, with NO
attack-conditional selectivity, converging from above toward the champ's
flat 0.036. Verdict: AVOIDANCE DRIFT — as forcing anneals out, the
policy prior regresses toward the money basin and the net converges to
the champ's own archetype (explaining the ~40% mirror cap). The
templates create winning trashing trajectories (t1/t4 beat unconstrained
seats through g20) but the 1.5M net does not RETAIN the archetype
unforced. Buffer trash rate decayed 4.05 -> 2.21 in step with lambda.

STANDING CONCLUSIONS: (1) the exploration package works as data
machinery — repertoire enters the buffer and converts to real strength
(g10->15 champ surge came exactly as template data compounded); (2)
anneal-to-zero is WRONG — retention needs either a lambda floor or more
capacity (or both); (3) obs-v3 + from-scratch transformer reaches
near-champ strength in 25 gens for ~$7 — iteration is cheap now.
C19 (scale d192/3L -> d320/5L/8H ~6M params) should also FLOOR the
anneal (proposal: lambda_final 0.15, p_unconstrained_final 0.7) so the
scale read isn't confounded by the known-broken schedule; the probe
(chapel_probe.py pattern, recreate from log) is the retention instrument.

SERVING STACK SHAKEOUT (2026-07-26/27, overnight): the shared inference
server (one GPU process serving 128 CPU-only workers via SHM rings)
went from design to validated production through five distinct failure
modes, each caught by live telemetry and fixed:
(1) NO COALESCING: first build fired per-request batches (mean 36,
    15K evals/s) — slower end-to-end than eager. Fix: bounded drain
    cycles (target rows / deadline / all-workers-pending).
(2) SERIALIZED CYCLE: coalescing knobs alone still ran every resident
    model's forward every cycle (~77ms cycle, waits p50==p99). Fix:
    per-model firing + drain-during-GPU-flight + contiguous scatter.
(3) MID-SERVE COMPILE STALLS: merged batches above the largest warmed
    bucket ran at exact size and triggered 30s+ torch.compile stalls
    (worker timeouts). Fix: cap merges at the largest compiled bucket.
(4) PRIORITY STARVATION: at saturation, continuously-eligible model 0
    starved league models; their host workers timed out. Fix:
    deadline-exceeded models preempt, oldest-first.
(5) n_games/RING MISMATCH: campaign config n_games=128 vs 64-row SHM
    request ring split every collect into two serialized round-trips —
    the "campaign slow, probe fast" mystery. Fix: n_games=64.
SELF-INFLICTED DETOURS, for the record: blanket OMP_NUM_THREADS=1
(fixing a real 9K-thread explosion) single-threaded the server's CPU
work and halved it — reverted entirely, thread explosion accepted as
benign; 337 zombie workers accumulated because stop-kills only swept
trainer+GPU pids while server-mode workers are CPU-only — new fleets
fought zombie herds (startup timeout crashes) until a full sweep.
VALIDATED END STATE: 709 games/hr blind-weights (vs 122 eager, 5.8x),
~930-1025 games/hr by gen 5-8, 55K evals/s aggregate, batches ~160-250,
zero timeouts. Duels run concurrently with training at ~400 games/hr.
OPS RULES ADDED: check cgroup cpu.max not nproc (one box advertised 192
cores, quota 46); kill by process-tree walk, never trainer+GPU pids;
never pattern-match processes in the same command that kills (killed own
ssh session twice); baseline log-grep counts on append-mode logs;
progress.json + heartbeat telemetry now mandatory on all long jobs.

C19 CAMPAIGN COMPLETE (2026-07-27/28, cut at gen 45 per rule): 4x scale
(CardTokenNet d320/5L/8H, 6.63M params) from scratch on obs-v3, anneal
FLOORS (lambda ->0.15, p_unc ->0.7), league of c18+v2 ancestors via obs
downgrade, no scripted training opponents. New 246-core vast box, 128
workers, shared inference server: 708 games/hr gen 1 -> 1,700+ mature;
45 generations in ~34h wall, ~$45 total incl. probes/duels.

RESULTS — every scripted/neural instrument at or past champ level:
- engine3: 21% (g1) -> 55.3% (g40, 400-game benchmark; champ's own
  definitive read 56.1 — statistical tie). bigmoney 83/100 (record).
  thinner 8.5 -> 71%.
- CHAMPION DUELS vs c15 gen_0045 (200g, 400 sims, seat-swapped):
  22.3 / 27.6 / 43.6 / 39.4 / 42.1 / 44.3 / 42.8 / 47.7 / 44.3 at
  g5..g45. Peak 47.7 (g40) — the closest any artifact has come to the
  champ; NO CROSSING. Seat structure stable: ~50-58% going first,
  ~29-38% going second — the entire remaining gap is second-seat play,
  uniform across game lengths (per-game telemetry refuted the
  short-wins/long-losses hypothesis; front-loading was seat blocking).
- League proxy runs 0-7pts HOT vs clean duels at later gens (mutual
  noise compresses skill gaps — randomness favors the underdog).

VERDICT (Jack, from arena/live play, confirmed by telemetry):
c19 converged to THE SAME ARCHETYPE AS THE CHAMP — money+attacks, near
zero unforced trasher/engine buys (Chapel ~0.13/game unconstrained,
over-buys attacks). Scale bought a stronger money bot, not a different
player. The anneal floor did not prevent repertoire regression; forced
openings produced games the net learned to BEAT, not to ADOPT (template
seats vs unconstrained hovered 45-55% — neutral gradient). Diagnosis
candidates for restructuring: (1) piloting chicken-and-egg (engines
lose under mediocre self-piloting, so the data honestly teaches money);
(2) kingdom-averaging (money decent everywhere beats engines-great-
somewhere for an unconditional prior); (3) attack signal density
(immediate credit vs 10-turn engine payoff horizon).

FLAGSHIP: campaign19/gen_0040 (47.7 champ duel, 55.3 engine3-400g).
All 45 checkpoints + replay + metrics banked. NEXT (pre-c20 audit,
convened early): value-head counterfactual probe (built engine deck vs
money deck — locates the failure in value/policy/data), then
restructuring per findings: curated engine-kingdom curriculum phases,
full-game guidance, and human-game imitation (the only source of
competent engine piloting). Serving stack + duel/benchmark/telemetry
harnesses carry forward unchanged.

VALUE-HEAD ENGINE PROBE (2026-07-28, local, post-c19): counterfactual
obs-editing on 14 real mid-game buy states (turn>=10, engine-dominant
kingdom: Village/Smithy/Lab/Market/Chapel/Festival/CR/Moat/Militia/
Witch). Own deck+discard composition swapped between matched variants
(equal VP): MONEY 7C/3S/2G/3E, ENGINE 3C/2S/2Vil/2Smi/2Lab/1Mkt/1Cha/3E
(thinned, built), JUNK 9C/1S/4Curse/3E control. Value head means:
  c15 gen_0045:  money +0.897 | engine +0.717 | junk -0.032
  c18 gen_0025:  money +0.819 | engine -0.686 | junk -0.948
  c19 g5/25/40:  money ~-0.86 | engine ~-0.98 | junk ~-1.00
FINDINGS: (1) EVERY net in the lineage prices a fully-built engine deck
BELOW a plain money deck on an engine board — the value head is where
engine lines die: MCTS backups steer away from engine plans regardless
of policy exploration, so no policy-side curriculum (templates, floors)
could ever convert. (2) The bias is honest-in-distribution: under the
nets' own mediocre engine piloting, engines DO lose — the piloting
chicken-and-egg is encoded in the value function. (3) c18's anti-engine
bias is extreme (built engine at -0.69). (4) c19's values saturate
hard-negative on ALL variants of these off-distribution states (even
money ~-0.86 where c18 reads +0.82 on identical inputs) — its tightly
fit value head (vloss 0.021, entropy 1.20) is OOD-brittle, a candidate
mechanism for the second-seat gap and human-play failures. Junk control
sane everywhere. Probe: scratchpad value_probe.py pattern (obs-edit,
recreate from log). IMPLICATION (ranking only, spec deferred): fixes
must change the OUTCOME DATA the value head trains on — engine wins
must actually occur in training games (human-game imitation; curated
engine-kingdom phases); policy-only guidance cannot work. Input to the
pre-c20 architecture audit.

HONEST RE-BASELINE — FIRST DETERMINIZED-SEARCH READS (2026-07-28,
local, new harness src/v2/train/honest_eval.py): the pre-c20 audit
found ALL training self-play AND every EvalRunner/duel.py number in
this log runs PERFECT-INFORMATION search — the tree sees opponent
hands and, via each node's cloned RNG, the exact future shuffle/draw
outcomes (deliberate Phase-T.1 scaffolding, selfplay.cpp ~1320-1333,
eval_runner.cpp:854; determinize() is reached only by the web/arena
DecisionSearcher paths and the scaffold bot). First honest reads at
the deployed operating point (400 sims, K=2 per-decision
determinization, 200g each, seat-blocked, seed 20260728, 0 errors):
  c15 gen_0045 vs engine3:      44.9% (88-108-4)   [clairvoyant 56.1]
  c19 gen_0040 vs engine3:      47.7% (95-104-1)   [clairvoyant 55.3]
  c19 gen_0040 vs c15 gen_0045: 45.6% (88-105-7)   [clairvoyant peak 47.7]
FINDINGS: (1) CLAIRVOYANCE GAP ~8-11 pts vs engine3 (~3 sigma at
n=200): both flagships are BELOW engine3 parity in the regime they
actually play — finally consistent with the arena's 39% vs humans.
(2) The NN-vs-NN duel is nearly unchanged (both seats cheated equally
before): RELATIVE reads in this log survive; ABSOLUTE reads vs
scripted opponents do not. (3) THE SEAT STRUCTURE IS REGIME-INVARIANT
and not a duel artifact: first/second-seat = 52.5/37.1 (c15 vs
engine3), 58.6/37.0 (c19 vs engine3), 53.1/37.9 (duel) — the
second-seat collapse reproduces against a scripted opponent under
honest information. (4) Ending mix healthy (province-majority 155/157
of 200 vs engine3; duel 105/95), zero truncations, mean turns 46-51.
Harness: serving-identical per-decision DecisionSearcher, 10 CPU
workers, 911 / 274 / 187 games/hr (c15 / c19 / duel). Honest sims
curve queued (c15 at 200/800/1600 + c19 at 1600; 400-point reuses the
baseline run — same seed, same boards). Results JSONs under
bench/honest_eval/.

HONEST SIMS CURVE (2026-07-29 overnight, honest_eval.py, 200g/point,
K=2, seed 20260728 — same boards as the re-baseline): c15 gen_0045 vs
engine3 by sims — 200: 48.7 | 400: 44.9 | 800: 47.7 | 1600: 45.5.
c19 gen_0040 @1600: 47.0 (vs 47.7 @400). One sigma ~3.5, zero errors.
FINDINGS: (1) THE DEEP-SEARCH REGRESSION DOES NOT REPRODUCE UNDER
HONEST SEARCH. The clairvoyant sweep's collapse past 1000 sims (39.7
@1600, 2026-07-20, 500g/point) was a perfect-information-regime
artifact — deep search over-exploiting a KNOWN future through a
miscalibrated value head. Under per-decision K=2 determinization the
curve is FLAT 200->1600 for both flagships. The standing
"depth-as-research-problem" conclusion is REVISED: honest depth is
neutral, not harmful; visit-scaled PUCT / c_puct-at-depth work was
chasing a cheat-regime pathology. (2) Neutral is its own message: 8x
sims buys ZERO points — the value head, not search budget, is the
binding ceiling on honest strength (independent confirmation of the
value-probe finding, from the search side). (3) Operating point: 400
sims stands (cheapest defensible point on a flat curve; 200 read
highest at 48.7 but within noise — candidate cost lever for the c20
regime A/B). (4) Seat structure invariant at every depth (54/37 c15,
56/38 c19 @1600) — see re-baseline entry; net second-seat deficit vs
engine3's own mirror splits (~58/41-43) is ~4-6 pts, not the raw
13-pt read. c19@1600 costs 66 games/hr on 10 CPU workers.

RESTRUCTURE BUILD WAVES 1-2 LANDED (2026-07-29 night, all Codex
delegations reviewed + suites re-run locally, 169 C++ / 166-168 py
green): (1) determinized self-play mode (selfplay.determinize =
off|per_decision|per_turn; fresh sample of the true state per search;
per_turn seeds by turn for hop damping; tree reuse auto-disabled;
encode-equivalence invariant tested; default off = legacy bit-exact);
(2) honest modes for EvalRunner/evaluate.py/duel.py (--honest;
default clairvoyant until c20); (3) TRUE legal-mask recording (the
policy>0 reconstruction bug is dead: legal-but-unvisited actions now
reach the CE denominator; crafted-case test shows loss 0 -> log 2);
(4) landscape-sentinel memset repair (see COMPAT WARNING below;
golden replay hashes legitimately regenerated — action sequences
byte-identical, hash-line-only diffs); (5) AdamW default (config
optimizer field, legacy adam for resumes, mismatch guard); (6) corpus
quarantine (109 real local + 2 hetzner games; 123 pytest artifacts +
92 stubs + 9 corrupt excluded; converter defaults clean; server tests
now export to tmp) + human-tuple exporter (12,734 human-seat tuples
with raw margins, exports/tuples/); (7) imitation stack in train.py
(BC pretrain, persistent anchor loss with schedule floor, optional
AWR, offline_fit --human-tuples/--human-only; anchor-off path
literally separate = legacy identical).

ENCODER COMPAT WARNING (2026-07-29, consequence of the sentinel
repair): pre-fix checkpoints (c13..c19, incl. the DEPLOYED c15
flagship) trained on ~34 constant-1.0 landscape/trait floats that now
correctly encode 0.0. Serving/probing them on post-fix builds is
OFF-DISTRIBUTION: c15's value probe collapses +0.897/+0.717/-0.032 ->
+0.076/-0.049/-0.378 (ordering survives, calibration destroyed). DO
NOT redeploy legacy checkpoints on post-fix engine builds (current
Hetzner image is safe — old build baked in). Legacy probe/benchmark
reference values are pre-fix-build-only. Guard task opened
(encoder-generation tag + legacy shim); c20 starts a fresh reference
series on the fixed encoder.

VALUE-TARGET GEOMETRY SWEEP (2026-07-29/30 overnight, task #15,
bench/value_target_sweep/): margin_blend alpha in {0.6 ctrl, 0.4,
0.2, 0.0}, c19 replay 250K slice (values EXACTLY inverted to raw
margins — the alpha=0.6 labels are bijective in-range), 13%
human-tuple mixing at matching alpha, 3 seeds x 3000 steps per arm,
identical init/data across arms, d192 CardTokenNet, probes via the
legacy shim (fits trained on pre-fix-encoder obs). Aggregate
(mean±sd over seeds; money/engine/junk = value-probe means):
  a=0.6: money +0.25±0.65  engine +0.53±0.31  junk -0.61±0.51  duchyDV -0.113±0.069
  a=0.4: money -0.31±0.79  engine +0.02±0.71  junk -0.91±0.05  duchyDV -0.022±0.007
  a=0.2: money +0.26±0.87  engine +0.17±0.82  junk -0.36±0.81  duchyDV -0.059±0.032
  a=0.0: money +0.87±0.16  engine +0.89±0.09  junk -0.26±0.75  duchyDV -0.057±0.020
READ: (1) PURE MARGIN (a=0.0) is uniquely SEED-STABLE and prices the
built engine deck AT PARITY with money on the engine board — the
exact property the restructure chases — replicated across seeds;
every alpha>=0.2 arm is seed-chaotic on these OOD-ish states (sd up
to ±0.87). Mid-range supervision stabilizes OOD behavior, consistent
with the saturation-cliff hypothesis. (2) The a=0.6 control shows the
WORST Duchy value-distortion at this scale, inverting the full-scale
c17 rationale; the c17 Duchy fix must be re-verified at campaign
scale whatever alpha ships — the Duchy probe is a pre-registered
non-regression gate for c20's early gens. (3) Caveat: quarter-scale
3000-step fits; comparative geometry read only, not a strength claim.
RECOMMENDATION to carry into c20 assembly: alpha=0.0 (pure margin)
as the leading candidate, 0.2-0.4 as fallback if the Duchy gate
trips. Pilot (single-seed, no humans, pre-shim) archived in
sweep_results.json; replication in sweep2_aggregate.json.

HONEST-GAP ATTRIBUTION REVISED (2026-07-29 night, matched-board
discriminators after the #9 honest-eval-mode build): (1) EvalRunner
matched A/B, same seed/scheduler, 200g each: clairvoyant 54.0 vs
honest-per-decision 54.8 — CLAIRVOYANCE IS WORTH ~ZERO vs engine3
(the divergence tests confirm hidden zones really are resampled; the
information just doesn't convert to strength against a chart bot at
400 sims). (2) Harness K=1 vs K=2 on the SAME boards (seed 20260728):
48.5 vs 44.9 — K=2 world-splitting costs ~3.6 pts (~1 sigma, mild).
(3) Residual ~6 pts between harness-K1 (48.5) and EvalRunner-honest
(54.8) is a STACK effect, not information — different samplers/seeds
so not yet matched-solid; prime suspect is DecisionSearcher's
within-decision leaf batching / virtual-loss waves (EvalRunner fills
batches across 128 games, ~few leaves per tree per pass; the serving
searcher must extract whole batches from one decision). CORRECTION to
the 2026-07-28 re-baseline entry: those numbers stand as
DEPLOYED-CONFIG reads, but attributing the eval-vs-serving gap to
perfect information was wrong — it is mostly serving-stack search
quality + K-splitting. Two consequences: (a) the deployed bot may be
leaving ~6-10 pts on the table at serve time (new investigation task:
serving-stack search parity; K=1 deploy candidate); (b) the c19-era
deep-search-regression story stays revised as before (EvalRunner-
clairvoyant-only pathology). c20 bars restated: ~45-48 (serving
harness, deployed config) AND ~54 (EvalRunner honest) — track both.

PROBE SUITE CODIFIED + MILITIA FINDING (2026-07-29, scripts/probes/):
the five standing behavior probes are now permanent instruments
(value_probe, chapel_probe, militia_probe, duchy_probe,
human_record_probe + run_all.py; one JSON + scorecard per checkpoint).
Validation against banked checkpoints: value probe reproduces the
2026-07-28 logged means EXACTLY (c15 +0.897/+0.717/-0.032; c18
+0.819/-0.686/-0.948; c19 -0.861/-0.976/-0.995); chapel probe matches
(champ flat 0.036). NEW FINDING: the MILITIA KEEP-INVERSION PERSISTS
IN c18 AND c19 (raw policy AND 400-sim search on the rebuilt hands) —
the obs-v3 select-semantic one-hot gave the net the INPUT to
distinguish keep-vs-discard but no training pressure ever taught it
to use the flag; self-play mirrors still never punish junk-keeps.
Same lesson as everything else this phase: representation without
outcome data does not convert. Human-record probe v1 baselines
(policy mass on winning humans' action buys / money-default rate):
c15 .110/53.5%, c18 .105/49.6%, c19 .122/46.8% — i.e. on decisions
where a winning human bought an action card, the nets' argmax buy is
still a treasure roughly half the time. Duchy-injection reference on
the NEW fixed states (longitudinal baseline going forward): c15
+0.89pts, c18 +2.42pts, c19 +0.40pts.
