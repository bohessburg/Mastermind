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
cold start.
