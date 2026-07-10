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
