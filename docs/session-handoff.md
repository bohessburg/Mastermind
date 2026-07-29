# Session Handoff — Training Program State

Snapshot: 2026-07-28. Read with `docs/training-log.md` (experiment history —
the c18/c19/serving-shakeout/value-probe entries at the bottom are the
current state), `IMPLEMENTATION_PLAN.md` (Phase T2 roadmap + pre-c20 gate),
`docs/human-games-analysis.md`, `docs/bot_strategies.md`,
`docs/web-ui-handoff.md` (web UI is a separate session's domain).

## THE CENTRAL FINDING (2026-07-28) — read this first

The value-head engine probe (training-log tail) closed the question the
whole program has circled since the human-games analysis: **every network
in the lineage — c15 champ, c18, c19 — prices a fully-built, thinned
engine deck BELOW a plain money deck on an engine-dominant board.** The
value head is where engine lines die: MCTS backups steer away from engine
plans no matter what the policy explores, so NO policy-side curriculum
(forced openings, anneal floors, obs enrichment, 4x scale — all tried,
c18+c19) can convert. The bias is honest-in-distribution: under the nets'
own mediocre engine piloting, engines DO lose in self-play — the
chicken-and-egg is encoded in the value function.

**Implication (ranking established, SPEC DELIBERATELY NOT YET DRAFTED —
Jack wants the restructuring design done fresh):** fixes must change the
OUTCOME DATA the value head trains on — engine wins must actually happen
in training games. Ranked candidates: (1) human-game imitation (the only
existing source of competent engine piloting; 113 competitive records in
`exports/records/` + the arena can harvest more); (2) curated
engine-kingdom curriculum phases (boards where even mediocre engine play
beats money); (3) full-game guidance as supplement only. Secondary
finding feeding the pre-c20 audit: c19's tightly-fit value head
(vloss 0.021) saturates hard-negative on off-distribution states —
OOD-brittleness is a candidate mechanism for its second-seat gap and
human-play failures.

## What is running RIGHT NOW

**Nothing.** c19 was cut at gen 45 (2026-07-28, pre-registered rule) and
the vast box destroyed. All artifacts banked locally. No GPU exists; the
next box gets provisioned when the restructuring design is ready.

**Serving:** flagship deployment on the Hetzner VPS (`deploy/` stack,
400 sims) — still c15 gen_0045 unless Jack switched it.

## Strength ladder (all clean 400-sim reads)

- **c15 gen_0045 remains the strongest artifact** (Jack's correction on
  record: strongest AND cheapest). engine3 56.1% (definitive sweep).
- **c19 gen_0040 is the strongest challenger ever**: 47.7% vs the champ
  head-to-head (peak; final band 42-48), engine3 55.3% (400g — statistical
  tie with champ), bigmoney 81-83%, thinner 71%. BUT Jack's live-play
  verdict, telemetry-confirmed: c19 converged to the champ's own
  archetype — money+attacks, ~0.13 unforced Chapel buys/game, over-buys
  attacks. A bigger champ, not a different player. Humans still beat it.
- Champ-duel seat structure (stable across 5 milestones): ~50-58% going
  FIRST, ~29-38% going SECOND — the whole gap is second-seat play,
  uniform across game lengths (per-game telemetry refuted the
  short-wins hypothesis).

## Campaign history this phase (details in training-log)

- **c18** (obs-v3 + forced openings, 1.5M): cut g25; avoidance drift —
  policy regressed to money as lambda annealed to zero.
- **c19** (4x scale d320/5L/8H + anneal floors 0.15/0.7): cut g45; same
  archetype convergence despite floors. 45 gens, ~$45, ~34h. Proved:
  capacity is NOT the constraint; serving stack does 1,700+ games/hr.
- League-proxy calibration: runs 0-7pts HOT vs clean duels at later gens
  (mutual selfplay noise compresses skill gaps — favors the underdog).
  Never quote league lines as duel forecasts.

## Next moves (in order)

1. **Restructuring design session** (Jack-led, not yet started): turn the
   value-head finding into a c20 recipe. Do NOT draft ahead of him.
2. **Pre-c20 gate (Phase T2, mandatory):** architecture/topology audit +
   literature review (set transformers, pointer nets, OpenAI Five/
   AlphaStar entity encoders, published deckbuilder AI). The probe
   findings and seat-gap data are its inputs.
3. Arena/human data: harvesting more human games is now doubly valuable
   (eval set AND imitation source). Militia arena bug was fixed by the
   other session; arena runs are unblocked.

## Local artifacts (all verified)

- `checkpoints/remote/campaign19/` — ALL 45 gens + metrics.csv +
  replay_state.npz + console19.log. Flagship gen_0040.
- `checkpoints/remote/campaign18/` — all 25 gens + metrics + replay.
- Earlier campaigns unchanged (c13/c14/c15/c16/c17 as before; c14
  replay_state.npz = 1.26M obs-v2 offline set).
- Configs: run_c18.json, run_c19.json (the floors recipe), probe scripts
  recreatable from training-log entries (chapel_probe, value_probe).

## Code facts that carry forward (all committed on v2-phase1)

- **Shared inference server** (`src/v2/train/inference_server.py`):
  `server_selfplay: true` routes all workers through one GPU process —
  coalesced cross-worker batches, per-model firing with deadline
  preemption, merge-cap at largest compiled bucket, obs-v3→v2 downgrade
  for league ancestors. 5.8x over eager. Worker count sized to the REAL
  cgroup quota (`cat /sys/fs/cgroup/cpu.max` or cpu.cfs_quota_us — NEVER
  nproc; one box advertised 192 cores with a 46-core quota).
- **duel.py**: clean NN-vs-NN checkpoint duels (400 sims, no noise,
  seat-BLOCKED halves — first 100 = A first player), cross-obs via
  downgrade, per-game telemetry lines `pg,index,a_seat,turns,result`.
  ~2x slower than evaluate.py by structure (both seats search) + config
  (n_games 64/max_batch 512 — bumpable to 128/2048 for ~2x, untested).
- **Progress telemetry**: trainer writes `<ckpt_dir>/progress.json`
  (~30s atomic) + 60s console heartbeats; duel/evaluate print periodic
  progress; server logs batching stats. PYTHONUNBUFFERED=1 always.
- CardTokenNet scales cleanly (d320/5L/8H = 6.63M proven); obs-v3 (1788)
  = v2 + trash section + select-semantic one-hot + source-card embedding.
- margin_blend targets, visit-scaled PUCT (negative, off), thinner
  sentinel (weak lower bound: champ beats it 73%) unchanged.

## Ops lessons (updated the hard way, 2026-07-26/28)

- **Stop = kill the whole tree**: trainer + spawn_main workers + GPU
  pids; server-mode workers are CPU-only and INVISIBLE to nvidia-smi
  (337 zombies once accumulated). Verify zero fleet procs + clean
  /dev/shm (kill -9 leaks segments AND sem.mp-* semaphores) before
  any relaunch.
- NEVER pattern-match processes in the same command that kills — killed
  our own ssh session twice. Collect PIDs first, kill by explicit list.
- Baseline log-grep counts on append-mode logs before alerting on them.
- Launch detached (setsid nohup, verify via second ssh); rsync
  --partial, never scp; md5sum checkpoints; bank at every milestone.
- Watchers exit-per-event and MUST be re-armed; notification queues can
  delay — milestone routines must re-check the generation counter, not
  trust firing order.
- ALL user-facing reporting goes in the FINAL message of a turn (mid-turn
  text is lost); interventions on running campaigns require the full
  incident report (problem/evidence/change/cost) in that final message.

## Eval discipline

- Honest strength = clean duels (duel.py) and evaluate.py at 400 sims.
  n=200 sigma ≈ ±3.5. League lines are directional only (hot bias).
- Counterfactual obs-editing probes remain the behavior-question tool:
  established results now include the engine value probe (see top),
  Chapel retention/avoidance, tit-for-tat, score-snapshot bias.
- Human play is ground truth. Telemetry counters can dress up "basically
  never" as a number — read them per-game (0.13 Chapels/game), not
  per-100-seat-games.

## Playable seats (web UI)

Unchanged: bot:nn, bot:nnmcts (NN_MCTS_SIMS, default 400), bot:engine3 /
bot:engine2 / bot / bot:thinner. Any checkpoint via DOMINION_NN_CHECKPOINT
or bot:nnmcts:<path>; loader is factory-aware. To play the c19 flagship:
DOMINION_NN_CHECKPOINT=checkpoints/remote/campaign19/gen_0040.pt.
