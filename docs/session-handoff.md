# Session Handoff — Training Program State

Snapshot: 2026-07-24. Read with `docs/training-log.md` (experiment history —
the c15/c16/c17 and phase-close entries at the bottom are the current state),
`IMPLEMENTATION_PLAN.md` (Phase T), `docs/bot_strategies.md` (scripted bots,
EngineBotV3), `docs/web-ui-handoff.md` (web UI is a separate session's domain).

## What is running RIGHT NOW

**No training.** The program is PAUSED: box 3 (vast.ai RTX 5090) ran out of
credit 2026-07-15 and is gone. Jack manages the vast console himself; we have
no VAST_API_KEY and cannot provision or destroy instances — wait for a box.

**Serving:** the flagship is deployed on a Hetzner VPS via the `deploy/`
stack (single Docker image: built Vite client + FastAPI from one origin,
optional Caddy TLS profile; see `deploy/README.md`). 400 sims. Local serving:
`DOMINION_NN_CHECKPOINT=checkpoints/remote/campaign15/gen_0045.pt` + uvicorn
(see `docs/play-the-net.md`).

## Flagship of record: c15 gen_0045 (card-token transformer)

Best engine3 read 62.8% (honest band ~56-57), and the first checkpoint ever
to take a game off Jack (41-40; his overall record vs the program is
otherwise unbeaten — human W-L is a tracked metric).

**Operating point everywhere: 400 sims, fixed c_puct 1.25.** Deep search
REGRESSES: the 2026-07-20 sims sweep (500 games/point vs engine3) shows a
flat plateau 200-1000 sims (~52-56%) then monotonic decline — 1600 sims
scores 39.7%. Do not raise sims to "play stronger." Policy-only (sims=1) is
4.0% vs engine3 — nearly all strength lives in search; the distillation gap
is a c18+ lever.

## Campaign results this phase (details in training-log)

- **c15** (card-token transformer, 1.5M params, mixed-arch league): the
  architecture thesis PROVED — −44% offline value MSE vs MLP, from-scratch
  engine3 curve 48.2% (g19) → 62.8% (g45) → 56-57 (g50/55, cut at 55). Every
  MLP-lineage flagship reads 34-37% on the same sentinel.
- **c16** (40% engine3 opponent mix, no league/curriculum, warm-start c15
  g45): sentinel FLAT ~50-55 through 25 gens. Engine pressure alone does not
  convert value accuracy into strength.
- **c17** (margin_blend value targets α=0.6, warm-start c16 g30): the
  score-snapshot bias is ELIMINATED (probe-verified: bot no longer panic-buys
  Duchies when behind), but the STRENGTH VERDICT IS INCOMPLETE — sentinel
  flat 48.5-51.5 through gen 15, interrupted ~gen 20 by the box lapse just
  as the replay buffer converted to majority-new-target data.

## Next moves (re-scoped 2026-07-25 after the human-games analysis)

**c17 is ABANDONED** (Jack's call, 2026-07-25): the human-games analysis
(`docs/human-games-analysis.md`) showed the binding constraint is plan-level
exploration / archetype monoculture, not value calibration — do not resume it.

The campaign roadmap c18–c21 is specced in `IMPLEMENTATION_PLAN.md`
(Phase T2). **c18 COMPLETE (2026-07-26, cut at gen 25 per pre-registered
rule):** from-scratch obs-v3 + forced openings reached 45% engine3 / 58%
thinner / 40% vs the c15 champ in 25 gens (~$7), then capped — the Chapel
probe showed AVOIDANCE DRIFT (unforced trasher-buying decays as lambda
anneals to zero; net converges to the champ's archetype). c18 flagship:
`checkpoints/remote/campaign18/gen_0025.pt`. Full entry at the tail of
`docs/training-log.md`. **Next: c19 = ~4x scale-up (d320/5L/8H) PLUS
anneal floor (lambda_final 0.15, p_unconstrained_final 0.7)** — the floor
is empirically required, not a discretionary second lever. Standing
policy: no scripted opponents in the training pool ever again — scripts
are sentinels/eval instruments only. New instruments that carry forward:
duel.py (clean NN-vs-NN champion duels, cross-obs via downgrade), the
thinner sentinel (weak lower bound), the Chapel retention probe.

Still true anytime: human-record games vs the flagship — the only benchmark
that has never saturated. Arena harness Militia discard bug is the top
pre-arena fix (25 of 64 arena losses; see human-games-analysis Finding 0).

## Local artifacts (all verified via torch.load)

- `checkpoints/remote/campaign13/` — 12 ckpts incl. gen_0065 (MLP-era
  flagship; 85.6% was vs the weak chart bot — 34.5% vs engine3).
- `checkpoints/remote/campaign14/` — 0001/0020/0025/0030/0035/0038 +
  metrics.csv + **replay_state.npz (1.26M obs-v2 positions — the offline
  validation set; c15 gate 1 ran on it)**.
- `checkpoints/remote/campaign15/` — eval gens 0005-0055 (~18 MB each).
- `checkpoints/remote/campaign16/` — 0025/0029/0030. `campaign17/` — 0015.
- Configs: `configs/run_c15.json` (full recipe), `run_c16.json`,
  `run_c17.json` (margin_blend). Exports of notable human games in
  `exports/` (T004… = first bot win vs Jack).

## Code facts that carry forward (all merged on v2-phase1)

- `build_model()` factory (`src/v2/train/model.py`): `model_config["arch"]`
  absent/"mlp" → DominionNet, "card_transformer" → CardTokenNet
  (`src/v2/train/card_transformer.py`). Checkpoints self-describe; mixed-arch
  league works (worker payload carries model_config).
- CardTokenNet: nn.Embedding(41,192) + 19 per-card scalars → 48 supply
  tokens + global token, 3 pre-LN layers, pointer policy head, tanh value
  off global. Zero C++ changes; tokenizer slices obs v2.
- Inference server: `server_compile` (torch.compile reduce-overhead,
  fullgraph), `server_autocast_bf16`, `server_batch_buckets` (static shapes).
  Compiled+bf16 = 368-439K evals/s on the 5090 vs ~70K eager.
- margin_blend value target (α=0.6, wins→[0.8,1.0]) — the c17 bias fix.
- Visit-scaled PUCT: implemented, NEGATIVE result, default off ("fixed").
- EngineBotV3 (`src/v2/bots/` — another session owns it; integrate, don't
  modify): selfplay opponent kind "engine3", eval sentinel columns
  sentinel_engine3_*, web seat "bot:engine3" ("Human vs EngineBot" mode).

## Ops lessons (expensive to relearn — next box will likely be vast again)

- NEVER pkill/pgrep a pattern present in your own ssh cmdline. Stop runs
  ONLY via `/root/stop_train.sh`; ALWAYS stop before any launch.
- Launch detached: `setsid nohup … >> console.log 2>&1 < /dev/null &`;
  verify via second ssh. Flaky links sever long sessions — poll with short
  connections; rsync -az --partial with retries, never scp (silent
  truncation); md5sum anything that matters.
- Box "death" can be a proxy outage: check `uptime` + trainer `ps etimes`
  before assuming a restart wipe. Real restarts wipe the container disk and
  reassign the ssh port — sync checkpoints home at every eval milestone.
- Fresh-box build: tarball includes tests/ (v2_fuzz sources), cmake
  -DBUILD_TESTS=OFF, pip pybind11 into /venv/main, target dominion_v2_py.
- Watcher pattern: local loop polling metrics.csv, campaign-specific state
  file, 3-consecutive-failure dead check; it EXITS after each report — must
  actually re-arm (verify the call, don't just claim it). Eval reports in
  chat as the FINAL message of a turn; never push notifications.

## Eval discipline

- The honest strength read is the **engine3 sentinel** (or evaluate.py
  --opponent engine3/engine2). The legacy "engine" chart bot is much weaker
  than the drivers bots — pre-reframe "~89% vs EngineBot" rows mean "vs
  chart money bot." Sentinel noise at n=200 is ~±3.5 for one sigma.
- Counterfactual obs-editing probes (policy forward passes on edited
  observations) beat metrics for behavior questions — established results:
  tit-for-tat attack retaliation (formed by gen 5), Moat-discount, and the
  score-snapshot bias before/after margin_blend.
- Human playtests are ground truth for repertoire gaps (gen 30 loss: never
  bought Sentry, panic-greened while behind — the probe-confirmed bias).

## Playable seats (web UI)

bot:nn (policy only), bot:nnmcts (search; NN_MCTS_SIMS env, default 400),
bot:engine3 / bot:engine2 / bot (scripted). Any checkpoint via
DOMINION_NN_CHECKPOINT or bot:nnmcts:<path>; the loader is factory-aware
(arch from checkpoint payload). DEFAULT_NN_CHECKPOINT currently
campaign13/gen_0065 — override to the flagship when serving. Finished games
persist to exports/<session>.json.
