# Session Handoff — Training Program State

Snapshot: 2026-07-09 late evening. Read with `docs/training-log.md` (experiment
history), `IMPLEMENTATION_PLAN.md` (Phase T), `docs/web-ui-handoff.md` (web UI
is a separate session's domain — do not touch src/v2/web from here).

## What is running RIGHT NOW

**Campaign 2** (gated training) on a rented vast.ai RTX 5090 box:
- SSH: `ssh -p 28890 root@69.176.92.135` (key already in ~/.ssh; account key
  registered on vast). Box costs $0.34/hr. 24-vCPU cgroup quota, 32GB disk.
- Repo at `~/dominion` on the box; venv python `/venv/main/bin/python`;
  bindings prebuilt at `~/dominion/build`.
- Run: `campaign2.json` (on box; local copy in scratchpad — regenerate from
  `src/v2/train/configs/run_gated.json` + campaign overrides if lost).
  100 gens × 1024 games, 128 sims, 8 CUDA workers, replay 2M, cosine LR,
  gate_warmup_generations=8, gate 60 games @64 sims threshold 0.55,
  league_fraction 0.2, eval 200 games vs EngineBot @400 sims every 5 gens.
- Monitor: `~/dominion/checkpoints/campaign2/metrics.csv` (label-parse it;
  columns include gate_result / gate_win_pct / best_generation / league_games).
  Liveness: `pgrep -f "python -m src.v2.train"` (expect ≥1; note pgrep matches
  its own ssh cmdline — pipe through `grep -v pgrep` or count >0 conservatively).
- Watcher pattern (local): background while-loop ssh-polling metrics for new
  eval rows or gate acceptances every 5-10 min; exits on new row → report to
  user → re-arm with a state file holding last reported generation.
  REPORTS MUST BE THE FINAL MESSAGE OF A TURN (mid-turn text may not render).
- Stop the run: `bash /root/stop_train.sh` (kills parent AND spawn_main
  workers — the parent-only kill orphaned 8x ~10GB workers per campaign cut
  until the container's 256GB cgroup OOM-killed a live worker on 2026-07-10;
  cgroup OOM kills are INVISIBLE in container dmesg and present as
  "self-play worker exited unexpectedly"). NEVER pkill a pattern that
  appears in your own ssh command line (self-kill; 4 occurrences now) —
  script files on the box only. Memory watch: /sys/fs/cgroup/memory/
  memory.usage_in_bytes vs memory.limit_in_bytes (~275GB); one run ≈
  90-100GB (parent w/ replay buffer + 8 workers ~10GB each).
- Launch pattern that works: `setsid nohup env OMP_NUM_THREADS=1
  PYTHONPATH=build /venv/main/bin/python -m src.v2.train.train --config X.json
  > log 2>&1 < /dev/null &` — the ssh session may hang after; that's cosmetic,
  verify via a second ssh (pgrep + nvidia-smi + console.log banner).

## Campaign 2 status at snapshot (gen 14)

Warmup gens 1-8 accepted unconditionally; gen 9-10 candidates gate-rejected
at 0.00; gen 12 accepted at 1.00 (best lineage 8→12). Gate scores are
near-binary (temp-0 determinism artifact between similar nets) — functional
but coarse. Eval 0/200 at gens 5/10 (matches campaign 1 pacing). League
plays ~205 games/gen. Throughput ~11-13K games/hr (gating costs ~2x vs
ungated 23K due to per-seat split — fix in flight, see Pending).

Decision rules agreed with Jack:
- Campaign 1 (ungated) peaked 30.3% at gen 25 then cycled down to ~17%.
  Campaign 2 exists to test whether gating holds the curve monotonic.
- Compare campaign 2's evals against campaign 1's at same generation
  (c1: 0/0/0/5.5%/15%/30.3% at gens 1-25). If c2 tracks or beats c1 through
  gen 25-35 WITHOUT the slide, gating works.
- Report eval results to Jack every 5 generations (he wants them promptly
  and IN CHAT, no push notifications).

## Pending / in flight

1. **Same-model gate fast path** (throughput fix, targets campaign 3): Codex
   task task-mreb8avi-4pcwdi has been running 1h44m — LIKELY STUCK; cancel
   via the codex-companion runtime and re-delegate (spec: when both seats
   share a model, evaluate leaves as one batch; bit-equality equivalence
   test mandatory; metrics fast_path_fraction). Check `git status` for
   partial writes first (it applied changes to src/v2/train/ mid-run —
   review or revert before re-delegating).
2. **Campaign 3 recipe** (agreed with Jack, not yet implemented): fast path
   + sampled-temperature gate matches (fix the 0-or-1 gate score artifact)
   + seed campaign-1 gen-25 checkpoint into the league pool as a standing
   external opponent (anti-cycling diversity without warm-start attribution
   mess). Possibly warm-start once optimizing for strength over science.
3. **Chunk W2 remainder**: infra tooling exists (scripts/infra/remote_run.py,
   Dockerfile.train) but the live provision→teardown E2E was never run
   (user manages the box manually; VAST_API_KEY never provided). Docker
   image never built (needs docker daemon). Optional completion.
4. **On-box bench of optimized inference server** (--bench-server, spin vs
   queue) — waiting for a campaign gap; targets bigger nets later.

## Key assets

- Campaign 1 artifacts: `checkpoints/remote/campaign1/` LOCAL (54 files,
  1.8GB); gen_0025.pt = 30.3% vs EngineBot checkpoint (verified loadable).
- Bot baselines (10K games, random kingdoms): Engine 74% vs BigMoney;
  scaffold-MCTS (EngineLike rollouts K=2) 65.8% vs Engine. Parity gate for
  Phase T = ≥50% vs EngineBot excl ties, ≥200 games, random kingdoms.
- Eval CLI: `python -m src.v2.train.evaluate --checkpoint X --opponent
  engine --games 200 --sims 400 [--ladder]` (PYTHONPATH=build).
- Codex delegation: gpt-5.6-terra @ xhigh (config in ~/.codex/config.toml,
  backup .bak has 5.5); shared session resumes with --resume-last; the
  rescue subagent only FORWARDS — orchestrator polls via
  `node ~/.claude/plugins/cache/openai-codex/codex/1.0.6/scripts/codex-companion.mjs status|result <task-id>`.

## Workflow reminders (from CLAUDE.md + learned this session)

- Delegate non-trivial implementation to Codex; review the ACTUAL diff;
  Codex sandbox has no network/GPU/docker/POSIX-shm — orchestrator runs
  those verifications.
- Commit style: <5 words, no body. Push after each reviewed chunk.
- Docs discipline: plan deviations → IMPLEMENTATION_PLAN.md notes;
  experiments → docs/training-log.md (append); benchmarks → docs/benchmarks.md.
- 2-player focus; 2nd-edition-removed cards out of scope.
- Engine is trusted ground truth (fuzz+goldens); training-layer bugs so far:
  cold-start gate deadlock (fixed: warmup), gated path bypassing worker pool
  (fixed), pkill self-match (procedural), ssh-hang-after-nohup (cosmetic).

## NN playtesting (added late 2026-07-09)

- Web server has a `bot:nn` seat (dropdown: "Human vs neural net"); loads
  checkpoints via DOMINION_NN_CHECKPOINT (default campaign1 gen_0025.pt) or
  `bot:nn:<path>`. Policy-head argmax, no search. Finished games auto-persist
  to `exports/<session_id>.json` (survives restarts; sessions themselves are
  in-memory only).
- Gen-25 policy characterization (offline probes + Jack's 2 games, one
  exported: exports dir + scratchpad nn_game2.json, seed 68969696969):
  * 3-4 coins → Silver at 78-90% (its one deep conviction).
  * **5 coins → Duchy ~80% ALWAYS** — turn 3 or turn 30, board-blind.
    Not reactive, not a clock: a constant learned from money-slog self-play.
  * ≤2-coin decisions are UNLEARNED (near-uniform: Pass≈Copper≈Moat≈Curse
    ~17-24%) → argmax noise buys: Coppers at 0 coins, Estates, late
    Workshops/Cellars at 15-21% confidence.
  * Value head is directionally well calibrated (tracked a lost game to
    -1.00 from turn 35) — policy learned unevenly, value learned well;
    explains why gen-25+MCTS hit 30.3% while raw policy lost 54-7 to Jack.
- Probe scripts pattern: encode obs via game.encode(player), masked softmax
  over policy logits, report top-k at chosen decision points (see scratchpad
  claudes_gambit.py / probe snippets in session history; worth promoting to
  a real tool if used again).
- Comparison idea for campaign 2 checkpoints: replay Jack's exported games,
  asking each checkpoint "what would you buy here" — qualitative diff
  between training runs.

## State update (2026-07-10 midday)

- RUNNING: **Campaign 13 clean run** (launch #4, 2026-07-12 ~03:30) —
  checkpoints/campaign13/, campaign13.json, /root/launch13.sh. FIRST
  SIGHTED NET: obs v2 (1717 floats, opponent collection/discard/set-aside)
  + input_scale 16 (conditioning fix — see 2026-07-12 postmortem) + margin
  targets + pile-aware opponents (EngineBot v2 baseline!) + async offload.
  Gen 5 = 55.1% vs EngineBot v2. Watcher pattern unchanged (state file
  scratchpad/c13_last_eval). OVERNIGHT CONTRACT: if stalled by gen 25,
  cut, bank best weights, design+launch c14. HYGIENE RULES (hard-won
  today): ALWAYS bash /root/stop_train.sh before ANY launch (double-run
  incident); verify singleton with ps + [.]-escaped pattern (pgrep
  self-match footgun); watcher state files must be reset when a campaign
  restarts from scratch; watchers die silently when the box has no
  training process during a stop window — re-check after every relaunch.
- HISTORICAL (was): **Campaign 10** (launched 2026-07-11 ~06:00) —
  checkpoints/campaign10/, campaign10.json, /root/launch10.sh. 512 sims,
  temp 20, BM anneal-down [[5,0],[6,.01],[25,.2],[40,.2],[60,.05]],
  engine [[10,0],[11,.01],[35,.25]]. Seed 20260718. Per-kind counters
  live (scripted_wins_bigmoney/engine columns; watcher prints both).
  Thesis: protect gen-30-style action exploration from money reconquest
  (see c9 closeout in training-log). RESUMED from gen 23 with the eval
  fix (131aca1) — all in-run evals before that are ~50x understated.
- **PARITY GATE CROSSED** (2026-07-11): eval had dropped the treasure-
  collapse flags since c7 — every eval ran checkpoints in the wrong
  decision space. Corrected: c8g35 80%, c9g30 56%, c9g60 87.7%, c10g20
  81% vs EngineBot. Provisional pending corrected scaffold + human
  playtest. Full story in training-log.
- PREVIOUS: **Campaign 9** (cut gen 89) —
  checkpoints/campaign9/, config campaign9.json, /root/launch9.sh. Dual
  ladder: treasure-collapse flags + BigMoney from gen 6 and ENGINE from
  gen 11, 1%/gen each to 20% caps. Seed 20260717. Campaigns 6 (curriculum
  collapse-adjacent stall), 7 (treasure collapse validated, 5% dose too
  early), 8 (stage-1 money play converged; ceiling mapped; gen-35 peak =
  stage-1 baseline, synced locally) all closed — see training-log for
  arcs. Probe playbook: argmax buy-preference probe at $5-$8 (fast,
  local), full-game buy log via DecisionSearcher, both in session history
  and cheap; run at every 5-gen eval past ~15.
- CLOSED: Campaign 5 cut at gen ~21 — pessimism collapse (value head
  pinned -0.9..-1.0, policy degenerated to Copper-only buys; 20% BM from
  cold start fed pure -1 into the value head). Full diagnosis in
  training-log. Also closed: campaigns 1-4, synced locally under
  checkpoints/remote/ (c1 peak 30.3%@25; c4 peak 35.2%@15, both slid).
  c4-gen15 vs scaffold-MCTS head-to-head: 0-60 — EngineBot win% is not
  transitive strength.
- STANDING INSTRUMENT: behavioral probe on synced checkpoints ~gen 15-20
  (argmax buy distribution + value-head min/mean/max vs BigMoney, control
  against c4-gen15) — eval columns alone hid the c5 collapse.
- ROOT-CAUSE FINDING: pile-out/money local optimum lives in the VALUE HEAD;
  search can't escape it (human beat c4-gen15-full-power easily). c5 tests
  scripted-opponent data as the cure.
- EVAL DISCIPLINE: EngineBot win% overstates strength (66-81% of games end
  by 3-pile vs its pile-blindness). Always read end_province/end_piles
  forensics columns; scaffold opponent (--opponent mcts) is the harder
  reference but ~50x slower (both sides search) — small game counts only.
- Box scripts: /root/box_ctl.sh (stop/patch/resume for campaign3-era paths
  — update paths before reuse), /root/launch5.sh pattern for launches.
  RESUME REQUIRES --config AND --resume together (bare --resume runs zero
  generations — see training-log postmortem).
- Watcher hygiene: one watcher per campaign with a CAMPAIGN-SPECIFIC state
  file (scratchpad/c5_last_eval etc.); stale watchers sharing state files
  have caused missed/duplicated reports twice.
- Playable seats: bot:nn (policy only), bot:nnmcts (full search;
  NN_MCTS_SIMS env, default 400). Any checkpoint via DOMINION_NN_CHECKPOINT
  or bot:nnmcts:<path>. Finished games persist to exports/<session>.json
  in the SERVER'S working directory.
