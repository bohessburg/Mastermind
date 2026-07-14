# Session Handoff — Training Program State

Snapshot: 2026-07-13. Read with `docs/training-log.md` (experiment history —
the c14/c15 entries at the bottom are the current state), `IMPLEMENTATION_PLAN.md`
(Phase T), `docs/web-ui-handoff.md` (web UI is a separate session's domain).

## What is running RIGHT NOW

**Nothing.** Campaign 14 was cut at gen 38 (2026-07-13, Jack's call — metrics
plateaued, see training log). The box-3 vast.ai instance (ssh -p 10913
root@198.2.214.6, RTX 5090) is IDLE with everything of value pulled home;
all-clear to destroy. The prior box (port 10169, same IP) was also cleared for
destruction. Jack manages the vast console himself; we have no VAST_API_KEY
and cannot destroy instances — give explicit all-clears instead.

**Active work: c15 card-token transformer** (design approved by Jack; full
spec in training-log "C15 DESIGN" entry). Model-side only — C++ encoder and
obs v2 untouched. Gates before campaign launch: (1) offline fit on c14 replay
beats the MLP on held-out loss; (2) eval throughput ≥70% of MLP.

## Local artifacts (all verified via torch.load)

- `checkpoints/remote/campaign13/` — 12 full checkpoints incl. flagship
  gen_0065.pt (89.2% EngineBot / 71.0% BM, loaded in web UI); lean league
  seed set in scratchpad/c13_lean (with `generation` key patched in).
- `checkpoints/remote/campaign14/` — gen_0001/0020/0030/0035/0038.pt +
  metrics.csv + **replay_state.npz (1.26M obs-v2 positions — the offline
  validation set for c15)**. Flagship candidates: gen_0020 (89.7% eval,
  BM sentinel 72/100 best) and gen_0030 (89.7% post-migration recovery).
- Configs: `configs/run_c14.json` is the recipe baseline for c15 (margin
  targets, obs v2 + input_scale 16, 12-seed league, curriculum, tree reuse,
  top-k 8 root-exempt+wildcard, eval sentinels).

## Recipe facts that carry to c15

- Margin value targets: v = sign(margin) × (0.5 + 0.5·min(|margin|,20)/20).
- obs v2 (OBS_SIZE_V2=1717): perfect-memory opponent composition; layout in
  `src/v2/encode/encoder.h` + `src/v2/encode/layout.md`.
- input_scale=16 divisor in DominionNet — the hack the c15 tokenizer retires
  (raw counts diverge training at any lr; log1p/embeddings are the fix).
- Slot-manifest runner is mandatory for throughput (see training-log
  "THROUGHPUT SAGA": batch fragmentation costs 10x; 10K games/hr healthy).
- League loader rebuilds opponents from their own payload config → mixed
  architectures OK; lean seeds MUST carry `generation`; obs-version guards
  exist. league_opponents_per_gen=3 cap. Resume: `--config` + `--resume`
  together, and pop `init_weights` from the config (mutual exclusion).

## Ops lessons (expensive to relearn)

- NEVER pkill/pgrep a pattern present in your own ssh cmdline (6+ self-kill
  incidents). Script files on the box only; verify with full `ps` listings
  and `[.]`-escaped patterns. `pgrep | head` grabs stale PIDs — wrong etimes
  readings misled status twice.
- Stop runs ONLY via `/root/stop_train.sh` (kills parent AND spawn_main
  workers; parent-only kills orphan ~10GB workers until invisible cgroup
  OOM kills). ALWAYS stop before any launch (double-trainer incident wrote
  garbage metrics rows).
- Launch pattern: `setsid nohup env OMP_NUM_THREADS=1 PYTHONPATH=build
  /venv/main/bin/python -m src.v2.train.train --config X.json >> console.log
  2>&1 < /dev/null &`; verify via second ssh (ps + nvidia-smi + banner).
- File transfer on vast links: rsync -az --partial with retry loop. scp
  silently truncates (three corrupted checkpoints). md5sum both ends for
  anything that matters.
- Vast instances can spontaneously restart: container disk WIPED, ssh port
  reassigned. Sync checkpoints home at milestones; keep a one-command
  re-bootstrap script; abandon the host if it repeats. Replay buffers are
  usually not worth migrating (cost ≈ 1-2 gens of freshness, see c14 gen 21).
- Fresh-box build: tarball must include tests/ (v2_fuzz sources); cmake with
  -DBUILD_TESTS=OFF (Catch2 fetch blocked); pip install pybind11 in
  /venv/main; build target dominion_v2_py -j32.
- Watcher pattern: local background loop ssh-polling metrics.csv for new
  eval rows; campaign-specific state file (scratchpad/c14_last_eval);
  exits after reporting → re-arm. REPORTS MUST BE THE FINAL MESSAGE OF A
  TURN. Dead-trainer check needs 3 consecutive failures (connection flakes).
  Jack wants eval reports in chat, never push notifications.

## Eval discipline

- EngineBot v2 win% saturates ~85-92%; read end_province/end_piles forensics
  and the BM sentinel columns (sentinel_bigmoney_wins/games). Sentinel noise
  at n=100 is ±10 for one sigma — don't overreact to single evals.
- Behavioral probes beat metrics for repertoire questions: policy buy
  distributions by coin level on an engine-forcing kingdom (see training-log
  "C14 REPERTOIRE PROBE" and the probe script pattern — argmax-greedy self
  vs scripted BM, softmax mass per card at each price point).
- Human playtests are the ground truth for repertoire gaps (c13-gen65 lost
  to Jack twice: money-only play, no attack response).

## Playable seats (web UI)

bot:nn (policy only), bot:nnmcts (full search; NN_MCTS_SIMS env, default
400). Any checkpoint via DOMINION_NN_CHECKPOINT or bot:nnmcts:<path>.
Server loads hidden_sizes/input_scale/obs_version from checkpoint payload
(v2-aware loader). DEFAULT_NN_CHECKPOINT = campaign13/gen_0065.pt.
Finished games persist to exports/<session>.json in the server's cwd.
