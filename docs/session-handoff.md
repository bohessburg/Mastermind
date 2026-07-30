# Session Handoff — Training Program State

Snapshot: 2026-07-30 (post-restructure-build night). Read with
`docs/training-log.md` (the 2026-07-28..30 entries at the tail are the
current state), `IMPLEMENTATION_PLAN.md` Phase T3 (the restructure build
order + status), `docs/human-games-analysis.md`, `docs/bot_strategies.md`,
`docs/web-ui-handoff.md` (web UI remains a separate session's domain).

## THE CENTRAL FINDING (2026-07-28) — still the premise

Every net in the lineage (c15 champ, c18, c19) prices a fully-built,
thinned engine deck BELOW a plain money deck on an engine-dominant board.
The value head is where engine lines die; no policy-side curriculum can
convert. Fixes must change the OUTCOME DATA the value head trains on and
the value function's ability to represent it. The c20 restructure
(Phase T3) is that program — its build phase is now COMPLETE except for
assembly.

## Revisions from the 2026-07-29/30 audit + build night (read the log)

- CLAIRVOYANCE: all training selfplay AND all EvalRunner/duel numbers in
  the log ran perfect-information search (opponent hands + exact future
  draws). BUT matched A/Bs show the information itself is worth ~0 vs
  engine3; the eval-vs-serving gap (~54 vs ~45-48) is STACK quality
  (K=2 world-splitting ~3.6 pts + suspected within-decision batching
  distortion ~6 pts, task #25). The 1600-sim "deep-search regression"
  was a clairvoyant-regime artifact; honest depth is flat 200-1600.
- c20 BARS (pre-registered): serving-harness honest @400/K2 — c15 44.9,
  c19 47.7, c19-vs-c15 45.6; EvalRunner-honest ~54. Track both.
- VALUE-TARGET GEOMETRY: 3-seed offline sweep — pure margin (alpha=0)
  is uniquely seed-stable and prices built engines at money parity;
  alpha=0.6's saturation geometry is the OOD cliff. Leading candidate
  for c20: alpha=0.0 (fallback 0.2-0.4); Duchy-injection probe is the
  full-scale non-regression gate (the c17 fix must hold).
- MILITIA KEEP-INVERSION PERSISTS in c18/c19 — obs-v3 select semantics
  gave the input, training never taught its use. Data problem again.
- ENCODER COMPAT: the landscape-sentinel repair changed ~34 obs floats
  that were constant-1.0 forever. Pre-fix checkpoints (c13..c19, incl.
  the deployed c15) MUST NOT run bare on post-fix builds — use
  --legacy-shim (exact: reproduces c15 probe means to 4 decimals) or
  pre-fix builds. Checkpoints now stamp encoder_generation; loaders
  guard. Current Hetzner image is safe (old build baked in).

## What is running RIGHT NOW

Nothing remote. No GPU box. All restructure code is LOCAL AND
UNCOMMITTED on v2-phase1 (tree snapshots in the session scratchpad;
Jack has not asked for commits). Serving: Hetzner VPS still c15
gen_0045 on its old baked image (safe).

## The restructure build (Phase T3) — landed 2026-07-29/30

All reviewed + full suites green (175 C++ / 189 py at close):
determinized selfplay (off|per_decision|per_turn; A/B: ~7-11% cost,
identical trajectories) · honest eval/duel modes (--honest) · true
legal-mask recording (policy>0 bug dead) · forced playouts + policy-
target pruning (KataGo, default-on candidate) · per-seat buy-phase
temperature schedule (temp_mode per_seat_buy, buys tau=1 thru turn 14)
· imitation stack (BC pretrain + floored persistent anchor + AWR
option; 12,734 human-seat tuples from 111 verified games incl. 2
Hetzner) · aux margin-distribution head (improves val MSE at equal
policy CE) · AdamW default · corpus quarantine (123 pytest artifacts +
92 stubs + 9 corrupt excluded; server tests export to tmp now) ·
duel-validated curriculum pools (configs/kingdom_pools_c20.json:
sentry_engine 94.5, thin_engine 88.4 vs BM; classic draw-engine board
REJECTED at 51.3 — validate every pool) · encoder-generation guard +
legacy shim · probe suite codified (scripts/probes/run_all.py) ·
honest_eval.py harness (serving-identical measurement).

## Open items (task list; Jack-gated marked ★)

1. ★ c20 ASSEMBLY (#21): run_c20.json + pre-registration. Decisions:
   alpha (rec: 0.0), template/SIL disposition (memo in task #20; rec:
   imitation-anchor-first, templates dormant, SIL built-but-off),
   determinize=per_turn, forced playouts on, temp per_seat_buy,
   curriculum pools + fractions, league seeding, sims budget WITHOUT
   tree reuse (forfeited under determinization), from-scratch d320.
2. ★ Data collection (#13): Jack local sessions are the gold channel
   (+50-100 games wanted; hot-seat human-vs-human uniquely valuable —
   zero exist). Server auto-export (#23) in flight tonight; VPS
   redeploy AFTER it lands (new build = c20-generation checkpoints
   only, or old image untouched).
3. Serving-stack search parity (#25): matched-board residual isolation,
   wave-size sweep, K=1-vs-K=2 deploy decision — potential free +6-10
   pts for the live bot. High value, orthogonal to training.
4. Arena value-only tuple pipeline (#24, blocked on regime flip in
   assembly): 216+ archived losses to real humans = the missing
   "money loses on engine boards" outcome data for the value head.
5. c21 neural exploiter league: deferred, unchanged.

## Eval discipline (updated)

- Honest reads only: honest_eval.py (serving-identical, deployed
  config) and evaluate.py/duel.py --honest. Clairvoyant mode retained
  solely for historical comparison columns.
- Legacy checkpoints: ALWAYS --legacy-shim (or pre-fix builds). Native
  EvalRunner/duel refuse mismatched generations by design.
- Probes: scripts/probes/run_all.py per milestone; value probe is the
  c20 headline instrument (first checkpoint pricing the probe's engine
  deck ABOVE money = the breakthrough signal); Duchy probe is the
  target-geometry non-regression gate; human-record probe tracks
  "recognizes winning human lines" (baseline money-default ~47-54%).
- League proxies stay directional-only; n=200 sigma ±3.5 unchanged.

## Ops lessons (carry forward, unchanged + additions)

- All prior stop/kill/rsync/watcher/cgroup lessons stand (see previous
  handoff revisions in git history).
- macOS: no setsid — detach with `nohup ... & disown`; no `timeout`
  binary; `==` breaks zsh echo args.
- Never pattern-kill; collect PIDs first (monitor shells self-match
  patterns like "run_ab.sh").
- Parallel Codex delegations: disjoint file sets only; anything
  touching the probe/loader path while probes run in subprocesses will
  race new guards (the encoder guard broke in-flight sweep probes —
  expected failures, re-ran with shim).
- Codex rescue agents are single-forward-only: poll/retrieve via the
  companion script from the main session (status/result verbs).

## Playable seats (web UI)

Unchanged mechanics (bot:nn, bot:nnmcts, engine3/engine2/thinner/
bigmoney, DOMINION_NN_CHECKPOINT override). NOTE: with the repaired
encoder in this tree, serving any pre-fix checkpoint from a FRESH build
requires the legacy shim (load_policy legacy_shim=True); the deployed
VPS image predates the fix and is unaffected.
