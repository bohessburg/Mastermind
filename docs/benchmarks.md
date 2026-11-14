# Engine Benchmarks and Performance

Current engine: v2 (`src/v2/`). All numbers Release build, 2-player.
Live per-machine numbers: `./build/v2_bench` (JSON), gated locally against
`bench/baseline.json` via `bench/check_regression.py` (±15%). CI runs the
bench informationally (runner hardware differs from baselines).

**Always build Release for benchmarks** — Debug is 8–16× slower:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
```

## v2 engine (M1 Max baseline, July 2026)

| Metric | Value | Budget (REFACTOR_PLAN §4) |
|---|---|---|
| `step()` median | 42 ns | < 200 ns |
| Clone (memcpy GameState) | ~118 ns | < 1 µs |
| `legal_actions` mask | 26 ns | < 100 ns |
| Random-agent games/sec | ~38.8K | 50K target: accepted deviation (decision count per game, not engine cost — see IMPLEMENTATION_PLAN Phase 3 note) |
| BigMoney games/sec | ~80K | — |
| Heuristic games/sec | ~60K | — |
| Engine bot games/sec | ~58K | — |
| MCTS sims/sec (1-thread, EngineLike rollouts, K=2) | ~14.5K | — |

v1 comparison (deleted 2026-07): ~12K games/sec with known correctness bugs
and a benchmark-skewing kingdom-setup bug.

## Bot win rates (10,000 seat-swapped games, random kingdoms, July 2026)

These rows are the drivers **EngineBot v2**. The current strongest scripted
bot is **EngineBot v3** (2026-07) — full duel numbers, the chart-bot vs
drivers-bot distinction, and CLI recipes are in `docs/bot_strategies.md`.
For NN strength reads use `evaluate.py --opponent engine3`; the legacy
"engine" eval opponent is a much weaker chart bot.

| Matchup | Win / Loss / Tie |
|---|---|
| EngineBot vs BigMoney | 74.0 / 21.3 / 4.8 |
| EngineBot vs Heuristic | 79.1 / 17.8 / 3.1 |
| Heuristic vs BigMoney | 33.3 / 62.5 / 4.2 (known weakness: buys actions on boards that don't reward them) |
| EngineBot vs Random | 100 / 0 / 0 |

On the fixed engine-friendly kingdom (Village/Smithy/Market/Festival/Lab/…),
EngineBot vs BigMoney is 81.1 / 16.5. On basics-only boards the bots converge
(47/43) — always evaluate on real kingdoms (`eval_matchup` parameterless
overload is basics-only; avoid it for strategy comparisons).

## MCTS (Phase 6 scaffold)

MCTS(1K sims, EngineLike rollouts, K=2 determinizations) vs EngineBot,
200 seat-swapped random-kingdom games: **65.8% excluding ties** (gate: >50%).
Full trial history and NN training throughput: `docs/training-log.md`.

## Network inference on Apple silicon (c31 net, M1 Max, November 2026)

`CardTokenNet` builds 49 tokens (1 global + 48 supply slots); a two-player
base-set game uses 17 piles. `model.token_trim` (default `true`) drops the
unused supply tokens before the encoder. They are attention-masked and never
read by a head, so outputs are unchanged (max policy difference 2.4e-7 on
8,192 real positions). The trim applies to eager PyTorch everywhere (training
steps, gating, evaluation, the arena bot). A `torch.compile`d server graph
keeps the full width, so the CUDA box's compiled serving path is unchanged.

An inference server on a Mac is selected with `server_device: "mps"` and two
independent switches:

| Config | Serving path |
| --- | --- |
| `server_backend: "torch"` (default) | PyTorch on Metal |
| `server_backend: "mlx"` | MLX port of the network (`pip install mlx`); PyTorch still owns the weights |
| `server_fp16: true` | float16 weights (either backend) |
| `model.token_trim: false` | all 49 tokens (either backend) |

End-to-end through the inference server (8 synthetic workers sending real game
rows, merged batches up to 4096, shared-memory transport):

| Backend | Precision | Trim | evals/s | vs untrimmed fp32 net: max value diff, top-1 agreement |
| --- | --- | --- | --- | --- |
| torch | fp32 | off | 16,400 | reference |
| torch | fp32 | on | 42,700 | 2.0e-6, 100% |
| torch | fp16 | on | 48,500 | 3.1e-3, 99.98% |
| mlx | fp32 | on | 45,300 | 2.1e-6, 100% |
| mlx | fp16 | on | 57,600 | 4.3e-3, 99.94% |
| mlx | fp16 | off | 23,700 | — |

Reproduce with:

```sh
PYTHONPATH=build ./.venv/bin/python -m src.v2.train.inference_server --bench-server \
  --device mps --model-config configs/run_c31.json --workers 8 --max-batch 4096 \
  --coalesce-rows 4096 --duration 8 [--backend mlx] [--fp16] [--no-token-trim]
```

Only self-play serving goes through these backends; the training step, gate
and evaluation matches run PyTorch directly and get the trim only.

#### Full trainer on the M1 Max (2026-11-13, not completed)

`configs/run_c31_mac.json` (c31's settings, MLX fp16 serving, 8 workers) was
started twice from c31 gen 27 and stopped both times during generation 1's
self-play, so there is no measured generation time.

- Real self-play serves 44-47k evals/s (merged batches about 1,600 rows). The
  GPU is the limit: the eight workers use 10-20% of a core each and the server
  process about 80% of one.
- A c31 generation is about 160M self-play evaluations (box: 1,024 games in
  19 min at 139k evals/s), so self-play would take about an hour at that rate.
- It does not fit in 64 GB. A worker runs all of its assigned games at once
  (`selfplay.n_games` is ignored when a slot manifest is used), each game
  holds about 85 MB of search tree, and 1,024 games come to about 88 GB
  whatever the worker count (11 GB per worker with eight). The first attempt
  also had 17 GB of MLX buffer cache, pushed the machine into memory
  compression, and fell from 44k to 25k evals/s within 18 minutes.
- `MLX_CACHE_LIMIT_BYTES` (2 GB) now caps that cache; the server held about
  3 GB in the second attempt with no loss of throughput.
- The gate and the training step were never reached on the Mac.

#### One full generation on the M1 Max (2026-11-14)

`selfplay.max_concurrent_games: 64` plays each worker's 128 games in waves
(at most 512 games, about 44 GB of trees, at once), which fits 64 GB with the
memory compressor at 2-3 GB. One generation from c31 gen 27, c31 settings,
with the arena client running on the same machine for most of it:

| Phase | Wall time | Notes |
|-------|-----------|-------|
| Self-play | 70 min | 1,024 games, 228,906 positions |
| Training step | 0.5 min | 80 steps, batch 2,048 |
| Gate | 34 min | 800 games at 128 sims (box: 17 min) |
| Generation | 104.7 min | `wall_time` 6,284 s |

- Self-play was not steady. The first 30 minutes ran at 41-46k evals/s and
  finished 719 games; the last 305 games took another 40 minutes. The
  generation average was 25.5k evals/s (mean server batch 165 rows).
- The tail is on the workers holding the league and scripted games.
  `split_segments_by_quotas` fills workers in plan order, so workers 0-5 got
  self-play only and workers 6-7 got all 102 league and 102 scripted games.
  First waves took 12-20 min on workers 0-5, 35 min on worker 6 (86 league
  games) and over 46 min on worker 7 (16 league, 102 scripted). Why those
  games are slow is not established.
- A fix was tried and reverted the same day: dealing every segment across
  all eight workers and sending a step's per-model server requests together.
  The second run was stopped at 37 minutes with 541 games finished (the
  first run had 719 at 30 minutes); the server fell from 38k to 9-22k
  evals/s at minute 18 with five of eight workers nearly idle. Spreading the
  slow games gave every worker a slow wave tail. The slowest worker of the
  first run held mostly scripted games, which use one model, so sequential
  per-model round trips were not the main cause.
- The gate rejected the candidate at 0.444 (343-432-25), about 3.5 standard
  errors below 0.5. That is lower than the box's two generations trained
  from an empty buffer (c31 gen 1 at 0.501, gen 7 at 0.494) and lower than
  any c31 gate on the box (minimum 0.466). Cause not established. Held-out
  fit was normal (policy CE 0.704, value r2 0.679). Known differences from
  those box generations: anchor weight 0.1 (box: 0.5 and about 0.37), fp16
  MLX serving, and training on MPS.

### Compiled CUDA serving (RTX 5090 box, November 2026)

The box serves through `torch.compile` with static batch buckets and bf16
autocast; `server_trim_piles: 17` fixes the compiled token width at 18.
Measured beside a running trainer (shared GPU), gen 22 net, evals/s:

| Measurement | 49 tokens | 18 tokens |
|---|---|---|
| Forward only, bucket 256 | 78.9k | 106.0k |
| Forward only, bucket 1024 | 94.5k | 186.7k |
| Inference server, 16 synthetic workers | 60.5k | 77.1k |
| c31 self-play, 64 workers (server rate) | 105-110k | 147-148k |

Server benchmark: `python -m src.v2.train.inference_server --bench-server
--device cuda --model-config configs/run_c31.json --workers 16 --max-batch
4096 --coalesce-rows 512 --duration 10 --trim-piles 17` (`--trim-piles 0`
for the full width).

