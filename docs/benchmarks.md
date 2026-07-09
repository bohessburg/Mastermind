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
