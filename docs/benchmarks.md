---
name: Engine benchmark results and performance data
description: Bot win rates, games/sec performance, optimization history, and build configuration notes
type: reference
---

## Build Configuration

**Always build with Release for benchmarks:**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
Debug mode (`-O0`) is 8-16x slower than Release (`-O2`). Debug showed 340 games/sec, Release shows 12,000.

## Performance (Release, post-string-optimization, March 2025)

| Matchup | Games/sec | Avg Turns |
|---------|-----------|-----------|
| BM vs BM | 11,821 | 38.5 |
| Heur vs BM | 7,640 | 42.1 |
| Eng vs BM | 7,221 | 41.9 |
| Eng vs Heur | 5,135 | 44.1 |
| Eng vs Eng | 4,969 | 42.5 |
| Heur vs Heur | 5,333 | 45.2 |

Full benchmark (9 matchups × 5000 games = 45,000 games) completes in ~6.5 seconds.

## Bot Win Rates (seat-adjusted, 5000 games/matchup, random 10-of-26 kingdoms)

| Bot A vs Bot B | A win% | B win% |
|----------------|--------|--------|
| Engine vs BigMoney | 56% | 41% |
| Engine vs Heuristic | 69% | 28% |
| BigMoney vs Heuristic | 57% | 40% |

**Ranking: Engine > BigMoney > Heuristic**

## Bot Strategies

- **BigMoney**: Pure money, no actions. Build to 16 total money value then green. Optimized duchy/estate dancing from wiki strategy guide.
- **HeuristicAgent**: Plays all actions by priority, buys most expensive card with diminishing returns on duplicates.
- **EngineBot**: Kingdom-aware — scans supply for Village/Chapel/draw. Three modes:
  - FULL_ENGINE: Chapel + Village + terminal draw available → build engine, then green
  - BM_PLUS_X: Has good terminals but no full engine support → BigMoney + 1-2 key actions
  - PURE_BM: Nothing useful → falls back to BigMoney

## Optimization History

1. **String→int optimization**: `card_def()` from double hash lookup to vector index, Supply iteration without allocating `all_pile_names()`. Result: 6K → 12K games/sec (2x).
2. **Release build**: Debug → Release. Result: 340 → 12K games/sec in benchmarks (8-16x depending on matchup).

## For MCTS Readiness

12K games/sec for pure BigMoney rollouts is in the right ballpark for engine-side MCTS performance. The bottleneck for training will be neural network inference, not the game engine. State clone + step operations (needed for MCTS tree search) are not yet implemented.
