# Bot Strategies

## Overview

Three heuristic bots with distinct strategies, used for benchmarking and as baselines for future AI training.

**Current rankings (seat-adjusted, 5000 games, random 10-of-32 kingdoms):**

| Matchup | Win rate |
|---------|---------|
| BigMoney vs Engine | ~58% / ~42% |
| BigMoney vs BetterRandom | ~100% / ~0% |
| Engine vs BetterRandom | ~98% / ~2% |

BigMoney is the strongest heuristic bot. Engine is competitive but slightly behind on random kingdoms. Both crush BetterRandom.

---

## BigMoneyAgent

**Source:** wiki.dominionstrategy.com "Big Money optimized"

**Strategy:** Never play actions. Only buy money and VP cards. No build phase — starts buying Provinces as soon as it draws $8.

**Buy rules:**
- **$8:** Province (unless very early game with no Gold and <5 Silvers → buy Gold instead)
- **$6-7:** Gold (unless ≤4 Provinces remaining → Duchy)
- **$5:** Silver (unless ≤5 Provinces remaining → Duchy)
- **$3-4:** Silver (unless ≤2 Provinces remaining → Estate)
- **$2:** Estate only if ≤3 Provinces remaining
- **$0-1:** Pass

**Strengths:** Consistent, fast, never times out. Averages ~36 VP in 38 turns. Simple and hard to beat because it never wastes turns on cards that don't directly produce money or VP.

**Weaknesses:** Ignores all kingdom cards. Can't use attacks, trashing, or draw. Gets crushed by engines on boards with Chapel + strong draw.

---

## EngineBot

**Strategy:** Kingdom-aware. Scans the supply and picks one of three modes.

### Mode Selection

**FULL_ENGINE** — requires Chapel/Sentry/Lookout (trasher) + Village/Festival (village) + terminal draw (Smithy/Witch/etc):
- Build phase: thin deck with Chapel, buy engine components by bottleneck analysis
- Green phase: buy VP when engine metrics are sufficient

**BM_PLUS_X** — has a good terminal or cantrip draw, but no full engine support:
- Buy Chapel first (if available), then 1-2 copies of the best terminal (Witch > Scholar > Smithy > ...), then Lab/Market, then money
- Green when total money ≥ 16 or Chapel has thinned the deck

**PURE_BM** — nothing useful in the kingdom:
- Identical to BigMoneyAgent

### FULL_ENGINE Details

**Build phase — bottleneck detection:**

Three metrics computed each buy decision:
- `draw_ratio = total_plus_cards / deck_size` — can we draw our deck?
- `action_ratio = total_plus_actions / deck_size` — can we chain actions?
- `money_density = (treasure_money + action_coins) / deck_size` — can we hit $8?

Buy priority depends on the bottleneck:
- Low actions → buy Villages urgently
- Low draw → buy Smithy/Lab urgently
- Low money → buy Gold/Silver

**Card yield data** used for density calculation (examples):
- Village: +2 Actions, +1 Card
- Smithy: +3 Cards
- Laboratory: +1 Action, +2 Cards
- Festival: +2 Actions, +2 Coins
- Market: +1 Action, +1 Card, +1 Coin
- Witch: +2 Cards
- Throne Room / King's Court: {0, 0, 0} — multipliers don't contribute standalone density

**Green trigger** (tuned via parameter sweep, best of 30+ configs):

| Metric | P1 threshold | P2 threshold |
|--------|-------------|-------------|
| draw_ratio | ≥ 0.45 | ≥ 0.3 |
| action_ratio | ≥ 0.25 | ≥ 0.15 |
| money_density | ≥ 1.2 | ≥ 1.0 |
| Failsafe turn | 5 | 4 |

P2 uses lower thresholds to compensate for going second (tempo disadvantage).

Also greens if Chapel has thinned to ≤10 cards with ≤1 junk and money_density ≥ 1.0.

**Engine build priority** (lower = buy first):

| Priority | Cards |
|----------|-------|
| 0-2 | Chapel, Lookout, Sentry (trashers) |
| 10-15 | Laboratory, Market, Festival, Sentry, Merchant, Hamlet (cantrips) |
| 20-21 | Village, Worker's Village |
| 25-37 | Witch, Smithy, Council Room, Scholar, Courtyard, Militia, Swindler, Moat (terminals) |
| 40-57 | Cellar, Harbinger, Throne Room, King's Court, Remodel, etc. (utility) |
| 80-81 | Silver, Gold (treasure fallback) |

Dynamic boosts during build:
- Witch boosted to 15 if < 2 terminal draw (attack + draw is very strong)
- TR/KC boosted to 22 if have targets to multiply

**Component limits:** 1 trasher, 2 terminal draw, 3 villages.

### Action Play Priority

All modes (except PURE_BM) play actions by priority:

| Tier | Cards |
|------|-------|
| Villages (play first) | Village, Festival, Worker's Village |
| Cantrips | Laboratory, Market, Sentry, Poacher, Merchant, Harbinger, Cellar, Hamlet, Lookout |
| Throne Room | Play before a terminal to double it |
| Terminals | Witch, Scholar, Smithy, Courtyard, Militia, Swindler, Library, Mine, Remodel, etc. |

### Sub-Decision Heuristics

- **Trash (Chapel/Remodel/etc):** Curses > Estates > Coppers. Money floor of $4 — won't trash Coppers if total deck money would drop below 4.
- **Sentry fate:** Trash Curses/Estates. Trash Coppers if money > 6, discard if money > 4, keep otherwise.
- **Discard (Militia/Cellar/etc):** Curses > Estates > Duchies > Coppers > other victory > actions > treasures.
- **Gain:** Pick most expensive option.
- **Moat:** Always reveal.
- **Throne Room:** Pick highest-priority action to double.
- **Yes/No:** Always yes.

---

## BetterRandomAgent

**Strategy:** Plays randomly but with basic common sense:
- Always plays all treasures (uses "Play all Treasures" shortcut)
- Always buys Province if affordable
- For multi-select decisions, picks randomly
- Prefers non-pass options over pass

**Purpose:** Baseline that's slightly better than pure random (which can't even end games). Still loses 100% to BigMoney and ~98% to Engine.

---

## What We Tried and What Worked

### Things that improved Engine vs BigMoney:
1. **Kingdom-aware strategy selection** (+14% overall) — falling back to BM_PLUS_X on boards without full engine support instead of building a bad engine
2. **Requiring Chapel for FULL_ENGINE** — Village+Smithy without trashing loses to money
3. **Density-based green trigger** — replaced component-counting with draw/action/money ratios
4. **Seat-aware thresholds** — P2 greens earlier to compensate for tempo disadvantage
5. **Smart trash money floor** — prevents Chapel from destroying buying power
6. **Sentry actually trashing** — was a no-op before (always kept cards)
7. **BM_PLUS_X mode winning 63% vs BigMoney** — the real workhorse on most boards
8. **Diminishing returns on action copies** — HeuristicAgent stopped buying 4 Witches

### Things that hurt or didn't help:
1. **Throne Room / King's Court as villages** — inflated deck profile metrics, made bot think it had enough +Actions when it didn't
2. **TR/KC at top build priority** — bought TR before having anything to throne
3. **TR/KC contributing to card_yield density** — inflated metrics, greened too early
4. **Sentry at priority 1 (above Lab/Witch)** — $5 trasher competing with best engine cards at same price point
5. **Higher draw thresholds (0.6-0.8)** — engine overbuilt and got outpaced by BigMoney's Province rush
6. **Build phase money threshold (16 total money)** — delayed greening, wiki BigMoney's immediate greening is faster

### Key insight:
On random kingdoms, ~40% of boards have no engine support at all. Another ~35% have partial support (BM_PLUS_X territory). Only ~25% have Chapel + Village + draw for a real engine. The engine bot's overall win rate is dragged down by the boards where it can't build an engine. Its real strength shows on the BM_PLUS_X boards where Chapel + 1-2 good terminals beats pure money 63% of the time.

---

## Performance

All measurements on Apple M1 Max, Release build (`-O2`).

| Matchup | Games/sec |
|---------|-----------|
| BigMoney vs BigMoney | ~12,000 |
| Engine vs BigMoney | ~4,500 |
| Engine vs Engine | ~3,000 |

Stress test: 100K Engine vs Engine games in ~3.3s (30K games/sec across 10 threads).
