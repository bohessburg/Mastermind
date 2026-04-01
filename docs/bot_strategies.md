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

---

## BM_PLUS_X Optimization (v2)

**Seat-adjusted win rate vs BigMoney: 42.5% -> 75.3%**

Major rework of the EngineBot, centered on making BM_PLUS_X the dominant strategy. FULL_ENGINE was disabled entirely as BM_PLUS_X outperforms it on all tested random kingdoms.

### Strategy selection changes

- **Disabled FULL_ENGINE** — on random 10-of-32 kingdoms, BM_PLUS_X beats BigMoney ~75% while FULL_ENGINE only won ~22%. The engine overbuilds and gets outpaced by BigMoney's Province rush.
- **Expanded BM_PLUS_X triggers** — added Militia, Swindler, Festival, Moneylender, Bandit to the list of cards that trigger BM_PLUS_X (previously only terminal draw + Chapel + cantrip draw).
- **Chapel-only boards fall to PURE_BM** — trashing without a good action to pair it with loses to pure money.

### Card ranking overhaul

BM_PLUS_X best_action priority (what the bot splashes into its money deck):

| Rank | Card | Rationale |
|------|------|-----------|
| 1 | Witch | +2 cards + Curses destroy BigMoney (-24% delta) |
| 2 | Bandit | Free Gold + trashes opponent's treasures |
| 3 | Militia | +$2 + force discard to 3 (-18% delta) |
| 4 | Laboratory | Non-terminal +2 cards |
| 5 | Smithy | +3 cards |
| 6 | Council Room | +4 cards +1 buy |
| 7 | Festival | Non-terminal +$2 +1 buy |
| 8 | Market | Non-terminal +$1 +1 buy |
| 9 | Swindler | +$2, disruptive attack |
| 10 | Courtyard | +3 cards (topdeck 1) |
| 11-14 | Library, Moat, Scholar, Moneylender | Weak options |

Key demotions: Scholar dropped from #3 to #13 (discard-hand-then-draw is bad in a money deck). Sentry removed from buy list entirely (only bought as trasher via special logic).

### Density-based action limits

Replaced fixed copy limits (max 2 terminals, etc.) with density-based limits tuned via parameter sweep:

- **Terminal density < 0.08** — roughly 1 terminal per 12 cards. Prevents terminal collision.
- **Action density < 0.35** — keeps the deck money-focused.

The bot only buys the single best terminal in the kingdom (doesn't settle for weaker ones on cheap hands — waits for the $5 hand to buy Witch rather than buying Smithy at $4).

### Greening trigger

Replaced `total_money >= 16` with density-based greening:

- **money_density >= 1.2 (P1) or 1.1 (P2)** — P2 greens slightly earlier for tempo.
- **Turn > 5 failsafe** — always green after turn 5.
- Removed the old `has_trasher && junk <= 2` trigger which caused premature greening on thin-but-poor decks.

### Chapel logic

Chapel's presence in the kingdom was the #1 factor in losses (+18% delta initially). Systematic fixes:

1. **Attack-only Chapel** — only buy Chapel when Militia or Bandit is in the kingdom. Witch boards don't need Chapel (Witch is strong enough alone). Trashing without attacks doesn't beat BigMoney's speed.
2. **Cheap-hand buying** — only buy Chapel on turns 1-2 with $2-$3 hands. Don't waste $4+ hands on a $2 card.
3. **First-play aggression** — first Chapel play trashes all junk with no money floor. Subsequent plays preserve $3 in hand money so the turn isn't wasted.
4. **Conditional play** — after first play, only play Chapel if it's the only terminal action in hand. Don't waste an action on Chapel when Witch/Militia is available.

Result: Chapel delta reduced from +18.2% to +6.0%.

### Sentry logic

- Always trash Estates, Coppers, and Curses on reveal.
- Always discard Duchies and Provinces (don't trash VP cards we bought).
- When ordering kept cards, put actions on top of deck over treasures.
- Removed from the general buy list — only bought as trasher backup on Militia/Bandit boards.

### Swindler decision logic

When Swindler trashes an opponent's card, give them the worst replacement:

- Cost 0: give Curse
- Cost 2: give Estate
- Cost 3-5: give the worst action card (reverse of best_action ranking), never a treasure
- Cost 6: give an action if available, otherwise Gold
- Cost 7+: give same card type

### Throne Room / King's Court

- Play priority 0 (play before everything else — can double villages, cantrips, or terminals).
- Never throne Chapel.
- Not bought in BM_PLUS_X — a money deck with ~1 terminal doesn't have enough targets to justify TR's $4 cost over Silver.

### What helped most (ordered by impact)

| Change | Win rate impact |
|--------|----------------|
| Disable FULL_ENGINE, expand BM_PLUS_X | +20% (42% -> 62%) |
| Militia ranked #3 (above Smithy/Lab) | +3% |
| Bandit ranked #2 | +2% |
| Chapel attack-only restriction | +2% |
| Scholar demotion | +1.5% |
| Density-based action limits | +1% |
| Aggressive Sentry trashing | +1% |
| Smart Swindler decisions | +0.5% |

### Loss analysis insights

Cards most correlated with Engine losses (delta = loss% - win%):

| Card | Delta | Explanation |
|------|-------|-------------|
| Chapel | +6.0% | Tempo cost of trashing, even on attack boards |
| Sentry | +3.5% | Same trasher issue |
| Festival | +3.2% | Bot buys it but doesn't help enough |
| Everything else | <+2% | Noise |

Cards most correlated with Engine wins:

| Card | Delta | Explanation |
|------|-------|-------------|
| Witch | -24.1% | Curses are devastating to BigMoney |
| Militia | -17.7% | Discard-to-3 cripples $8 Province hands |
| All others | <-1% | Only attacks meaningfully beat BigMoney |
