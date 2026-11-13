# Human-Games Loss Analysis — c15 gen_0045

Snapshot 2026-07-25. Data: all unified game records (`exports/records/`,
schema 1.0) where one seat is `bot:nnmcts` (flagship c15 gen_0045) and the
other is human — 112 arena games with decoded standings + 37 completed local
web games vs Jack. Analysis scripts and per-game feature dumps live in the
session job dir (`analyze_losses.py`, `analyze2.py`); everything below is
reproducible from `exports/records/` alone.

## Headline records

| Set | Raw record | Competitive record* | Win rate |
| --- | --- | --- | --- |
| Arena (dominion.games) | 44W–64L–4T (39%) | 33W–39L–4T | ~46% |
| Local vs Jack | 4W–33L | 4W–33L | 11% |

\* Competitive = excluding 25 technical losses (below) and 11 early
opponent-quits (humans bailing by turn ≤6 with 3–3 scores) that were recorded
as wins.

## Finding 0 — 39% of arena losses are one harness bug, not gameplay

**Every arena game in which the bot was hit by Militia died at the discard
question: 35/35, zero survivals.** 25 of the 64 recorded arena losses are
this bug: the bot submits an answer the server rejects (observed
`ANSWER_QUESTION` with 3 card indices against a `min=2 max=2` discard, e.g.
answers `[0,1,3]` for hand `[Estate,Estate,Copper,Estate,Copper]`; server
replies unhandled `msg_type 1` payload `ffffffff`), the game hangs, the human
waits ("hello?", "You there???"), then force-resigns the bot via the timeout
offer. Several aborted sessions show a second mechanism on the same question:
DOM gestures failing to select a second same-name card stack
(`status=no-effect` / `not-found` clicking the second Copper).

This is the top-priority arena fix — it alone moves the headline arena
record from 39% to ~50%. The answer-cardinality symptom (3 discards chosen
for a discard-down-to-3 from a 5-card hand) suggests the seeded Militia
interrupt frame is built over a 6-card shadow hand; verify the tracker/
StateBuilder hand count at interrupt seeding before touching the actuator.

Also recoverable: 20 completed arena games have `reason: game-finished` but
no decoded standings (older parser); a results backfill would grow the
sample.

## The strategic pattern — one coherent story

Every strength-relevant signal in the competitive losses points the same
way: **gen_0045 is a big-money+Witch bot, and it loses to humans who build
thin engines.** It converged to the archetype of its training opponents
(engine3 = bigmoney+x) and never explored out.

### 1. Repertoire collapse

Per-game average gains across all 112 competitive-record arena games:
Silver ~5, Gold ~4, Duchy ~3, plus Witch/Gardens/Bandit/Militia splash.
Cards the bot has **never bought in any human game, arena or local**:
Sentry, Chapel, Throne Room, Remodel, Workshop, Moneylender, Mine, Library.
Village, Market, Merchant, Festival, Smithy, Council Room are ≤0.15/game.
The gen-30 playtest note ("never bought Sentry") is still true at gen 45.
Opponents in losses average 5–8 engine-card gains (Laboratory 1.7, Merchant
1.5, Throne 1.4, Vassal 1.4, Sentry 1.3, Festival/Village/Market ~1).

### 2. Win rate collapses against engine-ness

Bucketing all competitive games by opponent action-plays-per-turn:

| Opp action plays/turn | Bot record | Win rate |
| --- | --- | --- |
| 0.4–1.5 (money-ish) | 10W–16L–2T | 36% |
| 1.5–2.2 | 13W–14L–1T | 46% |
| 2.2–3.1 | 9W–19L | 32% |
| 3.1–7.4 (real engines) | 5W–23L–1T | **17%** |

The bot is roughly even with money-playing humans and gets crushed by
engine players. (Bot's own action plays/turn: 0.26 — vs opponents' ~2.)

### 3. Deck thinning is the sharpest kingdom-level killer

- Opponents had trashing (Chapel/Sentry/Moneylender/Remodel/Mine) in
  **64/72 competitive losses** vs 28/37 wins; opponent cards-trashed
  averages 8.4/game in losses vs 5.7 in wins. The bot trashes ~0.
- Kingdom-presence win rates (arena, base ~43%, ±9pp at n≈30): **Sentry
  26%**, **Bandit 26%**, Bureaucrat/Militia 33%, Village 38%, Remodel 39%.
  Bot-favorable: Library 57%, Workshop 54%, Council Room/Vassal 53%,
  Festival 52% (terminal-draw kingdoms where money keeps pace).
- Bandit is doubly bad: a treasure-trashing attack is maximally effective
  against the bot's Silver/Gold deck (bot loses ~2.1 cards/game in Bandit
  losses), while the bot's own Bandit buys (~1/game in those losses) do
  little against thin decks.

### 4. Curses don't save it — Witch mirrors expose the archetype gap

The bot contests Witch fine (buys ~0.85 when present; Witch-kingdom win rate
48%). But in mirror games humans absorb curses and win anyway: in local
losses Jack took **3.5 curses/game and still won by a median 15 VP** —
thin engines shrug junk that cripples a money deck. Example (arena
181378972, 11–48): bot buys 4 Witches and curses relentlessly; opponent
opens Chapel, trashes down, buys 8 Laboratories/Merchants, and takes 8
Provinces in the last 4 turns.

### 5. Losses are blowouts, and greening explains the shape

- Loss margins: arena median −10 (12/39 worse than −15); local median −15.
  Only 11/39 arena losses are within one Province. These are deck-power
  gaps, not endgame coin flips.
- In losses the bot takes its first Province at turn ~10.8 vs the human's
  ~11.9, wins the Duchy split 3.4–1.4 — and loses Provinces 2.4–5.3 (local:
  1.9–6.1). Humans green later off a stronger deck and buy 2–4 Provinces
  per turn at the end; early greening + Duchies is exactly how a money deck
  loses to an engine, and no amount of value-head calibration (c17) fixes a
  deck-building policy gap.
- Pathology worth a probe: Militia-stacking — one local loss shows 8 Militia
  buys (terminal collision, zero draw); the bot overvalues the attack
  effect.

## Confirmed: the Militia keep-inversion is the network, not the harness

After the arena Militia fix went live (2026-07-25, verified across 10+
survived hits), every live discard showed the same pathology: the bot keeps
its cheapest cards (kept Estate+2 Copper over Silver; kept 2 Copper+Silver
while discarding 2 Silver; discarded Gold to keep Gardens+Copper). A headless
probe (`militia_probe.py`, session job dir) rebuilt these exact hands as
engine states — no arena, no web — and ran the identical arena decision path:

- **Arena search (400 sims, 2 det.) and raw greedy policy reproduce the live
  junk-keeps exactly**, including keep [C,C,E]→discard Silver and the control
  case keeping 2 Estates + Gold over Gold + 2 Silver.
- **The value head ranks keep-sets correctly in every case** (keep
  Gold+Silvers first, keep own Militia, pitch Estates). The policy head is
  inverted, and its prior is confident enough that 400 sims never escape it
  across the 3-step keep frame (value deltas are only ~0.03–0.10).

Mechanism (from `src/v2/encode/encoder.cpp` `encode_decision_v2`): the
observation's decision block is a 10-way decision-kind one-hot plus 5
scalars, where the deciding card is a single raw float def id. Militia's
keep-choice, Cellar's discard-choice, and Chapel's trash-choice all share
`DecisionKind::Choose` and differ only in that one scalar. The policy
generalizes one "select the junk" habit across all Choose frames — correct
for Cellar/Chapel/Poacher/Remodel fodder, exactly inverted for Militia's
keep-semantics. The value head is immune because it evaluates the
post-resolution state, which is unambiguous. Self-play never punished it
because both sides share the blind spot and money mirrors barely price it.

Fix direction (obs-v3/c18): embed the decision-source card properly (token
or one-hot, not a raw id scalar) and/or add a keep-vs-relinquish semantic
flag to the decision block; then check whether Militia/Bureaucrat kingdom
win rates recover.

## Why training never punished this

- The league (self-play + engine3 mix) contains no thin-engine opponent;
  engine3 is money-shaped, so the money archetype is never exploited in
  training. c16 already showed engine pressure alone doesn't convert.
- Trash composition is unencoded (known obs-v3 gap) — the bot cannot
  represent what thinning does to a deck.
- Value targets were fit on money-mirror trajectories where early greening
  is correct; vs engines the timing equilibrium is different.

## Implications for the next campaign (c18)

1. **Arena harness first:** fix the Militia discard bug (verify seeded
   interrupt hand size, then the answer path). Free +10pp on the only
   unsaturated benchmark. Backfill the 20 no-standings results.
2. **Break the archetype monoculture in the league.** Add engine-building
   scripted opponents (Chapel-thin money, Village/Smithy/Lab engine,
   Sentry-thinner) and/or seed league slots from checkpoints forced to buy
   engine cards. The human losses define the exploiter profile precisely:
   trash-heavy, 2+ action plays/turn, late greening.
3. **obs-v3 trash composition** is now empirically motivated, not just
   principled: the three worst kingdom effects (Sentry, Bandit, Chapel-heavy
   losses) all run through the trash.
4. **Exploration at buy nodes:** repertoire collapse (eight cards never
   bought in 149 human games) means self-play never tries these lines.
   Forced/randomized openings over kingdom cards, or opening-book
   diversification in self-play, so Sentry/Chapel/Village lines enter the
   replay buffer at all.
5. **New sentinel:** engine3 measures money-mirror strength and is nearly
   saturated (~57%); add a thin-engine scripted sentinel so campaign reads
   track the weakness humans actually exploit. The 113 competitive human
   games in `exports/records/` are a permanent eval set for "does the new
   net punish money?" regression probes.
