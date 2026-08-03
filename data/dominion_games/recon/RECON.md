# dominion.games Protocol Recon (Milestone 0)

## Leaderboard / ratings protocol re-derivation (2026-08-02) — PROVEN

This section is newer than the ordinal table below.  It was re-derived from
the deployed `dominion-webclient-body-2.2.9.min.js` response named by the live
`https://dominion.games` index (the saved Chromium HTTP response has `Date:
Sun, 02 Aug 2026 21:51:00 GMT`, `Content-Type: application/javascript`, and
`Content-Encoding: br`).  The body was Brotli-decoded and beautified before
reading the implementations below.  Re-derive again when the body version
changes: both ids are positional.

### Outbound `REQUEST_LEADERBOARD`

`ClientToServerIds.REQUEST_LEADERBOARD` is **ordinal 28** in 2.2.9.  The actual
`serverMessenger.requestLeaderboard` implementation is:

```js
n.writeInt(getOrdinal(ClientToServerIds, ClientToServerIds.REQUEST_LEADERBOARD));
n.writeInt(count);       // default 50 in the stock UI
n.writeBoolean(flag);    // stock UI calls requestLeaderboard(50, true)
```

So one outbound WebSocket frame is exactly:

```
[int32 type=28][int32 count][boolean flag]
```

The client source does not give the boolean a semantic name; record it as a
flag rather than guessing.  The ratings poller sends `count=2147483647` and
`flag=true`, leaving the server to enforce its own maximum.

### Inbound message 25: current `Leaderboard`, not the old four-field stub

Inbound processor slot **25** is `r.leaderboard`, which calls
`Leaderboard.parse(reader)`.  Its wire layout is an enum-to-object map, not a
flat `RankedPlayer[]`:

```
[int32 ratingTypeMapCount]
repeat ratingTypeMapCount times:
  [int32 RatingTypes ordinal]
  [int32 entryCount]
  repeat entryCount times:
    [int32 namedId.id][string namedId.name]
    [int32 rank]
    [double level][double levelChange]
    [double skill][double deviation][double volatility]
    [double convertedSkill][double convertedDeviation]
    [int32 gameCount]
```

`RatingTypes` is ordered `{RATINGS_2P=0, RATINGS_3P=1,
RATINGS_2P_BLITZ=2, RATINGS_3P_BLITZ=3}`.  The displayed leaderboard rating is
`level` and its trend is `levelChange`; those are stored as `rating` and
`trend` by `scripts/dgames_ratings.py`, while the raw Glicko-related fields are
also retained for future analysis.

The bundle still defines an unused `RankedPlayer(namedId, rank, rating,
trend)` UI class near the leaderboard component, but it has no parser and is
**not** the message-25 handler.  The earlier `{namedId, rank, rating, trend}`
description is therefore incomplete for 2.2.9.

### Per-player lookup investigation

**No dedicated per-player rating/profile lookup was found.** The complete 55
entry `ClientToServerIds` object contains no `REQUEST_PLAYER`,
`REQUEST_PROFILE`, or equivalent rating command.  The only player-name query
near this feature is `REQUEST_CARD_STATS` (ordinal 53; `double minLevel,
string playerName, string version`) and its `CardStats` reply contains card
statistics, not a rating.  `REQUEST_CARD_PER_TURN` (54) is likewise card
statistics.  Friend/blacklist commands accept `NamedId`, but their replies are
relationship updates, not a player profile.  Coverage must therefore come from
repeated broad leaderboard snapshots keyed by the stable numeric `namedId.id`.

Static reverse-engineering of the public web client, 2026-07-31.
Source: `https://dominion.games/js/dominion-webclient-{head,body}-2.2.9.min.js`,
beautified and read. **No server contact was made for this note** — every
claim below is derived from client code, and is marked PROVEN (quoted from
source) or INFERRED.

Companion artifacts in this directory:

| File | Contents |
|---|---|
| `dgames_card_vocab.json` | all 881 `CardNames` entries with wire ids |
| `card_id_map.json` | the 33 base-set cards -> our `DefId`s |
| `extract_card_vocab.js` | re-runnable extractor for the vocab table |

## Headline findings

1. **The live spectator stream is imperfect information.** Hidden zones are
   redacted on the wire with a `-1` sentinel; the client never receives an id
   it could resolve for a card in someone's hand or deck. This is structural,
   not a UI blur. We land in the handoff's **fallback tier**: buy-decision
   imitation, not full policy+value tuples. See "Hand visibility" below.
2. **Login is mandatory** — there is no anonymous or guest path. Jack needs to
   create one dedicated account for the collector.
3. **The card vocabulary is fully solved and maps perfectly onto our roster.**
   dominion.games' `BASE` expansion is wire ids 1-33 and is *exactly* the 2E
   base set — identical to our supported 26 kingdom cards plus 7 basics. All 33
   matched our `DefId`s by exact English name with zero ambiguity.
4. **Kingdom composition cannot be pre-filtered.** You must join a table and
   parse its game state before you know whether the kingdom is base-only, so
   the collector must join-then-decide rather than filter-then-join.

## Wire format (PROVEN)

Binary, not JSON. `binaryType = "arraybuffer"`.

Two hosts, `wss://prod-dominion-alpha.dominion.games:443` and
`wss://prod-dominion-beta.dominion.games:443`. These are **not** a
lobby/game split — they are a redundant failover pair speaking an identical
protocol. The client probes both with `REQUEST_SERVER_STATE`, throws the probe
sockets away, and opens a fresh connection to the winner.

**Observed live, 2026-07-31** (single anonymous page load, no login), which
both confirms the design and corrects two details:

```
open  sock1 wss://prod-dominion-alpha...   out type=45 (15 bytes)   in [seq=0][1]
close sock1
open  sock2 wss://prod-dominion-beta...    out type=45 (15 bytes)   in [seq=0][2]
close sock2
open  sock3 wss://prod-dominion-alpha...
```

- **`REQUEST_SERVER_STATE` is outbound type 45**, settling the 44-vs-45
  ambiguity in the derived id table. Treat the rest of that table as ±1
  uncertain until each id is likewise confirmed against live traffic.
- The probe reply is `[int32 seq][int32 stateOrdinal]` with the counter
  starting at 0, confirming the inbound framing.
- **The winner is not simply "higher ordinal".** The client compares
  `serverState.value`, and `ServerStates` members are objects carrying an
  explicit `.value`, not bare ordinals. Here alpha answered ordinal 1 and beta
  answered 2, yet the client reconnected to *alpha* — so the ordinal→meaning
  mapping I derived statically (`DOWN`=0, `BAD_CLIENT_VERSION`=1,
  `UP_NO_NEW_GAMES`=2, `UP_NORMAL`=3, `UP_ASKING_RECONNECT`=4) does not
  predict the choice and should not be relied on. Unresolved; re-derive from
  the `.value` properties if a native client ever needs to make this choice.

Framing is asymmetric:

- **Outbound**: `[int32 typeId][fields...]`, one logical message per websocket
  frame. No counter, no length prefix — the frame boundary is the delimiter.
- **Inbound**: `[int32 seqCounter][int32 typeId][fields...]`, repeated; the
  server batches several logical messages into one frame. The counter starts
  at 0 on open and increments per message. A mismatch only fires a diagnostic
  event, it does not drop the connection.

All integers are big-endian. Primitives:

| Type | Layout |
|---|---|
| `int` | 4 bytes BE int32 |
| `long` | two consecutive BE int32s, high then low |
| `double` | 8 bytes BE float64 |
| `boolean` | 1 byte |
| `string` | int32 UTF-8 byte length, then raw bytes, no terminator |
| `optionalX` | 1-byte present flag, then the value if present |
| arrays/maps | int32 count, then that many elements |

Message type ids are **not literals in the source**. They are the 0-based
declaration order of keys in an enum object literal, resolved at runtime by
`getOrdinal`/`getByOrdinal`. The tables below were computed by reading that
declaration order directly. Because ids are positional, they are only stable
as long as the client build does not reorder keys — re-derive them whenever
the bundle version changes. (Encouragingly, the vocab table contains 13
`UNUSED_SLOT_N` placeholders, showing the developers preserve ordinal
stability rather than renumbering when a card is cut.)

## Auth (PROVEN)

**No anonymous access exists.** `FailureReasons.LOGIN_REQUIRED` is a distinct
server rejection, and no guest path appears anywhere in the bundle.

The version string is enforced. Every auth-adjacent message carries
`VERSION` (currently `"2.2.9"`), and the server can answer
`BAD_CLIENT_VERSION` both as a probe state and as a command-failure reason;
the client treats it as fatal and clears stored credentials. A third-party
client must claim the current deployed build's version string. There is no
version negotiation — the client just sends its hardcoded constant, so we
will need to re-read this constant whenever they deploy.

Login options (client -> server):

- `LOGIN` — `string username, string password, string VERSION, int language`
- `LOGIN_WITH_SESSION` — `int playerId, string VERSION, string sessionId, int language`
- `RECONNECT_SESSION` — `int playerId, string VERSION, string sessionId`

`loginSuccess` returns `int playerId, string username, bool isReconnecting,
int previousReconnects, <user prefs>, string sessionId, ...`. The
`playerId`/`sessionId` pair is what gets persisted and reused, so the
collector should log in once with a password and then hold the session.

**Keepalive**: after 10 seconds with no inbound traffic the client sends
`PING` and expects `pong` within 5 seconds. A long-lived collector must
implement this or risk an idle disconnect.

## Discovery and pre-join metadata (PROVEN)

Table listing is **pull, not push, and unfilterable server-side**. The client
sends `REQUEST_UPDATE` with payload `int UpdateTypes.TABLES (=3)` and receives
a `tablesOverview` message containing a full snapshot array of `TableSummary`.
There is no cursor, no pagination, and no server-side filter — you always get
the whole public snapshot and filter locally.

`TableSummary` (the lobby row) carries:

```
long tableId, NamedId host (int id + string name), int players, int bots,
int spectators, int minPlayers, int maxPlayers, bool isObservable,
bool isJoinable, int status, [long startTime if status == RUNNING]
```

`TableStati` = `{NEW:0, POST_GAME:1, RUNNING:2, ABANDONED:3, TRANSFERRED:4}`.

**Table id and game id are different longs.** `GameStarted` carries both; the
game id is only minted when play actually begins. `JOIN_TABLE` accepts a
*table* id only, and nothing in the client resolves a game id back to a table
id.

Spectating is `JOIN_TABLE` with `long tableId, bool asPlayer` where
`asPlayer = false`. The reply is a full `TableDetails`; if a game is already
running, board state arrives separately as `fullGameState`, which replays a
count-prefixed list of game events to reconstruct the current position. **This
is the first point at which the actual kingdom becomes visible.**

Spectating can be refused per-table: `TableRuleIds.SPECTATE_RULES` is a group
of `Nobody` / `Everybody` / `FriendsOf(host)` / `ListPlayerIds(whitelist)`.
The server pre-evaluates this per viewer into the `isObservable` boolean on
the lobby row, so we can respect it cheaply. No numeric spectator cap exists
in the client.

What we can filter on **before** joining: player count (`minPlayers`/
`maxPlayers`), in-progress (`status == RUNNING`), and spectatability
(`isObservable`). What we **cannot**: the kingdom. Even full `TableDetails`
only exposes constraints (`bannedCards`, `requiredCards`, `usedExpansions`),
not the cards actually dealt. Player ratings are also absent from both the
summary and the details — ratings live only in a separate global leaderboard
query that would have to be name-matched, which is unreliable for unranked
players.

Two viable discovery channels: poll the lobby snapshot, or scan table ids
sequentially with `JOIN_TABLE(id, false)`. The latter bypasses the snapshot
entirely, but should be used sparingly and politely given it is a bare
enumeration of their id space.

## Hand visibility — the decisive finding (PROVEN)

The client builds one flat pool of `CardObject`s at game start, one per
physical copy of every card, each with a persistent integer index. Knowing
"there are 46 Coppers in this game" is public; **which zone a specific copy
sits in is the secret**, and that is exactly what gets redacted.

Zones are populated from a wire array of indices into that pool. A real index
resolves to a true card name. The sentinel `-1` means "a card occupies this
slot, but you get no identity" — rendered as an anonymous count on a
`CardNames.BACK` stack and never resolved. The redaction is protocol-level:
the client is not filtering data it holds, it simply never receives it. An
instrumented or patched client would gain nothing.

The `-1` pattern appears precisely around the zones that hide real Dominion
information — hand, draw pile, set-aside, and the various mats — and not
around the always-public ones (supply, trash, play area, reveal).

Corroborating this, there is a per-player preference `SPECTATORS_SEE_HAND`
("Spectators can see my cards"). It is opt-in and individual, which only makes
sense if the default is to withhold. Some fraction of players will have it
enabled; those games would carry full information, and the collector should
detect and prize them, but we cannot count on them.

**What remains public at every decision point**: total copies of every card,
every zone's size and ownership, the supply, the trash, and the identity of
every card actually played, revealed, or discarded face-up. Critically, at buy
time the played treasures and full board are public — which is exactly the
information the fallback tier needs.

## Card vocabulary (PROVEN)

Cards are wire-encoded as **numeric ordinals**, not strings.
`CardName.serialize` writes `getOrdinal(CardNames, this)` and `CardName.parse`
reads `getByOrdinal(CardNames, readInt())`, so the wire id is the entry's
index in the `CardNames` object literal.

The full 881-entry table is extracted to `dgames_card_vocab.json`. The part we
care about is contiguous and clean:

- id `0` = `BACK`, the card-back placeholder (also used as a general
  "no selection" sentinel — distinct from the `-1` hidden-slot sentinel).
- ids `1`-`33` = expansion `BASE` = **exactly** Curse, Copper, Silver, Gold,
  Estate, Duchy, Province, then the 26 kingdom cards.

That set is identical to our supported roster. All 33 map to our `DefId`s by
exact English name with no unmapped entries and no ambiguity; the mapping is
written to `card_id_map.json`. Our only uncovered defs are Colony, Platinum,
and Potion (excluded by the filter anyway) plus synthetic test defs.

Detecting a base-only kingdom therefore reduces to **"every observed card id
is <= 33"**, which is about as cheap a check as we could have hoped for.

Pseudo-cards that can appear in the vocabulary and must be handled if we ever
widen scope: `PRIZE_PILE`, `RUIN_PILE`, `BLACK_MARKET_PILE`, `LOOT_PILE`,
`REWARDS_PILE`, `BOON_DRAWPILE`, `HEX_DRAWPILE`, `DRUID_BOONS`, `STATE_LIMBO`,
the `CARD_OF_THE_*` tokens, and 13 `UNUSED_SLOT_N` reserved ordinals.

## CORRECTION #2 (2026-08-02) — HANDS ARE USUALLY VISIBLE. READ THIS FIRST.

**This supersedes both the original note and CORRECTION #1 below.** Measured
over 80 real captures:

| hands visible in `fullGameState` | games |
|---|---|
| **both** | 60 (75%) |
| one | 18 (22%) |
| neither | 2 (2%) |

So **97% of games expose at least one player's hand, and 75% expose both.**

The `-1` redaction mechanism documented below is real — it is what hides the
one invisible hand in the 22% case — but it is simply **not engaged most of
the time**. `SPECTATORS_SEE_HAND` is evidently on by default or near-
universally enabled. I inferred the opposite from the client code (reasoning
that an opt-in preference implies an off default); that inference was wrong,
and the data disproves it.

Example from a real capture:
```
hand owner=0 contents=('Copper','Laboratory','Copper','Duchy','Gold') anonymous=0
hand owner=1 contents=[]                                              anonymous=4
```

**Consequences — all favourable, all reversing CORRECTION #1:**
- **Hidden-information decisions ARE recoverable** in most games: Militia
  discards, Cellar, Chapel, Sentry ordering, Poacher. You can see the hand
  before and after the decision.
- **Strategy B (forced-deal replay) is viable again** where hands are visible,
  since drawn identities become knowable. `set_deck_order` already exists.
- Scope conversion per-game by what is actually visible: full decision tuples
  where both hands show, partial where one does, buy-only otherwise.

Note the separate, weaker fact that remains true: **hand->discard events name
only the resulting top discard card** (measured: 46.1% of discarded cards
named across 733 events). So do NOT try to infer discards from the discard
pile — read the hand contents directly instead.

## CORRECTION (2026-07-31, from a complete spectator capture)

Two claims earlier in this note are WRONG for scraped spectator data. Both
came from analysis of the ARENA path, where our bot is a **player** and sees
its own hand. Neither transfers to spectating.

1. **Spectators never receive `questionAsked` (37).** A complete captured game
   (181648216, 424 inbound messages) contains **zero** type-37 frames, as did
   the earlier manual spectator capture. The 933 seen in the arena reference
   recording were player-seat frames. Consequence: `DecisionEntry` answers are
   positional indices into an offered element list **we never receive**, so the
   raw decision integers are undecodable in general. The earlier advice to
   "record questionAsked alongside the log" is moot — there is nothing to
   record.
2. **Strategy B (forced-deal replay) does NOT work from spectator data.** It
   requires drawn-card identities to force the deal; spectators see
   `Draw(count=N, cards=())`. The `set_deck_order` hook is real and the arena
   uses it, but the arena knows its own draws. We do not.

### What IS recoverable, and therefore what conversion must target

From the semantic log (`gameLogInfo` LogEntry) and public `CardMove`s we get,
with real card identities: every **play**, **buy**, **gain**, **trash**,
face-up **discard**, and **reveal**, per seat, in order — plus explicit
`Shuffle` events, turn boundaries, and the full `GameResult` (real scores,
per-card VP breakdown, turn counts, both final deck histograms).

What we never get: hand contents, deck order, and drawn identities.

So the tractable target is the handoff's **fallback tier**: buy-decision
imitation over a reconstructed information set. At buy time this is a good
deal better than it sounds, because the player has already played their
treasures — coins available, supply state, both deck compositions (derivable
from the public gain/trash history), turn number, and score are all known. The
outcome labels are real, which is what the c20 value-head finding needs.

Full policy targets over hidden-information decisions (Cellar, Militia
discards, Sentry ordering) are **not** recoverable from spectating and should
be dropped from scope unless a player-seat data source appears.

## Conversion: Strategy B, as originally scoped for PLAYER data (PROVEN)

The handoff asked whether a forced-deal hook exists before writing one. It
does, and it is already in production for a harder version of this problem.

- `set_deck_order(GameState&, PlayerId, span<const DefId>)` —
  `src/v2/core/state_builder.cpp:778`. Reorders a player's existing deck to an
  exact draw order, validating that the permutation preserves composition.
- `build_game_from_snapshot(const Snapshot&)` — same file. Builds a full
  `GameState` from counted zone contents, including mid-attack pending
  decisions. This is Strategy A's injection constructor, already built.
- Both are exposed to Python at `src/v2/py/module.cpp:1705` as
  `game.determinize(seed)` and `game.set_deck_order(player, defs)`, and are
  consumed today by the arena scraper at `src/v2/arena/shadow/bridge.py:252`
  and `src/v2/arena/fsm/game.py:1859`.

Randomness enters `GameState` in exactly two places, both via `shuffle_zone`:
the initial deal (`src/v2/core/setup.cpp:90`) and the discard reshuffle
(`src/v2/core/interp.cpp:52`). Nothing else — starting player, tie-breaks, and
kingdom selection are all deterministic.

The initial deal turns out not to matter for our scope: the starting deck is
7 identical Coppers and 3 identical Estates, so its shuffle order carries no
information.

**The one non-obvious technique to copy.** `set_deck_order` can only permute
the *current* deck; it cannot pull cards from the discard. The engine
reshuffles inside `draw_cards` the moment the deck empties, with no callback
seam, so you cannot catch it mid-reshuffle. The arena code's workaround
(`src/v2/arena/fsm/game.py:1855`) is to **preempt** the reshuffle: rebuild the
state via `build_game_from_snapshot` with the discard already merged into the
deck, so there is nothing left to reshuffle, then immediately `set_deck_order`
with the observed draw sequence. Any forced dealer we write must do the same.

Legality checking at each ply comes free: `game.legal_mask()` / `game.step()`
reject an impossible action, which is exactly the golden verification the
handoff demands, mirroring `ReplayVerificationError` in
`src/v2/records/tuples.py:110`.

**Recommendation: Strategy B**, and it is far cheaper than the handoff
assumed. Strategy A would require rebuilding a full snapshot at every ply and
would forfeit the free legality check.

### Two claims from the handoff, checked

- **Action ids CONFIRMED.** `src/v2/core/actions.h:11` gives `A_PASS=0`,
  `A_PLAY_BASE=1`, `A_WAY_BASE=42`, `A_BUY_BASE=206`, `A_EVENT_BASE=247`,
  `A_SELECT_BASE=251`, `A_OPTION_BASE=292`. So plays are `1+def` and buys are
  `206+def`, matching the decode in `bench/buy_stats.py:46`.
- **Deck order does not leak into the observation — CONFIRMED.** Every
  reference to a player's deck in `src/v2/encode/encoder.cpp` is either
  `.size` or goes through `add_ordered_composition()`, which sums into a
  per-def histogram and is order-independent. The single order-sensitive call,
  `zone_top_def_id()`, is applied only to an opponent's *discard* top, which
  is genuinely public in real Dominion. This holds at all obs versions.

## The finished-game replay path — probably a dead end (INFERRED)

Worth documenting so nobody re-investigates it hopefully.

`ReplayInstructions` (`long gameId, int decisionIndex, PlayerList`) is **not a
"view a finished game" feature**. It is a *continue-an-old-game* mechanic: it
travels as a `TableRule` value inside `CHANGE_TABLE_RULE` or
`NEW_TABLE_REQUEST`, is host-only and lobby-only, and its `PlayerList` is a
seat-assignment list — an empty list means "anyone may sit down and take over
a seat". The UI labels are "Load Game", "Load Old Game", "Load from End"
(which sets `decisionIndex = -1`).

Crucially, a replay-configured table then starts and streams the **same**
`gameStarted` / `fullGameState` messages through the **same** parsers as any
live game. There is no replay-specific deserializer anywhere in the bundle. So
the `-1` redaction question is server-side and identical to live play, and
since the feature is designed for *different humans to sit down and play on*,
applying normal per-seat visibility is the only sound server behavior.

Verdict: do not build on this without a live packet capture proving otherwise.
Undo/"Rewind" is likewise a live in-game roll-back, not an information channel.

## End-of-game disclosure — the real prize (PROVEN)

`GameResult` carries, unconditionally and with no redaction branch:

```
long tableId, long gameId, optional<RatingType>,
CardName[] emptyPiles,
PlayerResult[] { int playerId, int rank, Score,
                 CardFrequency[] finalDeck,
                 int resignIndex, optional<ResignationType> },
bool autoContinue
```

where `CardFrequency = { CardName, int frequency }`.

`CardFrequency.parse` always resolves a real `CardName` — unlike the zone
parsers, it has **no `-1` / `BACK` branch at all**. That makes
`PlayerResult.finalDeck` a complete, real-identity histogram of every card
each player owned at game end, across all zones.

This is delivered as a routine game-end broadcast on the single shared
processor table; there is no role-branching anywhere in the client's message
pipeline, so spectators should receive it too (a strong inference, since no
spectator-specific protocol path exists to inspect either way).

Practical value: we get final scores, ranks, per-player final deck
composition, empty piles, and explicit resignation signalling (`resignIndex`
plus a `ResignationType`) — which cleanly answers the handoff's question about
distinguishing resignations, and gives real final scores for the margin
target on games that ended normally.

## In-game streams (PROVEN)

Three streams run in parallel, and we want all of them.

**`fullGameState` (id 38)** — one complete board snapshot when you join:
players, the flat card-instance pool, zones, pile markers, tokens, counters,
turn description, cost reductions, temporary effects. Then a count-prefixed
batch of queued change events.

**`gameEventInfo` (id 32)** — the continuous delta stream. Each frame is
`int32 subtype` plus a subtype-specific payload, over 21 change classes. The
ones that matter:

| # | Class | Payload |
|---|---|---|
| 0 | `CardMove` | `fromZone:int, toZone:int, cardIds:int[], cardIdsAfterMoving:int[], movementType, animationClass` |
| 3 | `TurnDescription` | `ownerId, turnNumber, turnType, controllerId` |
| 4 | `Shuffle` | `owner:int, shouldIncludeDiscardPile:bool` |
| 5 | `Inspection` | `playerIndex, cardIds:int[], fromZone, toZone, isPublic:bool, isLinkedToQuestion:bool` |
| 12 | `PilesStatus` | `drawIndex, drawSize, discardIndex, discardSize, topCardId` |
| 2 | `PileUpdate` | `index, topCardId` (-1 = empty) |

`CardMove` carries **two parallel id lists**, which is how revelation is
encoded: `cardIds[i] == -1, cardIdsAfterMoving[i] == realId` is a card
becoming visible; `-1` in both means it stayed hidden from us. Draws are just
`CardMove` with `movementType == DRAW`, so drawn identity reaches only the
drawing player's own connection.

`Shuffle` is an explicit, dedicated event — we do **not** have to infer
reshuffles from pile-size arithmetic. That is exactly the checkpoint our
forced-deal replay needs, and it lines up with the preempt-the-reshuffle
technique in the arena code.

**`gameLogInfo` (id 33)** — the semantic log: `int32 startIndex, int32 count`,
then `count` entries each tagged `0 = LogEntry` or `1 = DecisionEntry`, with a
running global index.

### The single best finding for the fallback tier

`DecisionEntry` = `decisionIndex:int, playerIndex:int, decision:int[],
autoPlayed:bool`.

**Spectators receive the literal integer answer each player submitted for each
decision**, not merely its downstream board effects — plus a flag for whether
it was auto-played (which we will want to filter out, since auto-played
treasures are not real decisions). No role-branching exists anywhere in the
client's message pipeline; visibility is enforced purely by the server
substituting `-1` for card identities, never by suppressing messages.

So the imitation corpus we can build is considerably better than "buy
decisions reconstructed from public state" — we get the actual decision
stream. The catch is interpretation (below).

## Decisions: `questionAsked` (id 37)

Layout: `int32 questionIndex, int32 questionClass`, then a class body, where
class is one of `ChoiceQuestion`, `NumberQuestion`, `ComplexQuestion`,
`DelayedQuestion`, `NameQuestion`. Every question shares a header of
`type` (an ordinal into ~50 `QuestionTypes`: PLAY, GAIN, DISCARD, TRASH,
BUY, ORDER_CARDS, TOPDECK, CHOOSE_MODE, …), an `association` card id, and a
`Story` carrying a fine-grained `questionId` from ~150 values — including
card-specific ones like `SENTRY_TRASH` / `SENTRY_DISCARD` / `SENTRY_TOPDECK`,
`HARBINGER_TOPDECK`, `ARTISAN_GAIN`, `REMODEL_TRASH` / `REMODEL_GAIN`.

`ChoiceQuestion` bodies carry `min, max, content, declineButtonId,
defaultAnswers:int[], affectedCards:int[]`, where `content` is itself tagged
into ten element kinds (raw card instances, abilities, zones, card modes, card
names, game buttons, buyable cards, …).

**The critical interpretation rule**: a `ChoiceQuestion` answer is a list of
**positions into the offered `content.elements` array**, *not* card ids. To
decode any recorded decision you must have parsed the matching `questionAsked`
frame and index into it positionally. `NameQuestion` answers are global
`CardNames` ordinals instead, and `NumberQuestion`/`DelayedQuestion` answers
are literal integers. This means the collector **must record `questionAsked`
frames alongside the log**, or the decision integers are uninterpretable —
worth stating plainly because it would be an easy and unrecoverable omission.

On the handoff's ordered-sub-decision worry: the separate per-stage
`questionId`s suggest Sentry arrives as up to three sequential top-level
questions rather than one bundled frame, which is *good* news for mapping onto
our choice frames. A `ComplexQuestion` bundling primitive does exist
(`COMPLEX_AND` flattens length-prefixed sub-answers; `COMPLEX_OR` prefixes the
chosen branch index), but no base-set card was tied to it. Verify empirically.

## Turn structure, end of game, and undo

Turn boundaries are explicit `TurnDescription` events carrying both `ownerId`
and `controllerId`. Phase boundaries are *not* first-class — they are implied
by which question is being asked (`GAME_ACTION_PHASE`, `GAME_BUY_PHASE`,
`GAME_CLEANUP_PHASE` question ids).

`GameFinished` (id 14) wraps `TableDetails`, `GameResult`, `continueAllowed`,
`matchCompleted`. Beyond the final-deck histogram noted above, `Score` breaks
down as `totalPoints, usedTurns, ScorePart[]`, where each `ScorePart` is
`cardName, points, frequency, explanation` — a **full per-card VP breakdown
plus turn count**. `ResignationTypes` distinguishes `MANUAL`,
`MANUAL_FROM_RECONNECT`, `FORCE_RESIGNED`, and `TIMED_OUT`, which cleanly
separates real resignations from clock-outs at conversion time.

**Undo is the nastiest recorder hazard.** There is no "undo executed" message.
`metagameInfo` (id 35) broadcasts request/grant/deny/cancel as
`kind:int, playerIndex:int, decisionIndex:int`, but the *actual* rollback is
signalled only implicitly: a newly arriving log entry whose `index` is **≤ an
index already buffered** means the server has rewound, and the client silently
truncates everything from that index forward. A recorder must therefore treat
index regression as the rollback trigger and keep state checkpoints at
decision boundaries (or at minimum at turn boundaries) to roll back to.
Because undo is two-party consent, a passive spectator can only react to the
regression after the fact — never assume a granted request took effect.

## Two id spaces — do not conflate

- `CardName` ordinals: static, cross-game, "what card type is this". These are
  the 1-33 values in `card_id_map.json`.
- `cardId` / `cardIndex`: an index into this game's flat instance pool, "which
  physical copy". Only meaningful within one game, and `-1` means hidden.

## What we already own, and the gap (PROVEN)

`src/v2/arena/protocol/` is a working, empirically-derived decoder for this
exact protocol, built for the arena bot. It cross-validates this note.

### Independent confirmation of the static analysis

The arena's `protocol/cards.py` was generated from bundle **2.2.8** and its
ids 0-33 match my 2.2.9 derivation **exactly, card for card**. That is a
genuine independent confirmation from a different bundle version, and it also
demonstrates the ordinals are stable across releases.

Also CONFIRMED by working code: the framing (inbound `[seq][type]`, outbound
`[type]`, big-endian, length-prefixed strings/arrays), the `-1` hidden-slot
sentinel, the flat per-game instance pool, and table id vs game id being
distinct `u64`s. `protocol/parser.py` claims 100% structural coverage
(21,203/21,203 frames) of inbound 32/33 and outbound 37 on its reference
recording.

NOT COVERED by the arena code, so still resting on my static analysis alone:
`JOIN_TABLE`'s layout, the mandatory-login/VERSION handshake, and
`SPECTATORS_SEE_HAND`. The arena never needed any of them (see below).

### The transport problem

**The arena does not speak the websocket.** It drives a real Chromium via
Playwright with a persistent logged-in profile, injects `browser/ws_hook.js`
before page load to subclass `window.WebSocket`, and forwards frames to Python.
`docs/arena-plan.md` gives the rationale: the page already maintains login,
keepalives, and message ordering, and a second connection would conflict.

That is why login and `JOIN_TABLE` are absent from our code — it free-rides on
the browser's authenticated socket.

### Existing captures we can use immediately

**2.2 GB of real recorded frames already on disk** (untracked, gitignored):
`arena-recordings/20260724T142103.096991Z/frames.jsonl` (26,131 lines, the
canonical reference recording) and 64 run directories under `exports/arena/`
from 2026-07-24 to 07-30, some with per-game `frames.jsonl` + `events.jsonl` +
`decisions.jsonl`.

These are player-seat, not spectator, so they do not satisfy the M0 criterion
of ≥3 spectator captures. But they let us validate parsing, the card mapping,
and the whole conversion path **offline, today, without touching the site**.

### Gap summary

| Component | Status |
|---|---|
| Frame/primitive decode, message + event parsing, card table | **Reusable as-is**, pure stdlib, zero repo imports |
| Raw-frame JSONL record shape and writers | Reusable, needs a gzip wrap |
| Browser WS sniffing hook | Reusable if we keep a browser |
| Parser's own-seat assumptions | Light adaptation; spectator is *simpler* (all public) |
| Single-page/single-game session + FSM | Needs multiplexing to ~10 concurrent |
| **Table discovery / listing** | **0% built** — lobby code only knows automatch |
| **Spectator join** | **0% built** — `JOIN_TABLE` appears nowhere |
| Per-game persistence, scheduler, restart safety | New |

The decisive constraint: `protocol/` imports nothing outside stdlib, so it can
ship to the VPS standalone under the no-repo-source rule. Everything coupled
to the C++ bindings (`shadow/`, `bot/`, `actuate/`) is irrelevant to a pure
recorder.

### Offline validation actually run (2026-07-31)

Step 1 below was executed against the reference recording, no site contact:

- **Card map: 33/33 exact match** between `card_id_map.json` (derived from
  2.2.9) and the arena's independent 2.2.8 table. Both tables are 881 entries.
  Id 34 is Courtyard (Intrigue), confirming the base set ends exactly at 33.
- **`protocol/recording.py` parsed the 26,131-frame reference capture cleanly**
  into 16,883 events across 2 sessions, 0 empty frames, no parse errors.
  Observed message-id histogram matches the static table exactly: inbound 32
  (8,403), 33 (11,868), 37 (933), 38 (3), 14 (3), 41, 34, 36; outbound 37
  (932), 44 (heartbeat, 100). The 60 `UnknownFrame`s are all lobby/account
  types that never touch the game feed.
- **The decision stream decodes today**: 7,213 `DecisionResolved` and 933
  `PendingDecision` events, with outbound answers (932) pairing against
  questions (933). Also 153 explicit `Shuffle` events and 3 `GameResult`s.

So the decode stack is proven working on real traffic; what is unproven is
everything specific to *spectating* rather than playing.

## Live spectator capture, 2026-07-31 — the questions static analysis could not answer

Source: `captures/20260731T220433.671341Z/frames.jsonl`, 2,127 binary frames
over 388s, three tables spectated. This supersedes inference wherever it
disagrees.

### `JOIN_TABLE` confirmed exactly as derived

`[int32 type=2][int64 tableId][bool asPlayer]`, 13 bytes, trailing `00` for
spectate. Observed joining tables 1275, 589, 1411. `LEAVE_TABLE` (type 6) is
`[int64 tableId][int32 playerId]`. `REQUEST_UPDATE` (type 11) was sent with
payload 3 (= `TABLES`), confirming the discovery message.

### Joining mid-game replays the ENTIRE history — the big one

On spectate-join of a game already in progress, the server sends
`fullGameState` followed immediately by `gameLogInfo startIndex=0`, with the
**complete log from the beginning**:

```
39.2s OUT JOIN_TABLE table=1275 spectate=True
39.3s IN  fullGameState  gameId=181645115
39.4s IN  gameLogInfo    startIndex=0 count=777     <- whole game so far
41.3s IN  gameLogInfo    startIndex=776 count=2     <- live tail continues
```

Parsing that capture yields **618 `DecisionResolved` events with
`question_index` running from 1**, each with seat, answer tuple, and
`auto_played` flag — the full decision record of a game we joined partway
through.

**Consequence: we do not need to catch games at their start.** Any running
game can be joined at any time and yields its complete decision history plus
the live tail to completion. This removes the need for start-detection,
speculative id probing, or pre-emptive subscription entirely.

### Table ids are not game ids, and are not a growing counter

Observed table ids span **6 to 1414** and are small and slot-like, while the
game ids in the same capture are large (`181645115`, `181645376`). The
admins' "ids increase by one" refers to **game** ids; `JOIN_TABLE` takes a
**table** id. Sequential enumeration of table ids is therefore not a
meaningful discovery strategy.

### Discovery is already solved by one message

A single `tablesOverview` reply carried **340-348 tables**, of which ~260 were
`RUNNING` and **~225 were 2-player and observable**. Three snapshots minutes
apart showed max table id moving 1390 -> 1408 -> 1414. There is no pagination
and no need for enumeration: one poll returns the entire public table list.
Only 1-2 tables sit in `NEW` at any instant, which would have made
catch-it-before-it-starts a poor strategy even if it were necessary.

### Hidden information CONFIRMED from real spectator traffic

```
Draw(seat=1, count=5, cards=(),                 from_zone='deck', to_zone='hand')
Play(seat=1, count=1, cards=('Silk Merchant',), from_zone='hand', to_zone='in-play')
Buy (seat=1, count=1, cards=('Sea Witch',),     from_zone='supply')
```

Draws carry a count and **no identities**; plays and buys carry real card
names. This is the `-1` redaction observed live, exactly as predicted.

### Operational gotcha: `REQUEST_UPDATE` can be silently ignored

`REQUEST_UPDATE(TABLES)` sent too early — before the page's own lobby
subscription has settled — is **silently dropped**. No error, no rejection,
no `commandFailed`; the snapshot simply never arrives. Observed reproducibly:
one run won the race and got 340 tables, the next two hung until timeout.

The collector must **re-send `REQUEST_UPDATE` on a cadence until a
`tablesOverview` actually arrives** rather than assuming one request suffices.
`scripts/dgames_yield_sample.py` retries every 6s, and the first attempt after
login does routinely go unanswered.

Generalise the lesson: this protocol has at least one request that fails by
silence. Anywhere the collector waits on a server reply, it needs a timeout
and a retry, never an unbounded wait.

### MEASURED: base-only yield is 18.3% (2026-07-31)

`scripts/dgames_yield_sample.py`, sampling live human 2-player games:

| Metric | Value |
|---|---|
| Tables in snapshot | 333 (252 RUNNING, 79 POST_GAME, 2 NEW) |
| Running + observable + 2-player | 222 |
| ...excluding tables with bots | **110 human 2p games live at once** |
| Sampled | 98 |
| Successfully read | 82 |
| **Base-only** | **15 = 18.3% of reads** |

Every base-only game showed exactly **17 card types** (10 kingdom + 7 basics),
an independent sanity check that the filter identifies real base games.

The most common out-of-set cards were Potion (15 games) and then a tight
cluster at 12 games each — Colony, Platinum, and the Plunder split piles
(Amphora, Doubloons, Endless Chalice, Figurehead, Hammer, Insignia, Jewels,
Orb, Puzzle Box, Prize Goat), plus Horse. Note Potion/Colony/Platinum are
individually cheap to support and account for the single largest slice; if we
ever want to widen the filter, those are the highest-yield additions.

This **supersedes the earlier 0-of-5 pessimism**, which was simply too small a
sample (0 of 5 at p=0.18 has ~37% probability).

### Throughput projection

Two independent estimates of human 2-player base-game supply:

- **Little's Law (preferred):** 110 concurrent human 2p games; at a typical
  10-20 minute game that is ~5.5-11 completed games/min, i.e. 8K-16K human 2p
  games/day, of which 18.3% base-only = **roughly 1,500-2,900 base games/day**.
- **Table-id creation rate (rough, 68s window):** max table id moved
  1390 -> 1414 in 68s = ~21 new tables/min across all table types. Consistent
  in order of magnitude, but noisy and includes bot/4p/abandoned tables. Also
  note this implies the small table-id space is **recycled**, not monotonic.

At ~2,000 base games/day, the corpus goes from today's ~108 games to ~10K in
under a week and 100K in roughly two months. The 100K target is therefore
realistic, not aspirational — a large upward revision.

### Operational notes from the sampling run

- **Snapshot staleness drives failures.** 16 of 98 joins timed out waiting for
  `fullGameState`, and failures clustered heavily at the end of the run —
  tables sampled ~13 minutes after the snapshot had the most time to finish.
  The collector should re-poll the lobby frequently rather than working a
  stale list, and treat a join timeout as "game already over", not an error.
- The run ended on the 5-consecutive-failure guard, which is correct
  behaviour, and still wrote a complete summary.
- Minor: a `TargetClosedError` traceback appears during teardown after the
  abort, when a final `LEAVE_TABLE` is attempted on an already-closing
  browser. Harmless — the summary is written — but worth silencing.

### Superseded risk note: base-set yield

The handoff assumed base-only games would be plentiful because free-tier
accounts are limited to the base set. **Both sampled games were heavy
full-expansion**: Silk Merchant, Staff, Camel Train, Goat (Menagerie/Allies),
Sea Witch (Seaside), Young Witch (Cornucopia), Monkey (Menagerie). Neither
would survive our base-only filter.

n=2, so this is a signal and not a conclusion — but it points the other way
from the handoff's assumption, and it directly drives time-to-100K. **Measure
the base-only rate over a few hundred running games before building anything
further**; it is cheap (join, read the kingdom from `fullGameState`, leave)
and it decides the scale of the whole project. If the rate is low, options
include widening our card roster or weighting toward the free-tier player
population.

## Recommendation

**Reuse `protocol/` and keep the browser transport.** Running N Playwright
pages in one authenticated context sidesteps reimplementing login, the
`VERSION` handshake, keepalives, and reconnect logic — all of which are
server-enforced, version-fragile, and exactly the parts my static analysis is
least able to guarantee. The recorder already supports multi-page attach, so
this is an adaptation rather than new architecture. Revisit a native
websocket client only if the browser proves too heavy at ~10 concurrent games.

### Tooling built for this (2026-07-31)

- `scripts/dgames_login.py` — headful browser on a persistent
  `dgames-profile/`, **installs no websocket hook and records nothing**. This
  separation is deliberate: the `LOGIN` message carries the password in
  plaintext over the socket, so the session you type credentials into must
  never be a recording session.
- `scripts/dgames_capture.py` — the recording session, reusing the arena's
  `ArenaRecorder` against that same profile so it never disturbs
  `arena-profile/` or the bot's live games. Outbound login-family messages
  (ids 1, 13, 14, 15, 16, 19, 24, 44, 45, 46) are redacted to type + byte
  length before anything is written, as a belt-and-braces guard against a
  mid-capture re-login. Emits a heartbeat so a long manual session is never
  opaque. Writes to `data/dominion_games/recon/captures/`.
- Both, plus `dgames-profile/` and `data/dominion_games/raw/`, are gitignored.

Validated: redaction unit-tested (login frames stripped, gameplay frames
preserved), Chromium present, and a 12-second live run captured and correctly
redacted the probe frames shown above.

Suggested order:

1. **Validate offline first.** Run the existing 2.2 GB corpus through
   `protocol/recording.py`, and golden-test all 33 cards round-tripping
   through `card_id_map.json`. Zero site contact, immediate signal.
2. **Capture the spectate handshake.** The one thing static analysis cannot
   settle is what the client actually sends when you click "watch", and
   whether hidden zones really arrive as `-1` for a spectator. One manual
   browser session with the recorder attached answers both and produces the
   M0 captures. **Needs the account (see below).**
3. **Build discovery**: poll `REQUEST_UPDATE(TABLES)`, filter to
   `status == RUNNING`, 2 players, `isObservable`.
4. **Build the join-then-decide recorder**: subscribe, read `fullGameState`,
   drop the game unless every card id is ≤ 33, else record to completion.
5. **Convert** via Strategy B, reusing `set_deck_order` and the
   preempt-the-reshuffle technique.

## Blockers for Jack

- **A dedicated account is required.** There is no anonymous access, and the
  collector should use exactly one account, never parallel farming.
- **The M0 criterion of ≥3 saved spectator captures cannot be met by static
  analysis.** It needs that account or a manual browser capture.
- **Training-use permission is still open.** The handoff notes scraping was
  granted but training use was not confirmed in writing. Unchanged, and still
  Jack's to own.

## Caveats

Everything here except the card table and the framing rests on reading a
minified bundle, not on observed traffic. Message ids are positional
(declaration order), so **re-derive them on every client release**, along with
the `VERSION` constant. Treat the first live capture as the real test of this
document.
