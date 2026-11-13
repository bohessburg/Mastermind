# Arena Stream — Progress Log

Running log for the "arena" stream: the best trained checkpoint (campaign 15
gen 45, card-token transformer) playing live base-set games against people on
dominion.games. The account is openly identified as a bot and the experiment
is run with permission. Architecture lives in `docs/arena-plan.md`; this file
tracks what is built and what is next.

## Context

dominion.games runs a standard web client that talks to the game server over
a public WebSocket. The same feed is what the site's own client uses to render
a game, and the site supports attaching to a running game's feed to spectate.
Our integration simply speaks that same client protocol: a browser session
(driven by Playwright) runs the normal client and uses its connection to handle
the lobby, we read the game feed the client already receives, and we send
question answers through that connection using the client's observed,
stateless `ANSWER_QUESTION` shape. Playwright still operates the normal lobby
and modal controls.

## Current actuation mode: protocol (2026-07-24)

In-game `PendingDecision` actuation now defaults to direct client message 37,
not DOM/canvas clicks. `src/v2/arena/browser/ws_hook.js` retains the current
open game socket across replacement/reconnect and exposes a raw-byte sender;
`ArenaSession.send_frame()` builds the outbound envelope. Sending still goes
through the hook's wrapped `WebSocket.send`, so every bot-generated answer is
mirrored into `frames.jsonl` in the same form as a client-generated outbound
frame and is decoded by the normal replay parser.

The trigger was three independent live aborts on the same Militia multi-select
prompt: region mapping, just-in-time/polled targeting, and occlusion-aware probe
points all failed in turn. In
`exports/arena/20260725T013442.698102Z/` (game 181369850), the final synthetic
click physically landed on the verified Copper target twice but the client's
canvas picker produced no selection effect. P1 had already proved that message
37 is the clean fallback reserved by design decision 3: 933 questions and 932
answers parse exactly as
`[questionIndex:u32][answerCount:u32][answers:u32...][autoPlayed:boolean]`.

The protocol actuator sends the mapper's first valid answer encoding with
`autoPlayed=false` after the existing think-time delay. The immediate
`DecisionResolved` verification and equivalent-encoding acceptance remain in
place. The old `PlaywrightActuator` question-click machinery is retained as
the operator fallback:

```json
{
  "actuation_mode": "protocol"
}
```

The only valid values are `"protocol"` (default) and `"clicks"`. Undo denial
remains a modal click, and game-ended/lobby controls (`OK`, `Start Search`,
`Ready`, and `Leave Table`) are unchanged. Corpus tests re-encode all 932
recorded outbound answers byte-for-byte, drive the three-game replay through a
fake protocol session (807 mapper-first frames byte-identical to the human
frame; 125 valid duplicate-card/order alternatives), and replay Militia
question 50 as one message-37 send with no game-DOM interaction.

## Autoplay/deferred-buy recovery (2026-07-25)

Live game 181375685 did not fail on a malformed follow-up from that game. The
session archive shows that the deferred state came from the preceding game
181375622:

```text
1784959146569 PendingDecision question_index=12 ... question_id='GAME_BUY_PHASE'
  offered=(..., '1:0:Copper', '1:0:Copper', '1:0:Copper',
  '2:AUTOPLAY_TREASURES') minimum=0 maximum=3
1784959149592 GameResult game_id=181375622 reason='game-finished' ...
1784959149903 DecisionResolved question_index=12 answers=(2, 0) seat=0 ...
```

The result arrived during the configured think delay, before the bot's
outbound autoplay answer. Consequently there was no follow-up buy question,
but the long-lived provider retained game 181375622's deferred buy. In the
next game, only the start handshake had been submitted before the abort. Its
first local buy frame was the ordinary initial question:

```text
1784959190593 PendingDecision question_index=5 ... question_id='GAME_BUY_PHASE'
  offered=('0:Curse', '0:Copper', '0:Silver', '0:Estate', '0:Chapel',
  '0:Harbinger', '0:Moat', '0:Village', '1:0:Copper', '1:0:Copper',
  '1:0:Copper', '2:AUTOPLAY_TREASURES') minimum=0 maximum=3
```

The old guard compared that frame with the stale `(game_id=181375622,
turn_number=3, seat=0)` deferred state. Game 181375685 was at turn 1, seat 1,
so it raised before sending any answer to question 5.

Providers now clear multi-question state at both game start and game end. A
same-game deviation no longer aborts merely because it is not the expected
treasure-free buy prompt: stale cross-turn state is dropped, interleaved
questions are planned while retaining the deferred buy, and a buy prompt that
still offers treasures is rebuilt from the tracker and replanned. Autoplay is
attempted at most twice per chain; subsequent prompts play one offered treasure
explicitly until the NN can be evaluated on a fully collapsed buy state. The
NN is never called while an offered hand treasure remains legal in the
planning clone.

Immediate `DecisionResolved` question/answer verification and all
mapper-equivalent encodings remain unchanged. Outcome verification is relaxed
only for an accepted `(2, 0)` followed by a buy prompt that re-offers exactly
the treasures missing from observed `Play` events. That narrow case discards
the shadow and rebuilds from the authoritative tracker. Every deviation logs
the complete source/current question shapes, planned engine actions,
mapper-acceptable answers, observed resolution answer, retry count, and
game/turn/seat/phase context.

## Lobby and supervisor hardening (2026-07-25)

The arena now fails closed instead of silently re-clicking a stuck automatch
lobby forever. `LobbyFSM` records every state transition and lobby control
click, confirms each `Start search` click within five seconds, and emits a
warning with elapsed/cumulative time for every fruitless search. The search is
escalated with a DOM capture and `LobbyError` after either three fruitless
attempts or 600 seconds total search-without-match, so the supervisor starts a
fresh browser/profile session.

The engagement signal is recorded rather than inferred: in
`exports/arena/20260724T234015.530752Z/dom-20260724T234502.180252Z-lobby-searching-wait.html`,
the normal `Start search` button is absent and the client exposes the
`$ctrl.automatch.cancel()` **Cancel search** button (while the automatch root
also has its `searching` class). The FSM uses the Cancel search control first,
with disappearance of Start search as the immediate-table fallback.

The new knobs are `idle_watchdog_seconds` (600) at the top level and, under
`lobby`, `search_engagement_timeout_seconds` (5),
`max_fruitless_search_attempts` (3), and
`search_without_match_budget_seconds` (600). The global idle watchdog remains
armed through all non-playing phases; after 600 seconds with no normalized
game-feed event and no active game, it saves `idle-stall.png` and a DOM
snapshot, writes `exit.json` with exit code 3, and closes the browser for the
supervisor. SIGINT and SIGTERM now cancel the bounded lobby waits and let the
session close its persistent browser context before process exit.

## Login and game-server moves (2026-11-11)

Restart after three idle months, serving c29 gen 10 through
`configs/arena_c29.json` (`obs_version` 3). Two failure classes closed:

- **Expired login.** The saved browser session had expired and the homepage
  wait timed out on the login page (exit 5, run `20261111T083535.064408Z`).
  The client pre-fills the credentials it remembers, so the lobby FSM now
  clicks its Login button (`_submit_remembered_login_if_ready`): only when
  both fields are non-empty and the button is enabled, at most three times,
  15 seconds apart. It never reads or types credentials; an empty form still
  needs the operator.
- **Game-server move.** The event stream listened to socket 3 only. When the
  client is moved to the other game server between games it opens a fresh
  socket (first outbound frame msg 46), so the driver never saw the next game:
  it left the start handshake unanswered, was resigned on timeout after about
  two minutes, and exited 3 on the idle watchdog (run
  `20261111T084201.458865Z`, game 185802340, not in `record.jsonl`). The
  archives hold eight such moves, including the last July exit.
  `GameChannelSelector` (`protocol/recording.py`) now follows the channel to
  any socket whose first outbound frame is a login (msg 24) or a move (msg
  46); probe sockets (msg 45) are ignored. Tests: `test_game_channel.py`.
- **Cleanup shuffle at game end.** A game that ends on our own turn never
  crosses a turn boundary, so the final step carries the cleanup shuffle. Its
  draw can include cards from this turn's hand, plays and gains, which the
  turn-start deck/discard pool cannot supply, so the shadow rebuild raised
  "shuffle reconstruction observed unavailable 'Province'" and the driver
  exited 4 on a game it had already won 30-19 (run `20261111T090857.411949Z`,
  game 185802782, not in `record.jsonl`). `settle` now drops the shadow
  when that rebuild fails and the game is over. Test:
  `test_game_ending_on_our_turn_with_a_cleanup_shuffle_completes`.
- **Throne Room on Chapel.** The second Chapel is a new native decision with
  the same player, kind and source as the first, so the planner's batching
  loop ran on into it and answered "trash nothing" inside the first
  question's plan. The shadow reached the buy phase while the client asked
  the second Chapel question, and the driver exited 4 with "native/client
  decision kinds differ" mid-game (run `20261111T092905.220503Z`, game
  185803487, abandoned on turn 13, not in `record.jsonl`). The loop now stops
  at a pass, or when the picks left do not go down (a fresh decision). The
  same applies to Throne Room on Cellar. Test:
  `test_throne_room_chapel_plans_each_chapel_question_separately`.
- **Cellar drawing its own discards.** Cellar discarded two Coppers and three
  Estates from hand with a short deck, so the shuffle took those cards in and
  the draw returned some of them. The turn-start deck/discard pool does not
  hold cards that left the hand this turn, so the rebuild raised "shuffle
  reconstruction observed unavailable 'Estate'" and the driver exited 4
  mid-turn (run `20261111T100415.808272Z`, game 185803914, abandoned on turn
  11, not in `record.jsonl`). The next question was a clean action phase,
  which rebuilds the shadow from the tracker anyway. `settle` is now told
  when the shadow is about to be replaced (a clean action or buy question, a
  question on the opponent's turn, game end) and drops the shadow instead of
  aborting when the turn replay fails there. Test:
  `test_cellar_discards_shuffled_back_into_its_own_draw_do_not_abort`.
  Still open: the same shuffle followed by a question that is not a clean
  phase boundary (Throne Room on Cellar's second Cellar, or a Library or
  Sentry draw after cards left the hand) still aborts, because the binding
  can order the current deck but cannot rebuild a post-shuffle deck from
  cards the turn-start pool lacks.
- **Several Merchants and the first Silver.** With four Merchants in play the
  native game asks in which order to resolve their "+$1 on the first Silver"
  triggers (decision kind 9, OrderTriggers); dominion.games asks nothing. The
  buy planner plays every offered treasure in a clone of the shadow game, so
  after the first Silver the clone sat on that question, the second Silver was
  illegal, and the driver exited 4 with "offered hand treasure is not legal in
  the shadow game" (run `20261111T103516.286873Z`, game 185804954, abandoned
  on turn 13, not in `record.jsonl`). The triggers are identical, so
  `_resolve_trigger_orders` now answers those questions with the first legal
  choice after every shadow step, both in the planner and when the shadow is
  stepped after the client confirms. Test:
  `test_silver_played_with_several_merchants_resolves_trigger_order`.
  The arena plays gen 17 from this relaunch.
- **Slow opponent answering our attack.** We played Militia on our turn and
  the opponent took more than two minutes to choose their discards. The stall
  watchdog counted every silent stretch of our own turn as ours, so it closed
  the browser after 120 s with the page showing "Waiting for" the opponent
  (exit 3, run `20261111T113053.076912Z`, game 185806100, turn 19, we led
  13 VP to 6, not in `record.jsonl`). After our own `Attack` event, and until
  the next local question or turn start, the silence limit is now
  `opponent_response_watchdog_seconds` (a `run_game_loop` argument, default
  600, not a config key). The site's own clock and timeout offer cover a
  slow opponent well before that; the longer limit only keeps a dead
  connection from hanging the run. Tests:
  `test_opponent_answering_our_attack_does_not_trigger_watchdog`,
  `test_silence_after_our_attack_still_stalls_at_the_longer_limit`.
- **Restart into a game whose opponent had timed out.** The relaunch after
  the abort above reconnected to the same game 185806100 (run
  `20261111T122844.386627Z`). The server replayed the log and sent a timeout
  offer for the opponent, but no `FullState` and no `GameStart`, and the
  board rendered just after the one startup check for a retained board. The
  homepage wait then found the board and returned "resumed", but
  `queue_next_game()` dropped that answer, so `main` waited for a new game's
  `GameStart` under the one-hour in-game limit with the page showing the
  client's "Make <opponent> resign?" prompt. The idle watchdog ended it after
  600 s (exit 3). Two changes: `queue_next_game()` now returns whether a
  retained board was resumed and `_start_or_resume_game` sends that case down
  the resume path (30 s for `FullState`, then recovery); and resume recovery
  answers Yes to that prompt before leaving the table, but only when the
  prompt names exactly the player the board's ticker says it is waiting for
  (`TIMEOUT_REQUEST_CLAIM_SELECTOR`, evidence
  `dom-20261111T124345.768671Z-idle-stall.html`; a live board has no Leave
  Table control). Tests:
  `test_board_that_renders_after_the_first_check_takes_the_resume_path`,
  `test_late_board_without_full_state_recovers_to_the_search_cycle`,
  `test_resume_recovery_claims_a_timed_out_opponents_game_then_requeues`,
  `test_resume_recovery_never_claims_unless_the_board_waits_on_that_player`.
  Still open: a reconnect into a live game that is waiting on an opponent who
  has not timed out gets no `FullState` either, and recovery then exits 5.
  The arena plays gen 21 from this relaunch (12:49 UTC, run
  `20261111T124901.220957Z`). Confirmed live there: the late board took the
  resume path, no `FullState` came in 30 s, the prompt was answered, the
  server ended game 185806100 within three seconds and the next game
  (185807194) started six seconds later. The game-finished message for
  185806100 could not be decoded without a known game id, so its result (a
  win by the opponent's forced resignation, as far as the prompt implies) is
  not in `record.jsonl`.

- Ninth abort, 15:09 UTC: the server rejected an answer (exit 4, run
  `20261111T124901.220957Z`, game 185811117, turn 12, abandoned and not in
  `record.jsonl`). The bot played Throne Room with only Council Room to
  double and chose to play nothing. The driver answered the `THRONE_ROOM`
  question (COMPLEX_AND, min 0, max 2) with a bare `[]`, and 30 ms later the
  server sent message type 1. It was the first decline in 88 live Throne Room
  decisions, and `[]` had been assumed, never observed. Opponents' own client
  answers, echoed in the log, show the decline as `[0, 0]` (an empty
  selection and an empty way list): game 181422355 decision 195, 181422553
  decisions 95 and 96, 181370596 decision 94, 181481244 decision 138. Fix
  (`efc9344`): `possible_answer_indices` sends `(0, 0)` for a Throne Room
  pass, and the replay decoders read `[0, 0]` as a pass. Tests:
  `test_declining_a_throne_room_target_sends_an_empty_selection`,
  `test_recorded_throne_room_decline_replays_as_a_pass`,
  `test_live_archive_shows_the_server_rejecting_a_bare_empty_throne_answer`,
  `test_human_throne_room_decline_is_logged_as_zero_zero`. Still open: other
  optional complex questions whose decline has never been sent live may also
  carry a guessed encoding; Cellar, Chapel, Harbinger and the action, buy and
  reaction passes have been accepted. That run finished 9-4 on gen 21 before
  the abort. The arena plays gen 26 from the next relaunch.

- Tenth abort, 15:52 UTC: an inbound question did not parse (exit 4, run
  `20261111T151610.685744Z`, game 185812617, turn 11, abandoned and not in
  `record.jsonl`). The opponent played Witch while the bot held a Moat, the
  first Moat reaction the bot has been asked since July. The 281-byte
  question is eight bytes longer than the July one. The site's web client is
  now bundle 2.3.7 (the parser was written against 2.2.8), and its
  `Ability.parse` reads two more ints, `copyIndex` and `copyTotal`, after the
  optional flag of every ability offered as a question element. Fix: the
  parser reads them, and falls back to the 2.2.8 layout when a question does
  not consume cleanly, so July recordings still replay. The answer encoding
  is unchanged: opponents' Moat reveals echoed in November logs are still
  `[1, 0, 1, 0, 0]`. Tests:
  `test_moat_reaction_parses_with_and_without_ability_copy_fields`,
  `test_live_281_byte_frame_parses_as_moat_reaction`,
  `test_live_moat_question_reaches_a_legal_reveal_in_the_shadow_game`.
  Still open: nothing else in the parser has been checked against bundle
  2.3.7; paths the bot has not hit since July may have changed too. That run
  went 2-1 on gen 26 before the abort. The arena plays gen 28 from the next
  relaunch.

- Eleventh abort, 18:10 UTC: the shadow game fell behind the client (exit 4,
  run `20261111T160015.763535Z`, game 185817129, turn 13, abandoned and not
  in `record.jsonl`). The bot played Throne Room on Vassal. The first Vassal
  discarded a Moneylender and the bot chose to play it. dominion.games
  trashes Moneylender's Copper without asking and logs the answer as
  auto-played; the native interpreter asks. After a Moneylender played from
  hand the next question is a clean action or buy phase, which rebuilds the
  shadow from the tracker, so the gap never showed. Here the next question
  was the second Vassal's, mid-chain, and the forced bridge refused two
  legal native answers: "native=4 client=6 legal=(0, 251)". Fix: the bridge
  counts the Coppers the client trashed from our hand after an auto-played
  answer since the last answer, and resolves that many native Moneylender
  questions by trashing a Copper. Test:
  `test_moneylender_played_by_a_throned_vassal_follows_the_client_auto_trash`.
  Still open: any other optional choice the client resolves by itself would
  abort the same way when it lands mid-chain; Moneylender is the only one
  seen. That run went 11-2-1 on gen 28 before the abort.
- Twelfth abort, 19:33 UTC: Sentry answers were matched to cards by position
  (exit 4, run `20261111T181356.236018Z`, game 185820625, turn 13, abandoned
  and not in `record.jsonl`). The bot played Throne Room on Sentry and kept
  both cards each time. The first Sentry put Vassal and Artisan back. The
  second drew Artisan and looked at that Vassal plus a fresh Sentry. The
  shadow looks at the deck top first (Vassal, Sentry); the client offered
  (Sentry, Vassal). Normally the shadow's deck is rigged from the client's
  reveal so the two orders agree, but a card put back earlier in the same
  event window is not in the deck yet when the rig runs. The order answer was
  applied by position, the shadow put Vassal on top where the client put
  Sentry, and the next step aborted with "shadow/tracker deck compositions
  differ before rigged step". The same mismatch would have sent a trash or
  discard to the wrong card on the client had the bot not kept both. Fix:
  both decision providers read the shadow's looked-at cards from
  `decision_context()` and translate classifications and the order option
  between the shadow's order and the client's offer by card
  (`_sentry_shadow_order`). Tests:
  `test_throned_sentry_orders_cards_by_name_not_by_offer_position`,
  `test_bot_sentry_choices_reach_the_client_by_card_name`. That run went
  4-4-0 on gen 28 before the abort. Relaunched on gen 35.
- Thirteenth abort, 22:03 UTC: the bot was asked to choose for the opponent
  (exit 4, run `20261111T194423.643533Z`, game 185826880, turn 15, abandoned
  and not in `record.jsonl`). The bot played Throne Room, Throne Room, Moat,
  then Militia. The native interpreter stops for the opponent's Militia
  choice (which three cards to keep). The client never shows us that
  question, only the resulting discard, and then asked what the second Throne
  Room plays. After a plain Militia the next question is a fresh action
  phase, which rebuilds the shadow from the tracker, so the opponent's
  pending choice never mattered. Inside a Throne Room chain the shadow
  carries on. Both questions are card selections, the forced-choice bridge
  compares decision kinds only, so the shadow stayed on the opponent's
  choice, the bot picked "keep Gold" for the opponent, and the send failed
  with "select action def 2 is absent from offered elements". Nothing wrong
  was sent to the server. Fix: `_step_actions` now answers the opponent's
  responses to our own plays in the shadow (`_resolve_opponent_responses`:
  Militia keep, Moat reveal, Bureaucrat topdeck, Bandit trash), following
  the observed outcome as far as the shadow's arbitrary opponent hand
  allows, and it does so in shuffle-time turn replays too. The game loop
  also refuses to plan when the shadow's pending decision belongs to the
  other seat. Tests:
  `test_opponent_militia_discard_is_resolved_inside_a_throne_room_chain`,
  `test_opponent_response_is_stepped_only_on_our_own_turn`,
  `test_local_question_is_never_answered_from_an_opponent_decision`. All 53
  games archived on 10 and 11 November replay clean through the patched
  loop. That run went 4-8-0 on gen 35 before the abort. Relaunched on gen 39.
- Fourteenth abort, 22:55 UTC: the shadow's deck was three cards short after
  a shuffle in the middle of a Throne Room chain (exit 4, run
  `20261111T221203.193765Z`, game 185829033, turn 11, abandoned and not in
  `record.jsonl`). The bot played Throne Room on Cellar. The first Cellar
  discarded three Coppers and its draw ran the deck out, so the shuffle took
  in the whole discard pile, those three Coppers included. The shadow rebuilt
  at a shuffle starts from the turn-start deck and discard pile merged, then
  replays the turn, which leaves cards discarded earlier in the same turn in
  the shadow's discard pile: 12 in the deck and 3 in the discard pile against
  the real 15 and 0. The second Cellar's question comes mid-chain, the shadow
  is not rebuilt from the tracker, and settling it failed with "shadow/tracker
  deck counts differ before rigged step: 12 vs 15". Fix: after the replay,
  `_return_shuffled_discards` moves every card in the shadow's discard pile
  that did not arrive after the shuffle under the deck, using a new native
  primitive `return_discards_to_deck` (`state_builder`, bound as
  `Game.return_discards_to_deck`). The next rigged step orders the deck from
  what is observed. Tests:
  `test_cards_discarded_before_a_mid_turn_shuffle_return_to_the_shadow_deck`,
  `test_shuffle_rebuild_without_the_discard_return_reproduces_the_abort`, and
  the native `v2 discards return under the deck oldest copy first`. All 58
  games archived on 10 and 11 November replay clean. Still open: if the draw
  that crosses the shuffle returns a card that exists only among the cards
  just discarded, the replay cannot supply it, and mid-chain that still
  aborts (the fifth fix covers only the case where the next question
  rebuilds the shadow). That run went 5-0-0 on gen 39 before the abort.
  Relaunched on gen 43.
- Fifteenth abort, 23:11 UTC: a native engine rules bug, found by the arena
  (exit 4, run `20261111T230432.642140Z`, game 185829493, turn 14, the first
  game on gen 43, abandoned and not in `record.jsonl`). The bot played three
  Throne Rooms and then Library, and Library drew an Artisan. The shadow set
  that Artisan aside without asking and asked about the next Action instead,
  one card further down the deck ("shadow/tracker deck counts differ before
  rigged step: 12 vs 13"). Cause: `Game::step` resumed the Throne Room
  choice, which ran the interpreter up to Library's keep-or-skip question,
  and then `drive_until_decision` ran the interpreter again on the suspended
  frame. DSL ops and Sentry re-issue their decision when that happens.
  `library_step` had no case for it, reported the frame done, and Throne
  Room's repeat counter restarted Library from the top. So in the engine a
  Throne Roomed Library was resolved once, with its first drawn Action set
  aside unasked. Vassal playing a Library had the same fault, and there the
  Library simply ended with the Action stranded in the set-aside zone. Every
  self-play and eval game up to now ran with this. Fix: `Game::step` no
  longer re-runs the interpreter when the resume already stopped on an
  effect decision, and `library_step` re-asks if it is run while suspended.
  Tests: native `v2 Throne Room Library asks about the first Action it
  draws` and `v2 Vassal Library asks about the first Action it draws`, arena
  `test_throned_library_asks_about_the_first_action_it_draws`. The golden
  replays and all 201 native tests pass unchanged. The box still runs the old
  engine; shipping this needs a stop, merge, rebuild, start between
  generations.

Open: `lobby.rated` is read from the config but nothing applies it. The
account's own automatch preference decides, and the saved lobby pages from
July show it set to Rated, so every arena game so far has been rated.

Abort, 2026-11-12 08:59 UTC: game 185844266 against paradhise (our seat 1,
turn 12) stopped with `BridgeError: shuffle reconstruction observed
unavailable 'Moat'` at frame 1717. The turn was Throne Room on Cellar: the
first Cellar discarded Chapel and Moat, the draw that followed shuffled, and
the feed showed Moat and Village drawn. The tracker did not count the Moat
just discarded as part of that shuffle. The game has no ledger row and its
outcome is unknown. Archive:
`exports/arena/20261111T231744.478115Z/20261112T085916.005746Z-game-185844266`.
That session ended 24-27-2 on c29 gen 43; the arena was restarted on c30
gen 4 (`checkpoints/remote/campaign30/gen_0004.pt`) with a rebuilt engine
module.

Fix for that abort (2026-11-12), the case left open under the fourteenth
abort. The turn rebuild at a shuffle started from the last clean question
with deck and discard pile merged. The Moat was in hand at that question, so
no replay of the turn could deal it out of the merged pool. The rebuild is
gone (`_rebuild_turn_after_shuffle`, `_return_shuffled_discards`,
`_deck_source_sequence`, the turn history). The live shadow now takes the
step itself: the interpreter reshuffles at the same point as the client, with
whatever is in its discard pile by then, and the bridge only tells it which
cards come out on top. `_local_shuffle_draws` reads those from the events
(the draw after a Shuffle event still holds the old deck's last cards, which
are not part of the shuffled order), `Game.rig_shuffles` installs them, and
`Game.clear_shuffle_rig` reports whether the reshuffle happened and could
supply them. Natively this is `ShuffleRig` / `install_shuffle_rig` in
`core/interp`: the shuffle runs first, so the RNG stream is the same with or
without a rig, then the prescribed cards are swapped to the top. A failed or
unused prescription is a `BridgeError`, forgiven as before only when the next
question rebuilds the shadow from the tracker. Tests:
`test_card_discarded_from_hand_and_redrawn_over_a_shuffle_does_not_abort`,
`test_local_shuffle_draws_separate_the_old_deck_from_the_shuffled_cards`,
`test_shadow_reshuffle_that_cannot_supply_the_observed_draw_is_an_error`, and
the native `v2 shuffle rig prescribes the top of the reshuffled deck`.
`Game.return_discards_to_deck` stays bound but the arena no longer calls it.
All 435 archived games were replayed through the old and the new loop: the
only game whose outcome changed is 185844266, which no longer aborts, and
the shadow's hand, deck count and discard count agree with the tracker at
the same questions in both. Deployed at the next arena restart: the running
process holds the old module, and `build/` is rebuilt only while it is down.

Restart, 2026-11-12 18:54 UTC: the c30 gen 4 session
(`exports/arena/20261112T090914.519833Z`) was stopped on purpose with SIGINT
after game 185860037 finished, exit 0, no abort in 61 games. It ended 42-19-0
(seven of the wins were opponents leaving before turn 10). `build/` was
rebuilt, so the shuffle rig above is now live, preflight was green (197 arena
tests, dry run 3/3), and the arena was relaunched on c31 gen 6
(`checkpoints/remote/campaign31/gen_0006.pt`). Ledger at restart: 413 rows.

Restart, 2026-11-12 21:40 UTC: c31 gen 12 was promoted at 21:32, so the gen 6
session (`exports/arena/20261112T185558.028777Z`) was stopped the same way
after game 14, exit 0, no abort. It ended 8-6-0. Preflight green (197 arena
tests, dry run 3/3), no engine change, relaunched on c31 gen 12
(`checkpoints/remote/campaign31/gen_0012.pt`). Ledger at restart: 427 rows.

Abort, 2026-11-12 23:40 UTC: game 185871928 against DuckStorm (our seat 1,
turn 8) stopped with `ActuationError: MILITIA native frame selects cards to
keep: expected 3 keep actions for 1 discards, got 1`. The bot had trashed
down to a four-card deck (Chapel, Vassal, Vassal, Militia), so the opponent's
Militia asked for one discard. `BotDecisionProvider.plan` treats a question
with exactly one client answer as one native action; Militia is the exception,
because the shadow picks the cards to keep. A five-card hand never hit this
(two discards, three keeps, and the loop bound of maximum + 1 happened to be
three). Fix: Militia skips the one-answer shortcut and the loop runs for
hand size minus discards. Test
`test_militia_against_a_four_card_hand_plans_three_keeps` replays the archive
`exports/arena/20261112T214112.021802Z/20261112T233826.547837Z-game-185871928`
to the question and fails without the change. The game has no ledger row and
its outcome is unknown. That session stood at 9-5-0 on c31 gen 12 over 14
recorded games; relaunched on gen 12 at 23:42 UTC, 198 arena tests, dry run
3/3. Ledger at restart: 441 rows.

The 23:42 and 23:49 UTC relaunches both exited 5: the abandoned game was
still live on the site, the reconnect sent no FullState, and Leave Table was
not actionable (logs `arena_c29_run20.log`, `arena_c29_run21.log`). No resign
was sent. A ten-minute retry loop got in at 23:57 UTC, once the opponent had
ended that game; first new game 185872600 at 23:59 UTC, run directory
`exports/arena/20261112T235753.655460Z`.

## Status by phase

- **P0 — Recording harness: complete (2026-07-24).**
  `src/v2/arena/recon/record.py` + `src/v2/arena/browser/ws_hook.js`. A
  Playwright recorder with a persistent login profile that saves the game
  feed (both directions) as JSONL alongside periodic screenshots and DOM
  snapshots, for use as offline fixtures. Clean shutdown on Ctrl-C. Covered
  by an offline test using a local echo server (`tests/v2/arena/
  test_recon_record.py`) — no network required.

- **P1 — Feed decoder + event parser: complete (2026-07-24).**
  `src/v2/arena/protocol/` (`frames.py`, `cards.py`, `events.py`,
  `parser.py`, `recording.py`, `dump.py`). Turns the raw client feed into a
  clean, normalized stream of `GameEvent` objects (game start, turn start,
  play/buy/gain/trash/discard/draw, reveals, shuffles, resource updates,
  pending decisions, decision-resolved, game end, chat). Verified against a
  real 3-game recording:
  - **100% coverage** of game-relevant messages (21,203 of 21,203).
  - All three games decode with correct kingdoms, player names, and seat
    assignment, including a mid-game reconnect.
  - The client's full-state message is decoded directly, giving exact card,
    zone, and counter state at game start and on reconnect.
  - The public/private information boundary is handled correctly out of the
    box: our own hand decodes to exact cards, while the opponent's draws
    decode as counts only — which is precisely the information model the
    engine's search already expects.
  - Card identities map cleanly onto engine card definitions via a shared
    symbol scheme (e.g. feed `THRONE_ROOM` ↔ display "Throne Room" ↔ engine
    `def_id("Throne Room")`).
  - 16/16 tests pass (`tests/v2/arena/`).

  Full protocol notes are in the "P1 protocol findings" section of
  `docs/arena-plan.md`.

- **P2 — Public-information tracker: complete (2026-07-24).**
  `src/v2/arena/shadow/tracker.py`. Folds the P1 event stream into live
  public state: supply piles, per-seat zones (our hand exact; opponent hand
  as a count plus the derivable hand∪deck composition; discard composition;
  in-play; set-aside), the trash, resources, turn/phase, and the current
  pending-decision context, exposed as a frozen `TrackerSnapshot` — the input
  P3 turns into an engine `GameState`. Zones are tracked as known-card
  multisets plus an anonymous count, with per-seat ownership totals; any
  disagreement with a server-reported count raises a loud `TrackerError`
  (the divergence tripwire). Supporting parser extensions: the full-state
  message now also emits a `FullState` event carrying per-zone contents and
  counter values (previously discarded), and all movement events carry
  normalized from/to zone kinds; `Topdeck`, `ZoneTransfer`, `PileUpdate`,
  and `PileReorder` events were added.
  Verified golden-style over the 3-game recording
  (`tests/v2/arena/test_shadow_tracker.py`): all 16,883 fixture events fold
  with non-negative counts and per-card conservation asserted after every
  event; all four full-states reconcile exactly (including the mid-game
  reconnect, handled as a replacement resync); 300+ decisions cross-checked
  offered-cards ⊆ tracked hand. 18/18 tests pass (`tests/v2/arena/`).

- **P3 — StateBuilder + bindings: complete (2026-07-24).**
  `src/v2/core/state_builder.{h,cpp}` constructs a clean or single-interrupt
  native `GameState` from tracker public state, with exact local hand,
  re-dealt opponent hand∪deck composition, public zones/resources, supply,
  trash, turn, and phase. The four base-set interrupt frames (Moat, Militia,
  Bureaucrat, Bandit) seed the existing interpreter layouts directly.
  `game.set_deck_order()` rigs full decks in next-draw-first order and rejects
  non-permutations; `game.validate()` checks structural invariants and
  base-set per-card conservation without changing the POD `GameState` layout
  or byte hashes. `src/v2/arena/shadow/bridge.py` maps tracker display names
  and phases to the native dict binding, including deterministic resolution
  of reconnect-era anonymous public-zone slots from known ownership.
  C++ tests cover 100+ sampled organic decision states, same-action behavior,
  all interrupts, rigged draws, invalid snapshots, and deliberate corruption.
  The real-recording golden builds 930 local-turn decision points and checks
  305 action plays, 130 buys, and 94 treasure plays against native legal
  masks, with 337 internal questions and 64 pass/unmapped phase answers
  counted explicitly. The full 144-test CTest suite and the bridge golden pass.

- **P4 — Decision service: complete (2026-07-24).**
  `src/v2/arena/bot/policy.py`: the checkpoint loader and NN / parked-leaf
  NN-MCTS decision functions factored out of the web server, parameterized by
  sims, determinizations, device, and an optional per-decision wall-clock
  cap. The server imports it, so there is one serving path. Note: a mid-search
  wall-clock timeout falls back to the first legal action (the binding's
  `best_action()` is only meaningful once the search completes), so the arena
  config should treat the cap as a generous safety valve, with pacing done via
  think-time delay. Server suite passes unchanged.

## Remaining phases (from the plan)

- **P5 — Actuator + supervised game loop: offline complete (2026-07-24).**
  `src/v2/arena/actuate/` supplies a pure engine-action → client-answer
  mapper, a DOM/canvas Playwright actuator, an offline mock actuator, and
  post-action verification. `src/v2/arena/fsm/game.py` is the supervised
  asyncio loop: tracker tripwire, kingdom gate, clean phase-boundary/native
  resync, seeded opponent interrupts, policy/provider boundary, polite
  think-time delay, observed-draw rigging, native stepping, and loud
  screenshot-capable divergence abort. `config.py`, `configs/arena.json`, and
  `archive.py` provide operator settings, environment-only credentials, and
  per-game JSONL/result records.

  Offline acceptance against the reference recording:
  - All **932/932 recorded answers** are producible by the pure mapper.
    **152** questions have multiple equivalent encodings (duplicate physical
    card copies, selection order, or manual-single-treasure vs autoplay);
    these are enumerated rather than guessed.
  - All three recorded games replay to `GameEnd` with **zero divergence
    aborts**. The mock actuator submits 932 live questions: three pre-turn
    start handshakes plus 929 engine decisions. One of the 933 pending
    questions is superseded by `GameEnd` without an answer. Every engine
    decision validates its native shadow, and 259 observed-outcome steps use
    deck rigging/reconstruction.
  - A deliberately corrupted `DecisionResolved` aborts immediately and stops
    the actuator.

  Python binding workaround: `set_deck_order()` can order only the current
  deck, not discard just before an interpreter-internal shuffle. The loop
  therefore retains the clean snapshot/action history for the current effect
  segment; after observing such a shuffle it rebuilds with the known
  deck+discard pool in observed source order and replays the native actions.
  No C++ or binding changes were needed. The remaining P5 step is the human
  operator's single supervised live game, which will verify the documented
  client-2.2.8 canvas selectors.

  The supervised live runner is now wired: `browser/session.py` launches the
  persistent headful profile, installs the same WebSocket hook before client
  navigation, archives every raw frame, and streams records through the shared
  decoder/parser into the game loop without a disk round-trip. `main.py`
  supports the operator-run live command and a queue-backed fixture dry run.
  Lobby setup remains deliberately manual for this phase. The outstanding P5
  acceptance step is still the operator-observed live game.

- **First supervised live-game follow-up (2026-07-24).**
  The first live game exposed two integration bugs and was stopped by the
  divergence tripwire. The archived turn-one buy question still had three
  Coppers in hand. `DecisionSearcher(auto_play_treasures=True)` correctly
  returned its first forced canonical Copper play without requesting an NN
  evaluation, but the arena provider treated that collapse step as the final
  client decision. The provider now mirrors organic self-play: on a buy
  question it clones the shadow, plays every offered treasure in ascending
  definition order, asks the policy for a buy/pass only on that collapsed
  state, submits the client's autoplay control, and defers the selected
  buy/pass until the feed supplies the treasure-free follow-up question. This
  keeps the checkpoint's evaluated root at the training distribution (played
  treasures and credited coins) while retaining manual-treasure encodings in
  recorded replay.

  The same question also exposed a DOM-region collision. The old actuator
  searched every visible card stack by display name, so the supply Copper was
  the first match for an intended hand Copper. Client 2.2.8's saved DOM maps
  the protocol regions as follows:

  - `0:<card>`: landscape supply-pile child of `div.card-stacks`, with a
    visible pile counter.
  - `1:0:<card>`: portrait local-hand child of `div.card-stacks`, with the
    client hand z-index range 2000–2999.
  - `2:AUTOPLAY_TREASURES`: the wide (at least 3.5:1) canvas under
    `div.game-buttons`; the separate rightmost canvas is the decline/end-phase
    control.
  - Effect choices: local-hand and supply questions reuse those regions;
    revealed-card overlays use portrait display stacks at z-index 10000+.

  Targets are now resolved by region plus exact identity, and actuation stops
  if that pair does not identify exactly one physical element. Finally,
  resolution answers are checked as soon as each `DecisionResolved` event
  arrives. Previously the loop buffered all outcome events and called the
  verifier only at the next `PendingDecision`; question 3's first mismatch was
  therefore detected at question 5, making the report look one turn late.
  Repeated/batched resolution copies are each checked immediately and cannot
  delay or weaken the first check.

- **Second supervised live-game follow-up (2026-07-24).**
  The second run reached turn 9 with 17 verified autoplay/buy decisions, then
  stopped at the first opponent-turn hand prompt: Militia question 55 offered
  bare card names and the mapper produced `(2, 0, 1)`, but the actuator treated
  every unknown bare-name question as a high-z-index revealed-card display.
  The first planned click was therefore Gold in the display region; the live
  Gold was a local-hand stack at z-index 2000–2999, so resolution found zero
  targets and aborted before clicking any card.

  Bare-name hand prompts now explicitly include Militia, Chapel, Cellar,
  Poacher, Remodel trash, Artisan topdeck, the Moat reaction window, and the
  other base-set hand selectors. Duplicate offered copies remain separate
  protocol indices but deliberately produce repeated clicks on one collapsed
  physical stack. Card-body clicks use the recorded stack center, outside the
  distinct lower-right `.all-button`, so “All” cannot turn one intended click
  into an unintended bulk selection.

  Multi-select effects now always finish with the recorded wide primary canvas
  (`Confirm Discarding` / `Confirm Trashing`), including when the mapped answer
  has already reached the prompt's terminal count. Before that final click the
  actuator reads the recorded `<selection-cross>` marker and selected-stack
  counter, and requires the selected card multiset to exactly match the mapped
  labels. A mismatch waits briefly for rendering and then aborts without
  confirming. The same primary-canvas mapping handles empty decline/skip
  submissions such as declining a Moat reaction; selecting Moat itself maps to
  its hand stack and auto-submits.

  Actuation failures now report the offered tuple, the complete resolved target
  plan, the failing step, `not-found` versus `click-error`, and the observed
  target/button facts. Offline regressions cover the saved duplicate-selection
  DOM (three selected Silvers in one counted stack), confirmation versus Undo
  canvas geometry, ambiguity rejection, three repeated Copper clicks on one
  logical stack, the exact Militia plan, Moat reveal/decline plans, and replay
  of the live-game-2 archive through question 55.

- **Third supervised live-game follow-up (2026-07-24).**
  The third run reached turn 9 and correctly resolved 22 decisions before the
  Library per-drawn-Action mode prompt exposed a batching boundary error.
  The native Library interpreter presents the same player/kind/source
  signature for each successive card, but the client exposes each choice as a
  separate `CHOOSE_MODE` question with exactly one answer. The provider now
  treats `CHOOSE_MODE` and non-complex exact-one prompts as one-action client
  questions before applying its engine-signature batching heuristic. The next
  identical prompt is therefore resynced from its observed outcome and planned
  independently.

  The action mapper now rejects an action plan that produces the wrong fixed
  answer arity, so an over-batched mode plan cannot reach the browser. Offline
  regression coverage replays the archived third game through Library
  questions 64 and 65 with the live provider on both prompts, verifies the
  repeated-signature unit case, and rejects the formerly emitted two-answer
  mapping. The tracker also normalizes Library's transient `zone-type-24` as
  player set-aside state, which lets the replay continue from question 64 to
  65. Provider regressions retain batching for Militia, Cellar, Chapel, and
  Sentry trash/discard/topdeck; the existing reference and game-2 replay
  goldens cover their client mappings.

- **Fifth supervised live-game follow-up (2026-07-24).**
  Game 4 (id 181363348) completed end-to-end. Game 5 (id 181363699) then
  exposed a false-positive verifier abort at action question 92: two identical
  Witches appeared as separate offered indices but one collapsed browser stack.
  The click submitted index 0 while the client resolved index 1; both encodings
  map to the same engine Witch play and the observed `Play`/`Attack` events
  confirmed that outcome. `IntendedAction` now retains the complete set of
  mapper-enumerated encodings for the submitted gesture, and every immediate
  and settled resolution check accepts only a member of that set. The expected
  downstream `Play`/`Buy` event verification remains unchanged. Offline
  regression coverage replays the archived game with the recorded submitted
  index and continues through its available terminal frames; unit coverage
  retains aborts for an out-of-set answer and a mismatched question index.

- **Lobby automation and chat silence (2026-07-24).**
  `src/v2/arena/fsm/lobby.py` now runs the supervised inter-game flow in the
  existing headful session: homepage → search → matched table/direct game →
  game loop → game-over → dismiss end-game modal → leave table → requeue. It
  has explicit HOMEPAGE, SEARCHING, TABLE_WAITING, IN_GAME, GAME_OVER, and
  LEAVING states; all state timeouts and the per-session limit are
  config-driven (`0` means requeue until Ctrl-C). A live protocol pump stays
  alive across games while each game is handed to a fresh loop/tracker and
  receives its own archive.

  Every lobby selector is tied to the reference capture:

  - `button.automatch-button[ng-click="$ctrl.automatch.searchNow()"]` is
    the **Start search** button in `dom-34066.html` and `dom-4341034.html`.
  - `score-table-buttons button.lobby-button[ng-click="$ctrl.readyClick()"]`
    is the recorded hosted-table start control (labelled **Ready** by client
    2.2.8) in `dom-530325.html`. This remains specific to hosted-table flow;
    automatch does not depend on it.
  - `game-area` is the in-play board signal, present in mid-game
    `dom-3978828.html` and `dom-2848087.html` but absent from the table-waiting
    `dom-530325.html`. Automatch goes directly from search to this board; the
    apparent **Start game** modal there is an in-feed question, not a lobby
    control. That table snapshot does contain `game-chat`, so chat is only a
    secondary diagnostic signal.
  - `score-table-buttons button.lobby-button[ng-click="$ctrl.leave()"]` is
    **Leave Table** in `dom-530325.html`.
  - `game-ended-notification modal-window button.lobby-button[ng-click="$ctrl.ok()"]`
    is **Ok** in the blocking game-ended modal in `dom-4326014.html`. Decoded
    inbound message 14 (`GameEnd`) is the authoritative game-over signal; the
    modal is dismissed and confirmed gone before the recorded `Leave Table`
    score-page control is clicked. If that modal is absent after its short
    config-driven wait, the FSM proceeds only when `Leave Table` is already
    actionable; otherwise it raises a loud `LobbyError`, retains the browser
    for the operator, and never guess-clicks.

  The bot now has no chat-send path: the game-start announcement/config was
  removed and the kingdom gate resigns silently.  Inbound `Chat` events remain
  decoded and archived as before.

  While SEARCHING or TABLE_WAITING, the live session archives a labeled DOM
  snapshot at state entry. A terminal missing-control resolution captures a
  second labeled snapshot before raising its `LobbyError`; these use
  `page.content()` only and do not touch the WebSocket/event feed.

- **Homepage cold-start hardening (2026-07-25).**
  The 20260725T012508.958883Z lobby failure was a logged-in, connected client
  whose automatch button had not yet rendered, rather than a wrong-page or
  disconnected state. The HOMEPAGE timeout is now 90 seconds. At its halfway
  point the FSM performs one reload only if there is no visible modal content
  and no rendered reconnecting/connecting/loading component; their persistent
  empty Angular shell tags do not count. A rendered reconnecting/loading state
  is allowed to recover without a reload, while rendered `reconnecting-failed`
  triggers that single reload immediately. The original terminal DOM capture
  and loud `LobbyError` remain intact.

  The reload retains WebSocket capture without a session change: the
  `__arenaFrame` exposed function belongs to the persistent browser context,
  and the context-level `add_init_script` reruns `ws_hook.js` in the new
  document before its page scripts create sockets. Its callbacks therefore
  continue to append to the same `frames.jsonl` mirror and frame queue. P0's
  recorder uses that same context-level hook/binding arrangement and tracks
  navigation on its existing page; the streaming parser treats the new socket
  `open` as a reconnect/session reset while preserving parser state.

- **Automatch start-handshake fix (2026-07-24).**
  The stalled game 181367084 archive shows `GameStart`, then local
  `question-411` (`CHOOSE_MODE`, exactly one `card-mode-*`, exactly one
  required answer), then the opponent's resolution and `GameEnd`, with no
  `TurnStart`. Working and reference captures resolve both seats' corresponding
  questions with `(0,)` before turn 1. The game loop now recognizes the local
  prompt from that full structure while no turn exists; the literal
  `question-411` localization key is corroborating evidence, not a dependency.
  It submits option zero directly without building or searching a native
  shadow game, records the decision, and verifies the ensuing
  `DecisionResolved` immediately.

  Actuation first uses the existing `CHOOSE_MODE` canvas path. A single visible
  mode canvas is clicked directly; if other game canvases are visible, exactly
  one wide primary canvas is accepted. Failure captures a labeled live DOM
  snapshot and raises `ActuationError` with the selector, visible button facts,
  and snapshot result. The live entrypoint supplies the session DOM callback.

  Any other local `PendingDecision` before turn context now produces a
  screenshot-backed divergence abort instead of the former silent `continue`.
  Lobby automatch likewise hands off as soon as `game-area` appears and never
  waits for a text **Start game** lobby button; only the recorded hosted-table
  **Ready** path remains. The three reference games contain one local start
  handshake each, so replay actuation increases from 929 to 932 gestures while
  the 932/932 paired recorded-answer mappings remain reconciled (the one
  terminal unanswered Sentry prompt is still not actuated).

- **Multi-step gesture re-render fix (2026-07-24).**
  Two later live games reached Militia's discard prompt with the correct
  Estate/Gold + Copper + Copper + Confirm plan, but stopped on the second
  Copper. After each selection the client re-renders `div.card-stacks`; the
  remaining unselected Copper stack is briefly non-clickable and may be a new
  DOM node. The actuator now retains the ordered logical target plan for error
  reporting while resolving each physical card/button immediately before its
  click. Every resolution polls for up to three seconds by default for the
  live predicate: an unselected, clickable region-qualified card stack, or a
  present, enabled button canvas.

  Selection clicks in confirmable multi-card prompts are additionally checked
  one at a time. The actuator requires the selected-copy count for that
  identity to equal the number of preceding logical clicks, clicks the current
  unselected sibling, then waits for the count to advance by exactly one. A
  no-effect click is retried once with a fresh DOM resolution; a second
  no-effect or an over-selection aborts with the original structured
  question/plan/step diagnostics. The exact duplicate accounting comes from
  recorded `dom-4326014.html`: selected cards are separate direct stack
  siblings carrying `<selection-cross>`, and one selected Silver sibling's
  visible numeric counter encodes all three selected Silvers while the
  remaining unselected Silver is another sibling. The existing exact
  selected-multiset check remains the final gate before Confirm.

  Offline fake-page regressions cover the one-poll non-clickable animation,
  Estate + Copper + Copper + Confirm completion, a one-retry no-effect
  failure, and replacement of the Copper node between clicks. Both archived
  Militia aborts still replay to their original correct four-step logical
  plans.

- **Occluded duplicate-hand-stack follow-up (2026-07-25).**
  Game 181368037 reproduced the second-Copper Militia failure after the
  per-step polling fix. The operator screenshot established that the state was
  permanent occlusion rather than animation: the selected Copper is popped
  upward and rotated across the top of the remaining Copper stack, whose lower
  card body stays visible.

  The recorded selected-state DOM provides the exact geometry. In
  `dom-4326014.html`, the unselected Silver hand sibling is
  `111.803px × 176.4px` at
  `translateX(466.197px) translateY(536.4px) rotateZ(0deg)`, while the
  selected three-Silver sibling is the same size at
  `translateX(455.017px) translateY(492.3px) rotateZ(-10deg)`—44.1px upward
  and 11.18px left. `dom-3631985.html` records the same layout for Market:
  unselected at `(399.496px, 536.4px)` and selected at
  `(388.315px, 492.3px) rotateZ(-10deg)`. The remaining stack's distinct
  `.all-button` occupies only the lower-right rectangle
  `left:79.1071px; top:150.94px; width:30.0762px; height:21.3444px`.

  Hand-stack resolution now hit-tests center, bottom-center, bottom-left, and
  two lower-third points with `document.elementFromPoint()`. A point is usable
  only when the hit is the target stack or an ordinary descendant; the
  distinct `.all-button` is explicitly excluded. Center remains first for the
  normal path, while bottom-center is the first occlusion escape because the
  recorded 44.1px upward pop leaves the card bottom exposed and it avoids the
  lower-right All control. The chosen verified point is passed to Playwright
  as a target-relative position. If Playwright still reports interception,
  the hand-only fallback issues mouse down/up at the already hit-tested
  viewport coordinate. Failure diagnostics list every point and the topmost
  element, stack identity, and selected state found there.

  Fake-page regressions permanently cover the remaining Copper's center while
  leaving its bottom exposed, exercise the coordinate-dispatch fallback, and
  verify Estate + Copper + Copper + Confirm completes. A fully covered variant
  fails at step 3 with all point/cover identities in the structured error.

- **Undo auto-deny and bounded tracker recovery (2026-07-25).**
  The failed game 181368463 in
  `exports/arena/20260725T002942.738597Z/` established that granting an
  opponent undo rewinds server state while the arena tracker is normally
  forward-only. Server message 35 is now decoded as three big-endian 32-bit
  fields, `[metagame-kind][player-seat][decision-index]`. Client bundle 2.2.8
  defines kind 0 as an undo request, 1 as a timeout offer, 2 as undo denied,
  and 3 as undo cancelled. The three game-181368463 frames are request
  `(0, 0, 21)`, cancellation `(3, 0, 21)`, and request `(0, 0, 39)`;
  the latter is followed by the recorded outbound grant and rewound
  FullState. Message 35 no longer remains unknown in that archive.

  `undo.auto_deny` is true in `configs/arena.json`, defaults true, and rejects
  false: granting is not a supported configuration. Bundle 2.2.8 supplies the
  exact negative control
  `undo-request modal-window button.lobby-button[ng-click="$ctrl.decline()"]`.
  Recorded DOM snapshots contain the undo component shells but no rendered
  request, so the actuator also has a guarded generic-button fallback confined
  to `undo-request modal-window`. It accepts exactly one visible button whose
  `ng-click` handler or label is negative and never selects a grant/accept
  handler. Ambiguity, absence, or click failure captures a labeled DOM
  snapshot, logs a loud warning, makes no click, and leaves the request to
  expire without aborting the game. A live modal monitor similarly snapshots
  each unrecognized modal fingerprint once per visible occurrence.

  A non-replacement mid-game FullState mismatch still raises `TrackerError` by
  default. The sole exception is a mismatch within both 30 seconds and 64
  normalized events of an unresolved undo request; a decoded denial or
  cancellation closes that window immediately. That FullState uses the
  existing reconnect replacement machinery as authoritative state, clears any
  pending question, and emits resync evidence. The game loop then invalidates
  the native shadow, turn reconstruction history, pending actuation, and
  provider multi-question state before the next decision. It logs at critical
  severity and writes an explicit `UndoResync` record into the per-game
  archive. The original failed archive now replays through game end; the
  identical FullState with undo signals removed still trips the original
  zone-count mismatch.

- **Opponent timeout claims (2026-07-25).**
  The parked game in
  `exports/arena/20260725T014909.665607Z/frames.jsonl` reaches
  `TimeoutOffer(player_seat=1, decision_index=32)` and then only WebSocket
  keepalives. The cached 2.2.8 body bundle identifies the exact control as
  `timeout-request modal-window
  button.lobby-button[ng-click="$ctrl.click()"]`. Its controller calls
  `metaGameModel.requestTimeout(getResignablePlayerIndex())`; the latter calls
  `serverMessenger.timeoutRequest({decisionIndex: -1, playerIndex: seat})`.
  The wire writer emits client message 40 (`TIMEOUT_REQUEST`) with two signed
  big-endian ints, `[-1, player-seat]`.

  Arena therefore uses the stateless protocol path in the default
  `actuation_mode: "protocol"`; `ArenaSession.send_frame()` keeps the outbound
  message mirrored in `frames.jsonl`. The parser recognizes that outbound
  message as `TimeoutClaim`, rather than leaving it as `UnknownFrame`. The
  clicks fallback uses only the same evidenced `$ctrl.click()` selector; an
  absent, duplicated, or failed control captures a labeled DOM snapshot and
  logs `TIMEOUT CLAIM FAILED SAFELY` without guessing.

  `timeout.claim_grace_seconds` defaults to 30 in `configs/arena.json`.
  While the offer is outstanding, the live async event pump polls its deadline
  even though the opponent's silence is not a stall-watchdog obligation. A
  decoded later `DecisionResolved` for the offered seat proves that player
  returned during grace, so the claim is discarded quietly. A timeout signal
  for a plausible local timeout is logged at critical severity and never sends
  a self-resign. The parked archive replay advances a fake clock and proves
  that message 40 fires 30 seconds after the offer; separate offline tests
  cover the cancellation, self-timeout, click fallback, and exact wire shape.

- **Reconnect into a running game (2026-07-25).**
  Startup now checks the recorded `game-area` board signal before waiting for
  the homepage **Start search** control. If a supervisor restart retained an
  active board, the lobby enters `IN_GAME` directly and buffers the reconnect
  feed until its authoritative `FullState` arrives; that buffer is then passed
  unchanged to the normal game loop, so pending local questions and timeout
  offers follow their ordinary paths. The check is locked to
  `dom-20260725T025402.631104Z-lobby-failure-homepage-start-search-not-found.html`,
  which contains `game-area` and no automatch control. A missing or rejected
  seed is loud but recoverable: after the config-driven
  `lobby.resume_full_state_timeout_seconds` (30 seconds by default), the DOM
  is archived as `lobby-resume-recovery`, the existing game-ended **Ok** / **Leave
  Table** path runs, and the normal search cycle resumes. A plain homepage
  keeps the independent 90-second, one-reload hardening unchanged.

- **Reconnect-limit startup modal (2026-07-25).**
  The supervisor-restart failure in
  `exports/arena/20260725T032324.753710Z/` rendered its reconnect-count prompt
  inside `reconnecting-failed > modal-window`, with a `div.timeout` and the
  observed handlers `$ctrl.reconnect()` and `$ctrl.decline()`. The initial
  implementation checked only once before the homepage wait; the later
  `20260725T033239.293293Z` capture showed that the modal can render after the
  loading page instead. Every homepage, search, table, and post-Leave-Table
  poll now rechecks the modal, board, and actionable control before sleeping.
  The default `lobby.reconnect_limit_policy` is
  `"return_to_lobby"`, which clicks the latter and resumes the ordinary
  homepage search; the documented alternative `"reconnect"` uses the former
  and then re-evaluates the retained-game path.

  The reconnect-limit decline selector is
  `reconnecting-failed modal-window:has(div.timeout)
  button.lobby-button[ng-click="$ctrl.decline()"]`, deliberately distinct
  from the undo selector `undo-request modal-window
  button.lobby-button[ng-click="$ctrl.decline()"]`. The evidence test locks
  both reconnect-limit button labels to the captured DOM; the undo evidence
  remains the separately scoped `undo-request` component documented above.
  Any other visible text-bearing `modal-window` during startup is archived as
  `lobby-failure-startup-unknown-modal` and raises a `LobbyError` containing
  its quoted visible text instead of waiting opaquely for the homepage timeout.

- **Overnight mode (2026-07-24).**
  The live process is now restart-safe for an external supervisor. The game
  loop runs a configurable stall watchdog (`stall_watchdog_seconds`, default
  120). It is armed only while the local client owes progress: either the
  current turn belongs to our seat, or a local `PendingDecision` remains
  unresolved (including opponent-turn attacks and the pre-game start
  handshake). Ordinary opponent-turn silence does not arm it. A timeout
  captures a screenshot and DOM snapshot and writes `stall.json` in the game
  archive with the frame index, last game-relevant event timestamp, elapsed
  silence, tracker summary, and pending-question detail.

  Server message 14 now emits a result-bearing `GameResult` end event instead
  of a bare `GameEnd`. Client bundle 2.2.8 defines its layout as:

  ```
  GameFinished =
    TableDetails
    GameResult {
      tableId: long
      gameId: long
      ratingType: optional enum
      emptyPiles: CardName[]
      playerResults: PlayerResult[]
      autoContinue: boolean
    }
    continueAllowed: boolean
    matchCompleted: boolean

  PlayerResult =
    playerId: int
    rank: int
    score: Score { totalPoints: int, usedTurns: int, parts: ScorePart[] }
    cardNames: CardFrequency[]
    resignIndex: int
    resignationType: optional enum
  ```

  The parser consumes the result tail exactly, maps player IDs back to seat
  order, and reports per-seat VP, placings, the unique winning seat, or a tie.
  Rank is authoritative: archive
  `exports/arena/20260724T220259.308785Z/` decodes games as:

  - 181364095: seats 0–1 scored 33–44, ranks 2–1 (our win).
  - 181364271: 25–39, ranks 2–1 (our loss).
  - 181364518: 27–28, ranks 2–1 (our loss).
  - 181364739: 3–3, ranks 2–1 (our win by the server result).
  - 181364786: 3–3, ranks 1–2 (our loss by the server result).

  All five VP tuples match the final per-seat `points` counters in the feed.
  Archive `exports/arena/20260724T214053.394532Z/` additionally decodes
  completed game 181363348 as 45–30, ranks 1–2 (our seat 1 lost); its following
  game 181363699 was the documented actuation abort and is not added to the
  completed-game ledger. If a future message-14 payload is unfamiliar, the
  parser logs an error, emits an unknown result while preserving the end
  marker, and never lets ledger bookkeeping crash the session.

  Each completed game's `result.json` includes our seat, opponent, outcome
  (`win`, `loss`, `tie`, or `unknown`), scores, placings, winning seat, and tie
  flag. Completion also appends one compact JSON object to the cross-session
  `exports/arena/record.jsonl`; the session prints its running W-L-T tally.

  Process exit codes are a supervisor-facing contract:

  | Code | Meaning |
  | ---: | --- |
  | 0 | Clean end: configured game cap reached, or Ctrl-C |
  | 1 | Unexpected crash/startup failure |
  | 3 | Stall-watchdog abort |
  | 4 | Tracker, bridge, verification, mapping, or actuation divergence |
  | 5 | Lobby FSM error/timeout |

  Every non-zero live path first writes one-line `exit.json` in the run
  directory (`exit_code`, `exit_reason`, `game_id`, and `archive_dir`) and then
  reaches `ArenaSession.stop()`, which closes the persistent browser context
  and Playwright before process exit. Divergence and lobby paths no longer
  wait indefinitely with the profile locked.

  `configs/arena.json` now sets `max_games_per_session` to `0`, retaining the
  established meaning “unlimited until Ctrl-C.” `scripts/arena_overnight.sh`
  launches the process under `caffeinate -is`; restart looping remains the
  responsibility of the outer orchestrator.

- **P6 — Lobby loop + supervisor: complete for local overnight running
  (2026-07-25).** `src/v2/arena/fsm/lobby.py` drives the full unattended
  cycle — Start Search, automatch, the in-feed start handshake, game-ended
  OK, Leave Table, requeue — with bounded re-search, search-engagement
  verification, blocking-modal handling, an idle watchdog, and per-state
  logging. `scripts/arena_overnight.sh` runs it under `caffeinate` with
  unlimited games. Restart-on-exit is the orchestrator's job (exit-code
  contract below). The planned Hetzner deployment was dropped: the operator
  runs sessions locally overnight instead.

## First live results (2026-07-24/25)

The first unattended overnight session played real rated-off base-set games
against humans. Running record is appended to `exports/arena/record.jsonl`;
after the first night it stood at 6W–14L–1T over 21 completed games.

Every live abort during that night became a permanent regression test built
from its own archive. The failure classes found and closed, in order:

1. Off-distribution buy decisions (collapse-trained checkpoint asked to act
   with unplayed treasures) — fixed by collapsing treasures before search.
2. Click-region confusion (buy pile vs hand card vs autoplay control).
3. Stacked-hand prompts (duplicate cards share one stack; confirm button).
4. Mode-question over-batching (Library re-presents an identical signature).
5. Encoding equivalence (duplicate copies: server's answer differs from ours
   but means the same move) — verifier now accepts the enumerated set.
6. Click timing/occlusion on multi-select prompts — ultimately retired
   wholesale by switching game answers to the protocol path (see above).
7. Lobby: cold-start render, the in-feed start handshake (the "Start game"
   prompt is a feed question, not a lobby button), reconnect-limit modal,
   custom-element visibility, silent re-search loop, idle hang.
8. Cross-game state leak (deferred buy surviving a game that ended during
   think-time).

## How to reproduce P1 results

```sh
# from repo root, with the venv and engine bindings built
./.venv/bin/python -m pytest tests/v2/arena/
# human-readable decode of a recording:
./.venv/bin/python -m src.v2.arena.protocol.dump arena-recordings/<run>/frames.jsonl
```

## Notes / carryovers

- Recordings and the browser profile live under `arena-recordings/` and
  `arena-profile/` (both gitignored). The current 3-game recording is the
  reference fixture for P1–P2 tests.
- Card and pile identities in the feed are numeric; the mapping to names is
  generated from the client's published card table (documented in
  `cards.py`) and regenerated when the client version changes.
- All arena work is committed on branch `v2-phase1`. Operator instructions
  for running live sessions are in `docs/arena-usage.md`.
- Per-game archives (`exports/arena/<session>/<ts>-game-<id>/`) keep
  `frames.jsonl`, `events.jsonl`, `decisions.jsonl`, and `result.json`; each
  is a replayable fixture, and past incidents are cited as test evidence by
  path. Do not delete `exports/arena/` runs referenced by tests.
- The web-server pytest suite intermittently deadlocks on this Mac inside
  torch's OpenMP threads at import (predates the arena stream; hung runs from
  July 10/14 observed). Workaround: `OMP_NUM_THREADS=1` when running
  `src/v2/web/server/test_server.py`.
