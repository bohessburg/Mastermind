# Unified game records

Unified game records are a third, analysis-only format. They do not replace or
change either source format:

- Local web exports remain seed replays: `exports/*.json`.
- Arena archives remain the raw and normalized evidence:
  `exports/arena/<session>/<game>/`.

The unified format is a self-describing JSON object with a flat ordered
`records` list. It is intended for `jq`, pandas, review tools, and backfills,
not for restoring an engine game.

## Creating records

Convert all local web exports:

```sh
OMP_NUM_THREADS=1 PYTHONPATH=build ./.venv/bin/python \
  -m src.v2.records.convert exports/*.json --out exports/records/local
```

Convert every arena game below every session:

```sh
OMP_NUM_THREADS=1 PYTHONPATH=build ./.venv/bin/python \
  -m src.v2.records.convert exports/arena --out exports/records/arena
```

The CLI also accepts one local JSON file, one arena game directory, one arena
session directory, or several mixed paths. Without `--out`, a local source
gets `<stem>.game-record.json` beside it and an arena source gets
`game-record.json` inside its game directory.

Directory/glob discovery selects complete games: local files must replay
legally through game end and match `final_state_hash`, and arena directories
must contain `result.json`. This excludes in-progress zero-action web snapshots
and abandoned partial arena directories from a backfill instead of presenting
them as completed games.

Future completed arena archives emit `game-record.json` automatically after
`result.json`. Emission is best effort: failure is logged but cannot interrupt
or invalidate the canonical arena archive. The web server's local export
writer is unchanged; use the local one-liner above.

## Top-level schema

Schema version `1.0` has these fields:

The machine-readable Draft 2020-12 schema is
`src/v2/records/game-record.schema.json`; the Python dataclasses and stricter
cross-field validation live in `src/v2/records/model.py`.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | Currently `"1.0"`. |
| `source` | string | `"local"` or `"arena"`. |
| `provenance` | string | Original export path or arena game directory. |
| `game_id` | string | Local session/export stem or arena game id. |
| `timestamp` | ISO-8601 string or null | Arena `GameStart` time. Local exports contain no game time. |
| `timestamp_visibility` | visibility | Whether `timestamp` was present in the source. |
| `kingdom` | card reference[] | Kingdom cards as both `def_id` and `name`. |
| `player_count` | integer | Number of seats. |
| `seats` | seat[] | Zero-based seat identity and controller metadata. |
| `obs_version` | integer or null | Local export observation version. Arena archives do not record it. |
| `obs_version_visibility` | visibility | `"known"` for a present local value, otherwise `"unknown"`. |
| `controlled_seat` | integer or null | Our arena seat. Local exports have no single controlled seat. |
| `controlled_seat_visibility` | visibility | Whether `controlled_seat` is known. |
| `results` | result[] | VP, placing, and outcome in seat order. |
| `records` | record[] | Ordered body described below. |

A card reference always has both fields:

```json
{"def_id": 15, "name": "Remodel"}
```

A seat has `index`, `kind`, `display_name`, `controlled`, and `bot`. Local
`kind` is copied unchanged, such as `human`, `bot:engine3`, or a checkpoint
bot string. Arena identifies our disclosed driver as `bot:nnmcts` and other
seats as `human`; display names come from `GameStart`.

A result has:

```json
{"seat": 0, "visibility": "known", "vp": 38, "placing": 1, "outcome": "win"}
```

When an archive has no decoded standings, all three result values are unknown:

```json
{"seat": 0, "visibility": "unknown", "vp": null, "placing": null, "outcome": "unknown"}
```

## Ordered records

Every body item has a sequential `index` and the original zero-based
`source_index`. The latter is an action index for local exports and an
`events.jsonl` row index for arena archives.

Context and action fields deliberately pair a value with visibility:

| Fields | Meaning |
| --- | --- |
| `record_type`, `event` | Stable broad class and source event name. |
| `timestamp_ms`, `timestamp_ms_visibility` | Arena event time; explicitly unknown for local actions. |
| `turn_number`, `turn_number_visibility` | One-based turn number for that seat. |
| `active_seat`, `active_seat_visibility` | Seat whose turn is in progress. |
| `phase`, `phase_visibility` | `action`, `buy`, `night`, `cleanup`, or unknown. |
| `actor_seat`, `actor_seat_visibility` | Seat making the decision/action, which can differ during reactions. |
| `engine_action_ids`, `engine_action_ids_visibility` | Zero or more engine action ids. |
| `action_labels`, `action_labels_visibility` | Labels aligned to action ids when both are known. |

Local records have `record_type: "action"` and `event: "EngineAction"`, one
per source action. Arena body rows retain meaningful normalized events:
`FullState`, `TurnStart`, `PendingDecision`, `Play`, `Buy`, `Gain`, `Trash`,
`Discard`, `Draw`, `Reveal`, `Topdeck`, `ZoneTransfer`, `Shuffle`,
`ResourceUpdate`, `PileReorder`, attacks, reactions, and game end. Repeated
protocol acknowledgements and redundant pile-top updates are not copied.

For arena, one observable server action can therefore be followed by separate
consequence rows. For example, `Buy Silver` is followed by `Gain Silver`;
their source ordering and timestamps preserve the relationship without
claiming that an unseeded remote engine action was replayed.

Each record also has five card consequence groups:

- `played`
- `bought`
- `gained`
- `trashed`
- `discarded`

Each group is `{visibility, count, cards}`. `cards` contains card references
only when identities were observed. Fields not evidenced by that arena event
are unknown, not zero.

`resources_after` is `{visibility, actions, buys, coins}`.
`zone_counts_after` is a flat list of `{seat, zone, visibility, count}` for
hand, deck, discard, in-play, set-aside, and global trash.

## Visibility and knowability

There are exactly three visibility values:

- `known`: the value and, for card groups, every card identity are known.
- `counts_only`: the numeric count is known but card identities are hidden.
- `unknown`: the source supplies no trustworthy value; scalar values are
  null and arrays are empty.

The rules are intentionally asymmetric:

| Information | Local seed replay | Arena archive |
| --- | --- | --- |
| Schema/source/provenance/game id/kingdom/seats | Always populated | Always populated when `GameStart` exists |
| Timestamp | Unknown (not in export) | Known from `GameStart`, otherwise unknown |
| Observation version | Known when present in export | Unknown (not archived) |
| Controlled seat | Unknown/not applicable | Known when decoded |
| Bot seats | Copied from local seat kinds | Our controlled seat is `bot:nnmcts` |
| Actions and labels | Known for every engine action | Known for observed plays/buys and our archived engine decisions; otherwise unknown |
| Played/bought/gained/trashed/discarded cards | Derived exactly from replay-visible engine state | Only identities and counts present in normalized events |
| Resources | Known after every local action | Known only after source counters establish all three values |
| Our arena hand/deck zone counts | N/A; every local seat count is known | Known |
| Opponent arena hand/deck | Fully recoverable locally | **Always `counts_only`; identities never appear** |
| Arena missing standings or incomplete context | N/A | Explicitly unknown |

No arena seed exists. The converter never runs `new_game`, determinizes an
opponent, or fills an unknown with a plausible card, resource, phase, score,
or action.

## Abridged examples

Local:

```json
{
  "schema_version": "1.0",
  "source": "local",
  "game_id": "1jMv_xAgobIUlMpC",
  "timestamp": null,
  "timestamp_visibility": "unknown",
  "kingdom": [{"def_id": 15, "name": "Remodel"}],
  "obs_version": 2,
  "obs_version_visibility": "known",
  "controlled_seat": null,
  "controlled_seat_visibility": "unknown",
  "records": [{
    "index": 0,
    "source_index": 0,
    "record_type": "action",
    "event": "EngineAction",
    "turn_number": 1,
    "turn_number_visibility": "known",
    "engine_action_ids": [0],
    "engine_action_ids_visibility": "known",
    "action_labels": ["Pass"],
    "action_labels_visibility": "known",
    "played": {"visibility": "known", "count": 0, "cards": []}
  }]
}
```

Arena:

```json
{
  "schema_version": "1.0",
  "source": "arena",
  "game_id": "181376119",
  "timestamp": "2026-07-25T06:22:30.203000+00:00",
  "timestamp_visibility": "known",
  "controlled_seat": 1,
  "controlled_seat_visibility": "known",
  "records": [{
    "event": "Play",
    "turn_number": 1,
    "turn_number_visibility": "known",
    "active_seat": 1,
    "active_seat_visibility": "known",
    "engine_action_ids": [1],
    "engine_action_ids_visibility": "known",
    "action_labels": ["Play Copper"],
    "played": {
      "visibility": "known",
      "count": 1,
      "cards": [{"def_id": 0, "name": "Copper"}]
    },
    "zone_counts_after": [
      {"seat": 0, "zone": "hand", "visibility": "counts_only", "count": 5},
      {"seat": 0, "zone": "deck", "visibility": "counts_only", "count": 5}
    ]
  }]
}
```

The examples omit unchanged required fields for readability; converter output
always contains the full shape.

## `jq` examples

Opening buys from one record:

```sh
jq -r '
  .game_id as $g
  | .records[]
  | select(.event == "Buy" and .turn_number <= 2)
  | [$g, .turn_number, .actor_seat, .bought.cards[].name]
  | @tsv
' game-record.json
```

Average turns across a directory (the maximum seat turn number in each game):

```sh
jq -s '
  map([.records[].turn_number | select(. != null)] | max)
  | add / length
' exports/records/*/*.json
```

Win rate by kingdom card, using the controlled arena seat:

```sh
jq -s '
  [
    .[]
    | select(.controlled_seat != null)
    | . as $g
    | $g.kingdom[].name as $card
    | {
        card: $card,
        win: ($g.results[$g.controlled_seat].outcome == "win")
      }
  ]
  | group_by(.card)
  | map({
      card: .[0].card,
      games: length,
      win_rate: (map(select(.win)) | length) / length
    })
' exports/records/arena/*.json
```

## pandas examples

```python
import glob
import json
import pandas as pd

games = [json.load(open(path)) for path in glob.glob("exports/records/**/*.json")]

headers = pd.DataFrame(games)
kingdoms = (
    headers[["game_id", "source", "kingdom"]]
    .explode("kingdom")
    .assign(card=lambda frame: frame["kingdom"].map(lambda card: card["name"]))
)

results = pd.json_normalize(
    [
        {"game_id": game["game_id"], "source": game["source"], **result}
        for game in games
        for result in game["results"]
    ]
)

records = pd.json_normalize(
    [
        {"game_id": game["game_id"], "source": game["source"], **record}
        for game in games
        for record in game["records"]
    ],
    sep=".",
)

average_turns = records.groupby("game_id")["turn_number"].max().mean()
opening_buys = records[
    (records["event"] == "Buy") & (records["turn_number"] <= 2)
]

controlled = {
    game["game_id"]: game["controlled_seat"]
    for game in games
    if game["controlled_seat"] is not None
}
arena_results = results[
    results.apply(lambda row: controlled.get(row.game_id) == row.seat, axis=1)
]
arena_results["won"] = arena_results["outcome"].eq("win")
win_rate_by_card = kingdoms.merge(arena_results, on=["game_id", "source"]).groupby(
    "card"
)["won"].mean()
```
