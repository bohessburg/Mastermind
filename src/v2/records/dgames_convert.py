"""Convert Dominion.games spectator captures into visibility-aware tuples.

The spectator protocol normally exposes both players' hands in the initial
``fullGameState``.  A hand can become anonymous later, though, so this module
tracks the *visible* side of every ``CardMove`` rather than assuming that an
initially visible hand stays visible.  It exports ordinary public buy rows for
the complete semantic history, then adds action and recovered choice rows only
where the relevant live hand/reveal transition is actually observable.

No spectator ``questionAsked`` frame is available.  Choice labels are therefore
derived from the observed state transition and validated against the native
engine's action mask.  Unknown deck/discard identities still require a
deterministic snapshot allocation; every shard row carries an explicit
observation-quality flag so that allocation is never represented as observed
detail.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeAlias

import numpy as np

import dominion_v2_py as dz

from src.v2.arena.protocol.events import FullState, GameStart, ResourceUpdate
from src.v2.arena.protocol.frames import DecodedFrame, Direction, ProtocolError, Reader
from src.v2.arena.protocol.parser import ArenaParser
from src.v2.arena.protocol.recording import decode_record_binary


OBS_VERSION = 3
OBS_WIDTH = 1788
ACTION_WIDTH = 357
A_PASS = 0
A_PLAY_BASE = 1
A_BUY_BASE = 206
A_SELECT_BASE = 251
A_OPTION_BASE = 292
DEFAULT_ALPHA = 0.6
DEFAULT_SCALE = 20.0
DEFAULT_RAW_ROOT = Path("data/dominion_games/raw")
DEFAULT_OUTPUT_DIR = Path("exports/tuples_dgames")
DEFAULT_CARD_MAP = Path("data/dominion_games/recon/card_id_map.json")
SOURCE_TAG = "dominion.games_spectator_visibility_aware_full_decisions_v2"

# Per-row provenance saved in the tuple shards.  ``FULLY_OBSERVED`` means the
# actor's decision-relevant hand or revealed cards were known from the live
# protocol transition.  It deliberately does *not* claim the normally hidden
# draw pile was observed.  ``PARTIALLY_INFERRED`` rows retain the previous
# public-buy reconstruction and any choice whose offered private zone was not
# fully visible.
OBSERVATION_PARTIALLY_INFERRED = 0
OBSERVATION_FULLY_OBSERVED = 1

DECISION_TYPE_IDS = {
    "buy": 0,
    "buy_pass": 1,
    "action_play": 2,
    "treasure_play": 3,
    "action_pass": 4,
    "militia_keep": 5,
    "cellar_discard": 6,
    "chapel_trash": 7,
    "poacher_discard": 8,
    "harbinger_topdeck": 9,
    "sentry_option": 10,
    "sentry_order": 11,
    "throne_room_target": 12,
    "remodel_trash": 13,
    "remodel_gain": 14,
    "mine_trash": 15,
    "mine_gain": 16,
    "artisan_gain": 17,
    "artisan_topdeck": 18,
    "vassal_option": 19,
    "library_option": 20,
    "bandit_trash": 21,
}
DECISION_TYPE_NAMES = tuple(name for name, _value in sorted(DECISION_TYPE_IDS.items(), key=lambda item: item[1]))

_TRACKED_LIVE_ZONE_KINDS = frozenset({"hand", "deck", "discard", "in-play", "set-aside"})
_TREASURE_NAMES = frozenset({"Copper", "Silver", "Gold"})
_ACTION_EFFECT_NAMES = frozenset(
    {
        "Cellar",
        "Chapel",
        "Poacher",
        "Harbinger",
        "Sentry",
        "Throne Room",
        "Remodel",
        "Mine",
        "Artisan",
        "Vassal",
        "Library",
        "Bandit",
        "Militia",
    }
)
# Vassal presents its yes/no choice only when the discarded card is an
# Action.  Keep this separate from _ACTION_EFFECT_NAMES: the latter is the
# smaller set whose follow-up choices this converter knows how to replay.
_ACTION_CARD_NAMES = frozenset(
    {
        "Artisan",
        "Bandit",
        "Bureaucrat",
        "Cellar",
        "Chapel",
        "Council Room",
        "Festival",
        "Harbinger",
        "Laboratory",
        "Library",
        "Market",
        "Merchant",
        "Militia",
        "Mine",
        "Moat",
        "Moneylender",
        "Poacher",
        "Remodel",
        "Sentry",
        "Smithy",
        "Throne Room",
        "Vassal",
        "Village",
        "Witch",
        "Workshop",
    }
)

# This converter deliberately supports the collector's base-only captures.
# The 33 cards in card_id_map.json are the full supported base card vocabulary.
BASE_CARD_LIMIT = 33
STARTING_DECK = Counter({"Copper": 7, "Estate": 3})

# The semantic log names used here have been validated against the complete
# 181648216 spectator capture.  They are semantic-log identifiers, not raw
# decision indices.
LOG_GAIN = 0
LOG_BUY_GAIN = 2
LOG_TRASH = 3
LOG_DISCARD = 4
LOG_PLAY = 6
LOG_TOPDECK = 9
LOG_DRAW = 10
LOG_REVEAL = 17
LOG_STARTING_CARDS = 29
LOG_BUY = 30
LOG_TURN_DESCRIPTION = 36
LOG_SHUFFLE = 35
LOG_COIN_BONUS_ONE = 151
LOG_COIN_BONUS = 152
# These are the card-attributed variants used by Merchant's first-Silver
# trigger.  They carry the same player and numeric amount arguments as the
# generic +Coin entries above, plus the originating card.  The server emits
# them once per resolving Merchant (and can coalesce multiple triggers), so
# treating them as a semantic no-op silently undercounts live coin totals.
LOG_CARD_COIN_BONUS_ONE = 153
LOG_CARD_COIN_BONUS = 154
LOG_TREASURES = 161
LOG_BUY_BONUS = 147

# ``CardNames.BACK`` is what a spectator semantic entry uses when the card
# identity would reveal a card that was in a hidden zone.  It is a real
# protocol value, not a card that can be put in an engine snapshot.
CARD_BACK_WIRE_ID = 0
STARTING_HAND_SIZE = 5

# Public semantic entries are nested under the card whose effect emitted
# them.  These tables are the complete base-set zone semantics relevant to a
# buy-time snapshot.  Keeping them explicit makes an unsupported card effect
# fail closed instead of accidentally treating a deck/set-aside movement as a
# hand movement.
HAND_TRASH_EFFECTS = frozenset({"Chapel", "Moneylender", "Mine", "Remodel"})
REVEALED_TRASH_EFFECTS = frozenset({"Bandit", "Sentry"})
HAND_DISCARD_EFFECTS = frozenset({"Cellar", "Militia", "Poacher"})
REVEALED_DISCARD_EFFECTS = frozenset({"Bandit", "Library", "Sentry", "Vassal"})
DISCARD_GAIN_EFFECTS = frozenset({"Bandit", "Remodel", "Witch", "Workshop"})
HAND_GAIN_EFFECTS = frozenset({"Artisan", "Mine"})
TOPDECK_GAIN_EFFECTS = frozenset({"Bureaucrat"})
HAND_TOPDECK_EFFECTS = frozenset({"Artisan", "Bureaucrat"})

# The supported domain is exactly the 2E base-card vocabulary.  Buy-resource
# updates are not represented in the semantic log, so retain the authoritative
# coin costs here when a turn has more than one buy.
BASE_BUY_COSTS = {
    "Curse": 0,
    "Copper": 0,
    "Silver": 3,
    "Gold": 6,
    "Estate": 2,
    "Duchy": 5,
    "Province": 8,
    "Cellar": 2,
    "Chapel": 2,
    "Village": 3,
    "Smithy": 4,
    "Workshop": 3,
    "Remodel": 4,
    "Mine": 5,
    "Merchant": 3,
    "Militia": 4,
    "Witch": 5,
    "Moat": 2,
    "Bureaucrat": 4,
    "Market": 5,
    "Festival": 5,
    "Laboratory": 5,
    "Gardens": 4,
    "Moneylender": 4,
    "Poacher": 4,
    "Vassal": 3,
    "Harbinger": 3,
    "Throne Room": 4,
    "Council Room": 5,
    "Artisan": 6,
    "Bandit": 5,
    "Library": 5,
    "Sentry": 5,
}

CounterInt: TypeAlias = Counter[int]
LogArgument: TypeAlias = tuple[int, object]


class ConversionError(ValueError):
    """A capture cannot safely be converted into public buy tuples."""


class FinalDeckMismatch(ConversionError):
    """The public ownership reconstruction disagrees with GameResult truth."""

    def __init__(self, message: str, deck_matches: tuple[bool, ...]) -> None:
        super().__init__(message)
        self.deck_matches = deck_matches


@dataclass(frozen=True)
class CardMap:
    """Bidirectional card mapping supplied by the recon artifact."""

    wire_to_def: dict[int, int]
    name_to_def: dict[str, int]
    def_to_name: dict[int, str]


@dataclass(frozen=True)
class SemanticLogEntry:
    """A final semantic-log entry, after server replacements at one index."""

    index: int
    name: int
    depth: int
    arguments: tuple[LogArgument, ...]


@dataclass(frozen=True)
class ObservedBuyResources:
    """Authoritative live counters captured when a Buy log first appears."""

    seat: int
    coins: int
    buys: int


@dataclass(frozen=True)
class CardArgument:
    """Known card identities plus redacted spectator card-back slots."""

    known: CounterInt
    hidden_count: int

    @property
    def total(self) -> int:
        return _counter_total(self.known) + self.hidden_count


@dataclass(frozen=True)
class LiveCardMove:
    """One post-snapshot CardMove with both endpoint visibility lists.

    ``source_cards`` is the identity list visible before the move and
    ``destination_cards`` is the list visible afterwards.  ``None`` denotes
    the protocol's ``-1`` redaction sentinel; it is never resolved from the
    parallel raw list or from a synthetic allocation.
    """

    event_index: int
    movement: str
    seat: int | None
    from_zone: str
    to_zone: str
    from_zone_index: int
    to_zone_index: int
    source_cards: tuple[str | None, ...]
    destination_cards: tuple[str | None, ...]
    count: int


@dataclass(frozen=True)
class LiveResourceUpdate:
    event_index: int
    seat: int | None
    resource: str
    value: int


@dataclass(frozen=True)
class LiveTurnDescription:
    event_index: int
    seat: int
    turn_number: int
    turn_type: int
    controller_seat: int


@dataclass(frozen=True)
class LiveShuffle:
    event_index: int
    seat: int
    included_discard: bool


LiveEvent: TypeAlias = LiveCardMove | LiveResourceUpdate | LiveTurnDescription | LiveShuffle


@dataclass(frozen=True)
class LiveSegment:
    """One FullState boundary and only the deltas that follow that snapshot."""

    full_state: FullState
    initial_log_entries: tuple[SemanticLogEntry, ...]
    live_events: tuple[LiveEvent, ...]


@dataclass(frozen=True)
class ProtocolCapture:
    """Protocol-decoded material needed for one capture reconstruction."""

    raw_path: Path
    game_start: GameStart
    full_state: FullState
    log_entries: tuple[SemanticLogEntry, ...]
    initial_log_entries: tuple[SemanticLogEntry, ...]
    live_events: tuple[LiveEvent, ...]
    live_segments: tuple[LiveSegment, ...]
    observed_buy_resources: dict[int, ObservedBuyResources]


@dataclass(frozen=True)
class OutcomeSeat:
    """Terminal labels for one game seat from the GameResult manifest."""

    player_id: int
    player_name: str
    rank: int
    score: int
    turns_used: int
    final_deck: CounterInt
    resigned: bool


@dataclass(frozen=True)
class Outcome:
    """All terminal labels and deck ground truth for a complete game."""

    seats: tuple[OutcomeSeat, ...]
    winner: int | None
    margin_valid: bool


@dataclass(frozen=True)
class PublicBuyRow:
    """One encoded position and an action validated by the native mask."""

    obs: np.ndarray
    legal: np.ndarray
    action: int
    seat_index: int
    ply_index: int
    turn_number: int
    source_log_index: int
    coins: int
    buys: int
    score: int
    decision_type: str
    observation_quality: int
    source_event_index: int


@dataclass(frozen=True)
class VisibilitySummary:
    """Initial and sustained hand visibility for a two-seat capture."""

    initial_visible: tuple[bool, ...]
    visible_throughout: tuple[bool, ...]
    first_unknown_event: tuple[int | None, ...]
    grade: str
    segment_count: int = 1


@dataclass(frozen=True)
class RecoveredLiveRows:
    """Rows reconstructed from the post-snapshot CardMove stream."""

    rows: tuple[PublicBuyRow, ...]
    visibility: VisibilitySummary
    skipped_by_type: dict[str, int]
    militia_examples: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ConvertedGame:
    """Successful in-memory conversion, retained until all gates pass."""

    game_id: str
    source_path: Path
    rows: tuple[PublicBuyRow, ...]
    outcome: Outcome
    deck_matches: tuple[bool, ...]
    per_seat_decisions: tuple[int, ...]
    observed_resource_buy_rows: int
    player_names: tuple[str, ...]
    kingdom: tuple[str, ...]
    visibility: VisibilitySummary
    recovered_skips: dict[str, int]
    militia_examples: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class QuarantineEntry:
    """A non-emitted capture and the reason it was retained only for audit."""

    game_id: str
    manifest_path: Path
    raw_path: Path | None
    category: str
    reason: str
    capture_status: str | None
    deck_matches: tuple[bool, ...] | None = None

    def as_json(self) -> dict[str, object]:
        """Return a stable JSON-compatible audit record."""

        result: dict[str, object] = {
            "game_id": self.game_id,
            "manifest_path": self.manifest_path.as_posix(),
            "raw_path": None if self.raw_path is None else self.raw_path.as_posix(),
            "category": self.category,
            "reason": self.reason,
            "capture_status": self.capture_status,
            "emitted": False,
        }
        if self.deck_matches is not None:
            result["deck_matches"] = list(self.deck_matches)
        return result


class _ShardAccumulator:
    """Write standard tuple shards plus spectator-specific outcome arrays."""

    def __init__(self, output_dir: Path, shard_size: int) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.output_dir = output_dir
        self.shard_size = shard_size
        self._rows: list[tuple[PublicBuyRow, Outcome, int]] = []
        self.paths: list[Path] = []
        self.total_rows = 0

    def append(self, row: PublicBuyRow, outcome: Outcome, game_index: int) -> None:
        """Queue one row after its game has passed every correctness gate."""

        self._rows.append((row, outcome, game_index))
        self.total_rows += 1
        if len(self._rows) >= self.shard_size:
            self._flush()

    def finish(self) -> tuple[Path, ...]:
        """Flush any remaining rows and return the manifest-declared shards."""

        if self._rows:
            self._flush()
        return tuple(self.paths)

    def _flush(self) -> None:
        index = len(self.paths)
        destination = self.output_dir / f"tuples-{index:05d}.npz"
        rows = self._rows
        margins = np.asarray(
            [_seat_margin(outcome.seats, row.seat_index) if outcome.margin_valid else 0 for row, outcome, _ in rows],
            dtype=np.int16,
        )
        signs = np.asarray(
            [_outcome_sign(outcome, row.seat_index) for row, outcome, _ in rows],
            dtype=np.int8,
        )
        values = np.asarray(
            [
                _margin_blend_value(int(margin)) if valid else float(sign)
                for margin, valid, sign in zip(
                    margins,
                    (outcome.margin_valid for _, outcome, _ in rows),
                    signs,
                    strict=True,
                )
            ],
            dtype=np.float32,
        )
        winners = np.asarray(
            [-1 if outcome.winner is None else outcome.winner for _, outcome, _ in rows],
            dtype=np.int8,
        )
        margin_valid = np.asarray(
            [outcome.margin_valid for _, outcome, _ in rows], dtype=np.bool_
        )
        turns_used = np.asarray(
            [outcome.seats[row.seat_index].turns_used for row, outcome, _ in rows], dtype=np.int16
        )
        np.savez_compressed(
            destination,
            # Existing seeded-export arrays: human_data.py can validate their
            # shapes and hard action legality without any special path.
            obs=np.stack([row.obs for row, _, _ in rows]).astype(np.float32, copy=False),
            action=np.asarray([row.action for row, _, _ in rows], dtype=np.int32),
            legal=np.stack([row.legal for row, _, _ in rows]).astype(np.bool_, copy=False),
            value=values,
            margin=margins,
            winner=winners,
            seat_index=np.asarray([row.seat_index for row, _, _ in rows], dtype=np.int16),
            game_index=np.asarray([game_index for _, _, game_index in rows], dtype=np.int32),
            # Stable dominion.games identity for the seat that acted on this
            # row.  Ratings remain external/temporal in the ratings sidecar;
            # this immutable key is all a later training filter needs here.
            player_id=np.asarray(
                [outcome.seats[row.seat_index].player_id for row, outcome, _ in rows],
                dtype=np.int64,
            ),
            ply_index=np.asarray([row.ply_index for row, _, _ in rows], dtype=np.int32),
            turn_number=np.asarray([row.turn_number for row, _, _ in rows], dtype=np.int32),
            # Spectator-specific labels.  ``margin`` is deliberately zero for
            # a resignation, and ``margin_valid`` tells consumers to use this
            # sign-only target instead of a made-up score margin.
            outcome_sign=signs,
            margin_valid=margin_valid,
            turns_used=turns_used,
            coins=np.asarray([row.coins for row, _, _ in rows], dtype=np.int16),
            buys=np.asarray([row.buys for row, _, _ in rows], dtype=np.int16),
            score_at_decision=np.asarray([row.score for row, _, _ in rows], dtype=np.int16),
            source_log_index=np.asarray([row.source_log_index for row, _, _ in rows], dtype=np.int32),
            source_event_index=np.asarray([row.source_event_index for row, _, _ in rows], dtype=np.int32),
            decision_type=np.asarray(
                [DECISION_TYPE_IDS[row.decision_type] for row, _, _ in rows],
                dtype=np.uint8,
            ),
            observation_quality=np.asarray(
                [row.observation_quality for row, _, _ in rows], dtype=np.uint8
            ),
        )
        self.paths.append(destination)
        self._rows = []


def convert_dgames_corpus(
    raw_root: Path | str = DEFAULT_RAW_ROOT,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    card_map_path: Path | str = DEFAULT_CARD_MAP,
    shard_size: int = 10_000,
) -> dict[str, object]:
    """Convert all manifests below ``raw_root`` into a separate tuple store.

    An incomplete capture is reported but never decoded for tuple emission.
    A complete capture is emitted only after the reconstructed final deck for
    *every* seat equals its GameResult histogram exactly.
    """

    root = Path(raw_root)
    destination = Path(output_dir)
    mapping = _load_card_map(Path(card_map_path))
    destination.mkdir(parents=True, exist_ok=True)
    accumulator = _ShardAccumulator(destination, shard_size)
    games: list[dict[str, object]] = []
    quarantined: list[QuarantineEntry] = []
    complete_seen = 0
    deck_checked_games = 0
    deck_matched_games = 0
    observed_resource_buy_rows = 0
    visibility_grades: Counter[str] = Counter()
    initial_visibility_grades: Counter[str] = Counter()
    decision_type_counts: Counter[str] = Counter()
    observation_quality_counts: Counter[str] = Counter()
    recovered_skip_counts: Counter[str] = Counter()
    militia_examples: list[dict[str, object]] = []
    manifests = sorted(root.glob("*.manifest.json"), key=_manifest_sort_key)

    print(f"dgames converter: discovered {len(manifests)} manifest(s)", file=sys.stderr, flush=True)
    for manifest_path in manifests:
        game_id = manifest_path.name.removesuffix(".manifest.json")
        raw_path = root / f"{game_id}.jsonl.gz"
        try:
            manifest = _read_json_object(manifest_path)
        except (OSError, ValueError) as error:
            quarantined.append(
                QuarantineEntry(
                    game_id=game_id,
                    manifest_path=manifest_path,
                    raw_path=raw_path if raw_path.exists() else None,
                    category="malformed_manifest",
                    reason=str(error),
                    capture_status=None,
                )
            )
            print(f"dgames converter: quarantined {game_id}: {error}", file=sys.stderr, flush=True)
            continue

        capture_status = manifest.get("capture_status")
        if capture_status != "complete":
            detail = manifest.get("incomplete_reason")
            reason = f"capture_status={capture_status!r}"
            if detail:
                reason += f": {detail}"
            quarantined.append(
                QuarantineEntry(
                    game_id=str(manifest.get("game_id", game_id)),
                    manifest_path=manifest_path,
                    raw_path=raw_path if raw_path.exists() else None,
                    category="incomplete_capture",
                    reason=reason,
                    capture_status=str(capture_status) if capture_status is not None else None,
                )
            )
            print(f"dgames converter: skipped incomplete {game_id}: {reason}", file=sys.stderr, flush=True)
            continue

        complete_seen += 1
        if not raw_path.exists():
            reason = "complete manifest has no matching .jsonl.gz raw capture"
            quarantined.append(
                QuarantineEntry(
                    game_id=str(manifest.get("game_id", game_id)),
                    manifest_path=manifest_path,
                    raw_path=None,
                    category="missing_raw_capture",
                    reason=reason,
                    capture_status="complete",
                )
            )
            print(f"dgames converter: quarantined {game_id}: {reason}", file=sys.stderr, flush=True)
            continue

        try:
            converted = _convert_complete_capture(raw_path, manifest, mapping)
        except FinalDeckMismatch as error:
            deck_checked_games += 1
            quarantined.append(
                QuarantineEntry(
                    game_id=str(manifest.get("game_id", game_id)),
                    manifest_path=manifest_path,
                    raw_path=raw_path,
                    category="final_deck_mismatch",
                    reason=str(error),
                    capture_status="complete",
                    deck_matches=error.deck_matches,
                )
            )
            print(f"dgames converter: quarantined {game_id}: {error}", file=sys.stderr, flush=True)
            continue
        except (OSError, json.JSONDecodeError, ProtocolError, ConversionError, ValueError) as error:
            reason = str(error)
            quarantined.append(
                QuarantineEntry(
                    game_id=str(manifest.get("game_id", game_id)),
                    manifest_path=manifest_path,
                    raw_path=raw_path,
                    category=_conversion_failure_category(reason),
                    reason=reason,
                    capture_status="complete",
                )
            )
            print(f"dgames converter: quarantined {game_id}: {reason}", file=sys.stderr, flush=True)
            continue

        # A successful return is possible only after every final-deck equality
        # check passed; keep the defensive guard close to emission anyway.
        deck_checked_games += 1
        if not all(converted.deck_matches):
            raise AssertionError("converted game bypassed final-deck correctness gate")
        deck_matched_games += 1
        observed_resource_buy_rows += converted.observed_resource_buy_rows
        visibility_grades[converted.visibility.grade] += 1
        initial_count = sum(converted.visibility.initial_visible)
        initial_visibility_grades[
            "full" if initial_count == len(converted.visibility.initial_visible) else "partial" if initial_count else "buy_only"
        ] += 1
        recovered_skip_counts.update(converted.recovered_skips)

        game_index = len(games)
        per_seat_exported = [0 for _ in converted.outcome.seats]
        for row in converted.rows:
            accumulator.append(row, converted.outcome, game_index)
            per_seat_exported[row.seat_index] += 1
            decision_type_counts[row.decision_type] += 1
            observation_quality_counts[
                "fully_observed"
                if row.observation_quality == OBSERVATION_FULLY_OBSERVED
                else "partially_inferred"
            ] += 1
        for example in converted.militia_examples:
            if len(militia_examples) < 16:
                militia_examples.append({"game_id": converted.game_id, **example})
        games.append(
            {
                "index": game_index,
                "id": converted.game_id,
                "source_path": _relative_path(root, converted.source_path),
                "source_tag": SOURCE_TAG,
                "seat_kinds": ["human" for _ in converted.outcome.seats],
                "player_ids": [seat.player_id for seat in converted.outcome.seats],
                "player_names": list(converted.player_names),
                "kingdom": list(converted.kingdom),
                "decision_counts": list(converted.per_seat_decisions),
                "exported_decision_counts": per_seat_exported,
                "observed_resource_buy_rows": converted.observed_resource_buy_rows,
                "visibility": {
                    "grade": converted.visibility.grade,
                    "full_state_segments": converted.visibility.segment_count,
                    "initial_hand_visible": list(converted.visibility.initial_visible),
                    "hand_visible_throughout": list(converted.visibility.visible_throughout),
                    "first_unknown_event": list(converted.visibility.first_unknown_event),
                },
                "decision_counts_by_type": dict(
                    sorted(Counter(row.decision_type for row in converted.rows).items())
                ),
                "observation_quality_counts": {
                    "fully_observed": sum(
                        row.observation_quality == OBSERVATION_FULLY_OBSERVED
                        for row in converted.rows
                    ),
                    "partially_inferred": sum(
                        row.observation_quality == OBSERVATION_PARTIALLY_INFERRED
                        for row in converted.rows
                    ),
                },
                "recovered_skip_counts": converted.recovered_skips,
                "final_deck_match": {
                    "all_seats": True,
                    "per_seat": list(converted.deck_matches),
                },
                "outcome": {
                    "winner": converted.outcome.winner,
                    "scores": [seat.score for seat in converted.outcome.seats],
                    "turns_used": [seat.turns_used for seat in converted.outcome.seats],
                    "margins": [
                        _seat_margin(converted.outcome.seats, seat_index)
                        for seat_index in range(len(converted.outcome.seats))
                    ],
                    "margin_valid": converted.outcome.margin_valid,
                    "outcome_signs": [
                        _outcome_sign(converted.outcome, seat_index)
                        for seat_index in range(len(converted.outcome.seats))
                    ],
                    "resigned": [seat.resigned for seat in converted.outcome.seats],
                },
            }
        )
        print(
            f"dgames converter: emitted {converted.game_id}: "
            f"{len(converted.rows)} tuple(s), visibility={converted.visibility.grade}, "
            f"final decks match {list(converted.deck_matches)}",
            file=sys.stderr,
            flush=True,
        )

    shard_paths = accumulator.finish()
    complete_divergences = deck_checked_games - deck_matched_games
    match_rate = 1.0 if deck_checked_games == 0 else deck_matched_games / deck_checked_games
    divergence_rate = 0.0 if deck_checked_games == 0 else complete_divergences / deck_checked_games
    quarantine_json = [entry.as_json() for entry in quarantined]
    quarantine_path = destination / "quarantine.json"
    quarantine_path.write_text(json.dumps(quarantine_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "schema_version": 3,
        "source_tag": SOURCE_TAG,
        "obs_version": OBS_VERSION,
        "obs_width": OBS_WIDTH,
        "action_width": ACTION_WIDTH,
        "value_target": {
            "name": "margin_blend",
            "alpha": DEFAULT_ALPHA,
            "scale": DEFAULT_SCALE,
            "resignation": "outcome_sign only; margin=0 and margin_valid=false",
        },
        "observation_reconstruction": {
            "method": "visibility_aware_engine_snapshot_determinization",
            "public_facts": [
                "starting deck plus public gains minus public trashes",
                "supply from card totals minus owned collections and trash",
                "current public in-play cards and turn state",
                "authoritative live ResourceUpdate coins/buys at observed buy boundaries, with semantic fallback for join-history rows",
                "base-card effect scope distinguishes hand, deck/reveal, discard, and topdeck moves",
                "initial fullGameState hand contents and anonymous-count visibility",
                "post-snapshot CardMove source/destination identities, retaining -1 redactions",
                "each reconnect's FullState starts a separate tracked CardMove segment",
                "face-up active-turn gains/discards whose zone placement remains unambiguous",
            ],
            "sampled_not_observed": [
                "hands after a CardMove destination is redacted",
                "hidden deck identities and order",
                "unresolved discard identities and hidden-zone allocation",
                "BACK-card identities in spectator semantic entries",
            ],
            "sampling": "deterministic BLAKE2b-seeded allocation per game/live-event/seat",
            "draw_card_identities_used": "only identities present in CardMove destination arrays",
            "decision_entry_answers_used": False,
        },
        "row_provenance": {
            "decision_type_ids": {name: value for name, value in DECISION_TYPE_IDS.items()},
            "observation_quality": {
                str(OBSERVATION_PARTIALLY_INFERRED): "partially_inferred: a required private offered zone was sampled or this is a public-history buy row",
                str(OBSERVATION_FULLY_OBSERVED): "fully_observed: the actor's decision-relevant hand/reveal transition was visible; hidden deck allocation is not a label source",
            },
            "arrays": [
                "decision_type",
                "observation_quality",
                "source_event_index",
                "player_id",
            ],
        },
        "games": games,
        # Keep the immutable tuple-to-capture join explicit.  A training
        # reader combines this with each row's player_id and the mutable
        # data/dominion_games/ratings/game_ratings.json sidecar; ratings must
        # never be copied into these NPZ shards because they change over time.
        "game_index_to_game_id": {
            str(game["index"]): str(game["id"])
            for game in games
        },
        "shards": [
            {"path": path.name, "tuples": _shard_tuple_count(path)} for path in shard_paths
        ],
        "quarantine_path": quarantine_path.name,
        "totals": {
            "games_discovered": len(manifests),
            "complete_games_seen": complete_seen,
            "games_emitted": len(games),
            "tuples_exported": accumulator.total_rows,
            "tuples_per_emitted_game": (
                0.0 if not games else accumulator.total_rows / len(games)
            ),
            "tuples_by_decision_type": dict(sorted(decision_type_counts.items())),
            "rows_by_observation_quality": dict(sorted(observation_quality_counts.items())),
            "visibility_grades": dict(sorted(visibility_grades.items())),
            "initial_visibility_grades": dict(sorted(initial_visibility_grades.items())),
            "recovered_skip_counts": dict(sorted(recovered_skip_counts.items())),
            "observed_resource_buy_rows": observed_resource_buy_rows,
            "skipped_incomplete": sum(entry.category == "incomplete_capture" for entry in quarantined),
            "quarantined": len(quarantined),
            "final_deck_checked_games": deck_checked_games,
            "complete_games_without_final_deck_check": complete_seen - deck_checked_games,
            "final_deck_matched_games": deck_matched_games,
            "final_deck_match_rate": match_rate,
            "final_deck_divergences": complete_divergences,
            "final_deck_divergence_rate": divergence_rate,
            "divergence_requires_investigation": divergence_rate > 0.02,
        },
        "militia_sanity_examples": militia_examples,
        "quarantine_counts_by_category": dict(
            sorted(Counter(entry.category for entry in quarantined).items())
        ),
        "quarantine_reasons": dict(
            sorted(Counter(entry.reason for entry in quarantined).items())
        ),
        "quarantine": quarantine_json,
    }
    manifest_path = destination / "tuple_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "manifest_path": manifest_path,
        "quarantine_path": quarantine_path,
        "tuples_exported": accumulator.total_rows,
        "games_emitted": len(games),
        "quarantined": len(quarantined),
        "final_deck_match_rate": match_rate,
        "final_deck_divergence_rate": divergence_rate,
    }


def _conversion_failure_category(reason: str) -> str:
    """Give structurally unencodable captures an audit-friendly family."""

    if "only two-player captures are supported" in reason:
        return "unsupported_player_count"
    if "MAX_IN_PLAY" in reason:
        return "engine_in_play_capacity"
    return "conversion_failed"


class _LiveRecoveryError(ValueError):
    """One live decision cannot be reconstructed without inventing state."""


@dataclass
class _LiveZone:
    known: CounterInt
    anonymous: int

    @property
    def count(self) -> int:
        return _counter_total(self.known) + self.anonymous

    def clone(self) -> "_LiveZone":
        return _LiveZone(known=Counter(self.known), anonymous=self.anonymous)


def _live_zone_kind(kind: str) -> str:
    """Normalize the set-aside aliases emitted by the protocol."""

    return "set-aside" if kind == "zone-type-24" else kind


@dataclass
class _LiveState:
    """Visible live zones plus exact public ownership at one delta boundary."""

    game_id: str
    card_totals: CounterInt
    kingdom: tuple[str, ...]
    zones: dict[int, _LiveZone]
    zone_kind: dict[int, str]
    zone_owner: dict[int, int | None]
    resources: list[dict[str, int]]
    collections: list[CounterInt]
    trash: CounterInt
    current_turn_seat: int | None
    player_turn_number: int
    phase: str
    initial_visible: tuple[bool, ...]
    visible_throughout: list[bool]
    first_unknown_event: list[int | None]
    ownership_safe: bool = True

    def clone(self) -> "_LiveState":
        return _LiveState(
            game_id=self.game_id,
            card_totals=Counter(self.card_totals),
            kingdom=self.kingdom,
            zones={index: zone.clone() for index, zone in self.zones.items()},
            zone_kind=self.zone_kind.copy(),
            zone_owner=self.zone_owner.copy(),
            resources=[resource.copy() for resource in self.resources],
            collections=[Counter(collection) for collection in self.collections],
            trash=Counter(self.trash),
            current_turn_seat=self.current_turn_seat,
            player_turn_number=self.player_turn_number,
            phase=self.phase,
            initial_visible=self.initial_visible,
            visible_throughout=self.visible_throughout.copy(),
            first_unknown_event=self.first_unknown_event.copy(),
            ownership_safe=self.ownership_safe,
        )

    @property
    def seats(self) -> int:
        return len(self.collections)

    def resource_tuple(self, seat: int) -> tuple[int, int, int]:
        self._check_seat(seat)
        values = self.resources[seat]
        return (values.get("actions", 0), values.get("buys", 0), values.get("coins", 0))

    def set_resource(self, seat: int, resource: str, value: int) -> None:
        self._check_seat(seat)
        self.resources[seat][resource] = value

    def aggregate(self, seat: int, kind: str) -> _LiveZone:
        self._check_seat(seat)
        normalized = _live_zone_kind(kind)
        result = _LiveZone(known=Counter(), anonymous=0)
        for index, zone in self.zones.items():
            if self.zone_owner.get(index) != seat or self.zone_kind.get(index) != normalized:
                continue
            result.known.update(zone.known)
            result.anonymous += zone.anonymous
        return result

    def hand_is_exact(self, seat: int) -> bool:
        return self.aggregate(seat, "hand").anonymous == 0

    def zone_is_exact(self, seat: int, kind: str) -> bool:
        return self.aggregate(seat, kind).anonymous == 0

    def apply_move(self, move: LiveCardMove, mapping: CardMap) -> None:
        """Apply endpoint visibility without consulting a redacted identity."""

        source_owner = self.zone_owner.get(move.from_zone_index)
        if source_owner is None:
            source_owner = move.seat
        destination_owner = self.zone_owner.get(move.to_zone_index)
        if destination_owner is None:
            destination_owner = move.seat
        source = self._movement_zone(
            move.from_zone_index,
            _live_zone_kind(move.from_zone),
            source_owner,
            destination=False,
        )
        destination = self._movement_zone(
            move.to_zone_index,
            _live_zone_kind(move.to_zone),
            destination_owner,
            destination=True,
        )
        if source is not None:
            self._take_from_zone(source, move.source_cards, move.count, move)
        if destination is not None:
            self._put_in_zone(destination, move.destination_cards, move.count)
        self._update_ownership(move, mapping)
        self._refresh_visibility(move.event_index)

    def _movement_zone(
        self,
        index: int,
        kind: str,
        owner: int | None,
        *,
        destination: bool,
    ) -> _LiveZone | None:
        if kind not in _TRACKED_LIVE_ZONE_KINDS:
            return None
        zone = self.zones.get(index)
        if zone is not None:
            known_kind = self.zone_kind.get(index)
            if known_kind != kind:
                raise _LiveRecoveryError(
                    f"live zone {index} changed kind from {known_kind!r} to {kind!r}"
                )
            if self.zone_owner.get(index) is None and owner is not None:
                self.zone_owner[index] = owner
            return zone
        if not destination:
            raise _LiveRecoveryError(f"live CardMove source zone {index} is unknown")
        self.zone_kind[index] = kind
        self.zone_owner[index] = owner
        zone = _LiveZone(known=Counter(), anonymous=0)
        self.zones[index] = zone
        return zone

    def _take_from_zone(
        self,
        zone: _LiveZone,
        cards: Sequence[str | None],
        count: int,
        move: LiveCardMove,
    ) -> None:
        if zone.count < count:
            raise _LiveRecoveryError(
                f"live CardMove {move.event_index} removes {count} cards from a zone holding {zone.count}"
            )
        for name in cards:
            if name is not None:
                def_id = self._def_id(name, move.event_index)
                if zone.known.get(def_id, 0):
                    _counter_add(zone.known, Counter({def_id: 1}), -1, context="live zone removal")
                elif zone.anonymous:
                    zone.anonymous -= 1
                else:
                    raise _LiveRecoveryError(
                        f"live CardMove {move.event_index} names {name} outside its source zone"
                    )
                continue
            if not zone.anonymous:
                # A source-side redaction means we no longer know which one
                # of these cards remains.  Convert, do not guess.
                zone.anonymous += _counter_total(zone.known)
                zone.known.clear()
            if not zone.anonymous:
                raise _LiveRecoveryError(
                    f"live CardMove {move.event_index} redacts a card absent from its source zone"
                )
            zone.anonymous -= 1

    def _put_in_zone(self, zone: _LiveZone, cards: Sequence[str | None], count: int) -> None:
        if len(cards) != count:
            raise _LiveRecoveryError("live CardMove destination length is inconsistent")
        for name in cards:
            if name is None:
                zone.anonymous += 1
            else:
                zone.known[self._def_id(name, -1)] += 1

    def _update_ownership(self, move: LiveCardMove, mapping: CardMap) -> None:
        if move.seat is None or move.seat < 0 or move.seat >= self.seats:
            return
        known = _known_move_counter(move, mapping)
        if move.from_zone == "supply":
            if known is None:
                self.ownership_safe = False
                return
            _counter_add(self.collections[move.seat], known, 1, context="live supply gain")
            return
        if move.to_zone == "trash":
            if known is None:
                self.ownership_safe = False
                return
            _counter_add(self.collections[move.seat], known, -1, context="live trash")
            self.trash.update(known)

    def _refresh_visibility(self, event_index: int) -> None:
        for seat in range(self.seats):
            if self.hand_is_exact(seat):
                continue
            if self.visible_throughout[seat]:
                self.visible_throughout[seat] = False
                self.first_unknown_event[seat] = event_index

    def _def_id(self, name: str, event_index: int) -> int:
        try:
            return int(dz.def_id(name))
        except (TypeError, ValueError) as error:
            context = "initial FullState" if event_index < 0 else f"live CardMove {event_index}"
            raise _LiveRecoveryError(f"{context}: engine has no card {name!r}") from error

    def _check_seat(self, seat: int) -> None:
        if seat < 0 or seat >= self.seats:
            raise _LiveRecoveryError(f"live event has invalid seat {seat}")


@dataclass(frozen=True)
class _LiveMoveRecord:
    move: LiveCardMove
    before: _LiveState | None
    after: _LiveState | None
    card_name: str | None
    action_resources: tuple[int, int, int] | None
    buy_resources: tuple[int, int, int] | None
    before_hand_exact: tuple[bool, ...]
    after_hand_exact: tuple[bool, ...]

    @property
    def normal_action(self) -> bool:
        return self.action_resources is not None


def _known_move_counter(move: LiveCardMove, mapping: CardMap) -> CounterInt | None:
    """Return a move's observed identities, or ``None`` if any are redacted."""

    names = move.source_cards
    if any(name is None for name in names):
        names = move.destination_cards
    if any(name is None for name in names) or len(names) != move.count:
        return None
    result: CounterInt = Counter()
    for name in names:
        assert name is not None
        result[_def_for_name(name, mapping, f"live CardMove {move.event_index}")] += 1
    return result


def _single_move_card_name(move: LiveCardMove) -> str | None:
    """Return an unambiguous card identity from a one-card move."""

    if move.count != 1:
        return None
    source = move.source_cards[0]
    destination = move.destination_cards[0]
    if source is not None and destination is not None and source != destination:
        return None
    return source if source is not None else destination


def _initial_public_ownership(
    entries: Sequence[SemanticLogEntry],
    seats: int,
    mapping: CardMap,
    raw_path: Path,
) -> tuple[list[CounterInt], CounterInt]:
    """Replay public ownership only through the state sent at spectate join."""

    collections = [Counter(_counter_from_names(STARTING_DECK, mapping)) for _ in range(seats)]
    trash: CounterInt = Counter()
    for entry in entries:
        if entry.name not in (LOG_GAIN, LOG_BUY_GAIN, LOG_TRASH):
            continue
        seat = _seat_argument(entry, raw_path)
        _check_seat(seat, seats, raw_path, entry.index)
        cards = _require_known_cards(
            entry,
            mapping,
            raw_path,
            operation="initial live ownership",
        )
        if entry.name in (LOG_GAIN, LOG_BUY_GAIN):
            _counter_add(collections[seat], cards, 1, context="initial live gain")
        else:
            _counter_add(collections[seat], cards, -1, context="initial live trash")
            trash.update(cards)
    return collections, trash


def _initial_turn(entries: Sequence[SemanticLogEntry], raw_path: Path) -> tuple[int | None, int]:
    """Recover the current normal turn from the historical semantic suffix."""

    current: tuple[int, int] | None = None
    for entry in entries:
        if entry.name != LOG_TURN_DESCRIPTION:
            continue
        turn = _turn_description(entry, raw_path)
        if turn is None or turn[2] != 0:
            continue
        owner, turn_number, _turn_type, controller = turn
        if owner == controller:
            current = (owner, turn_number)
    return (None, 0) if current is None else current


def _live_state_from_capture(
    capture: ProtocolCapture,
    *,
    game_id: str,
    mapping: CardMap,
) -> _LiveState:
    """Seed live zone/ownership tracking at the initial FullState boundary."""

    seats = len(capture.game_start.player_ids)
    collections, trash = _initial_public_ownership(
        capture.initial_log_entries,
        seats,
        mapping,
        capture.raw_path,
    )
    zones: dict[int, _LiveZone] = {}
    zone_kind: dict[int, str] = {}
    zone_owner: dict[int, int | None] = {}
    hand_zone_counts = [0 for _ in range(seats)]
    initial_visible = [False for _ in range(seats)]
    for zone in capture.full_state.zones:
        kind = _live_zone_kind(zone.kind)
        zone_kind[zone.index] = kind
        zone_owner[zone.index] = zone.owner
        if kind not in _TRACKED_LIVE_ZONE_KINDS:
            continue
        known: CounterInt = Counter()
        for name in zone.contents:
            known[_def_for_name(name, mapping, "initial FullState zone")] += 1
        zones[zone.index] = _LiveZone(known=known, anonymous=zone.anonymous_count)
        if kind == "hand" and zone.owner is not None and 0 <= zone.owner < seats:
            hand_zone_counts[zone.owner] += 1
            # An empty hand with anonymous_count zero is fully known too.  It
            # is the zone's explicit empty contents, not an absent zone.
            initial_visible[zone.owner] = zone.anonymous_count == 0
    if any(count != 1 for count in hand_zone_counts):
        raise _LiveRecoveryError(
            f"initial FullState has hand-zone counts {hand_zone_counts}, expected one per seat"
        )
    turn_owner, turn_number = _initial_turn(capture.initial_log_entries, capture.raw_path)
    resources = [dict() for _ in range(seats)]
    for counter in capture.full_state.counters:
        if counter.owner is not None and 0 <= counter.owner < seats:
            resources[counter.owner][counter.name] = counter.value
    card_totals = _counter_from_named_pairs(
        capture.full_state.card_counts,
        mapping,
        context="initial FullState card totals",
    )
    return _LiveState(
        game_id=game_id,
        card_totals=card_totals,
        kingdom=tuple(capture.game_start.kingdom),
        zones=zones,
        zone_kind=zone_kind,
        zone_owner=zone_owner,
        resources=resources,
        collections=collections,
        trash=trash,
        current_turn_seat=turn_owner,
        player_turn_number=turn_number,
        phase="action",
        initial_visible=tuple(initial_visible),
        visible_throughout=initial_visible.copy(),
        first_unknown_event=[None for _ in range(seats)],
    )


def _visibility_summary(state: _LiveState) -> VisibilitySummary:
    visible = tuple(state.visible_throughout)
    count = sum(visible)
    grade = "full" if count == len(visible) else "partial" if count else "buy_only"
    return VisibilitySummary(
        initial_visible=state.initial_visible,
        visible_throughout=visible,
        first_unknown_event=tuple(state.first_unknown_event),
        grade=grade,
    )


def _initial_visibility_from_full_state(capture: ProtocolCapture, seats: int) -> tuple[bool, ...]:
    """Read only explicit initial hand visibility for a conservative fallback."""

    visibility = [False for _ in range(seats)]
    hand_zone_counts = [0 for _ in range(seats)]
    for zone in capture.full_state.zones:
        if _live_zone_kind(zone.kind) != "hand" or zone.owner is None:
            continue
        if not 0 <= zone.owner < seats:
            continue
        hand_zone_counts[zone.owner] += 1
        if zone.anonymous_count == 0:
            visibility[zone.owner] = True
    # A missing or duplicated hand zone is not evidence that the hand was
    # known.  This mirrors the strict check in _live_state_from_capture.
    return tuple(
        visible if hand_zone_counts[seat] == 1 else False
        for seat, visible in enumerate(visibility)
    )


def _build_live_trace(
    capture: ProtocolCapture,
    *,
    game_id: str,
    mapping: CardMap,
) -> tuple[list[_LiveMoveRecord], _LiveState]:
    """Fold post-snapshot deltas while retaining decision-boundary snapshots."""

    state = _live_state_from_capture(capture, game_id=game_id, mapping=mapping)
    records: list[_LiveMoveRecord] = []
    pending_actions: dict[int, tuple[int, int, int]] = {}
    pending_buys: dict[int, tuple[int, int, int]] = {}
    for event in capture.live_events:
        if isinstance(event, LiveResourceUpdate):
            if event.seat is None or event.seat < 0 or event.seat >= state.seats:
                continue
            previous = state.resources[event.seat].get(event.resource, 0)
            if event.resource == "actions" and event.value < previous:
                pending_actions[event.seat] = state.resource_tuple(event.seat)
            elif event.resource == "buys" and event.value < previous:
                pending_buys[event.seat] = state.resource_tuple(event.seat)
            state.set_resource(event.seat, event.resource, event.value)
            continue
        if isinstance(event, LiveTurnDescription):
            if event.seat < 0 or event.seat >= state.seats:
                raise _LiveRecoveryError(f"live turn has invalid seat {event.seat}")
            state.current_turn_seat = event.seat
            state.player_turn_number = event.turn_number
            state.phase = "action" if event.turn_type == 0 else "cleanup"
            pending_actions.clear()
            pending_buys.clear()
            continue
        if isinstance(event, LiveShuffle):
            # CardMoves carry the actual zone transfer.  A shuffle changes
            # order, which snapshots intentionally represent as sampled.
            continue
        if not isinstance(event, LiveCardMove):
            continue

        card_name = _single_move_card_name(event)
        action_resources: tuple[int, int, int] | None = None
        buy_resources: tuple[int, int, int] | None = None
        if (
            event.movement == "PLAY"
            and event.from_zone == "hand"
            and event.seat is not None
            and event.seat in pending_actions
        ):
            action_resources = pending_actions.pop(event.seat)
        if (
            event.movement == "GAIN"
            and event.from_zone == "supply"
            and event.seat is not None
            and event.seat in pending_buys
        ):
            buy_resources = pending_buys.pop(event.seat)
        capture_before = (
            event.movement in {"PLAY", "GAIN", "TRASH", "DISCARD", "TOPDECK", "REVEAL", "LOOK_AT"}
        )
        before = state.clone() if capture_before else None
        before_exact = tuple(state.hand_is_exact(seat) for seat in range(state.seats))
        state.apply_move(event, mapping)
        after = state.clone() if capture_before else None
        after_exact = tuple(state.hand_is_exact(seat) for seat in range(state.seats))
        records.append(
            _LiveMoveRecord(
                move=event,
                before=before,
                after=after,
                card_name=card_name,
                action_resources=action_resources,
                buy_resources=buy_resources,
                before_hand_exact=before_exact,
                after_hand_exact=after_exact,
            )
        )
        if event.movement == "PLAY" and event.from_zone == "hand" and card_name in _TREASURE_NAMES:
            state.phase = "buy"
        elif action_resources is not None:
            state.phase = "action"
    return records, state


def _live_snapshot(
    state: _LiveState,
    *,
    our_player: int,
    current_player: int,
    phase: str,
    mapping: CardMap,
    source_event_index: int,
    resource_override: tuple[int, int, int] | None = None,
    forced_deck: Sequence[int] = (),
) -> tuple[dict[str, object], dict[int, list[int]]]:
    """Build one conservative engine snapshot from live visible zones.

    Unknown slots are allocated only after all named zones have been removed
    from public ownership.  ``forced_deck`` reserves observed upcoming draws
    in the actor's otherwise anonymous deck, allowing a source card to be
    replayed up to its pending choice without fabricating a draw identity.
    """

    if not state.ownership_safe:
        raise _LiveRecoveryError("a live gain/trash had redacted ownership")
    if our_player < 0 or our_player >= state.seats:
        raise _LiveRecoveryError(f"invalid snapshot observer seat {our_player}")
    if current_player < 0 or current_player >= state.seats:
        raise _LiveRecoveryError(f"invalid snapshot current seat {current_player}")
    if phase not in {"action", "buy"}:
        raise _LiveRecoveryError(f"unsupported live snapshot phase {phase!r}")

    resolved: list[dict[str, CounterInt]] = []
    deck_orders: dict[int, list[int]] = {}
    force_counts: CounterInt = Counter(forced_deck)
    for seat in range(state.seats):
        raw = {
            kind: state.aggregate(seat, kind)
            for kind in ("hand", "deck", "discard", "in-play", "set-aside")
        }
        known_total: CounterInt = Counter()
        for zone in raw.values():
            known_total.update(zone.known)
        try:
            remaining = _counter_subtract(
                state.collections[seat],
                known_total,
                context=f"live snapshot seat {seat}: named zones exceed ownership",
            )
        except ConversionError as error:
            raise _LiveRecoveryError(str(error)) from error
        anonymous_total = sum(zone.anonymous for zone in raw.values())
        if _counter_total(remaining) != anonymous_total:
            raise _LiveRecoveryError(
                f"live snapshot seat {seat}: {anonymous_total} anonymous zone slots but "
                f"{_counter_total(remaining)} unassigned owned cards"
            )
        if seat == our_player and raw["hand"].anonymous:
            raise _LiveRecoveryError(f"live snapshot observer seat {seat} has an anonymous hand")

        seat_resolved = {kind: Counter(zone.known) for kind, zone in raw.items()}
        deck_anonymous = raw["deck"].anonymous
        if seat == current_player and force_counts:
            extras: CounterInt = Counter()
            for def_id, wanted in force_counts.items():
                missing = wanted - seat_resolved["deck"].get(def_id, 0)
                if missing > 0:
                    extras[def_id] = missing
            if _counter_total(extras) > deck_anonymous:
                raise _LiveRecoveryError(
                    "observed forced draws do not fit in the live deck's anonymous slots"
                )
            try:
                remaining = _counter_subtract(
                    remaining,
                    extras,
                    context="observed forced draws are absent from public ownership",
                )
            except ConversionError as error:
                raise _LiveRecoveryError(str(error)) from error
            seat_resolved["deck"].update(extras)
            deck_anonymous -= _counter_total(extras)

        for kind in ("hand", "deck", "discard", "in-play", "set-aside"):
            anonymous = deck_anonymous if kind == "deck" else raw[kind].anonymous
            if anonymous:
                sampled = _sample_counter(
                    remaining,
                    anonymous,
                    _sample_seed(state.game_id, source_event_index * 11 + seat * 5 + len(kind), seat),
                )
                seat_resolved[kind].update(sampled)
                _counter_add(
                    remaining,
                    sampled,
                    -1,
                    context="live snapshot anonymous allocation",
                )
        if remaining:
            raise _LiveRecoveryError(
                f"live snapshot seat {seat} left unallocated cards {dict(remaining)}"
            )
        resolved.append(seat_resolved)
        if seat == current_player and force_counts:
            deck = Counter(seat_resolved["deck"])
            ordered = list(forced_deck)
            _counter_add(deck, force_counts, -1, context="forced deck order")
            ordered.extend(def_id for def_id in sorted(deck) for _ in range(deck[def_id]))
            deck_orders[seat] = ordered

    supply = _supply_from_public_totals(state.card_totals, state.collections, state.trash)
    players: list[dict[str, object]] = []
    for seat, zones in enumerate(resolved):
        hand = zones["hand"]
        deck = zones["deck"]
        actions, buys, coins = state.resource_tuple(seat)
        if seat == current_player and resource_override is not None:
            actions, buys, coins = resource_override
        players.append(
            {
                "hand": _snapshot_counter(hand) if seat == our_player else {},
                "hand_count": _counter_total(hand),
                "hand_deck": _snapshot_counter(Counter(hand) + Counter(deck)),
                "deck_count": _counter_total(deck),
                "discard": _snapshot_counter(zones["discard"]),
                "in_play": _snapshot_counter(zones["in-play"]),
                "set_aside": _snapshot_counter(zones["set-aside"]),
                "actions": actions,
                "buys": buys,
                "coins": coins,
            }
        )
    return (
        {
            "num_players": state.seats,
            "our_player": our_player,
            "supply": _snapshot_counter(supply, include_zero_defs=state.card_totals),
            "kingdom_order": [_def_for_name(name, mapping, "live kingdom") for name in state.kingdom],
            "players": players,
            "trash": _snapshot_counter(state.trash),
            "card_totals": _snapshot_counter(state.card_totals),
            "turn_number": max(0, state.player_turn_number - 1),
            "phase": phase,
            "current_player": current_player,
        },
        deck_orders,
    )


def _new_live_game(
    state: _LiveState,
    *,
    our_player: int,
    current_player: int,
    phase: str,
    mapping: CardMap,
    source_event_index: int,
    resource_override: tuple[int, int, int] | None = None,
    forced_deck: Sequence[int] = (),
    interrupt: dict[str, int | str] | None = None,
):
    """Instantiate and validate a native game for one recovered decision."""

    snapshot, deck_orders = _live_snapshot(
        state,
        our_player=our_player,
        current_player=current_player,
        phase=phase,
        mapping=mapping,
        source_event_index=source_event_index,
        resource_override=resource_override,
        forced_deck=forced_deck,
    )
    if interrupt is not None:
        snapshot["interrupt"] = interrupt
    try:
        game = dz.game_from_snapshot(snapshot)
        for seat, order in deck_orders.items():
            game.set_deck_order(seat, order)
        game.validate()
        return game
    except (TypeError, ValueError, RuntimeError) as error:
        raise _LiveRecoveryError(f"native live snapshot rejected: {error}") from error


def _row_from_game(
    game,
    *,
    actor: int,
    action: int,
    decision_type: str,
    source_event_index: int,
    turn_number: int,
    observation_quality: int,
    ply_index: int,
) -> PublicBuyRow:
    """Encode a native decision state and reject every non-legal label."""

    if decision_type not in DECISION_TYPE_IDS:
        raise _LiveRecoveryError(f"unknown decision type {decision_type!r}")
    try:
        decision = dict(game.current_decision())
        decision_player = int(decision["player"])
        mask = np.asarray(game.legal_mask(), dtype=np.bool_)
        obs = np.asarray(game.encode(actor, OBS_VERSION), dtype=np.float32)
        resources = dict(game.resources(-1))
    except (TypeError, ValueError, RuntimeError, KeyError) as error:
        raise _LiveRecoveryError(f"native live decision could not be encoded: {error}") from error
    if decision_player != actor:
        raise _LiveRecoveryError(
            f"native live decision belongs to seat {decision_player}, expected seat {actor}"
        )
    if obs.shape != (OBS_WIDTH,) or mask.shape != (ACTION_WIDTH,):
        raise _LiveRecoveryError(
            f"native live decision has obs/mask shapes {obs.shape}/{mask.shape}"
        )
    if action < 0 or action >= ACTION_WIDTH or not bool(mask[action]):
        raise _LiveRecoveryError(
            f"native live {decision_type} action {action} is not legal for seat {actor}"
        )
    return PublicBuyRow(
        obs=obs.copy(),
        legal=mask.copy(),
        action=action,
        seat_index=actor,
        ply_index=ply_index,
        turn_number=turn_number,
        source_log_index=-1,
        coins=int(resources.get("coins", 0)),
        buys=int(resources.get("buys", 0)),
        score=int(game.score(actor)),
        decision_type=decision_type,
        observation_quality=observation_quality,
        source_event_index=source_event_index,
    )


def _append_game_action(
    rows: list[PublicBuyRow],
    game,
    *,
    actor: int,
    action: int,
    decision_type: str,
    source_event_index: int,
    turn_number: int,
    observation_quality: int,
) -> None:
    """Append one row then advance its native continuation."""

    rows.append(
        _row_from_game(
            game,
            actor=actor,
            action=action,
            decision_type=decision_type,
            source_event_index=source_event_index,
            turn_number=turn_number,
            observation_quality=observation_quality,
            ply_index=len(rows),
        )
    )
    try:
        game.step(action)
    except (TypeError, ValueError, RuntimeError) as error:
        raise _LiveRecoveryError(f"native live {decision_type} continuation failed: {error}") from error


def _effect_game(
    record: _LiveMoveRecord,
    *,
    mapping: CardMap,
    forced_deck: Sequence[int] = (),
):
    """Replay one normally played source action up to its first choice."""

    if record.before is None or record.move.seat is None or record.card_name is None:
        raise _LiveRecoveryError("source action lacks a pre-move state or identity")
    if record.action_resources is None:
        raise _LiveRecoveryError("source action was not a normal action-phase play")
    actor = record.move.seat
    game = _new_live_game(
        record.before,
        our_player=actor,
        current_player=actor,
        phase="action",
        mapping=mapping,
        source_event_index=record.move.event_index,
        resource_override=record.action_resources,
        forced_deck=forced_deck,
    )
    _append_game_action(
        [],
        game,
        actor=actor,
        action=A_PLAY_BASE + _def_for_name(record.card_name, mapping, "live source action"),
        decision_type="action_play",
        source_event_index=record.move.event_index,
        turn_number=record.before.player_turn_number,
        observation_quality=OBSERVATION_FULLY_OBSERVED,
    )
    return game


def _is_hand_play(record: _LiveMoveRecord) -> bool:
    move = record.move
    return move.movement == "PLAY" and move.from_zone == "hand" and move.to_zone == "in-play"


def _is_normal_source(record: _LiveMoveRecord, *, eligible: Sequence[bool]) -> bool:
    seat = record.move.seat
    return (
        _is_hand_play(record)
        and record.normal_action
        and record.card_name in _ACTION_EFFECT_NAMES
        and seat is not None
        and 0 <= seat < len(eligible)
        and eligible[seat]
        and bool(record.before_hand_exact[seat])
        and record.before is not None
        and record.before.ownership_safe
    )


def _record_counter(record: _LiveMoveRecord, mapping: CardMap) -> CounterInt:
    cards = _known_move_counter(record.move, mapping)
    if cards is None:
        raise _LiveRecoveryError(
            f"live CardMove {record.move.event_index} has a redacted selected card"
        )
    return cards


def _append_optional_pass(
    rows: list[PublicBuyRow],
    game,
    *,
    actor: int,
    decision_type: str,
    source_event_index: int,
    turn_number: int,
    observation_quality: int,
) -> None:
    """Append a demonstrated optional completion only when native state permits it."""

    decision = dict(game.current_decision())
    mask = np.asarray(game.legal_mask(), dtype=np.bool_)
    if int(decision["player"]) != actor or not bool(mask[A_PASS]):
        return
    _append_game_action(
        rows,
        game,
        actor=actor,
        action=A_PASS,
        decision_type=decision_type,
        source_event_index=source_event_index,
        turn_number=turn_number,
        observation_quality=observation_quality,
    )


def _append_select_counter(
    rows: list[PublicBuyRow],
    game,
    cards: CounterInt,
    *,
    actor: int,
    decision_type: str,
    source_event_index: int,
    turn_number: int,
    observation_quality: int,
) -> None:
    """Emit a deterministic sequence for a set-valued native card choice."""

    for def_id in sorted(cards):
        for _ in range(cards[def_id]):
            _append_game_action(
                rows,
                game,
                actor=actor,
                action=A_SELECT_BASE + def_id,
                decision_type=decision_type,
                source_event_index=source_event_index,
                turn_number=turn_number,
                observation_quality=observation_quality,
            )


def _next_record(
    records: Sequence[_LiveMoveRecord],
    index: int,
    *,
    predicate,
    stop=None,
) -> _LiveMoveRecord | None:
    for candidate in records[index + 1 :]:
        if predicate(candidate):
            return candidate
        if stop is not None and stop(candidate):
            return None
    return None


def _next_proven_deck_top(
    records: Sequence[_LiveMoveRecord],
    index: int,
    *,
    actor: int,
    mapping: CardMap,
) -> int | None:
    """Return the next proven top-deck card, without assuming wire ordering.

    After Sentry returns two different cards, an intervening Action may draw
    one of them.  That draw still proves the Sentry ordering, so unlike the
    effect-local searches this deliberately crosses subsequent normal plays.
    It stops as soon as another topdeck can have overwritten the order, or
    when a multi-card/anonymous deck removal makes the top card ambiguous.
    """

    for candidate in records[index + 1 :]:
        move = candidate.move
        if move.seat != actor:
            continue
        if move.to_zone == "deck" and move.movement == "TOPDECK":
            return None
        if move.from_zone != "deck":
            continue
        if move.count != 1:
            return None
        try:
            cards = _record_counter(candidate, mapping)
        except _LiveRecoveryError:
            # The classification rows are already proven; only the optional
            # order label depends on this later removal being named.
            return None
        if _counter_total(cards) != 1:
            return None
        return next(iter(cards))
    return None


def _stops_at_next_normal_play(record: _LiveMoveRecord, actor: int) -> bool:
    return _is_hand_play(record) and record.move.seat == actor and record.normal_action


def _direct_live_rows(
    records: Sequence[_LiveMoveRecord],
    *,
    eligible: Sequence[bool],
    mapping: CardMap,
) -> tuple[list[PublicBuyRow], Counter[str]]:
    """Emit observed normal action, treasure, and live buy decisions."""

    rows: list[PublicBuyRow] = []
    skipped: Counter[str] = Counter()
    for record in records:
        move = record.move
        actor = move.seat
        if actor is None or actor < 0 or actor >= len(eligible) or not eligible[actor]:
            continue
        if (
            record.before is None
            or not record.before.ownership_safe
            or not record.before_hand_exact[actor]
        ):
            continue
        try:
            if _is_hand_play(record) and record.card_name is not None:
                def_id = _def_for_name(record.card_name, mapping, "live action play")
                if record.normal_action:
                    game = _new_live_game(
                        record.before,
                        our_player=actor,
                        current_player=actor,
                        phase="action",
                        mapping=mapping,
                        source_event_index=move.event_index,
                        resource_override=record.action_resources,
                    )
                    _append_game_action(
                        rows,
                        game,
                        actor=actor,
                        action=A_PLAY_BASE + def_id,
                        decision_type="action_play",
                        source_event_index=move.event_index,
                        turn_number=record.before.player_turn_number,
                        observation_quality=OBSERVATION_FULLY_OBSERVED,
                    )
                elif record.card_name in _TREASURE_NAMES:
                    # The first treasure in a turn proves the preceding
                    # action-phase pass.  A later treasure is already in buy.
                    if record.before.phase != "buy":
                        action_game = _new_live_game(
                            record.before,
                            our_player=actor,
                            current_player=actor,
                            phase="action",
                            mapping=mapping,
                            source_event_index=move.event_index,
                        )
                        _append_game_action(
                            rows,
                            action_game,
                            actor=actor,
                            action=A_PASS,
                            decision_type="action_pass",
                            source_event_index=move.event_index,
                            turn_number=record.before.player_turn_number,
                            observation_quality=OBSERVATION_FULLY_OBSERVED,
                        )
                    buy_game = _new_live_game(
                        record.before,
                        our_player=actor,
                        current_player=actor,
                        phase="buy",
                        mapping=mapping,
                        source_event_index=move.event_index,
                    )
                    _append_game_action(
                        rows,
                        buy_game,
                        actor=actor,
                        action=A_PLAY_BASE + def_id,
                        decision_type="treasure_play",
                        source_event_index=move.event_index,
                        turn_number=record.before.player_turn_number,
                        observation_quality=OBSERVATION_FULLY_OBSERVED,
                    )
            if (
                move.movement == "GAIN"
                and move.from_zone == "supply"
                and record.buy_resources is not None
                and record.card_name is not None
            ):
                game = _new_live_game(
                    record.before,
                    our_player=actor,
                    current_player=actor,
                    phase="buy",
                    mapping=mapping,
                    source_event_index=move.event_index,
                    resource_override=record.buy_resources,
                )
                _append_game_action(
                    rows,
                    game,
                    actor=actor,
                    action=A_BUY_BASE + _def_for_name(record.card_name, mapping, "live buy"),
                    decision_type="buy",
                    source_event_index=move.event_index,
                    turn_number=record.before.player_turn_number,
                    observation_quality=OBSERVATION_FULLY_OBSERVED,
                )
        except _LiveRecoveryError:
            skipped["phase_or_play"] += 1
    return rows, skipped


def _merge_live_rows_with_public_buys(
    public_rows: list[PublicBuyRow],
    live_rows: Sequence[PublicBuyRow],
) -> tuple[list[PublicBuyRow], list[PublicBuyRow], int]:
    """Replace, rather than duplicate, public buys proven by live hand state.

    The semantic and live streams have different indexes, but a real buy has
    one exact shared boundary: actor, card/action, turn, coins, and remaining
    buys.  A ``buys`` decrement can otherwise remain pending until an attack
    gives its victim a Curse, so an unmatched live supply gain is not safe
    evidence of a purchase and is deliberately suppressed.
    """

    def signature(row: PublicBuyRow) -> tuple[int, int, int, int, int]:
        return (row.seat_index, row.action, row.turn_number, row.coins, row.buys)

    public_buy_indices: dict[tuple[int, int, int, int, int], list[int]] = defaultdict(list)
    for row_index, row in enumerate(public_rows):
        if row.decision_type == "buy":
            public_buy_indices[signature(row)].append(row_index)

    remaining_live: list[PublicBuyRow] = []
    unmatched_live_buys = 0
    for row in live_rows:
        if row.decision_type != "buy" or row.observation_quality != OBSERVATION_FULLY_OBSERVED:
            remaining_live.append(row)
            continue
        candidates = public_buy_indices.get(signature(row))
        if not candidates:
            unmatched_live_buys += 1
            continue
        public_index = candidates.pop()
        public_rows[public_index] = replace(
            row,
            # Preserve the semantic history index alongside the raw
            # CardMove index so either stream can be audited later.
            source_log_index=public_rows[public_index].source_log_index,
        )
    return public_rows, remaining_live, unmatched_live_buys


def _recover_cellar_or_chapel(
    records: Sequence[_LiveMoveRecord],
    index: int,
    *,
    mapping: CardMap,
    eligible: Sequence[bool],
) -> list[PublicBuyRow]:
    record = records[index]
    assert record.card_name in {"Cellar", "Chapel"}
    if not _is_normal_source(record, eligible=eligible):
        return []
    actor = record.move.seat
    assert actor is not None and record.before is not None
    wanted_movement = "DISCARD" if record.card_name == "Cellar" else "TRASH"
    selected: CounterInt = Counter()
    for candidate in records[index + 1 :]:
        move = candidate.move
        if (
            move.seat == actor
            and move.movement == wanted_movement
            and move.from_zone == "hand"
        ):
            selected.update(_record_counter(candidate, mapping))
            continue
        if _stops_at_next_normal_play(candidate, actor):
            break
        if record.card_name == "Cellar" and move.seat == actor and move.movement == "DRAW":
            break
        if record.card_name == "Chapel" and move.seat == actor and move.movement in {"DISCARD", "DRAW"}:
            break
    game = _effect_game(record, mapping=mapping)
    rows: list[PublicBuyRow] = []
    decision_type = "cellar_discard" if record.card_name == "Cellar" else "chapel_trash"
    _append_select_counter(
        rows,
        game,
        selected,
        actor=actor,
        decision_type=decision_type,
        source_event_index=record.move.event_index,
        turn_number=record.before.player_turn_number,
        observation_quality=OBSERVATION_FULLY_OBSERVED,
    )
    _append_optional_pass(
        rows,
        game,
        actor=actor,
        decision_type=decision_type,
        source_event_index=record.move.event_index,
        turn_number=record.before.player_turn_number,
        observation_quality=OBSERVATION_FULLY_OBSERVED,
    )
    return rows


def _recover_throne_room(
    records: Sequence[_LiveMoveRecord],
    index: int,
    *,
    mapping: CardMap,
    eligible: Sequence[bool],
) -> list[PublicBuyRow]:
    record = records[index]
    if not _is_normal_source(record, eligible=eligible):
        return []
    actor = record.move.seat
    assert actor is not None and record.before is not None
    target: int | None = None
    for candidate in records[index + 1 :]:
        if candidate.move.seat != actor:
            continue
        if _is_hand_play(candidate):
            if candidate.normal_action:
                break
            if candidate.card_name is None:
                raise _LiveRecoveryError("Throne Room target play has no visible identity")
            target = _def_for_name(candidate.card_name, mapping, "Throne Room target")
            break
        if candidate.move.movement in {"DISCARD", "DRAW"} and candidate.move.from_zone == "in-play":
            break
    game = _effect_game(record, mapping=mapping)
    rows: list[PublicBuyRow] = []
    _append_game_action(
        rows,
        game,
        actor=actor,
        action=A_PASS if target is None else A_SELECT_BASE + target,
        decision_type="throne_room_target",
        source_event_index=record.move.event_index,
        turn_number=record.before.player_turn_number,
        observation_quality=OBSERVATION_FULLY_OBSERVED,
    )
    return rows


def _recover_remodel_or_mine(
    records: Sequence[_LiveMoveRecord],
    index: int,
    *,
    mapping: CardMap,
    eligible: Sequence[bool],
) -> list[PublicBuyRow]:
    record = records[index]
    assert record.card_name in {"Remodel", "Mine"}
    if not _is_normal_source(record, eligible=eligible):
        return []
    actor = record.move.seat
    assert actor is not None and record.before is not None
    trash = _next_record(
        records,
        index,
        predicate=lambda candidate: (
            candidate.move.seat == actor
            and candidate.move.movement == "TRASH"
            and candidate.move.from_zone == "hand"
        ),
        stop=lambda candidate: _stops_at_next_normal_play(candidate, actor),
    )
    if trash is None:
        raise _LiveRecoveryError(f"{record.card_name} has no observed hand trash")
    trashed = _record_counter(trash, mapping)
    if _counter_total(trashed) != 1:
        raise _LiveRecoveryError(f"{record.card_name} trash did not select exactly one card")
    gain = _next_record(
        records,
        _record_index(records, trash),
        predicate=lambda candidate: (
            candidate.move.seat == actor
            and candidate.move.movement == "GAIN"
            and candidate.move.from_zone == "supply"
        ),
        stop=lambda candidate: _stops_at_next_normal_play(candidate, actor),
    )
    if gain is None:
        raise _LiveRecoveryError(f"{record.card_name} has no observed gain")
    gained = _record_counter(gain, mapping)
    if _counter_total(gained) != 1:
        raise _LiveRecoveryError(f"{record.card_name} gain did not select exactly one pile")
    game = _effect_game(record, mapping=mapping)
    rows: list[PublicBuyRow] = []
    prefix = record.card_name.lower().replace(" ", "_")
    _append_select_counter(
        rows,
        game,
        trashed,
        actor=actor,
        decision_type=f"{prefix}_trash",
        source_event_index=record.move.event_index,
        turn_number=record.before.player_turn_number,
        observation_quality=OBSERVATION_FULLY_OBSERVED,
    )
    _append_select_counter(
        rows,
        game,
        gained,
        actor=actor,
        decision_type=f"{prefix}_gain",
        source_event_index=record.move.event_index,
        turn_number=record.before.player_turn_number,
        observation_quality=OBSERVATION_FULLY_OBSERVED,
    )
    return rows


def _recover_artisan(
    records: Sequence[_LiveMoveRecord],
    index: int,
    *,
    mapping: CardMap,
    eligible: Sequence[bool],
) -> list[PublicBuyRow]:
    record = records[index]
    if not _is_normal_source(record, eligible=eligible):
        return []
    actor = record.move.seat
    assert actor is not None and record.before is not None
    gain = _next_record(
        records,
        index,
        predicate=lambda candidate: (
            candidate.move.seat == actor
            and candidate.move.movement == "GAIN"
            and candidate.move.from_zone == "supply"
            and candidate.move.to_zone == "hand"
        ),
        stop=lambda candidate: _stops_at_next_normal_play(candidate, actor),
    )
    if gain is None:
        raise _LiveRecoveryError("Artisan has no observed gain to hand")
    topdeck = _next_record(
        records,
        _record_index(records, gain),
        predicate=lambda candidate: (
            candidate.move.seat == actor
            and candidate.move.movement == "TOPDECK"
            and candidate.move.from_zone == "hand"
        ),
        stop=lambda candidate: _stops_at_next_normal_play(candidate, actor),
    )
    if topdeck is None:
        raise _LiveRecoveryError("Artisan has no observed hand topdeck")
    gained = _record_counter(gain, mapping)
    topdecked = _record_counter(topdeck, mapping)
    if _counter_total(gained) != 1 or _counter_total(topdecked) != 1:
        raise _LiveRecoveryError("Artisan choice is not single-card")
    game = _effect_game(record, mapping=mapping)
    rows: list[PublicBuyRow] = []
    _append_select_counter(
        rows,
        game,
        gained,
        actor=actor,
        decision_type="artisan_gain",
        source_event_index=record.move.event_index,
        turn_number=record.before.player_turn_number,
        observation_quality=OBSERVATION_FULLY_OBSERVED,
    )
    # If the server redacted the gained card on arrival, the actual selected
    # topdeck remains recoverable from the outgoing move but the full hand
    # offered to the player was not observed.  Preserve it only as partial.
    quality = (
        OBSERVATION_FULLY_OBSERVED
        if gain.after_hand_exact[actor]
        else OBSERVATION_PARTIALLY_INFERRED
    )
    _append_select_counter(
        rows,
        game,
        topdecked,
        actor=actor,
        decision_type="artisan_topdeck",
        source_event_index=topdeck.move.event_index,
        turn_number=record.before.player_turn_number,
        observation_quality=quality,
    )
    return rows


def _record_index(records: Sequence[_LiveMoveRecord], target: _LiveMoveRecord) -> int:
    """Locate by identity without dataclass-comparing full live snapshots."""

    for index, record in enumerate(records):
        if record is target:
            return index
    raise AssertionError("live move record is absent from its trace")


def _recover_militia(
    records: Sequence[_LiveMoveRecord],
    index: int,
    *,
    mapping: CardMap,
    eligible: Sequence[bool],
) -> tuple[list[PublicBuyRow], dict[str, object] | None]:
    """Recover a victim's retained-three action sequence from a hand delta."""

    source = records[index]
    attacker = source.move.seat
    if attacker is None:
        return [], None
    discard = _next_record(
        records,
        index,
        predicate=lambda candidate: (
            candidate.move.seat is not None
            and candidate.move.seat != attacker
            and candidate.move.movement == "DISCARD"
            and candidate.move.from_zone == "hand"
        ),
        stop=lambda candidate: _stops_at_next_normal_play(candidate, attacker),
    )
    if (
        discard is None
        or discard.before is None
        or not discard.before.ownership_safe
        or discard.move.seat is None
    ):
        raise _LiveRecoveryError("Militia has no victim hand discard in the live trace")
    victim = discard.move.seat
    if victim < 0 or victim >= len(eligible) or not eligible[victim]:
        return [], None
    if not discard.before_hand_exact[victim]:
        raise _LiveRecoveryError("Militia victim hand was not fully visible")
    discarded = _record_counter(discard, mapping)
    before_hand = discard.before.aggregate(victim, "hand").known
    kept = _counter_subtract(
        before_hand,
        discarded,
        context="Militia kept-card derivation",
    )
    if _counter_total(kept) != 3:
        raise _LiveRecoveryError(
            f"Militia victim keeps {_counter_total(kept)} cards rather than exactly three"
        )
    game = _new_live_game(
        discard.before,
        our_player=victim,
        current_player=attacker,
        phase="action",
        mapping=mapping,
        source_event_index=discard.move.event_index,
        interrupt={"kind": "militia_discard", "attacker": attacker, "defender": victim},
    )
    rows: list[PublicBuyRow] = []
    _append_select_counter(
        rows,
        game,
        kept,
        actor=victim,
        decision_type="militia_keep",
        source_event_index=discard.move.event_index,
        turn_number=discard.before.player_turn_number,
        observation_quality=OBSERVATION_FULLY_OBSERVED,
    )
    example = {
        "event_index": discard.move.event_index,
        "attacker": attacker,
        "victim": victim,
        "hand_before": _named_counter(before_hand, mapping),
        "discarded": _named_counter(discarded, mapping),
        "hand_after": _named_counter(discard.after.aggregate(victim, "hand").known, mapping)
        if discard.after is not None
        else [],
    }
    return rows, example


def _recover_bandit(
    records: Sequence[_LiveMoveRecord],
    index: int,
    *,
    mapping: CardMap,
    eligible: Sequence[bool],
) -> list[PublicBuyRow]:
    """Seed the native Bandit interrupt after two observed reveals."""

    source = records[index]
    attacker = source.move.seat
    if attacker is None:
        return []
    trash = _next_record(
        records,
        index,
        predicate=lambda candidate: (
            candidate.move.seat is not None
            and candidate.move.seat != attacker
            and candidate.move.movement == "TRASH"
            and candidate.move.from_zone == "set-aside"
        ),
        stop=lambda candidate: _stops_at_next_normal_play(candidate, attacker),
    )
    if (
        trash is None
        or trash.before is None
        or not trash.before.ownership_safe
        or trash.move.seat is None
    ):
        return []
    victim = trash.move.seat
    if victim < 0 or victim >= len(eligible) or not eligible[victim]:
        return []
    revealed = trash.before.aggregate(victim, "set-aside")
    if revealed.anonymous or _counter_total(revealed.known) != 2:
        raise _LiveRecoveryError("Bandit revealed cards were not exactly visible")
    trashable = [
        def_id
        for def_id in revealed.known
        if mapping.def_to_name[def_id] in {"Silver", "Gold"}
    ]
    # Bandit auto-resolves when there is at most one distinct eligible
    # Treasure.  A later Trash CardMove in that case is an outcome, not a
    # player decision and must not become an imitation label.
    if len(trashable) != 2 or trashable[0] == trashable[1]:
        return []
    chosen = _record_counter(trash, mapping)
    if _counter_total(chosen) != 1:
        raise _LiveRecoveryError("Bandit trash did not select exactly one revealed card")
    game = _new_live_game(
        trash.before,
        our_player=victim,
        current_player=attacker,
        phase="action",
        mapping=mapping,
        source_event_index=trash.move.event_index,
        interrupt={"kind": "bandit_trash", "attacker": attacker, "defender": victim},
    )
    rows: list[PublicBuyRow] = []
    _append_select_counter(
        rows,
        game,
        chosen,
        actor=victim,
        decision_type="bandit_trash",
        source_event_index=trash.move.event_index,
        turn_number=trash.before.player_turn_number,
        observation_quality=OBSERVATION_FULLY_OBSERVED,
    )
    return rows


def _recover_poacher(
    records: Sequence[_LiveMoveRecord],
    index: int,
    *,
    mapping: CardMap,
    eligible: Sequence[bool],
) -> list[PublicBuyRow]:
    record = records[index]
    if not _is_normal_source(record, eligible=eligible):
        return []
    actor = record.move.seat
    assert actor is not None and record.before is not None
    draw = _next_record(
        records,
        index,
        predicate=lambda candidate: (
            candidate.move.seat == actor
            and candidate.move.movement == "DRAW"
            and candidate.move.from_zone == "deck"
            and candidate.move.to_zone == "hand"
        ),
        stop=lambda candidate: _stops_at_next_normal_play(candidate, actor),
    )
    if draw is None or not draw.after_hand_exact[actor]:
        raise _LiveRecoveryError("Poacher draw did not leave a fully visible hand")
    drawn = _record_counter(draw, mapping)
    if _counter_total(drawn) != 1:
        raise _LiveRecoveryError("Poacher did not draw exactly one observed card")
    discarded: CounterInt = Counter()
    draw_index = _record_index(records, draw)
    for candidate in records[draw_index + 1 :]:
        if (
            candidate.move.seat == actor
            and candidate.move.movement == "DISCARD"
            and candidate.move.from_zone == "hand"
        ):
            discarded.update(_record_counter(candidate, mapping))
            continue
        if _stops_at_next_normal_play(candidate, actor):
            break
        if candidate.move.seat == actor and candidate.move.movement in {"PLAY", "GAIN", "TRASH", "TOPDECK"}:
            break
    game = _effect_game(record, mapping=mapping, forced_deck=list(drawn.elements()))
    rows: list[PublicBuyRow] = []
    _append_select_counter(
        rows,
        game,
        discarded,
        actor=actor,
        decision_type="poacher_discard",
        source_event_index=record.move.event_index,
        turn_number=record.before.player_turn_number,
        observation_quality=OBSERVATION_FULLY_OBSERVED,
    )
    return rows


def _recover_harbinger(
    records: Sequence[_LiveMoveRecord],
    index: int,
    *,
    mapping: CardMap,
    eligible: Sequence[bool],
) -> list[PublicBuyRow]:
    record = records[index]
    if not _is_normal_source(record, eligible=eligible):
        return []
    actor = record.move.seat
    assert actor is not None and record.before is not None
    draw = _next_record(
        records,
        index,
        predicate=lambda candidate: (
            candidate.move.seat == actor
            and candidate.move.movement == "DRAW"
            and candidate.move.from_zone == "deck"
            and candidate.move.to_zone == "hand"
        ),
        stop=lambda candidate: _stops_at_next_normal_play(candidate, actor),
    )
    if draw is None or not draw.after_hand_exact[actor]:
        raise _LiveRecoveryError("Harbinger draw did not leave a fully visible hand")
    drawn = _record_counter(draw, mapping)
    if _counter_total(drawn) != 1:
        raise _LiveRecoveryError("Harbinger did not draw exactly one observed card")
    topdeck = _next_record(
        records,
        _record_index(records, draw),
        predicate=lambda candidate: (
            candidate.move.seat == actor
            and candidate.move.movement == "TOPDECK"
            and candidate.move.from_zone == "discard"
        ),
        stop=lambda candidate: _stops_at_next_normal_play(candidate, actor),
    )
    game = _effect_game(record, mapping=mapping, forced_deck=list(drawn.elements()))
    rows: list[PublicBuyRow] = []
    quality = (
        OBSERVATION_FULLY_OBSERVED
        if record.before.zone_is_exact(actor, "discard")
        else OBSERVATION_PARTIALLY_INFERRED
    )
    if topdeck is not None:
        chosen = _record_counter(topdeck, mapping)
        if _counter_total(chosen) != 1:
            raise _LiveRecoveryError("Harbinger topdeck did not select exactly one card")
        _append_select_counter(
            rows,
            game,
            chosen,
            actor=actor,
            decision_type="harbinger_topdeck",
            source_event_index=topdeck.move.event_index,
            turn_number=record.before.player_turn_number,
            observation_quality=quality,
        )
    else:
        _append_optional_pass(
            rows,
            game,
            actor=actor,
            decision_type="harbinger_topdeck",
            source_event_index=record.move.event_index,
            turn_number=record.before.player_turn_number,
            observation_quality=quality,
    )
    return rows


def _recover_vassal(
    records: Sequence[_LiveMoveRecord],
    index: int,
    *,
    mapping: CardMap,
    eligible: Sequence[bool],
) -> list[PublicBuyRow]:
    record = records[index]
    if not _is_normal_source(record, eligible=eligible):
        return []
    actor = record.move.seat
    assert actor is not None and record.before is not None
    discarded = _next_record(
        records,
        index,
        predicate=lambda candidate: (
            candidate.move.seat == actor
            and candidate.move.movement == "DISCARD"
            and candidate.move.from_zone == "deck"
        ),
        stop=lambda candidate: _stops_at_next_normal_play(candidate, actor),
    )
    if discarded is None:
        return []
    cards = _record_counter(discarded, mapping)
    if _counter_total(cards) != 1:
        raise _LiveRecoveryError("Vassal discarded more than one card")
    revealed_def = next(iter(cards))
    revealed_name = mapping.def_to_name[revealed_def]
    if revealed_name not in _ACTION_CARD_NAMES:
        # It was not an Action, so Vassal presented no yes/no decision.
        return []
    game = _effect_game(record, mapping=mapping, forced_deck=[revealed_def])
    # A forced play from discard is the only observed affirmative outcome.
    play_from_discard = _next_record(
        records,
        _record_index(records, discarded),
        predicate=lambda candidate: (
            candidate.move.seat == actor
            and candidate.move.movement == "PLAY"
            and candidate.move.from_zone == "discard"
        ),
        stop=lambda candidate: _stops_at_next_normal_play(candidate, actor),
    )
    rows: list[PublicBuyRow] = []
    _append_game_action(
        rows,
        game,
        actor=actor,
        # Engine option 1 means play; option 0 means decline.
        action=A_OPTION_BASE + (1 if play_from_discard is not None else 0),
        decision_type="vassal_option",
        source_event_index=discarded.move.event_index,
        turn_number=record.before.player_turn_number,
        observation_quality=OBSERVATION_FULLY_OBSERVED,
    )
    return rows


def _recover_sentry(
    records: Sequence[_LiveMoveRecord],
    index: int,
    *,
    mapping: CardMap,
    eligible: Sequence[bool],
) -> list[PublicBuyRow]:
    """Recover Sentry's option sequence and, when proven, its topdeck order."""

    record = records[index]
    if not _is_normal_source(record, eligible=eligible):
        return []
    actor = record.move.seat
    assert actor is not None and record.before is not None
    draw = _next_record(
        records,
        index,
        predicate=lambda candidate: (
            candidate.move.seat == actor
            and candidate.move.movement == "DRAW"
            and candidate.move.from_zone == "deck"
            and candidate.move.to_zone == "hand"
        ),
        stop=lambda candidate: _stops_at_next_normal_play(candidate, actor),
    )
    if draw is None or not draw.after_hand_exact[actor]:
        raise _LiveRecoveryError("Sentry draw did not leave a fully visible hand")
    drawn = _record_counter(draw, mapping)
    if _counter_total(drawn) != 1:
        raise _LiveRecoveryError("Sentry did not draw exactly one card")
    reveal = _next_record(
        records,
        _record_index(records, draw),
        predicate=lambda candidate: (
            candidate.move.seat == actor
            and candidate.move.movement in {"LOOK_AT", "REVEAL"}
            and candidate.move.from_zone == "deck"
            and candidate.move.to_zone == "set-aside"
        ),
        stop=lambda candidate: _stops_at_next_normal_play(candidate, actor),
    )
    if reveal is None:
        return []
    revealed = _record_counter(reveal, mapping)
    reveal_order = [
        _def_for_name(name, mapping, "Sentry reveal")
        for name in reveal.move.source_cards
        if name is not None
    ]
    if len(reveal_order) != reveal.move.count or not 1 <= len(reveal_order) <= 2:
        raise _LiveRecoveryError("Sentry revealed cards are not fully visible")
    forced = [*drawn.elements(), *reveal_order]
    trash: CounterInt = Counter()
    discard: CounterInt = Counter()
    topdeck_record: _LiveMoveRecord | None = None
    reveal_index = _record_index(records, reveal)
    for candidate in records[reveal_index + 1 :]:
        move = candidate.move
        if move.seat == actor and move.from_zone == "set-aside":
            if move.movement == "TRASH":
                trash.update(_record_counter(candidate, mapping))
                continue
            if move.movement == "DISCARD":
                discard.update(_record_counter(candidate, mapping))
                continue
            if move.movement == "TOPDECK":
                topdeck_record = candidate
                break
        if _stops_at_next_normal_play(candidate, actor):
            break
    # Duplicate revealed card names with different outcomes are genuinely
    # ambiguous from the aggregate visible transition.  Do not invent which
    # reveal position got which Sentry option.
    outcomes_by_def: dict[int, set[int]] = defaultdict(set)
    for def_id, count in trash.items():
        if count:
            outcomes_by_def[def_id].add(0)
    for def_id, count in discard.items():
        if count:
            outcomes_by_def[def_id].add(1)
    kept_counter = Counter(revealed)
    _counter_add(kept_counter, trash, -1, context="Sentry trash outcome")
    _counter_add(kept_counter, discard, -1, context="Sentry discard outcome")
    for def_id, count in kept_counter.items():
        if count:
            outcomes_by_def[def_id].add(2)
    if any(len(outcomes_by_def.get(def_id, {2})) > 1 for def_id in reveal_order):
        raise _LiveRecoveryError("Sentry duplicate revealed cards have different outcomes")
    option_values = [next(iter(outcomes_by_def.get(def_id, {2}))) for def_id in reveal_order]
    game = _effect_game(record, mapping=mapping, forced_deck=forced)
    rows: list[PublicBuyRow] = []
    for option in option_values:
        _append_game_action(
            rows,
            game,
            actor=actor,
            action=A_OPTION_BASE + option,
            decision_type="sentry_option",
            source_event_index=reveal.move.event_index,
            turn_number=record.before.player_turn_number,
            observation_quality=OBSERVATION_FULLY_OBSERVED,
        )
    kept_order = [def_id for def_id, option in zip(reveal_order, option_values, strict=True) if option == 2]
    if len(kept_order) != 2 or kept_order[0] == kept_order[1] or topdeck_record is None:
        return rows
    next_def = _next_proven_deck_top(
        records,
        _record_index(records, topdeck_record),
        actor=actor,
        mapping=mapping,
    )
    if next_def is None:
        return rows
    if next_def == kept_order[0]:
        order_option = 0
    elif next_def == kept_order[1]:
        order_option = 1
    else:
        return rows
    _append_game_action(
        rows,
        game,
        actor=actor,
        action=A_OPTION_BASE + order_option,
        decision_type="sentry_order",
        source_event_index=topdeck_record.move.event_index,
        turn_number=record.before.player_turn_number,
        observation_quality=OBSERVATION_FULLY_OBSERVED,
    )
    return rows


def _recover_library(
    records: Sequence[_LiveMoveRecord],
    index: int,
    *,
    mapping: CardMap,
    eligible: Sequence[bool],
) -> list[PublicBuyRow]:
    """Replay Library's visible deck sequence through its yes/no choices."""

    record = records[index]
    if not _is_normal_source(record, eligible=eligible):
        return []
    actor = record.move.seat
    assert actor is not None and record.before is not None
    forced: list[int] = []
    options: list[int] = []
    cursor = index + 1
    while cursor < len(records):
        candidate = records[cursor]
        move = candidate.move
        if _stops_at_next_normal_play(candidate, actor):
            break
        if move.seat != actor:
            cursor += 1
            continue
        if (
            move.movement == "DRAW"
            and move.from_zone == "deck"
            and move.to_zone == "hand"
        ):
            if not candidate.after_hand_exact[actor]:
                raise _LiveRecoveryError("Library draw made the acting hand anonymous")
            forced.extend(_record_counter(candidate, mapping).elements())
            cursor += 1
            continue
        if (
            move.movement in {"LOOK_AT", "REVEAL"}
            and move.from_zone == "deck"
            and move.to_zone == "set-aside"
        ):
            revealed = _record_counter(candidate, mapping)
            if _counter_total(revealed) != 1:
                raise _LiveRecoveryError("Library look-at did not expose one Action")
            forced.extend(revealed.elements())
            next_candidate = records[cursor + 1] if cursor + 1 < len(records) else None
            keep = (
                next_candidate is not None
                and next_candidate.move.seat == actor
                and next_candidate.move.movement == "RETURN_TO"
                and next_candidate.move.from_zone == "set-aside"
                and next_candidate.move.to_zone == "hand"
            )
            if keep and not next_candidate.after_hand_exact[actor]:
                raise _LiveRecoveryError("Library kept Action made the hand anonymous")
            options.append(0 if keep else 1)
            cursor += 1
            continue
        cursor += 1
    if not options:
        return []
    game = _effect_game(record, mapping=mapping, forced_deck=forced)
    rows: list[PublicBuyRow] = []
    for option in options:
        _append_game_action(
            rows,
            game,
            actor=actor,
            action=A_OPTION_BASE + option,
            decision_type="library_option",
            source_event_index=record.move.event_index,
            turn_number=record.before.player_turn_number,
            observation_quality=OBSERVATION_FULLY_OBSERVED,
        )
    return rows


def _named_counter(counter: CounterInt, mapping: CardMap) -> list[str]:
    """Return repeated card names for small audit examples."""

    result: list[str] = []
    for def_id in sorted(counter):
        result.extend([mapping.def_to_name[def_id]] * counter[def_id])
    return result


def _recover_trace_rows(
    records: Sequence[_LiveMoveRecord],
    *,
    eligible: Sequence[bool],
    mapping: CardMap,
) -> tuple[list[PublicBuyRow], Counter[str], list[dict[str, object]]]:
    """Recover rows from one coherent FullState-to-reconnect window."""

    direct_rows, skipped = _direct_live_rows(records, eligible=eligible, mapping=mapping)
    recovered: list[PublicBuyRow] = []
    militia_examples: list[dict[str, object]] = []
    for index, record in enumerate(records):
        name = record.card_name
        if not _is_hand_play(record) or name is None:
            continue
        try:
            if name == "Cellar":
                recovered.extend(_recover_cellar_or_chapel(records, index, mapping=mapping, eligible=eligible))
            elif name == "Chapel":
                recovered.extend(_recover_cellar_or_chapel(records, index, mapping=mapping, eligible=eligible))
            elif name == "Throne Room":
                recovered.extend(_recover_throne_room(records, index, mapping=mapping, eligible=eligible))
            elif name in {"Remodel", "Mine"}:
                recovered.extend(_recover_remodel_or_mine(records, index, mapping=mapping, eligible=eligible))
            elif name == "Artisan":
                recovered.extend(_recover_artisan(records, index, mapping=mapping, eligible=eligible))
            elif name == "Poacher":
                recovered.extend(_recover_poacher(records, index, mapping=mapping, eligible=eligible))
            elif name == "Harbinger":
                recovered.extend(_recover_harbinger(records, index, mapping=mapping, eligible=eligible))
            elif name == "Vassal":
                recovered.extend(_recover_vassal(records, index, mapping=mapping, eligible=eligible))
            elif name == "Sentry":
                recovered.extend(_recover_sentry(records, index, mapping=mapping, eligible=eligible))
            elif name == "Library":
                recovered.extend(_recover_library(records, index, mapping=mapping, eligible=eligible))
            elif name == "Militia":
                rows, example = _recover_militia(records, index, mapping=mapping, eligible=eligible)
                recovered.extend(rows)
                if example is not None and len(militia_examples) < 8:
                    militia_examples.append(example)
            elif name == "Bandit":
                recovered.extend(_recover_bandit(records, index, mapping=mapping, eligible=eligible))
        except (ConversionError, _LiveRecoveryError):
            skipped[name.lower().replace(" ", "_")] += 1
    return [*direct_rows, *recovered], skipped, militia_examples


def _recover_live_rows(
    capture: ProtocolCapture,
    *,
    game_id: str,
    outcome: Outcome,
    mapping: CardMap,
) -> RecoveredLiveRows:
    """Recover only live tuples whose visibility is sound across snapshots."""

    segments = capture.live_segments or (
        LiveSegment(
            full_state=capture.full_state,
            initial_log_entries=capture.initial_log_entries,
            live_events=capture.live_events,
        ),
    )
    skipped: Counter[str] = Counter()
    traces: list[tuple[list[_LiveMoveRecord] | None, _LiveState | None, VisibilitySummary]] = []
    for segment in segments:
        segment_capture = replace(
            capture,
            full_state=segment.full_state,
            initial_log_entries=segment.initial_log_entries,
            live_events=segment.live_events,
        )
        try:
            records, state = _build_live_trace(segment_capture, game_id=game_id, mapping=mapping)
        except (ConversionError, _LiveRecoveryError, ValueError):
            # A failure in one reconnect window does not turn its initial
            # hand contents into evidence of sustained visibility.  Earlier
            # proven rows remain valid, but later windows are not eligible.
            initial = _initial_visibility_from_full_state(segment_capture, len(outcome.seats))
            summary = VisibilitySummary(
                initial_visible=initial,
                visible_throughout=tuple(False for _ in outcome.seats),
                first_unknown_event=tuple(None for _ in outcome.seats),
                grade="buy_only",
            )
            traces.append((None, None, summary))
            skipped["live_trace"] += 1
            continue
        traces.append((records, state, _visibility_summary(state)))

    initial = traces[0][2].initial_visible
    visible = list(initial)
    first_unknown = [None for _ in outcome.seats]
    trace_eligibility: list[tuple[bool, ...]] = []
    for _records, _state, summary in traces:
        eligible = tuple(
            visible[seat] and summary.visible_throughout[seat]
            for seat in range(len(outcome.seats))
        )
        for seat, allowed in enumerate(eligible):
            if visible[seat] and not allowed:
                first_unknown[seat] = summary.first_unknown_event[seat]
            visible[seat] = allowed
        trace_eligibility.append(eligible)

    visible_tuple = tuple(visible)
    count = sum(visible_tuple)
    summary = VisibilitySummary(
        initial_visible=initial,
        visible_throughout=visible_tuple,
        first_unknown_event=tuple(first_unknown),
        grade="full" if count == len(visible_tuple) else "partial" if count else "buy_only",
        segment_count=len(segments),
    )
    final_state = traces[-1][1]
    if final_state is None or not final_state.ownership_safe:
        skipped["live_ownership"] += 1
        return RecoveredLiveRows(
            rows=(),
            visibility=summary,
            skipped_by_type=dict(sorted(skipped.items())),
            militia_examples=(),
        )
    if len(final_state.collections) != len(outcome.seats):
        skipped["live_ownership"] += 1
        return RecoveredLiveRows(
            rows=(),
            visibility=summary,
            skipped_by_type=dict(sorted(skipped.items())),
            militia_examples=(),
        )
    live_matches = tuple(
        _normalized_counter(final_state.collections[seat])
        == _normalized_counter(outcome.seats[seat].final_deck)
        for seat in range(len(outcome.seats))
    )
    if not all(live_matches):
        # Do not bridge a live tail from ownership that is not independently
        # conserved all the way to GameResult.  The primary semantic gate has
        # already established final-deck truth; this is an extra gate for the
        # optional recovered rows from every reconnect window.
        skipped["live_ownership_final_mismatch"] += 1
        return RecoveredLiveRows(
            rows=(),
            visibility=summary,
            skipped_by_type=dict(sorted(skipped.items())),
            militia_examples=(),
        )

    rows: list[PublicBuyRow] = []
    militia_examples: list[dict[str, object]] = []
    for (records, _state, _segment_summary), eligible in zip(traces, trace_eligibility, strict=True):
        if records is None or not any(eligible):
            continue
        segment_rows, segment_skips, segment_examples = _recover_trace_rows(
            records,
            eligible=eligible,
            mapping=mapping,
        )
        rows.extend(segment_rows)
        skipped.update(segment_skips)
        for example in segment_examples:
            if len(militia_examples) < 8:
                militia_examples.append(example)
    return RecoveredLiveRows(
        rows=tuple(rows),
        visibility=summary,
        skipped_by_type=dict(sorted(skipped.items())),
        militia_examples=tuple(militia_examples),
    )


def _convert_complete_capture(
    raw_path: Path,
    manifest: dict[str, object],
    mapping: CardMap,
) -> ConvertedGame:
    """Reconstruct, verify, and encode one manifest-marked complete capture."""

    capture = _load_protocol_capture(raw_path)
    game_id = str(manifest.get("game_id", capture.game_start.game_id))
    if int(game_id) != capture.game_start.game_id:
        raise ConversionError(
            f"{raw_path}: manifest game_id {game_id} does not match protocol {capture.game_start.game_id}"
        )
    if len(capture.game_start.player_ids) != 2:
        raise ConversionError(f"{raw_path}: only two-player captures are supported")
    if len(capture.game_start.kingdom) != 10:
        raise ConversionError(f"{raw_path}: expected ten kingdom piles, got {len(capture.game_start.kingdom)}")
    _validate_base_capture(capture, mapping)
    outcome = _parse_outcome(manifest, capture.game_start, mapping)
    card_totals = _counter_from_named_pairs(capture.full_state.card_counts, mapping, context="FullState card totals")
    if not card_totals:
        raise ConversionError(f"{raw_path}: FullState has no card totals")

    collections = [Counter(_counter_from_names(STARTING_DECK, mapping)) for _ in outcome.seats]
    trash: CounterInt = Counter()
    in_play = [Counter() for _ in outcome.seats]
    known_discard = [Counter() for _ in outcome.seats]
    # Each player normally receives an initial five-card hand before the first
    # normal turn.  Depth-zero Draw entries replace this with the observed
    # initial/cleanup count (which can be less than five in a thinned deck).
    hand_counts: list[int] = [STARTING_HAND_SIZE for _ in outcome.seats]
    # Once cleanup has drawn a new hand, the next turn boundary still needs to
    # emit the just-finished turn's pass using its *pre*-cleanup hand.  Keep
    # that one snapshot value while ``hand_counts`` advances immediately, so
    # attacks and Council Room draws between cleanup and the next turn operate
    # on the real new hand.
    cleanup_pass_hand_counts: list[int | None] = [None for _ in outcome.seats]
    coins = [0 for _ in outcome.seats]
    treasure_coins = [0 for _ in outcome.seats]
    coins_spent = [0 for _ in outcome.seats]
    buy_counts = [0 for _ in outcome.seats]
    current_turn_seat: int | None = None
    current_turn_number = 0
    global_turn_index = -1
    turn_had_purchase = False
    rows: list[PublicBuyRow] = []
    per_seat_decisions = [0 for _ in outcome.seats]
    observed_resource_buy_rows = 0
    effect_cards: dict[int, str] = {}

    def available_coins(seat: int) -> int:
        return coins[seat] + treasure_coins[seat] - coins_spent[seat]

    def emit_pass_if_observable(source_index: int, *, terminal: bool = False) -> None:
        """Emit a demonstrated end-buy pass when the public boundary proves it."""

        if current_turn_seat is None:
            return
        # A terminal purchase can end the game while unspent buys remain.  No
        # post-game pass was demonstrated in that case.
        if terminal and turn_had_purchase:
            return
        if buy_counts[current_turn_seat] <= 0:
            return
        # A start of the next turn (or a terminal no-buy turn) is an observable
        # end-of-buy boundary.  This remains sound for Market/Festival/Council
        # Room turns that retain buys after one or more purchases.
        snapshot_hand_counts = list(hand_counts)
        cleanup_hand_count = cleanup_pass_hand_counts[current_turn_seat]
        if cleanup_hand_count is not None:
            snapshot_hand_counts[current_turn_seat] = cleanup_hand_count
        rows.append(
            _encode_row(
                game_id=game_id,
                source_log_index=source_index,
                action=A_PASS,
                actor=current_turn_seat,
                player_turn_number=current_turn_number,
                global_turn_index=global_turn_index,
                collections=collections,
                card_totals=card_totals,
                trash=trash,
                in_play=in_play,
                known_discard=known_discard,
                hand_counts=snapshot_hand_counts,
                coins=available_coins(current_turn_seat),
                buys=buy_counts[current_turn_seat],
                kingdom=capture.game_start.kingdom,
                mapping=mapping,
                ply_index=len(rows),
            )
        )
        per_seat_decisions[current_turn_seat] += 1
        buy_counts[current_turn_seat] = 0

    for entry in capture.log_entries:
        if entry.name == LOG_STARTING_CARDS:
            _validate_starting_cards(entry, mapping, len(outcome.seats), raw_path)
            continue

        if entry.name == LOG_TURN_DESCRIPTION:
            turn = _turn_description(entry, raw_path)
            if turn is None or turn[2] != 0:
                continue
            owner, player_turn_number, _turn_type, controller = turn
            if owner != controller:
                raise ConversionError(
                    f"{raw_path}: turn log {entry.index} has nonstandard controller {controller} for owner {owner}"
                )
            _check_seat(owner, len(outcome.seats), raw_path, entry.index)
            emit_pass_if_observable(entry.index)
            if current_turn_seat is not None:
                # Cleanup completes before the next turn boundary.  Its
                # depth-zero Draw has already set this player's exact next
                # hand size (which can be fewer than five in a thinned deck),
                # so only the public play area needs clearing here.
                cleanup_pass_hand_counts[current_turn_seat] = None
                in_play[current_turn_seat].clear()
            # Preserve face-up discard facts only within the active turn.  An
            # older card may have crossed a hidden shuffle/draw boundary, so
            # retaining it would make a sampled snapshot inconsistent.
            for discard in known_discard:
                discard.clear()
            effect_cards.clear()
            current_turn_seat = owner
            current_turn_number = player_turn_number
            global_turn_index += 1
            turn_had_purchase = False
            # Do not reset the incoming player's hand here.  Their preceding
            # cleanup drew five cards when *their* last turn ended, but an
            # opponent can subsequently change that hand before this boundary
            # (Council Room draws one, Militia discards down to three).  Those
            # depth-one logs have already updated ``hand_counts[owner]``.
            # Resetting it to five erased that public information and made
            # later action/treasure plays appear to create or destroy cards.
            coins[owner] = 0
            treasure_coins[owner] = 0
            coins_spent[owner] = 0
            buy_counts[owner] = 1
            continue

        if entry.name == LOG_SHUFFLE:
            seat = _seat_argument(entry, raw_path)
            _check_seat(seat, len(outcome.seats), raw_path, entry.index)
            # A public shuffle makes previous discard identities hidden again.
            known_discard[seat].clear()
            continue

        if entry.name == LOG_PLAY:
            seat = _seat_argument(entry, raw_path)
            _check_seat(seat, len(outcome.seats), raw_path, entry.index)
            cards = _require_known_cards(entry, mapping, raw_path, operation="play")
            if _counter_total(cards) != 1:
                raise ConversionError(
                    f"{raw_path}: play at semantic log {entry.index} has "
                    f"{_counter_total(cards)} cards, expected one"
                )
            parent = _effect_source(effect_cards, entry.depth)
            if entry.depth == 0 or parent == "Throne Room":
                _change_hand_count(
                    hand_counts,
                    seat,
                    -1,
                    raw_path=raw_path,
                    log_index=entry.index,
                    reason="play from hand",
                )
            elif parent == "Vassal":
                # Vassal first discards the revealed deck card, then may play
                # it from discard.  Its prior discard evidence is no longer a
                # current discard fact once the card enters play.
                known_discard[seat].clear()
            else:
                raise ConversionError(
                    f"{raw_path}: nested play at log {entry.index} has unsupported "
                    f"source effect {parent!r}"
                )
            in_play[seat].update(cards)
            _replace_effect_card(
                effect_cards,
                entry.depth,
                _single_card_name(cards, mapping, raw_path, entry.index),
            )
            continue

        if entry.name == LOG_TREASURES:
            seat = _seat_argument(entry, raw_path)
            _check_seat(seat, len(outcome.seats), raw_path, entry.index)
            cards = _require_known_cards(entry, mapping, raw_path, operation="treasure play")
            total = _int_argument(entry, raw_path)
            if total < 0:
                raise ConversionError(f"{raw_path}: negative treasure coins at log {entry.index}")
            # A semantic LOG_TREASURES entry is one (possibly coalesced)
            # treasure-play group, not the player's entire in-play treasure
            # set.  Replacements of *this same log index* have already been
            # collapsed by _collect_log_entries, while a later index is an
            # additional group.  Replacing every previous treasure here both
            # loses coins from earlier groups and corrupts the hand count when
            # a later group is smaller than an earlier one.
            in_play[seat].update(cards)
            _change_hand_count(
                hand_counts,
                seat,
                -_counter_total(cards),
                raw_path=raw_path,
                log_index=entry.index,
                reason="treasure play",
            )
            treasure_coins[seat] += total
            continue

        if entry.name == LOG_BUY_BONUS:
            seat = _seat_argument(entry, raw_path)
            _check_seat(seat, len(outcome.seats), raw_path, entry.index)
            bonus = _int_argument(entry, raw_path)
            if bonus < 0:
                raise ConversionError(f"{raw_path}: negative buy bonus at log {entry.index}")
            buy_counts[seat] += bonus
            continue

        if entry.name in (
            LOG_COIN_BONUS_ONE,
            LOG_COIN_BONUS,
            LOG_CARD_COIN_BONUS_ONE,
            LOG_CARD_COIN_BONUS,
        ):
            seat = _seat_argument(entry, raw_path)
            _check_seat(seat, len(outcome.seats), raw_path, entry.index)
            coins[seat] += _int_argument(entry, raw_path)
            continue

        if entry.name == LOG_DRAW:
            # Deliberately consume only the count.  The identities present in
            # the semantic argument are unavailable spectator information.
            seat = _seat_argument(entry, raw_path)
            _check_seat(seat, len(outcome.seats), raw_path, entry.index)
            draw_count = _hidden_card_count(entry, raw_path)
            if entry.depth == 0:
                # In the base-only domain, depth-zero Draw entries are the
                # initial deal or cleanup draw.  They replace the old hand,
                # rather than add to it; a heavily trashed deck can draw less
                # than five.  Opponent effects between this cleanup and the
                # following TurnDescription are then applied as deltas below.
                if seat == current_turn_seat:
                    cleanup_pass_hand_counts[seat] = hand_counts[seat]
                hand_counts[seat] = draw_count
            else:
                _change_hand_count(
                    hand_counts,
                    seat,
                    draw_count,
                    raw_path=raw_path,
                    log_index=entry.index,
                    reason="card draw",
                )
            continue

        if entry.name == LOG_REVEAL:
            # Library moves every looked-at card through a public reveal.  A
            # kept card returns to the hidden hand without a Draw log, while
            # an Action set aside is later logged as a discard.  Count it now
            # and reconcile that set-aside discard below.
            source = _effect_source(effect_cards, entry.depth)
            if source == "Library":
                seat = _seat_argument(entry, raw_path)
                _check_seat(seat, len(outcome.seats), raw_path, entry.index)
                cards = _card_argument(entry, mapping, raw_path)
                _change_hand_count(
                    hand_counts,
                    seat,
                    cards.total,
                    raw_path=raw_path,
                    log_index=entry.index,
                    reason="Library reveal provisionally kept in hand",
                )
            continue

        if entry.name == LOG_DISCARD:
            seat = _seat_argument(entry, raw_path)
            _check_seat(seat, len(outcome.seats), raw_path, entry.index)
            cards = _card_argument(entry, mapping, raw_path)
            source = _effect_source(effect_cards, entry.depth)
            if source in HAND_DISCARD_EFFECTS:
                _change_hand_count(
                    hand_counts,
                    seat,
                    -cards.total,
                    raw_path=raw_path,
                    log_index=entry.index,
                    reason=f"{source} discard from hand",
                )
            elif source == "Library":
                # Library set-asides were provisionally counted when revealed
                # so the post-effect hand size is exact even without its
                # hidden offered-element list.
                _change_hand_count(
                    hand_counts,
                    seat,
                    -cards.total,
                    raw_path=raw_path,
                    log_index=entry.index,
                    reason="Library set-aside discard",
                )
            elif source not in REVEALED_DISCARD_EFFECTS:
                raise ConversionError(
                    f"{raw_path}: discard at log {entry.index} has unsupported "
                    f"source effect {source!r}"
                )
            # A BACK slot tells us only that a hidden card was discarded.  Do
            # not fabricate a card identity for the public discard snapshot.
            known_discard[seat].update(cards.known)
            continue

        if entry.name == LOG_TRASH:
            seat = _seat_argument(entry, raw_path)
            _check_seat(seat, len(outcome.seats), raw_path, entry.index)
            cards = _require_known_cards(entry, mapping, raw_path, operation="trash")
            source = _effect_source(effect_cards, entry.depth)
            if source in HAND_TRASH_EFFECTS:
                _change_hand_count(
                    hand_counts,
                    seat,
                    -_counter_total(cards),
                    raw_path=raw_path,
                    log_index=entry.index,
                    reason=f"{source} trash from hand",
                )
            elif source not in REVEALED_TRASH_EFFECTS:
                raise ConversionError(
                    f"{raw_path}: trash at log {entry.index} has unsupported "
                    f"source effect {source!r}"
                )
            _counter_add(collections[seat], cards, -1, context=f"{raw_path}: trash at log {entry.index}")
            trash.update(cards)
            continue

        if entry.name == LOG_TOPDECK:
            seat = _seat_argument(entry, raw_path)
            _check_seat(seat, len(outcome.seats), raw_path, entry.index)
            cards = _card_argument(entry, mapping, raw_path)
            source = _effect_source(effect_cards, entry.depth)
            if source in HAND_TOPDECK_EFFECTS:
                _change_hand_count(
                    hand_counts,
                    seat,
                    -cards.total,
                    raw_path=raw_path,
                    log_index=entry.index,
                    reason=f"{source} topdeck from hand",
                )
            elif source == "Harbinger":
                # The selected card can be indistinguishable from an older,
                # deliberately untracked discard copy.  Dropping this
                # partial fact is exact epistemic bookkeeping; keeping it
                # would assert a card remains in discard when it may not.
                known_discard[seat].clear()
            elif source != "Sentry":
                raise ConversionError(
                    f"{raw_path}: topdeck at log {entry.index} has unsupported "
                    f"source effect {source!r}"
                )
            continue

        if entry.name in (LOG_GAIN, LOG_BUY_GAIN):
            seat = _seat_argument(entry, raw_path)
            _check_seat(seat, len(outcome.seats), raw_path, entry.index)
            cards = _require_known_cards(entry, mapping, raw_path, operation="gain")
            if entry.name == LOG_BUY_GAIN:
                if current_turn_seat is None:
                    raise ConversionError(f"{raw_path}: buy before first turn at semantic log {entry.index}")
                if seat != current_turn_seat:
                    raise ConversionError(
                        f"{raw_path}: buy log {entry.index} belongs to seat {seat}, active seat is {current_turn_seat}"
                    )
                if len(cards) != 1:
                    raise ConversionError(
                        f"{raw_path}: buy log {entry.index} coalesces different card identities; "
                        "their decision order is not recoverable"
                    )
                observed = capture.observed_buy_resources.get(entry.index)
                if observed is not None:
                    if observed.seat != seat:
                        raise ConversionError(
                            f"{raw_path}: observed resources at buy log {entry.index} belong to seat "
                            f"{observed.seat}, not buyer {seat}"
                        )
                    if observed.coins < 0 or observed.buys <= 0:
                        raise ConversionError(
                            f"{raw_path}: observed resources at buy log {entry.index} are invalid "
                            f"(coins={observed.coins}, buys={observed.buys})"
                        )
                    # ResourceUpdate is the authoritative public counter
                    # stream.  It is captured immediately before this
                    # transient Buy line is replaced by BuyGain, so prefer it
                    # to any semantic arithmetic for the row itself.  The
                    # reconstructed counters resume from this observed state
                    # after the purchase for historical/fallback rows.
                    coins[seat] = observed.coins
                    treasure_coins[seat] = 0
                    coins_spent[seat] = 0
                    buy_counts[seat] = observed.buys
                bought_def = next(iter(cards))
                for purchase_offset in range(cards[bought_def]):
                    if buy_counts[seat] <= 0:
                        raise ConversionError(
                            f"{raw_path}: buy log {entry.index} exceeds the public buy count"
                        )
                    coins_before_buy = available_coins(seat)
                    rows.append(
                        _encode_row(
                            game_id=game_id,
                            source_log_index=entry.index,
                            action=A_BUY_BASE + bought_def,
                            actor=seat,
                            player_turn_number=current_turn_number,
                            global_turn_index=global_turn_index,
                            collections=collections,
                            card_totals=card_totals,
                            trash=trash,
                            in_play=in_play,
                            known_discard=known_discard,
                            hand_counts=hand_counts,
                            coins=coins_before_buy,
                            buys=buy_counts[seat],
                            kingdom=capture.game_start.kingdom,
                            mapping=mapping,
                            ply_index=len(rows),
                        )
                    )
                    if observed is not None and purchase_offset == 0:
                        observed_resource_buy_rows += 1
                    per_seat_decisions[seat] += 1
                    cost = _buy_coin_cost(bought_def, mapping)
                    if cost > coins_before_buy:
                        raise ConversionError(
                            f"{raw_path}: buy log {entry.index} spends {cost} coins with only "
                            f"{coins_before_buy} public coins"
                        )
                    buy_counts[seat] -= 1
                    coins_spent[seat] += cost
                    _counter_add(
                        collections[seat],
                        Counter({bought_def: 1}),
                        1,
                        context=f"{raw_path}: buy gain at log {entry.index}",
                    )
                    known_discard[seat][bought_def] += 1
                    turn_had_purchase = True
                continue
            _counter_add(collections[seat], cards, 1, context=f"{raw_path}: gain at log {entry.index}")
            source = _effect_source(effect_cards, entry.depth)
            if source in DISCARD_GAIN_EFFECTS:
                known_discard[seat].update(cards)
            elif source in HAND_GAIN_EFFECTS:
                _change_hand_count(
                    hand_counts,
                    seat,
                    _counter_total(cards),
                    raw_path=raw_path,
                    log_index=entry.index,
                    reason=f"{source} gain to hand",
                )
            elif source not in TOPDECK_GAIN_EFFECTS:
                raise ConversionError(
                    f"{raw_path}: gain at log {entry.index} has unsupported "
                    f"source effect {source!r}"
                )
            continue

    # The terminal GameResult is an observable boundary too.  This covers a
    # final turn that ended its buy phase without a purchase.
    emit_pass_if_observable(capture.log_entries[-1].index if capture.log_entries else 0, terminal=True)
    if current_turn_seat is None:
        raise ConversionError(f"{raw_path}: no normal turn boundaries in semantic log")
    if any(count < 0 for count in hand_counts):
        # A negative count cannot be supplied to the snapshot; report it
        # plainly rather than clipping it into a fabricated hand.
        raise ConversionError(f"{raw_path}: public hand-count reconstruction became negative: {hand_counts}")

    deck_matches = tuple(
        _normalized_counter(collections[seat]) == _normalized_counter(outcome.seats[seat].final_deck)
        for seat in range(len(outcome.seats))
    )
    if not all(deck_matches):
        details = "; ".join(
            _deck_difference(collections[seat], outcome.seats[seat].final_deck, seat, mapping)
            for seat, matched in enumerate(deck_matches)
            if not matched
        )
        raise FinalDeckMismatch(
            f"{raw_path}: final deck histogram mismatch: {details}",
            deck_matches,
        )

    recovered_live = _recover_live_rows(
        capture,
        game_id=game_id,
        outcome=outcome,
        mapping=mapping,
    )
    # Public buy rows retain their historic semantic ordering.  When a live
    # CardMove proves one of those buys from a fully visible hand, replace
    # that public approximation in situ instead of giving training two copies
    # of the same choice.  The remaining live decisions are appended because
    # the two protocol streams do not expose a reliable global interleaving.
    rows, remaining_live_rows, unmatched_live_buys = _merge_live_rows_with_public_buys(
        rows,
        recovered_live.rows,
    )
    recovered_skips = Counter(recovered_live.skipped_by_type)
    if unmatched_live_buys:
        recovered_skips["live_buy_without_semantic_match"] += unmatched_live_buys
    all_rows = [*rows, *remaining_live_rows]
    all_rows = [replace(row, ply_index=index) for index, row in enumerate(all_rows)]
    emitted_per_seat = [0 for _ in outcome.seats]
    for row in all_rows:
        emitted_per_seat[row.seat_index] += 1

    return ConvertedGame(
        game_id=game_id,
        source_path=raw_path,
        rows=tuple(all_rows),
        outcome=outcome,
        deck_matches=deck_matches,
        per_seat_decisions=tuple(emitted_per_seat),
        observed_resource_buy_rows=observed_resource_buy_rows,
        player_names=tuple(capture.game_start.players),
        kingdom=tuple(capture.game_start.kingdom),
        visibility=recovered_live.visibility,
        recovered_skips=dict(sorted(recovered_skips.items())),
        militia_examples=recovered_live.militia_examples,
    )


def _encode_row(
    *,
    game_id: str,
    source_log_index: int,
    action: int,
    actor: int,
    player_turn_number: int,
    global_turn_index: int,
    collections: Sequence[CounterInt],
    card_totals: CounterInt,
    trash: CounterInt,
    in_play: Sequence[CounterInt],
    known_discard: Sequence[CounterInt],
    hand_counts: Sequence[int],
    coins: int,
    buys: int,
    kingdom: Sequence[str],
    mapping: CardMap,
    ply_index: int,
    decision_type: str | None = None,
    observation_quality: int = OBSERVATION_PARTIALLY_INFERRED,
    source_event_index: int = -1,
) -> PublicBuyRow:
    """Build and encode one synthetic-but-consistent public buy snapshot."""

    if actor < 0 or actor >= len(collections):
        raise ConversionError(f"invalid buy actor {actor}")
    if coins < 0:
        raise ConversionError(f"buy at log {source_log_index} has negative public coins {coins}")
    if buys <= 0:
        raise ConversionError(f"buy at log {source_log_index} has non-positive public buys {buys}")
    supply = _supply_from_public_totals(card_totals, collections, trash)
    players: list[dict[str, object]] = []
    for seat, collection in enumerate(collections):
        unresolved = _counter_subtract(
            collection,
            in_play[seat],
            context=f"buy log {source_log_index}: seat {seat} in-play exceeds collection",
        )
        unresolved = _counter_subtract(
            unresolved,
            known_discard[seat],
            context=f"buy log {source_log_index}: seat {seat} known discard exceeds collection",
        )
        requested_hand = hand_counts[seat]
        if requested_hand < 0:
            raise ConversionError(
                f"buy log {source_log_index}: seat {seat} has negative public hand count {requested_hand}"
            )
        available = _counter_total(unresolved)
        if requested_hand > available:
            raise ConversionError(
                f"buy log {source_log_index}: seat {seat} hand count {requested_hand} exceeds "
                f"{available} unresolved cards"
            )
        hand = _sample_counter(
            unresolved,
            requested_hand,
            _sample_seed(game_id, source_log_index, seat),
        )
        players.append(
            {
                "hand": _snapshot_counter(hand) if seat == actor else {},
                "hand_count": requested_hand,
                "hand_deck": _snapshot_counter(unresolved),
                "deck_count": available - requested_hand,
                "discard": _snapshot_counter(known_discard[seat]),
                "in_play": _snapshot_counter(in_play[seat]),
                "set_aside": {},
                "actions": 0,
                "buys": buys if seat == actor else 0,
                "coins": coins if seat == actor else 0,
            }
        )
    snapshot = {
        "num_players": len(collections),
        "our_player": actor,
        # Zero-count piles must remain present in a snapshot: pile presence
        # and pile count are distinct in the engine ABI.
        "supply": _snapshot_counter(supply, include_zero_defs=card_totals),
        "kingdom_order": [_def_for_name(name, mapping, "kingdom") for name in kingdom],
        "players": players,
        "trash": _snapshot_counter(trash),
        "card_totals": _snapshot_counter(card_totals),
        "turn_number": global_turn_index,
        "phase": "buy",
        "current_player": actor,
    }
    try:
        game = dz.game_from_snapshot(snapshot)
        game.validate()
        obs = np.asarray(game.encode(actor, OBS_VERSION), dtype=np.float32)
        legal = np.asarray(game.legal_mask(), dtype=np.bool_)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ConversionError(f"buy log {source_log_index}: engine snapshot rejected: {error}") from error
    if obs.shape != (OBS_WIDTH,):
        raise ConversionError(f"buy log {source_log_index}: obs shape {obs.shape}, expected ({OBS_WIDTH},)")
    if legal.shape != (ACTION_WIDTH,):
        raise ConversionError(
            f"buy log {source_log_index}: legal mask shape {legal.shape}, expected ({ACTION_WIDTH},)"
        )
    if action < 0 or action >= ACTION_WIDTH or not bool(legal[action]):
        legal_buys = [
            action_id
            for action_id in range(A_BUY_BASE, ACTION_WIDTH)
            if bool(legal[action_id])
        ]
        raise ConversionError(
            f"buy log {source_log_index}: demonstrated action {action} is not legal "
            f"with public coins={coins}; legal buy actions={legal_buys}"
        )
    return PublicBuyRow(
        obs=obs.copy(),
        legal=legal.copy(),
        action=action,
        seat_index=actor,
        ply_index=ply_index,
        turn_number=player_turn_number,
        source_log_index=source_log_index,
        coins=coins,
        buys=buys,
        # The snapshot preserves the actor's full collection, so the engine
        # also handles count-dependent VP cards without inventing a Python
        # scoring rule.
        score=int(game.score(actor)),
        decision_type=("buy_pass" if action == A_PASS else "buy")
        if decision_type is None
        else decision_type,
        observation_quality=observation_quality,
        source_event_index=source_event_index,
    )


def _load_protocol_capture(raw_path: Path) -> ProtocolCapture:
    """Decode recorder JSONL through the existing arena protocol parser.

    ``ArenaParser`` already owns the binary grammar and card decoding.  This
    function only retains the final semantic entry at each global log index,
    because a transient Buy log can later be replaced by its completed Gain
    log at the same index.
    """

    parser = ArenaParser()
    game_start: GameStart | None = None
    full_state: FullState | None = None
    logical: dict[int, SemanticLogEntry | None] = {}
    initial_log_entries: tuple[SemanticLogEntry, ...] = ()
    live_events: list[LiveEvent] = []
    live_segments: list[LiveSegment] = []
    next_live_event_index = 0
    current_resources: dict[tuple[int, str], int] = {}
    observed_buy_resources: dict[int, ObservedBuyResources] = {}
    # A spectator joins with a FullState plus one complete historical
    # gameLogInfo dump.  FullState's counters describe the *current* board,
    # not every earlier row in that dump, so resource observations begin only
    # after it has been loaded.  Subsequent Buy -> BuyGain replacements are
    # live and can be paired with the immediately preceding ResourceUpdates.
    awaiting_history_dump = False
    history_loaded = False
    with gzip.open(raw_path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ConversionError(f"{raw_path}:{line_number}: invalid JSON: {error}") from error
            if record.get("kind") != "binary":
                continue
            frame = decode_record_binary(record, source=f"{raw_path}:{line_number}")
            if frame is None:
                continue
            for event in parser.parse_frame(frame):
                if isinstance(event, GameStart):
                    game_start = event
                elif isinstance(event, FullState):
                    if full_state is not None and history_loaded:
                        live_segments.append(
                            LiveSegment(
                                full_state=full_state,
                                initial_log_entries=initial_log_entries,
                                live_events=tuple(live_events),
                            )
                        )
                    full_state = event
                    # A reconnect can send another complete FullState and a
                    # fresh historical gameLogInfo dump.  Deltas collected
                    # before that snapshot describe the old connection
                    # boundary and must never be folded into this newer
                    # point-in-time state.
                    live_events = []
                    current_resources = {
                        (counter.owner, counter.name): counter.value
                        for counter in event.counters
                        if counter.owner is not None
                    }
                    observed_buy_resources.clear()
                    awaiting_history_dump = True
                    history_loaded = False
                elif isinstance(event, ResourceUpdate) and event.seat is not None:
                    current_resources[(event.seat, event.resource)] = event.value
            if frame.direction is Direction.INBOUND and frame.msg_type == 33:
                start_index, updated_entries = _collect_log_entries(frame, parser, logical, raw_path)
                if awaiting_history_dump:
                    # Do not associate FullState's point-in-time resources
                    # with a joining spectator's backfilled history.
                    awaiting_history_dump = False
                    history_loaded = True
                    initial_log_entries = tuple(
                        value for _index, value in sorted(logical.items()) if value is not None
                    )
                    continue
                if not history_loaded:
                    # A malformed/out-of-order capture cannot safely claim
                    # that a counter observation belongs to old log history.
                    history_loaded = True
                    continue

                replacements = {
                    start_index + offset: entry
                    for offset, entry in enumerate(updated_entries)
                }
                # gameLogInfo replaces its suffix.  Preserve a pre-Buy
                # observation only when that exact index becomes BuyGain;
                # every other replacement (including an undo) invalidates it.
                for index in tuple(observed_buy_resources):
                    if index < start_index:
                        continue
                    replacement = replacements.get(index)
                    if replacement is None or replacement.name != LOG_BUY_GAIN:
                        del observed_buy_resources[index]

                for offset, entry in enumerate(updated_entries):
                    if entry is None or entry.name != LOG_BUY:
                        continue
                    seat = _seat_argument(entry, raw_path)
                    coins = current_resources.get((seat, "coins"))
                    buys = current_resources.get((seat, "buys"))
                    if coins is None or buys is None:
                        # Historical fallback remains available if a malformed
                        # live stream omitted one of the public counters.
                        continue
                    observed_buy_resources[start_index + offset] = ObservedBuyResources(
                        seat=seat,
                        coins=coins,
                        buys=buys,
                    )
                continue

            # CardMove's two identity arrays are more detailed than the
            # normalized arena events: that parser intentionally suppresses
            # draw identities for a non-player spectator.  Decode the live
            # delta directly here, after ArenaParser has refreshed its zone
            # and card-instance tables from FullState.
            if (
                history_loaded
                and frame.direction is Direction.INBOUND
                and frame.msg_type == 32
            ):
                live_event = _decode_live_event(frame, parser, next_live_event_index, raw_path)
                if live_event is not None:
                    live_events.append(live_event)
                    next_live_event_index += 1
    if game_start is None:
        raise ConversionError(f"{raw_path}: missing GameStart")
    if full_state is None:
        raise ConversionError(f"{raw_path}: missing FullState")
    if full_state.game_id != game_start.game_id:
        raise ConversionError(f"{raw_path}: GameStart and FullState game ids disagree")
    if not logical:
        raise ConversionError(f"{raw_path}: no gameLogInfo entries")
    expected = set(range(max(logical) + 1))
    if set(logical) != expected:
        raise ConversionError(f"{raw_path}: semantic log is not a contiguous history from index zero")
    entries = tuple(value for _index, value in sorted(logical.items()) if value is not None)
    if not entries:
        raise ConversionError(f"{raw_path}: semantic log contains no LogEntry records")
    if history_loaded:
        live_segments.append(
            LiveSegment(
                full_state=full_state,
                initial_log_entries=initial_log_entries,
                live_events=tuple(live_events),
            )
        )
    if not live_segments:
        raise ConversionError(f"{raw_path}: FullState was not followed by a gameLogInfo history dump")
    return ProtocolCapture(
        raw_path=raw_path,
        game_start=game_start,
        full_state=full_state,
        log_entries=entries,
        initial_log_entries=initial_log_entries,
        live_events=tuple(live_events),
        live_segments=tuple(live_segments),
        observed_buy_resources=observed_buy_resources,
    )


def _decode_live_event(
    frame: DecodedFrame,
    parser: ArenaParser,
    event_index: int,
    raw_path: Path,
) -> LiveEvent | None:
    """Decode the live subset of ``gameEventInfo`` needed by conversion.

    This intentionally reads CardMove's two endpoint arrays directly instead
    of reusing :meth:`ArenaParser._movement_cards`: the arena operates as a
    player and hides spectator draws by policy, while this converter must
    faithfully retain whether the server exposed the destination identity.
    """

    reader = Reader(frame.payload)
    subtype = reader.u32()
    if subtype == 0:
        from_zone_index = reader.s32()
        to_zone_index = reader.s32()
        source_ids = reader.s32_array()
        destination_ids = reader.s32_array()
        movement_id = reader.u32()
        reader.u32()  # animation class
        reader.finish()
        try:
            from src.v2.arena.protocol.parser import MOVEMENT_TYPES

            movement = MOVEMENT_TYPES[movement_id]
        except (ImportError, IndexError) as error:
            raise ConversionError(
                f"{raw_path}: live event {event_index} has unknown movement type {movement_id}"
            ) from error
        count = max(len(source_ids), len(destination_ids))
        if count == 0:
            return None
        source_cards = _live_card_names(source_ids, parser, raw_path, event_index)
        destination_cards = _live_card_names(destination_ids, parser, raw_path, event_index)
        if len(source_cards) not in {0, count} or len(destination_cards) not in {0, count}:
            raise ConversionError(
                f"{raw_path}: live CardMove {event_index} has incompatible endpoint lengths "
                f"{len(source_cards)} and {len(destination_cards)}"
            )
        return LiveCardMove(
            event_index=event_index,
            movement=movement,
            seat=parser._movement_seat(movement_id, from_zone_index, to_zone_index),
            from_zone=parser._zone_kind(from_zone_index),
            to_zone=parser._zone_kind(to_zone_index),
            from_zone_index=from_zone_index,
            to_zone_index=to_zone_index,
            source_cards=_pad_live_cards(source_cards, count),
            destination_cards=_pad_live_cards(destination_cards, count),
            count=count,
        )
    if subtype == 1:
        counter_index = reader.s32()
        value = reader.s32()
        reader.finish()
        seat, resource = parser.counter_info.get(counter_index, (None, f"counter-{counter_index}"))
        return LiveResourceUpdate(
            event_index=event_index,
            seat=seat,
            resource=resource,
            value=value,
        )
    if subtype == 3:
        seat = reader.s32()
        turn_number = reader.s32()
        turn_type = reader.s32()
        controller_seat = reader.s32()
        reader.finish()
        return LiveTurnDescription(
            event_index=event_index,
            seat=seat,
            turn_number=turn_number,
            turn_type=turn_type,
            controller_seat=controller_seat,
        )
    if subtype == 4:
        seat = reader.s32()
        included_discard = reader.boolean()
        reader.finish()
        return LiveShuffle(
            event_index=event_index,
            seat=seat,
            included_discard=included_discard,
        )
    # The remaining event subtypes either describe visuals/counters already
    # represented above or do not change a base-game decision state.
    return None


def _live_card_names(
    instance_ids: Sequence[int],
    parser: ArenaParser,
    raw_path: Path,
    event_index: int,
) -> tuple[str | None, ...]:
    """Resolve only non-redacted CardMove ids into card names."""

    result: list[str | None] = []
    for instance_id in instance_ids:
        if instance_id < 0:
            result.append(None)
            continue
        name = parser.card_by_instance.get(instance_id)
        if name is None:
            raise ConversionError(
                f"{raw_path}: live CardMove {event_index} references unknown card instance {instance_id}"
            )
        result.append(name)
    return tuple(result)


def _pad_live_cards(cards: tuple[str | None, ...], count: int) -> tuple[str | None, ...]:
    """Treat an omitted endpoint list as fully redacted, never as empty."""

    if len(cards) == count:
        return cards
    if not cards:
        return (None,) * count
    raise AssertionError("live CardMove endpoint length was not prevalidated")


def _collect_log_entries(
    frame: DecodedFrame,
    parser: ArenaParser,
    destination: dict[int, SemanticLogEntry | None],
    raw_path: Path,
) -> tuple[int, tuple[SemanticLogEntry | None, ...]]:
    """Read one gameLogInfo frame using ArenaParser's shared argument reader."""

    reader = Reader(frame.payload)
    start_index = reader.u32()
    entry_count = reader.u32()
    # gameLogInfo is a replace-from-index stream, not an append-only log.  A
    # normal in-flight line (for example Buy -> BuyGain) re-sends its trailing
    # window, while an accepted Undo rewinds to an earlier index and sends a
    # shorter replacement.  Keeping entries beyond that boundary replays the
    # abandoned branch as well as the replacement branch, which is precisely
    # how otherwise impossible hand and in-play counts arise.  This mirrors
    # the web client's truncation behavior documented in RECON.md.
    for previous_index in tuple(destination):
        if previous_index >= start_index:
            del destination[previous_index]
    updated_entries: list[SemanticLogEntry | None] = []
    for offset in range(entry_count):
        entry_type = reader.u32()
        index = start_index + offset
        if entry_type == 0:
            name = reader.u32()
            depth = reader.s32()
            arguments = reader.array(lambda: parser._read_log_argument(reader))
            entry = SemanticLogEntry(
                index=index,
                name=name,
                depth=depth,
                arguments=arguments,
            )
            destination[index] = entry
            updated_entries.append(entry)
        elif entry_type == 1:
            # Deliberately discard opaque raw answers.  In spectator captures
            # the associated offered-element list never arrives.
            reader.s32()
            reader.s32()
            reader.s32_array()
            reader.boolean()
            destination[index] = None
            updated_entries.append(None)
        else:
            raise ConversionError(f"{raw_path}: unknown semantic log entry type {entry_type} at {index}")
    reader.finish()
    return start_index, tuple(updated_entries)


def _parse_outcome(manifest: dict[str, object], game_start: GameStart, mapping: CardMap) -> Outcome:
    """Read the collector's lossless GameResult manifest payload."""

    try:
        outcome_raw = manifest["outcome"]
        result = outcome_raw["game_result"]  # type: ignore[index]
        players = result["players"]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise ConversionError("complete manifest lacks decoded outcome.game_result.players") from error
    if not isinstance(players, list) or len(players) != len(game_start.player_ids):
        raise ConversionError("GameResult players do not match protocol seat count")
    by_player_id: dict[int, dict[str, object]] = {}
    for raw_player in players:
        if not isinstance(raw_player, dict):
            raise ConversionError("GameResult player entry is not an object")
        player_id = _required_int(raw_player, "player_id", "GameResult player")
        if player_id in by_player_id:
            raise ConversionError(f"GameResult repeats player_id {player_id}")
        by_player_id[player_id] = raw_player
    seats: list[OutcomeSeat] = []
    for player_id, protocol_name in zip(game_start.player_ids, game_start.players, strict=True):
        raw_player = by_player_id.get(player_id)
        if raw_player is None:
            raise ConversionError(f"GameResult is missing protocol player {player_id}")
        score_raw = raw_player.get("score")
        if not isinstance(score_raw, dict):
            raise ConversionError(f"GameResult player {player_id} lacks score")
        histogram = raw_player.get("final_deck_histogram")
        if not isinstance(histogram, list):
            raise ConversionError(f"GameResult player {player_id} lacks final_deck_histogram")
        final_deck: CounterInt = Counter()
        for item in histogram:
            if not isinstance(item, dict):
                raise ConversionError(f"GameResult player {player_id} has invalid deck histogram entry")
            name = item.get("card_name")
            if not isinstance(name, str):
                raise ConversionError(f"GameResult player {player_id} has unnamed deck card")
            def_id = _def_for_name(name, mapping, "GameResult final deck")
            frequency = _required_int(item, "frequency", "GameResult final deck")
            if frequency < 0:
                raise ConversionError(f"GameResult player {player_id} has negative {name} frequency")
            final_deck[def_id] += frequency
        resignation_type = raw_player.get("resignation_type")
        resign_index = _required_int(raw_player, "resign_index", "GameResult player")
        seats.append(
            OutcomeSeat(
                player_id=player_id,
                player_name=str(raw_player.get("player_name", protocol_name)),
                rank=_required_int(raw_player, "rank", "GameResult player"),
                score=_required_int(score_raw, "total_points", "GameResult score"),
                turns_used=_required_int(score_raw, "used_turns", "GameResult score"),
                final_deck=final_deck,
                resigned=resignation_type is not None or resign_index >= 0,
            )
        )
    lowest_rank = min(seat.rank for seat in seats)
    winners = [seat_index for seat_index, seat in enumerate(seats) if seat.rank == lowest_rank]
    winner = winners[0] if len(winners) == 1 else None
    return Outcome(seats=tuple(seats), winner=winner, margin_valid=not any(seat.resigned for seat in seats))


def _validate_base_capture(capture: ProtocolCapture, mapping: CardMap) -> None:
    """Reject unknown cards instead of inventing a snapshot outside v2 scope."""

    base_defs = {
        def_id
        for wire_id, def_id in mapping.wire_to_def.items()
        if 1 <= wire_id <= BASE_CARD_LIMIT
    }
    for name in (*capture.game_start.kingdom, *(name for name, _ in capture.full_state.card_counts)):
        def_id = _def_for_name(name, mapping, "capture")
        if def_id not in base_defs:
            raise ConversionError(f"{capture.raw_path}: unsupported non-base card {name!r}")


def _load_card_map(path: Path) -> CardMap:
    """Load the recon card map rather than maintaining a second vocab table."""

    raw = _read_json_object(path)
    entries = raw.get("map")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: card map must contain a list named map")
    wire_to_def: dict[int, int] = {}
    name_to_def: dict[str, int] = {}
    def_to_name: dict[int, str] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: card map entry is not an object")
        wire_id = _required_int(item, "dgames_id", "card map")
        def_id = _required_int(item, "def_id", "card map")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}: card map has invalid name")
        if wire_id in wire_to_def or name in name_to_def or def_id in def_to_name:
            raise ValueError(f"{path}: card map has a duplicate card entry")
        wire_to_def[wire_id] = def_id
        name_to_def[name] = def_id
        def_to_name[def_id] = name
    return CardMap(wire_to_def=wire_to_def, name_to_def=name_to_def, def_to_name=def_to_name)


def _turn_description(entry: SemanticLogEntry, raw_path: Path) -> tuple[int, int, int, int] | None:
    for argument_type, value in entry.arguments:
        if argument_type == 6:
            if not isinstance(value, tuple) or len(value) != 4 or not all(isinstance(item, int) for item in value):
                raise ConversionError(f"{raw_path}: malformed turn description at log {entry.index}")
            return value  # type: ignore[return-value]
    return None


def _seat_argument(entry: SemanticLogEntry, raw_path: Path) -> int:
    for argument_type, value in entry.arguments:
        if argument_type == 1 and isinstance(value, int):
            return value
    raise ConversionError(f"{raw_path}: log {entry.index} has no player argument")


def _int_argument(entry: SemanticLogEntry, raw_path: Path) -> int:
    for argument_type, value in entry.arguments:
        if argument_type == 4 and isinstance(value, int):
            return value
    raise ConversionError(f"{raw_path}: log {entry.index} has no integer resource argument")


def _card_argument(entry: SemanticLogEntry, mapping: CardMap, raw_path: Path) -> CardArgument:
    """Decode known card identities and spectator-redacted card-back slots."""

    for argument_type, value in entry.arguments:
        if argument_type != 0:
            continue
        if not isinstance(value, tuple):
            raise ConversionError(f"{raw_path}: malformed card argument at log {entry.index}")
        result: CounterInt = Counter()
        hidden_count = 0
        for pair in value:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ConversionError(f"{raw_path}: malformed card frequency at log {entry.index}")
            wire_id, frequency = pair
            if not isinstance(wire_id, int) or not isinstance(frequency, int) or frequency < 0:
                raise ConversionError(f"{raw_path}: invalid card frequency at log {entry.index}")
            if wire_id == CARD_BACK_WIRE_ID:
                hidden_count += frequency
                continue
            try:
                def_id = mapping.wire_to_def[wire_id]
            except KeyError as error:
                raise ConversionError(f"{raw_path}: unmapped card wire id {wire_id} at log {entry.index}") from error
            result[def_id] += frequency
        return CardArgument(known=result, hidden_count=hidden_count)
    raise ConversionError(f"{raw_path}: log {entry.index} has no card argument")


def _require_known_cards(
    entry: SemanticLogEntry,
    mapping: CardMap,
    raw_path: Path,
    *,
    operation: str,
) -> CounterInt:
    """Return a card argument only when every identity is public.

    Gains and trashes affect the authoritative ownership conservation check;
    inventing a type for a ``BACK`` slot would make that check meaningless.
    Moves whose identities are irrelevant to ownership (for example Artisan's
    hidden topdeck) use :func:`_card_argument` directly instead.
    """

    cards = _card_argument(entry, mapping, raw_path)
    if cards.hidden_count:
        raise ConversionError(
            f"{raw_path}: {operation} at log {entry.index} has "
            f"{cards.hidden_count} hidden card identity/identities"
        )
    return cards.known


def _single_card_name(cards: CounterInt, mapping: CardMap, raw_path: Path, log_index: int) -> str:
    """Resolve a one-card play into its source-effect name."""

    if _counter_total(cards) != 1:
        raise ConversionError(f"{raw_path}: log {log_index} does not name exactly one card")
    def_id = next(iter(cards))
    try:
        return mapping.def_to_name[def_id]
    except KeyError as error:
        raise ConversionError(f"{raw_path}: log {log_index} has unmapped def {def_id}") from error


def _effect_source(effect_cards: dict[int, str], depth: int) -> str | None:
    """Return the nearest enclosing played card for a nested semantic entry."""

    candidates = [effect_depth for effect_depth in effect_cards if effect_depth < depth]
    return effect_cards[max(candidates)] if candidates else None


def _replace_effect_card(effect_cards: dict[int, str], depth: int, card_name: str) -> None:
    """Begin a new semantic effect scope at ``depth``."""

    for effect_depth in tuple(effect_cards):
        if effect_depth >= depth:
            del effect_cards[effect_depth]
    effect_cards[depth] = card_name


def _change_hand_count(
    hand_counts: list[int],
    seat: int,
    delta: int,
    *,
    raw_path: Path,
    log_index: int,
    reason: str,
) -> None:
    """Apply a public hand-size movement without permitting underflow."""

    next_count = hand_counts[seat] + delta
    if next_count < 0:
        raise ConversionError(
            f"{raw_path}: log {log_index}: {reason} makes seat {seat} public hand count "
            f"negative ({next_count})"
        )
    hand_counts[seat] = next_count


def _buy_coin_cost(def_id: int, mapping: CardMap) -> int:
    """Return the supported base-card coin cost for a recorded purchase."""

    try:
        card_name = mapping.def_to_name[def_id]
        return BASE_BUY_COSTS[card_name]
    except KeyError as error:
        raise ConversionError(f"buy cost is unavailable for def {def_id}") from error


def _hidden_card_count(entry: SemanticLogEntry, raw_path: Path) -> int:
    """Return only a Draw count, intentionally discarding all card identities."""

    for argument_type, value in entry.arguments:
        if argument_type != 0:
            continue
        if not isinstance(value, tuple):
            raise ConversionError(f"{raw_path}: malformed draw count at log {entry.index}")
        total = 0
        for pair in value:
            if not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[1], int):
                raise ConversionError(f"{raw_path}: malformed draw count at log {entry.index}")
            if pair[1] < 0:
                raise ConversionError(f"{raw_path}: negative draw count at log {entry.index}")
            total += pair[1]
        return total
    raise ConversionError(f"{raw_path}: draw log {entry.index} has no count")


def _validate_starting_cards(
    entry: SemanticLogEntry,
    mapping: CardMap,
    seats: int,
    raw_path: Path,
) -> None:
    """Check the claimed standard starting cards without using them as truth."""

    seat = _seat_argument(entry, raw_path)
    _check_seat(seat, seats, raw_path, entry.index)
    cards = _require_known_cards(entry, mapping, raw_path, operation="starting cards")
    # Dominion.games writes the seven Coppers and three Estates as separate
    # semantic entries, not one combined starting-deck record.
    copper = _def_for_name("Copper", mapping, "starting deck")
    estate = _def_for_name("Estate", mapping, "starting deck")
    expected_entries = (Counter({copper: 7}), Counter({estate: 3}))
    if all(_normalized_counter(cards) != expected for expected in expected_entries):
        raise ConversionError(f"{raw_path}: nonstandard starting cards at log {entry.index}")


def _supply_from_public_totals(
    card_totals: CounterInt,
    collections: Sequence[CounterInt],
    trash: CounterInt,
) -> CounterInt:
    """Recover each supply count by conservation of public card ownership."""

    supply = Counter(card_totals)
    for collection in collections:
        _counter_add(supply, collection, -1, context="public supply reconstruction")
    _counter_add(supply, trash, -1, context="public supply reconstruction")
    return _normalized_counter(supply)


def _counter_from_named_pairs(
    pairs: Iterable[tuple[str, int]],
    mapping: CardMap,
    *,
    context: str,
) -> CounterInt:
    result: CounterInt = Counter()
    for name, count in pairs:
        if count < 0:
            raise ConversionError(f"{context}: negative count for {name}")
        result[_def_for_name(name, mapping, context)] += count
    return result


def _counter_from_names(counts: Counter[str], mapping: CardMap) -> CounterInt:
    return Counter({_def_for_name(name, mapping, "starting deck"): count for name, count in counts.items()})


def _counter_add(destination: CounterInt, source: CounterInt, multiplier: int, *, context: str) -> None:
    for def_id, count in source.items():
        destination[def_id] += multiplier * count
        if destination[def_id] < 0:
            raise ConversionError(f"{context}: negative count for def {def_id}")
        if destination[def_id] == 0:
            del destination[def_id]


def _counter_subtract(source: CounterInt, subtrahend: CounterInt, *, context: str) -> CounterInt:
    result = Counter(source)
    _counter_add(result, subtrahend, -1, context=context)
    return result


def _normalized_counter(counter: CounterInt) -> CounterInt:
    return Counter({def_id: count for def_id, count in counter.items() if count > 0})


def _counter_total(counter: CounterInt) -> int:
    return sum(counter.values())


def _sample_seed(game_id: str, log_index: int, seat: int) -> int:
    payload = f"{SOURCE_TAG}|{game_id}|{log_index}|{seat}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _sample_counter(pool: CounterInt, count: int, seed: int) -> CounterInt:
    """Sample a deterministic multiset without replacement from ``pool``."""

    if count < 0 or count > _counter_total(pool):
        raise ConversionError(f"cannot sample {count} cards from pool of {_counter_total(pool)}")
    if count == 0:
        return Counter()
    cards = [def_id for def_id in sorted(pool) for _ in range(pool[def_id])]
    generator = np.random.default_rng(seed)
    selected = generator.choice(len(cards), size=count, replace=False)
    return Counter(cards[int(index)] for index in selected)


def _snapshot_counter(
    counter: CounterInt,
    *,
    include_zero_defs: Iterable[int] = (),
) -> dict[int, int]:
    result = {int(def_id): int(count) for def_id, count in sorted(_normalized_counter(counter).items())}
    for def_id in include_zero_defs:
        result.setdefault(int(def_id), 0)
    return result


def _deck_difference(actual: CounterInt, expected: CounterInt, seat: int, mapping: CardMap) -> str:
    names = sorted(set(actual) | set(expected))
    differences = [
        f"{mapping.def_to_name.get(def_id, f'def-{def_id}')}={actual.get(def_id, 0)} vs {expected.get(def_id, 0)}"
        for def_id in names
        if actual.get(def_id, 0) != expected.get(def_id, 0)
    ]
    return f"seat {seat}: " + ", ".join(differences)


def _seat_margin(seats: Sequence[OutcomeSeat], actor: int) -> int:
    opponents = [seat.score for index, seat in enumerate(seats) if index != actor]
    return seats[actor].score - max(opponents) if opponents else 0


def _outcome_sign(outcome: Outcome, actor: int) -> int:
    if outcome.winner is None:
        return 0
    return 1 if actor == outcome.winner else -1


def _margin_blend_value(margin: int) -> float:
    if margin == 0:
        return 0.0
    sign = 1.0 if margin > 0 else -1.0
    graded = min(abs(float(margin)), DEFAULT_SCALE) / DEFAULT_SCALE
    return sign * (DEFAULT_ALPHA + (1.0 - DEFAULT_ALPHA) * (0.5 + 0.5 * graded))


def _def_for_name(name: str, mapping: CardMap, context: str) -> int:
    try:
        return mapping.name_to_def[name]
    except KeyError as error:
        raise ConversionError(f"{context}: unmapped card name {name!r}") from error


def _check_seat(seat: int, seats: int, raw_path: Path, log_index: int) -> None:
    if seat < 0 or seat >= seats:
        raise ConversionError(f"{raw_path}: log {log_index} has invalid seat {seat}")


def _required_int(raw: dict[str, object], key: str, context: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConversionError(f"{context}: {key} must be an integer")
    return value


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"{path}: cannot read: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return raw


def _manifest_sort_key(path: Path) -> tuple[int, str]:
    stem = path.name.removesuffix(".manifest.json")
    try:
        return int(stem), stem
    except ValueError:
        return sys.maxsize, stem


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _shard_tuple_count(path: Path) -> int:
    with np.load(path, allow_pickle=False) as shard:
        return int(shard["action"].shape[0])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert complete dominion.games spectator captures into visibility-aware decision tuples."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--card-map", type=Path, default=DEFAULT_CARD_MAP)
    parser.add_argument("--shard-size", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the converter CLI and return a conventional process status."""

    args = _parser().parse_args(argv)
    try:
        result = convert_dgames_corpus(
            args.raw_root,
            output_dir=args.out,
            card_map_path=args.card_map,
            shard_size=args.shard_size,
        )
    except (OSError, ValueError, ConversionError) as error:
        print(f"dgames conversion failed: {error}", file=sys.stderr, flush=True)
        return 2
    print(result["manifest_path"])
    print(
        "dgames converter: "
        f"emitted {result['tuples_exported']} tuple(s) from {result['games_emitted']} game(s); "
        f"quarantined {result['quarantined']}; "
        f"final-deck match rate {float(result['final_deck_match_rate']):.1%}; "
        f"divergence {float(result['final_deck_divergence_rate']):.1%}",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
