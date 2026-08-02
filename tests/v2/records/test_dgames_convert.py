"""Regression gate for every locally captured Dominion.games spectator game."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.v2.records.dgames_convert import (
    LOG_CARD_COIN_BONUS,
    LOG_CARD_COIN_BONUS_ONE,
    LOG_DRAW,
    LOG_TREASURES,
    LOG_TURN_DESCRIPTION,
    _hidden_card_count,
    _load_protocol_capture,
    _seat_argument,
    _turn_description,
    convert_dgames_corpus,
)


ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data/dominion_games/raw"
CARD_MAP = ROOT / "data/dominion_games/recon/card_id_map.json"


def test_all_discovered_spectator_captures_have_the_expected_outcome(tmp_path: Path) -> None:
    """Regression gate for resource, hidden-hand, and capacity failure families.

    ``raw/`` is collector-owned and gitignored, so discovery rather than a
    fixture id list is intentional.  This catches a regression that turns any
    completed local capture back into a silent conversion failure while still
    allowing a collector to be writing an explicitly incomplete manifest.
    """

    if not tuple(RAW_ROOT.glob("*.manifest.json")):
        pytest.skip("no dominion.games raw captures are available")

    result = convert_dgames_corpus(
        RAW_ROOT,
        output_dir=tmp_path / "tuples_dgames",
        card_map_path=CARD_MAP,
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    totals = manifest["totals"]

    complete_quarantine = [
        entry for entry in manifest["quarantine"] if entry["capture_status"] == "complete"
    ]

    # Every emitted game crosses the strict terminal-deck gate.  A complete
    # game may remain quarantined only when the current engine ABI cannot
    # represent its public snapshot (more than 24 cards in play), or when it
    # is outside the converter's intentionally two-player scope.
    assert totals["complete_games_seen"] == totals["games_emitted"] + len(complete_quarantine)
    assert totals["complete_games_without_final_deck_check"] == len(complete_quarantine)
    assert totals["final_deck_checked_games"] == totals["games_emitted"]
    assert totals["final_deck_matched_games"] == totals["games_emitted"]
    assert totals["final_deck_match_rate"] == 1.0
    assert totals["observed_resource_buy_rows"] > 0
    assert totals["observed_resource_buy_rows"] == sum(
        game["observed_resource_buy_rows"] for game in manifest["games"]
    )
    assert manifest["source_tag"].endswith("visibility_aware_full_decisions_v2")
    assert totals["tuples_per_emitted_game"] > 53.0

    # A live full-decision row has an explicit provenance bit and may only
    # belong to a seat whose hand stayed fully visible.  The existing public
    # buy rows remain deliberately marked partially inferred.
    assert set(totals["visibility_grades"]) <= {"full", "partial", "buy_only"}
    assert sum(totals["visibility_grades"].values()) == totals["games_emitted"]
    game_visibility = {
        game["index"]: game["visibility"]["hand_visible_throughout"]
        for game in manifest["games"]
    }
    assert any(game["visibility"]["full_state_segments"] > 1 for game in manifest["games"])
    for game in manifest["games"]:
        visibility = game["visibility"]["hand_visible_throughout"]
        grade = game["visibility"]["grade"]
        assert game["visibility"]["full_state_segments"] >= 1
        assert grade == (
            "full" if all(visibility) else "partial" if any(visibility) else "buy_only"
        )

    id_to_decision = {
        value: name
        for name, value in manifest["row_provenance"]["decision_type_ids"].items()
    }
    decision_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    rows_seen = 0
    for shard in manifest["shards"]:
        with np.load(tmp_path / "tuples_dgames" / shard["path"], allow_pickle=False) as data:
            assert {
                "action",
                "legal",
                "game_index",
                "seat_index",
                "source_event_index",
                "decision_type",
                "observation_quality",
            } <= set(data.files)
            actions = data["action"]
            legal = data["legal"]
            decision_type = data["decision_type"]
            quality = data["observation_quality"]
            game_index = data["game_index"]
            seat_index = data["seat_index"]
            source_event_index = data["source_event_index"]
            assert np.all(legal[np.arange(actions.size), actions])
            assert set(np.unique(quality)) <= {0, 1}
            assert np.all(source_event_index[quality == 1] >= 0)
            for row_index, type_id in enumerate(decision_type):
                name = id_to_decision[int(type_id)]
                decision_counts[name] += 1
                quality_counts[
                    "fully_observed" if int(quality[row_index]) == 1 else "partially_inferred"
                ] += 1
                if int(quality[row_index]) == 1:
                    assert game_visibility[int(game_index[row_index])][int(seat_index[row_index])]
            rows_seen += int(actions.size)

    assert rows_seen == totals["tuples_exported"]
    assert dict(sorted(decision_counts.items())) == totals["tuples_by_decision_type"]
    assert dict(sorted(quality_counts.items())) == totals["rows_by_observation_quality"]
    assert decision_counts["action_play"] > 0
    assert decision_counts["treasure_play"] > 0
    assert decision_counts["militia_keep"] > 0
    assert decision_counts["sentry_order"] > 0

    # Audit one of the recovered attack decisions from the dynamically
    # discovered corpus: exactly the observed discards separate the before
    # and after hand, and Militia leaves three cards.
    assert manifest["militia_sanity_examples"]
    for example in manifest["militia_sanity_examples"]:
        before = Counter(example["hand_before"])
        discarded = Counter(example["discarded"])
        after = Counter(example["hand_after"])
        assert before - discarded == after
        assert not discarded - before
        assert sum(after.values()) == 3

    for game in manifest["games"]:
        final_deck_match = game["final_deck_match"]
        assert final_deck_match["all_seats"] is True
        assert all(final_deck_match["per_seat"])

    assert {
        entry["category"] for entry in complete_quarantine
    } <= {"engine_in_play_capacity", "unsupported_player_count"}
    assert all(
        "MAX_IN_PLAY" in entry["reason"]
        if entry["category"] == "engine_in_play_capacity"
        else "only two-player captures are supported" in entry["reason"]
        for entry in complete_quarantine
    )

    # These were the former high-volume corruption families.  Any recurrence
    # is an accounting regression, not an acceptable quarantine reason.
    forbidden_reasons = ("public coins", "hand count", "public hand-count", "makes seat")
    assert not any(
        any(fragment in entry["reason"] for fragment in forbidden_reasons)
        for entry in complete_quarantine
    )
    assert all(
        entry["category"] in {"incomplete_capture", "engine_in_play_capacity", "unsupported_player_count"}
        for entry in manifest["quarantine"]
    )

    # Exercise the exact structures behind the former high-volume failures,
    # without pinning the live collector corpus to individual game ids.
    emitted_ids = {game["id"] for game in manifest["games"]}
    merchant_bonus_ids: set[str] = set()
    multi_treasure_ids: set[str] = set()
    short_cleanup_ids: set[str] = set()
    opponent_draw_ids: set[str] = set()
    for source_manifest in RAW_ROOT.glob("*.manifest.json"):
        source = json.loads(source_manifest.read_text(encoding="utf-8"))
        if source.get("capture_status") != "complete":
            continue
        game_id = source_manifest.name.removesuffix(".manifest.json")
        raw_path = RAW_ROOT / f"{game_id}.jsonl.gz"
        if not raw_path.exists():
            continue
        capture = _load_protocol_capture(raw_path)
        active_seat: int | None = None
        turn_index = 0
        treasure_groups: dict[tuple[int, int], int] = {}
        for entry in capture.log_entries:
            if entry.name == LOG_TURN_DESCRIPTION:
                turn = _turn_description(entry, capture.raw_path)
                if turn is not None and turn[2] == 0:
                    active_seat = turn[0]
                    turn_index += 1
                continue
            if entry.name in (LOG_CARD_COIN_BONUS_ONE, LOG_CARD_COIN_BONUS):
                merchant_bonus_ids.add(game_id)
            if entry.name == LOG_TREASURES:
                seat = _seat_argument(entry, capture.raw_path)
                key = (turn_index, seat)
                treasure_groups[key] = treasure_groups.get(key, 0) + 1
                if treasure_groups[key] >= 2:
                    multi_treasure_ids.add(game_id)
            if entry.name == LOG_DRAW:
                seat = _seat_argument(entry, capture.raw_path)
                draw_count = _hidden_card_count(entry, capture.raw_path)
                if entry.depth == 0 and draw_count < 5:
                    short_cleanup_ids.add(game_id)
                if entry.depth > 0 and active_seat is not None and seat != active_seat:
                    opponent_draw_ids.add(game_id)

    assert merchant_bonus_ids & emitted_ids
    assert multi_treasure_ids & emitted_ids
    assert short_cleanup_ids & emitted_ids
    assert opponent_draw_ids & emitted_ids
