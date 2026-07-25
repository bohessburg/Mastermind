"""Real-source coverage for the unified game record converters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dominion_v2_py as dz
import pytest

from src.v2.arena.archive import GameArchive, ResultSummary
from src.v2.records.arena import convert_arena_archive
from src.v2.records.convert import convert_paths, discover_sources
from src.v2.records.local import PHASE_NAMES, convert_local_export
from src.v2.records.model import GameRecord, validate_record
from src.v2.web.server.defs import def_id, def_name


ROOT = Path(__file__).resolve().parents[3]
LOCAL_EXPORTS = (
    ROOT / "exports/1jMv_xAgobIUlMpC.json",
    ROOT / "exports/mJMhsSGYQV-TUZG4.json",
    ROOT / "exports/eoURgfuadtw6DlsR.json",
)
ARENA_ARCHIVES = (
    ROOT
    / "exports/arena/20260725T060501.751790Z"
    / "20260725T062230.198982Z-game-181376119",
    ROOT
    / "exports/arena/20260725T053730.304380Z"
    / "20260725T053737.746324Z-game-181375294",
    ROOT
    / "exports/arena/20260725T060501.751790Z"
    / "20260725T063728.308223Z-game-181376377",
)
ARENA_BODY_EVENTS = {
    "FullState",
    "TurnStart",
    "PendingDecision",
    "ReactionWindow",
    "Play",
    "Buy",
    "Attack",
    "Gain",
    "Trash",
    "Discard",
    "Draw",
    "Reveal",
    "Topdeck",
    "ZoneTransfer",
    "Shuffle",
    "ResourceUpdate",
    "PileReorder",
    "GameEnd",
    "GameResult",
}


def _require_real(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"real export is not present: {path}")
    return path


@pytest.mark.parametrize("path", LOCAL_EXPORTS)
def test_real_local_exports_replay_to_valid_records(path: Path) -> None:
    path = _require_real(path)
    source = json.loads(path.read_text(encoding="utf-8"))
    record = convert_local_export(path)

    validate_record(record)
    assert record.source == "local"
    assert [card.def_id for card in record.kingdom] == source["kingdom"]
    assert [card.name for card in record.kingdom] == [
        def_name(card) for card in source["kingdom"]
    ]
    assert [seat.kind for seat in record.seats] == source["seats"]
    assert [seat.index for seat in record.seats if seat.bot] == [
        index
        for index, kind in enumerate(source["seats"])
        if kind.startswith("bot")
    ]
    assert len(record.records) == len(source["actions"])
    assert all(item.engine_action_ids_visibility == "known" for item in record.records)
    assert [item.engine_action_ids[0] for item in record.records] == source["actions"]

    game = _independent_replay(source, record)
    assert [result.vp for result in record.results] == [
        game.score(seat) for seat in range(game.num_players())
    ]
    assert all(result.visibility == "known" for result in record.results)


def test_local_detail_matches_independent_engine_replay() -> None:
    path = _require_real(LOCAL_EXPORTS[0])
    source = json.loads(path.read_text(encoding="utf-8"))
    record = convert_local_export(path)
    game = dz.new_game(
        dz.Setup(players=len(source["seats"]), kingdom=source["kingdom"]),
        source["seed"],
    )
    checkpoints = {0, len(source["actions"]) // 2, len(source["actions"]) - 1}

    for index, action in enumerate(source["actions"]):
        item = record.records[index]
        turn_counter = int(game.turn())
        if index in checkpoints:
            assert item.turn_number == turn_counter // game.num_players() + 1
            assert item.active_seat == turn_counter % game.num_players()
            assert item.actor_seat == int(game.current_decision()["player"])
            assert item.phase == PHASE_NAMES[int(game.phase())]
        game.step(int(action))
        if index not in checkpoints:
            continue
        resources = game.resources()
        assert (
            item.resources_after.actions,
            item.resources_after.buys,
            item.resources_after.coins,
        ) == (
            resources["actions"],
            resources["buys"],
            resources["coins"],
        )
        counts = {
            (zone.seat, zone.zone): zone.count
            for zone in item.zone_counts_after
        }
        for seat in range(game.num_players()):
            assert counts[(seat, "hand")] == game.hand_count(seat)
            assert counts[(seat, "deck")] == game.deck_count(seat)
            assert counts[(seat, "discard")] == game.discard_count(seat)
            assert counts[(seat, "in-play")] == len(game.in_play(seat))
            assert counts[(seat, "set-aside")] == len(game.set_aside(seat))


@pytest.mark.parametrize("path", ARENA_ARCHIVES)
def test_real_arena_archives_fold_only_observed_information(path: Path) -> None:
    path = _require_real(path)
    event_rows = _jsonl(path / "events.jsonl")
    start = next(row["event"] for row in event_rows if row["event_type"] == "GameStart")
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    record = convert_arena_archive(path)

    validate_record(record)
    assert record.source == "arena"
    assert record.game_id == str(result["game_id"])
    assert [card.name for card in record.kingdom] == start["kingdom"]
    assert [card.def_id for card in record.kingdom] == [
        def_id(name) for name in start["kingdom"]
    ]
    assert [seat.display_name for seat in record.seats] == start["players"]
    assert record.controlled_seat == result["our_seat"] == start["our_seat"]
    assert [seat.controlled for seat in record.seats].count(True) == 1
    assert [seat.bot for seat in record.seats] == [
        index == result["our_seat"] for index in range(len(start["players"]))
    ]
    assert [standing.vp for standing in record.results] == result["scores"]
    assert [standing.placing for standing in record.results] == result["placings"]

    expected_body = [
        row for row in event_rows if row["event_type"] in ARENA_BODY_EVENTS
    ]
    assert len(record.records) == len(expected_body)
    assert sum(item.event == "TurnStart" for item in record.records) == sum(
        row["event_type"] == "TurnStart" for row in event_rows
    )
    assert sum(item.record_type == "action" for item in record.records) == sum(
        row["event_type"] in {"Play", "Buy", "Attack"} for row in event_rows
    )

    opponent = 1 - int(record.controlled_seat)
    hidden_zones = [
        zone
        for item in record.records
        for zone in item.zone_counts_after
        if zone.seat == opponent
        and zone.zone in {"hand", "deck"}
        and zone.count is not None
    ]
    assert hidden_zones
    assert all(zone.visibility == "counts_only" for zone in hidden_zones)
    assert not any(zone.visibility == "known" for zone in hidden_zones)
    gains = [item for item in record.records if item.event == "Gain"]
    assert gains
    assert all(item.engine_action_ids_visibility == "unknown" for item in gains)
    assert all(not item.engine_action_ids for item in gains)
    treasure_plays = [
        item
        for item in record.records
        if item.event == "Play"
        and item.played.cards
        and any(
            card.name in {"Copper", "Silver", "Gold"}
            for card in item.played.cards
        )
    ]
    assert treasure_plays
    assert all(item.phase == "buy" for item in treasure_plays)


def test_archive_session_and_local_directory_are_backfill_friendly(
    tmp_path: Path,
) -> None:
    local = _require_real(LOCAL_EXPORTS[0])
    arena = _require_real(ARENA_ARCHIVES[0])
    assert local in discover_sources(local.parent)
    assert arena in discover_sources(arena.parent)

    written = convert_paths((local, arena), output_dir=tmp_path)
    assert {path.name for path in written} == {
        f"local-{local.stem}.game-record.json",
        "arena-181376119.game-record.json",
    }
    for path in written:
        validate_record(json.loads(path.read_text(encoding="utf-8")))


def test_game_archive_finish_invokes_unified_emission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[Path] = []

    def capture(path: Path) -> Path:
        emitted.append(Path(path))
        return Path(path) / "game-record.json"

    monkeypatch.setattr("src.v2.records.convert.emit_arena_record", capture)
    archive = GameArchive(tmp_path, game_id=123)
    archive.finish(
        ResultSummary(
            game_id=123,
            completed=True,
            divergence_aborted=False,
            decisions=0,
            reason="test",
        )
    )
    assert emitted == [archive.path]
    assert (archive.path / "result.json").is_file()


def _independent_replay(
    source: dict[str, Any],
    record: GameRecord,
) -> Any:
    game = dz.new_game(
        dz.Setup(players=len(source["seats"]), kingdom=source["kingdom"]),
        source["seed"],
    )
    max_turn = 0
    for index, action in enumerate(source["actions"]):
        mask = game.legal_mask()
        assert bool(mask[int(action)])
        max_turn = max(
            max_turn,
            int(game.turn()) // game.num_players() + 1,
        )
        game.step(int(action))
    assert max(item.turn_number or 0 for item in record.records) == max_turn
    assert f"0x{game.state_hash():016x}" == source["final_state_hash"].lower()
    return game


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
