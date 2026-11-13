"""End-to-end checks for replay-verified imitation tuple output."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import dominion_v2_py as dz

from src.v2.records.corpus import build_manifest, load_or_create_manifest
from src.v2.records.tuples import export_real_corpus, replay_export


ROOT = Path(__file__).resolve().parents[3]


def _big_money_action(game) -> int:
    mask = game.legal_mask()
    for def_id in (dz.DEF_PLATINUM, dz.DEF_GOLD, dz.DEF_SILVER, dz.DEF_COPPER, dz.DEF_POTION):
        action = dz.A_PLAY_BASE + def_id
        if action < dz.ACTION_SPACE_SIZE and mask[action]:
            return int(action)
    for def_id in (dz.DEF_PROVINCE, dz.DEF_GOLD, dz.DEF_SILVER):
        action = dz.A_BUY_BASE + def_id
        if action < dz.ACTION_SPACE_SIZE and mask[action]:
            return int(action)
    if mask[dz.A_PASS]:
        return int(dz.A_PASS)
    return int(next(iter(mask.nonzero()[0])))


@pytest.fixture
def tiny_complete_export(tmp_path: Path) -> tuple[Path, int]:
    """A deterministic, complete two-player export used as a tiny corpus fixture."""
    game = dz.new_game(dz.Setup(players=2, kingdom=[]), 0x5151)
    actions: list[int] = []
    human_decisions = 0
    for _ in range(10_000):
        if game.game_over():
            break
        if int(game.current_decision()["player"]) == 0:
            human_decisions += 1
        action = _big_money_action(game)
        actions.append(action)
        game.step(action)
    else:
        raise AssertionError("tiny fixture game did not finish")
    assert len(actions) == 163
    assert human_decisions == 86
    root = tmp_path / "exports"
    root.mkdir()
    source = root / "tiny.json"
    source.write_text(
        json.dumps(
            {
                "seed": 0x5151,
                "kingdom": [],
                "seats": ["human", "bot:bigmoney"],
                "actions": actions,
                "obs_version": int(dz.OBS_VERSION),
                "final_state_hash": f"0x{int(game.state_hash()):016x}",
            }
        ),
        encoding="utf-8",
    )
    return root, human_decisions


def _load_rows(result) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {}
    for shard_path in result.shard_paths:
        with np.load(shard_path) as shard:
            for name in shard.files:
                arrays.setdefault(name, []).append(shard[name])
    return {name: np.concatenate(parts) for name, parts in arrays.items()}


def test_tiny_fixture_exports_exact_human_tuple_count_and_obs_width(
    tiny_complete_export: tuple[Path, int],
) -> None:
    root, expected_human_decisions = tiny_complete_export
    build_manifest(root)
    result = export_real_corpus(root, output_dir=root / "tuples", shard_size=30)
    rows = _load_rows(result)

    assert result.games_processed == 1
    assert result.tuples_exported == expected_human_decisions == 86
    assert rows["obs"].shape == (86, 1788)
    assert rows["action"].dtype == np.int32
    assert rows["legal"].shape == (86, 357)
    assert rows["legal"].dtype == np.bool_
    assert np.all(rows["seat_index"] == 0)
    assert np.all(rows["legal"][np.arange(86), rows["action"]])


def test_tuple_value_sign_matches_the_winner(
    tiny_complete_export: tuple[Path, int],
) -> None:
    root, _ = tiny_complete_export
    build_manifest(root)
    result = export_real_corpus(
        root,
        output_dir=root / "tuples",
        include_bot_seats=True,
        shard_size=500,
    )
    rows = _load_rows(result)
    winner = int(rows["winner"][0])

    assert winner == 0
    assert np.all(rows["winner"] == winner)
    assert np.all(rows["value"][rows["seat_index"] == winner] > 0)
    assert np.all(rows["value"][rows["seat_index"] != winner] < 0)
    assert np.all(rows["margin"][rows["seat_index"] == winner] > 0)
    assert np.all(rows["margin"][rows["seat_index"] != winner] < 0)


def test_real_export_decision_counts_match_its_unified_record() -> None:
    exports = ROOT / "exports"
    records = exports / "records/local"
    if not exports.is_dir() or not records.is_dir():
        pytest.skip("real local corpus or unified records are not present")
    manifest = load_or_create_manifest(exports)
    real_paths = {
        str(entry["path"])
        for entry in manifest["files"]
        if entry["classification"] == "real"
    }
    pair: tuple[Path, dict] | None = None
    for record_path in sorted(records.glob("local-*.game-record.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        provenance = str(record.get("provenance", ""))
        try:
            relative = Path(provenance).relative_to("exports").as_posix()
        except ValueError:
            continue
        if relative in real_paths and (exports / relative).is_file():
            pair = (exports / relative, record)
            break
    if pair is None:
        pytest.skip("no unified local record still has a real source export")

    source, record = pair
    replayed = replay_export(
        json.loads(source.read_text(encoding="utf-8")),
        source_path=source,
        game_id=source.stem,
        include_bot_seats=True,
    )
    record_counts = Counter(int(item["actor_seat"]) for item in record["records"])
    assert tuple(record_counts[seat] for seat in range(len(replayed.seat_kinds))) == replayed.per_seat_decisions
