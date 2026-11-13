"""Offline protocol and time-join tests for dominion.games ratings."""

from __future__ import annotations

import pytest

from scripts.dgames_ratings import (
    LEADERBOARD,
    LeaderboardEntry,
    decode_leaderboard_payload,
    encode_leaderboard_request,
)
from scripts.dgames_ratings_join import (
    RatingObservation,
    join_manifests,
    nearest_observation,
    parse_utc_timestamp,
)
from src.v2.arena.protocol.frames import Direction, ProtocolError, Writer, decode_frame


def _write_entry(
    writer: Writer,
    *,
    player_id: int,
    name: str,
    rank: int,
    level: float,
    level_change: float,
    skill: float = 1700.0,
    deviation: float = 50.0,
    volatility: float = 0.06,
    converted_skill: float = 6.2,
    converted_deviation: float = 0.4,
    game_count: int = 42,
) -> None:
    writer.s32(player_id).string(name).s32(rank)
    writer.f64(level).f64(level_change)
    writer.f64(skill).f64(deviation).f64(volatility)
    writer.f64(converted_skill).f64(converted_deviation).s32(game_count)


def _observation(
    player_id: int,
    observed_at_utc: str,
    *,
    rating: float = 6.0,
    rating_type: str = "RATINGS_2P",
) -> RatingObservation:
    return RatingObservation(
        player_id=player_id,
        observed_at_utc=observed_at_utc,
        rank=10,
        rating=rating,
        trend=0.1,
        rating_type=rating_type,
    )


def _manifest(
    captured_at: str,
    player_ids: list[int] | None,
    *,
    rating_type_ordinal: int = 0,
) -> dict[str, object]:
    players = [] if player_ids is None else [{"player_id": player_id} for player_id in player_ids]
    return {
        "capture_time_utc": captured_at,
        "outcome": {
            "game_result": {
                "rating_type_ordinal": rating_type_ordinal,
                "players": players,
            }
        },
    }


def test_current_leaderboard_payload_decoder_handles_typed_map_and_full_rows() -> None:
    """Message 25 is a map of full LeaderboardEntry rows, not RankedPlayer[]."""

    payload_writer = Writer().s32(2)
    payload_writer.s32(0).s32(2)
    _write_entry(
        payload_writer,
        player_id=101,
        name="Alice",
        rank=1,
        level=6.4,
        level_change=0.2,
        skill=1820.0,
        deviation=42.0,
        volatility=0.055,
        converted_skill=6.6,
        converted_deviation=0.1,
        game_count=99,
    )
    _write_entry(
        payload_writer,
        player_id=102,
        name="Bob",
        rank=2,
        level=6.1,
        level_change=-0.1,
    )
    payload_writer.s32(2).s32(1)
    _write_entry(
        payload_writer,
        player_id=201,
        name="Carol",
        rank=3,
        level=5.9,
        level_change=0.0,
    )
    payload = payload_writer.build()

    inbound = decode_frame(
        Writer().u32(17).u32(LEADERBOARD).bytes(payload).build(),
        Direction.INBOUND,
    )
    assert inbound is not None
    assert inbound.sequence == 17
    assert inbound.msg_type == LEADERBOARD

    decoded = decode_leaderboard_payload(inbound.payload)
    assert [entry.rating_type for entry in decoded] == [
        "RATINGS_2P",
        "RATINGS_2P",
        "RATINGS_2P_BLITZ",
    ]
    assert decoded[0] == LeaderboardEntry(
        player_id=101,
        player_name="Alice",
        rank=1,
        rating=6.4,
        trend=0.2,
        rating_type="RATINGS_2P",
        skill=1820.0,
        deviation=42.0,
        volatility=0.055,
        converted_skill=6.6,
        converted_deviation=0.1,
        game_count=99,
    )
    assert decoded[1].player_name == "Bob"
    assert decoded[2].rating == 5.9


def test_leaderboard_request_has_current_type_count_flag_order() -> None:
    payload = encode_leaderboard_request(count=123_456, flag=True)
    assert payload == b"\x00\x01\xe2@\x01"


def test_leaderboard_decoder_rejects_unknown_rating_type_and_trailing_bytes() -> None:
    with pytest.raises(ProtocolError, match="unknown rating type ordinal"):
        decode_leaderboard_payload(Writer().s32(1).s32(99).s32(0).build())
    with pytest.raises(ProtocolError, match="trailing bytes"):
        decode_leaderboard_payload(Writer().s32(0).u8(0).build())


def test_nearest_observation_prefers_closest_then_earlier_on_a_tie() -> None:
    capture_time = parse_utc_timestamp("2026-08-02T12:00:00Z", field="capture")
    before = _observation(7, "2026-08-02T11:59:00Z", rating=5.8)
    after = _observation(7, "2026-08-02T12:01:00Z", rating=6.2)
    farther = _observation(7, "2026-08-02T12:05:00Z", rating=7.0)

    assert nearest_observation(capture_time, (farther, after, before)) is before


def test_join_uses_nearest_matching_type_and_reports_missing_players_honestly() -> None:
    capture_time = "2026-08-02T12:00:00Z"
    observations = (
        _observation(11, "2026-08-02T11:58:00Z", rating=5.5),
        _observation(11, "2026-08-02T12:00:30Z", rating=6.5),
        _observation(22, "2026-08-02T12:03:00Z", rating=7.5),
        # A different rating type must not be silently substituted for player 23.
        _observation(
            23,
            "2026-08-02T12:00:00Z",
            rating=9.0,
            rating_type="RATINGS_3P",
        ),
    )
    manifests = (
        ("both", _manifest(capture_time, [11, 22])),
        ("one", _manifest(capture_time, [11, 23])),
        ("neither", _manifest(capture_time, [24, 25])),
        ("no-player-ids", _manifest(capture_time, None)),
    )

    sidecar, coverage = join_manifests(manifests, observations)

    assert sidecar["both"]["11"] is not None
    assert sidecar["both"]["11"]["rating"] == 6.5
    assert sidecar["both"]["11"]["staleness_seconds"] == 30.0
    assert sidecar["both"]["22"] is not None
    assert sidecar["both"]["22"]["staleness_seconds"] == 180.0
    assert sidecar["one"]["11"] is not None
    assert sidecar["one"]["23"] is None
    assert sidecar["neither"] == {"24": None, "25": None}
    assert sidecar["no-player-ids"] == {}

    report = coverage.as_json()
    assert report["games_total"] == 4
    assert report["both_players_rated"] == {"games": 1, "fraction": 0.25}
    assert report["one_player_rated"] == {"games": 1, "fraction": 0.25}
    assert report["neither_player_rated"] == {"games": 2, "fraction": 0.5}
    assert report["manifests_without_player_ids"] == 1
