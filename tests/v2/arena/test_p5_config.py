from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.v2.arena.config import ArenaConfig


def test_default_arena_config_loads_credentials_only_from_environment() -> None:
    config = ArenaConfig.load(
        "configs/arena.json",
        environ={"ARENA_USER": "operator", "ARENA_PASS": "secret"},
    )

    assert config.sims == 400
    assert config.checkpoint_path == Path(
        "checkpoints/remote/campaign19/gen_0015.pt"
    )
    assert config.arena_user == "operator"
    assert config.arena_pass == "secret"
    assert config.lobby.max_games_per_session == 0
    assert config.stall_watchdog_seconds == 120.0
    assert config.idle_watchdog_seconds == 600.0
    assert config.lobby.homepage_timeout_seconds == 90.0
    assert config.lobby.reconnect_limit_policy == "return_to_lobby"
    assert config.lobby.resume_full_state_timeout_seconds == 30.0
    assert config.lobby.searching_timeout_seconds == 180.0
    assert config.lobby.search_engagement_timeout_seconds == 5.0
    assert config.lobby.max_fruitless_search_attempts == 3
    assert config.lobby.search_without_match_budget_seconds == 600.0
    assert config.undo.auto_deny is True
    assert config.timeout.claim_grace_seconds == 30.0
    assert config.actuation_mode == "protocol"


def test_arena_config_rejects_invalid_pacing(tmp_path: Path) -> None:
    path = tmp_path / "arena.json"
    path.write_text(
        json.dumps(
            {
                "think_time": {"min_seconds": 3, "max_seconds": 2},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="maximum think time"):
        ArenaConfig.load(path, environ={})


def test_arena_config_rejects_unknown_actuation_mode(tmp_path: Path) -> None:
    path = tmp_path / "arena.json"
    path.write_text(
        json.dumps({"actuation_mode": "canvas"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="actuation_mode"):
        ArenaConfig.load(path, environ={})

    clicks = tmp_path / "clicks.json"
    clicks.write_text(
        json.dumps({"actuation_mode": "clicks"}),
        encoding="utf-8",
    )
    assert ArenaConfig.load(clicks, environ={}).actuation_mode == "clicks"


def test_arena_config_accepts_unlimited_lobby_sessions_and_rejects_timeouts(
    tmp_path: Path,
) -> None:
    unlimited = tmp_path / "unlimited.json"
    unlimited.write_text(
        json.dumps({"lobby": {"max_games_per_session": 0}}),
        encoding="utf-8",
    )
    assert ArenaConfig.load(unlimited, environ={}).lobby.max_games_per_session == 0

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps({"lobby": {"leaving_timeout_seconds": 0}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leaving timeout"):
        ArenaConfig.load(invalid, environ={})


def test_arena_config_allows_only_boolean_true_undo_auto_deny(
    tmp_path: Path,
) -> None:
    disabled = tmp_path / "disabled.json"
    disabled.write_text(
        json.dumps({"undo": {"auto_deny": False}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="undo.auto_deny must be true"):
        ArenaConfig.load(disabled, environ={})

    not_boolean = tmp_path / "not-boolean.json"
    not_boolean.write_text(
        json.dumps({"undo": {"auto_deny": "true"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="'auto_deny' must be a boolean"):
        ArenaConfig.load(not_boolean, environ={})


def test_arena_config_rejects_negative_timeout_claim_grace(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid-timeout.json"
    invalid.write_text(
        json.dumps({"timeout": {"claim_grace_seconds": -1}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="timeout claim grace"):
        ArenaConfig.load(invalid, environ={})


def test_arena_config_rejects_zero_resume_full_state_timeout(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid-resume.json"
    invalid.write_text(
        json.dumps({"lobby": {"resume_full_state_timeout_seconds": 0}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="resume_full_state timeout"):
        ArenaConfig.load(invalid, environ={})


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"idle_watchdog_seconds": 0}, "idle watchdog"),
        (
            {"lobby": {"max_fruitless_search_attempts": 0}},
            "max_fruitless_search_attempts",
        ),
        (
            {"lobby": {"search_engagement_timeout_seconds": 0}},
            "search_engagement timeout",
        ),
    ],
)
def test_arena_config_rejects_invalid_idle_and_search_bounds(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    invalid = tmp_path / "invalid-watchdog.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        ArenaConfig.load(invalid, environ={})


def test_arena_config_rejects_unknown_reconnect_limit_policy(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid-reconnect-limit-policy.json"
    invalid.write_text(
        json.dumps({"lobby": {"reconnect_limit_policy": "wait"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reconnect_limit_policy"):
        ArenaConfig.load(invalid, environ={})
