"""Configuration loading for supervised arena sessions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_CHECKPOINT = Path("checkpoints/remote/campaign15/gen_0045.pt")


@dataclass(frozen=True, kw_only=True)
class LobbyConfig:
    """Automatch and timeout settings for the lobby state machine."""

    card_pool: str = "base"
    rated: bool = False
    max_games_per_session: int = 0
    homepage_timeout_seconds: float = 90.0
    resume_full_state_timeout_seconds: float = 30.0
    searching_timeout_seconds: float = 180.0
    table_waiting_timeout_seconds: float = 30.0
    in_game_timeout_seconds: float = 3600.0
    game_over_timeout_seconds: float = 30.0
    game_ended_dialog_timeout_seconds: float = 5.0
    leaving_timeout_seconds: float = 30.0


@dataclass(frozen=True, kw_only=True)
class UndoConfig:
    """Safety policy for metagame undo requests."""

    auto_deny: bool = True


@dataclass(frozen=True, kw_only=True)
class TimeoutConfig:
    """Safety policy for an opponent's metagame timeout offer."""

    claim_grace_seconds: float = 30.0


@dataclass(frozen=True, kw_only=True)
class ArenaConfig:
    """All operator-controlled settings for one arena process."""

    checkpoint_path: Path = DEFAULT_CHECKPOINT
    obs_version: int | None = 2
    sims: int = 400
    determinizations: int = 2
    wall_clock_cap_seconds: float | None = 30.0
    stall_watchdog_seconds: float = 120.0
    think_time_min_seconds: float = 1.25
    think_time_max_seconds: float = 2.75
    actuation_mode: str = "protocol"
    lobby: LobbyConfig = LobbyConfig()
    undo: UndoConfig = UndoConfig()
    timeout: TimeoutConfig = TimeoutConfig()
    arena_user: str | None = None
    arena_pass: str | None = None

    @classmethod
    def load(
        cls,
        path: Path | str = "configs/arena.json",
        *,
        environ: Mapping[str, str] | None = None,
    ) -> ArenaConfig:
        """Load JSON settings and credentials from the process environment."""
        config_path = Path(path)
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"cannot read arena config: {config_path}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid arena config JSON: {config_path}") from error
        if not isinstance(raw, dict):
            raise ValueError("arena config root must be an object")

        search = _object(raw, "search")
        pacing = _object(raw, "think_time")
        lobby_raw = _object(raw, "lobby")
        undo_raw = _object(raw, "undo")
        timeout_raw = _object(raw, "timeout")
        env = os.environ if environ is None else environ

        cap_value = search.get("wall_clock_cap_seconds", 30.0)
        cap = None if cap_value is None else float(cap_value)
        obs_value = search.get("obs_version", 2)
        obs_version = None if obs_value is None else int(obs_value)
        config = cls(
            checkpoint_path=Path(
                raw.get("checkpoint_path", str(DEFAULT_CHECKPOINT))
            ),
            obs_version=obs_version,
            sims=int(search.get("sims", 400)),
            determinizations=int(search.get("determinizations", 2)),
            wall_clock_cap_seconds=cap,
            stall_watchdog_seconds=float(
                raw.get("stall_watchdog_seconds", 120.0)
            ),
            think_time_min_seconds=float(pacing.get("min_seconds", 1.25)),
            think_time_max_seconds=float(pacing.get("max_seconds", 2.75)),
            actuation_mode=str(raw.get("actuation_mode", "protocol")),
            lobby=LobbyConfig(
                card_pool=str(lobby_raw.get("card_pool", "base")),
                rated=bool(lobby_raw.get("rated", False)),
                max_games_per_session=int(
                    lobby_raw.get("max_games_per_session", 0)
                ),
                homepage_timeout_seconds=float(
                    lobby_raw.get("homepage_timeout_seconds", 90.0)
                ),
                resume_full_state_timeout_seconds=float(
                    lobby_raw.get("resume_full_state_timeout_seconds", 30.0)
                ),
                searching_timeout_seconds=float(
                    lobby_raw.get("searching_timeout_seconds", 180.0)
                ),
                table_waiting_timeout_seconds=float(
                    lobby_raw.get("table_waiting_timeout_seconds", 30.0)
                ),
                in_game_timeout_seconds=float(
                    lobby_raw.get("in_game_timeout_seconds", 3600.0)
                ),
                game_over_timeout_seconds=float(
                    lobby_raw.get("game_over_timeout_seconds", 30.0)
                ),
                game_ended_dialog_timeout_seconds=float(
                    lobby_raw.get("game_ended_dialog_timeout_seconds", 5.0)
                ),
                leaving_timeout_seconds=float(
                    lobby_raw.get("leaving_timeout_seconds", 30.0)
                ),
            ),
            undo=UndoConfig(
                auto_deny=_boolean(undo_raw, "auto_deny", default=True),
            ),
            timeout=TimeoutConfig(
                claim_grace_seconds=float(
                    timeout_raw.get("claim_grace_seconds", 30.0)
                ),
            ),
            arena_user=env.get("ARENA_USER"),
            arena_pass=env.get("ARENA_PASS"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Reject unsafe or internally inconsistent runtime settings."""
        if self.obs_version not in (None, 1, 2):
            raise ValueError("obs_version must be 1, 2, or null")
        if self.sims <= 0:
            raise ValueError("sims must be greater than zero")
        if self.determinizations <= 0:
            raise ValueError("determinizations must be greater than zero")
        if (
            self.wall_clock_cap_seconds is not None
            and self.wall_clock_cap_seconds <= 0
        ):
            raise ValueError("wall-clock cap must be positive or null")
        if self.stall_watchdog_seconds <= 0:
            raise ValueError("stall watchdog must be positive")
        if self.think_time_min_seconds < 0:
            raise ValueError("minimum think time cannot be negative")
        if self.think_time_max_seconds < self.think_time_min_seconds:
            raise ValueError("maximum think time must be at least the minimum")
        if self.actuation_mode not in {"protocol", "clicks"}:
            raise ValueError("actuation_mode must be 'protocol' or 'clicks'")
        if self.lobby.max_games_per_session < 0:
            raise ValueError("lobby max_games_per_session cannot be negative")
        if not self.undo.auto_deny:
            raise ValueError(
                "undo.auto_deny must be true; granting undo is unsupported"
            )
        if self.timeout.claim_grace_seconds < 0:
            raise ValueError("timeout claim grace must be non-negative")
        for name, value in (
            ("homepage", self.lobby.homepage_timeout_seconds),
            ("resume_full_state", self.lobby.resume_full_state_timeout_seconds),
            ("searching", self.lobby.searching_timeout_seconds),
            ("table_waiting", self.lobby.table_waiting_timeout_seconds),
            ("in_game", self.lobby.in_game_timeout_seconds),
            ("game_over", self.lobby.game_over_timeout_seconds),
            ("game_ended_dialog", self.lobby.game_ended_dialog_timeout_seconds),
            ("leaving", self.lobby.leaving_timeout_seconds),
        ):
            if value <= 0:
                raise ValueError(f"lobby {name} timeout must be positive")


def _object(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"arena config {key!r} must be an object")
    return value


def _boolean(
    raw: Mapping[str, object],
    key: str,
    *,
    default: bool,
) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"arena config {key!r} must be a boolean")
    return value
