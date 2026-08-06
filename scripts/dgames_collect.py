"""Collect complete base-only dominion.games spectator games, raw frame first.

This collector intentionally uses the conservative single-table architecture:
one long-lived authenticated browser websocket spectates one game at a time.
It is the safe mode to use until ``scripts/dgames_concurrency_probe.py`` has
produced live evidence for a higher single-account attachment limit.  A running
game is randomly sampled from a configurable moderate age band, then its
kingdom is read before the one session is committed to recording it.

The collector normally reuses the already logged-in ``dgames-profile/``
Chromium profile and the existing WebSocket hook.  If ``DGAMES_USER`` and
``DGAMES_PASS`` are available from the environment or the repository-root
``.env``, it can self-heal a wiped profile by filling the page's own login
form.  Credential-bearing reauthentication frames are redacted from its
session diagnostic archive and are never included in per-game raw files.

Usage::

    PYTHONUNBUFFERED=1 ./.venv/bin/python scripts/dgames_collect.py
    PYTHONUNBUFFERED=1 ./.venv/bin/python scripts/dgames_collect.py \\
        --run-seconds 900
    PYTHONUNBUFFERED=1 ./.venv/bin/python scripts/dgames_collect.py --headful

Fleet coordination is opt-in. A direct coordinated account uses a unique
account/profile pair, for example::

    PYTHONUNBUFFERED=1 ./.venv/bin/python scripts/dgames_collect.py \\
        --account-id 1 --coordination-dir data/dominion_games/coord

That account reads ``DGAMES_USER_1`` / ``DGAMES_PASS_1`` from the process
environment or repository ``.env``. ``scripts/dgames_fleet.py`` launches many
such isolated collectors from a non-secret accounts JSON file.

For a standalone VPS deployment, ship this script plus ``src/v2/arena``'s
``browser/session.py``, ``browser/ws_hook.js``, ``archive.py``, and the full
pure-stdlib ``protocol/`` directory.  The only non-stdlib runtime dependency
is Playwright Chromium; no C++ binding, shadow, bot, or actuate module is
imported.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import gzip
import json
import os
from pathlib import Path
import random
import secrets
import signal
import sys
import time
from typing import Any, TypeVar


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v2.arena.archive import serialize_frame_record  # noqa: E402
from src.v2.arena.browser.session import ArenaSession  # noqa: E402
from src.v2.arena.protocol.cards import card_name  # noqa: E402
from src.v2.arena.protocol.events import FullState, GameStart  # noqa: E402
from src.v2.arena.protocol.frames import DecodedFrame, Direction, ProtocolError, Reader, Writer  # noqa: E402
from src.v2.arena.protocol.parser import ArenaParser  # noqa: E402
from src.v2.arena.protocol.recording import decode_record_binary  # noqa: E402
from scripts.dgames_coordination import (  # noqa: E402
    ClaimLease,
    CoordinationBackend,
    CoordinationUnavailable,
    FilesystemCoordinationBackend,
)


DEFAULT_PROFILE = Path("dgames-profile")
DEFAULT_FLEET_PROFILE_ROOT = Path("dgames-profiles")
DEFAULT_RAW_ROOT = Path("data/dominion_games/raw")
DEFAULT_COORDINATION_DIR = Path("data/dominion_games/coord")
DEFAULT_CARD_MAP = Path("data/dominion_games/recon/card_id_map.json")
DEFAULT_URL = "https://dominion.games"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
FAILED_HANDSHAKE_FIXTURE = Path(
    "data/dominion_games/raw/runs/20260801T020913.387278Z/frames.jsonl"
)
# This successful browser session contains completed capture 181648216.
HEALTHY_HANDSHAKE_FIXTURE = Path(
    "data/dominion_games/raw/runs/20260731T235136.707435Z/frames.jsonl"
)

LOGIN = 1
REQUEST_UPDATE = 11
UPDATE_TABLES = 3
JOIN_TABLE = 2
LEAVE_TABLE = 6
TABLES_OVERVIEW = 3
GAME_FINISHED = 14
RUNNING = 2
LOGIN_SUCCESS = 2
LOGIN_WITH_SESSION = 24
REMOVE_ACTIVE_SESSIONS = 35
CLIENT_HEARTBEAT = 44
REQUEST_SERVER_STATE = 45

TABLES_RETRY_SECONDS = 6.0
MIN_JOIN_DELAY_SECONDS = 0.5
DEFAULT_TARGET_MIN_AGE_SECONDS = 4.0 * 60.0
DEFAULT_TARGET_MAX_AGE_SECONDS = 12.0 * 60.0
DEFAULT_SESSION_LOGIN_GRACE_SECONDS = 5.0
DEFAULT_CREDENTIALED_LOGIN_TIMEOUT_SECONDS = 15.0
DEFAULT_CREDENTIALED_RELOGIN_BACKOFF_SECONDS = 10.0
DEFAULT_MAX_CREDENTIALED_RELOGINS_PER_HOUR = 5
CREDENTIALED_RELOGIN_WINDOW_SECONDS = 60.0 * 60.0
DEFAULT_CLAIM_TTL_SECONDS = 90.0
DEFAULT_CLAIM_REFRESH_INTERVAL_SECONDS = 20.0
DEFAULT_SNAPSHOT_CACHE_TTL_SECONDS = 20.0
DEFAULT_SNAPSHOT_REFRESH_TTL_SECONDS = 30.0
# Deliberately no-burst: the shared bucket has capacity one, so even a fleet
# startup cannot emit a 32-browser JOIN_TABLE burst.  Thirty joins/minute is
# one aggregate join every two seconds, a deliberately polite ceiling.
DEFAULT_FLEET_JOIN_RATE_PER_MINUTE = 30.0
SESSION_EXPIRED_EXIT_CODE = 2
SESSION_EXPIRED_MESSAGE = (
    "SESSION EXPIRED: the saved profile session is no longer valid. "
    "Run: ./.venv/bin/python scripts/dgames_login.py"
)
MAX_TRACKED_GAME_STARTS = 64
MAX_REMEMBERED_TABLES = 4_096
MAX_REJECTED_GAME_IDS = 4_096
MAX_PENDING_JOIN_BUFFER_BYTES = 32 * 1024 * 1024
# Client->server messages that can carry a password, session id, or login code,
# plus the recovery-only active-session cleanup request.
# NOTE: 44 is the client HEARTBEAT and carries no payload -- it must NOT be in
# this set.  Treating it as sensitive made every heartbeat during a capture look
# like a re-login, which aborted the game and restarted the browser.
SENSITIVE_OUTBOUND_TYPES = frozenset(
    {
        LOGIN,
        13,
        14,
        15,
        16,
        19,
        LOGIN_WITH_SESSION,
        REMOVE_ACTIVE_SESSIONS,
        REQUEST_SERVER_STATE,
        46,
    }
)
RESIGNATION_TYPES = {
    0: "MANUAL",
    1: "MANUAL_FROM_RECONNECT",
    2: "FORCE_RESIGNED",
    3: "TIMED_OUT",
}


@dataclass(frozen=True)
class TableSummary:
    """One tablesOverview row, including RUNNING-only startTime."""

    table_id: int
    host_id: int
    host_name: str
    players: int
    bots: int
    spectators: int
    min_players: int
    max_players: int
    is_observable: bool
    is_joinable: bool
    status: int
    start_time: int | None

    @property
    def is_two_player(self) -> bool:
        return self.min_players < 3 and self.max_players > 1


@dataclass(frozen=True)
class DgamesCredentials:
    """Credential pair kept in memory only for credentialed self-heal."""

    username: str = field(repr=False)
    password: str = field(repr=False)


class ParkedLoginAction(Enum):
    """The safe action to take when startup handshake evidence is examined."""

    WAIT = "wait"
    CREDENTIALED_RELOGIN = "credentialed_relogin"
    SESSION_EXPIRED = "session_expired"


class CredentialedLoginOutcome(Enum):
    """Only the outcomes needed to supervise a page-driven login attempt."""

    SUCCEEDED = "succeeded"
    ALREADY_LOGGED_IN = "already_logged_in"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class CredentialedReloginSlot:
    """Availability of one credentialed re-login attempt at a given instant."""

    permitted: bool
    backoff_seconds: float
    attempts_in_window: int
    reset_after_seconds: float | None


class CredentialedReloginRateLimiter:
    """Bound credentialed form submissions across page/browser recoveries."""

    def __init__(
        self,
        *,
        max_attempts_per_hour: int,
        minimum_backoff_seconds: float,
    ) -> None:
        self.max_attempts_per_hour = max_attempts_per_hour
        self.minimum_backoff_seconds = minimum_backoff_seconds
        self._attempted_at: deque[float] = deque()
        self._last_attempt_at: float | None = None

    def _discard_expired(self, now: float) -> None:
        cutoff = now - CREDENTIALED_RELOGIN_WINDOW_SECONDS
        while self._attempted_at and self._attempted_at[0] <= cutoff:
            self._attempted_at.popleft()

    def check(self, *, now: float) -> CredentialedReloginSlot:
        """Return a non-mutating decision, pruning expired attempts first."""
        self._discard_expired(now)
        attempts_in_window = len(self._attempted_at)
        if attempts_in_window >= self.max_attempts_per_hour:
            reset_after = max(
                0.0,
                self._attempted_at[0] + CREDENTIALED_RELOGIN_WINDOW_SECONDS - now,
            )
            return CredentialedReloginSlot(
                permitted=False,
                backoff_seconds=0.0,
                attempts_in_window=attempts_in_window,
                reset_after_seconds=reset_after,
            )
        backoff = 0.0
        if self._last_attempt_at is not None:
            backoff = max(0.0, self._last_attempt_at + self.minimum_backoff_seconds - now)
        return CredentialedReloginSlot(
            permitted=True,
            backoff_seconds=backoff,
            attempts_in_window=attempts_in_window,
            reset_after_seconds=None,
        )

    def record(self, *, now: float) -> None:
        """Record an attempt after the caller has consumed an available slot."""
        self._discard_expired(now)
        self._attempted_at.append(now)
        self._last_attempt_at = now


@dataclass(frozen=True)
class FrameNotice:
    """One matching hook record with its decoded frame and normalized events."""

    record: dict[str, Any]
    frame: DecodedFrame | None
    events: tuple[object, ...]


@dataclass
class PendingJoin:
    """A bounded join attempt buffered until kingdom eligibility is known."""

    table: TableSummary
    started_monotonic: float
    deadline_monotonic: float
    raw_records: list[dict[str, Any]] = field(default_factory=list)
    raw_bytes: int = 0
    game_start: GameStart | None = None


@dataclass
class ActiveCapture:
    """A durable base-only capture awaiting its authoritative gameFinished."""

    table: TableSummary
    game_id: int
    capture_time_utc: str
    kingdom_card_names: tuple[str, ...]
    card_type_names: tuple[str, ...]
    player_names: tuple[str, ...]
    player_names_by_id: dict[int, str]
    raw_path: Path
    raw_staging_path: Path
    manifest_path: Path
    manifest: dict[str, object]


@dataclass
class CollectorStats:
    started_at: datetime
    snapshots_received: int = 0
    joins_attempted: int = 0
    joins_timed_out_as_over: int = 0
    tables_probed: int = 0
    base_only_tables_probed: int = 0
    skipped_non_base: int = 0
    skipped_duplicates: int = 0
    captured_started: int = 0
    captured_completed: int = 0
    captures_incomplete: int = 0
    failed: int = 0
    in_page_recoveries: int = 0
    hard_browser_relaunches: int = 0
    browser_restarts: int = 0
    credentialed_relogins: int = 0
    credentialed_self_heal_available: bool = False
    last_snapshot_tables: int = 0
    last_snapshot_candidates: int = 0
    coordination_enabled: bool = False
    account_id: str | None = None
    table_claim_attempts: int = 0
    table_claim_contentions: int = 0
    game_claim_attempts: int = 0
    game_claim_contentions: int = 0
    stale_claims_reclaimed: int = 0
    claim_refreshes: int = 0
    claim_refresh_failures: int = 0
    shared_snapshot_hits: int = 0
    shared_snapshot_refreshes: int = 0
    snapshot_cache_degradations: int = 0
    rate_limit_waits: int = 0

    def claim_attempts(self) -> int:
        return self.table_claim_attempts + self.game_claim_attempts

    def claim_contentions(self) -> int:
        return self.table_claim_contentions + self.game_claim_contentions

    def claim_contention_rate(self) -> float | None:
        attempts = self.claim_attempts()
        return self.claim_contentions() / attempts if attempts else None

    def games_per_hour(self) -> float:
        elapsed_hours = (datetime.now(timezone.utc) - self.started_at).total_seconds() / 3600
        return self.captured_completed / elapsed_hours if elapsed_hours > 0 else 0.0

    def base_only_probe_rate(self) -> float | None:
        """Return the observed base-only share of joins that reached FullState."""
        if self.tables_probed == 0:
            return None
        return self.base_only_tables_probed / self.tables_probed


class StopRequested(Exception):
    """The operator requested a clean stop at the next safe state boundary."""


class RestartBrowser(Exception):
    """The page/browser is dead, so only a fresh Playwright context can recover."""


class RecoverPage(Exception):
    """Recover a live browser context by reloading its existing page in place."""


class SessionExpired(Exception):
    """The persistent profile no longer has a usable dominion.games session."""


class CredentialedReloginRateLimitExceeded(Exception):
    """Credentialed self-heal reached its operator-configured safety cap."""


class SustainedFailure(Exception):
    """Bounded in-page recovery failed repeatedly without returning to health."""


T = TypeVar("T")


def decode_tables_overview(payload: bytes) -> tuple[TableSummary, ...]:
    """Decode one tablesOverview message exactly in its documented wire order."""
    reader = Reader(payload)
    count = reader.s32()
    if count < 0:
        raise ProtocolError(f"negative tablesOverview count {count}")
    tables: list[TableSummary] = []
    for _ in range(count):
        table_id = reader.u64()
        host_id = reader.s32()
        host_name = reader.string()
        players = reader.s32()
        bots = reader.s32()
        spectators = reader.s32()
        min_players = reader.s32()
        max_players = reader.s32()
        is_observable = reader.boolean()
        is_joinable = reader.boolean()
        status = reader.s32()
        start_time = reader.u64() if status == RUNNING else None
        tables.append(
            TableSummary(
                table_id=table_id,
                host_id=host_id,
                host_name=host_name,
                players=players,
                bots=bots,
                spectators=spectators,
                min_players=min_players,
                max_players=max_players,
                is_observable=is_observable,
                is_joinable=is_joinable,
                status=status,
                start_time=start_time,
            )
        )
    reader.finish()
    return tuple(tables)


def _table_to_cache_document(table: TableSummary) -> dict[str, object]:
    """Serialize a lobby row for the backend-neutral shared cache."""
    return {
        "table_id": table.table_id,
        "host_id": table.host_id,
        "host_name": table.host_name,
        "players": table.players,
        "bots": table.bots,
        "spectators": table.spectators,
        "min_players": table.min_players,
        "max_players": table.max_players,
        "is_observable": table.is_observable,
        "is_joinable": table.is_joinable,
        "status": table.status,
        "start_time": table.start_time,
    }


def _table_from_cache_document(document: Mapping[str, object]) -> TableSummary:
    """Validate a backend cache row before it can influence a JOIN_TABLE."""
    integer_fields = (
        "table_id",
        "host_id",
        "players",
        "bots",
        "spectators",
        "min_players",
        "max_players",
        "status",
    )
    values: dict[str, int] = {}
    for name in integer_fields:
        value = document.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"invalid cached lobby field {name}")
        values[name] = value
    host_name = document.get("host_name")
    if not isinstance(host_name, str):
        raise ValueError("invalid cached lobby host_name")
    booleans: dict[str, bool] = {}
    for name in ("is_observable", "is_joinable"):
        value = document.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"invalid cached lobby field {name}")
        booleans[name] = value
    start_time = document.get("start_time")
    if start_time is not None and (not isinstance(start_time, int) or isinstance(start_time, bool)):
        raise ValueError("invalid cached lobby start_time")
    return TableSummary(
        table_id=values["table_id"],
        host_id=values["host_id"],
        host_name=host_name,
        players=values["players"],
        bots=values["bots"],
        spectators=values["spectators"],
        min_players=values["min_players"],
        max_players=values["max_players"],
        is_observable=booleans["is_observable"],
        is_joinable=booleans["is_joinable"],
        status=values["status"],
        start_time=start_time,
    )


def _record_socket_id(record: dict[str, Any]) -> int:
    """Normalize one hook socket id without trusting its original JSON type."""
    try:
        return int(record.get("sock", -1))
    except (TypeError, ValueError):
        return -1


def _record_message_type(record: dict[str, Any]) -> int | None:
    """Read a safe frame type from raw bytes or an already-redacted archive row."""
    if record.get("kind") != "binary":
        return None
    if record.get("redacted") is True:
        archived_type = record.get("msg_type")
        if isinstance(archived_type, int) and not isinstance(archived_type, bool):
            return archived_type
    if not record.get("b64"):
        return None
    try:
        raw = base64.b64decode(str(record.get("data", "")), validate=True)
    except (TypeError, ValueError):
        return None
    offset = 4 if record.get("dir") == "in" else 0
    if len(raw) < offset + 4:
        return None
    return int.from_bytes(raw[offset : offset + 4], "big")


def _is_sensitive_record(record: dict[str, Any], *, game_socket: int = 3) -> bool:
    """Whether this record can carry authentication material.

    Probe replies use the same inbound framing as normal game traffic, so a
    server-state ordinal of 2 is *not* loginSuccess unless it arrived on the
    selected game socket.
    """
    msg_type = _record_message_type(record)
    return (
        record.get("dir") == "in"
        and _record_socket_id(record) == game_socket
        and msg_type == LOGIN_SUCCESS
    ) or (
        record.get("dir") == "out" and msg_type in SENSITIVE_OUTBOUND_TYPES
    )


def _redacted_archive_record(
    record: dict[str, Any],
    *,
    game_socket: int = 3,
) -> dict[str, Any]:
    """Keep diagnostics useful without ever persisting auth/session payloads."""
    archived = dict(record)
    msg_type = _record_message_type(archived)
    if _is_sensitive_record(archived, game_socket=game_socket):
        archived["data"] = ""
        archived["b64"] = False
        archived["redacted"] = True
        archived["msg_type"] = msg_type
    return archived


@dataclass
class SessionHandshakeTracker:
    """Track only safe login signal metadata while a page is starting.

    A fresh dominion.games page probes alpha/beta with REQUEST_SERVER_STATE,
    opens its selected game socket, then sends LOGIN_WITH_SESSION when the
    persistent profile is valid.  A dead stored session leaves the rendered
    page at the login form instead: the socket is open and probes happened, but
    neither the login request nor loginSuccess ever appears.

    Probe replies are intentionally ignored as login evidence: their payload
    begins with a server-state ordinal that can equal 2, the same numeric value
    as game-socket ``loginSuccess``.
    """

    game_socket: int
    game_socket_opened_at: float | None = None
    request_server_state_at: float | None = None
    login_with_session_at: float | None = None
    credential_login_at: float | None = None
    credential_login_requests: int = 0
    login_success_at: float | None = None
    parked_since: float | None = None
    game_socket_generations: int = 0

    def _current_candidate_started_at(self) -> float | None:
        if self.game_socket_opened_at is None or self.request_server_state_at is None:
            return None
        return max(self.game_socket_opened_at, self.request_server_state_at)

    def _clear_parked_evidence(self) -> None:
        self.parked_since = None

    def _note_parked_candidate(self) -> None:
        if (
            self.login_with_session_at is not None
            or self.credential_login_at is not None
            or self.login_success_at is not None
        ):
            return
        candidate_started_at = self._current_candidate_started_at()
        if candidate_started_at is not None and self.parked_since is None:
            self.parked_since = candidate_started_at

    def _begin_game_socket_generation(self, *, now: float) -> None:
        """Start a fresh auth generation while retaining prior no-login evidence."""
        self.game_socket_generations += 1
        self.game_socket_opened_at = now
        # A prior LOGIN_WITH_SESSION can fail and make the client clear the
        # profile before this reload.  Its existence must not mask the next
        # generation's no-login signature.
        self.login_with_session_at = None
        self.credential_login_at = None
        self.login_success_at = None
        self._note_parked_candidate()

    def observe(self, record: dict[str, Any], *, now: float) -> None:
        """Record type-level handshake evidence without retaining frame payloads."""
        socket_id = _record_socket_id(record)
        if record.get("kind") == "open" and socket_id == self.game_socket:
            self._begin_game_socket_generation(now=now)

        msg_type = _record_message_type(record)
        if record.get("dir") == "out" and msg_type == REQUEST_SERVER_STATE:
            self.request_server_state_at = now
            self._note_parked_candidate()
        elif (
            record.get("dir") == "out"
            and socket_id == self.game_socket
            and msg_type == LOGIN_WITH_SESSION
        ):
            self.login_with_session_at = now
            self._clear_parked_evidence()
        elif record.get("dir") == "out" and socket_id == self.game_socket and msg_type == LOGIN:
            self.credential_login_at = now
            self.credential_login_requests += 1
            self._clear_parked_evidence()
        elif (
            record.get("dir") == "in"
            and socket_id == self.game_socket
            and msg_type == LOGIN_SUCCESS
        ):
            self.login_success_at = now
            self._clear_parked_evidence()

    def is_parked_at_login_form(self, *, now: float, grace_seconds: float) -> bool:
        """Whether the wire signature proves the saved session is unavailable."""
        if (
            self.login_with_session_at is not None
            or self.credential_login_at is not None
            or self.login_success_at is not None
        ):
            return False
        candidate_started_at = self._current_candidate_started_at()
        if candidate_started_at is None:
            return False
        self._note_parked_candidate()
        # The probes normally precede the game socket.  Evidence survives an
        # in-page reload with no login traffic, so rapid consecutive reloads
        # cannot keep resetting the grace period forever.
        started_at = self.parked_since if self.parked_since is not None else candidate_started_at
        return now >= started_at + grace_seconds


def _parked_login_action(
    tracker: SessionHandshakeTracker,
    *,
    credentials: DgamesCredentials | None,
    now: float,
    grace_seconds: float,
) -> ParkedLoginAction:
    """Choose fast-fail versus self-heal from the proved parked-login signature."""
    if not tracker.is_parked_at_login_form(now=now, grace_seconds=grace_seconds):
        return ParkedLoginAction.WAIT
    return (
        ParkedLoginAction.CREDENTIALED_RELOGIN
        if credentials is not None
        else ParkedLoginAction.SESSION_EXPIRED
    )


class CollectorSession(ArenaSession):
    """Browser transport with a redacted, per-run diagnostic frame archive."""

    def _receive_frame(self, record: dict[str, Any]) -> None:
        if self._stopped:
            return
        copied = dict(record)
        if self._frames_file is not None:
            self._frames_file.write(
                serialize_frame_record(
                    _redacted_archive_record(record, game_socket=self.game_socket)
                )
            )
            self._frames_file.flush()
        self.frame_queue.put_nowait(copied)


class LivePump:
    """Consume a single known game socket, parse it, and preserve raw notices."""

    def __init__(
        self,
        session: ArenaSession,
        *,
        ignore_records_before_ms: int | None = None,
        login_handshake: SessionHandshakeTracker | None = None,
    ) -> None:
        self.session = session
        self.parser = ArenaParser()
        if login_handshake is not None and login_handshake.game_socket != session.game_socket:
            raise ValueError("login handshake tracker has a different game socket")
        self.login_handshake = login_handshake or SessionHandshakeTracker(session.game_socket)
        self.ignore_records_before_ms = ignore_records_before_ms
        self.awaiting_generation_open = ignore_records_before_ms is not None
        self.socket_open = False
        self.socket_closed = False
        # Joining a table can emit a GameStart for every replayed event.  Keep
        # only a small recent lookup window; the pending join has its own
        # bounded lifetime and is the only consumer of this mapping.
        self.game_starts: OrderedDict[int, GameStart] = OrderedDict()
        self.inbound_type_counts: Counter[int] = Counter()

    def _process(self, record: dict[str, Any] | None) -> FrameNotice | None:
        if record is None:
            raise RestartBrowser("browser transport stopped")
        timestamp = record.get("ts")
        if (
            self.ignore_records_before_ms is not None
            and isinstance(timestamp, int)
            and timestamp < self.ignore_records_before_ms
        ):
            # A reload creates a fresh in-page socket generation.  Ignore late
            # lifecycle records from the one just replaced rather than letting
            # an old close event restart the new page cycle.
            return None
        socket_id = _record_socket_id(record)
        # Reloading deliberately closes the old window's sock-3.  Its close
        # callback can race into the binding after the reload timestamp, so do
        # not mistake it for a failure of the replacement page.  The new page
        # emits its own sock-3 open before any relevant game traffic.
        if (
            socket_id == self.session.game_socket
            and self.awaiting_generation_open
            and record.get("kind") != "open"
        ):
            return None
        if socket_id == self.session.game_socket and record.get("kind") == "open":
            self.awaiting_generation_open = False
        self.login_handshake.observe(record, now=time.monotonic())
        if socket_id != self.session.game_socket:
            return None
        kind = record.get("kind")
        if kind == "open":
            self.socket_open = True
            self.socket_closed = False
            return FrameNotice(record=dict(record), frame=None, events=())
        if kind == "close":
            self.socket_open = False
            self.socket_closed = True
            return FrameNotice(record=dict(record), frame=None, events=())
        if kind != "binary":
            return None
        self.socket_open = True
        frame = decode_record_binary(record)
        if frame is None:
            return FrameNotice(record=dict(record), frame=None, events=())
        if frame.direction is Direction.INBOUND:
            self.inbound_type_counts[frame.msg_type] += 1
        events = tuple(self.parser.parse_frame(frame))
        for event in events:
            if isinstance(event, GameStart):
                self.game_starts[event.game_id] = event
                self.game_starts.move_to_end(event.game_id)
                while len(self.game_starts) > MAX_TRACKED_GAME_STARTS:
                    self.game_starts.popitem(last=False)
        return FrameNotice(record=dict(record), frame=frame, events=events)

    def drain(self) -> tuple[FrameNotice, ...]:
        notices: list[FrameNotice] = []
        while True:
            try:
                record = self.session.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                return tuple(notices)
            notice = self._process(record)
            if notice is not None:
                notices.append(notice)

    async def next_notice(self, timeout: float) -> FrameNotice | None:
        try:
            record = await asyncio.wait_for(self.session.frame_queue.get(), timeout=timeout)
        except TimeoutError:
            return None
        return self._process(record)

    async def wait_for(
        self,
        ready: Callable[[], T | None],
        *,
        timeout: float,
        label: str,
        stop_requested: asyncio.Event,
        on_notice: Callable[[FrameNotice], None] | None = None,
    ) -> T:
        deadline = time.monotonic() + timeout
        next_heartbeat = time.monotonic() + 5.0
        while True:
            value = ready()
            if value is not None:
                return value
            if stop_requested.is_set():
                raise StopRequested()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {label} after {timeout:.1f}s")
            notice = await self.next_notice(min(1.0, remaining))
            if notice is None:
                if time.monotonic() >= next_heartbeat:
                    print(f"waiting for {label}...", flush=True)
                    next_heartbeat += 5.0
                continue
            if on_notice is not None:
                on_notice(notice)


def _load_base_names(path: Path) -> frozenset[str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        names = frozenset(str(entry["name"]) for entry in document["map"])
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not load base card map {path}: {error}") from error
    if len(names) != 33:
        raise RuntimeError(f"expected 33 distinct base names in {path}, got {len(names)}")
    return names


def _eligible_tables(snapshot: tuple[TableSummary, ...]) -> list[TableSummary]:
    """Apply precisely the documented pre-join candidate filter."""
    return [
        table
        for table in snapshot
        if table.status == RUNNING
        and table.is_observable
        and table.is_two_player
        and table.bots == 0
    ]


def _table_age_seconds(table: TableSummary, *, now_epoch_ms: int) -> float | None:
    """Return elapsed running time from TableSummary.startTime, if present."""
    if table.start_time is None:
        return None
    return max(0.0, (now_epoch_ms - table.start_time) / 1_000.0)


def _select_probe_table(
    snapshot: tuple[TableSummary, ...],
    *,
    now_epoch_ms: int,
    target_min_age_seconds: float,
    target_max_age_seconds: float,
    excluded_table_keys: set[tuple[int, int | None]],
    rng: random.Random,
) -> TableSummary | None:
    """Choose one unprobed table, uniformly within the moderate age band.

    Reading ``fullGameState`` is the actual base-only filter, so a non-base
    probe costs only one bounded join/leave cycle.  In contrast, keeping a
    base game monopolizes the one session until ``gameFinished``.  Roughly,
    expected time per retained game is ``probe_cost / P(base | band)`` plus
    its remaining base-game wait; the first term is seconds, the latter can
    be minutes.  The band therefore favors the shorter base-game population
    without repeatedly selecting the old, expansion-heavy survival tail.  If
    a sparse snapshot has no row in the band, sample its remaining candidates
    uniformly rather than restoring an old-first ordering.
    """
    candidates = [
        table
        for table in _eligible_tables(snapshot)
        if (table.table_id, table.start_time) not in excluded_table_keys
    ]
    if not candidates:
        return None
    in_band = [
        table
        for table in candidates
        if (
            (age := _table_age_seconds(table, now_epoch_ms=now_epoch_ms)) is not None
            and target_min_age_seconds <= age <= target_max_age_seconds
        )
    ]
    return rng.choice(in_band if in_band else candidates)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _failure_reason(error: BaseException) -> str:
    detail = str(error).splitlines()[0] if str(error) else ""
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def _credential_environment_keys(account_id: str | None) -> tuple[str, str]:
    """Return the non-secret environment variable names for one account.

    The legacy no-account path deliberately remains ``DGAMES_USER`` and
    ``DGAMES_PASS``.  Fleet accounts must use distinct indexed/suffixed keys;
    they never silently fall back to the legacy pair and accidentally share a
    browser session.
    """
    if account_id is None:
        return ("DGAMES_USER", "DGAMES_PASS")
    if not account_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
        for character in account_id
    ):
        raise ValueError("account id must contain only letters, digits, or underscore")
    return (f"DGAMES_USER_{account_id}", f"DGAMES_PASS_{account_id}")


def _load_dotenv_credentials(path: Path, *, keys: tuple[str, str]) -> dict[str, str]:
    """Read only requested credential keys without echoing file contents."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or key not in set(keys):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _load_dgames_credentials(
    *,
    environment: Mapping[str, str] | None = None,
    env_file: Path = DEFAULT_ENV_FILE,
    account_id: str | None = None,
) -> DgamesCredentials | None:
    """Load credentials from the environment, filling absent keys from ``.env``.

    The returned object deliberately has a redacted repr.  Callers must use it
    only to drive the page form; no caller receives raw values for logging or
    persistence.
    """
    source = os.environ if environment is None else environment
    user_key, password_key = _credential_environment_keys(account_id)
    dotenv = _load_dotenv_credentials(env_file, keys=(user_key, password_key))
    username = source.get(user_key)
    password = source.get(password_key)
    if username is None:
        username = dotenv.get(user_key)
    if password is None:
        password = dotenv.get(password_key)
    if not username or not password:
        return None
    return DgamesCredentials(username=username, password=password)


def _atomic_json_write(destination: Path, document: dict[str, object]) -> None:
    """Atomically replace one small manifest, fsyncing the completed file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(document, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _fsync_directory(directory: Path) -> None:
    """Best-effort namespace durability for coordinated output publication."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_json_create(destination: Path, document: dict[str, object]) -> bool:
    """Publish a new JSON file without ever replacing another owner's file.

    ``os.link`` is an atomic create-if-absent publication on the local
    filesystem.  The source temporary lives beside its destination, so this
    works without cross-device rename assumptions and avoids the overwrite
    behavior of ``os.replace``.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(document, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            return False
        _fsync_directory(destination.parent)
        return True
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _publish_staged_capture(staging_path: Path, destination: Path) -> bool:
    """Atomically publish a staged raw capture only if its final name is free."""
    if staging_path == destination:
        return destination.exists()
    if not staging_path.exists():
        return destination.exists()
    try:
        os.link(staging_path, destination)
    except FileExistsError:
        return False
    _fsync_directory(destination.parent)
    return True


def _append_gzip_jsonl_record(destination: Path, record: dict[str, Any]) -> None:
    """Append one independently closed gzip member for crash-safe JSONL.

    Concatenated gzip members are valid gzip.  Closing and fsyncing every
    record means an interrupted write can affect only its last member, never a
    previously durable game frame.  Python's gzip reader transparently reads
    the concatenation.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_frame_record(record).encode("utf-8")
    with destination.open("ab") as raw_file:
        with gzip.GzipFile(fileobj=raw_file, mode="wb", mtime=0) as member:
            member.write(payload)
        raw_file.flush()
        os.fsync(raw_file.fileno())


def _load_seen_game_ids(raw_root: Path) -> set[int]:
    """Treat complete and incomplete on-disk games as permanently non-duplicable."""
    seen: set[int] = set()
    for manifest_path in raw_root.glob("*.manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            seen.add(int(manifest["game_id"]))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            # A malformed operator-edited manifest should not make the collector
            # overwrite a possible capture named by its path.
            try:
                seen.add(int(manifest_path.name.removesuffix(".manifest.json")))
            except ValueError:
                pass
    for raw_path in raw_root.glob("*.jsonl.gz"):
        try:
            seen.add(int(raw_path.name.removesuffix(".jsonl.gz")))
        except ValueError:
            pass
    return seen


def _decode_game_result_details(
    *,
    payload: bytes,
    table_id: int,
    game_id: int,
    player_names_by_id: dict[int, str],
) -> dict[str, object]:
    """Decode the complete spectator GameResult, including final decks.

    The shared parser intentionally normalizes message 14 to scores/placings.
    The collector needs the richer wire result, so this uses the same bounded
    ``Reader`` and log-argument skipper without changing the arena pipeline.
    """
    if len(payload) < 16:
        raise ProtocolError("truncated GameFinished payload")
    marker = table_id.to_bytes(8, "big") + game_id.to_bytes(8, "big")
    offset = payload.find(marker, 8)
    if offset < 0:
        raise ProtocolError("could not locate GameResult table/game boundary")
    if payload.find(marker, offset + 1) >= 0:
        raise ProtocolError("ambiguous GameResult table/game boundary")
    reader = Reader(payload[offset:])
    if reader.u64() != table_id or reader.u64() != game_id:
        raise ProtocolError("GameResult table/game ids do not match active capture")
    rating_type = reader.s32()
    empty_piles = [card_name(wire_id) for wire_id in reader.u32_array()]
    # Reuse the pure-stdlib parser's stable generic log argument reader to
    # skip ScorePart explanations, which evolve independently of GameResult.
    log_argument_reader = ArenaParser()
    players: list[dict[str, object]] = []
    for _ in range(reader.u32()):
        player_id = reader.s32()
        rank = reader.s32()
        total_points = reader.s32()
        used_turns = reader.s32()
        score_parts: list[dict[str, object]] = []
        for _ in range(reader.u32()):
            card = card_name(reader.u32())
            points = reader.s32()
            frequency = reader.s32()
            explanation_display_field = reader.s32()
            explanation_name = reader.u32()
            explanation_depth = reader.s32()
            reader.array(lambda: log_argument_reader._read_log_argument(reader))
            score_parts.append(
                {
                    "card_name": card,
                    "points": points,
                    "frequency": frequency,
                    "explanation_display_field": explanation_display_field,
                    "explanation_name_ordinal": explanation_name,
                    "explanation_depth": explanation_depth,
                }
            )
        final_deck = [
            {"card_name": card_name(reader.u32()), "frequency": reader.s32()}
            for _ in range(reader.u32())
        ]
        resign_index = reader.s32()
        resignation_ordinal = reader.s32()
        players.append(
            {
                "player_id": player_id,
                "player_name": player_names_by_id.get(player_id, str(player_id)),
                "rank": rank,
                "score": {
                    "total_points": total_points,
                    "used_turns": used_turns,
                    "score_parts": score_parts,
                },
                "final_deck_histogram": final_deck,
                "resign_index": resign_index,
                "resignation_type": (
                    None
                    if resignation_ordinal == -1
                    else RESIGNATION_TYPES.get(
                        resignation_ordinal,
                        f"UNKNOWN_{resignation_ordinal}",
                    )
                ),
            }
        )
    auto_continue = reader.boolean()
    continue_allowed = reader.boolean()
    match_completed = reader.boolean()
    reader.finish()
    return {
        "table_id": table_id,
        "game_id": game_id,
        "rating_type_ordinal": None if rating_type == -1 else rating_type,
        "empty_piles": empty_piles,
        "players": players,
        "auto_continue": auto_continue,
        "continue_allowed": continue_allowed,
        "match_completed": match_completed,
    }


class DominionCollector:
    """One long-lived browser collector, optionally coordinated with peers."""

    def __init__(self, args: argparse.Namespace, stats: CollectorStats) -> None:
        self.args = args
        self.stats = stats
        self.account_id = args.account_id
        self.coordination: CoordinationBackend | None = (
            None
            if args.coordination_dir is None
            else FilesystemCoordinationBackend(args.coordination_dir)
        )
        self.stats.coordination_enabled = self.coordination is not None
        self.stats.account_id = self.account_id
        self.credentials = _load_dgames_credentials(account_id=self.account_id)
        self.stats.credentialed_self_heal_available = self.credentials is not None
        self.credentialed_relogin_limiter = CredentialedReloginRateLimiter(
            max_attempts_per_hour=args.max_credentialed_relogins_per_hour,
            minimum_backoff_seconds=args.credentialed_relogin_backoff,
        )
        self.base_names = _load_base_names(args.card_map)
        self.raw_root = args.raw_root
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.seen_game_ids = _load_seen_game_ids(self.raw_root)
        self.session: CollectorSession | None = None
        self.pump: LivePump | None = None
        self.login_handshake: SessionHandshakeTracker | None = None
        self.player_id: int | None = None
        self.snapshot: tuple[TableSummary, ...] | None = None
        self.snapshot_received_monotonic: float | None = None
        self.snapshot_generation = 0
        # Table ids are recycled, so pair each id with its RUNNING startTime.
        # Bounded caches skip a rejected/recently-ended game on later lobby
        # polls without permanently blacklisting a future game in that slot.
        self.attempted_table_keys: OrderedDict[tuple[int, int | None], None] = OrderedDict()
        self.rejected_table_keys: OrderedDict[tuple[int, int | None], None] = OrderedDict()
        self.rejected_game_ids: OrderedDict[int, None] = OrderedDict()
        # One stream for the whole browser session makes each probe choice
        # independent of lobby row order while still leaving tests injectable.
        self.selection_rng = random.Random()
        self.pending: PendingJoin | None = None
        self.active: ActiveCapture | None = None
        # A table is claimed before JOIN_TABLE.  It may briefly wait here for a
        # fleet-wide token before becoming ``pending``.
        self.claimed_table: TableSummary | None = None
        self.table_claim: ClaimLease | None = None
        self.game_claim: ClaimLease | None = None
        self.snapshot_refresh_claim: ClaimLease | None = None
        self.next_claim_refresh_monotonic = 0.0
        self._last_claim_contention_notice_monotonic = 0.0
        self._last_rate_limit_notice_monotonic = 0.0
        self._last_cache_degradation_notice_monotonic = 0.0
        self.poll_inflight = False
        self.poll_attempts = 0
        self.next_poll_monotonic = 0.0
        self.next_poll_retry_monotonic = 0.0
        self.next_join_allowed_monotonic = 0.0
        self.leave_tasks: dict[int, asyncio.Task[None]] = {}
        self.recovery_table_ids: set[int] = set()
        self._in_page_recovery_streak = 0
        self.last_session_reached_healthy_state = False
        # One REMOVE_ACTIVE_SESSIONS request is enough for one unhealthy
        # episode.  Keep this across in-page reloads so a broken server-side
        # session cannot make cleanup traffic routine.
        self._active_sessions_cleanup_attempted = False
        self._publish_coordination_heartbeat(status="starting")

    @property
    def _session(self) -> CollectorSession:
        if self.session is None:
            raise RuntimeError("collector browser session is unavailable")
        return self.session

    @property
    def _pump(self) -> LivePump:
        if self.pump is None:
            raise RuntimeError("collector frame pump is unavailable")
        return self.pump

    @staticmethod
    def _table_key(table: TableSummary) -> tuple[int, int | None]:
        return (table.table_id, table.start_time)

    @staticmethod
    def _remember_bounded(
        cache: OrderedDict[T, None],
        value: T,
        *,
        limit: int,
    ) -> None:
        cache[value] = None
        cache.move_to_end(value)
        while len(cache) > limit:
            cache.popitem(last=False)

    def _remember_table_attempt(self, table: TableSummary) -> None:
        self._remember_bounded(
            self.attempted_table_keys,
            self._table_key(table),
            limit=MAX_REMEMBERED_TABLES,
        )

    def _remember_rejected_table(self, table: TableSummary) -> None:
        self._remember_bounded(
            self.rejected_table_keys,
            self._table_key(table),
            limit=MAX_REMEMBERED_TABLES,
        )

    def _remember_rejected_game(self, game_id: int) -> None:
        self._remember_bounded(
            self.rejected_game_ids,
            game_id,
            limit=MAX_REJECTED_GAME_IDS,
        )

    def _publish_coordination_heartbeat(self, *, status: str) -> None:
        """Expose only aggregate health; credentials never enter this document."""
        coordination = self.coordination
        if coordination is None or self.account_id is None:
            return
        contention_rate = self.stats.claim_contention_rate()
        try:
            coordination.write_account_heartbeat(
                account_id=self.account_id,
                document={
                    "schema_version": 1,
                    "pid": os.getpid(),
                    "status": status,
                    "started_at_utc": self.stats.started_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "games_live": 1 if self.active is not None else 0,
                    "captured_completed": self.stats.captured_completed,
                    "captured_started": self.stats.captured_started,
                    "tables_probed": self.stats.tables_probed,
                    "skipped_non_base": self.stats.skipped_non_base,
                    "joins_attempted": self.stats.joins_attempted,
                    "claim_attempts": self.stats.claim_attempts(),
                    "claim_contentions": self.stats.claim_contentions(),
                    "claim_contention_rate": contention_rate,
                    "claim_refresh_failures": self.stats.claim_refresh_failures,
                    "shared_snapshot_hits": self.stats.shared_snapshot_hits,
                    "rate_limit_waits": self.stats.rate_limit_waits,
                    "games_per_hour": self.stats.games_per_hour(),
                },
            )
        except CoordinationUnavailable:
            # Health publication must not bring down a healthy browser.  Claim
            # and token operations remain fail-closed elsewhere.
            return

    def _claim_refresh_period(self) -> float:
        limits = [self.args.claim_refresh_interval]
        if self.table_claim is not None or self.game_claim is not None:
            limits.append(self.args.claim_ttl / 3.0)
        if self.snapshot_refresh_claim is not None:
            limits.append(self.args.snapshot_refresh_ttl / 3.0)
        return min(limits)

    def _schedule_claim_refresh(self, now: float | None = None) -> None:
        if self.coordination is not None:
            self.next_claim_refresh_monotonic = (
                time.monotonic() if now is None else now
            ) + self._claim_refresh_period()

    def _release_claim_lease(self, lease: ClaimLease, *, label: str) -> None:
        coordination = self.coordination
        if coordination is None:
            return
        try:
            coordination.release_claim(lease)
        except CoordinationUnavailable as error:
            # The TTL makes a failed final release recoverable after a crash or
            # filesystem outage.  Do not let it mask a durable capture result.
            print(f"could not release {label} claim: {_failure_reason(error)}", flush=True)

    def _release_table_claim(self) -> None:
        lease, self.table_claim = self.table_claim, None
        self.claimed_table = None
        if lease is not None:
            self._release_claim_lease(lease, label="table")

    def _release_game_claim(self) -> None:
        lease, self.game_claim = self.game_claim, None
        if lease is not None:
            self._release_claim_lease(lease, label="game")

    def _release_snapshot_refresh_claim(self) -> None:
        lease, self.snapshot_refresh_claim = self.snapshot_refresh_claim, None
        if lease is not None:
            self._release_claim_lease(lease, label="snapshot refresh")

    def _release_capture_claims(self) -> None:
        self._release_game_claim()
        self._release_table_claim()

    def _acquire_table_claim(self, table: TableSummary, *, now: float) -> bool:
        coordination = self.coordination
        if coordination is None:
            return True
        assert self.account_id is not None
        self.stats.table_claim_attempts += 1
        try:
            acquisition = coordination.acquire_claim(
                scope="table",
                key=table.table_id,
                owner_id=self.account_id,
                pid=os.getpid(),
                ttl_seconds=self.args.claim_ttl,
            )
        except CoordinationUnavailable as error:
            # Never JOIN_TABLE if exclusive ownership cannot be established.
            self.next_join_allowed_monotonic = now + 1.0
            print(f"table claim unavailable; deferring join: {_failure_reason(error)}", flush=True)
            return False
        if not acquisition.acquired:
            self.stats.table_claim_contentions += 1
            if now - self._last_claim_contention_notice_monotonic >= 10.0:
                print(f"table {table.table_id} already claimed by a fleet peer", flush=True)
                self._last_claim_contention_notice_monotonic = now
            return False
        assert acquisition.lease is not None
        self.table_claim = acquisition.lease
        self.claimed_table = table
        if acquisition.reclaimed_stale:
            self.stats.stale_claims_reclaimed += 1
        self._schedule_claim_refresh(now)
        return True

    def _acquire_game_claim(self, game_id: int) -> bool:
        coordination = self.coordination
        if coordination is None:
            return True
        assert self.account_id is not None
        self.stats.game_claim_attempts += 1
        try:
            acquisition = coordination.acquire_claim(
                scope="game",
                key=game_id,
                owner_id=self.account_id,
                pid=os.getpid(),
                ttl_seconds=self.args.claim_ttl,
            )
        except CoordinationUnavailable as error:
            print(f"game claim unavailable; skipping game {game_id}: {_failure_reason(error)}", flush=True)
            return False
        if not acquisition.acquired:
            self.stats.game_claim_contentions += 1
            return False
        assert acquisition.lease is not None
        self.game_claim = acquisition.lease
        if acquisition.reclaimed_stale:
            self.stats.stale_claims_reclaimed += 1
        self._schedule_claim_refresh()
        return True

    def _refresh_coordination_claims(self, now: float) -> None:
        """Keep live table/game leases fresh before their stale TTL.

        A lost table or game lease is treated as a transport recovery boundary:
        continuing to record after ownership is uncertain would be less safe
        than publishing the durable prefix as incomplete and detaching.
        """
        coordination = self.coordination
        if coordination is None or now < self.next_claim_refresh_monotonic:
            return
        for label, lease in (("table", self.table_claim), ("game", self.game_claim)):
            if lease is None:
                continue
            try:
                refreshed = coordination.refresh_claim(lease)
            except CoordinationUnavailable as error:
                self.stats.claim_refresh_failures += 1
                raise RecoverPage(
                    f"could not refresh {label} claim: {_failure_reason(error)}"
                ) from error
            if not refreshed:
                self.stats.claim_refresh_failures += 1
                if label == "table":
                    self.table_claim = None
                    self.claimed_table = None
                else:
                    self.game_claim = None
                raise RecoverPage(f"lost {label} claim ownership during collection")
            self.stats.claim_refreshes += 1
        if self.snapshot_refresh_claim is not None:
            try:
                if not coordination.refresh_claim(self.snapshot_refresh_claim):
                    self.snapshot_refresh_claim = None
            except CoordinationUnavailable:
                # Cache coordination is explicitly allowed to degrade to an
                # ordinary per-account lobby poll.
                self.snapshot_refresh_claim = None
                self.stats.snapshot_cache_degradations += 1
        self._schedule_claim_refresh(now)

    def _staging_raw_path(self, game_id: int, destination: Path) -> Path:
        if self.coordination is None:
            return destination
        safe_account_id = self.account_id or "unknown"
        return destination.with_name(
            f".{game_id}.{safe_account_id}.{os.getpid()}.{secrets.token_hex(8)}.jsonl.gz.tmp"
        )

    def _reserve_capture_manifest(self, path: Path, manifest: dict[str, object]) -> bool:
        if self.coordination is None:
            _atomic_json_write(path, manifest)
            return True
        return _atomic_json_create(path, manifest)

    def _publish_active_raw_capture(
        self,
        active: ActiveCapture,
        *,
        allow_missing_staging: bool = False,
    ) -> bool:
        if self.coordination is None:
            return True
        if not active.raw_staging_path.exists():
            return allow_missing_staging and not active.raw_path.exists()
        published = _publish_staged_capture(active.raw_staging_path, active.raw_path)
        if published and active.raw_staging_path != active.raw_path:
            try:
                active.raw_staging_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        return published

    def _mark_session_healthy(self) -> None:
        """Reset transient recovery pressure after real authenticated operation."""
        self.last_session_reached_healthy_state = True
        self._in_page_recovery_streak = 0
        self._active_sessions_cleanup_attempted = False
        if self.login_handshake is not None:
            self.login_handshake._clear_parked_evidence()

    def _reset_page_cycle_state(self) -> None:
        """Forget transport-local state while keeping the browser context alive."""
        self.player_id = None
        self.snapshot = None
        self.snapshot_received_monotonic = None
        self.snapshot_generation = 0
        self.poll_inflight = False
        self.poll_attempts = 0
        self.next_poll_monotonic = 0.0
        self.next_poll_retry_monotonic = 0.0
        self.next_join_allowed_monotonic = 0.0

    def _page_is_usable(self) -> bool:
        session = self.session
        if session is None or session.context is None or session.page is None:
            return False
        try:
            return not session.page.is_closed()
        except Exception:
            return False

    async def _cancel_leave_tasks(self) -> None:
        tasks, self.leave_tasks = tuple(self.leave_tasks.values()), {}
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _discard_queued_frames(self) -> None:
        """Drop old-socket records before installing a new page generation pump."""
        while True:
            try:
                record = self._session.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if record is None:
                # Preserve the wakeup for any code that observes the queue
                # after this hard transport failure.
                self._session.frame_queue.put_nowait(None)
                raise RestartBrowser("browser transport stopped during page recovery")

    async def run_browser_session(self, stop_requested: asyncio.Event) -> None:
        """Run one Playwright context, recovering live sockets in the same page."""
        self.session = CollectorSession(
            output_root=self.raw_root / "runs",
            profile_dir=self.args.profile,
            url=self.args.url,
            headless=not self.args.headful,
        )
        self.pump = None
        self.login_handshake = None
        self.last_session_reached_healthy_state = False
        self._in_page_recovery_streak = 0
        try:
            print(
                f"launching authenticated browser ({'headful' if self.args.headful else 'headless'})...",
                flush=True,
            )
            print(
                "credentialed self-heal enabled"
                if self.credentials is not None
                else "credentialed self-heal unavailable; using profile-session-only mode",
                flush=True,
            )
            try:
                await self._session.start()
            except Exception as error:
                raise RestartBrowser(
                    f"could not launch browser context: {_failure_reason(error)}"
                ) from error
            self.login_handshake = SessionHandshakeTracker(self._session.game_socket)
            self.pump = LivePump(
                self._session,
                login_handshake=self.login_handshake,
            )
            while True:
                try:
                    await self._run_page_cycle(stop_requested)
                    return
                except RecoverPage as error:
                    await self._recover_page_in_place(error, stop_requested)
        except (
            StopRequested,
            RestartBrowser,
            SessionExpired,
            CredentialedReloginRateLimitExceeded,
            SustainedFailure,
        ):
            raise
        finally:
            await self._shutdown_browser_session()

    async def _run_page_cycle(self, stop_requested: asyncio.Event) -> None:
        """Authenticate and collect until this page needs an in-place reload."""
        self._reset_page_cycle_state()
        await self._wait_for_login(stop_requested)
        await self._leave_recovered_tables()
        self.next_poll_monotonic = time.monotonic()
        self.next_poll_retry_monotonic = time.monotonic()
        await self._obtain_initial_snapshot(stop_requested)
        # A successful authenticated lobby cycle is a genuine healthy state;
        # starting/completing a base capture also calls this below.
        self._mark_session_healthy()
        await self._main_loop(stop_requested)

    async def _recover_page_in_place(
        self,
        error: RecoverPage,
        stop_requested: asyncio.Event,
    ) -> None:
        """Reload a live page without destroying its authenticated profile/context."""
        if stop_requested.is_set():
            raise StopRequested()
        await self._abandon_transport_attachments(str(error))
        if not self._page_is_usable():
            raise RestartBrowser(f"page is unavailable after transport failure: {error}")

        next_streak = self._in_page_recovery_streak + 1
        if next_streak >= self.args.max_consecutive_failures:
            self.stats.failed += 1
            raise SustainedFailure(
                f"{next_streak} consecutive in-page transport recoveries without "
                f"returning to healthy operation; last failure: {error}"
            )

        self._in_page_recovery_streak = next_streak
        print(
            f"recovering in the existing browser page: {error}; "
            f"attempt {next_streak}",
            flush=True,
        )
        # Records can arrive after a close notification.  The timestamp gate
        # on the replacement pump prevents such stale records from poisoning
        # the new page generation's socket state.
        recovery_started_ms = int(time.time() * 1_000)
        self._discard_queued_frames()
        self.pump = LivePump(
            self._session,
            ignore_records_before_ms=recovery_started_ms,
            login_handshake=self.login_handshake,
        )
        try:
            assert self._session.page is not None
            await self._session.page.reload(wait_until="domcontentloaded")
        except Exception as reload_error:
            raise RestartBrowser(
                f"could not recover the live page in place: {_failure_reason(reload_error)}"
            ) from reload_error
        self.stats.failed += 1
        self.stats.in_page_recoveries += 1

    async def _abandon_transport_attachments(self, reason: str) -> None:
        """Durably close capture state before a socket/page recovery loses frames."""
        if self.active is not None:
            self.recovery_table_ids.add(self.active.table.table_id)
            self._mark_active_incomplete(f"transport recovery: {reason}")
        if self.pending is not None:
            self.recovery_table_ids.add(self.pending.table.table_id)
            self.pending = None
            self._release_capture_claims()
        if self.claimed_table is not None:
            self._release_table_claim()
        self._release_snapshot_refresh_claim()
        self.recovery_table_ids.update(self.leave_tasks)
        await self._cancel_leave_tasks()

    async def _leave_recovered_tables(self) -> None:
        """Detach any table the reloaded page may have reattached automatically."""
        for table_id in tuple(sorted(self.recovery_table_ids)):
            await self._leave_table(table_id)
            self.recovery_table_ids.discard(table_id)
        if self.recovery_table_ids:
            # The loop above either removes every entry or raises RecoverPage.
            raise RecoverPage("could not leave a table recovered after page reload")

    async def _shutdown_browser_session(self) -> None:
        """Best-effort teardown; a summary must survive a closing browser/page."""
        session = self.session
        table_ids = set(self.recovery_table_ids)
        if self.active is not None:
            table_ids.add(self.active.table.table_id)
            self._mark_active_incomplete("browser session ended before gameFinished")
        if self.pending is not None:
            table_ids.add(self.pending.table.table_id)
            self.pending = None
            self._release_capture_claims()
        if self.claimed_table is not None:
            self._release_table_claim()
        self._release_snapshot_refresh_claim()
        table_ids.update(self.leave_tasks)
        await self._cancel_leave_tasks()
        if session is not None and self._page_is_usable() and self.player_id is not None:
            for table_id in sorted(table_ids):
                try:
                    await self._leave_table(table_id)
                except Exception:
                    # The context is about to close; never let a final LEAVE
                    # mask the durable manifest or run summary.
                    pass
        self.recovery_table_ids.clear()
        try:
            if session is not None:
                await session.stop()
        finally:
            self.session = None
            self.pump = None
            self.login_handshake = None
            self.player_id = None

    async def _wait_for_login(self, stop_requested: asyncio.Event) -> None:
        print("waiting for the game websocket and existing logged-in session...", flush=True)
        deadline = time.monotonic() + self.args.startup_timeout
        next_progress = time.monotonic() + 5.0
        while True:
            player_id = self._pump.parser.our_player_id
            if self._pump.socket_open and player_id is not None:
                self.player_id = player_id
                print(f"authenticated browser session as player {player_id}", flush=True)
                return
            if stop_requested.is_set():
                raise StopRequested()
            if not self._page_is_usable():
                raise RestartBrowser("browser page closed while waiting for login")
            now = time.monotonic()
            parked_action = _parked_login_action(
                self._pump.login_handshake,
                credentials=self.credentials,
                now=now,
                grace_seconds=self.args.session_login_grace,
            )
            if parked_action is ParkedLoginAction.SESSION_EXPIRED:
                login_form_visible = await self._login_form_is_visible()
                corroboration = " (visible login form corroborated)" if login_form_visible else ""
                message = (
                    f"{SESSION_EXPIRED_MESSAGE}; credentialed self-heal unavailable{corroboration}"
                )
                print(message, flush=True)
                raise SessionExpired(message)
            if parked_action is ParkedLoginAction.CREDENTIALED_RELOGIN:
                player_id = await self._credentialed_relogin(stop_requested)
                self.player_id = player_id
                print(
                    "credentialed re-login restored authenticated browser session "
                    f"as player {player_id}",
                    flush=True,
                )
                return
            remaining = deadline - now
            if remaining <= 0:
                raise RecoverPage("timed out waiting for game websocket login")
            notice = await self._pump.next_notice(min(0.25, remaining))
            if notice is not None:
                self._handle_notice(notice)
            elif now >= next_progress:
                print("waiting for game websocket login...", flush=True)
                next_progress += 5.0

    async def _login_form_is_visible(self) -> bool:
        """Optionally corroborate the wire signature without depending on UI text."""
        if not self._page_is_usable():
            return False
        try:
            assert self._session.page is not None
            return bool(
                await asyncio.wait_for(
                    self._session.page.evaluate(
                        """() => Array.from(document.querySelectorAll(
                            'input[type="password"]'
                        )).some((input) => {
                            const style = window.getComputedStyle(input);
                            return style.display !== 'none' && style.visibility !== 'hidden'
                                && input.getClientRects().length > 0;
                        })"""
                    ),
                    timeout=1.0,
                )
            )
        except Exception:
            return False

    async def _wait_for_stop_or_delay(
        self,
        delay_seconds: float,
        stop_requested: asyncio.Event,
    ) -> None:
        """Wait responsively, limiting each unattended wait slice to one minute."""
        deadline = time.monotonic() + delay_seconds
        while True:
            if stop_requested.is_set():
                raise StopRequested()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                await asyncio.wait_for(stop_requested.wait(), timeout=min(60.0, remaining))
            except TimeoutError:
                continue
            raise StopRequested()

    async def _reserve_credentialed_relogin_slot(
        self,
        stop_requested: asyncio.Event,
    ) -> int:
        """Apply the per-hour cap and backoff before one page form submission."""
        while True:
            now = time.monotonic()
            slot = self.credentialed_relogin_limiter.check(now=now)
            if not slot.permitted:
                raise CredentialedReloginRateLimitExceeded(
                    "credentialed re-login safety cap reached: "
                    f"{self.args.max_credentialed_relogins_per_hour} attempts in the past hour"
                )
            if slot.backoff_seconds <= 0:
                self.credentialed_relogin_limiter.record(now=now)
                self.stats.credentialed_relogins += 1
                return self.stats.credentialed_relogins
            print(
                f"credentialed re-login backoff: waiting {slot.backoff_seconds:.1f}s",
                flush=True,
            )
            await self._wait_for_stop_or_delay(slot.backoff_seconds, stop_requested)

    async def _submit_credentialed_login_form(self, credentials: DgamesCredentials) -> None:
        """Fill and submit the client's own visible login form via Playwright.

        The page remains responsible for the versioned LOGIN handshake and for
        persisting the new player/session pair in its Chromium profile.  Every
        exception deliberately becomes a credential-free recovery message.
        """
        if not self._page_is_usable():
            raise RestartBrowser("browser page closed before credentialed re-login")
        assert self._session.page is not None
        page = self._session.page
        try:
            prepared = await asyncio.wait_for(
                page.evaluate(
                    """() => {
                        const marker = "data-dgames-collector-login";
                        const visible = (element) => {
                            const style = window.getComputedStyle(element);
                            return !element.disabled
                                && style.display !== "none"
                                && style.visibility !== "hidden"
                                && element.getClientRects().length > 0;
                        };
                        for (const element of document.querySelectorAll(`[${marker}]`)) {
                            element.removeAttribute(marker);
                        }
                        const password = Array.from(document.querySelectorAll("input"))
                            .filter(visible)
                            .find((input) => input.type.toLowerCase() === "password");
                        if (!password) {
                            return null;
                        }
                        const scope = password.closest("form") || document;
                        const username = Array.from(scope.querySelectorAll("input"))
                            .filter((input) => input !== password && visible(input))
                            .filter((input) => ["text", "email", ""].includes(
                                input.type.toLowerCase()
                            ))
                            .sort((left, right) => {
                                const score = (input) => {
                                    const hint = [
                                        input.autocomplete,
                                        input.name,
                                        input.id,
                                        input.placeholder,
                                    ].join(" ").toLowerCase();
                                    return (input.autocomplete === "username" ? 100 : 0)
                                        + (/user|email|login|name/.test(hint) ? 20 : 0)
                                        + (input.type.toLowerCase() === "email" ? 5 : 0);
                                };
                                return score(right) - score(left);
                            })[0];
                        if (!username) {
                            return null;
                        }
                        const submitCandidates = Array.from(
                            scope.querySelectorAll("button, input[type=submit]")
                        ).filter(visible);
                        const submit = submitCandidates.find(
                            (element) => element.getAttribute("type") === "submit"
                        ) || submitCandidates.find((element) =>
                            /log\\s*in|sign\\s*in/i.test(
                                "value" in element ? element.value : element.textContent || ""
                            )
                        );
                        username.setAttribute(marker, "username");
                        password.setAttribute(marker, "password");
                        if (submit) {
                            submit.setAttribute(marker, "submit");
                        }
                        return {has_submit: Boolean(submit)};
                    }"""
                ),
                timeout=3.0,
            )
        except Exception as error:
            raise RecoverPage("could not locate the dominion.games login form") from error
        if not isinstance(prepared, dict):
            raise RecoverPage("could not locate the dominion.games login form")
        try:
            username = page.locator('[data-dgames-collector-login="username"]')
            password = page.locator('[data-dgames-collector-login="password"]')
            await username.fill(credentials.username, timeout=3_000)
            await password.fill(credentials.password, timeout=3_000)
            if prepared.get("has_submit"):
                await page.locator('[data-dgames-collector-login="submit"]').click(timeout=3_000)
            else:
                await password.press("Enter", timeout=3_000)
        except Exception as error:
            raise RecoverPage("could not submit credentialed login form") from error

    async def _login_form_reports_already_logged_in(self) -> bool:
        """Read only a boolean UI signal; never retain the form's text or values."""
        if not self._page_is_usable():
            return False
        try:
            assert self._session.page is not None
            return bool(
                await asyncio.wait_for(
                    self._session.page.evaluate(
                        """() => {
                            const text = (document.body?.innerText || "").toLowerCase();
                            return text.includes("already_logged_in")
                                || text.includes("already logged in");
                        }"""
                    ),
                    timeout=1.0,
                )
            )
        except Exception:
            return False

    async def _await_credentialed_login_outcome(
        self,
        *,
        previous_login_requests: int,
        stop_requested: asyncio.Event,
    ) -> tuple[CredentialedLoginOutcome, int | None]:
        """Wait for loginSuccess or the visible ALREADY_LOGGED_IN page response."""
        deadline = time.monotonic() + self.args.credentialed_login_timeout
        while True:
            player_id = self._pump.parser.our_player_id
            if self._pump.socket_open and player_id is not None:
                return (CredentialedLoginOutcome.SUCCEEDED, player_id)
            if stop_requested.is_set():
                raise StopRequested()
            if not self._page_is_usable():
                raise RestartBrowser("browser page closed during credentialed re-login")
            if (
                self._pump.login_handshake.credential_login_requests > previous_login_requests
                and await self._login_form_reports_already_logged_in()
            ):
                return (CredentialedLoginOutcome.ALREADY_LOGGED_IN, None)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return (CredentialedLoginOutcome.TIMED_OUT, None)
            notice = await self._pump.next_notice(min(0.25, remaining))
            if notice is not None:
                self._handle_notice(notice)

    async def _credentialed_login_attempt(
        self,
        credentials: DgamesCredentials,
        stop_requested: asyncio.Event,
    ) -> tuple[CredentialedLoginOutcome, int | None]:
        """Perform one rate-limited form submit and await its safe outcome."""
        previous_login_requests = self._pump.login_handshake.credential_login_requests
        attempt = await self._reserve_credentialed_relogin_slot(stop_requested)
        print(
            f"credentialed re-login attempt {attempt} "
            f"(limit {self.args.max_credentialed_relogins_per_hour}/hour)",
            flush=True,
        )
        await self._submit_credentialed_login_form(credentials)
        return await self._await_credentialed_login_outcome(
            previous_login_requests=previous_login_requests,
            stop_requested=stop_requested,
        )

    async def _credentialed_relogin(self, stop_requested: asyncio.Event) -> int:
        """Self-heal a wiped profile by using the page's supported login path."""
        credentials = self.credentials
        if credentials is None:
            raise RuntimeError("credentialed re-login was invoked without credentials")
        print(
            "saved profile session unavailable; starting credentialed re-login self-heal",
            flush=True,
        )
        outcome, player_id = await self._credentialed_login_attempt(credentials, stop_requested)
        if outcome is CredentialedLoginOutcome.SUCCEEDED:
            assert player_id is not None
            return player_id
        if outcome is CredentialedLoginOutcome.ALREADY_LOGGED_IN:
            if self._active_sessions_cleanup_attempted:
                raise RecoverPage(
                    "credentialed login reported another active session after prior cleanup"
                )
            self._active_sessions_cleanup_attempted = True
            print(
                "credentialed login reported ALREADY_LOGGED_IN; requesting active-session "
                "cleanup and retrying once",
                flush=True,
            )
            try:
                await self._session.send_frame(REMOVE_ACTIVE_SESSIONS, b"")
            except Exception as error:
                raise RecoverPage("could not request active-session cleanup") from error
            outcome, player_id = await self._credentialed_login_attempt(credentials, stop_requested)
            if outcome is CredentialedLoginOutcome.SUCCEEDED:
                assert player_id is not None
                return player_id
            raise RecoverPage(
                "credentialed re-login did not reach login success after active-session cleanup"
            )
        raise RecoverPage("credentialed re-login did not reach login success")

    async def _obtain_initial_snapshot(self, stop_requested: asyncio.Event) -> None:
        """Request/retry a fresh lobby snapshot before any spectator join."""
        deadline = time.monotonic() + self.args.startup_timeout
        self._send_lobby_request(force=True)
        observed_generation = self.snapshot_generation
        while self.snapshot_generation == observed_generation:
            if stop_requested.is_set():
                raise StopRequested()
            if not self._page_is_usable():
                raise RestartBrowser("browser page closed while waiting for tablesOverview")
            now = time.monotonic()
            if now >= deadline:
                raise RecoverPage("no tablesOverview before startup deadline")
            self._refresh_coordination_claims(now)
            await self._service_poll(now)
            notice = await self._pump.next_notice(min(1.0, deadline - now))
            if notice is not None:
                self._handle_notice(notice)
        assert self.snapshot is not None
        print(
            f"fresh snapshot: {len(self.snapshot)} tables; "
            f"{self.stats.last_snapshot_candidates} eligible human observable 2-player running",
            flush=True,
        )

    def _send_lobby_request(self, *, force: bool = False) -> None:
        """Schedule a bounded request; actual send is awaited by _service_poll."""
        if force:
            self.next_poll_monotonic = 0.0
            self.next_poll_retry_monotonic = 0.0

    def _try_use_shared_snapshot(self, now: float) -> bool:
        """Use a fresh peer snapshot, with cache failure falling back safely."""
        coordination = self.coordination
        if coordination is None:
            return False
        try:
            cached = coordination.read_lobby_snapshot(
                max_age_seconds=min(
                    self.args.snapshot_cache_ttl,
                    self.args.max_snapshot_age,
                ),
            )
        except CoordinationUnavailable as error:
            self.stats.snapshot_cache_degradations += 1
            if now - self._last_cache_degradation_notice_monotonic >= 10.0:
                print(
                    f"shared lobby cache unavailable; using per-account polling: "
                    f"{_failure_reason(error)}",
                    flush=True,
                )
                self._last_cache_degradation_notice_monotonic = now
            return False
        if cached is None:
            return False
        try:
            snapshot = tuple(_table_from_cache_document(row) for row in cached.payload)
        except (TypeError, ValueError) as error:
            self.stats.snapshot_cache_degradations += 1
            print(
                f"shared lobby cache was invalid; using per-account polling: "
                f"{_failure_reason(error)}",
                flush=True,
            )
            return False
        self.stats.shared_snapshot_hits += 1
        self._receive_snapshot(snapshot, cached_age_seconds=cached.age_seconds)
        return True

    def _try_acquire_snapshot_refresh(self, now: float) -> bool:
        """Elect exactly one account to issue the next lobby poll."""
        coordination = self.coordination
        if coordination is None:
            return True
        assert self.account_id is not None
        try:
            acquisition = coordination.acquire_snapshot_refresh(
                owner_id=self.account_id,
                pid=os.getpid(),
                ttl_seconds=self.args.snapshot_refresh_ttl,
            )
        except CoordinationUnavailable as error:
            self.stats.snapshot_cache_degradations += 1
            if now - self._last_cache_degradation_notice_monotonic >= 10.0:
                print(
                    f"shared lobby refresh unavailable; using per-account polling: "
                    f"{_failure_reason(error)}",
                    flush=True,
                )
                self._last_cache_degradation_notice_monotonic = now
            # Cache unavailability is explicitly non-fatal and should never
            # prevent the legacy request path from making progress.
            return True
        if acquisition.acquired:
            assert acquisition.lease is not None
            self.snapshot_refresh_claim = acquisition.lease
            self.stats.shared_snapshot_refreshes += 1
            self._schedule_claim_refresh(now)
            return True
        # A peer is now fetching the same snapshot.  Poll again shortly: it
        # will either become readable from cache or safely retry after lock TTL.
        self.next_poll_monotonic = now + min(1.0, self.args.snapshot_cache_ttl)
        if now - self._last_claim_contention_notice_monotonic >= 10.0:
            print("shared lobby refresh held by a fleet peer", flush=True)
            self._last_claim_contention_notice_monotonic = now
        return False

    async def _service_poll(self, now: float) -> None:
        """Keep polling without blocking game frame processing or hammering silence."""
        due = (
            (not self.poll_inflight and now >= self.next_poll_monotonic)
            or (self.poll_inflight and now >= self.next_poll_retry_monotonic)
        )
        if not due:
            return
        if self.poll_inflight and self.poll_attempts >= self.args.max_poll_retries:
            self._release_snapshot_refresh_claim()
            raise RecoverPage(
                f"tablesOverview stayed silent across {self.poll_attempts} bounded retries"
            )
        if not self.poll_inflight:
            if self._try_use_shared_snapshot(now):
                return
            if not self._try_acquire_snapshot_refresh(now):
                return
        try:
            await self._session.send_frame(
                REQUEST_UPDATE,
                Writer().s32(UPDATE_TABLES).build(),
            )
        except Exception as error:
            raise RecoverPage(
                f"could not send lobby poll: {_failure_reason(error)}"
            ) from error
        self.poll_inflight = True
        self.poll_attempts += 1
        self.next_poll_retry_monotonic = now + TABLES_RETRY_SECONDS
        print(
            f"lobby poll request {self.poll_attempts} "
            f"({'retry' if self.poll_attempts > 1 else 'new'})",
            flush=True,
        )

    def _receive_snapshot(
        self,
        snapshot: tuple[TableSummary, ...],
        *,
        cached_age_seconds: float | None = None,
    ) -> None:
        received_monotonic = time.monotonic()
        if cached_age_seconds is None and self.snapshot_refresh_claim is not None:
            try:
                self.coordination is not None and self.coordination.write_lobby_snapshot(
                    lease=self.snapshot_refresh_claim,
                    payload=[_table_to_cache_document(table) for table in snapshot],
                )
            except CoordinationUnavailable as error:
                self.stats.snapshot_cache_degradations += 1
                print(
                    f"could not update shared lobby cache; continuing locally: "
                    f"{_failure_reason(error)}",
                    flush=True,
                )
            finally:
                self._release_snapshot_refresh_claim()
        self.snapshot = snapshot
        self.snapshot_received_monotonic = received_monotonic - max(0.0, cached_age_seconds or 0.0)
        self.snapshot_generation += 1
        self.poll_inflight = False
        self.poll_attempts = 0
        if cached_age_seconds is None:
            self.next_poll_monotonic = self.snapshot_received_monotonic + self.args.poll_interval
        else:
            until_cache_stale = max(
                0.05,
                self.args.snapshot_cache_ttl - max(0.0, cached_age_seconds),
            )
            self.next_poll_monotonic = received_monotonic + min(
                self.args.poll_interval,
                until_cache_stale,
            )
        self.stats.snapshots_received += 1
        self.stats.last_snapshot_tables = len(snapshot)
        self.stats.last_snapshot_candidates = len(_eligible_tables(snapshot))

    def _handle_notice(self, notice: FrameNotice) -> None:
        """Process one frame in arrival order and mirror it into the active capture."""
        if self.pending is not None:
            self._buffer_pending_notice(notice)
        if self.active is not None:
            if _is_sensitive_record(
                notice.record,
                game_socket=self._session.game_socket,
            ):
                # A browser re-login may contain a session token.  Preserve the
                # already durable game prefix, mark it incomplete, and recover
                # the live page without ever putting that record in a raw game
                # archive.
                self.recovery_table_ids.add(self.active.table.table_id)
                self._mark_active_incomplete("browser reauthentication observed")
                raise RecoverPage("browser reauthenticated during active capture")
            _append_gzip_jsonl_record(self.active.raw_staging_path, notice.record)

        frame = notice.frame
        if frame is None:
            if self._pump.socket_closed:
                raise RecoverPage("game websocket closed")
            return
        if frame.direction is Direction.INBOUND and frame.msg_type == TABLES_OVERVIEW:
            self._receive_snapshot(decode_tables_overview(frame.payload))
        if self.pending is not None:
            self._handle_pending_events(notice)
        if self.active is not None and frame.direction is Direction.INBOUND and frame.msg_type == GAME_FINISHED:
            self._complete_active_capture(frame)

    def _buffer_pending_notice(self, notice: FrameNotice) -> None:
        """Keep only a bounded pre-eligibility buffer for one short join attempt."""
        pending = self.pending
        assert pending is not None
        data = notice.record.get("data", "")
        data_bytes = len(data) if isinstance(data, (bytes, str)) else 0
        next_size = pending.raw_bytes + data_bytes + 256
        if next_size > MAX_PENDING_JOIN_BUFFER_BYTES:
            self.recovery_table_ids.add(pending.table.table_id)
            self.pending = None
            self._release_table_claim()
            raise RecoverPage(
                f"pending join buffer exceeded {MAX_PENDING_JOIN_BUFFER_BYTES // (1024 * 1024)} MiB"
            )
        pending.raw_records.append(notice.record)
        pending.raw_bytes = next_size

    def _handle_pending_events(self, notice: FrameNotice) -> None:
        pending = self.pending
        assert pending is not None
        for event in notice.events:
            if isinstance(event, GameStart):
                pending.game_start = event
            elif isinstance(event, FullState):
                self._decide_pending_join(pending, event)
                return

    def _decide_pending_join(self, pending: PendingJoin, full_state: FullState) -> None:
        """Perform the mandatory join-then-decide filter and game-id claim."""
        card_type_names = tuple(sorted(name for name, _count in full_state.card_counts))
        foreign = tuple(name for name in card_type_names if name not in self.base_names)
        self.stats.tables_probed += 1
        if not foreign:
            self.stats.base_only_tables_probed += 1
        if foreign and not self.args.no_base_filter:
            self._remember_rejected_game(full_state.game_id)
            self._remember_rejected_table(pending.table)
            self.stats.skipped_non_base += 1
            print(
                f"skipping table {pending.table.table_id}, game {full_state.game_id}: "
                f"non-base cards {', '.join(foreign[:4])}{'...' if len(foreign) > 4 else ''}",
                flush=True,
            )
            self.pending = None
            self._release_table_claim()
            self.next_join_allowed_monotonic = time.monotonic() + self.args.join_delay
            self._schedule_leave(pending.table.table_id)
            return
        if (
            full_state.game_id in self.seen_game_ids
            or full_state.game_id in self.rejected_game_ids
        ):
            self._remember_rejected_table(pending.table)
            self.stats.skipped_duplicates += 1
            print(
                f"skipping seen game {full_state.game_id} from table {pending.table.table_id}",
                flush=True,
            )
            self.pending = None
            self._release_table_claim()
            self.next_join_allowed_monotonic = time.monotonic() + self.args.join_delay
            self._schedule_leave(pending.table.table_id)
            return

        # A table id is only a pre-join exclusion key.  Once fullGameState
        # supplies the durable game id, acquire the second lease before any
        # shared-output reservation or raw-frame write.
        if not self._acquire_game_claim(full_state.game_id):
            self._remember_rejected_table(pending.table)
            self.stats.skipped_duplicates += 1
            print(
                f"skipping game {full_state.game_id} from table {pending.table.table_id}: "
                "claimed by a fleet peer or coordination unavailable",
                flush=True,
            )
            self.pending = None
            self._release_table_claim()
            self.next_join_allowed_monotonic = time.monotonic() + self.args.join_delay
            self._schedule_leave(pending.table.table_id)
            return

        game_start = pending.game_start or self._pump.game_starts.get(full_state.game_id)
        player_ids = self._pump.parser.player_ids
        player_names_by_id = dict(self._pump.parser.player_names_by_id)
        player_names = (
            game_start.players
            if game_start is not None
            else tuple(player_names_by_id.get(player_id, str(player_id)) for player_id in player_ids)
        )
        kingdom_card_names = game_start.kingdom if game_start is not None else ()
        capture_time_utc = _utc_now()
        raw_path = self.raw_root / f"{full_state.game_id}.jsonl.gz"
        manifest_path = self.raw_root / f"{full_state.game_id}.manifest.json"
        if self.coordination is not None and (raw_path.exists() or manifest_path.exists()):
            self._remember_rejected_table(pending.table)
            self.stats.skipped_duplicates += 1
            print(
                f"skipping game {full_state.game_id} from table {pending.table.table_id}: "
                "shared output already exists",
                flush=True,
            )
            self.pending = None
            self._release_capture_claims()
            self.next_join_allowed_monotonic = time.monotonic() + self.args.join_delay
            self._schedule_leave(pending.table.table_id)
            return
        manifest: dict[str, object] = {
            "schema_version": 1,
            "capture_time_utc": capture_time_utc,
            "table_id": pending.table.table_id,
            "game_id": full_state.game_id,
            "kingdom_card_names": list(kingdom_card_names),
            "card_type_names": list(card_type_names),
            "player_names": list(player_names),
            "outcome": None,
            "capture_reached_game_finished": False,
            "capture_status": "incomplete",
            "incomplete_reason": "capture in progress",
        }
        if self.args.no_base_filter:
            # ``card_type_names`` is the fullGameState-observed kingdom/card
            # population even if a GameStart record was unavailable.
            manifest["capture_filter"] = "everything"
            manifest["observed_kingdom_card_names"] = list(
                kingdom_card_names or card_type_names
            )
        # Mark before writing the buffered stream.  An interruption at any
        # later point leaves an explicit incomplete manifest and blocks a
        # duplicate game id on restart.
        if not self._reserve_capture_manifest(manifest_path, manifest):
            self._remember_rejected_table(pending.table)
            self.stats.skipped_duplicates += 1
            print(
                f"skipping game {full_state.game_id} from table {pending.table.table_id}: "
                "another collector reserved its manifest",
                flush=True,
            )
            self.pending = None
            self._release_capture_claims()
            self.next_join_allowed_monotonic = time.monotonic() + self.args.join_delay
            self._schedule_leave(pending.table.table_id)
            return
        self.seen_game_ids.add(full_state.game_id)
        active = ActiveCapture(
            table=pending.table,
            game_id=full_state.game_id,
            capture_time_utc=capture_time_utc,
            kingdom_card_names=tuple(kingdom_card_names),
            card_type_names=card_type_names,
            player_names=tuple(player_names),
            player_names_by_id=player_names_by_id,
            raw_path=raw_path,
            raw_staging_path=self._staging_raw_path(full_state.game_id, raw_path),
            manifest_path=manifest_path,
            manifest=manifest,
        )
        self.pending = None
        self.claimed_table = None
        self.active = active
        for record in pending.raw_records:
            if _is_sensitive_record(record, game_socket=self._session.game_socket):
                self.recovery_table_ids.add(active.table.table_id)
                self._mark_active_incomplete("browser reauthentication observed in join buffer")
                raise RecoverPage("browser reauthenticated while attaching base game")
            _append_gzip_jsonl_record(active.raw_staging_path, record)
        self.stats.captured_started += 1
        self._mark_session_healthy()
        capture_kind = "base game" if not foreign else "game (non-base filter disabled)"
        print(
            f"keeping {capture_kind} {full_state.game_id} from table {pending.table.table_id}: "
            f"{len(card_type_names)} card types; awaiting gameFinished",
            flush=True,
        )

    def _schedule_leave(self, table_id: int) -> None:
        """Issue LEAVE_TABLE and gate every later join on its completion."""
        if self.player_id is None:
            return
        if table_id in self.leave_tasks:
            return
        self.leave_tasks[table_id] = asyncio.create_task(
            self._leave_table(table_id),
            name=f"leave-table-{table_id}",
        )

    async def _service_leave_tasks(self) -> None:
        """Surface cleanup failures rather than silently orphaning a table join."""
        for table_id, task in tuple(self.leave_tasks.items()):
            if not task.done():
                continue
            del self.leave_tasks[table_id]
            try:
                task.result()
            except RecoverPage:
                self.recovery_table_ids.add(table_id)
                raise
            except Exception as error:
                self.recovery_table_ids.add(table_id)
                raise RecoverPage(
                    f"LEAVE_TABLE task failed for {table_id}: {_failure_reason(error)}"
                ) from error

    async def _leave_table(self, table_id: int) -> None:
        try:
            await self._session.send_frame(
                LEAVE_TABLE,
                Writer().u64(table_id).s32(self.player_id or 0).build(),
            )
            print(f"left table {table_id}", flush=True)
        except Exception as error:
            # Refusing to issue further joins after a leave failure is safer
            # than risking an accidental lingering attachment.  A live page
            # gets an in-place reload first; only an unusable page reaches a
            # hard browser relaunch.
            self.recovery_table_ids.add(table_id)
            raise RecoverPage(
                f"LEAVE_TABLE failed for {table_id}: {_failure_reason(error)}"
            ) from error

    def _complete_active_capture(self, frame: DecodedFrame) -> None:
        active = self.active
        assert active is not None
        try:
            result = _decode_game_result_details(
                payload=frame.payload,
                table_id=active.table.table_id,
                game_id=active.game_id,
                player_names_by_id=active.player_names_by_id,
            )
            outcome: dict[str, object] = {"kind": "gameFinished", "game_result": result}
        except Exception as error:
            outcome = {
                "kind": "gameFinished",
                "game_result_decode_error": _failure_reason(error),
            }
        if not self._publish_active_raw_capture(active):
            active.manifest.update(
                {
                    "capture_reached_game_finished": False,
                    "capture_status": "incomplete",
                    "incomplete_reason": "raw output target already existed at publication",
                    "stopped_at_utc": _utc_now(),
                }
            )
            _atomic_json_write(active.manifest_path, active.manifest)
            self.stats.captures_incomplete += 1
            print(
                f"marked game {active.game_id} incomplete: raw output target already existed",
                flush=True,
            )
            table_id = active.table.table_id
            self.active = None
            self._release_capture_claims()
            self.next_join_allowed_monotonic = time.monotonic() + self.args.join_delay
            self._schedule_leave(table_id)
            self.snapshot_received_monotonic = None
            self._send_lobby_request(force=True)
            return
        active.manifest.update(
            {
                "outcome": outcome,
                "capture_reached_game_finished": True,
                "capture_status": "complete",
                "incomplete_reason": None,
                "completed_at_utc": _utc_now(),
            }
        )
        _atomic_json_write(active.manifest_path, active.manifest)
        self.stats.captured_completed += 1
        self._mark_session_healthy()
        capture_kind = "game" if active.manifest.get("capture_filter") == "everything" else "base game"
        print(
            f"completed {capture_kind} {active.game_id}; "
            f"games/hour={self.stats.games_per_hour():.2f}",
            flush=True,
        )
        table_id = active.table.table_id
        self.active = None
        self._release_capture_claims()
        self.next_join_allowed_monotonic = time.monotonic() + self.args.join_delay
        self._schedule_leave(table_id)
        # Never choose another row from a snapshot that pre-dates a complete
        # game wait.  A forced new poll will gate the next join.
        self.snapshot_received_monotonic = None
        self._send_lobby_request(force=True)

    def _mark_active_incomplete(self, reason: str) -> None:
        active = self.active
        if active is None:
            return
        if not self._publish_active_raw_capture(active, allow_missing_staging=True):
            reason = f"{reason}; raw output target already existed at publication"
        active.manifest.update(
            {
                "capture_reached_game_finished": False,
                "capture_status": "incomplete",
                "incomplete_reason": reason,
                "stopped_at_utc": _utc_now(),
            }
        )
        _atomic_json_write(active.manifest_path, active.manifest)
        self.stats.captures_incomplete += 1
        print(f"marked game {active.game_id} incomplete: {reason}", flush=True)
        self.active = None
        self._release_capture_claims()

    def _snapshot_is_fresh(self, now: float) -> bool:
        return (
            self.snapshot is not None
            and self.snapshot_received_monotonic is not None
            and now - self.snapshot_received_monotonic <= self.args.max_snapshot_age
        )

    async def _begin_next_join_if_ready(self, now: float, stop_requested: asyncio.Event) -> None:
        if self.pending is not None or self.active is not None:
            return
        if self.leave_tasks:
            # The task begins immediately, but do not overlap its protocol
            # action with another join on this intentionally single-table socket.
            return
        if now < self.next_join_allowed_monotonic:
            return
        if self.claimed_table is not None:
            if not self._snapshot_is_fresh(now):
                # Do not hold a table claim across an old lobby view while a
                # rate token is unavailable; table ids can be recycled.
                self._release_table_claim()
                self._send_lobby_request(force=True)
                return
            table = self.claimed_table
        else:
            if not self._snapshot_is_fresh(now):
                self._send_lobby_request(force=True)
                return
            assert self.snapshot is not None
            now_epoch_ms = int(time.time() * 1_000)
            table = _select_probe_table(
                self.snapshot,
                now_epoch_ms=now_epoch_ms,
                target_min_age_seconds=self.args.target_min_age_seconds,
                target_max_age_seconds=self.args.target_max_age_seconds,
                excluded_table_keys=set(self.attempted_table_keys) | set(self.rejected_table_keys),
                rng=self.selection_rng,
            )
            if table is None:
                # All rows in this fresh snapshot have already been
                # seen/rejected.  Keep normal poll cadence rather than making
                # a tight loop of identical requests.
                return
            self._remember_table_attempt(table)
            if not self._acquire_table_claim(table, now=now):
                return

        if self.coordination is not None:
            try:
                token = self.coordination.take_join_token(
                    rate_per_minute=self.args.fleet_join_rate_per_minute,
                )
            except CoordinationUnavailable as error:
                self._release_table_claim()
                self.next_join_allowed_monotonic = now + 1.0
                print(
                    f"fleet join limiter unavailable; deferring join: {_failure_reason(error)}",
                    flush=True,
                )
                return
            if not token.permitted:
                self.stats.rate_limit_waits += 1
                delay = max(self.args.join_delay, token.retry_after_seconds)
                self.next_join_allowed_monotonic = now + delay
                if now - self._last_rate_limit_notice_monotonic >= 10.0:
                    print(
                        f"fleet JOIN_TABLE rate limit; holding table {table.table_id} "
                        f"for {delay:.1f}s",
                        flush=True,
                    )
                    self._last_rate_limit_notice_monotonic = now
                return

        now_epoch_ms = int(time.time() * 1_000)
        # The claim has served its pre-join selection role, but its lease stays
        # owned through pending/full-state/capture cleanup.
        self.claimed_table = None
        self.stats.joins_attempted += 1
        self.pending = PendingJoin(
            table=table,
            started_monotonic=now,
            deadline_monotonic=now + self.args.join_timeout,
        )
        self.next_join_allowed_monotonic = now + self.args.join_delay
        age = _table_age_seconds(table, now_epoch_ms=now_epoch_ms)
        age_text = "unknown" if age is None else f"{age / 60.0:.1f}m"
        print(
            f"joining table {table.table_id} (age={age_text}, target="
            f"{self.args.target_min_age_seconds / 60.0:.0f}-"
            f"{self.args.target_max_age_seconds / 60.0:.0f}m, "
            f"uniform-band sample, join={self.stats.joins_attempted}, "
            f"skipped-non-base={self.stats.skipped_non_base})",
            flush=True,
        )
        try:
            await self._session.send_frame(
                JOIN_TABLE,
                Writer().u64(table.table_id).boolean(False).build(),
            )
        except Exception as error:
            self.pending = None
            self._release_table_claim()
            raise RecoverPage(
                f"JOIN_TABLE failed for {table.table_id}: {_failure_reason(error)}"
            ) from error
        # The delay applies after every join setup, while the frame loop still
        # receives the server's FullState immediately.
        await self._sleep_until_or_stop(now + self.args.join_delay, stop_requested)

    async def _sleep_until_or_stop(self, deadline: float, stop_requested: asyncio.Event) -> None:
        while True:
            if stop_requested.is_set():
                raise StopRequested()
            if not self._page_is_usable():
                raise RestartBrowser("browser page closed during join delay")
            self._refresh_coordination_claims(time.monotonic())
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            notice = await self._pump.next_notice(min(0.25, remaining))
            if notice is not None:
                self._handle_notice(notice)

    async def _main_loop(self, stop_requested: asyncio.Event) -> None:
        next_heartbeat = time.monotonic() + self.args.heartbeat_interval
        end_monotonic = (
            None
            if self.args.run_seconds == 0
            else time.monotonic() + self.args.run_seconds
        )
        while True:
            if stop_requested.is_set():
                raise StopRequested()
            if not self._page_is_usable():
                raise RestartBrowser("browser page closed during collection")
            now = time.monotonic()
            if end_monotonic is not None and now >= end_monotonic:
                print("configured run duration reached", flush=True)
                return
            await self._service_leave_tasks()
            self._refresh_coordination_claims(now)
            await self._service_poll(now)
            await self._begin_next_join_if_ready(now, stop_requested)
            now = time.monotonic()
            if self.pending is not None and now >= self.pending.deadline_monotonic:
                timed_out_table = self.pending.table
                table_id = timed_out_table.table_id
                self.pending = None
                self._release_table_claim()
                self._remember_rejected_table(timed_out_table)
                self.stats.joins_timed_out_as_over += 1
                self.next_join_allowed_monotonic = time.monotonic() + self.args.join_delay
                print(
                    f"table {table_id} timed out waiting for fullGameState; treating as already over",
                    flush=True,
                )
                await self._leave_table(table_id)
            if now >= next_heartbeat:
                self._heartbeat()
                next_heartbeat += self.args.heartbeat_interval

            deadlines = [next_heartbeat, self.next_poll_monotonic]
            if self.coordination is not None and (
                self.table_claim is not None
                or self.game_claim is not None
                or self.snapshot_refresh_claim is not None
            ):
                deadlines.append(self.next_claim_refresh_monotonic)
            if self.poll_inflight:
                deadlines.append(self.next_poll_retry_monotonic)
            if self.pending is not None:
                deadlines.append(self.pending.deadline_monotonic)
            if end_monotonic is not None:
                deadlines.append(end_monotonic)
            timeout = max(0.05, min(1.0, min(deadlines) - time.monotonic()))
            notice = await self._pump.next_notice(timeout)
            if notice is not None:
                self._handle_notice(notice)

    def _heartbeat(self) -> None:
        # The page owns the protocol-level CLIENT_HEARTBEAT (type 44) and its
        # pong deadline.  This loop wakes at least once per second and keeps
        # lobby polling active, so it never blocks that browser-managed
        # keepalive; type 44 remains deliberately non-sensitive above.
        games_live = 1 if self.active is not None else 0
        probe_rate = self.stats.base_only_probe_rate()
        probe_rate_text = "n/a" if probe_rate is None else f"{probe_rate:.1%}"
        message = (
            "heartbeat: "
            f"games-live={games_live} captured={self.stats.captured_completed} "
            f"started={self.stats.captured_started} "
            f"probed={self.stats.tables_probed} base-probe-rate={probe_rate_text} "
            f"skipped-non-base={self.stats.skipped_non_base} "
            f"join-timeouts={self.stats.joins_timed_out_as_over} "
            f"failed={self.stats.failed} in-page-recoveries={self.stats.in_page_recoveries} "
            f"hard-relaunches={self.stats.hard_browser_relaunches} "
            f"credentialed-relogins={self.stats.credentialed_relogins} "
            f"games/hour={self.stats.games_per_hour():.2f}"
        )
        if self.coordination is not None:
            claim_rate = self.stats.claim_contention_rate()
            claim_rate_text = "n/a" if claim_rate is None else f"{claim_rate:.1%}"
            message += f" claim-contention={claim_rate_text} rate-limit-waits={self.stats.rate_limit_waits}"
        print(message, flush=True)
        self._publish_coordination_heartbeat(status="running")


def _summary_document(
    *,
    stats: CollectorStats,
    outcome: str,
    abort_reason: str | None,
    seen_game_ids: int,
) -> dict[str, object]:
    document: dict[str, object] = {
        "started_at_utc": stats.started_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "finished_at_utc": _utc_now(),
        "outcome": outcome,
        "abort_reason": abort_reason,
        "session_expired": outcome == "session_expired",
        "architecture": "single-table-long-lived-browser-target-age-band",
        "maximum_active_spectated_games": 1,
        "snapshots_received": stats.snapshots_received,
        "last_snapshot_tables": stats.last_snapshot_tables,
        "last_snapshot_eligible_human_two_player": stats.last_snapshot_candidates,
        "joins_attempted": stats.joins_attempted,
        "joins_timed_out_treated_as_over": stats.joins_timed_out_as_over,
        "tables_probed": stats.tables_probed,
        "base_only_tables_probed": stats.base_only_tables_probed,
        "base_only_rate_among_probed_tables": stats.base_only_probe_rate(),
        "skipped_non_base": stats.skipped_non_base,
        "skipped_duplicate_game_id": stats.skipped_duplicates,
        "base_captures_started": stats.captured_started,
        "base_captures_completed": stats.captured_completed,
        "captures_marked_incomplete": stats.captures_incomplete,
        "failed": stats.failed,
        "in_page_recoveries": stats.in_page_recoveries,
        "hard_browser_relaunches": stats.hard_browser_relaunches,
        # Retained for consumers of earlier summary schemas.  It now means
        # only a real Playwright context relaunch, never a normal page reload.
        "browser_restarts": stats.browser_restarts,
        "credentialed_self_heal_available": stats.credentialed_self_heal_available,
        "credentialed_relogins": stats.credentialed_relogins,
        "known_game_ids_on_disk": seen_game_ids,
        "achieved_games_per_hour": stats.games_per_hour(),
    }
    if stats.coordination_enabled:
        document.update(
            {
                "coordination_enabled": True,
                "account_id": stats.account_id,
                "table_claim_attempts": stats.table_claim_attempts,
                "table_claim_contentions": stats.table_claim_contentions,
                "game_claim_attempts": stats.game_claim_attempts,
                "game_claim_contentions": stats.game_claim_contentions,
                "claim_contention_rate": stats.claim_contention_rate(),
                "stale_claims_reclaimed": stats.stale_claims_reclaimed,
                "claim_refreshes": stats.claim_refreshes,
                "claim_refresh_failures": stats.claim_refresh_failures,
                "shared_snapshot_hits": stats.shared_snapshot_hits,
                "shared_snapshot_refreshes": stats.shared_snapshot_refreshes,
                "snapshot_cache_degradations": stats.snapshot_cache_degradations,
                "fleet_rate_limit_waits": stats.rate_limit_waits,
            }
        )
    return document


def _write_run_summary(raw_root: Path, document: dict[str, object]) -> Path:
    runs_root = raw_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = runs_root / f"collector-summary-{timestamp}.json"
    _atomic_json_write(destination, document)
    return destination


async def _run(args: argparse.Namespace) -> int:
    stats = CollectorStats(started_at=datetime.now(timezone.utc))
    collector: DominionCollector | None = None
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_requested.set)
            installed_signals.append(signum)
        except (NotImplementedError, RuntimeError):
            pass

    outcome = "completed"
    abort_reason: str | None = None
    exit_code = 0
    consecutive_hard_failures = 0
    try:
        collector = DominionCollector(args, stats)
        while True:
            try:
                await collector.run_browser_session(stop_requested)
                break
            except StopRequested:
                outcome = "interrupted"
                abort_reason = "interrupted by user signal"
                print("interrupt received; active capture was marked incomplete if needed", flush=True)
                break
            except SessionExpired as error:
                outcome = "session_expired"
                abort_reason = str(error)
                exit_code = SESSION_EXPIRED_EXIT_CODE
                print(f"aborting: {abort_reason}", flush=True)
                break
            except CredentialedReloginRateLimitExceeded as error:
                outcome = "credentialed_relogin_rate_limited"
                abort_reason = str(error)
                exit_code = 1
                print(f"aborting: {abort_reason}", flush=True)
                break
            except SustainedFailure as error:
                outcome = "aborted"
                abort_reason = str(error)
                exit_code = 1
                print(f"aborting: {abort_reason}", flush=True)
                break
            except RestartBrowser as error:
                if collector.last_session_reached_healthy_state:
                    # Authenticated lobby operation and every base capture
                    # reset the hard-relaunch streak.  Isolated transport
                    # blips over a healthy multi-hour run therefore cannot
                    # exhaust the safety limit.
                    consecutive_hard_failures = 0
                stats.failed += 1
                consecutive_hard_failures += 1
                if consecutive_hard_failures >= args.max_consecutive_failures:
                    outcome = "aborted"
                    exit_code = 1
                    abort_reason = (
                        f"{consecutive_hard_failures} consecutive hard browser failures "
                        f"reached the safety limit; last failure: {error}"
                    )
                    print(f"aborting: {abort_reason}", flush=True)
                    break
                stats.browser_restarts += 1
                stats.hard_browser_relaunches += 1
                delay = min(
                    args.max_backoff,
                    args.retry_base_seconds * (2 ** (consecutive_hard_failures - 1)),
                )
                print(
                    f"hard browser failure: {error}; relaunch {stats.hard_browser_relaunches} "
                    f"after {delay:.1f}s backoff",
                    flush=True,
                )
                try:
                    await asyncio.wait_for(stop_requested.wait(), timeout=delay)
                except TimeoutError:
                    continue
                outcome = "interrupted"
                abort_reason = "interrupted by user signal during backoff"
                break
            except Exception as error:
                outcome = "aborted"
                abort_reason = f"unhandled collector failure: {_failure_reason(error)}"
                exit_code = 1
                print(f"aborting: {abort_reason}", flush=True)
                break
    except Exception as error:
        outcome = "aborted"
        abort_reason = f"collector startup failure: {_failure_reason(error)}"
        exit_code = 1
        print(f"aborting: {abort_reason}", flush=True)
    except KeyboardInterrupt:
        outcome = "interrupted"
        abort_reason = "interrupted by user signal"
        print("interrupt received; writing run summary", flush=True)
    finally:
        for signum in installed_signals:
            try:
                loop.remove_signal_handler(signum)
            except (NotImplementedError, RuntimeError):
                pass
        if collector is not None:
            collector._publish_coordination_heartbeat(status=outcome)
        summary = _summary_document(
            stats=stats,
            outcome=outcome,
            abort_reason=abort_reason,
            seen_game_ids=len(collector.seen_game_ids) if collector is not None else 0,
        )
        summary_path = _write_run_summary(args.raw_root, summary)
        print(
            f"done: outcome={outcome}; completed={stats.captured_completed}; "
            f"skipped-non-base={stats.skipped_non_base}; failed={stats.failed}; "
            f"games/hour={stats.games_per_hour():.2f}",
            flush=True,
        )
        print(f"summary: {summary_path.resolve()}", flush=True)
    return exit_code


def _run_self_test() -> None:
    """Offline smoke test for durable raw framing and protocol fixtures."""
    from tempfile import TemporaryDirectory

    sample_record = {
        "ts": 1,
        "sock": 3,
        "dir": "in",
        "kind": "binary",
        "url": "wss://example.invalid",
        "data": "AA==",
        "b64": True,
    }
    with TemporaryDirectory() as temporary_directory:
        destination = Path(temporary_directory) / "123.jsonl.gz"
        _append_gzip_jsonl_record(destination, sample_record)
        _append_gzip_jsonl_record(destination, sample_record)
        with gzip.open(destination, "rt", encoding="utf-8") as source:
            records = [json.loads(line) for line in source]
    if records != [sample_record, sample_record]:
        raise AssertionError("concatenated append-safe gzip members did not round-trip")
    _run_session_handshake_self_test()
    _run_archived_handshake_fixture_self_test()
    # Exercise the same tables decoder against the saved live spectator fixture.
    fixture = Path("data/dominion_games/recon/captures/20260731T220433.671341Z/frames.jsonl")
    expected_counts = (340, 347, 348)
    counts: list[int] = []
    with fixture.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("kind") != "binary" or record.get("dir") != "in" or not record.get("b64"):
                continue
            frame = decode_record_binary(record)
            if frame is not None and frame.msg_type == TABLES_OVERVIEW:
                counts.append(len(decode_tables_overview(frame.payload)))
    if tuple(counts) != expected_counts:
        raise AssertionError(f"expected fixture table counts {expected_counts}, got {tuple(counts)}")
    # The recon capture deliberately ended before the three games finished;
    # use the existing canonical player-seat capture to exercise all rich
    # GameResult fields without live site contact.  It is optional so a lean
    # VPS shipment can still use --self-test with only recon fixtures.
    canonical = Path("arena-recordings/20260724T142103.096991Z/frames.jsonl")
    decoded_results = 0
    if canonical.is_file():
        parser = ArenaParser()
        table_id: int | None = None
        with canonical.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if record.get("kind") != "binary" or not record.get("b64"):
                    continue
                frame = decode_record_binary(record)
                if frame is None:
                    continue
                if frame.direction is Direction.INBOUND and frame.msg_type == 10:
                    table_id = Reader(frame.payload).u64()
                parser.parse_frame(frame)
                if frame.direction is Direction.INBOUND and frame.msg_type == GAME_FINISHED:
                    if table_id is None or parser.game_id is None:
                        raise AssertionError("GameFinished fixture arrived without current table/game ids")
                    result = _decode_game_result_details(
                        payload=frame.payload,
                        table_id=table_id,
                        game_id=parser.game_id,
                        player_names_by_id=parser.player_names_by_id,
                    )
                    if len(result["players"]) != 2:
                        raise AssertionError("expected two final player results")
                    decoded_results += 1
        if decoded_results != 3:
            raise AssertionError(f"expected three canonical GameResults, got {decoded_results}")
    print(
        "self-test passed: gzip append members, session-expiry/self-heal handshake detection, "
        "failed/healthy archived handshake fixtures, credentialed re-login rate limiting, "
        "three saved lobby snapshots, "
        f"and {decoded_results} canonical GameResults",
        flush=True,
    )


def _synthetic_binary_record(
    *,
    direction: str,
    msg_type: int,
    socket: int = 3,
) -> dict[str, Any]:
    """Build a payload-free hook record for offline message-type tests."""
    prefix = b"\x00\x00\x00\x00" if direction == "in" else b""
    raw = prefix + msg_type.to_bytes(4, "big")
    return {
        "ts": 0,
        "sock": socket,
        "dir": direction,
        "kind": "binary",
        "url": "wss://example.invalid",
        "data": base64.b64encode(raw).decode("ascii"),
        "b64": True,
    }


def _replay_archived_handshake_fixture(
    path: Path,
    *,
    game_socket: int = 3,
) -> tuple[SessionHandshakeTracker, float]:
    """Replay an archived hook stream in order using only safe type metadata.

    Auth frames are deliberately redacted in diagnostic archives.  Their
    ``msg_type`` metadata is sufficient for handshake state tests, while the
    original payload never needs to be restored or inspected.
    """
    tracker = SessionHandshakeTracker(game_socket=game_socket)
    last_now = 0.0
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            timestamp = record.get("ts")
            if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
                last_now = float(timestamp) / 1_000.0
            tracker.observe(record, now=last_now)
    return tracker, last_now


def _run_session_handshake_self_test() -> None:
    """Exercise valid and expired persistent-profile handshakes without Chromium."""
    opened = {
        "ts": 0,
        "sock": 3,
        "dir": "in",
        "kind": "open",
        "url": "wss://example.invalid",
        "data": "",
        "b64": False,
    }
    healthy = SessionHandshakeTracker(game_socket=3)
    healthy_sequence = (
        (opened, 0.0),
        (_synthetic_binary_record(direction="out", msg_type=REQUEST_SERVER_STATE, socket=1), 0.1),
        (_synthetic_binary_record(direction="out", msg_type=LOGIN_WITH_SESSION), 0.2),
        (_synthetic_binary_record(direction="in", msg_type=LOGIN_SUCCESS), 0.3),
    )
    for record, now in healthy_sequence:
        healthy.observe(record, now=now)
    if healthy.is_parked_at_login_form(
        now=DEFAULT_SESSION_LOGIN_GRACE_SECONDS + 1.0,
        grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
    ):
        raise AssertionError("healthy LOGIN_WITH_SESSION/loginSuccess sequence was marked expired")
    if _parked_login_action(
        healthy,
        credentials=DgamesCredentials(username="test-user", password="test-password"),
        now=DEFAULT_SESSION_LOGIN_GRACE_SECONDS + 1.0,
        grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
    ) is not ParkedLoginAction.WAIT:
        raise AssertionError("healthy login sequence incorrectly requested self-heal")

    dead = SessionHandshakeTracker(game_socket=3)
    dead_sequence = (
        (opened, 0.0),
        # REQUEST_SERVER_STATE on a probe socket plus the game socket opening
        # is the captured dead-profile signature.  There is intentionally no
        # outbound LOGIN_WITH_SESSION (24) and no inbound loginSuccess (2).
        (_synthetic_binary_record(direction="out", msg_type=REQUEST_SERVER_STATE, socket=1), 0.1),
    )
    for record, now in dead_sequence:
        dead.observe(record, now=now)
    dead_deadline = 0.1 + DEFAULT_SESSION_LOGIN_GRACE_SECONDS
    if dead.is_parked_at_login_form(
        now=dead_deadline - 0.01,
        grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
    ):
        raise AssertionError("dead session was reported before its short login grace period")
    if not dead.is_parked_at_login_form(
        now=dead_deadline + 0.01,
        grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
    ):
        raise AssertionError("dead session did not report expiry after its short login grace period")
    if _parked_login_action(
        dead,
        credentials=None,
        now=dead_deadline + 0.01,
        grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
    ) is not ParkedLoginAction.SESSION_EXPIRED:
        raise AssertionError("dead session without credentials did not fast-fail")
    if _parked_login_action(
        dead,
        credentials=DgamesCredentials(username="test-user", password="test-password"),
        now=dead_deadline + 0.01,
        grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
    ) is not ParkedLoginAction.CREDENTIALED_RELOGIN:
        raise AssertionError("dead session with credentials did not request credentialed re-login")

    limiter = CredentialedReloginRateLimiter(
        max_attempts_per_hour=2,
        minimum_backoff_seconds=10.0,
    )
    if (slot := limiter.check(now=0.0)).permitted is not True or slot.backoff_seconds != 0.0:
        raise AssertionError("first credentialed re-login slot was unexpectedly unavailable")
    limiter.record(now=0.0)
    if (slot := limiter.check(now=5.0)).backoff_seconds != 5.0:
        raise AssertionError("credentialed re-login backoff was not enforced")
    limiter.record(now=10.0)
    if (slot := limiter.check(now=11.0)).permitted or slot.attempts_in_window != 2:
        raise AssertionError("credentialed re-login hourly cap was not enforced")
    if not limiter.check(now=CREDENTIALED_RELOGIN_WINDOW_SECONDS).permitted:
        raise AssertionError("credentialed re-login cap did not expire after one hour")

    if _is_sensitive_record(
        _synthetic_binary_record(direction="out", msg_type=CLIENT_HEARTBEAT)
    ):
        raise AssertionError("client heartbeat type 44 must remain non-sensitive")
    if _is_sensitive_record(
        _synthetic_binary_record(direction="in", msg_type=LOGIN_SUCCESS, socket=1)
    ):
        raise AssertionError("probe server-state ordinal 2 was misclassified as loginSuccess")
    for sensitive_type in (
        LOGIN,
        13,
        14,
        15,
        16,
        19,
        LOGIN_WITH_SESSION,
        REMOVE_ACTIVE_SESSIONS,
        REQUEST_SERVER_STATE,
        46,
    ):
        if not _is_sensitive_record(
            _synthetic_binary_record(direction="out", msg_type=sensitive_type)
        ):
            raise AssertionError(f"sensitive outbound type {sensitive_type} was not redacted")


def _run_archived_handshake_fixture_self_test() -> None:
    """Keep the real parked-login and successful-login captures as regression gates."""
    failed, failed_now = _replay_archived_handshake_fixture(FAILED_HANDSHAKE_FIXTURE)
    if failed.game_socket_generations != 5:
        raise AssertionError(
            "failed handshake fixture did not preserve all five socket-3 generations"
        )
    if failed.login_with_session_at is not None or failed.login_success_at is not None:
        raise AssertionError("probe-socket state replies were misclassified as game login")
    failed_check_at = failed_now + DEFAULT_SESSION_LOGIN_GRACE_SECONDS + 0.01
    if not failed.is_parked_at_login_form(
        now=failed_check_at,
        grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
    ):
        raise AssertionError("failed handshake fixture did not trigger parked-login detection")
    if _parked_login_action(
        failed,
        credentials=None,
        now=failed_check_at,
        grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
    ) is not ParkedLoginAction.SESSION_EXPIRED:
        raise AssertionError("failed handshake fixture did not fast-fail without credentials")
    if _parked_login_action(
        failed,
        credentials=DgamesCredentials(username="fixture-user", password="fixture-password"),
        now=failed_check_at,
        grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
    ) is not ParkedLoginAction.CREDENTIALED_RELOGIN:
        raise AssertionError("failed handshake fixture did not select credentialed self-heal")

    healthy, healthy_now = _replay_archived_handshake_fixture(HEALTHY_HANDSHAKE_FIXTURE)
    if healthy.login_with_session_at is None or healthy.login_success_at is None:
        raise AssertionError("healthy handshake fixture did not contain game-socket login evidence")
    healthy_check_at = healthy_now + DEFAULT_SESSION_LOGIN_GRACE_SECONDS + 0.01
    if healthy.is_parked_at_login_form(
        now=healthy_check_at,
        grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
    ):
        raise AssertionError("healthy handshake fixture was misdetected as parked")
    if _parked_login_action(
        healthy,
        credentials=None,
        now=healthy_check_at,
        grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
    ) is not ParkedLoginAction.WAIT:
        raise AssertionError("healthy handshake fixture requested unnecessary recovery")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--account-id",
        default=None,
        help="fleet account suffix (uses DGAMES_USER_<id>/DGAMES_PASS_<id>)",
    )
    parser.add_argument(
        "--coordination-dir",
        type=Path,
        default=None,
        help=(
            "opt in to shared fleet coordination under this directory "
            f"(fleet default: {DEFAULT_COORDINATION_DIR})"
        ),
    )
    parser.add_argument(
        "--claim-ttl",
        type=float,
        default=DEFAULT_CLAIM_TTL_SECONDS,
        help="seconds before an unrefreshed table/game claim can be reclaimed",
    )
    parser.add_argument(
        "--claim-refresh-interval",
        type=float,
        default=DEFAULT_CLAIM_REFRESH_INTERVAL_SECONDS,
        help="seconds between live table/game claim mtime refreshes",
    )
    parser.add_argument(
        "--snapshot-cache-ttl",
        type=float,
        default=DEFAULT_SNAPSHOT_CACHE_TTL_SECONDS,
        help="maximum age of a peer's shared lobby snapshot",
    )
    parser.add_argument(
        "--snapshot-refresh-ttl",
        type=float,
        default=DEFAULT_SNAPSHOT_REFRESH_TTL_SECONDS,
        help="stale TTL for the shared lobby-refresh mutex",
    )
    parser.add_argument(
        "--fleet-join-rate-per-minute",
        type=float,
        default=DEFAULT_FLEET_JOIN_RATE_PER_MINUTE,
        help="global shared JOIN_TABLE cap when coordination is enabled",
    )
    parser.add_argument("--card-map", type=Path, default=DEFAULT_CARD_MAP)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument(
        "--max-snapshot-age",
        type=float,
        default=30.0,
        help="never join a row older than this many seconds",
    )
    parser.add_argument("--join-timeout", type=float, default=12.0)
    parser.add_argument("--join-delay", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument(
        "--session-login-grace",
        type=float,
        default=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
        help="fail as session_expired after this many seconds of the parked-login wire signature",
    )
    parser.add_argument(
        "--credentialed-login-timeout",
        type=float,
        default=DEFAULT_CREDENTIALED_LOGIN_TIMEOUT_SECONDS,
        help="wait this long for one page-driven credentialed login attempt",
    )
    parser.add_argument(
        "--credentialed-relogin-backoff",
        type=float,
        default=DEFAULT_CREDENTIALED_RELOGIN_BACKOFF_SECONDS,
        help="minimum delay between credentialed self-heal attempts",
    )
    parser.add_argument(
        "--max-credentialed-relogins-per-hour",
        type=int,
        default=DEFAULT_MAX_CREDENTIALED_RELOGINS_PER_HOUR,
        help="maximum page-driven credentialed logins in any rolling hour",
    )
    parser.add_argument(
        "--target-min-age-seconds",
        type=float,
        default=DEFAULT_TARGET_MIN_AGE_SECONDS,
        help="uniformly sample running tables at least this old (default: 4 minutes)",
    )
    parser.add_argument(
        "--target-max-age-seconds",
        type=float,
        default=DEFAULT_TARGET_MAX_AGE_SECONDS,
        help="uniformly sample running tables no older than this (default: 12 minutes)",
    )
    parser.add_argument("--heartbeat-interval", type=float, default=30.0)
    parser.add_argument("--max-poll-retries", type=int, default=5)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--max-backoff", type=float, default=60.0)
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=0.0,
        help="0 means run indefinitely; use 900 for a 15-minute verification",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="show Chromium instead of the default headless browser",
    )
    parser.add_argument(
        "--no-base-filter",
        action="store_true",
        help="capture every observed kingdom while retaining kingdom details in the manifest",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run offline gzip and saved-capture checks without launching a browser",
    )
    args = parser.parse_args(argv)
    for name in (
        "poll_interval",
        "max_snapshot_age",
        "join_timeout",
        "startup_timeout",
        "session_login_grace",
        "credentialed_login_timeout",
        "credentialed_relogin_backoff",
        "target_max_age_seconds",
        "heartbeat_interval",
        "retry_base_seconds",
        "max_backoff",
        "claim_ttl",
        "claim_refresh_interval",
        "snapshot_cache_ttl",
        "snapshot_refresh_ttl",
        "fleet_join_rate_per_minute",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.join_delay < MIN_JOIN_DELAY_SECONDS:
        parser.error(f"--join-delay must be at least {MIN_JOIN_DELAY_SECONDS:.1f} seconds")
    if args.target_min_age_seconds < 0:
        parser.error("--target-min-age-seconds cannot be negative")
    if args.target_min_age_seconds >= args.target_max_age_seconds:
        parser.error("--target-min-age-seconds must be less than --target-max-age-seconds")
    if args.max_poll_retries < 1:
        parser.error("--max-poll-retries must be at least one")
    if args.max_consecutive_failures < 1:
        parser.error("--max-consecutive-failures must be at least one")
    if args.max_credentialed_relogins_per_hour < 1:
        parser.error("--max-credentialed-relogins-per-hour must be at least one")
    if args.run_seconds < 0:
        parser.error("--run-seconds cannot be negative")
    if args.account_id is not None:
        try:
            _credential_environment_keys(args.account_id)
        except ValueError as error:
            parser.error(str(error))
    if args.coordination_dir is not None:
        if args.account_id is None:
            parser.error("--coordination-dir requires --account-id for accountable claims")
        if args.claim_refresh_interval >= args.claim_ttl:
            parser.error("--claim-refresh-interval must be less than --claim-ttl")
        if args.snapshot_refresh_ttl <= args.snapshot_cache_ttl:
            parser.error("--snapshot-refresh-ttl must exceed --snapshot-cache-ttl")
        # The legacy default profile remains untouched without coordination.
        # In opt-in fleet mode, deriving a per-account profile avoids an easy
        # accidental session/profile collision for direct manual launches.
        if args.profile == DEFAULT_PROFILE:
            args.profile = DEFAULT_FLEET_PROFILE_ROOT / f"account-{args.account_id}"
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        try:
            _run_self_test()
        except Exception as error:
            print(f"self-test failed: {_failure_reason(error)}", file=sys.stderr, flush=True)
            return 1
        return 0
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
