"""Poll the dominion.games leaderboard into an append-only ratings history.

The browser page owns the authenticated Dominion session.  This script reuses
the collector's persistent-profile, credentialed-self-heal machinery, but it
does not join tables or write game frames.  One leaderboard request is made per
poll interval (one per hour by default), then every returned rating type is
appended as a separate observation.

Usage::

    PYTHONUNBUFFERED=1 ./.venv/bin/python scripts/dgames_ratings.py --once
    PYTHONUNBUFFERED=1 ./.venv/bin/python scripts/dgames_ratings.py

The current client bundle (2.2.9) defines REQUEST_LEADERBOARD as outbound
message 28 with ``int count, boolean flag`` and supplies message 25 as a map
of rating types to full leaderboard entries.  See recon/RECON.md for the
versioned wire layout.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v2.arena.protocol.frames import Direction, ProtocolError, Reader, Writer
from scripts.dgames_collect import (
    DEFAULT_CREDENTIALED_LOGIN_TIMEOUT_SECONDS,
    DEFAULT_CREDENTIALED_RELOGIN_BACKOFF_SECONDS,
    DEFAULT_FLEET_PROFILE_ROOT,
    DEFAULT_MAX_CREDENTIALED_RELOGINS_PER_HOUR,
    DEFAULT_PROFILE,
    DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
    DEFAULT_URL,
    CredentialedReloginRateLimitExceeded,
    CredentialedReloginRateLimiter,
    CollectorSession,
    DominionCollector,
    LivePump,
    RecoverPage,
    RestartBrowser,
    SessionExpired,
    SessionHandshakeTracker,
    StopRequested,
    _credential_environment_keys,
    _failure_reason,
    _load_dgames_credentials,
)


# Positional ids re-derived from dominion-webclient-body-2.2.9.min.js.
REQUEST_LEADERBOARD = 28
LEADERBOARD = 25
RATING_TYPE_NAMES = (
    "RATINGS_2P",
    "RATINGS_3P",
    "RATINGS_2P_BLITZ",
    "RATINGS_3P_BLITZ",
)

DEFAULT_RATINGS_ROOT = Path("data/dominion_games/ratings")
DEFAULT_POLL_INTERVAL_SECONDS = 60.0 * 60.0
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0
DEFAULT_RESPONSE_TIMEOUT_SECONDS = 30.0
DEFAULT_STARTUP_TIMEOUT_SECONDS = 45.0
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5
DEFAULT_RETRY_BASE_SECONDS = 2.0
DEFAULT_MAX_BACKOFF_SECONDS = 60.0
MAX_INT32 = 2_147_483_647
SESSION_EXPIRED_EXIT_CODE = 2


@dataclass(frozen=True)
class LeaderboardEntry:
    """One current-client leaderboard row, flattened from its rating-type map."""

    player_id: int
    player_name: str
    rank: int
    # ``level`` is the rating displayed by the current leaderboard UI.
    rating: float
    trend: float
    rating_type: str
    skill: float
    deviation: float
    volatility: float
    converted_skill: float
    converted_deviation: float
    game_count: int


@dataclass(frozen=True)
class LeaderboardResponse:
    """A decoded response paired with the UTC instant it was received."""

    observed_at_utc: str
    entries: tuple[LeaderboardEntry, ...]


@dataclass
class RatingsStats:
    """Aggregate-only health data; no credentials or frame payloads are retained."""

    started_at: datetime
    polls_requested: int = 0
    leaderboard_responses: int = 0
    observations_appended: int = 0
    failed: int = 0
    in_page_recoveries: int = 0
    hard_browser_relaunches: int = 0
    credentialed_relogins: int = 0
    credentialed_self_heal_available: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def encode_leaderboard_request(*, count: int, flag: bool = True) -> bytes:
    """Encode the payload after outbound type 28 in current-client wire order."""

    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_INT32:
        raise ValueError(f"leaderboard count must be an int in [1, {MAX_INT32}]")
    return Writer().s32(count).boolean(flag).build()


def _read_nonnegative_count(reader: Reader, *, label: str, minimum_item_bytes: int) -> int:
    count = reader.s32()
    if count < 0:
        raise ProtocolError(f"negative {label} count {count}")
    if count > reader.remaining // minimum_item_bytes:
        raise ProtocolError(
            f"implausible {label} count {count} with {reader.remaining} bytes remaining"
        )
    return count


def decode_leaderboard_payload(payload: bytes) -> tuple[LeaderboardEntry, ...]:
    """Decode current inbound message 25, including all rating-type buckets.

    The client reads an enum-to-object map rather than the obsolete four-field
    ``RankedPlayer`` shape.  Each wire row is ``NamedId, rank, seven doubles,
    gameCount``; the displayed rating/trend are ``level`` and ``levelChange``.
    """

    reader = Reader(payload)
    type_count = reader.s32()
    if type_count < 0:
        raise ProtocolError(f"negative leaderboard rating-type count {type_count}")
    if type_count > len(RATING_TYPE_NAMES):
        raise ProtocolError(
            f"unsupported leaderboard rating-type count {type_count}; "
            f"client defines {len(RATING_TYPE_NAMES)}"
        )

    entries: list[LeaderboardEntry] = []
    seen_types: set[int] = set()
    # NamedId is at least eight bytes (int + empty string length), followed by
    # rank, seven doubles, and gameCount: 8 + 4 + 56 + 4 = 72 bytes.
    minimum_entry_bytes = 72
    for _ in range(type_count):
        rating_type_ordinal = reader.s32()
        if not 0 <= rating_type_ordinal < len(RATING_TYPE_NAMES):
            raise ProtocolError(f"unknown rating type ordinal {rating_type_ordinal}")
        if rating_type_ordinal in seen_types:
            raise ProtocolError(f"duplicate rating type ordinal {rating_type_ordinal}")
        seen_types.add(rating_type_ordinal)
        rating_type = RATING_TYPE_NAMES[rating_type_ordinal]
        entry_count = _read_nonnegative_count(
            reader,
            label=f"{rating_type} leaderboard entry",
            minimum_item_bytes=minimum_entry_bytes,
        )
        for _ in range(entry_count):
            player_id = reader.s32()
            player_name = reader.string()
            rank = reader.s32()
            level = reader.f64()
            level_change = reader.f64()
            skill = reader.f64()
            deviation = reader.f64()
            volatility = reader.f64()
            converted_skill = reader.f64()
            converted_deviation = reader.f64()
            game_count = reader.s32()
            entries.append(
                LeaderboardEntry(
                    player_id=player_id,
                    player_name=player_name,
                    rank=rank,
                    rating=level,
                    trend=level_change,
                    rating_type=rating_type,
                    skill=skill,
                    deviation=deviation,
                    volatility=volatility,
                    converted_skill=converted_skill,
                    converted_deviation=converted_deviation,
                    game_count=game_count,
                )
            )
    reader.finish()
    return tuple(entries)


def observation_documents(response: LeaderboardResponse) -> tuple[dict[str, object], ...]:
    """Convert one response to append-only JSONL records without dropping fields."""

    return tuple(
        {
            "schema_version": 1,
            "player_id": entry.player_id,
            "player_name": entry.player_name,
            "observed_at_utc": response.observed_at_utc,
            "rank": entry.rank,
            "rating": entry.rating,
            "trend": entry.trend,
            "rating_type": entry.rating_type,
            "skill": entry.skill,
            "deviation": entry.deviation,
            "volatility": entry.volatility,
            "converted_skill": entry.converted_skill,
            "converted_deviation": entry.converted_deviation,
            "game_count": entry.game_count,
        }
        for entry in response.entries
    )


def append_observations(path: Path, documents: tuple[dict[str, object], ...]) -> int:
    """Durably append one response; existing observations are never rewritten."""

    if not documents:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as output:
        for document in documents:
            output.write(json.dumps(document, sort_keys=True, separators=(",", ":")))
            output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    return len(documents)


class RatingsPoller(DominionCollector):
    """Collector-authenticated transport specialized to leaderboard polling.

    This intentionally inherits the collector's authentication helpers without
    calling its capture-oriented constructor.  In particular, ``_wait_for_login``
    and its credentialed form self-heal stay single-sourced in
    ``scripts.dgames_collect``; this class supplies only the small amount of
    leaderboard-specific state those helpers need.
    """

    def __init__(self, args: argparse.Namespace, stats: RatingsStats) -> None:
        self.args = args
        self.stats = stats
        self.account_id = args.account_id
        self.credentials = _load_dgames_credentials(account_id=self.account_id)
        self.stats.credentialed_self_heal_available = self.credentials is not None
        self.credentialed_relogin_limiter = CredentialedReloginRateLimiter(
            max_attempts_per_hour=args.max_credentialed_relogins_per_hour,
            minimum_backoff_seconds=args.credentialed_relogin_backoff,
        )
        self.session: CollectorSession | None = None
        self.pump: LivePump | None = None
        self.login_handshake: SessionHandshakeTracker | None = None
        self.player_id: int | None = None
        self._active_sessions_cleanup_attempted = False
        self._in_page_recovery_streak = 0
        self.last_session_reached_healthy_state = False
        self._pending_leaderboard: LeaderboardResponse | None = None

    def _reset_page_cycle_state(self) -> None:
        """Reset only ratings-page state between a local page reload."""

        self.player_id = None
        self._pending_leaderboard = None

    def _handle_notice(self, notice: Any) -> None:
        """Consume only type-level transport state and decoded leaderboard data.

        ``CollectorSession`` already redacts its diagnostic archive using the
        collector's sensitive-message set.  This poller keeps no raw records
        and never logs a payload, so reauthentication data cannot enter the
        ratings store or console output.
        """

        frame = notice.frame
        if frame is None:
            if self._pump.socket_closed:
                raise RecoverPage("game websocket closed")
            return
        if frame.direction is not Direction.INBOUND or frame.msg_type != LEADERBOARD:
            return
        try:
            entries = decode_leaderboard_payload(frame.payload)
        except ProtocolError as error:
            raise RecoverPage(f"malformed leaderboard response: {error}") from error
        self._pending_leaderboard = LeaderboardResponse(
            observed_at_utc=_utc_now(),
            entries=entries,
        )

    def _take_pending_leaderboard(self) -> LeaderboardResponse | None:
        response, self._pending_leaderboard = self._pending_leaderboard, None
        return response

    async def _poll_once(self, stop_requested: asyncio.Event) -> None:
        """Send exactly one polite type-28 request and append its type-25 reply."""

        self._pending_leaderboard = None
        try:
            await self._session.send_frame(
                REQUEST_LEADERBOARD,
                encode_leaderboard_request(count=self.args.count, flag=True),
            )
        except Exception as error:
            raise RecoverPage(
                f"could not send leaderboard request: {_failure_reason(error)}"
            ) from error
        self.stats.polls_requested += 1
        print(
            f"leaderboard poll {self.stats.polls_requested}: requested count={self.args.count} "
            "flag=true",
            flush=True,
        )
        try:
            response = await self._pump.wait_for(
                self._take_pending_leaderboard,
                timeout=self.args.response_timeout,
                label="leaderboard response",
                stop_requested=stop_requested,
                on_notice=self._handle_notice,
            )
        except TimeoutError as error:
            raise RecoverPage(
                f"leaderboard response timed out after {self.args.response_timeout:.1f}s"
            ) from error

        appended = append_observations(
            self.args.observations,
            observation_documents(response),
        )
        self.stats.leaderboard_responses += 1
        self.stats.observations_appended += appended
        rating_types = len({entry.rating_type for entry in response.entries})
        print(
            f"leaderboard response: {len(response.entries)} row(s) across {rating_types} "
            f"rating type(s); appended {appended} observation(s)",
            flush=True,
        )

    async def _wait_until_next_poll(self, stop_requested: asyncio.Event) -> None:
        """Idle without blocking browser traffic, emitting bounded heartbeats."""

        deadline = time.monotonic() + self.args.poll_interval
        next_heartbeat = time.monotonic() + self.args.heartbeat_interval
        while True:
            if stop_requested.is_set():
                raise StopRequested()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            until_heartbeat = max(0.0, next_heartbeat - time.monotonic())
            try:
                await asyncio.wait_for(
                    stop_requested.wait(),
                    timeout=min(remaining, until_heartbeat, 60.0),
                )
            except TimeoutError:
                pass
            else:
                raise StopRequested()
            now = time.monotonic()
            if now >= next_heartbeat:
                print(
                    "heartbeat: ratings poller healthy; "
                    f"polls={self.stats.polls_requested} "
                    f"observations={self.stats.observations_appended} "
                    f"next-poll-in={max(0.0, deadline - now):.0f}s",
                    flush=True,
                )
                next_heartbeat = now + self.args.heartbeat_interval

    async def _poll_loop(self, stop_requested: asyncio.Event) -> None:
        while True:
            await self._poll_once(stop_requested)
            if self.args.once:
                return
            await self._wait_until_next_poll(stop_requested)

    async def _recover_page_in_place(
        self,
        error: RecoverPage,
        stop_requested: asyncio.Event,
    ) -> None:
        """Reload one browser page while retaining its persistent profile/session."""

        if stop_requested.is_set():
            raise StopRequested()
        if not self._page_is_usable():
            raise RestartBrowser(f"page unavailable after ratings transport failure: {error}")
        next_streak = self._in_page_recovery_streak + 1
        if next_streak >= self.args.max_consecutive_failures:
            raise RestartBrowser(
                f"{next_streak} consecutive in-page ratings recoveries; last failure: {error}"
            )
        self._in_page_recovery_streak = next_streak
        recovery_started_ms = int(time.time() * 1_000)
        print(
            f"recovering ratings page in place: {error}; attempt {next_streak}",
            flush=True,
        )
        self._discard_queued_frames()
        self._pending_leaderboard = None
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
                "could not reload ratings page: " + _failure_reason(reload_error)
            ) from reload_error
        self.stats.failed += 1
        self.stats.in_page_recoveries += 1

    async def run_browser_session(self, stop_requested: asyncio.Event) -> None:
        """Run one long-lived browser, delegating login/self-heal to the collector."""

        self.session = CollectorSession(
            output_root=self.args.ratings_root / "runs",
            profile_dir=self.args.profile,
            url=self.args.url,
            headless=not self.args.headful,
        )
        self.pump = None
        self.login_handshake = None
        self._in_page_recovery_streak = 0
        self.last_session_reached_healthy_state = False
        try:
            print(
                f"launching authenticated ratings browser "
                f"({'headful' if self.args.headful else 'headless'})...",
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
            self.pump = LivePump(self._session, login_handshake=self.login_handshake)
            while True:
                self._reset_page_cycle_state()
                try:
                    await self._wait_for_login(stop_requested)
                    self._mark_session_healthy()
                    await self._poll_loop(stop_requested)
                    return
                except RecoverPage as error:
                    await self._recover_page_in_place(error, stop_requested)
        finally:
            session, self.session = self.session, None
            self.pump = None
            self.login_handshake = None
            self.player_id = None
            if session is not None:
                await session.stop()


def _summary_document(
    *,
    stats: RatingsStats,
    outcome: str,
    abort_reason: str | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "started_at_utc": stats.started_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "finished_at_utc": _utc_now(),
        "outcome": outcome,
        "abort_reason": abort_reason,
        "poll_interval_seconds": args.poll_interval,
        "maximum_request_rate_per_hour": 3600.0 / args.poll_interval,
        "request_count": args.count,
        "request_flag": True,
        "polls_requested": stats.polls_requested,
        "leaderboard_responses": stats.leaderboard_responses,
        "observations_appended": stats.observations_appended,
        "failed": stats.failed,
        "in_page_recoveries": stats.in_page_recoveries,
        "hard_browser_relaunches": stats.hard_browser_relaunches,
        "credentialed_self_heal_available": stats.credentialed_self_heal_available,
        "credentialed_relogins": stats.credentialed_relogins,
        "observations_path": str(args.observations),
    }


def _write_summary(args: argparse.Namespace, document: dict[str, object]) -> Path:
    runs_root = args.ratings_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = runs_root / f"ratings-summary-{timestamp}.json"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


async def _run(args: argparse.Namespace) -> int:
    stats = RatingsStats(started_at=datetime.now(timezone.utc))
    poller: RatingsPoller | None = None
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
        poller = RatingsPoller(args, stats)
        print(
            "ratings polling rate: at most "
            f"{3600.0 / args.poll_interval:.3f} leaderboard request(s)/hour "
            f"({args.poll_interval:.0f}s interval)",
            flush=True,
        )
        while True:
            try:
                await poller.run_browser_session(stop_requested)
                break
            except StopRequested:
                outcome = "interrupted"
                abort_reason = "interrupted by user signal"
                print("interrupt received; no partial leaderboard response was appended", flush=True)
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
            except RestartBrowser as error:
                if poller.last_session_reached_healthy_state:
                    consecutive_hard_failures = 0
                stats.failed += 1
                consecutive_hard_failures += 1
                if consecutive_hard_failures >= args.max_consecutive_failures:
                    outcome = "aborted"
                    abort_reason = (
                        f"{consecutive_hard_failures} consecutive hard browser failures; "
                        f"last failure: {error}"
                    )
                    exit_code = 1
                    print(f"aborting: {abort_reason}", flush=True)
                    break
                stats.hard_browser_relaunches += 1
                delay = min(
                    args.max_backoff,
                    args.retry_base_seconds * (2 ** (consecutive_hard_failures - 1)),
                )
                print(
                    f"hard ratings browser failure: {error}; retry after {delay:.1f}s",
                    flush=True,
                )
                try:
                    await asyncio.wait_for(stop_requested.wait(), timeout=delay)
                except TimeoutError:
                    continue
                outcome = "interrupted"
                abort_reason = "interrupted by user signal during retry backoff"
                break
            except Exception as error:
                outcome = "aborted"
                abort_reason = f"unhandled ratings poller failure: {_failure_reason(error)}"
                exit_code = 1
                print(f"aborting: {abort_reason}", flush=True)
                break
    except KeyboardInterrupt:
        outcome = "interrupted"
        abort_reason = "interrupted by user signal"
    finally:
        for signum in installed_signals:
            try:
                loop.remove_signal_handler(signum)
            except (NotImplementedError, RuntimeError):
                pass
        summary_path = _write_summary(
            args,
            _summary_document(
                stats=stats,
                outcome=outcome,
                abort_reason=abort_reason,
                args=args,
            ),
        )
        print(
            f"done: outcome={outcome}; polls={stats.polls_requested}; "
            f"responses={stats.leaderboard_responses}; "
            f"observations={stats.observations_appended}; failed={stats.failed}",
            flush=True,
        )
        print(f"summary: {summary_path.resolve()}", flush=True)
    return exit_code


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--ratings-root", type=Path, default=DEFAULT_RATINGS_ROOT)
    parser.add_argument(
        "--observations",
        type=Path,
        default=None,
        help="append-only JSONL destination (default: <ratings-root>/observations.jsonl)",
    )
    parser.add_argument("--once", action="store_true", help="request one snapshot and exit")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help="seconds between requests in polling mode (default: 3600)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=MAX_INT32,
        help="maximum leaderboard row count to request; the server may clamp it",
    )
    parser.add_argument("--response-timeout", type=float, default=DEFAULT_RESPONSE_TIMEOUT_SECONDS)
    parser.add_argument("--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT_SECONDS)
    parser.add_argument(
        "--session-login-grace",
        type=float,
        default=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
    )
    parser.add_argument(
        "--credentialed-login-timeout",
        type=float,
        default=DEFAULT_CREDENTIALED_LOGIN_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--credentialed-relogin-backoff",
        type=float,
        default=DEFAULT_CREDENTIALED_RELOGIN_BACKOFF_SECONDS,
    )
    parser.add_argument(
        "--max-credentialed-relogins-per-hour",
        type=int,
        default=DEFAULT_MAX_CREDENTIALED_RELOGINS_PER_HOUR,
    )
    parser.add_argument("--heartbeat-interval", type=float, default=DEFAULT_HEARTBEAT_INTERVAL_SECONDS)
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
    )
    parser.add_argument("--retry-base-seconds", type=float, default=DEFAULT_RETRY_BASE_SECONDS)
    parser.add_argument("--max-backoff", type=float, default=DEFAULT_MAX_BACKOFF_SECONDS)
    parser.add_argument("--headful", action="store_true")
    args = parser.parse_args(argv)

    for name in (
        "poll_interval",
        "response_timeout",
        "startup_timeout",
        "session_login_grace",
        "credentialed_login_timeout",
        "credentialed_relogin_backoff",
        "heartbeat_interval",
        "retry_base_seconds",
        "max_backoff",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if not 1 <= args.count <= MAX_INT32:
        parser.error(f"--count must be in [1, {MAX_INT32}]")
    if args.max_credentialed_relogins_per_hour < 1:
        parser.error("--max-credentialed-relogins-per-hour must be at least one")
    if args.max_consecutive_failures < 1:
        parser.error("--max-consecutive-failures must be at least one")
    if args.account_id is not None:
        try:
            _credential_environment_keys(args.account_id)
        except ValueError as error:
            parser.error(str(error))
        if args.profile == DEFAULT_PROFILE:
            args.profile = DEFAULT_FLEET_PROFILE_ROOT / f"account-{args.account_id}"
    if args.observations is None:
        args.observations = args.ratings_root / "observations.jsonl"
    return args


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
