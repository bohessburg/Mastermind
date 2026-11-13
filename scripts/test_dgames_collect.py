"""Offline session-state tests for the dominion.games collector."""

from __future__ import annotations

import asyncio
import base64
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts.dgames_collect import (
    CLIENT_HEARTBEAT,
    CREDENTIALED_RELOGIN_WINDOW_SECONDS,
    DEFAULT_FLEET_PROFILE_ROOT,
    DEFAULT_TARGET_MAX_AGE_SECONDS,
    DEFAULT_TARGET_MIN_AGE_SECONDS,
    DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
    FAILED_HANDSHAKE_FIXTURE,
    HEALTHY_HANDSHAKE_FIXTURE,
    LOGIN,
    LOGIN_SUCCESS,
    LOGIN_WITH_SESSION,
    REMOVE_ACTIVE_SESSIONS,
    REQUEST_SERVER_STATE,
    CredentialedLoginOutcome,
    CredentialedReloginRateLimiter,
    CollectorStats,
    DgamesCredentials,
    DominionCollector,
    ParkedLoginAction,
    SessionExpired,
    SessionHandshakeTracker,
    TableSummary,
    _load_dgames_credentials,
    _is_sensitive_record,
    _parked_login_action,
    _parse_args,
    _redacted_archive_record,
    _replay_archived_handshake_fixture,
    _select_probe_table,
    _summary_document,
)


def _binary_record(*, direction: str, msg_type: int, socket: int = 3) -> dict[str, object]:
    prefix = b"\x00\x00\x00\x00" if direction == "in" else b""
    return {
        "ts": 0,
        "sock": socket,
        "dir": direction,
        "kind": "binary",
        "url": "wss://example.invalid",
        "data": base64.b64encode(prefix + msg_type.to_bytes(4, "big")).decode("ascii"),
        "b64": True,
    }


def _open_record() -> dict[str, object]:
    return {
        "ts": 0,
        "sock": 3,
        "dir": "in",
        "kind": "open",
        "url": "wss://example.invalid",
        "data": "",
        "b64": False,
    }


SELECTION_NOW_MS = 10_000_000


def _table(*, table_id: int, age_seconds: float) -> TableSummary:
    return TableSummary(
        table_id=table_id,
        host_id=table_id,
        host_name=f"host-{table_id}",
        players=2,
        bots=0,
        spectators=0,
        min_players=2,
        max_players=2,
        is_observable=True,
        is_joinable=True,
        status=2,
        start_time=SELECTION_NOW_MS - int(age_seconds * 1_000),
    )


class TableSelectionPolicyTest(unittest.TestCase):
    def _select(
        self,
        snapshot: tuple[TableSummary, ...],
        *,
        excluded: set[tuple[int, int | None]] | None = None,
        rng: random.Random | None = None,
    ) -> TableSummary | None:
        return _select_probe_table(
            snapshot,
            now_epoch_ms=SELECTION_NOW_MS,
            target_min_age_seconds=DEFAULT_TARGET_MIN_AGE_SECONDS,
            target_max_age_seconds=DEFAULT_TARGET_MAX_AGE_SECONDS,
            excluded_table_keys=set() if excluded is None else excluded,
            rng=random.Random(20260801) if rng is None else rng,
        )

    def test_default_band_is_moderate_and_configurable(self) -> None:
        args = _parse_args([])
        self.assertEqual(args.target_min_age_seconds, 4.0 * 60.0)
        self.assertEqual(args.target_max_age_seconds, 12.0 * 60.0)

    def test_opt_in_fleet_flags_preserve_legacy_default_and_derive_unique_profile(self) -> None:
        legacy = _parse_args([])
        self.assertIsNone(legacy.coordination_dir)
        self.assertFalse(legacy.no_base_filter)

        fleet = _parse_args(
            [
                "--account-id",
                "7",
                "--coordination-dir",
                "data/dominion_games/coord",
                "--no-base-filter",
            ]
        )
        self.assertTrue(fleet.no_base_filter)
        self.assertEqual(fleet.profile, DEFAULT_FLEET_PROFILE_ROOT / "account-7")

    def test_uniform_sampling_stays_in_band_and_does_not_sort_oldest_first(self) -> None:
        snapshot = (
            _table(table_id=1, age_seconds=20.0 * 60.0),
            _table(table_id=2, age_seconds=12.0 * 60.0),
            _table(table_id=3, age_seconds=4.0 * 60.0),
            _table(table_id=4, age_seconds=8.0 * 60.0),
            _table(table_id=5, age_seconds=2.0 * 60.0),
        )
        rng = random.Random(7)
        selections = {
            selected.table_id
            for _ in range(40)
            if (selected := self._select(snapshot, rng=rng)) is not None
        }

        self.assertEqual(selections, {2, 3, 4})

    def test_already_probed_stale_table_is_not_rejoined_but_recycled_slot_is_allowed(self) -> None:
        stale = _table(table_id=91, age_seconds=45.0 * 60.0)
        first = self._select((stale,))
        self.assertIs(first, stale)
        excluded = {(stale.table_id, stale.start_time)}
        self.assertIsNone(self._select((stale,), excluded=excluded))

        replacement = _table(table_id=91, age_seconds=6.0 * 60.0)
        self.assertIs(self._select((replacement,), excluded=excluded), replacement)

    def test_summary_reports_base_only_rate_from_successful_probes(self) -> None:
        stats = CollectorStats(started_at=datetime.now(timezone.utc))
        stats.tables_probed = 11
        stats.base_only_tables_probed = 2

        summary = _summary_document(
            stats=stats,
            outcome="completed",
            abort_reason=None,
            seen_game_ids=0,
        )

        self.assertEqual(summary["tables_probed"], 11)
        self.assertEqual(summary["base_only_tables_probed"], 2)
        self.assertAlmostEqual(summary["base_only_rate_among_probed_tables"], 2 / 11)


class SessionHandshakeTrackerTest(unittest.TestCase):
    def test_healthy_login_is_not_expired(self) -> None:
        tracker = SessionHandshakeTracker(game_socket=3)
        for record, now in (
            (_open_record(), 0.0),
            (_binary_record(direction="out", msg_type=REQUEST_SERVER_STATE, socket=1), 0.1),
            (_binary_record(direction="out", msg_type=LOGIN_WITH_SESSION), 0.2),
            (_binary_record(direction="in", msg_type=LOGIN_SUCCESS), 0.3),
        ):
            tracker.observe(record, now=now)

        self.assertFalse(
            tracker.is_parked_at_login_form(
                now=DEFAULT_SESSION_LOGIN_GRACE_SECONDS + 1.0,
                grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
            )
        )
        self.assertIs(
            _parked_login_action(
                tracker,
                credentials=DgamesCredentials(username="test-user", password="test-password"),
                now=DEFAULT_SESSION_LOGIN_GRACE_SECONDS + 1.0,
                grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
            ),
            ParkedLoginAction.WAIT,
        )

    def test_probe_without_login_reports_expiry_after_short_grace(self) -> None:
        tracker = SessionHandshakeTracker(game_socket=3)
        tracker.observe(_open_record(), now=0.0)
        tracker.observe(
            _binary_record(direction="out", msg_type=REQUEST_SERVER_STATE, socket=1),
            now=0.1,
        )
        deadline = 0.1 + DEFAULT_SESSION_LOGIN_GRACE_SECONDS

        self.assertFalse(
            tracker.is_parked_at_login_form(
                now=deadline - 0.01,
                grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
            )
        )
        self.assertTrue(
            tracker.is_parked_at_login_form(
                now=deadline + 0.01,
                grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
            )
        )
        self.assertIs(
            _parked_login_action(
                tracker,
                credentials=None,
                now=deadline + 0.01,
                grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
            ),
            ParkedLoginAction.SESSION_EXPIRED,
        )
        self.assertIs(
            _parked_login_action(
                tracker,
                credentials=DgamesCredentials(username="test-user", password="test-password"),
                now=deadline + 0.01,
                grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
            ),
            ParkedLoginAction.CREDENTIALED_RELOGIN,
        )

    def test_probe_state_ordinal_two_is_not_game_socket_login_success(self) -> None:
        tracker = SessionHandshakeTracker(game_socket=3)
        tracker.observe(
            _binary_record(direction="out", msg_type=REQUEST_SERVER_STATE, socket=1),
            now=0.0,
        )
        tracker.observe(_open_record(), now=0.1)
        # Probe frames use [sequence][server-state ordinal].  An ordinal of 2
        # must not be mistaken for inbound loginSuccess (also protocol type 2).
        probe_reply = _binary_record(direction="in", msg_type=LOGIN_SUCCESS, socket=1)
        tracker.observe(probe_reply, now=0.2)

        self.assertIsNone(tracker.login_success_at)
        self.assertFalse(_is_sensitive_record(probe_reply))
        self.assertNotIn("redacted", _redacted_archive_record(probe_reply))
        self.assertTrue(
            _is_sensitive_record(
                _binary_record(direction="in", msg_type=LOGIN_SUCCESS, socket=3)
            )
        )
        self.assertTrue(
            tracker.is_parked_at_login_form(
                now=DEFAULT_SESSION_LOGIN_GRACE_SECONDS + 0.2,
                grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
            )
        )

    def test_no_login_evidence_survives_socket_three_reload_generations(self) -> None:
        tracker = SessionHandshakeTracker(game_socket=3)
        tracker.observe(
            _binary_record(direction="out", msg_type=REQUEST_SERVER_STATE, socket=1),
            now=0.0,
        )
        tracker.observe(_open_record(), now=0.1)
        self.assertFalse(
            tracker.is_parked_at_login_form(
                now=4.9,
                grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
            )
        )
        # A fresh page gets the same socket numbering.  Its own short setup
        # window must not erase the first generation's no-login evidence.
        tracker.observe(
            _binary_record(direction="out", msg_type=REQUEST_SERVER_STATE, socket=1),
            now=5.0,
        )
        tracker.observe(_open_record(), now=5.1)

        self.assertEqual(tracker.game_socket_generations, 2)
        self.assertTrue(
            tracker.is_parked_at_login_form(
                now=5.1,
                grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
            )
        )

    def test_heartbeat_stays_non_sensitive(self) -> None:
        heartbeat = _binary_record(direction="out", msg_type=CLIENT_HEARTBEAT)
        self.assertFalse(_is_sensitive_record(heartbeat))
        self.assertNotIn("redacted", _redacted_archive_record(heartbeat))
        self.assertTrue(
            _is_sensitive_record(_binary_record(direction="out", msg_type=LOGIN_WITH_SESSION))
        )
        self.assertTrue(
            _is_sensitive_record(_binary_record(direction="out", msg_type=REQUEST_SERVER_STATE))
        )

    def test_all_credentialed_login_traffic_stays_sensitive(self) -> None:
        for msg_type in (
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
            with self.subTest(msg_type=msg_type):
                record = _binary_record(direction="out", msg_type=msg_type)
                self.assertTrue(_is_sensitive_record(record))
                archived = _redacted_archive_record(record)
                self.assertTrue(archived["redacted"])
                self.assertEqual(archived["data"], "")
                self.assertFalse(archived["b64"])


class CredentialedReloginRateLimiterTest(unittest.TestCase):
    def test_hourly_cap_and_minimum_backoff(self) -> None:
        limiter = CredentialedReloginRateLimiter(
            max_attempts_per_hour=2,
            minimum_backoff_seconds=10.0,
        )
        self.assertTrue(limiter.check(now=0.0).permitted)
        limiter.record(now=0.0)
        self.assertEqual(limiter.check(now=5.0).backoff_seconds, 5.0)
        self.assertTrue(limiter.check(now=10.0).permitted)
        limiter.record(now=10.0)

        capped = limiter.check(now=11.0)
        self.assertFalse(capped.permitted)
        self.assertEqual(capped.attempts_in_window, 2)
        self.assertIsNotNone(capped.reset_after_seconds)
        self.assertTrue(limiter.check(now=CREDENTIALED_RELOGIN_WINDOW_SECONDS).permitted)


class ArchivedHandshakeFixtureTest(unittest.TestCase):
    def test_failed_fixture_escalates_to_the_correct_credential_action(self) -> None:
        tracker, last_now = _replay_archived_handshake_fixture(FAILED_HANDSHAKE_FIXTURE)
        check_at = last_now + DEFAULT_SESSION_LOGIN_GRACE_SECONDS + 0.01

        self.assertEqual(tracker.game_socket_generations, 5)
        self.assertIsNone(tracker.login_with_session_at)
        self.assertIsNone(tracker.login_success_at)
        self.assertTrue(
            tracker.is_parked_at_login_form(
                now=check_at,
                grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
            )
        )
        self.assertIs(
            _parked_login_action(
                tracker,
                credentials=None,
                now=check_at,
                grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
            ),
            ParkedLoginAction.SESSION_EXPIRED,
        )
        self.assertIs(
            _parked_login_action(
                tracker,
                credentials=DgamesCredentials(
                    username="fixture-user",
                    password="fixture-password",
                ),
                now=check_at,
                grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
            ),
            ParkedLoginAction.CREDENTIALED_RELOGIN,
        )

    def test_healthy_fixture_never_selects_parked_login_recovery(self) -> None:
        tracker, last_now = _replay_archived_handshake_fixture(HEALTHY_HANDSHAKE_FIXTURE)
        check_at = last_now + DEFAULT_SESSION_LOGIN_GRACE_SECONDS + 0.01

        self.assertIsNotNone(tracker.login_with_session_at)
        self.assertIsNotNone(tracker.login_success_at)
        self.assertFalse(
            tracker.is_parked_at_login_form(
                now=check_at,
                grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
            )
        )
        self.assertIs(
            _parked_login_action(
                tracker,
                credentials=None,
                now=check_at,
                grace_seconds=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
            ),
            ParkedLoginAction.WAIT,
        )


class LoginWaitPathTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _wiped_session_tracker() -> SessionHandshakeTracker:
        tracker = SessionHandshakeTracker(game_socket=3)
        tracker.observe(_open_record(), now=0.0)
        # The dead-profile signature deliberately contains probe traffic only:
        # no outbound LOGIN_WITH_SESSION (24), no inbound loginSuccess (2).
        tracker.observe(
            _binary_record(direction="out", msg_type=REQUEST_SERVER_STATE, socket=1),
            now=0.1,
        )
        return tracker

    def _collector_for(self, credentials: DgamesCredentials | None) -> DominionCollector:
        collector = object.__new__(DominionCollector)
        collector.args = SimpleNamespace(
            startup_timeout=30.0,
            session_login_grace=DEFAULT_SESSION_LOGIN_GRACE_SECONDS,
        )
        collector.credentials = credentials
        collector.player_id = None
        collector.pump = SimpleNamespace(
            parser=SimpleNamespace(our_player_id=None),
            socket_open=True,
            login_handshake=self._wiped_session_tracker(),
        )
        collector._page_is_usable = lambda: True
        return collector

    async def test_wiped_session_calls_credentialed_relogin_when_available(self) -> None:
        collector = self._collector_for(
            DgamesCredentials(username="test-user", password="test-password")
        )
        calls = 0

        async def credentialed_relogin(_stop_requested: asyncio.Event) -> int:
            nonlocal calls
            calls += 1
            return 123

        collector._credentialed_relogin = credentialed_relogin
        with redirect_stdout(StringIO()):
            await collector._wait_for_login(asyncio.Event())

        self.assertEqual(calls, 1)
        self.assertEqual(collector.player_id, 123)

    async def test_wiped_session_fast_fails_without_credentials(self) -> None:
        collector = self._collector_for(None)
        called = False

        async def login_form_is_visible() -> bool:
            return False

        async def credentialed_relogin(_stop_requested: asyncio.Event) -> int:
            nonlocal called
            called = True
            return 123

        collector._login_form_is_visible = login_form_is_visible
        collector._credentialed_relogin = credentialed_relogin

        with redirect_stdout(StringIO()):
            with self.assertRaisesRegex(SessionExpired, "credentialed self-heal unavailable"):
                await collector._wait_for_login(asyncio.Event())
        self.assertFalse(called)

    async def test_already_logged_in_requests_cleanup_then_retries_once(self) -> None:
        collector = object.__new__(DominionCollector)
        collector.credentials = DgamesCredentials(username="test-user", password="test-password")
        collector._active_sessions_cleanup_attempted = False
        sent_frames: list[tuple[int, bytes]] = []

        class Session:
            async def send_frame(self, msg_type: int, payload: bytes) -> None:
                sent_frames.append((msg_type, payload))

        outcomes = [
            ("already_logged_in", None),
            ("succeeded", 321),
        ]

        async def credentialed_login_attempt(
            _credentials: DgamesCredentials,
            _stop_requested: asyncio.Event,
        ) -> tuple[CredentialedLoginOutcome, int | None]:
            outcome, player_id = outcomes.pop(0)
            if outcome == "already_logged_in":
                return (CredentialedLoginOutcome.ALREADY_LOGGED_IN, player_id)
            return (CredentialedLoginOutcome.SUCCEEDED, player_id)

        collector.session = Session()
        collector._credentialed_login_attempt = credentialed_login_attempt
        with redirect_stdout(StringIO()):
            player_id = await collector._credentialed_relogin(asyncio.Event())

        self.assertEqual(player_id, 321)
        self.assertEqual(sent_frames, [(REMOVE_ACTIVE_SESSIONS, b"")])
        self.assertEqual(outcomes, [])


class CredentialSourceTest(unittest.TestCase):
    def test_env_file_fills_missing_environment_keys_without_exposable_repr(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            env_file = Path(temporary_directory) / ".env"
            env_file.write_text(
                "DGAMES_USER=offline-user\nDGAMES_PASS=offline-password\n",
                encoding="utf-8",
            )
            credentials = _load_dgames_credentials(environment={}, env_file=env_file)
            environment_wins = _load_dgames_credentials(
                environment={"DGAMES_USER": "environment-user"},
                env_file=env_file,
            )

        self.assertIsNotNone(credentials)
        assert credentials is not None
        self.assertEqual(credentials.username, "offline-user")
        self.assertEqual(credentials.password, "offline-password")
        self.assertNotIn("offline-user", repr(credentials))
        self.assertNotIn("offline-password", repr(credentials))
        self.assertIsNotNone(environment_wins)
        assert environment_wins is not None
        self.assertEqual(environment_wins.username, "environment-user")
        self.assertEqual(environment_wins.password, "offline-password")

    def test_indexed_fleet_credentials_do_not_fall_back_to_legacy_pair(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            env_file = Path(temporary_directory) / ".env"
            env_file.write_text(
                "DGAMES_USER=legacy-user\n"
                "DGAMES_PASS=legacy-password\n"
                "DGAMES_USER_7=fleet-user\n"
                "DGAMES_PASS_7=fleet-password\n",
                encoding="utf-8",
            )
            indexed = _load_dgames_credentials(
                environment={},
                env_file=env_file,
                account_id="7",
            )
            missing_indexed = _load_dgames_credentials(
                environment={},
                env_file=env_file,
                account_id="8",
            )

        self.assertIsNotNone(indexed)
        assert indexed is not None
        self.assertEqual(indexed.username, "fleet-user")
        self.assertEqual(indexed.password, "fleet-password")
        self.assertIsNone(missing_indexed)


if __name__ == "__main__":
    unittest.main()
