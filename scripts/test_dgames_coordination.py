"""Offline concurrency tests for the dominion.games filesystem fleet backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest

from scripts.dgames_collect import (
    REQUEST_UPDATE,
    CollectorStats,
    DominionCollector,
    _atomic_json_create,
    _publish_staged_capture,
)
from scripts.dgames_coordination import (
    CoordinationUnavailable,
    FilesystemCoordinationBackend,
)


WORKERS = 32


class FilesystemClaimProtocolTest(unittest.TestCase):
    def _concurrent_claims(
        self,
        root: Path,
        *,
        scope: str,
        key: int,
        ttl_seconds: float = 2.0,
    ) -> list[object]:
        barrier = threading.Barrier(WORKERS)

        def acquire(index: int) -> object:
            backend = FilesystemCoordinationBackend(root)
            barrier.wait(timeout=10.0)
            return backend.acquire_claim(
                scope=scope,
                key=key,
                owner_id=f"worker_{index}",
                pid=os.getpid(),
                ttl_seconds=ttl_seconds,
            )

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            return list(pool.map(acquire, range(WORKERS)))

    def test_32_concurrent_workers_have_exactly_one_table_and_game_winner(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            backend = FilesystemCoordinationBackend(root)
            for scope, key in (("table", 781), ("game", 181648216)):
                with self.subTest(scope=scope):
                    outcomes = self._concurrent_claims(root, scope=scope, key=key)
                    winners = [outcome for outcome in outcomes if outcome.acquired]
                    self.assertEqual(len(winners), 1)
                    self.assertTrue(winners[0].contended is False)
                    assert winners[0].lease is not None
                    self.assertTrue(backend.release_claim(winners[0].lease))

    def test_stale_claim_is_reclaimable(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            backend = FilesystemCoordinationBackend(Path(temporary_directory))
            original = backend.acquire_claim(
                scope="table",
                key=99,
                owner_id="holder",
                pid=os.getpid(),
                ttl_seconds=0.1,
            )
            assert original.lease is not None
            stale_at = time.time() - 2.0
            os.utime(original.lease.path, (stale_at, stale_at))

            replacement = backend.acquire_claim(
                scope="table",
                key=99,
                owner_id="reclaimer",
                pid=os.getpid(),
                ttl_seconds=0.1,
            )

            self.assertTrue(replacement.acquired)
            self.assertTrue(replacement.reclaimed_stale)
            self.assertFalse(backend.refresh_claim(original.lease))
            assert replacement.lease is not None
            self.assertTrue(backend.release_claim(replacement.lease))

    def test_refreshed_claim_is_never_stolen_by_32_contenders(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            holder_backend = FilesystemCoordinationBackend(root)
            held = holder_backend.acquire_claim(
                scope="game",
                key=1234,
                owner_id="holder",
                pid=os.getpid(),
                ttl_seconds=0.5,
            )
            assert held.lease is not None
            stop_refresher = threading.Event()
            refresh_failures: list[bool] = []

            def refresh() -> None:
                while not stop_refresher.is_set():
                    refresh_failures.append(holder_backend.refresh_claim(held.lease))
                    time.sleep(0.02)

            refresher = threading.Thread(target=refresh, daemon=True)
            refresher.start()
            # Wait longer than TTL, proving that the current lease is live by
            # refresh rather than merely by its original creation time.
            time.sleep(0.7)
            outcomes = self._concurrent_claims(
                root,
                scope="game",
                key=1234,
                ttl_seconds=0.5,
            )
            stop_refresher.set()
            refresher.join(timeout=2.0)

            self.assertTrue(refresh_failures)
            self.assertTrue(all(refresh_failures))
            self.assertEqual(sum(outcome.acquired for outcome in outcomes), 0)
            self.assertTrue(holder_backend.release_claim(held.lease))

    def test_stopped_refresher_simulates_crash_and_releases_after_ttl(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            holder_backend = FilesystemCoordinationBackend(root)
            held = holder_backend.acquire_claim(
                scope="table",
                key=456,
                owner_id="crashed_holder",
                pid=os.getpid(),
                ttl_seconds=0.12,
            )
            assert held.lease is not None
            keep_refreshing = threading.Event()
            keep_refreshing.set()

            def refresh_until_removed() -> None:
                while keep_refreshing.is_set():
                    holder_backend.refresh_claim(held.lease)
                    time.sleep(0.015)

            refresher = threading.Thread(target=refresh_until_removed, daemon=True)
            refresher.start()
            time.sleep(0.08)
            # Simulate a crashed collector by removing its refresher without
            # calling release_claim; no PID liveness heuristic is required.
            keep_refreshing.clear()
            refresher.join(timeout=2.0)
            time.sleep(0.16)

            reclaimer = FilesystemCoordinationBackend(root).acquire_claim(
                scope="table",
                key=456,
                owner_id="replacement",
                pid=os.getpid(),
                ttl_seconds=0.12,
            )
            self.assertTrue(reclaimer.acquired)
            self.assertTrue(reclaimer.reclaimed_stale)
            assert reclaimer.lease is not None
            self.assertTrue(FilesystemCoordinationBackend(root).release_claim(reclaimer.lease))


class SharedSnapshotCacheTest(unittest.TestCase):
    def test_exactly_one_of_32_refreshers_wins_and_everyone_reads_cache(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            barrier = threading.Barrier(WORKERS)

            def acquire(index: int) -> object:
                backend = FilesystemCoordinationBackend(root)
                barrier.wait(timeout=10.0)
                return backend.acquire_snapshot_refresh(
                    owner_id=f"cache_{index}",
                    pid=os.getpid(),
                    ttl_seconds=2.0,
                )

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                outcomes = list(pool.map(acquire, range(WORKERS)))
            winners = [outcome for outcome in outcomes if outcome.acquired]
            self.assertEqual(len(winners), 1)
            assert winners[0].lease is not None
            writer = FilesystemCoordinationBackend(root)
            payload = [
                {
                    "table_id": 7,
                    "host_id": 2,
                    "host_name": "offline-host",
                    "players": 2,
                    "bots": 0,
                    "spectators": 0,
                    "min_players": 2,
                    "max_players": 2,
                    "is_observable": True,
                    "is_joinable": True,
                    "status": 2,
                    "start_time": 123,
                }
            ]
            self.assertTrue(writer.write_lobby_snapshot(lease=winners[0].lease, payload=payload))
            self.assertTrue(writer.release_claim(winners[0].lease))

            read_barrier = threading.Barrier(WORKERS)

            def read(_: int) -> object:
                read_barrier.wait(timeout=10.0)
                return FilesystemCoordinationBackend(root).read_lobby_snapshot(max_age_seconds=2.0)

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                snapshots = list(pool.map(read, range(WORKERS)))
            self.assertTrue(all(snapshot is not None for snapshot in snapshots))
            self.assertTrue(all(snapshot.payload == payload for snapshot in snapshots))

    def test_cache_unavailability_falls_back_to_per_account_lobby_poll(self) -> None:
        class UnavailableCache:
            def read_lobby_snapshot(self, *, max_age_seconds: float) -> object:
                raise CoordinationUnavailable("offline cache failure")

            def acquire_snapshot_refresh(
                self,
                *,
                owner_id: str,
                pid: int,
                ttl_seconds: float,
            ) -> object:
                raise CoordinationUnavailable("offline cache failure")

        class Session:
            def __init__(self) -> None:
                self.frames: list[tuple[int, bytes]] = []

            async def send_frame(self, msg_type: int, payload: bytes) -> None:
                self.frames.append((msg_type, payload))

        collector = object.__new__(DominionCollector)
        collector.coordination = UnavailableCache()
        collector.account_id = "one"
        collector.args = SimpleNamespace(
            snapshot_cache_ttl=20.0,
            snapshot_refresh_ttl=30.0,
            max_snapshot_age=30.0,
            max_poll_retries=5,
        )
        collector.stats = CollectorStats(started_at=datetime.now(timezone.utc))
        collector._last_cache_degradation_notice_monotonic = float("-inf")
        collector._last_claim_contention_notice_monotonic = float("-inf")
        collector.snapshot_refresh_claim = None
        collector.poll_inflight = False
        collector.poll_attempts = 0
        collector.next_poll_monotonic = 0.0
        collector.next_poll_retry_monotonic = 0.0
        collector.session = Session()

        asyncio.run(collector._service_poll(now=1.0))

        self.assertEqual(len(collector.session.frames), 1)
        self.assertEqual(collector.session.frames[0][0], REQUEST_UPDATE)
        self.assertTrue(collector.poll_inflight)
        self.assertEqual(collector.stats.snapshot_cache_degradations, 2)


class GlobalJoinRateLimitTest(unittest.TestCase):
    def test_aggregate_cap_is_respected_under_32_concurrent_demands(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            barrier = threading.Barrier(WORKERS)

            def take(_: int) -> object:
                backend = FilesystemCoordinationBackend(root)
                barrier.wait(timeout=10.0)
                return backend.take_join_token(rate_per_minute=60.0)

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                decisions = list(pool.map(take, range(WORKERS)))

            self.assertEqual(sum(decision.permitted for decision in decisions), 1)
            self.assertTrue(all(decision.retry_after_seconds > 0 for decision in decisions if not decision.permitted))

            # The no-burst bucket admits at most one additional token after a
            # full one-second refill at 60 joins/minute.
            time.sleep(1.05)
            second = FilesystemCoordinationBackend(root).take_join_token(rate_per_minute=60.0)
            immediate = [
                FilesystemCoordinationBackend(root).take_join_token(rate_per_minute=60.0)
                for _ in range(WORKERS)
            ]
            self.assertTrue(second.permitted)
            self.assertEqual(sum(decision.permitted for decision in immediate), 0)


class SharedOutputSafetyTest(unittest.TestCase):
    def test_32_manifest_reservations_and_raw_publication_never_overwrite(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "181648216.manifest.json"
            barrier = threading.Barrier(WORKERS)

            def reserve(index: int) -> bool:
                barrier.wait(timeout=10.0)
                return _atomic_json_create(manifest, {"owner": index})

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                reservations = list(pool.map(reserve, range(WORKERS)))
            self.assertEqual(sum(reservations), 1)

            first_staging = root / ".first.jsonl.gz.tmp"
            second_staging = root / ".second.jsonl.gz.tmp"
            destination = root / "181648216.jsonl.gz"
            first_staging.write_bytes(b"first")
            second_staging.write_bytes(b"second")
            self.assertTrue(_publish_staged_capture(first_staging, destination))
            self.assertFalse(_publish_staged_capture(second_staging, destination))
            self.assertEqual(destination.read_bytes(), b"first")


if __name__ == "__main__":
    unittest.main()
