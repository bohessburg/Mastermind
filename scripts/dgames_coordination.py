"""Swappable coordination primitives for dominion.games collector fleets.

The first backend deliberately uses only a shared local filesystem.  Its
public ``CoordinationBackend`` interface is intentionally small so a future
shared-mount or database implementation can preserve the collector-facing
contract without changing collector control flow.

Filesystem claims are leases, not PID liveness checks: the owner refreshes a
claim's mtime well before its TTL.  Reclamation and refresh share a tiny
per-claim guard file, which prevents a stale reclaimer from unlinking a claim
while its owner is refreshing it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
import time
from typing import Any, Iterator, Mapping


class CoordinationError(RuntimeError):
    """A coordination operation could not be completed safely."""


class CoordinationUnavailable(CoordinationError):
    """The filesystem backend is temporarily unavailable."""


@dataclass(frozen=True)
class ClaimLease:
    """Opaque ownership token for one table, game, or short-lived mutex."""

    scope: str
    key: str
    owner_id: str
    pid: int
    token: str
    path: Path


@dataclass(frozen=True)
class ClaimAcquisition:
    """Result of attempting an exclusive claim."""

    lease: ClaimLease | None
    contended: bool
    reclaimed_stale: bool = False

    @property
    def acquired(self) -> bool:
        return self.lease is not None


@dataclass(frozen=True)
class CachedSnapshot:
    """A fresh shared lobby payload and its wall-clock age."""

    payload: list[dict[str, object]]
    generated_at_epoch: float
    age_seconds: float


@dataclass(frozen=True)
class RateLimitDecision:
    """One global JOIN_TABLE token-bucket decision."""

    permitted: bool
    retry_after_seconds: float
    tokens_remaining: float


class CoordinationBackend(ABC):
    """Backend contract used by collectors and the fleet supervisor.

    Implementations must make claim acquisition exclusive across all workers
    that share the backend, and must never silently permit a rate-limited join.
    Cache failures may be surfaced as ``CoordinationUnavailable``; collectors
    then safely fall back to their normal per-account lobby poll.
    """

    @abstractmethod
    def acquire_claim(
        self,
        *,
        scope: str,
        key: int | str,
        owner_id: str,
        pid: int,
        ttl_seconds: float,
    ) -> ClaimAcquisition:
        """Acquire a lease, or return a contended result without joining."""

    @abstractmethod
    def refresh_claim(self, lease: ClaimLease) -> bool:
        """Refresh a lease mtime; return false only when ownership was lost."""

    @abstractmethod
    def release_claim(self, lease: ClaimLease) -> bool:
        """Release a lease iff this caller still owns its opaque token."""

    @abstractmethod
    def read_lobby_snapshot(self, *, max_age_seconds: float) -> CachedSnapshot | None:
        """Return a fresh cached lobby payload, or ``None`` when stale/missing."""

    @abstractmethod
    def acquire_snapshot_refresh(
        self,
        *,
        owner_id: str,
        pid: int,
        ttl_seconds: float,
    ) -> ClaimAcquisition:
        """Acquire the one-at-a-time shared lobby refresh mutex."""

    @abstractmethod
    def write_lobby_snapshot(
        self,
        *,
        lease: ClaimLease,
        payload: list[dict[str, object]],
    ) -> bool:
        """Publish a snapshot only while the refresh lease remains owned."""

    @abstractmethod
    def take_join_token(self, *, rate_per_minute: float) -> RateLimitDecision:
        """Consume one fleet-wide JOIN_TABLE token, or provide a retry delay."""

    @abstractmethod
    def write_account_heartbeat(
        self,
        *,
        account_id: str,
        document: Mapping[str, object],
    ) -> None:
        """Publish non-sensitive per-account health for a fleet supervisor."""

    @abstractmethod
    def read_account_heartbeats(
        self,
        *,
        account_ids: list[str],
    ) -> dict[str, dict[str, object]]:
        """Read the latest valid health document for each requested account."""

    @abstractmethod
    def write_fleet_heartbeat(self, *, document: Mapping[str, object]) -> None:
        """Publish the fleet supervisor's aggregate non-sensitive heartbeat."""


class FilesystemCoordinationBackend(CoordinationBackend):
    """Atomic-file coordination suitable for collectors on one machine.

    All mutable files live below ``root``.  Claim data contains no credentials:
    only account identity, process id, timestamps, and a random lease token.
    """

    _GUARD_STALE_SECONDS = 10.0
    _GUARD_WAIT_SECONDS = 2.0
    _GUARD_RETRY_SECONDS = 0.005
    _RATE_BUCKET_CAPACITY = 1.0

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.claims_root = self.root / "claims"
        self.heartbeats_root = self.root / "heartbeats"
        try:
            self.claims_root.mkdir(parents=True, exist_ok=True)
            self.heartbeats_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise CoordinationUnavailable(
                f"could not initialize coordination directory {self.root}"
            ) from error

    @staticmethod
    def _safe_component(value: int | str, *, label: str) -> str:
        text = str(value)
        if not text or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in text):
            raise ValueError(f"{label} must contain only letters, digits, underscore, or hyphen")
        return text

    def _claim_path(self, *, scope: str, key: int | str) -> Path:
        safe_scope = self._safe_component(scope, label="claim scope")
        safe_key = self._safe_component(key, label="claim key")
        return self.claims_root / f"{safe_scope}-{safe_key}.claim"

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        """Best-effort directory sync after namespace changes."""
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

    @staticmethod
    def _write_descriptor_json(descriptor: int, document: Mapping[str, object]) -> None:
        payload = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)

    def _atomic_json_replace(self, destination: Path, document: Mapping[str, object]) -> None:
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8") as output:
                json.dump(document, output, sort_keys=True, separators=(",", ":"))
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            self._fsync_directory(destination.parent)
        except OSError as error:
            raise CoordinationUnavailable(f"could not update coordination file {destination}") from error
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as error:
            raise CoordinationUnavailable(f"could not read coordination file {path}") from error
        if not isinstance(loaded, dict):
            raise CoordinationUnavailable(f"invalid coordination document in {path}")
        return loaded

    @contextmanager
    def _guard(self, protected_path: Path) -> Iterator[None]:
        """Serialize refresh/reclaim/update decisions for a tiny critical section."""
        guard_path = protected_path.with_name(f".{protected_path.name}.guard")
        token = secrets.token_hex(12)
        deadline = time.monotonic() + self._GUARD_WAIT_SECONDS
        while True:
            try:
                descriptor = os.open(
                    guard_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                try:
                    age = time.time() - guard_path.stat().st_mtime
                except FileNotFoundError:
                    continue
                except OSError as error:
                    raise CoordinationUnavailable(f"could not inspect coordination guard {guard_path}") from error
                if age > self._GUARD_STALE_SECONDS:
                    # Guard holders never perform I/O that can legitimately
                    # take this long.  A crashed holder must not deadlock all
                    # claim refreshes forever.
                    try:
                        guard_path.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError as error:
                        raise CoordinationUnavailable(
                            f"could not reclaim stale coordination guard {guard_path}"
                        ) from error
                    continue
                if time.monotonic() >= deadline:
                    raise CoordinationUnavailable(
                        f"timed out waiting for coordination guard {guard_path}"
                    )
                time.sleep(self._GUARD_RETRY_SECONDS)
                continue
            except OSError as error:
                raise CoordinationUnavailable(f"could not create coordination guard {guard_path}") from error
            try:
                self._write_descriptor_json(
                    descriptor,
                    {
                        "pid": os.getpid(),
                        "token": token,
                        "created_at_epoch": time.time(),
                    },
                )
            except OSError as error:
                raise CoordinationUnavailable(f"could not initialize coordination guard {guard_path}") from error
            finally:
                os.close(descriptor)
            break
        try:
            yield
        finally:
            try:
                guard_document = self._read_json(guard_path)
                if guard_document is not None and guard_document.get("token") == token:
                    guard_path.unlink()
                    self._fsync_directory(guard_path.parent)
            except (CoordinationUnavailable, OSError):
                # A later stale-guard reclamation is safe; never make a
                # best-effort release mask collector shutdown.
                pass

    def _create_claim(
        self,
        *,
        path: Path,
        scope: str,
        key: int | str,
        owner_id: str,
        pid: int,
    ) -> ClaimLease | None:
        token = secrets.token_hex(16)
        document = {
            "owner_account_id": owner_id,
            "pid": pid,
            "claimed_at_epoch": time.time(),
            "token": token,
        }
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return None
        except OSError as error:
            raise CoordinationUnavailable(f"could not create claim {path}") from error
        try:
            self._write_descriptor_json(descriptor, document)
        except OSError as error:
            raise CoordinationUnavailable(f"could not write claim {path}") from error
        finally:
            os.close(descriptor)
        self._fsync_directory(path.parent)
        return ClaimLease(
            scope=str(scope),
            key=str(key),
            owner_id=owner_id,
            pid=pid,
            token=token,
            path=path,
        )

    @staticmethod
    def _matches_lease(document: Mapping[str, object] | None, lease: ClaimLease) -> bool:
        return bool(document) and document.get("token") == lease.token

    def acquire_claim(
        self,
        *,
        scope: str,
        key: int | str,
        owner_id: str,
        pid: int,
        ttl_seconds: float,
    ) -> ClaimAcquisition:
        if ttl_seconds <= 0:
            raise ValueError("claim TTL must be greater than zero")
        safe_owner = self._safe_component(owner_id, label="account id")
        path = self._claim_path(scope=scope, key=key)
        direct = self._create_claim(
            path=path,
            scope=scope,
            key=key,
            owner_id=safe_owner,
            pid=pid,
        )
        if direct is not None:
            return ClaimAcquisition(lease=direct, contended=False)

        # A direct O_EXCL miss is the common contention path.  Take the
        # per-claim guard only to make a refreshed live claim impossible to
        # steal between its mtime check and stale reclamation.
        with self._guard(path):
            try:
                stat = path.stat()
            except FileNotFoundError:
                replacement = self._create_claim(
                    path=path,
                    scope=scope,
                    key=key,
                    owner_id=safe_owner,
                    pid=pid,
                )
                return ClaimAcquisition(
                    lease=replacement,
                    contended=replacement is None,
                    reclaimed_stale=False,
                )
            except OSError as error:
                raise CoordinationUnavailable(f"could not inspect claim {path}") from error
            if time.time() - stat.st_mtime <= ttl_seconds:
                return ClaimAcquisition(lease=None, contended=True)
            try:
                path.unlink()
                self._fsync_directory(path.parent)
            except FileNotFoundError:
                pass
            except OSError as error:
                raise CoordinationUnavailable(f"could not reclaim stale claim {path}") from error
            replacement = self._create_claim(
                path=path,
                scope=scope,
                key=key,
                owner_id=safe_owner,
                pid=pid,
            )
            return ClaimAcquisition(
                lease=replacement,
                contended=replacement is None,
                reclaimed_stale=replacement is not None,
            )

    def refresh_claim(self, lease: ClaimLease) -> bool:
        try:
            with self._guard(lease.path):
                document = self._read_json(lease.path)
                if not self._matches_lease(document, lease):
                    return False
                try:
                    os.utime(lease.path, None)
                except OSError as error:
                    raise CoordinationUnavailable(f"could not refresh claim {lease.path}") from error
                return True
        except FileNotFoundError:
            return False

    def release_claim(self, lease: ClaimLease) -> bool:
        try:
            with self._guard(lease.path):
                document = self._read_json(lease.path)
                if not self._matches_lease(document, lease):
                    return False
                try:
                    lease.path.unlink()
                    self._fsync_directory(lease.path.parent)
                except FileNotFoundError:
                    return False
                except OSError as error:
                    raise CoordinationUnavailable(f"could not release claim {lease.path}") from error
                return True
        except FileNotFoundError:
            return False

    def read_lobby_snapshot(self, *, max_age_seconds: float) -> CachedSnapshot | None:
        if max_age_seconds <= 0:
            raise ValueError("snapshot cache TTL must be greater than zero")
        path = self.root / "lobby-snapshot.json"
        document = self._read_json(path)
        if document is None:
            return None
        generated_at = document.get("generated_at_epoch")
        payload = document.get("tables")
        if (
            not isinstance(generated_at, (int, float))
            or isinstance(generated_at, bool)
            or not isinstance(payload, list)
            or not all(isinstance(item, dict) for item in payload)
        ):
            raise CoordinationUnavailable(f"invalid lobby snapshot cache {path}")
        age = max(0.0, time.time() - float(generated_at))
        if age > max_age_seconds:
            return None
        return CachedSnapshot(
            payload=[dict(item) for item in payload],
            generated_at_epoch=float(generated_at),
            age_seconds=age,
        )

    def acquire_snapshot_refresh(
        self,
        *,
        owner_id: str,
        pid: int,
        ttl_seconds: float,
    ) -> ClaimAcquisition:
        return self.acquire_claim(
            scope="snapshot-refresh",
            key="lobby",
            owner_id=owner_id,
            pid=pid,
            ttl_seconds=ttl_seconds,
        )

    def write_lobby_snapshot(
        self,
        *,
        lease: ClaimLease,
        payload: list[dict[str, object]],
    ) -> bool:
        if lease.scope != "snapshot-refresh" or lease.key != "lobby":
            raise ValueError("lobby snapshot requires a snapshot-refresh lease")
        try:
            with self._guard(lease.path):
                if not self._matches_lease(self._read_json(lease.path), lease):
                    return False
                self._atomic_json_replace(
                    self.root / "lobby-snapshot.json",
                    {
                        "schema_version": 1,
                        "generated_at_epoch": time.time(),
                        "tables": payload,
                    },
                )
                return True
        except FileNotFoundError:
            return False

    def take_join_token(self, *, rate_per_minute: float) -> RateLimitDecision:
        if rate_per_minute <= 0:
            raise ValueError("fleet join rate per minute must be greater than zero")
        path = self.root / "join-rate-limit.json"
        now = time.time()
        try:
            with self._guard(path):
                document = self._read_json(path)
                if document is None:
                    tokens = self._RATE_BUCKET_CAPACITY
                    updated_at = now
                else:
                    raw_tokens = document.get("tokens")
                    raw_updated_at = document.get("updated_at_epoch")
                    if (
                        not isinstance(raw_tokens, (int, float))
                        or isinstance(raw_tokens, bool)
                        or not isinstance(raw_updated_at, (int, float))
                        or isinstance(raw_updated_at, bool)
                    ):
                        raise CoordinationUnavailable(f"invalid join rate-limit state {path}")
                    tokens = min(self._RATE_BUCKET_CAPACITY, max(0.0, float(raw_tokens)))
                    updated_at = max(float(raw_updated_at), now)
                    elapsed = max(0.0, now - float(raw_updated_at))
                    tokens = min(
                        self._RATE_BUCKET_CAPACITY,
                        tokens + elapsed * rate_per_minute / 60.0,
                    )
                permitted = tokens >= 1.0
                if permitted:
                    tokens = max(0.0, tokens - 1.0)
                    retry_after = 0.0
                else:
                    retry_after = max(0.01, (1.0 - tokens) * 60.0 / rate_per_minute)
                self._atomic_json_replace(
                    path,
                    {
                        "schema_version": 1,
                        "tokens": tokens,
                        "updated_at_epoch": updated_at,
                        "rate_per_minute": rate_per_minute,
                    },
                )
                return RateLimitDecision(
                    permitted=permitted,
                    retry_after_seconds=retry_after,
                    tokens_remaining=tokens,
                )
        except FileNotFoundError:
            # A concurrent unlink can only occur while a stale guard is being
            # reclaimed.  Retrying on the next event-loop tick is fail-closed.
            return RateLimitDecision(False, retry_after_seconds=0.25, tokens_remaining=0.0)

    def write_account_heartbeat(
        self,
        *,
        account_id: str,
        document: Mapping[str, object],
    ) -> None:
        safe_account_id = self._safe_component(account_id, label="account id")
        payload = dict(document)
        payload["account_id"] = safe_account_id
        payload["updated_at_epoch"] = time.time()
        self._atomic_json_replace(
            self.heartbeats_root / f"account-{safe_account_id}.json",
            payload,
        )

    def read_account_heartbeats(
        self,
        *,
        account_ids: list[str],
    ) -> dict[str, dict[str, object]]:
        results: dict[str, dict[str, object]] = {}
        for account_id in account_ids:
            safe_account_id = self._safe_component(account_id, label="account id")
            document = self._read_json(self.heartbeats_root / f"account-{safe_account_id}.json")
            if document is not None:
                results[safe_account_id] = document
        return results

    def write_fleet_heartbeat(self, *, document: Mapping[str, object]) -> None:
        payload = dict(document)
        payload["updated_at_epoch"] = time.time()
        self._atomic_json_replace(self.root / "fleet-heartbeat.json", payload)
