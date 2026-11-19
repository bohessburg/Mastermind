"""Supervise isolated dominion.games collector accounts on one machine.

The accounts file intentionally contains no credentials.  It is JSON with a
single ``accounts`` array, for example::

    {"accounts": [{"id": "1"}, {"id": "2", "profile": "dgames-profiles/two"}]}

Each account id uses ``DGAMES_USER_<id>`` and ``DGAMES_PASS_<id>`` from the
normal process environment or repository ``.env``.  The supervisor passes no
credentials on its command line and never reads or logs their values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Mapping


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dgames_coordination import (  # noqa: E402
    CoordinationUnavailable,
    FilesystemCoordinationBackend,
)


DEFAULT_COORDINATION_DIR = Path("data/dominion_games/coord")
DEFAULT_RAW_ROOT = Path("data/dominion_games/raw")
DEFAULT_PROFILE_ROOT = Path("dgames-profiles")
DEFAULT_STARTUP_STAGGER_SECONDS = 5.0
DEFAULT_RESTART_DELAY_SECONDS = 10.0
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0
DEFAULT_HEARTBEAT_STALE_SECONDS = 90.0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class FleetAccount:
    """One non-secret account/process configuration."""

    account_id: str
    profile: Path


@dataclass
class ManagedCollector:
    account: FleetAccount
    next_start_monotonic: float
    process: subprocess.Popen[str] | None = None
    reader: threading.Thread | None = None
    starts: int = 0
    last_exit_code: int | None = None
    last_exit_monotonic: float | None = None


def _safe_account_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("account id must be a string or integer")
    account_id = str(value)
    if not account_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
        for character in account_id
    ):
        raise ValueError("account id must contain only letters, digits, or underscore")
    return account_id


def _load_accounts(
    path: Path,
    *,
    profile_root: Path,
) -> list[FleetAccount]:
    """Load only account ids/profile paths; reject credential-shaped fields."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read accounts file {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"accounts file {path} is not valid JSON") from error
    rows: object
    if isinstance(document, dict):
        rows = document.get("accounts")
    else:
        rows = document
    if not isinstance(rows, list) or not rows:
        raise ValueError("accounts file must contain a non-empty 'accounts' array")

    accounts: list[FleetAccount] = []
    seen_ids: set[str] = set()
    seen_profiles: set[Path] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"accounts[{index}] must be an object")
        unknown = set(row) - {"id", "profile", "enabled"}
        if unknown:
            raise ValueError(
                f"accounts[{index}] contains unsupported fields {sorted(unknown)}; "
                "credentials belong only in .env"
            )
        enabled = row.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"accounts[{index}].enabled must be boolean")
        if not enabled:
            continue
        if "id" not in row:
            raise ValueError(f"accounts[{index}] is missing id")
        account_id = _safe_account_id(row["id"])
        raw_profile = row.get("profile")
        if raw_profile is None:
            profile = profile_root / f"account-{account_id}"
        elif isinstance(raw_profile, str) and raw_profile:
            profile = Path(raw_profile)
        else:
            raise ValueError(f"accounts[{index}].profile must be a non-empty path string")
        resolved_profile = profile.resolve()
        if account_id in seen_ids:
            raise ValueError(f"duplicate account id {account_id}")
        if resolved_profile in seen_profiles:
            raise ValueError(f"duplicate Chromium profile path {profile}")
        seen_ids.add(account_id)
        seen_profiles.add(resolved_profile)
        accounts.append(FleetAccount(account_id=account_id, profile=profile))
    if not accounts:
        raise ValueError("accounts file has no enabled accounts")
    return accounts


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class FleetSupervisor:
    """Launch, relay, restart, and observe process-per-account collectors."""

    def __init__(self, args: argparse.Namespace, accounts: list[FleetAccount]) -> None:
        self.args = args
        self.backend = FilesystemCoordinationBackend(args.coordination_dir)
        initial = time.monotonic()
        self.workers = [
            ManagedCollector(
                account=account,
                next_start_monotonic=initial + index * args.startup_stagger_seconds,
            )
            for index, account in enumerate(accounts)
        ]
        self.output_lines: queue.SimpleQueue[tuple[str, str | None]] = queue.SimpleQueue()
        self.stop_requested = False
        self.stop_reason: str | None = None
        self._last_heartbeat_monotonic = 0.0

    def _collector_command(self, account: FleetAccount) -> list[str]:
        command = [
            self.args.collector_python,
            "-u",
            str(Path(__file__).with_name("dgames_collect.py")),
            "--account-id",
            account.account_id,
            "--coordination-dir",
            str(self.args.coordination_dir),
            "--profile",
            str(account.profile),
            "--raw-root",
            str(self.args.raw_root),
        ]
        command.extend(self.args.collector_arg)
        return command

    def _relay_output(self, account_id: str, stream: Any) -> None:
        try:
            for line in iter(stream.readline, ""):
                self.output_lines.put((account_id, line.rstrip("\n")))
        finally:
            try:
                stream.close()
            except Exception:
                pass
            self.output_lines.put((account_id, None))

    def _start_worker(self, worker: ManagedCollector) -> None:
        command = self._collector_command(worker.account)
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            worker.process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            worker.last_exit_code = None
            worker.last_exit_monotonic = time.monotonic()
            worker.next_start_monotonic = time.monotonic() + self.args.restart_delay_seconds
            print(
                f"fleet: could not start account {worker.account.account_id}; "
                f"retrying in {self.args.restart_delay_seconds:.1f}s: {type(error).__name__}",
                flush=True,
            )
            return
        assert worker.process.stdout is not None
        worker.starts += 1
        worker.reader = threading.Thread(
            target=self._relay_output,
            args=(worker.account.account_id, worker.process.stdout),
            name=f"dgames-output-{worker.account.account_id}",
            daemon=True,
        )
        worker.reader.start()
        print(
            f"fleet: started account {worker.account.account_id} pid={worker.process.pid} "
            f"(start {worker.starts})",
            flush=True,
        )

    def _drain_output(self) -> None:
        while True:
            try:
                account_id, line = self.output_lines.get_nowait()
            except queue.Empty:
                return
            if line:
                print(f"[account {account_id}] {line}", flush=True)

    def _inspect_exits(self, now: float) -> None:
        for worker in self.workers:
            process = worker.process
            if process is None:
                if not self.stop_requested and now >= worker.next_start_monotonic:
                    self._start_worker(worker)
                continue
            exit_code = process.poll()
            if exit_code is None:
                continue
            worker.last_exit_code = exit_code
            worker.last_exit_monotonic = now
            worker.process = None
            print(
                f"fleet: account {worker.account.account_id} exited code={exit_code}",
                flush=True,
            )
            if not self.stop_requested:
                worker.next_start_monotonic = now + self.args.restart_delay_seconds
                print(
                    f"fleet: restarting account {worker.account.account_id} in "
                    f"{self.args.restart_delay_seconds:.1f}s",
                    flush=True,
                )

    @staticmethod
    def _number(document: Mapping[str, object], key: str) -> float:
        value = document.get(key, 0)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return 0.0

    def _heartbeat(self, now: float, *, status: str) -> None:
        now_epoch = time.time()
        account_ids = [worker.account.account_id for worker in self.workers]
        try:
            documents = self.backend.read_account_heartbeats(account_ids=account_ids)
        except CoordinationUnavailable as error:
            documents = {}
            print(f"fleet: could not read collector heartbeats: {type(error).__name__}", flush=True)

        total_games_per_hour = 0.0
        claim_attempts = 0.0
        claim_contentions = 0.0
        tables_probed = 0.0
        skipped_non_base = 0.0
        completed = 0.0
        health: dict[str, str] = {}
        for worker in self.workers:
            account_id = worker.account.account_id
            document = documents.get(account_id)
            process_live = worker.process is not None and worker.process.poll() is None
            if document is None:
                state = "missing" if process_live else "down"
            else:
                age = max(0.0, now_epoch - self._number(document, "updated_at_epoch"))
                child_status = str(document.get("status", "unknown"))
                if not process_live:
                    state = "down"
                elif age > self.args.heartbeat_stale_seconds:
                    state = "stale"
                else:
                    state = child_status
                total_games_per_hour += self._number(document, "games_per_hour")
                claim_attempts += self._number(document, "claim_attempts")
                claim_contentions += self._number(document, "claim_contentions")
                tables_probed += self._number(document, "tables_probed")
                skipped_non_base += self._number(document, "skipped_non_base")
                completed += self._number(document, "captured_completed")
            health[account_id] = state
        contention_rate = (
            None if claim_attempts == 0 else claim_contentions / claim_attempts
        )
        non_base_skip_rate = None if tables_probed == 0 else skipped_non_base / tables_probed
        document: dict[str, object] = {
            "schema_version": 1,
            "status": status,
            "supervisor_pid": os.getpid(),
            "started_accounts": sum(worker.starts for worker in self.workers),
            "completed_visible": int(completed),
            "total_games_per_hour": total_games_per_hour,
            "claim_attempts": int(claim_attempts),
            "claim_contentions": int(claim_contentions),
            "claim_contention_rate": contention_rate,
            "tables_probed": int(tables_probed),
            "skipped_non_base": int(skipped_non_base),
            "non_base_skip_rate": non_base_skip_rate,
            "per_account_health": health,
            "heartbeat_at_utc": _utc_now(),
        }
        try:
            self.backend.write_fleet_heartbeat(document=document)
        except CoordinationUnavailable as error:
            print(f"fleet: could not write fleet heartbeat: {type(error).__name__}", flush=True)

        claim_text = "n/a" if contention_rate is None else f"{contention_rate:.1%}"
        non_base_text = "n/a" if non_base_skip_rate is None else f"{non_base_skip_rate:.1%}"
        health_text = ",".join(f"{account_id}:{state}" for account_id, state in health.items())
        print(
            "fleet heartbeat: "
            f"games/hour={total_games_per_hour:.2f} completed={int(completed)} "
            f"claim-contention={claim_text} non-base-skip={non_base_text} "
            f"accounts={health_text}",
            flush=True,
        )
        self._last_heartbeat_monotonic = now

    def request_stop(self, reason: str) -> None:
        if self.stop_requested:
            return
        self.stop_requested = True
        self.stop_reason = reason
        print(f"fleet: shutdown requested ({reason})", flush=True)

    def _graceful_shutdown(self) -> None:
        live = [worker for worker in self.workers if worker.process and worker.process.poll() is None]
        for worker in live:
            assert worker.process is not None
            try:
                worker.process.send_signal(signal.SIGINT)
            except OSError:
                pass
        deadline = time.monotonic() + self.args.shutdown_timeout_seconds
        while live and time.monotonic() < deadline:
            self._drain_output()
            self._inspect_exits(time.monotonic())
            live = [worker for worker in live if worker.process and worker.process.poll() is None]
            time.sleep(0.1)
        for worker in live:
            assert worker.process is not None
            print(f"fleet: terminating account {worker.account.account_id} after grace timeout", flush=True)
            try:
                worker.process.terminate()
            except OSError:
                pass
        terminate_deadline = time.monotonic() + 5.0
        while live and time.monotonic() < terminate_deadline:
            self._drain_output()
            live = [worker for worker in live if worker.process and worker.process.poll() is None]
            time.sleep(0.1)
        for worker in live:
            assert worker.process is not None
            print(f"fleet: killing unresponsive account {worker.account.account_id}", flush=True)
            try:
                worker.process.kill()
            except OSError:
                pass
        self._drain_output()

    def run(self) -> int:
        end_monotonic = (
            None
            if self.args.run_seconds == 0
            else time.monotonic() + self.args.run_seconds
        )
        previous_handlers: dict[signal.Signals, Any] = {}

        def handle_signal(signum: int, _frame: object) -> None:
            self.request_stop(signal.Signals(signum).name)

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_signal)
        try:
            while not self.stop_requested:
                now = time.monotonic()
                if end_monotonic is not None and now >= end_monotonic:
                    self.request_stop("configured run duration")
                    break
                self._drain_output()
                self._inspect_exits(now)
                if now - self._last_heartbeat_monotonic >= self.args.heartbeat_interval:
                    self._heartbeat(now, status="running")
                time.sleep(0.1)
        finally:
            self._graceful_shutdown()
            self._heartbeat(time.monotonic(), status="stopped")
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)
        return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--accounts-file", type=Path, required=True)
    parser.add_argument("--coordination-dir", type=Path, default=DEFAULT_COORDINATION_DIR)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument("--collector-python", default=sys.executable)
    parser.add_argument(
        "--collector-arg",
        action="append",
        default=[],
        help="one additional non-identity argument passed to every collector",
    )
    parser.add_argument(
        "--startup-stagger-seconds",
        type=float,
        default=DEFAULT_STARTUP_STAGGER_SECONDS,
    )
    parser.add_argument(
        "--restart-delay-seconds",
        type=float,
        default=DEFAULT_RESTART_DELAY_SECONDS,
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--heartbeat-stale-seconds",
        type=float,
        default=DEFAULT_HEARTBEAT_STALE_SECONDS,
    )
    parser.add_argument(
        "--shutdown-timeout-seconds",
        type=float,
        default=DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=0.0,
        help="0 runs until SIGINT/SIGTERM; useful for offline supervisor smoke checks",
    )
    args = parser.parse_args(argv)
    if args.startup_stagger_seconds < 0:
        parser.error("--startup-stagger-seconds cannot be negative")
    for name in (
        "restart_delay_seconds",
        "heartbeat_interval",
        "heartbeat_stale_seconds",
        "shutdown_timeout_seconds",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.run_seconds < 0:
        parser.error("--run-seconds cannot be negative")
    forbidden_prefixes = (
        "--account-id",
        "--coordination-dir",
        "--profile",
        "--raw-root",
        "--self-test",
    )
    for value in args.collector_arg:
        if any(value == prefix or value.startswith(f"{prefix}=") for prefix in forbidden_prefixes):
            parser.error(f"--collector-arg {value!r} would break fleet account isolation")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        accounts = _load_accounts(args.accounts_file, profile_root=args.profile_root)
        supervisor = FleetSupervisor(args, accounts)
    except (CoordinationUnavailable, ValueError) as error:
        print(f"fleet setup failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    print(
        f"fleet: supervising {len(accounts)} account(s); startup stagger "
        f"{args.startup_stagger_seconds:.1f}s; coordination={args.coordination_dir}",
        flush=True,
    )
    try:
        return supervisor.run()
    except KeyboardInterrupt:
        supervisor.request_stop("KeyboardInterrupt")
        return 0


if __name__ == "__main__":
    sys.exit(main())
