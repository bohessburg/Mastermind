"""Measure the base-set-only yield among live dominion.games tables.

This is a deliberately small, polite one-off sampler.  It reuses the logged-in
``dgames-profile/`` Chromium profile and the arena browser transport, requests
one lobby snapshot, then spectates a randomized sequential sample of eligible
human two-player games.  It never sends a login or credential message.

Usage::

    ./.venv/bin/python scripts/dgames_yield_sample.py --limit 150
    ./.venv/bin/python scripts/dgames_yield_sample.py --self-test

Use ``--headful`` if Chromium needs visible UI for a local troubleshooting
run.  Results are written below ``data/dominion_games/recon/yield/``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any, TypeVar


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v2.arena.archive import serialize_frame_record  # noqa: E402
from src.v2.arena.browser.session import ArenaSession  # noqa: E402
from src.v2.arena.protocol.events import FullState  # noqa: E402
from src.v2.arena.protocol.frames import (  # noqa: E402
    Direction,
    ProtocolError,
    Reader,
    Writer,
)
from src.v2.arena.protocol.parser import ArenaParser  # noqa: E402
from src.v2.arena.protocol.recording import decode_record_binary  # noqa: E402


DEFAULT_PROFILE = Path("dgames-profile")
DEFAULT_OUTPUT_ROOT = Path("data/dominion_games/recon/yield")
DEFAULT_CARD_MAP = Path("data/dominion_games/recon/card_id_map.json")
DEFAULT_SELF_TEST_CAPTURE = Path(
    "data/dominion_games/recon/captures/20260731T220433.671341Z/frames.jsonl"
)
DEFAULT_URL = "https://dominion.games"

REQUEST_UPDATE = 11
UPDATE_TABLES = 3
JOIN_TABLE = 2
LEAVE_TABLE = 6
TABLES_OVERVIEW = 3
RUNNING = 2

HARD_SAMPLE_LIMIT = 150
MIN_DELAY_SECONDS = 0.5
# How long to wait for a tablesOverview before re-sending REQUEST_UPDATE.
TABLES_RETRY_SECONDS = 6.0
MAX_FAILURE_BACKOFF_SECONDS = 30.0
WAIT_HEARTBEAT_SECONDS = 5.0

STATUS_NAMES = {
    0: "NEW",
    1: "POST_GAME",
    2: "RUNNING",
    3: "ABANDONED",
    4: "TRANSFERRED",
}

# These are the login-family ordinal ids documented in dgams_capture.py and
# RECON.md.  The sampler never sends them; this list only prevents a reused
# browser session from persisting a session token if the client reauthenticates.
SENSITIVE_OUTBOUND_TYPES = frozenset({1, 13, 14, 15, 16, 19, 24, 44, 45, 46})


@dataclass(frozen=True)
class TableSummary:
    """One ``tablesOverview`` row, decoded in the server's wire order."""

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
        """Mirror the Dominion client helper used to identify 2-player tables."""
        return self.min_players < 3 and self.max_players > 1


@dataclass(frozen=True)
class SampleResult:
    """The persisted outcome from one attempted spectator join."""

    table_id: int
    game_id: int | None
    ok: bool
    failure_reason: str | None
    card_names: tuple[str, ...]
    base_only: bool | None
    foreign_names: tuple[str, ...]
    leave_failed: bool = False


class StopRequested(Exception):
    """The user asked the sampler to stop at a safe table boundary."""


T = TypeVar("T")


def decode_tables_overview(payload: bytes) -> tuple[TableSummary, ...]:
    """Decode one inbound message-3 payload and reject trailing bytes.

    ``TableSummary.start_time`` is present only for ``RUNNING`` tables; this is
    the conditional field that makes a generic dataclass decoder unsafe here.
    """
    reader = Reader(payload)
    count = reader.s32()
    if count < 0:
        raise ProtocolError(f"negative tablesOverview count {count}")

    summaries: list[TableSummary] = []
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
        summaries.append(
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
    return tuple(summaries)


def _record_message_type(record: dict[str, Any]) -> int | None:
    """Return a captured frame's message type without exposing its payload."""
    if record.get("kind") != "binary" or not record.get("b64"):
        return None
    try:
        raw = base64.b64decode(str(record.get("data", "")), validate=True)
    except (ValueError, TypeError):
        return None
    offset = 4 if record.get("dir") == "in" else 0
    if len(raw) < offset + 4:
        return None
    return int.from_bytes(raw[offset : offset + 4], "big")


class DgamesYieldSession(ArenaSession):
    """Arena transport with a fixed measurement directory and safe archive."""

    def __init__(self, *, measurement_dir: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.measurement_dir = measurement_dir

    def _create_run_dir(self) -> Path:
        """Use the caller-created yield directory, without a nested timestamp."""
        return self.measurement_dir

    def _receive_frame(self, record: dict[str, Any]) -> None:
        """Archive a redacted copy while retaining frames in memory for parsing.

        ``loginSuccess`` itself contains the persisted session id after the
        player id.  Keeping its original bytes only on the in-process queue is
        necessary to obtain that id, while ensuring neither direction of the
        login exchange ends up in ``frames.jsonl``.
        """
        if self._stopped:
            return
        copied = dict(record)
        archive_record = dict(record)
        msg_type = _record_message_type(archive_record)
        if (
            archive_record.get("dir") == "in" and msg_type == 2
        ) or (
            archive_record.get("dir") == "out"
            and msg_type in SENSITIVE_OUTBOUND_TYPES
        ):
            archive_record["data"] = ""
            archive_record["b64"] = False
            archive_record["redacted"] = True
            archive_record["msg_type"] = msg_type
        if self._frames_file is not None:
            self._frames_file.write(serialize_frame_record(archive_record))
            self._frames_file.flush()
        self.frame_queue.put_nowait(copied)


class LiveProtocolPump:
    """Single consumer of an :class:`ArenaSession` raw-frame queue."""

    def __init__(self, session: ArenaSession) -> None:
        self.session = session
        self.parser = ArenaParser()
        self.socket_open = False
        self.tables_overviews: list[tuple[TableSummary, ...]] = []
        self.full_states: list[FullState] = []

    def drain(self) -> None:
        """Process every currently queued record before starting a new action."""
        while True:
            try:
                record = self.session.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._process_record(record)

    def _process_record(self, record: dict[str, Any] | None) -> None:
        if record is None:
            raise RuntimeError("game websocket closed")
        if record.get("sock") != self.session.game_socket:
            return
        kind = record.get("kind")
        if kind == "open":
            self.socket_open = True
            return
        if kind == "close":
            self.socket_open = False
            return
        if kind != "binary":
            return

        self.socket_open = True
        frame = decode_record_binary(record)
        if frame is None:
            return
        if frame.direction is Direction.INBOUND and frame.msg_type == TABLES_OVERVIEW:
            self.tables_overviews.append(decode_tables_overview(frame.payload))
        for event in self.parser.parse_frame(frame):
            if isinstance(event, FullState):
                self.full_states.append(event)

    async def wait_for(
        self,
        ready: Callable[[], T | None],
        *,
        timeout: float,
        label: str,
        stop_requested: asyncio.Event,
    ) -> T:
        """Wait for one predicate while emitting a periodic no-silence update."""
        deadline = time.monotonic() + timeout
        next_heartbeat = time.monotonic() + WAIT_HEARTBEAT_SECONDS
        while True:
            value = ready()
            if value is not None:
                return value
            if stop_requested.is_set():
                raise StopRequested()

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {label} after {timeout:.1f}s")
            try:
                record = await asyncio.wait_for(
                    self.session.frame_queue.get(), timeout=min(1.0, remaining)
                )
            except TimeoutError:
                if time.monotonic() >= next_heartbeat:
                    print(f"waiting for {label}...", flush=True)
                    next_heartbeat += WAIT_HEARTBEAT_SECONDS
                continue
            self._process_record(record)


def _unique_run_dir(output_root: Path) -> Path:
    """Reserve a UTC-named output directory for this measurement."""
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    candidate = output_root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{timestamp}-{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _load_base_names(path: Path) -> frozenset[str]:
    """Load the authoritative 33-name supported-base roster."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        names = frozenset(str(entry["name"]) for entry in document["map"])
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not load base card map {path}: {error}") from error
    if len(names) != 33:
        raise RuntimeError(f"expected 33 distinct base names in {path}, got {len(names)}")
    return names


def _status_name(status: int) -> str:
    return STATUS_NAMES.get(status, f"UNKNOWN_{status}")


def _failure_reason(error: BaseException) -> str:
    detail = str(error)
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def _write_result(destination: Any, result: SampleResult) -> None:
    """Append one durable JSONL result, including failed join attempts."""
    record = {
        "table_id": result.table_id,
        "game_id": result.game_id,
        "ok": result.ok,
        "failure_reason": result.failure_reason,
        "card_names": list(result.card_names),
        "base_only": result.base_only,
        "foreign_names": list(result.foreign_names),
    }
    destination.write(json.dumps(record, sort_keys=True) + "\n")
    destination.flush()


async def _sleep_with_stop(delay: float, stop_requested: asyncio.Event) -> None:
    """Apply a politeness pause while still honoring Ctrl-C promptly."""
    try:
        await asyncio.wait_for(stop_requested.wait(), timeout=delay)
    except TimeoutError:
        return
    raise StopRequested()


async def _sample_table(
    *,
    session: ArenaSession,
    pump: LiveProtocolPump,
    table: TableSummary,
    player_id: int,
    base_names: frozenset[str],
    join_timeout: float,
    stop_requested: asyncio.Event,
) -> SampleResult:
    """Spectate one table, read its full state, and always issue LEAVE_TABLE."""
    join_attempted = False
    failure_reason: str | None = None
    full_state: FullState | None = None
    leave_failed = False

    try:
        # Do not mistake an already-buffered full-state for this new join.
        pump.drain()
        full_state_count = len(pump.full_states)
        join_attempted = True
        await session.send_frame(
            JOIN_TABLE,
            Writer().u64(table.table_id).boolean(False).build(),
        )
        full_state = await pump.wait_for(
            lambda: (
                pump.full_states[full_state_count]
                if len(pump.full_states) > full_state_count
                else None
            ),
            timeout=join_timeout,
            label=f"fullGameState for table {table.table_id}",
            stop_requested=stop_requested,
        )
    except StopRequested:
        # Persist the in-flight attempt after the mandatory leave below.  The
        # outer loop sees the same signal and exits without attempting another
        # join, so Ctrl-C never turns into an unrecorded table attachment.
        failure_reason = "interrupted by user signal"
    except Exception as error:
        failure_reason = _failure_reason(error)
    finally:
        if join_attempted:
            try:
                await session.send_frame(
                    LEAVE_TABLE,
                    Writer().u64(table.table_id).s32(player_id).build(),
                )
            except Exception as error:
                leave_failed = True
                leave_reason = f"leave failed: {_failure_reason(error)}"
                failure_reason = (
                    leave_reason
                    if failure_reason is None
                    else f"{failure_reason}; {leave_reason}"
                )

    if full_state is None or failure_reason is not None:
        return SampleResult(
            table_id=table.table_id,
            game_id=None if full_state is None else full_state.game_id,
            ok=False,
            failure_reason=failure_reason or "fullGameState was not received",
            card_names=(),
            base_only=None,
            foreign_names=(),
            leave_failed=leave_failed,
        )

    card_names = tuple(sorted(name for name, count in full_state.card_counts if count > 0))
    foreign_names = tuple(name for name in card_names if name not in base_names)
    return SampleResult(
        table_id=table.table_id,
        game_id=full_state.game_id,
        ok=True,
        failure_reason=None,
        card_names=card_names,
        base_only=not foreign_names,
        foreign_names=foreign_names,
        leave_failed=False,
    )


def run_self_test(capture_path: Path = DEFAULT_SELF_TEST_CAPTURE) -> None:
    """Validate the tables decoder against all three recorded live snapshots."""
    expected_counts = (340, 347, 348)
    decoded: list[tuple[TableSummary, ...]] = []
    try:
        source = capture_path.open(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"could not open self-test capture {capture_path}: {error}") from error

    with source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{capture_path}:{line_number}: invalid JSON") from error
            if (
                record.get("kind") != "binary"
                or record.get("dir") != "in"
                or not record.get("b64")
            ):
                continue
            frame = decode_record_binary(record, source=f"{capture_path}:{line_number}")
            if frame is None or frame.msg_type != TABLES_OVERVIEW:
                continue
            decoded.append(decode_tables_overview(frame.payload))

    counts = tuple(len(snapshot) for snapshot in decoded)
    if counts != expected_counts:
        raise AssertionError(f"expected table counts {expected_counts}, got {counts}")
    for snapshot in decoded:
        if not all(6 <= table.table_id <= 1414 for table in snapshot):
            raise AssertionError("table ids fell outside observed 6..1414 range")
        if not all(0 <= table.status <= 4 for table in snapshot):
            raise AssertionError("encountered a table status outside 0..4")
        if not all(2 <= table.min_players <= table.max_players <= 6 for table in snapshot):
            raise AssertionError("encountered implausible minPlayers/maxPlayers")
    print(
        "self-test passed: decoded tablesOverview snapshots "
        + ", ".join(str(count) for count in counts)
        + " with no trailing bytes",
        flush=True,
    )


def _build_summary(
    *,
    started_at: datetime,
    duration: float,
    total_tables: int,
    status_counts: Counter[str],
    observable_two_player_candidates: int,
    excluded_bots: int,
    human_candidates: int,
    sampled: int,
    successful: int,
    base_only: int,
    failures: int,
    foreign_counts: Counter[str],
    outcome: str,
    abort_reason: str | None,
) -> dict[str, object]:
    """Construct the stable, machine-readable run summary."""
    counts_by_status = {
        _status_name(status): status_counts.get(_status_name(status), 0)
        for status in sorted(STATUS_NAMES)
    }
    counts_by_status.update(
        {
            name: count
            for name, count in status_counts.items()
            if name not in counts_by_status
        }
    )
    return {
        "started_at_utc": started_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "outcome": outcome,
        "abort_reason": abort_reason,
        "total_tables_in_snapshot": total_tables,
        "counts_by_status": counts_by_status,
        "two_player_running_observable_candidates": observable_two_player_candidates,
        "excluded_bots": excluded_bots,
        "human_two_player_running_observable_candidates": human_candidates,
        "number_sampled": sampled,
        "number_successfully_read": successful,
        "base_only_count": base_only,
        "base_only_rate": (base_only / successful) if successful else None,
        "failures": failures,
        "most_common_foreign_names": [
            {"name": name, "count": count}
            for name, count in sorted(
                foreign_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "wall_clock_duration_seconds": round(duration, 3),
    }


async def _run(args: argparse.Namespace) -> int:
    """Run one bounded sampler invocation and persist partial results on exit."""
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    run_dir = _unique_run_dir(args.out)
    games_path = run_dir / "games.jsonl"
    summary_path = run_dir / "summary.json"
    base_names = _load_base_names(args.card_map)

    session = DgamesYieldSession(
        measurement_dir=run_dir,
        output_root=args.out,
        profile_dir=args.profile,
        url=args.url,
        headless=not args.headful,
    )
    pump = LiveProtocolPump(session)
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_requested.set)
            installed_signals.append(signum)
        except (NotImplementedError, RuntimeError):
            pass

    total_tables = 0
    status_counts: Counter[str] = Counter()
    observable_two_player_candidates = 0
    excluded_bots = 0
    human_candidates = 0
    sampled = 0
    successful = 0
    base_only = 0
    failures = 0
    foreign_counts: Counter[str] = Counter()
    outcome = "completed"
    abort_reason: str | None = None
    games_file: Any | None = None

    print(f"yield run: {run_dir.resolve()}", flush=True)
    print(
        f"launching authenticated browser ({'headful' if args.headful else 'headless'})...",
        flush=True,
    )
    try:
        games_file = games_path.open("a", encoding="utf-8", buffering=1)
        await session.start()
        print("waiting for the game websocket and login success...", flush=True)
        player_id = await pump.wait_for(
            lambda: (
                pump.parser.our_player_id
                if pump.socket_open and pump.parser.our_player_id is not None
                else None
            ),
            timeout=args.startup_timeout,
            label="game websocket login",
            stop_requested=stop_requested,
        )
        print(f"authenticated as player {player_id}; requesting table snapshot...", flush=True)
        pump.drain()
        overview_count = len(pump.tables_overviews)
        # A single REQUEST_UPDATE is unreliable: if it lands before the page's
        # own lobby subscription has settled, the server simply never answers
        # and there is no error to observe.  Re-ask on a slow cadence until a
        # snapshot arrives, bounded by the startup timeout.
        snapshot = None
        deadline = asyncio.get_running_loop().time() + args.startup_timeout
        attempt = 0
        while snapshot is None:
            attempt += 1
            await session.send_frame(REQUEST_UPDATE, Writer().s32(UPDATE_TABLES).build())
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"no tablesOverview after {attempt} REQUEST_UPDATE attempts "
                    f"in {args.startup_timeout:.0f}s"
                )
            try:
                snapshot = await pump.wait_for(
                    lambda: (
                        pump.tables_overviews[overview_count]
                        if len(pump.tables_overviews) > overview_count
                        else None
                    ),
                    timeout=min(TABLES_RETRY_SECONDS, remaining),
                    label="tablesOverview",
                    stop_requested=stop_requested,
                )
            except TimeoutError:
                print(
                    f"no tablesOverview yet (attempt {attempt}); re-requesting...",
                    flush=True,
                )

        total_tables = len(snapshot)
        status_counts.update(_status_name(table.status) for table in snapshot)
        observable_two_player = [
            table
            for table in snapshot
            if table.status == RUNNING and table.is_observable and table.is_two_player
        ]
        observable_two_player_candidates = len(observable_two_player)
        excluded_bots = sum(table.bots > 0 for table in observable_two_player)
        candidates = [table for table in observable_two_player if table.bots == 0]
        human_candidates = len(candidates)
        target = min(args.limit, len(candidates))
        selected = random.SystemRandom().sample(candidates, target)
        print(
            f"snapshot: {total_tables} tables; {observable_two_player_candidates} "
            f"running/observable 2p; excluded {excluded_bots} with bots; "
            f"sampling {target}.",
            flush=True,
        )

        consecutive_failures = 0
        for table in selected:
            if stop_requested.is_set():
                raise StopRequested()
            print(
                f"sampling {sampled + 1}/{target}: table {table.table_id} "
                f"(base-only={base_only}, failures={failures})",
                flush=True,
            )
            result = await _sample_table(
                session=session,
                pump=pump,
                table=table,
                player_id=player_id,
                base_names=base_names,
                join_timeout=args.join_timeout,
                stop_requested=stop_requested,
            )
            sampled += 1
            _write_result(games_file, result)

            if result.ok:
                successful += 1
                consecutive_failures = 0
                if result.base_only:
                    base_only += 1
                # Count each foreign card once per game: split piles (Amphora,
                # Doubloons, ...) appear twice in card_counts and would
                # otherwise outrank ordinary cards in the tally.
                foreign_counts.update(set(result.foreign_names))
                print(
                    f"read game {result.game_id}: "
                    f"{'base-only' if result.base_only else 'has foreign cards'}; "
                    f"{len(result.card_names)} card types.",
                    flush=True,
                )
            else:
                failures += 1
                consecutive_failures += 1
                print(
                    f"table {table.table_id} failed: {result.failure_reason}",
                    flush=True,
                )
                if result.leave_failed:
                    outcome = "aborted"
                    abort_reason = "LEAVE_TABLE failed; refusing to join another table"
                    break

            if stop_requested.is_set():
                print(
                    f"heartbeat {sampled}/{target}: base-only={base_only}, "
                    f"failures={failures}",
                    flush=True,
                )
                raise StopRequested()

            print(
                f"heartbeat {sampled}/{target}: base-only={base_only}, failures={failures}",
                flush=True,
            )
            if consecutive_failures >= args.max_consecutive_failures:
                outcome = "aborted"
                abort_reason = (
                    f"{consecutive_failures} consecutive failures "
                    "reached the configured safety limit"
                )
                print(f"aborting: {abort_reason}", flush=True)
                break
            if sampled < target:
                delay = args.delay
                if consecutive_failures:
                    delay = min(
                        MAX_FAILURE_BACKOFF_SECONDS,
                        args.delay * (2 ** (consecutive_failures - 1)),
                    )
                    print(
                        f"failure backoff: waiting {delay:.1f}s before the next table...",
                        flush=True,
                    )
                else:
                    print(f"politeness delay: waiting {delay:.1f}s...", flush=True)
                await _sleep_with_stop(delay, stop_requested)
    except StopRequested:
        outcome = "interrupted"
        abort_reason = "interrupted by user signal"
        print("interrupt received; stopping after leaving the current table.", flush=True)
    except asyncio.CancelledError:
        outcome = "interrupted"
        abort_reason = "sampler task cancelled"
        print("sampler cancelled; stopping after leaving the current table.", flush=True)
    except Exception as error:
        outcome = "failed"
        abort_reason = _failure_reason(error)
        print(f"sampler failed: {abort_reason}", file=sys.stderr, flush=True)
    finally:
        if games_file is not None:
            games_file.close()
        await session.stop()
        for signum in installed_signals:
            loop.remove_signal_handler(signum)
        summary = _build_summary(
            started_at=started_at,
            duration=time.monotonic() - started_monotonic,
            total_tables=total_tables,
            status_counts=status_counts,
            observable_two_player_candidates=observable_two_player_candidates,
            excluded_bots=excluded_bots,
            human_candidates=human_candidates,
            sampled=sampled,
            successful=successful,
            base_only=base_only,
            failures=failures,
            foreign_counts=foreign_counts,
            outcome=outcome,
            abort_reason=abort_reason,
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rate = "n/a" if successful == 0 else f"{base_only / successful:.1%}"
        print(
            f"done: outcome={outcome}; sampled={sampled}; read={successful}; "
            f"base-only={base_only} ({rate}); failures={failures}; "
            f"duration={summary['wall_clock_duration_seconds']}s",
            flush=True,
        )
        print(f"summary: {summary_path.resolve()}", flush=True)
    return 0 if outcome in {"completed", "interrupted"} else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--card-map", type=Path, default=DEFAULT_CARD_MAP)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--limit", type=int, default=HARD_SAMPLE_LIMIT)
    parser.add_argument("--join-timeout", type=float, default=8.0)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument(
        "--headful",
        action="store_true",
        help="launch Chromium with visible UI instead of the default headless mode",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="offline-validate the tablesOverview decoder against the saved capture",
    )
    parser.add_argument(
        "--self-test-capture",
        type=Path,
        default=DEFAULT_SELF_TEST_CAPTURE,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= HARD_SAMPLE_LIMIT:
        parser.error(f"--limit must be between 1 and {HARD_SAMPLE_LIMIT}")
    if args.join_timeout <= 0:
        parser.error("--join-timeout must be greater than zero")
    if args.delay < MIN_DELAY_SECONDS:
        parser.error(f"--delay must be at least {MIN_DELAY_SECONDS:.1f} seconds")
    if args.max_consecutive_failures < 1:
        parser.error("--max-consecutive-failures must be at least one")
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be greater than zero")
    return args


def main(argv: list[str] | None = None) -> int:
    """CLI entry point used by the repository virtual environment."""
    args = _parse_args(argv)
    if args.self_test:
        try:
            run_self_test(args.self_test_capture)
        except Exception as error:
            print(f"self-test failed: {_failure_reason(error)}", file=sys.stderr, flush=True)
            return 1
        return 0
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        # Platforms without loop-level signal handlers still get a clean
        # browser shutdown through asyncio.run's cancellation path.
        return 0


if __name__ == "__main__":
    sys.exit(main())
