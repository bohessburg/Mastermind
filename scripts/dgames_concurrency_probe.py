"""Empirically determine safe single-account dominion.games spectator concurrency.

This is deliberately a small, polite architecture probe, not a collector.  It
uses one already-authenticated persistent Chromium profile, requests fresh
lobby snapshots with the known retry-on-silence behavior, and touches at most
four running human two-player tables:

1. Join table A on one websocket, then table B without leaving A.  The probe
   retains game-log index evidence from traffic after the second join.
2. In the *same* browser context, hold one spectate join per each of two pages
   and watch for command failures, reconnect instructions, socket closes, and
   page-specific live traffic.

It never sends a login message, never asks for credentials, and redacts any
browser reauthentication frames before writing its evidence archive.  Every
joined table is sent LEAVE_TABLE during cleanup, including after failures.

Usage::

    PYTHONUNBUFFERED=1 ./.venv/bin/python scripts/dgames_concurrency_probe.py
    PYTHONUNBUFFERED=1 ./.venv/bin/python scripts/dgames_concurrency_probe.py \
        --observe-seconds 30 --headful

The durable result is ``summary.json`` in the printed output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import time
from typing import Any, TypeVar


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v2.arena.archive import serialize_frame_record  # noqa: E402
from src.v2.arena.browser.session import ArenaSession  # noqa: E402
from src.v2.arena.protocol.events import FullState  # noqa: E402
from src.v2.arena.protocol.frames import Direction, ProtocolError, Reader, Writer  # noqa: E402
from src.v2.arena.protocol.parser import ArenaParser  # noqa: E402
from src.v2.arena.protocol.recording import decode_record_binary  # noqa: E402


DEFAULT_PROFILE = Path("dgames-profile")
DEFAULT_OUTPUT_ROOT = Path("data/dominion_games/recon/concurrency")
DEFAULT_URL = "https://dominion.games"
DEFAULT_SELF_TEST_CAPTURE = Path(
    "data/dominion_games/recon/captures/20260731T220433.671341Z/frames.jsonl"
)

REQUEST_UPDATE = 11
UPDATE_TABLES = 3
JOIN_TABLE = 2
LEAVE_TABLE = 6
TABLES_OVERVIEW = 3
GAME_EVENT_INFO = 32
GAME_LOG_INFO = 33
COMMAND_FAILED = 1
INSTRUCT_TO_RECONNECT = 39
RUNNING = 2

TABLES_RETRY_SECONDS = 6.0
WAIT_HEARTBEAT_SECONDS = 5.0
MIN_JOIN_DELAY_SECONDS = 0.5
# Same conservative belt-and-braces list used by the existing capture tools.
SENSITIVE_OUTBOUND_TYPES = frozenset({1, 13, 14, 15, 16, 19, 24, 44, 45, 46})


@dataclass(frozen=True)
class TableSummary:
    """One tablesOverview row, decoded in the server's wire order."""

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
class LogHeader:
    """A non-sensitive gameLogInfo identity proxy (its global log interval)."""

    start_index: int
    count: int

    @property
    def end_index(self) -> int:
        return self.start_index + self.count


@dataclass(frozen=True)
class JoinEvidence:
    table_id: int
    game_id: int | None
    full_state_received: bool
    error: str | None
    initial_log_headers: tuple[LogHeader, ...]


class StopRequested(Exception):
    """Raised at a safe point when the operator sends SIGINT or SIGTERM."""


T = TypeVar("T")


def decode_tables_overview(payload: bytes) -> tuple[TableSummary, ...]:
    """Decode a tablesOverview payload and reject trailing bytes."""
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


def _record_message_type(record: dict[str, Any]) -> int | None:
    """Read just a frame type; never decode or print its payload."""
    if record.get("kind") != "binary" or not record.get("b64"):
        return None
    try:
        raw = base64.b64decode(str(record.get("data", "")), validate=True)
    except (TypeError, ValueError):
        return None
    offset = 4 if record.get("dir") == "in" else 0
    if len(raw) < offset + 4:
        return None
    return int.from_bytes(raw[offset : offset + 4], "big")


def _redacted_archive_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a recorder-shaped record safe to persist for probe evidence."""
    archived = dict(record)
    msg_type = _record_message_type(archived)
    if (archived.get("dir") == "in" and msg_type == 2) or (
        archived.get("dir") == "out" and msg_type in SENSITIVE_OUTBOUND_TYPES
    ):
        archived["data"] = ""
        archived["b64"] = False
        archived["redacted"] = True
        archived["msg_type"] = msg_type
    return archived


class ProbeSession(ArenaSession):
    """ArenaSession with redacted evidence and opt-in page-tagged frame feeds.

    ``ws_hook.js`` gives each page its own socket-number sequence, so socket
    number alone cannot distinguish two pages.  After each page is settled we
    replace only its forwarding callback with a tiny tag-preserving wrapper.
    The hook itself, browser transport, and normal websocket behavior stay
    untouched.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tag_queues: dict[str, asyncio.Queue[dict[str, Any] | None]] = {}
        self._tag_binding_installed = False

    def _receive_frame(self, record: dict[str, Any]) -> None:
        if self._stopped:
            return
        copied = dict(record)
        if self._frames_file is not None:
            self._frames_file.write(serialize_frame_record(_redacted_archive_record(record)))
            self._frames_file.flush()
        self.frame_queue.put_nowait(copied)

    async def install_page_tag(self, page: Any, tag: str) -> asyncio.Queue[dict[str, Any] | None]:
        """Route future hook records from one settled page to a tagged queue."""
        if self.context is None:
            raise RuntimeError("start the probe session before tagging pages")
        if not self._tag_binding_installed:
            await self.context.expose_binding(
                "__dgamesProbeFrame", self._receive_tagged_frame
            )
            self._tag_binding_installed = True
        if tag in self.tag_queues:
            raise RuntimeError(f"page tag {tag!r} is already installed")
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.tag_queues[tag] = queue
        await page.evaluate(
            """(tag) => {
                window.__arenaFrame = (record) =>
                    window.__dgamesProbeFrame({ ...record, page_tag: tag });
                window.__dgamesProbePageTag = tag;
            }""",
            tag,
        )
        return queue

    async def send_frame_on_page(self, page: Any, msg_type: int, payload: bytes) -> None:
        """Send a raw protocol frame through a particular page's live socket."""
        raw = Writer().u32(msg_type).bytes(payload).build()
        encoded = base64.b64encode(raw).decode("ascii")
        await page.evaluate("(bytes) => window.__arenaSend(bytes)", encoded)

    def _receive_tagged_frame(self, _source: Any, record: dict[str, Any]) -> None:
        if self._stopped:
            return
        copied = dict(record)
        if self._frames_file is not None:
            self._frames_file.write(serialize_frame_record(_redacted_archive_record(record)))
            self._frames_file.flush()
        tag = str(copied.get("page_tag", ""))
        queue = self.tag_queues.get(tag)
        if queue is not None:
            queue.put_nowait(copied)

    async def stop(self) -> None:
        for queue in self.tag_queues.values():
            queue.put_nowait(None)
        await super().stop()


class FramePump:
    """Parse one raw browser frame stream and retain only probe-safe evidence."""

    def __init__(
        self,
        queue: asyncio.Queue[dict[str, Any] | None],
        *,
        socket: int | None,
        label: str,
    ) -> None:
        self.queue = queue
        self.socket = socket
        self.label = label
        self.parser = ArenaParser()
        self.socket_open = False
        self.socket_closes = 0
        self.tables_overviews: list[tuple[TableSummary, ...]] = []
        self.full_states: list[FullState] = []
        self.log_headers: list[LogHeader] = []
        self.inbound_types: Counter[int] = Counter()
        self.game_event_info_count = 0

    def drain(self) -> None:
        while True:
            try:
                record = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._process(record)

    def _matches(self, record: dict[str, Any]) -> bool:
        return self.socket is None or record.get("sock") == self.socket

    def _process(self, record: dict[str, Any] | None) -> None:
        if record is None:
            raise RuntimeError(f"{self.label}: websocket stream closed")
        if not self._matches(record):
            return
        kind = record.get("kind")
        if kind == "open":
            self.socket_open = True
            return
        if kind == "close":
            self.socket_open = False
            self.socket_closes += 1
            return
        if kind != "binary":
            return
        self.socket_open = True
        frame = decode_record_binary(record)
        if frame is None:
            return
        if frame.direction is Direction.INBOUND:
            self.inbound_types[frame.msg_type] += 1
            if frame.msg_type == TABLES_OVERVIEW:
                self.tables_overviews.append(decode_tables_overview(frame.payload))
            elif frame.msg_type == GAME_LOG_INFO:
                reader = Reader(frame.payload)
                self.log_headers.append(LogHeader(reader.s32(), reader.s32()))
            elif frame.msg_type == GAME_EVENT_INFO:
                self.game_event_info_count += 1
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
                record = await asyncio.wait_for(self.queue.get(), timeout=min(1.0, remaining))
            except TimeoutError:
                if time.monotonic() >= next_heartbeat:
                    print(f"waiting for {label}...", flush=True)
                    next_heartbeat += WAIT_HEARTBEAT_SECONDS
                continue
            self._process(record)

    async def observe(self, seconds: float, *, stop_requested: asyncio.Event) -> None:
        """Consume live traffic for a bounded observation window with heartbeats."""
        deadline = time.monotonic() + seconds
        next_heartbeat = time.monotonic() + WAIT_HEARTBEAT_SECONDS
        while True:
            if stop_requested.is_set():
                raise StopRequested()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                record = await asyncio.wait_for(self.queue.get(), timeout=min(1.0, remaining))
            except TimeoutError:
                if time.monotonic() >= next_heartbeat:
                    print(f"observing {self.label} traffic...", flush=True)
                    next_heartbeat += WAIT_HEARTBEAT_SECONDS
                continue
            self._process(record)


def _failure_reason(error: BaseException) -> str:
    detail = str(error).splitlines()[0] if str(error) else ""
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


async def _sleep_with_stop(delay: float, stop_requested: asyncio.Event) -> None:
    try:
        await asyncio.wait_for(stop_requested.wait(), timeout=delay)
    except TimeoutError:
        return
    raise StopRequested()


async def _request_snapshot(
    *,
    send: Callable[[int, bytes], Awaitable[None]],
    pump: FramePump,
    timeout: float,
    stop_requested: asyncio.Event,
) -> tuple[TableSummary, ...]:
    """Retry REQUEST_UPDATE until a fresh tablesOverview actually arrives."""
    pump.drain()
    before = len(pump.tables_overviews)
    deadline = time.monotonic() + timeout
    attempt = 0
    while True:
        if stop_requested.is_set():
            raise StopRequested()
        attempt += 1
        await send(REQUEST_UPDATE, Writer().s32(UPDATE_TABLES).build())
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"no tablesOverview after {attempt} REQUEST_UPDATE attempts in {timeout:.1f}s"
            )
        try:
            return await pump.wait_for(
                lambda: (
                    pump.tables_overviews[before]
                    if len(pump.tables_overviews) > before
                    else None
                ),
                timeout=min(TABLES_RETRY_SECONDS, remaining),
                label="tablesOverview",
                stop_requested=stop_requested,
            )
        except TimeoutError:
            print(f"no tablesOverview yet (attempt {attempt}); re-requesting...", flush=True)


def _candidate_tables(snapshot: tuple[TableSummary, ...]) -> list[TableSummary]:
    return [
        table
        for table in snapshot
        if table.status == RUNNING
        and table.is_observable
        and table.is_two_player
        and table.bots == 0
    ]


def _elapsed_sort_key(table: TableSummary) -> int:
    # A missing start time is impossible for RUNNING, but keep ordering total.
    return table.start_time if table.start_time is not None else 2**63 - 1


def _pick_distinct(
    tables: list[TableSummary],
    *,
    excluded: set[int],
    count: int,
) -> list[TableSummary]:
    """Select spread-out fresh candidates to give log indexes distinct ranges."""
    available = [table for table in sorted(tables, key=_elapsed_sort_key) if table.table_id not in excluded]
    if len(available) < count:
        raise RuntimeError(
            f"need {count} distinct eligible tables, found {len(available)} after exclusions"
        )
    if count == 1:
        return [available[len(available) // 2]]
    positions = [round(index * (len(available) - 1) / (count - 1)) for index in range(count)]
    chosen: list[TableSummary] = []
    seen: set[int] = set()
    for position in positions:
        candidate = available[position]
        if candidate.table_id not in seen:
            chosen.append(candidate)
            seen.add(candidate.table_id)
    for candidate in available:
        if len(chosen) == count:
            break
        if candidate.table_id not in seen:
            chosen.append(candidate)
            seen.add(candidate.table_id)
    return chosen


async def _join_and_read(
    *,
    send: Callable[[int, bytes], Awaitable[None]],
    pump: FramePump,
    table: TableSummary,
    join_timeout: float,
    stop_requested: asyncio.Event,
) -> JoinEvidence:
    """Join once and associate the resulting FullState with bounded evidence."""
    pump.drain()
    full_state_before = len(pump.full_states)
    logs_before = len(pump.log_headers)
    try:
        await send(JOIN_TABLE, Writer().u64(table.table_id).boolean(False).build())
        state = await pump.wait_for(
            lambda: (
                pump.full_states[full_state_before]
                if len(pump.full_states) > full_state_before
                else None
            ),
            timeout=join_timeout,
            label=f"fullGameState for table {table.table_id}",
            stop_requested=stop_requested,
        )
        # The history usually arrives immediately after FullState; drain queued
        # records so its gameLogInfo headers are in this join's evidence.
        await asyncio.sleep(0)
        pump.drain()
        return JoinEvidence(
            table_id=table.table_id,
            game_id=state.game_id,
            full_state_received=True,
            error=None,
            initial_log_headers=tuple(pump.log_headers[logs_before:]),
        )
    except StopRequested:
        raise
    except Exception as error:
        return JoinEvidence(
            table_id=table.table_id,
            game_id=None,
            full_state_received=False,
            error=_failure_reason(error),
            initial_log_headers=tuple(pump.log_headers[logs_before:]),
        )


async def _leave(
    *,
    send: Callable[[int, bytes], Awaitable[None]],
    table_id: int,
    player_id: int,
    label: str,
) -> str | None:
    try:
        await send(LEAVE_TABLE, Writer().u64(table_id).s32(player_id).build())
        print(f"left {label} table {table_id}", flush=True)
        return None
    except Exception as error:
        detail = _failure_reason(error)
        print(f"could not leave {label} table {table_id}: {detail}", flush=True)
        return detail


def _log_end(evidence: JoinEvidence) -> int | None:
    return max((header.end_index for header in evidence.initial_log_headers), default=None)


def _classify_post_second_join(
    *,
    first: JoinEvidence,
    second: JoinEvidence,
    post_headers: tuple[LogHeader, ...],
) -> dict[str, object]:
    """Classify tail headers only when their log-index ranges are distinguishable.

    gameLogInfo does not carry a game id.  Its global index is still a useful
    empirical discriminator because every running game's whole history starts
    at zero on join.  We make no claim from ambiguous ranges or from silence.
    """
    first_end = _log_end(first)
    second_end = _log_end(second)
    first_like: list[LogHeader] = []
    second_like: list[LogHeader] = []
    ambiguous: list[LogHeader] = []
    if first_end is None or second_end is None or abs(first_end - second_end) < 12:
        ambiguous = list(post_headers)
    else:
        first_anchor = first_end - 1
        second_anchor = second_end - 1
        for header in post_headers:
            first_distance = abs(header.start_index - first_anchor)
            second_distance = abs(header.start_index - second_anchor)
            if first_distance <= 8 and first_distance < second_distance:
                first_like.append(header)
            elif second_distance <= 8 and second_distance < first_distance:
                second_like.append(header)
            else:
                ambiguous.append(header)

    if first_like and second_like:
        conclusion = "multi-table-supported"
        reason = "distinct A- and B-range gameLogInfo tails arrived after joining B"
    elif second_like and not first_like:
        conclusion = "single-table-indicated"
        reason = "only B-range gameLogInfo tails arrived after joining B"
    elif not post_headers:
        conclusion = "inconclusive-no-live-log-traffic"
        reason = "no post-join gameLogInfo arrived during the bounded window"
    else:
        conclusion = "inconclusive-ambiguous-log-ranges"
        reason = "post-join log indexes could not be attributed safely"
    return {
        "first_history_end": first_end,
        "second_history_end": second_end,
        "first_like_tail_headers": [asdict(header) for header in first_like],
        "second_like_tail_headers": [asdict(header) for header in second_like],
        "ambiguous_tail_headers": [asdict(header) for header in ambiguous],
        "conclusion": conclusion,
        "reason": reason,
    }


def _failure_signals(pump: FramePump) -> dict[str, int]:
    return {
        "command_failed_type_1": pump.inbound_types.get(COMMAND_FAILED, 0),
        "instruct_to_reconnect_type_39": pump.inbound_types.get(
            INSTRUCT_TO_RECONNECT, 0
        ),
        "socket_closes": pump.socket_closes,
    }


def _as_jsonable_evidence(evidence: JoinEvidence) -> dict[str, object]:
    return {
        "table_id": evidence.table_id,
        "game_id": evidence.game_id,
        "full_state_received": evidence.full_state_received,
        "error": evidence.error,
        "initial_log_headers": [asdict(header) for header in evidence.initial_log_headers],
    }


async def _single_socket_probe(
    *,
    session: ProbeSession,
    pump: FramePump,
    first: TableSummary,
    second: TableSummary,
    player_id: int,
    args: argparse.Namespace,
    stop_requested: asyncio.Event,
) -> dict[str, object]:
    """Run Stage 1 question 1 with two sequential joins on one websocket."""
    send = session.send_frame
    first_joined = False
    second_joined = False
    first_evidence: JoinEvidence | None = None
    second_evidence: JoinEvidence | None = None
    leave_failures: dict[str, str] = {}
    post_headers: tuple[LogHeader, ...] = ()
    post_events = 0
    result: dict[str, object] = {}
    try:
        print(f"single-socket: joining A table {first.table_id}", flush=True)
        first_evidence = await _join_and_read(
            send=send,
            pump=pump,
            table=first,
            join_timeout=args.join_timeout,
            stop_requested=stop_requested,
        )
        first_joined = True
        print(
            f"single-socket A: game={first_evidence.game_id}; "
            f"full-state={first_evidence.full_state_received}; error={first_evidence.error}",
            flush=True,
        )
        if not first_evidence.full_state_received:
            result = {
                "outcome": "inconclusive",
                "reason": "could not attach table A",
                "first": _as_jsonable_evidence(first_evidence),
                "second": None,
            }
            return result
        pre_logs_before = len(pump.log_headers)
        pre_events_before = pump.game_event_info_count
        print(
            f"single-socket: confirming A traffic for {args.pre_second_observe_seconds:.0f}s",
            flush=True,
        )
        await pump.observe(args.pre_second_observe_seconds, stop_requested=stop_requested)
        pre_headers = tuple(pump.log_headers[pre_logs_before:])
        pre_events = pump.game_event_info_count - pre_events_before
        await _sleep_with_stop(args.join_delay, stop_requested)
        print(
            f"single-socket: joining B table {second.table_id} without leaving A",
            flush=True,
        )
        second_evidence = await _join_and_read(
            send=send,
            pump=pump,
            table=second,
            join_timeout=args.join_timeout,
            stop_requested=stop_requested,
        )
        second_joined = True
        print(
            f"single-socket B: game={second_evidence.game_id}; "
            f"full-state={second_evidence.full_state_received}; error={second_evidence.error}",
            flush=True,
        )
        logs_before = len(pump.log_headers)
        events_before = pump.game_event_info_count
        print(
            f"single-socket: observing traffic for {args.observe_seconds:.0f}s after B join",
            flush=True,
        )
        await pump.observe(args.observe_seconds, stop_requested=stop_requested)
        post_headers = tuple(pump.log_headers[logs_before:])
        post_events = pump.game_event_info_count - events_before
        if second_evidence.full_state_received:
            attribution = _classify_post_second_join(
                first=first_evidence,
                second=second_evidence,
                post_headers=post_headers,
            )
        else:
            attribution = {
                "conclusion": "inconclusive",
                "reason": "could not attach table B",
            }
        result = {
            "outcome": attribution["conclusion"],
            "reason": attribution["reason"],
            "first": _as_jsonable_evidence(first_evidence),
            "second": _as_jsonable_evidence(second_evidence),
            "pre_second_join_a_log_headers": [asdict(header) for header in pre_headers],
            "pre_second_join_a_game_event_info_count": pre_events,
            "post_second_join_log_headers": [asdict(header) for header in post_headers],
            "post_second_join_game_event_info_count": post_events,
            "attribution": attribution,
            "failure_signals": _failure_signals(pump),
        }
        return result
    finally:
        # Sending LEAVE for A is intentional even if B displaced it; it makes
        # cleanup explicit and harmlessly covers both server semantics.
        if second_joined:
            error = await _leave(
                send=send,
                table_id=second.table_id,
                player_id=player_id,
                label="single-socket B",
            )
            if error is not None:
                leave_failures["B"] = error
        if first_joined:
            error = await _leave(
                send=send,
                table_id=first.table_id,
                player_id=player_id,
                label="single-socket A",
            )
            if error is not None:
                leave_failures["A"] = error
        if leave_failures:
            print(f"single-socket cleanup failures: {leave_failures}", flush=True)
        if result:
            result["leave_failures"] = leave_failures


async def _two_page_probe(
    *,
    session: ProbeSession,
    page_one: Any,
    first: TableSummary,
    second: TableSummary,
    player_id: int,
    args: argparse.Namespace,
    stop_requested: asyncio.Event,
) -> dict[str, object]:
    """Run Stage 1 question 2 in two pages sharing one persistent context."""
    if session.context is None:
        raise RuntimeError("browser context unexpectedly unavailable")
    page_two: Any | None = None
    first_joined = False
    second_joined = False
    leave_failures: dict[str, str] = {}
    first_evidence: JoinEvidence | None = None
    second_evidence: JoinEvidence | None = None
    result: dict[str, object] = {}
    try:
        # Tag after the stage-one first page is settled.  The tag affects only
        # forwarding of future observations, not the browser's websocket.
        queue_one = await session.install_page_tag(page_one, "page-1")
        pump_one = FramePump(queue_one, socket=None, label="page 1")
        page_two = await session.context.new_page()
        await page_two.goto(args.url, wait_until="domcontentloaded")
        # Let the app create/reuse its authenticated websocket before the
        # forwarding callback is replaced.  This is a wait, not a request.
        await _sleep_with_stop(args.page_settle_seconds, stop_requested)
        queue_two = await session.install_page_tag(page_two, "page-2")
        pump_two = FramePump(queue_two, socket=None, label="page 2")

        send_one = lambda msg_type, payload: session.send_frame_on_page(page_one, msg_type, payload)
        send_two = lambda msg_type, payload: session.send_frame_on_page(page_two, msg_type, payload)

        # A snapshot reply on page two proves its own socket is usable before
        # we attach a game there.  Retrying remains bounded and polite.
        print("two-page: confirming page 2 lobby socket", flush=True)
        await _request_snapshot(
            send=send_two,
            pump=pump_two,
            timeout=args.startup_timeout,
            stop_requested=stop_requested,
        )

        print(f"two-page: page 1 joining table {first.table_id}", flush=True)
        first_evidence = await _join_and_read(
            send=send_one,
            pump=pump_one,
            table=first,
            join_timeout=args.join_timeout,
            stop_requested=stop_requested,
        )
        first_joined = True
        print(
            f"two-page page 1: game={first_evidence.game_id}; "
            f"full-state={first_evidence.full_state_received}; error={first_evidence.error}",
            flush=True,
        )
        await _sleep_with_stop(args.join_delay, stop_requested)
        print(f"two-page: page 2 joining table {second.table_id}", flush=True)
        second_evidence = await _join_and_read(
            send=send_two,
            pump=pump_two,
            table=second,
            join_timeout=args.join_timeout,
            stop_requested=stop_requested,
        )
        second_joined = True
        print(
            f"two-page page 2: game={second_evidence.game_id}; "
            f"full-state={second_evidence.full_state_received}; error={second_evidence.error}",
            flush=True,
        )

        logs_one_before = len(pump_one.log_headers)
        logs_two_before = len(pump_two.log_headers)
        events_one_before = pump_one.game_event_info_count
        events_two_before = pump_two.game_event_info_count
        print(
            f"two-page: observing both page streams for {args.observe_seconds:.0f}s",
            flush=True,
        )
        await asyncio.gather(
            pump_one.observe(args.observe_seconds, stop_requested=stop_requested),
            pump_two.observe(args.observe_seconds, stop_requested=stop_requested),
        )
        first_signals = _failure_signals(pump_one)
        second_signals = _failure_signals(pump_two)
        first_ok = first_evidence.full_state_received
        second_ok = second_evidence.full_state_received
        displaced = any(
            value > 0
            for signals in (first_signals, second_signals)
            for value in signals.values()
        )
        if first_ok and second_ok and not displaced:
            outcome = "two-pages-supported"
            reason = "both pages received fullGameState without command/reconnect/close signals"
        elif second_ok and (first_signals["instruct_to_reconnect_type_39"] or first_signals["socket_closes"]):
            outcome = "second-page-displaced-first"
            reason = "page 1 received a reconnect instruction or socket close after page 2 joined"
        elif not second_ok:
            outcome = "second-page-not-supported"
            reason = "page 2 did not receive fullGameState"
        else:
            outcome = "two-pages-inconclusive"
            reason = "both pages attached but transport failure signals were observed"
        result = {
            "outcome": outcome,
            "reason": reason,
            "page_1": _as_jsonable_evidence(first_evidence),
            "page_2": _as_jsonable_evidence(second_evidence),
            "page_1_post_join_log_headers": [
                asdict(header) for header in pump_one.log_headers[logs_one_before:]
            ],
            "page_2_post_join_log_headers": [
                asdict(header) for header in pump_two.log_headers[logs_two_before:]
            ],
            "page_1_post_join_game_event_info_count": (
                pump_one.game_event_info_count - events_one_before
            ),
            "page_2_post_join_game_event_info_count": (
                pump_two.game_event_info_count - events_two_before
            ),
            "page_1_failure_signals": first_signals,
            "page_2_failure_signals": second_signals,
        }
        return result
    finally:
        if page_two is not None:
            send_two = lambda msg_type, payload: session.send_frame_on_page(page_two, msg_type, payload)
            if second_joined:
                error = await _leave(
                    send=send_two,
                    table_id=second.table_id,
                    player_id=player_id,
                    label="page 2",
                )
                if error is not None:
                    leave_failures["page_2"] = error
        send_one = lambda msg_type, payload: session.send_frame_on_page(page_one, msg_type, payload)
        if first_joined:
            error = await _leave(
                send=send_one,
                table_id=first.table_id,
                player_id=player_id,
                label="page 1",
            )
            if error is not None:
                leave_failures["page_1"] = error
        if page_two is not None:
            try:
                await page_two.close()
            except Exception:
                pass
        if leave_failures:
            print(f"two-page cleanup failures: {leave_failures}", flush=True)
        if result:
            result["leave_failures"] = leave_failures


def _maximum_safe_concurrency(
    single_socket: dict[str, object] | None,
    two_pages: dict[str, object] | None,
) -> tuple[int | None, str]:
    """Make an intentionally conservative conclusion from observed probe evidence."""
    if two_pages is not None and two_pages.get("outcome") == "two-pages-supported":
        return 2, "two independent page sockets stayed attached with no displacement signals"
    if single_socket is not None and single_socket.get("outcome") == "multi-table-supported":
        return 2, "one websocket demonstrably retained two distinguishable game streams"
    if single_socket is not None and single_socket.get("outcome") == "single-table-indicated":
        return 1, "one websocket showed only the second table's identifiable live log tail"
    if two_pages is not None and two_pages.get("outcome") in {
        "second-page-displaced-first",
        "second-page-not-supported",
    }:
        return 1, "the second same-context page could not maintain an independent attachment"
    return None, "the bounded probe did not produce enough attributable live traffic"


def _run_self_test(capture_path: Path) -> None:
    """Exercise the duplicated lobby decoder and conservative attribution offline."""
    expected_counts = (340, 347, 348)
    counts: list[int] = []
    with capture_path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if (
                record.get("kind") != "binary"
                or record.get("dir") != "in"
                or not record.get("b64")
            ):
                continue
            frame = decode_record_binary(record)
            if frame is not None and frame.msg_type == TABLES_OVERVIEW:
                counts.append(len(decode_tables_overview(frame.payload)))
    if tuple(counts) != expected_counts:
        raise AssertionError(f"expected table counts {expected_counts}, got {tuple(counts)}")
    first = JoinEvidence(1, 101, True, None, (LogHeader(0, 100),))
    second = JoinEvidence(2, 202, True, None, (LogHeader(0, 500),))
    both = _classify_post_second_join(
        first=first,
        second=second,
        post_headers=(LogHeader(99, 1), LogHeader(499, 1)),
    )
    only_second = _classify_post_second_join(
        first=first,
        second=second,
        post_headers=(LogHeader(499, 1),),
    )
    if both["conclusion"] != "multi-table-supported":
        raise AssertionError("could not classify two distinguishable log tails")
    if only_second["conclusion"] != "single-table-indicated":
        raise AssertionError("could not classify an isolated second-table tail")
    print(
        "self-test passed: three saved lobby snapshots and log-tail attribution",
        flush=True,
    )


async def _run(args: argparse.Namespace) -> int:
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_requested.set)
            installed_signals.append(signum)
        except (NotImplementedError, RuntimeError):
            pass

    session = ProbeSession(
        output_root=args.out,
        profile_dir=args.profile,
        url=args.url,
        headless=not args.headful,
    )
    status = "completed"
    error: str | None = None
    single_socket: dict[str, object] | None = None
    two_pages: dict[str, object] | None = None
    max_concurrency: int | None = None
    concurrency_reason = "not run"
    snapshot_counts: list[int] = []
    try:
        print("launching authenticated Chromium for the bounded concurrency probe...", flush=True)
        await session.start()
        if session.page is None:
            raise RuntimeError("probe browser did not create its first page")
        pump = FramePump(session.frame_queue, socket=session.game_socket, label="initial page")
        print("waiting for the game websocket and existing logged-in session...", flush=True)
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
        print(f"authenticated browser session as player {player_id}", flush=True)

        first_snapshot = await _request_snapshot(
            send=session.send_frame,
            pump=pump,
            timeout=args.startup_timeout,
            stop_requested=stop_requested,
        )
        snapshot_counts.append(len(first_snapshot))
        first_candidates = _candidate_tables(first_snapshot)
        print(
            f"fresh snapshot: {len(first_snapshot)} tables; "
            f"{len(first_candidates)} eligible human observable 2-player running tables",
            flush=True,
        )
        first, second = _pick_distinct(first_candidates, excluded=set(), count=2)
        single_socket = await _single_socket_probe(
            session=session,
            pump=pump,
            first=first,
            second=second,
            player_id=player_id,
            args=args,
            stop_requested=stop_requested,
        )

        # Never use the earlier snapshot for the second experiment: stale
        # lobby rows caused join timeouts in the live recon run.
        await _sleep_with_stop(args.join_delay, stop_requested)
        second_snapshot = await _request_snapshot(
            send=session.send_frame,
            pump=pump,
            timeout=args.startup_timeout,
            stop_requested=stop_requested,
        )
        snapshot_counts.append(len(second_snapshot))
        second_candidates = _candidate_tables(second_snapshot)
        excluded = {first.table_id, second.table_id}
        page_first, page_second = _pick_distinct(
            second_candidates,
            excluded=excluded,
            count=2,
        )
        two_pages = await _two_page_probe(
            session=session,
            page_one=session.page,
            first=page_first,
            second=page_second,
            player_id=player_id,
            args=args,
            stop_requested=stop_requested,
        )
        max_concurrency, concurrency_reason = _maximum_safe_concurrency(
            single_socket, two_pages
        )
    except StopRequested:
        status = "interrupted"
        error = "interrupted by user signal"
        print("interrupt received; cleanup is underway", flush=True)
    except Exception as exc:
        status = "failed"
        error = _failure_reason(exc)
        print(f"probe failed: {error}", file=sys.stderr, flush=True)
    finally:
        run_dir = session.run_dir
        await session.stop()
        for signum in installed_signals:
            loop.remove_signal_handler(signum)
        max_concurrency, concurrency_reason = _maximum_safe_concurrency(
            single_socket, two_pages
        )
        summary = {
            "started_at_utc": started_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "finished_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "status": status,
            "error": error,
            "snapshot_table_counts": snapshot_counts,
            "single_websocket": single_socket,
            "two_pages_same_context": two_pages,
            "maximum_safe_single_account_concurrency": max_concurrency,
            "maximum_concurrency_reason": concurrency_reason,
            "wall_clock_duration_seconds": round(time.monotonic() - started_monotonic, 3),
        }
        if run_dir is not None:
            summary_path = run_dir / "summary.json"
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"probe summary: {summary_path.resolve()}", flush=True)
        print(
            "probe result: "
            f"maximum safe single-account concurrency={max_concurrency}; "
            f"{concurrency_reason}",
            flush=True,
        )
    return 0 if status == "completed" else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--join-timeout", type=float, default=12.0)
    parser.add_argument("--join-delay", type=float, default=1.0)
    parser.add_argument("--observe-seconds", type=float, default=20.0)
    parser.add_argument(
        "--pre-second-observe-seconds",
        type=float,
        default=5.0,
        help="bounded A-only traffic observation before the second single-socket join",
    )
    parser.add_argument("--page-settle-seconds", type=float, default=3.0)
    parser.add_argument(
        "--headful",
        action="store_true",
        help="show Chromium instead of using the default headless browser",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="offline-validate saved lobby snapshots and log-tail attribution",
    )
    parser.add_argument(
        "--self-test-capture",
        type=Path,
        default=DEFAULT_SELF_TEST_CAPTURE,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be greater than zero")
    if args.join_timeout <= 0:
        parser.error("--join-timeout must be greater than zero")
    if args.join_delay < MIN_JOIN_DELAY_SECONDS:
        parser.error(f"--join-delay must be at least {MIN_JOIN_DELAY_SECONDS:.1f} seconds")
    if args.observe_seconds <= 0:
        parser.error("--observe-seconds must be greater than zero")
    if args.pre_second_observe_seconds <= 0:
        parser.error("--pre-second-observe-seconds must be greater than zero")
    if args.page_settle_seconds <= 0:
        parser.error("--page-settle-seconds must be greater than zero")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        try:
            _run_self_test(args.self_test_capture)
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
