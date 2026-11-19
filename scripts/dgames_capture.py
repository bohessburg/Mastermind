"""Record a manual dominion.games spectating session for scraper recon.

Reuses the arena's ArenaRecorder (Playwright + the ws_hook.js frame sniffer)
but against the collector's own profile, so it never disturbs the arena bot's
profile or its live games.

Intended use is Milestone 0: log in once via ``scripts/dgames_login.py``, run
this, then manually click into the lobby and spectate games start-to-finish.
The resulting ``frames.jsonl`` answers the questions static analysis could not:
what the client sends to start spectating, and whether hidden zones really
arrive as ``-1`` for a spectator.

Credential safety: outbound login-family messages are redacted to their type
and byte length before anything is written, so a stray re-login mid-capture
cannot put a password or session id into the recording.  Even so, prefer to
log in with ``dgames_login.py`` (which records nothing at all).

Usage: ``./.venv/bin/python scripts/dgames_capture.py``
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import signal
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v2.arena.recon.record import ArenaRecorder  # noqa: E402


DEFAULT_PROFILE = Path("dgames-profile")
DEFAULT_OUT = Path("data/dominion_games/recon/captures")
DEFAULT_URL = "https://dominion.games"

# Client->server messages that carry a password, a session id, or a login
# code.  Ids are the positional ordinals derived in RECON.md; the range is
# deliberately wide because those ordinals shift between client releases and
# over-redacting an outbound frame costs us nothing.
# 44 is the client heartbeat and carries no payload -- deliberately excluded.
SENSITIVE_OUTBOUND = frozenset({1, 13, 14, 15, 16, 19, 24, 45, 46})


def _outbound_msg_type(payload: bytes) -> int | None:
    """Return the leading int32 message type of an outbound frame."""
    if len(payload) < 4:
        return None
    return int(struct.unpack(">I", payload[:4])[0])


class DgamesRecorder(ArenaRecorder):
    """ArenaRecorder that refuses to persist credential-bearing frames."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.redacted = 0
        self.binary_frames = 0

    def _record_frame(self, record: dict[str, Any]) -> None:
        if record.get("kind") == "binary":
            self.binary_frames += 1
        if record.get("dir") == "out" and record.get("b64") and record.get("data"):
            try:
                payload = base64.b64decode(record["data"])
            except Exception:
                payload = b""
            msg_type = _outbound_msg_type(payload)
            if msg_type in SENSITIVE_OUTBOUND:
                record = dict(record)
                record["data"] = ""
                record["b64"] = False
                record["redacted"] = True
                record["msg_type"] = msg_type
                record["byte_len"] = len(payload)
                self.redacted += 1
        super()._record_frame(record)


async def _heartbeat(recorder: DgamesRecorder, interval: float) -> None:
    """Emit progress so a long manual capture is never silently opaque."""
    elapsed = 0.0
    while True:
        await asyncio.sleep(interval)
        elapsed += interval
        print(
            f"[{int(elapsed)}s] binary frames: {recorder.binary_frames}"
            f"  redacted: {recorder.redacted}",
            flush=True,
        )


async def _run(args: argparse.Namespace) -> None:
    recorder = DgamesRecorder(
        args.out,
        args.profile,
        screenshot_interval=args.screenshot_interval,
        dom_interval=args.dom_interval,
        headless=False,
    )

    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_requested.set)
            installed.append(signum)
        except (NotImplementedError, RuntimeError):
            pass

    beat: asyncio.Task[None] | None = None
    try:
        run_dir = await recorder.start()
        print(f"recording to: {run_dir.resolve()}", flush=True)
        print("Spectate games start-to-finish, then press Ctrl-C.", flush=True)
        page = await recorder.new_page()
        await page.goto(args.url, wait_until="domcontentloaded")
        beat = asyncio.create_task(_heartbeat(recorder, args.heartbeat))
        await stop_requested.wait()
    finally:
        if beat is not None:
            beat.cancel()
        for signum in installed:
            loop.remove_signal_handler(signum)
        await recorder.stop()
        print(
            f"done. binary frames: {recorder.binary_frames}, redacted: {recorder.redacted}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--screenshot-interval", type=float, default=10.0)
    parser.add_argument("--dom-interval", type=float, default=30.0)
    parser.add_argument("--heartbeat", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
