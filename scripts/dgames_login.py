"""Park a headful browser on dominion.games so a human can log in once.

The collector authenticates by reusing a persistent Chromium profile rather
than by storing credentials: dominion.games keeps its playerId/sessionId in
the profile, so logging in here once is enough for later capture runs.

This script deliberately installs **no** WebSocket hook and records nothing.
The site's LOGIN message carries the password in plaintext over the socket,
so a recording session must never be the one you type credentials into.
Use ``scripts/dgames_capture.py`` for recording, after logging in here.

Setup: ``pip install -r src/v2/arena/requirements.txt && .venv/bin/python -m
playwright install chromium``.

Usage: ``./.venv/bin/python scripts/dgames_login.py``
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from playwright.async_api import async_playwright


DEFAULT_PROFILE = Path("dgames-profile")
DEFAULT_URL = "https://dominion.games"


async def _run(profile_dir: Path, url: str) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)

    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_requested.set)
            installed.append(signum)
        except (NotImplementedError, RuntimeError):
            pass

    playwright = await async_playwright().start()
    context = None
    try:
        context = await playwright.chromium.launch_persistent_context(
            str(profile_dir),
            headless=False,
        )
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(url, wait_until="domcontentloaded")

        print(f"profile: {profile_dir.resolve()}", flush=True)
        print("Log in with the collector's account, then leave it logged in.", flush=True)
        print("Nothing is being recorded. Press Ctrl-C here when done.", flush=True)

        # Exit early if the human just closes the window.
        closed = asyncio.create_task(context.wait_for_event("close"))
        stopped = asyncio.create_task(stop_requested.wait())
        await asyncio.wait({closed, stopped}, return_when=asyncio.FIRST_COMPLETED)
        for task in (closed, stopped):
            task.cancel()
    finally:
        for signum in installed:
            loop.remove_signal_handler(signum)
        if context is not None:
            try:
                await context.close()
            except Exception:
                pass
        await playwright.stop()
        print("session saved to the profile", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args(argv)
    try:
        asyncio.run(_run(args.profile, args.url))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
