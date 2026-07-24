"""Live Playwright session and in-process WebSocket event feed."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, AsyncIterator

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright

from ..archive import serialize_frame_record
from ..protocol.events import GameEvent
from ..protocol.live import RawFrameQueue, events_from_queue


HOOK_PATH = Path(__file__).with_name("ws_hook.js")


class ArenaSession:
    """Own the persistent, headful browser used by one supervised run.

    The observer is the same ``add_init_script`` hook as the recorder.  Its
    exposed binding writes an archival JSONL mirror and pushes the original
    record straight onto ``frame_queue``; the live control path never reads
    frames back from disk.
    """

    def __init__(
        self,
        *,
        output_root: Path | str = "exports/arena",
        profile_dir: Path | str = "arena-profile",
        url: str = "https://dominion.games",
        game_socket: int = 3,
        headless: bool = False,
    ) -> None:
        self.output_root = Path(output_root)
        self.profile_dir = Path(profile_dir)
        self.url = url
        self.game_socket = game_socket
        self.headless = headless

        self.run_dir: Path | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self.frame_queue: RawFrameQueue = asyncio.Queue()
        self._frames_file: Any | None = None
        self._playwright: Playwright | None = None
        self._stopped = False

    @property
    def frames_path(self) -> Path:
        """Return the raw-frame mirror path after the session has started."""
        if self.run_dir is None:
            raise RuntimeError("start the arena session before accessing its archive")
        return self.run_dir / "frames.jsonl"

    async def start(self) -> Path:
        """Launch the browser, install the hook, and navigate the live client."""
        if self.context is not None:
            raise RuntimeError("arena session is already running")

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self._create_run_dir()
        self._frames_file = self.frames_path.open("w", encoding="utf-8", buffering=1)
        self.frame_queue = asyncio.Queue()
        self._stopped = False

        try:
            self._playwright = await async_playwright().start()
            self.context = await self._playwright.chromium.launch_persistent_context(
                str(self.profile_dir),
                headless=self.headless,
            )
            await self.context.expose_function("__arenaFrame", self._receive_frame)
            # This must precede page creation/navigation.  It intentionally
            # matches ArenaRecorder's capture installation exactly.
            await self.context.add_init_script(path=str(HOOK_PATH))
            self.page = await self.context.new_page()
            await self.page.goto(self.url, wait_until="domcontentloaded")
        except BaseException:
            await self.stop()
            raise
        return self.run_dir

    async def stop(self) -> None:
        """Close browser resources and wake an event pump awaiting input."""
        if self._stopped:
            return
        self._stopped = True

        context, self.context = self.context, None
        if context is not None:
            try:
                await context.close()
            except Exception:
                pass
        self.page = None

        playwright, self._playwright = self._playwright, None
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                pass

        frames_file, self._frames_file = self._frames_file, None
        if frames_file is not None:
            frames_file.close()
        self.frame_queue.put_nowait(None)

    async def screenshot(self, path: Path | str) -> Path:
        """Capture the live page at ``path`` for an archive report."""
        if self.page is None or self.page.is_closed():
            raise RuntimeError("Playwright page is unavailable for screenshot")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await self.page.screenshot(path=str(destination))
        return destination

    async def snapshot_dom(self, label: str) -> Path:
        """Archive the current DOM without affecting the live frame feed.

        This deliberately mirrors ``ArenaRecorder._save_dom`` but is invoked
        only at lobby state boundaries and terminal selector failures.  The
        supplied label makes a missing live selector easy to identify among
        the run's normal frame archive.
        """
        if self.run_dir is None:
            raise RuntimeError("start the arena session before capturing DOM")
        if self.page is None or self.page.is_closed():
            raise RuntimeError("Playwright page is unavailable for DOM capture")
        safe_label = re.sub(r"[^a-z0-9]+", "-", label.casefold()).strip("-")
        if not safe_label:
            safe_label = "snapshot"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        destination = self.run_dir / f"dom-{timestamp}-{safe_label}.html"
        contents = await self.page.content()
        destination.write_text(contents, encoding="utf-8")
        return destination

    async def events(self) -> AsyncIterator[GameEvent]:
        """Yield parsed game events while idling on the raw-frame queue."""
        async for event in events_from_queue(self.frame_queue, socket=self.game_socket):
            yield event

    async def __aenter__(self) -> "ArenaSession":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    def _receive_frame(self, record: dict[str, Any]) -> None:
        """Receive the recorder hook's JSON-safe record in Playwright's loop."""
        if self._stopped:
            return
        copied = dict(record)
        if self._frames_file is not None:
            self._frames_file.write(serialize_frame_record(copied))
            self._frames_file.flush()
        self.frame_queue.put_nowait(copied)

    def _create_run_dir(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir = self.output_root / timestamp
        suffix = 1
        while run_dir.exists():
            run_dir = self.output_root / f"{timestamp}-{suffix}"
            suffix += 1
        run_dir.mkdir()
        return run_dir
