"""Record manual arena sessions.

Setup: ``pip install -r src/v2/arena/requirements.txt && .venv/bin/python -m
playwright install chromium``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, time
from typing import Any

from playwright.async_api import BrowserContext, ConsoleMessage, Frame, Page, Playwright
from playwright.async_api import async_playwright


HOOK_PATH = Path(__file__).resolve().parents[1] / "browser" / "ws_hook.js"


class ArenaRecorder:
    """Owns a Playwright context and writes its browser observations to disk."""

    def __init__(
        self,
        output_root: Path | str,
        profile_dir: Path | str,
        *,
        screenshot_interval: float = 5.0,
        dom_interval: float = 15.0,
        headless: bool = False,
    ) -> None:
        if screenshot_interval <= 0:
            raise ValueError("screenshot_interval must be greater than zero")
        if dom_interval <= 0:
            raise ValueError("dom_interval must be greater than zero")

        self.output_root = Path(output_root)
        self.profile_dir = Path(profile_dir)
        self.screenshot_interval = screenshot_interval
        self.dom_interval = dom_interval
        self.headless = headless

        self.run_dir: Path | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self._frames_file: Any | None = None
        self._playwright: Playwright | None = None
        self._periodic_tasks: list[asyncio.Task[None]] = []
        self._started_at: float | None = None
        self._attached_pages: set[int] = set()
        self._stopped = False

    async def start(self) -> Path:
        """Start the persistent browser context and return this run's directory."""
        if self.context is not None:
            raise RuntimeError("recorder is already running")

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self._create_run_dir()
        self._frames_file = (self.run_dir / "frames.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self._started_at = monotonic()
        self._stopped = False

        try:
            self._playwright = await async_playwright().start()
            self.context = await self._playwright.chromium.launch_persistent_context(
                str(self.profile_dir),
                headless=self.headless,
            )
            await self.context.expose_function("__arenaFrame", self._record_frame)
            await self.context.add_init_script(path=str(HOOK_PATH))
            self.context.on("page", self._attach_page)
            for page in self.context.pages:
                self._attach_page(page)

            self._periodic_tasks = [
                asyncio.create_task(
                    self._periodic_capture(self.screenshot_interval, self._save_screenshot),
                    name="arena-screenshot-recorder",
                ),
                asyncio.create_task(
                    self._periodic_capture(self.dom_interval, self._save_dom),
                    name="arena-dom-recorder",
                ),
            ]
        except BaseException:
            await self.stop()
            raise

        return self.run_dir

    async def new_page(self) -> Page:
        """Create a page in the instrumented recording context."""
        if self.context is None:
            raise RuntimeError("start the recorder before creating a page")

        page = await self.context.new_page()
        self._attach_page(page)
        self.page = page
        return page

    async def stop(self) -> None:
        """Stop periodic capture and release browser and file resources."""
        if self._stopped:
            return
        self._stopped = True

        tasks, self._periodic_tasks = self._periodic_tasks, []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        context, self.context = self.context, None
        if context is not None:
            try:
                await context.close()
            except Exception:
                pass

        playwright, self._playwright = self._playwright, None
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                pass

        frames_file, self._frames_file = self._frames_file, None
        if frames_file is not None:
            frames_file.close()

    def _create_run_dir(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir = self.output_root / timestamp
        suffix = 1
        while run_dir.exists():
            run_dir = self.output_root / f"{timestamp}-{suffix}"
            suffix += 1
        run_dir.mkdir()
        return run_dir

    def _attach_page(self, page: Page) -> None:
        if id(page) in self._attached_pages:
            return
        self._attached_pages.add(id(page))
        page.on("console", self._record_console)
        page.on("framenavigated", self._record_navigation)

    def _record_frame(self, record: dict[str, Any]) -> None:
        self._write_record(record)

    def _record_console(self, message: ConsoleMessage) -> None:
        if message.type not in {"error", "warning"}:
            return
        self._write_record(
            {
                "ts": int(time() * 1000),
                "kind": "console",
                "level": message.type,
                "data": message.text,
                "url": message.location.get("url", ""),
            }
        )

    def _record_navigation(self, frame: Frame) -> None:
        self._write_record(
            {
                "ts": int(time() * 1000),
                "kind": "nav",
                "url": frame.url,
            }
        )

    def _write_record(self, record: dict[str, Any]) -> None:
        if self._frames_file is None:
            return
        self._frames_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._frames_file.flush()

    async def _periodic_capture(
        self,
        interval: float,
        capture: Any,
    ) -> None:
        while True:
            await asyncio.sleep(interval)
            await capture()

    async def _save_screenshot(self) -> None:
        page = self._capture_page()
        if page is None or self.run_dir is None:
            return
        try:
            await page.screenshot(path=str(self.run_dir / self._capture_name("screenshot", "png")))
        except Exception:
            # Navigation or page closure should not stop the manual recorder.
            pass

    async def _save_dom(self) -> None:
        page = self._capture_page()
        if page is None or self.run_dir is None:
            return
        try:
            contents = await page.content()
            (self.run_dir / self._capture_name("dom", "html")).write_text(
                contents, encoding="utf-8"
            )
        except Exception:
            # Navigation or page closure should not stop the manual recorder.
            pass

    def _capture_page(self) -> Page | None:
        if self.page is not None and not self.page.is_closed():
            return self.page
        if self.context is None:
            return None
        pages = [page for page in self.context.pages if not page.is_closed()]
        return pages[-1] if pages else None

    def _capture_name(self, prefix: str, suffix: str) -> str:
        assert self._started_at is not None
        elapsed_ms = int((monotonic() - self._started_at) * 1000)
        return f"{prefix}-{elapsed_ms}.{suffix}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record manual WebSocket arena sessions")
    parser.add_argument("--out", type=Path, default=Path("arena-recordings"))
    parser.add_argument("--url", default="https://dominion.games")
    parser.add_argument("--screenshot-interval", type=float, default=5.0)
    parser.add_argument("--dom-interval", type=float, default=15.0)
    args = parser.parse_args(argv)
    if args.screenshot_interval <= 0:
        parser.error("--screenshot-interval must be greater than zero")
    if args.dom_interval <= 0:
        parser.error("--dom-interval must be greater than zero")
    return args


async def _run_cli(args: argparse.Namespace) -> None:
    recorder = ArenaRecorder(
        args.out,
        Path("arena-profile"),
        screenshot_interval=args.screenshot_interval,
        dom_interval=args.dom_interval,
        headless=False,
    )
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_requested.set)
            installed_signals.append(signum)
        except (NotImplementedError, RuntimeError):
            pass

    try:
        run_dir = await recorder.start()
        print(run_dir, flush=True)
        page = await recorder.new_page()
        await page.goto(args.url, wait_until="domcontentloaded")
        await stop_requested.wait()
    finally:
        for signum in installed_signals:
            loop.remove_signal_handler(signum)
        await recorder.stop()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        asyncio.run(_run_cli(args))
    except KeyboardInterrupt:
        # This fallback is for platforms without asyncio signal handlers.
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
