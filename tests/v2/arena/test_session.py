from __future__ import annotations

import asyncio
import base64

from src.v2.arena.browser.session import ArenaSession
from src.v2.arena.protocol.frames import Writer


class _FakePage:
    def is_closed(self) -> bool:
        return False

    async def content(self) -> str:
        return "<html><body>lobby evidence</body></html>"


class _FakeSendPage:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def is_closed(self) -> bool:
        return False

    async def evaluate(self, expression: str, argument: str) -> None:
        self.calls.append((expression, argument))


def test_session_snapshot_dom_writes_labeled_run_archive(tmp_path) -> None:
    session = ArenaSession(output_root=tmp_path)
    session.run_dir = tmp_path / "run"
    session.run_dir.mkdir()
    session.page = _FakePage()  # type: ignore[assignment]

    destination = asyncio.run(session.snapshot_dom("Lobby failure: Start game"))

    assert destination.parent == session.run_dir
    assert destination.name.endswith("-lobby-failure-start-game.html")
    assert destination.read_text(encoding="utf-8") == (
        "<html><body>lobby evidence</body></html>"
    )


def test_session_send_frame_builds_outbound_envelope_for_page_hook() -> None:
    session = ArenaSession()
    page = _FakeSendPage()
    session.page = page  # type: ignore[assignment]

    asyncio.run(session.send_frame(37, b"answer"))

    assert len(page.calls) == 1
    expression, argument = page.calls[0]
    assert "__arenaSend" in expression
    assert base64.b64decode(argument) == (
        Writer().u32(37).bytes(b"answer").build()
    )
