from __future__ import annotations

import asyncio

from src.v2.arena.browser.session import ArenaSession


class _FakePage:
    def is_closed(self) -> bool:
        return False

    async def content(self) -> str:
        return "<html><body>lobby evidence</body></html>"


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
