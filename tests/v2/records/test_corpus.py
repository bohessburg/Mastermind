"""Tests for non-destructive local-export corpus classification."""

from __future__ import annotations

import json
from pathlib import Path

from src.v2.records.convert import discover_sources
from src.v2.records.corpus import build_manifest, load_or_create_manifest


def test_manifest_and_discovery_quarantine_stub_and_pytest_sources(tmp_path: Path) -> None:
    root = tmp_path / "exports"
    root.mkdir()
    (root / "stub.json").write_text(
        json.dumps({"seed": 1, "kingdom": [], "seats": ["human", "human"], "actions": []}),
        encoding="utf-8",
    )
    (root / "pytest.json").write_text(
        json.dumps(
            {
                "seed": 1,
                "kingdom": [],
                "seats": ["human", "bot:nn:/tmp/pytest-of-user/tiny.pt"],
                "actions": [0],
            }
        ),
        encoding="utf-8",
    )
    (root / "bad.json").write_text("{not json", encoding="utf-8")

    manifest = build_manifest(root)
    classifications = {entry["path"]: entry["classification"] for entry in manifest["files"]}
    assert classifications == {
        "bad.json": "incomplete",
        "pytest.json": "pytest_artifact",
        "stub.json": "stub",
    }
    assert discover_sources(root) == ()
    assert {path.name for path in discover_sources(root, include_all=True)} == {
        "pytest.json",
        "stub.json",
    }

    (root / "new-stub.json").write_text(
        json.dumps({"seed": 2, "kingdom": [], "seats": ["human", "human"], "actions": []}),
        encoding="utf-8",
    )
    refreshed = load_or_create_manifest(root)
    assert len(refreshed["files"]) == 4
