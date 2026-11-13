"""CLI wrapper for :mod:`src.v2.records.dgames_convert`."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Dominion.games spectator tuple converter from ``scripts/``."""

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    build = root / "build"
    if build.exists() and str(build) not in sys.path:
        sys.path.insert(0, str(build))
    from src.v2.records.dgames_convert import main as convert_main

    return convert_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
