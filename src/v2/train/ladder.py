"""Bradley-Terry rating ladder over every recorded checkpoint match.

Replaces "pointwise bar vs one fixed champion" with a joint rating of all
checkpoints (and scripted bots) from every honest_eval / sprt result JSON.
One opponent measures closeness to one style; a ladder pools all evidence,
exposes non-transitivity, and gives each rating a bootstrap CI.

    PYTHONPATH=build .venv/bin/python -m src.v2.train.ladder \
        bench/honest_eval bench/ --anchor checkpoints/remote/campaign15/gen_0045.pt

Ratings are Elo-scaled (400*log10 odds), anchored so ``--anchor`` = 0. Ties
count as half a win for each side. Errors are ignored.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _player(config: dict[str, Any], side: str) -> str | None:
    if side == "a":
        return str(config["a"]) if config.get("a") else None
    if config.get("b"):
        return str(config["b"])
    if config.get("opponent"):
        return f"bot:{config['opponent']}"
    return None


def load_matches(paths: Iterable[Path]) -> list[tuple[str, str, float]]:
    """Return (a, b, score_for_a) per game from result JSONs found under paths."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.suffix == ".json":
            files.append(path)
    games: list[tuple[str, str, float]] = []
    seen: set[Path] = set()
    for file in files:
        resolved = file.resolve()
        if resolved in seen or file.name.endswith(".batch.json"):
            continue
        seen.add(resolved)
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
            continue
        if not isinstance(payload.get("games"), list):
            continue
        a, b = _player(payload["config"], "a"), _player(payload["config"], "b")
        if a is None or b is None or a == b:
            continue
        for game in payload["games"]:
            result = game.get("result") if isinstance(game, dict) else None
            score = {"W": 1.0, "T": 0.5, "L": 0.0}.get(result)
            if score is not None:
                games.append((a, b, score))
    return games


def fit_bradley_terry(
    games: list[tuple[str, str, float]],
    *,
    iterations: int = 2000,
    prior_games: float = 1.0,
) -> dict[str, float]:
    """MM fit of BT strengths; a weak prior (virtual draws vs a mean player)
    keeps ratings finite for players with all wins or all losses."""
    players = sorted({p for a, b, _ in games for p in (a, b)})
    index = {player: i for i, player in enumerate(players)}
    n = len(players)
    wins = np.zeros(n)
    pair_games: dict[tuple[int, int], float] = defaultdict(float)
    for a, b, score in games:
        i, j = index[a], index[b]
        wins[i] += score
        wins[j] += 1.0 - score
        pair_games[(min(i, j), max(i, j))] += 1.0
    strength = np.ones(n)
    for _ in range(iterations):
        # MM update: s_i = W_i / sum_j n_ij/(s_i+s_j), plus `prior_games`
        # virtual draws against a fixed strength-1 reference player.
        denom = prior_games / (strength + 1.0)
        for (i, j), count in pair_games.items():
            total = strength[i] + strength[j]
            denom[i] += count / total
            denom[j] += count / total
        updated = (wins + 0.5 * prior_games) / denom
        updated /= math.exp(np.mean(np.log(updated)))
        if np.max(np.abs(np.log(updated) - np.log(strength))) < 1.0e-9:
            strength = updated
            break
        strength = updated
    return {player: 400.0 * math.log10(strength[index[player]]) for player in players}


def ladder(
    games: list[tuple[str, str, float]],
    *,
    anchor: str | None,
    bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    ratings = fit_bradley_terry(games)
    offset = ratings.get(anchor, 0.0) if anchor else 0.0
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    array = list(games)
    for _ in range(bootstrap):
        picks = rng.integers(0, len(array), size=len(array))
        boot = fit_bradley_terry([array[int(k)] for k in picks], iterations=400)
        boot_offset = boot.get(anchor, 0.0) if anchor else 0.0
        for player, rating in boot.items():
            samples[player].append(rating - boot_offset)
    counts: dict[str, int] = defaultdict(int)
    for a, b, _ in games:
        counts[a] += 1
        counts[b] += 1
    rows = []
    for player, rating in ratings.items():
        boot = samples.get(player, [])
        low, high = (np.percentile(boot, [2.5, 97.5]).tolist() if len(boot) >= 10 else [float("nan")] * 2)
        rows.append(
            {
                "player": player,
                "elo": rating - offset,
                "ci95": [low, high],
                "games": counts[player],
            }
        )
    rows.sort(key=lambda row: -row["elo"])
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bradley-Terry ladder over honest_eval/sprt JSONs")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--anchor", default="checkpoints/remote/campaign15/gen_0045.pt")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    games = load_matches(args.paths)
    if not games:
        raise SystemExit("no match results found")
    rows = ladder(games, anchor=args.anchor, bootstrap=args.bootstrap, seed=args.seed)
    print(f"{'elo':>7} {'95% CI':>17} {'games':>6}  player")
    for row in rows:
        low, high = row["ci95"]
        print(f"{row['elo']:+7.0f} [{low:+7.0f},{high:+7.0f}] {row['games']:6d}  {row['player']}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
