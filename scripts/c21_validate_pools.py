"""Duel-validate c21 curriculum pools: engine3 vs bigmoney across sampled kingdoms.

For each pool in configs/kingdom_pools_c21.json, sample N distinct 10-card
kingdoms (the same uniform draw the engine uses), play seat-balanced scripted
games, and report engine3's win rate overall and on the worst kingdom.
Launch bar: >=85% engine-dominant overall.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import dominion_v2_py as dz
from src.v2.web.server.defs import def_id

KINGDOMS_PER_POOL = 30
GAMES_PER_KINGDOM = 40  # seat-balanced


def play(kingdom: list[int], seed: int, engine_seat: int) -> int:
    game = dz.new_game(dz.Setup(players=2, kingdom=sorted(kingdom)), seed)
    bots = {engine_seat: dz.ScriptedBot("engine3"), 1 - engine_seat: dz.ScriptedBot("bigmoney")}
    plies = 0
    while not game.game_over() and plies < 4000:
        plies += 1
        seat = int(game.current_decision()["player"])
        action = int(bots[seat].choose(game))
        if not game.legal_mask()[action]:
            return -1
        game.step(action)
    winner = game.winner()
    if winner is None:
        return 0
    return 1 if int(winner) == engine_seat else 0


def main() -> int:
    pools = json.load(open(REPO / "configs/kingdom_pools_c21.json"))
    report = {}
    ok = True
    for name, cards in pools.items():
        if name.startswith("_"):
            continue
        defs = [def_id(c) for c in cards]
        rng = random.Random(20260802)
        results = []
        for k in range(KINGDOMS_PER_POOL):
            kingdom = rng.sample(defs, 10)
            wins = games = 0
            for g in range(GAMES_PER_KINGDOM):
                r = play(kingdom, 400000 + k * 1000 + g, g % 2)
                if r >= 0:
                    games += 1
                    wins += r
            results.append({"kingdom": sorted(kingdom), "win_pct": 100.0 * wins / max(games, 1), "games": games})
            print(f"{name} kingdom {k+1}/{KINGDOMS_PER_POOL}: {results[-1]['win_pct']:.1f}%", flush=True)
        overall = sum(r["win_pct"] for r in results) / len(results)
        worst = min(results, key=lambda r: r["win_pct"])
        passed = overall >= 85.0
        ok = ok and passed
        report[name] = {"overall_win_pct": overall, "worst_kingdom_pct": worst["win_pct"], "worst_kingdom": worst["kingdom"], "passed": passed}
        print(f"== {name}: overall {overall:.1f}% | worst kingdom {worst['win_pct']:.1f}% | {'PASS' if passed else 'FAIL'}", flush=True)
    (REPO / "bench/c21_pool_validation.json").write_text(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
