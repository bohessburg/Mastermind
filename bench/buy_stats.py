"""Per-card buy/play tallies from live games (NN vs engine3).

Plays real searched games (128 sims, K=1 for speed; honest determinized
serving path) and counts the NN seat's buys and action plays per card.
Compares checkpoints on engine-pool and random boards.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

REPO = Path("/Users/paisho/Projects/Mastermind")
sys.path.insert(0, str(REPO))

SENTRY_POOL = ["Sentry", "Moneylender", "Village", "Laboratory", "Market", "Festival", "Smithy", "Merchant", "Poacher", "Witch"]
GAMES = 24
SIMS = 128


def play_one(args):
    ckpt, board, seed = args
    import dominion_v2_py as dz
    from src.v2.arena.bot.policy import load_policy, choose_nnmcts_action

    policy = load_policy(Path(ckpt), legacy_shim="auto")
    from src.v2.web.server.defs import kingdom_def_ids, def_id
    pool = kingdom_def_ids()
    rng = random.Random(seed)
    if board == "sentry":
        kingdom = sorted(def_id(n) for n in SENTRY_POOL)
    else:
        kingdom = sorted(rng.sample(pool, 10))
    game = dz.new_game(dz.Setup(players=2, kingdom=kingdom), seed)
    bot = dz.ScriptedBot("engine3")
    nn_seat = seed % 2
    buys: Counter = Counter()
    plays: Counter = Counter()
    while not game.game_over():
        seat = int(game.current_decision()["player"])
        if seat == nn_seat:
            a = choose_nnmcts_action(game, seat, policy, sims=SIMS, determinizations=1)
            if 206 <= a < 247:
                buys[a - 206] += 1
            elif 1 <= a < 42:
                plays[a - 1] += 1
        else:
            a = int(bot.choose(game))
        if not game.legal_mask()[a]:
            break
        game.step(a)
    return buys, plays, int(game.winner() == nn_seat if game.winner() is not None else -1)


def main():
    import dominion_v2_py as dz
    from src.v2.web.server.defs import load_defs
    name_of = {int(c["id"]): c["name"] for c in load_defs()["defs"]}
    results = {}
    for ckpt_label, ckpt in [("gen5", "checkpoints/campaign20/gen_0005.pt"), ("champ", "checkpoints/remote/campaign15/gen_0045.pt")]:
        for board in ("sentry", "random"):
            jobs = [(ckpt, board, 9000 + i) for i in range(GAMES)]
            with Pool(8) as p:
                outs = p.map(play_one, jobs)
            buys: Counter = Counter()
            plays: Counter = Counter()
            wins = 0
            for b, pl, w in outs:
                buys.update(b)
                plays.update(pl)
                wins += 1 if w == 1 else 0
            results[f"{ckpt_label}:{board}"] = {
                "wins": wins,
                "games": GAMES,
                "buys_per_game": {name_of.get(d, str(d)): round(c / GAMES, 2) for d, c in buys.most_common(14)},
                "plays_per_game": {name_of.get(d, str(d)): round(c / GAMES, 2) for d, c in plays.most_common(8)},
            }
            print(f"{ckpt_label}:{board} done ({wins}/{GAMES} wins)", flush=True)
    Path("bench/buy_stats.json").write_text(json.dumps(results, indent=2))
    for key, r in results.items():
        print(f"== {key} ({r['wins']}/{r['games']} wins) buys/game: {r['buys_per_game']}", flush=True)


if __name__ == "__main__":
    main()
