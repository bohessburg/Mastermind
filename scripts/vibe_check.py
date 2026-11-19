#!/usr/bin/env python3
"""Vibe check: watch one checkpoint play full games against a scripted bot.

Prints a turn-by-turn log (actions played, coins, buys, gains, trashes) and
the final decks, so a human can judge HOW the net plays, not just how often it
wins. The net moves with the arena's honest NN-MCTS (it does not see hidden
cards); the opponent is a scripted bot (engine3 by default).

  python scripts/vibe_check.py checkpoints/campaign29/gen_0006.pt
  python scripts/vibe_check.py CKPT --boards chapel --seats 0 --seed 7
  python scripts/vibe_check.py CKPT --boards random,random:5   # 10 random kingdom cards

Boards (fixed here so later vibes are comparable; the c24-era vibe script and
its board pools were lost with that box, so older vibe scores do not compare):
  sentry  Sentry, Merchant, Village, Smithy, Market, Militia, Witch, Cellar, Moat, Remodel
  chapel  Chapel, Village, Smithy, Laboratory, Market, Festival, Council Room, Moat, Militia, Witch
  random[:N]  10 cards drawn from the 26-card pool with seed N (default: --seed)
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import dominion_v2_py as dz  # noqa: E402
from src.v2.arena.bot.policy import choose_nnmcts_action, load_policy  # noqa: E402
from src.v2.web.server.defs import def_name, kingdom_def_ids  # noqa: E402

BOARDS = {
    "sentry": ("Sentry", "Merchant", "Village", "Smithy", "Market", "Militia", "Witch", "Cellar", "Moat", "Remodel"),
    "chapel": ("Chapel", "Village", "Smithy", "Laboratory", "Market", "Festival", "Council Room", "Moat", "Militia", "Witch"),
}
OPPONENTS = {
    "engine3": dz.EvalScriptedBotKind.EngineV3,
    "engine": dz.EvalScriptedBotKind.Engine,
    "bigmoney": dz.EvalScriptedBotKind.BigMoney,
    "thinner": dz.EvalScriptedBotKind.Thinner,
}


def board_cards(board: str, seed: int) -> tuple[str, ...]:
    if board in BOARDS:
        return BOARDS[board]
    if board == "random" or board.startswith("random:"):
        draw_seed = int(board.split(":", 1)[1]) if ":" in board else seed
        pool = sorted(def_name(card) for card in kingdom_def_ids())
        return tuple(random.Random(draw_seed).sample(pool, 10))
    raise SystemExit(f"unknown board {board!r}; choose from {', '.join(BOARDS)} or random[:N]")


TREASURES = {"Copper", "Silver", "Gold", "Platinum"}
VICTORY = ("Province", "Duchy", "Estate", "Gardens", "Curse")
PHASE_KINDS = (1, 2)  # PhaseAction, PhaseBuy: the decider is the player whose turn it is


def tally(names: list[str]) -> str:
    counts = Counter(names)
    return ", ".join(name if n == 1 else f"{name} x{n}" for name, n in counts.items())


def deck_line(deck: Counter[str]) -> str:
    def order(item: tuple[str, int]) -> tuple[int, str]:
        name = item[0]
        group = 2 if name in VICTORY else 1 if name in TREASURES else 0
        return group, name

    return ", ".join(f"{n} {name}" for name, n in sorted(deck.items(), key=order) if n > 0)


class Turn:
    def __init__(self, number: int, player: int) -> None:
        self.number = number
        self.player = player
        self.plays: list[str] = []
        self.buys: list[str] = []
        self.gains: list[str] = []
        self.trashes: list[str] = []
        self.given: list[str] = []
        self.coins = 0

    def line(self, label: str) -> str:
        parts = [f"R{self.number // 2 + 1:02d} {label:<8} ${self.coins}"]
        if self.plays:
            parts.append("plays " + tally(self.plays))
        if self.trashes:
            parts.append("TRASHES " + tally(self.trashes))
        if self.gains:
            parts.append("gains " + tally(self.gains))
        if self.given:
            parts.append("gives " + tally(self.given))
        parts.append("buys " + (tally(self.buys) if self.buys else "nothing"))
        return " | ".join(parts)


def play_game(policy, board: str, kingdom: tuple[str, ...], nn_seat: int, seed: int, opponent: str, sims: int, quiet: bool) -> dict:
    game = dz.new_game(dz.Setup(2, list(kingdom)), seed)
    bot = dz.ScriptedBot(OPPONENTS[opponent])
    labels = {nn_seat: "NET", 1 - nn_seat: opponent}
    decks = {0: Counter(Copper=7, Estate=3), 1: Counter(Copper=7, Estate=3)}
    first_province: dict[int, int | None] = {0: None, 1: None}
    actions_played = {0: 0, 1: 0}
    turns_taken = {0: 0, 1: 0}
    turn: Turn | None = None
    last_action_card = ""
    started = time.time()

    def flush() -> None:
        if turn is not None and not quiet:
            print(turn.line(labels[turn.player]), flush=True)

    while not game.game_over():
        decision = game.current_decision()
        player = int(decision["player"])
        kind = int(decision["kind"])
        number = int(game.turn())
        if turn is None or number != turn.number:
            flush()
            turn = Turn(number, player if kind in PHASE_KINDS else number % 2)
            turns_taken[turn.player] += 1
            last_action_card = ""
        if kind == 2 and player == turn.player:
            turn.coins = max(turn.coins, int(game.resources()["coins"]))
        legal = np.flatnonzero(game.legal_mask())
        if player == nn_seat:
            action = int(legal[0]) if legal.size == 1 else choose_nnmcts_action(game, player, policy, sims=sims)
        else:
            action = int(bot.choose(game))
        supply_before = dict(game.supply())
        trash_before = Counter(dict(game.trash()))
        game.step(action)
        owner, other = turn.player, 1 - turn.player
        bought = ""
        if dz.A_PLAY_BASE <= action < dz.A_PLAY_BASE + dz.ACTION_DEF_COUNT:
            name = def_name(action - dz.A_PLAY_BASE)
            if name not in TREASURES:
                turn.plays.append(name)
                actions_played[player] += 1
                last_action_card = name
        elif dz.A_BUY_BASE <= action < dz.A_BUY_BASE + dz.ACTION_DEF_COUNT:
            bought = def_name(action - dz.A_BUY_BASE)
            turn.buys.append(bought)
            decks[owner][bought] += 1
            if bought == "Province" and first_province[owner] is None:
                first_province[owner] = turn.number // 2 + 1
        for card, count in game.supply():
            taken = supply_before.get(card, 0) - count
            name = def_name(card)
            if name == bought:
                taken -= 1
                bought = ""
            for _ in range(max(0, taken)):
                if name == "Curse":
                    turn.given.append(name)
                    decks[other][name] += 1
                else:
                    turn.gains.append(name)
                    decks[owner][name] += 1
        trash_after = Counter(dict(game.trash()))
        for card, count in trash_after.items():
            name = def_name(card)
            victim = other if last_action_card == "Bandit" else owner
            for _ in range(count - trash_before.get(card, 0)):
                turn.trashes.append(name)
                decks[victim][name] -= 1
    flush()

    scores = {seat: int(game.score(seat)) for seat in (0, 1)}
    winner = game.winner()
    empty = [def_name(card) for card, count in game.supply() if count == 0]
    result = "TIE" if winner is None else ("WIN" if int(winner) == nn_seat else "LOSS")
    print(
        f"== {board} board, NET seat {nn_seat}, seed {seed}: {result} {scores[nn_seat]}-{scores[1 - nn_seat]} "
        f"after {int(game.turn()) // 2 + 1} rounds"
        f"{' (TRUNCATED)' if game.truncated() else ''}; empty piles: {', '.join(empty) or 'none'}; {time.time() - started:.0f}s",
        flush=True,
    )
    for seat in (nn_seat, 1 - nn_seat):
        cards = sum(decks[seat].values())
        print(
            f"   {labels[seat]:<8} {cards} cards: {deck_line(decks[seat])}\n"
            f"            first Province round {first_province[seat]}, "
            f"{actions_played[seat] / max(turns_taken[seat], 1):.1f} actions played per turn",
            flush=True,
        )
    return {"board": board, "seat": nn_seat, "result": result, "score": scores[nn_seat], "opponent_score": scores[1 - nn_seat]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--boards", default="sentry,chapel")
    parser.add_argument("--seats", default="0,1", help="seats the net takes; one game per board per seat")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--opponent", choices=sorted(OPPONENTS), default="engine3")
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quiet", action="store_true", help="final summaries only")
    args = parser.parse_args()

    policy = load_policy(args.checkpoint, device=args.device)
    print(f"vibe_check {args.checkpoint} vs {args.opponent}, {args.sims} sims, honest search", flush=True)
    results = []
    for board in args.boards.split(","):
        kingdom = board_cards(board, args.seed)
        for seat in (int(s) for s in args.seats.split(",")):
            print(f"\n-- {board} board ({', '.join(kingdom)}), NET in seat {seat}", flush=True)
            results.append(play_game(policy, board, kingdom, seat, args.seed, args.opponent, args.sims, args.quiet))
    print("\n" + "  ".join(f"{r['board']}/s{r['seat']} {r['result']} {r['score']}-{r['opponent_score']}" for r in results), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
