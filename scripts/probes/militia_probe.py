"""Standing reconstruction of the human-games-analysis Militia keep-inversion.

The cited section explicitly gives three complete hands: [Copper, Copper,
Estate, Silver], [Copper, Copper, Silver, Silver, Silver], and [Gold,
Estate, Estate, Silver, Silver].  It also mentions a Gardens/Copper/Gold
incident and a hand containing own Militia, but does not preserve those full
five-card hands; this probe deliberately does not fabricate their missing
cards.  Each retained fixture is built headlessly as a seeded native
``militia_discard`` interrupt and kept cards are the three Select decisions.

Raw reads are greedy policy reads.  Search uses the arena's own
``choose_nnmcts_action`` wrapper, whose DecisionSearcher construction uses
the production 400 simulations, K=2 determinizations, c_puct=1.25,
auto-played treasures, and state-derived seed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import dominion_v2_py as dz

from _common import (
    ENGINE_KINGDOM,
    action_card,
    card_name,
    default_output,
    evaluate,
    load_checkpoint,
    seed_everything,
    select_action,
    snapshot_game,
    write_json,
)
from src.v2.arena.bot.policy import choose_nnmcts_action


SEED = 0xA11171A
KEEP_COUNT = 3
HANDS: tuple[tuple[str, Counter[str]], ...] = (
    ("copper_copper_estate_silver", Counter({"Copper": 2, "Estate": 1, "Silver": 1})),
    ("two_copper_three_silver", Counter({"Copper": 2, "Silver": 3})),
    ("two_estates_gold_two_silver_control", Counter({"Estate": 2, "Gold": 1, "Silver": 2})),
)


def _game(hand: Counter[str]) -> Any:
    return snapshot_game(
        kingdom=ENGINE_KINGDOM,
        player0={
            "hand": {},
            "deck": {"Copper": 7, "Estate": 3},
            "in_play": {"Militia": 1},
            "actions": 0,
            "buys": 1,
            "coins": 0,
        },
        player1={
            "hand": dict(hand),
            "deck": {"Copper": 5, "Estate": 2},
            "actions": 1,
            "buys": 1,
            "coins": 0,
        },
        turn=20,
        phase="action",
        current_player=0,
        our_player=1,
        interrupt={"kind": "militia_discard", "attacker": 0, "defender": 1},
    )


def _counter_as_list(cards: Iterable[str]) -> list[str]:
    return sorted(cards, key=lambda name: (int(dz.def_id(name)), name))


def _counter_as_dict(counter: Counter[str]) -> dict[str, int]:
    return {name: int(counter[name]) for name in _counter_as_list(counter.elements()) if counter[name]}


def _keep_by_raw_policy(policy: Any, hand: Counter[str]) -> tuple[list[str], float]:
    game = _game(hand)
    keeps: list[str] = []
    first_value = 0.0
    while int(game.current_decision()["player"]) == 1 and int(game.current_decision()["kind"]) == 4:
        probabilities, value = evaluate(policy, game, 1)
        if not keeps:
            first_value = value
        action = int(probabilities.argmax())
        name = action_card(action, base=int(dz.A_SELECT_BASE))
        if name is None or not bool(game.legal_mask()[action]):
            raise RuntimeError("raw policy did not return a legal Militia selection")
        keeps.append(name)
        game.step(action)
    return _counter_as_list(keeps), first_value


def _keep_by_arena_search(policy: Any, hand: Counter[str]) -> list[str]:
    game = _game(hand)
    keeps: list[str] = []
    while int(game.current_decision()["player"]) == 1 and int(game.current_decision()["kind"]) == 4:
        action = choose_nnmcts_action(
            game,
            1,
            policy,
            sims=400,
            determinizations=2,
        )
        name = action_card(action, base=int(dz.A_SELECT_BASE))
        if name is None or not bool(game.legal_mask()[action]):
            raise RuntimeError("arena DecisionSearcher did not return a legal Militia selection")
        keeps.append(name)
        game.step(action)
    return _counter_as_list(keeps)


def _unique_keeps(hand: Counter[str], wanted: int = KEEP_COUNT) -> list[Counter[str]]:
    names = sorted(hand, key=lambda name: int(dz.def_id(name)))
    answer: list[Counter[str]] = []

    def visit(index: int, remaining: int, selected: Counter[str]) -> None:
        if index == len(names):
            if remaining == 0:
                answer.append(selected.copy())
            return
        name = names[index]
        for count in range(min(hand[name], remaining) + 1):
            if count:
                selected[name] = count
            visit(index + 1, remaining - count, selected)
            selected.pop(name, None)

    visit(0, wanted, Counter())
    return answer


def _value_optimal_keep(policy: Any, hand: Counter[str]) -> tuple[list[str], float, list[dict[str, Any]]]:
    ranked: list[tuple[float, Counter[str]]] = []
    for keep in _unique_keeps(hand):
        game = _game(hand)
        for name in _counter_as_list(keep.elements()):
            game.step(select_action(name))
        _, value = evaluate(policy, game, 1)
        ranked.append((value, keep))
    ranked.sort(key=lambda item: (-item[0], _counter_as_list(item[1].elements())))
    best_value, best = ranked[0]
    table = [
        {"keep": _counter_as_list(keep.elements()), "value": value}
        for value, keep in ranked
    ]
    return _counter_as_list(best.elements()), best_value, table


def _discarded(hand: Counter[str], kept: list[str]) -> list[str]:
    cards = hand.copy()
    cards.subtract(Counter(kept))
    cards += Counter()
    return _counter_as_list(cards.elements())


def run(checkpoint: str | Path, *, legacy_shim: bool = False) -> dict[str, Any]:
    seed_everything(SEED)
    policy = load_checkpoint(checkpoint, legacy_shim=legacy_shim)
    cases: list[dict[str, Any]] = []
    for label, hand in HANDS:
        raw_keep, raw_root_value = _keep_by_raw_policy(policy, hand)
        search_keep = _keep_by_arena_search(policy, hand)
        optimal_keep, optimal_value, ranking = _value_optimal_keep(policy, hand)
        cases.append(
            {
                "label": label,
                "hand": _counter_as_list(hand.elements()),
                "raw_policy_keep": raw_keep,
                "raw_policy_discard": _discarded(hand, raw_keep),
                "search_keep": search_keep,
                "search_discard": _discarded(hand, search_keep),
                "value_optimal_keep": optimal_keep,
                "value_optimal_discard": _discarded(hand, optimal_keep),
                "raw_matches_value_optimal": raw_keep == optimal_keep,
                "search_matches_value_optimal": search_keep == optimal_keep,
                "raw_root_value": raw_root_value,
                "value_optimal_post_resolution_value": optimal_value,
                "value_ranking": ranking,
            }
        )
    raw_inversion = any(not case["raw_matches_value_optimal"] for case in cases)
    search_inversion = any(not case["search_matches_value_optimal"] for case in cases)
    return {
        "probe": "militia_probe",
        "checkpoint": str(checkpoint),
        "obs_version": policy.obs_version,
        "seed": SEED,
        "protocol": {
            "sims": 400,
            "determinizations": 2,
            "mode": "raw greedy policy plus production arena DecisionSearcher",
            "omitted_incomplete_log_cases": [
                "Gardens/Copper/Gold incident (full hand not logged)",
                "own-Militia incident (full hand not logged)",
            ],
        },
        "summary": {
            "raw_policy": "inversion_present" if raw_inversion else "fixed",
            "arena_search": "inversion_present" if search_inversion else "fixed",
        },
        "cases": cases,
    }


def scorecard(result: dict[str, Any]) -> str:
    summary = result["summary"]
    mismatches = [case["label"] for case in result["cases"] if not case["raw_matches_value_optimal"]]
    return (
        "militia_probe "
        f"raw={summary['raw_policy']} search={summary['arena_search']} "
        f"raw_mismatch_cases={','.join(mismatches) or 'none'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--legacy-shim", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run(args.checkpoint, legacy_shim=args.legacy_shim)
    output = write_json(args.out or default_output(args.checkpoint, "militia_probe"), result)
    print(scorecard(result))
    print(f"json={output}")


if __name__ == "__main__":
    main()
