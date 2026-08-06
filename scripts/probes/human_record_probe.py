"""v1 protocol: replay local human games and measure raw-policy imitation.

This is intentionally labelled v1 because it is the first standing version of
the human-record instrument.  It follows each replayable local export exactly,
using ``is_complete_local_export_data(data)`` before replaying.  Exports whose
seat kinds contain pytest paths and zero-action stubs are excluded separately.
At every human buy of an Action costing at least $3 it records the raw policy
mass on the action actually bought and whether the policy's best *buy* is a
Treasure.  No MCTS is run.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import dominion_v2_py as dz

from _common import default_output, evaluate, load_checkpoint, seed_everything, write_json
from src.v2.records.local import is_complete_local_export_data
from src.v2.web.server.defs import def_by_id, def_name


SEED = 0x4A11
MIN_GAMES_FOR_MEANINGFUL = 10
MIN_DECISIONS_FOR_MEANINGFUL = 20


def _is_action_costing_three(def_id: int) -> bool:
    card = def_by_id(def_id)
    return "Action" in card["types"] and int(card["cost"]["coins"]) >= 3


def _buy_argmax_is_treasure(probabilities: np.ndarray, legal: np.ndarray) -> tuple[str | None, bool]:
    start = int(dz.A_BUY_BASE)
    stop = start + int(dz.ACTION_DEF_COUNT)
    legal_buys = np.flatnonzero(legal[start:stop]) + start
    if legal_buys.size == 0:
        return None, False
    action = int(legal_buys[int(np.argmax(probabilities[legal_buys]))])
    def_id = action - start
    card = def_by_id(def_id)
    return str(card["name"]), "Treasure" in card["types"]


def _outcome(game: Any, human_seat: int) -> str:
    own = int(game.score(human_seat))
    other_scores = [int(game.score(seat)) for seat in range(int(game.num_players())) if seat != human_seat]
    if own > max(other_scores):
        return "won"
    if own < max(other_scores):
        return "lost"
    return "tied"


def _empty_bucket() -> dict[str, Any]:
    return {"probability_mass": [], "money_defaults": 0, "decisions": 0, "games": set()}


def _summarize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    decisions = int(bucket["decisions"])
    games = len(bucket["games"])
    return {
        "qualifying_decisions": decisions,
        "qualifying_games": games,
        "mean_probability_mass": (
            float(np.mean(bucket["probability_mass"])) if decisions else None
        ),
        "money_default_cases": int(bucket["money_defaults"]),
        "money_default_rate": (
            float(bucket["money_defaults"] / decisions) if decisions else None
        ),
    }


def run(
    checkpoint: str | Path,
    exports: str | Path = "exports",
    *,
    legacy_shim: bool = False,
) -> dict[str, Any]:
    seed_everything(SEED)
    policy = load_checkpoint(checkpoint, legacy_shim=legacy_shim)
    export_root = Path(exports)
    buckets: dict[str, dict[str, Any]] = defaultdict(_empty_bucket)
    records: list[dict[str, Any]] = []
    game_outcomes: dict[str, dict[int, str]] = {}
    counts: dict[str, int] = defaultdict(int)

    for path in sorted(export_root.rglob("*.json")):
        counts["files_seen"] += 1
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            counts["invalid_json"] += 1
            continue
        seats = [str(kind) for kind in data.get("seats", [])] if isinstance(data, dict) else []
        actions = data.get("actions", []) if isinstance(data, dict) else []
        if not actions:
            counts["zero_action_stubs"] += 1
            continue
        if any("pytest" in kind.lower() for kind in seats):
            counts["pytest_path_exports"] += 1
            continue
        if not is_complete_local_export_data(data):
            counts["incomplete_or_unreplayable"] += 1
            continue
        human_seats = [seat for seat, kind in enumerate(seats) if kind.lower() == "human"]
        if not human_seats:
            counts["no_human_seat"] += 1
            continue
        counts["qualifying_games"] += 1

        game = dz.new_game(
            dz.Setup(players=len(seats), kingdom=[int(card) for card in data["kingdom"]]),
            int(data["seed"]),
        )
        pending: list[dict[str, Any]] = []
        for index, raw_action in enumerate(data["actions"]):
            action = int(raw_action)
            decision = game.current_decision()
            actor = int(decision["player"])
            legal = game.legal_mask()
            if actor in human_seats and int(decision["kind"]) == 2:
                start = int(dz.A_BUY_BASE)
                def_id = action - start
                if start <= action < start + int(dz.ACTION_DEF_COUNT) and _is_action_costing_three(def_id):
                    probabilities, _ = evaluate(policy, game, actor)
                    argmax_name, argmax_treasure = _buy_argmax_is_treasure(probabilities, legal)
                    pending.append(
                        {
                            "source": str(path),
                            "action_index": index,
                            "human_seat": actor,
                            "chosen_def": def_id,
                            "chosen_card": def_name(def_id),
                            "chosen_probability_mass": float(probabilities[action]),
                            "argmax_buy": argmax_name,
                            "money_default": bool(argmax_treasure),
                            "state_hash": f"0x{int(game.state_hash()):016x}",
                            "turn_counter": int(game.turn()),
                        }
                    )
            if not bool(legal[action]):
                raise RuntimeError(f"complete export replay became illegal: {path} action {index}")
            game.step(action)

        outcomes = {seat: _outcome(game, seat) for seat in human_seats}
        game_outcomes[str(path)] = outcomes
        for row in pending:
            outcome = outcomes[row["human_seat"]]
            row["outcome"] = outcome
            bucket = buckets[outcome]
            bucket["probability_mass"].append(row["chosen_probability_mass"])
            bucket["money_defaults"] += int(row["money_default"])
            bucket["decisions"] += 1
            bucket["games"].add(row["source"])
            records.append(row)

    split = {outcome: _summarize_bucket(buckets[outcome]) for outcome in ("won", "lost", "tied")}
    won_lost_games = split["won"]["qualifying_games"] + split["lost"]["qualifying_games"]
    won_lost_decisions = split["won"]["qualifying_decisions"] + split["lost"]["qualifying_decisions"]
    meaningful = (
        won_lost_games >= MIN_GAMES_FOR_MEANINGFUL
        and won_lost_decisions >= MIN_DECISIONS_FOR_MEANINGFUL
    )
    return {
        "probe": "human_record_probe",
        "protocol_version": "v1",
        "checkpoint": str(checkpoint),
        "obs_version": policy.obs_version,
        "exports": str(export_root),
        "filter_counts": dict(sorted(counts.items())),
        "meaningful_sample": meaningful,
        "sample_note": (
            "enough qualifying win/loss games and decisions for a directional comparison"
            if meaningful
            else (
                "too few replayable qualifying human games/decisions for a reliable statistic; "
                "the split is reported for inspection only"
            )
        ),
        "by_outcome": split,
        "records": records,
        "game_outcomes": game_outcomes,
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def scorecard(result: dict[str, Any]) -> str:
    split = result["by_outcome"]
    won = split["won"]
    lost = split["lost"]
    warning = "" if result["meaningful_sample"] else " sample=TOO_FEW"
    return (
        "human_record_probe[v1] "
        f"won(mass={_fmt(won['mean_probability_mass'])},default={_fmt(won['money_default_rate'])},n={won['qualifying_decisions']}) "
        f"lost(mass={_fmt(lost['mean_probability_mass'])},default={_fmt(lost['money_default_rate'])},n={lost['qualifying_decisions']})"
        f"{warning}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--legacy-shim", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--exports", type=Path, default=Path("exports"))
    args = parser.parse_args()
    result = run(args.checkpoint, args.exports, legacy_shim=args.legacy_shim)
    output = write_json(args.out or default_output(args.checkpoint, "human_record_probe"), result)
    print(scorecard(result))
    print(f"json={output}")


if __name__ == "__main__":
    main()
