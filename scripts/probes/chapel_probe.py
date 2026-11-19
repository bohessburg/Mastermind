"""Standing reconstruction of c18's ROOT CAUSE OF THE CAP Chapel probe.

The cited protocol is raw-policy only: make ten seeded Chapel kingdoms with
one attack and ten without, drive three seeded games per kingdom with a
net-independent money-ish policy, then collect up to four Chapel-affordable
turn-1..4 buy nodes from each game.  That yields the logged 120 attack and
120 no-attack nodes.  No MCTS is run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

import dominion_v2_py as dz

from _common import (
    default_output,
    evaluate_observation,
    load_checkpoint,
    seed_everything,
    write_json,
)
from src.v2.train.observation import downgrade_v3_observations


SEED = 7
POOL = (
    "Cellar", "Moat", "Harbinger", "Merchant", "Vassal", "Village",
    "Workshop", "Bureaucrat", "Gardens", "Moneylender", "Poacher", "Remodel",
    "Smithy", "Throne Room", "Bandit", "Council Room", "Festival", "Laboratory",
    "Library", "Market", "Mine", "Sentry",
)
ATTACKS = ("Witch", "Militia")
OBS_DECISION_OFFSET = 1702
TURN_OFFSET = 1692
N_KINGDOMS = 10
SEEDS_PER_KINGDOM = 3
NODES_PER_GAME = 4


def _kingdoms() -> tuple[list[list[str]], list[list[str]]]:
    rng = np.random.default_rng(SEED)

    def make(with_attack: bool) -> list[list[str]]:
        result: list[list[str]] = []
        for _ in range(N_KINGDOMS):
            count = 8 if with_attack else 9
            rest = list(rng.choice(POOL, size=count, replace=False))
            kingdom = ["Chapel", *rest]
            if with_attack:
                kingdom.append(str(rng.choice(ATTACKS, size=1)[0]))
            result.append(kingdom[:10])
        return result

    # Preserve the original shared-generator order: attack set first.
    return make(True), make(False)


def _buy_nodes(
    kingdom: list[str], seed: int
) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    """Exact fixed driver from the original session script."""
    game = dz.new_game(dz.Setup(players=2, kingdom=kingdom), seed)
    chapel_buy = int(dz.A_BUY_BASE) + int(dz.DEF_CHAPEL)
    nodes: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    treasure_defs = {int(dz.DEF_COPPER), int(dz.DEF_SILVER), int(dz.DEF_GOLD)}
    silver_buy = int(dz.A_BUY_BASE) + int(dz.DEF_SILVER)
    for _ in range(400):
        legal = game.legal_mask()
        actions = np.flatnonzero(legal)
        if actions.size == 0:
            break
        treasures = [
            int(action)
            for action in actions
            if int(dz.A_PLAY_BASE) <= int(action) < int(dz.A_BUY_BASE)
            and int(action) - int(dz.A_PLAY_BASE) in treasure_defs
        ]
        if bool(legal[chapel_buy]):
            for seat in (0, 1):
                observation = game.encode(seat, 3)
                if int(round(float(observation[OBS_DECISION_OFFSET + 11]))) == 1:
                    turn = int(round(float(observation[TURN_OFFSET + 7])))
                    if turn <= 4:
                        nodes.append(
                            (
                                f"0x{int(game.state_hash()):016x}",
                                turn,
                                observation.copy(),
                                legal.copy(),
                            )
                        )
                    break
        if treasures:
            game.step(treasures[0])
        else:
            game.step(silver_buy if bool(legal[silver_buy]) else int(actions[0]))
        if len(nodes) >= NODES_PER_GAME:
            break
    return nodes


def _node_sets() -> dict[str, list[dict[str, Any]]]:
    attack_kingdoms, no_attack_kingdoms = _kingdoms()
    groups: dict[str, list[dict[str, Any]]] = {"attack_present": [], "no_attack": []}
    for label, kingdoms, seed_base in (
        ("attack_present", attack_kingdoms, 1000),
        ("no_attack", no_attack_kingdoms, 2000),
    ):
        for kingdom_index, kingdom in enumerate(kingdoms):
            for offset in range(SEEDS_PER_KINGDOM):
                seed = seed_base + offset
                for state_hash, turn, observation, legal in _buy_nodes(kingdom, seed):
                    groups[label].append(
                        {
                            "kingdom_index": kingdom_index,
                            "kingdom": kingdom,
                            "seed": seed,
                            "state_hash": state_hash,
                            "turn_counter": turn,
                            "observation": observation,
                            "legal_mask": legal,
                        }
                    )
    for label, nodes in groups.items():
        if len(nodes) != 120:
            raise RuntimeError(f"Chapel protocol collected {len(nodes)} {label} nodes, expected 120")
    return groups


def _reference(checkpoint: str | Path) -> dict[str, float] | None:
    lowered = str(checkpoint).lower()
    if "campaign18" in lowered:
        return {"no_attack": 0.070, "attack_present": 0.056}
    if "campaign15" in lowered:
        return {"no_attack": 0.036, "attack_present": 0.036}
    return None


def run(checkpoint: str | Path, *, legacy_shim: bool = False) -> dict[str, Any]:
    seed_everything(SEED)
    policy = load_checkpoint(checkpoint, legacy_shim=legacy_shim)
    groups = _node_sets()
    chapel_action = int(dz.A_BUY_BASE) + int(dz.DEF_CHAPEL)
    output_groups: dict[str, Any] = {}
    means: dict[str, float] = {}
    for label, nodes in groups.items():
        rows: list[dict[str, Any]] = []
        masses: list[float] = []
        for index, node in enumerate(nodes):
            observation = node["observation"]
            model_observation = (
                downgrade_v3_observations(observation) if policy.obs_version == 2 else observation
            )
            probabilities, _ = evaluate_observation(policy, model_observation, node["legal_mask"])
            mass = float(probabilities[chapel_action])
            masses.append(mass)
            rows.append(
                {
                    "index": index,
                    "kingdom_index": node["kingdom_index"],
                    "kingdom": node["kingdom"],
                    "seed": node["seed"],
                    "state_hash": node["state_hash"],
                    "turn_counter": node["turn_counter"],
                    "p_buy_chapel": mass,
                }
            )
        means[label] = float(np.mean(masses))
        output_groups[label] = {"nodes": len(rows), "states": rows}
    return {
        "probe": "chapel_probe",
        "checkpoint": str(checkpoint),
        "obs_version": policy.obs_version,
        "seed": SEED,
        "protocol": {
            "nodes": 240,
            "attack_nodes": 120,
            "no_attack_nodes": 120,
            "turns": "turn counters 0..4 (logged turn-1..4 opening window)",
            "mode": "raw unforced policy only",
            "reference_note": (
                "c18 logged no-attack 0.070 and attack 0.056; c15 champion was flat "
                "around 0.036. These are scorecard comparisons, not gates."
            ),
        },
        "means": means,
        "reference": _reference(checkpoint),
        "groups": output_groups,
    }


def scorecard(result: dict[str, Any]) -> str:
    means = result["means"]
    reference = result.get("reference")
    suffix = ""
    if reference is not None:
        suffix = (
            f" reference(no-attack={reference['no_attack']:.3f},"
            f"attack={reference['attack_present']:.3f})"
        )
    return (
        "chapel_probe "
        f"attack-present={means['attack_present']:.3f} "
        f"no-attack={means['no_attack']:.3f}{suffix}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--legacy-shim", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run(args.checkpoint, legacy_shim=args.legacy_shim)
    output = write_json(args.out or default_output(args.checkpoint, "chapel_probe"), result)
    print(scorecard(result))
    print(f"json={output}")


if __name__ == "__main__":
    main()
