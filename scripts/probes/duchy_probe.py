"""Standing C15 score-snapshot bias / c17 margin-blend regression probe.

The cited C15 entry used 42 mid-game buy states and an observation-only edit:
give the opponent two Duchies (+6 VP), leaving the acting player's deck
untouched.  C15 logged a +3.7 percentage-point Duchy-buy shift and -0.21
value shift.  The c17 verification logged +0.2 points and -0.001 after
margin_blend.  This version regenerates 42 fixed EngineV3-play mid-game buy
states, then changes exactly the opponent perfect-memory collection slot for
Duchy by +2.  It does not run MCTS.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

import dominion_v2_py as dz

from _common import (
    ENGINE_KINGDOM,
    buy_action,
    default_output,
    evaluate_observation,
    fixed_buy_states,
    load_checkpoint,
    seed_everything,
    write_json,
)


SEED = 0xD0C4
STATE_SEEDS = tuple(SEED + offset for offset in range(128))
STATE_COUNT = 42
MAX_SLOTS = int(dz.MAX_SLOTS)
OBS_V2_OPPONENT_OFFSET = 4 + 5 * MAX_SLOTS
OBS_V2_OPPONENT_COLLECTION_OFFSET = 75
DUCHY_SLOT = 4  # canonical basic-pile slot: Copper, Silver, Gold, Estate, Duchy
REFERENCE = {
    "campaign15": {"p_buy_duchy_delta_points": 3.7, "value_delta": -0.21},
    "campaign17": {"p_buy_duchy_delta_points": 0.2, "value_delta": -0.001},
}


def _add_two_opponent_duchies(observation: np.ndarray) -> np.ndarray:
    edited = observation.copy()
    collection_index = (
        OBS_V2_OPPONENT_OFFSET + OBS_V2_OPPONENT_COLLECTION_OFFSET + DUCHY_SLOT
    )
    if collection_index >= edited.size:
        raise ValueError("checkpoint observation layout has no v2 opponent collection block")
    edited[collection_index] += 2.0
    return edited


def _reference(checkpoint: str | Path) -> dict[str, float] | None:
    lowered = str(checkpoint).lower()
    return next((value for key, value in REFERENCE.items() if key in lowered), None)


def run(checkpoint: str | Path, *, legacy_shim: bool = False) -> dict[str, Any]:
    seed_everything(SEED)
    policy = load_checkpoint(checkpoint, legacy_shim=legacy_shim)
    if policy.obs_version == 1:
        raise RuntimeError("duchy probe requires v2/v3 opponent collection observations")
    states = fixed_buy_states(
        kingdom=ENGINE_KINGDOM,
        seeds=STATE_SEEDS,
        min_player_turn=10,
        max_player_turn=None,
        wanted=STATE_COUNT,
    )
    duchy_action = buy_action("Duchy")
    rows: list[dict[str, Any]] = []
    deltas_probability: list[float] = []
    deltas_value: list[float] = []
    for index, (game, seat) in enumerate(states):
        observation = game.encode(seat, policy.obs_version)
        legal = game.legal_mask()
        base_probs, base_value = evaluate_observation(policy, observation, legal)
        injected_probs, injected_value = evaluate_observation(
            policy, _add_two_opponent_duchies(observation), legal
        )
        p_delta = float(injected_probs[duchy_action] - base_probs[duchy_action])
        v_delta = float(injected_value - base_value)
        deltas_probability.append(p_delta)
        deltas_value.append(v_delta)
        rows.append(
            {
                "index": index,
                "state_hash": f"0x{int(game.state_hash()):016x}",
                "turn_counter": int(game.turn()),
                "player_turn": int(game.turn()) // 2 + 1,
                "acting_seat": seat,
                "duchy_legal": bool(legal[duchy_action]),
                "base_p_buy_duchy": float(base_probs[duchy_action]),
                "injected_p_buy_duchy": float(injected_probs[duchy_action]),
                "p_buy_duchy_delta": p_delta,
                "base_value": base_value,
                "injected_value": injected_value,
                "value_delta": v_delta,
            }
        )
    means = {
        "p_buy_duchy_delta": float(np.mean(deltas_probability)),
        "p_buy_duchy_delta_points": float(100.0 * np.mean(deltas_probability)),
        "value_delta": float(np.mean(deltas_value)),
    }
    return {
        "probe": "duchy_probe",
        "checkpoint": str(checkpoint),
        "obs_version": policy.obs_version,
        "seed": SEED,
        "protocol": {
            "states": STATE_COUNT,
            "minimum_player_turn": 10,
            "counterfactual": "opponent collection Duchy count +2 (+6 VP)",
            "mode": "raw policy/value heads only",
        },
        "means": means,
        "reference": _reference(checkpoint),
        "states": rows,
    }


def scorecard(result: dict[str, Any]) -> str:
    means = result["means"]
    reference = result.get("reference")
    suffix = ""
    if reference is not None:
        suffix = (
            f" reference=({reference['p_buy_duchy_delta_points']:+.3f}pts,"
            f"{reference['value_delta']:+.3f})"
        )
    return (
        "duchy_probe "
        f"P(Duchy) delta={means['p_buy_duchy_delta_points']:+.3f}pts "
        f"value delta={means['value_delta']:+.4f}{suffix}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--legacy-shim", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run(args.checkpoint, legacy_shim=args.legacy_shim)
    output = write_json(args.out or default_output(args.checkpoint, "duchy_probe"), result)
    print(scorecard(result))
    print(f"json={output}")


if __name__ == "__main__":
    main()
