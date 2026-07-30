"""Standing reconstruction of the 2026-07-28 VALUE-HEAD ENGINE PROBE.

The logged protocol has 14 turn>=10 player-0 buy states on the fixed
Village/Smithy/Laboratory/Market/Chapel/Festival/Council Room/Moat/Militia/
Witch kingdom.  Its source states were not committed, but its state recipe is
deterministic: seeds 5000..5013, a money-ish driver, and the first player-0
buy node at turn>=10.  The three equal-VP variants replace the combined
deck+discard with cards placed in the deck block and an empty discard block.
Only the encoded observation is edited; this probe never runs MCTS.

The +/-0.15 acceptance tolerance is deliberately retained for future
reconstructions: the original state hashes are otherwise irretrievable and a
minor engine-state or observation-layout change can move a saturated mean.
It is broad enough for that recovery uncertainty, but narrow enough to reject
a changed value ranking or saturation regime.  It is never a tuning target.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

import dominion_v2_py as dz

from _common import (
    ENGINE_KINGDOM,
    default_output,
    evaluate_observation,
    load_checkpoint,
    seed_everything,
    write_json,
)
from src.v2.train.observation import downgrade_v3_observations


SEED = 0xA11CE
STATE_SEEDS = tuple(5000 + offset for offset in range(14))
VARIANTS: dict[str, dict[str, int]] = {
    "money": {"Copper": 7, "Silver": 3, "Gold": 2, "Estate": 3},
    "engine": {
        "Copper": 3,
        "Silver": 2,
        "Village": 2,
        "Smithy": 2,
        "Laboratory": 2,
        "Market": 1,
        "Chapel": 1,
        "Estate": 3,
    },
    "junk": {"Copper": 9, "Silver": 1, "Curse": 4, "Estate": 3},
}
REFERENCE: dict[str, dict[str, float]] = {
    "campaign15": {"money": 0.897, "engine": 0.717, "junk": -0.032},
    "campaign18": {"money": 0.819, "engine": -0.686, "junk": -0.948},
    "campaign19": {"money": -0.86, "engine": -0.98, "junk": -1.00},
}
NAME_TO_DEF = {
    "Copper": int(dz.DEF_COPPER),
    "Silver": int(dz.DEF_SILVER),
    "Gold": int(dz.DEF_GOLD),
    "Estate": int(dz.DEF_ESTATE),
    "Curse": int(dz.DEF_CURSE),
    "Village": int(dz.DEF_VILLAGE),
    "Smithy": int(dz.DEF_SMITHY),
    "Laboratory": int(dz.DEF_LABORATORY),
    "Market": int(dz.DEF_MARKET),
    "Chapel": int(dz.DEF_CHAPEL),
}
OBS_OWN_OFFSET = 4
DECK_OFFSET = 64
DISCARD_OFFSET = 128
MAX_SLOTS = int(dz.MAX_SLOTS)
OBS_SUPPLY_OFFSET = 1125
PILE_BLOCK = 11
TURN_OFFSET = 1692
DECISION_OFFSET = 1702


def _def_to_slot(observation: np.ndarray) -> dict[int, int]:
    slots: dict[int, int] = {}
    for pile in range(48):
        base = int(round(float(observation[OBS_SUPPLY_OFFSET + pile * PILE_BLOCK + 2])))
        if base > 0:
            slots[base - 1] = pile
    return slots


def _counterfactual_observation(observation: np.ndarray, composition: dict[str, int]) -> np.ndarray:
    edited = observation.copy()
    deck = edited[OBS_OWN_OFFSET + DECK_OFFSET : OBS_OWN_OFFSET + DECK_OFFSET + MAX_SLOTS]
    discard = edited[
        OBS_OWN_OFFSET + DISCARD_OFFSET : OBS_OWN_OFFSET + DISCARD_OFFSET + MAX_SLOTS
    ]
    deck.fill(0.0)
    discard.fill(0.0)
    slots = _def_to_slot(observation)
    for name, count in composition.items():
        deck[slots[NAME_TO_DEF[name]]] = float(count)
    return edited


def _collect_states() -> list[tuple[int, str, int, np.ndarray, np.ndarray]]:
    """Replay the archived deterministic money-ish, one-state-per-seed driver."""
    states: list[tuple[int, str, int, np.ndarray, np.ndarray]] = []
    silver = int(dz.A_BUY_BASE) + int(dz.DEF_SILVER)
    gold = int(dz.A_BUY_BASE) + int(dz.DEF_GOLD)
    treasure_defs = {int(dz.DEF_COPPER), int(dz.DEF_SILVER), int(dz.DEF_GOLD)}
    for seed in STATE_SEEDS:
        game = dz.new_game(dz.Setup(players=2, kingdom=list(ENGINE_KINGDOM)), seed)
        for _ in range(600):
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
            buys = [
                int(action)
                for action in actions
                if int(dz.A_BUY_BASE) <= int(action) < int(dz.A_SELECT_BASE)
            ]
            # The archived driver always advances every treasure before it
            # considers the buy node.  In particular it does not capture the
            # pre-treasure PhaseBuy frame merely because a buy action is legal.
            if treasures:
                game.step(treasures[0])
                continue
            if buys:
                observation = game.encode(0, 3)
                turn = int(round(float(observation[TURN_OFFSET + 7])))
                player_zero_to_act = int(round(float(observation[DECISION_OFFSET + 11]))) == 1
                if turn >= 10 and player_zero_to_act:
                    states.append(
                        (
                            seed,
                            f"0x{int(game.state_hash()):016x}",
                            turn,
                            observation.copy(),
                            legal.copy(),
                        )
                    )
                    break
            if buys:
                game.step(gold if bool(legal[gold]) else (silver if bool(legal[silver]) else buys[0]))
            else:
                game.step(int(actions[0]))
    if len(states) != 14:
        raise RuntimeError(f"archived value protocol collected {len(states)} states, expected 14")
    return states


def _reference_for(checkpoint: str | Path) -> tuple[str | None, dict[str, float] | None]:
    lowered = str(checkpoint).lower()
    for key, values in REFERENCE.items():
        if key in lowered:
            return key, values
    return None, None


def run(checkpoint: str | Path, *, legacy_shim: bool = False) -> dict[str, Any]:
    seed_everything(SEED)
    policy = load_checkpoint(checkpoint, legacy_shim=legacy_shim)
    if policy.obs_version not in (2, 3):
        raise RuntimeError("value probe requires a v2/v3 checkpoint")
    states = _collect_states()
    rows: list[dict[str, Any]] = []
    by_variant: dict[str, list[float]] = {name: [] for name in VARIANTS}
    for index, (seed, state_hash, turn, observation, legal_mask) in enumerate(states):
        values: dict[str, float] = {}
        for name, composition in VARIANTS.items():
            edited = _counterfactual_observation(observation, composition)
            model_observation = (
                downgrade_v3_observations(edited) if policy.obs_version == 2 else edited
            )
            _, value = evaluate_observation(policy, model_observation, legal_mask)
            values[name] = value
            by_variant[name].append(value)
        rows.append(
            {
                "index": index,
                "seed": seed,
                "state_hash": state_hash,
                "turn_counter": turn,
                "acting_seat": 0,
                "values": values,
            }
        )
    means = {name: float(np.mean(values)) for name, values in by_variant.items()}
    reference_name, reference = _reference_for(checkpoint)
    if reference is None:
        acceptance: dict[str, Any] = {"checked": False}
    else:
        within = {name: abs(means[name] - reference[name]) <= 0.15 for name in VARIANTS}
        ordering = means["money"] > means["engine"] > means["junk"]
        saturation = all(means[name] <= -0.8 for name in VARIANTS)
        regime = (
            ordering and means["junk"] <= 0.15
            if reference_name == "campaign15"
            else (means["engine"] < -0.5 if reference_name == "campaign18" else saturation)
        )
        acceptance = {
            "checked": True,
            "reference": reference,
            "tolerance": 0.15,
            "within_tolerance": within,
            "ordering_money_engine_junk": ordering,
            "saturated_negative": saturation,
            "regime_pass": regime,
            "pass": bool(all(within.values()) and regime),
        }
    return {
        "probe": "value_probe",
        "checkpoint": str(checkpoint),
        "obs_version": policy.obs_version,
        "seed": SEED,
        "protocol": {
            "kingdom": list(ENGINE_KINGDOM),
            "states": 14,
            "minimum_turn_counter": 10,
            "state_seeds": list(STATE_SEEDS),
            "variants": VARIANTS,
            "mode": "raw value head; encoded deck+discard counterfactual only",
        },
        "means": means,
        "acceptance": acceptance,
        "states": rows,
    }


def scorecard(result: dict[str, Any]) -> str:
    means = result["means"]
    acceptance = result["acceptance"]
    status = "n/a" if not acceptance["checked"] else ("PASS" if acceptance["pass"] else "CHECK")
    return (
        "value_probe "
        f"money={means['money']:+.3f} engine={means['engine']:+.3f} "
        f"junk={means['junk']:+.3f} acceptance={status}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--legacy-shim", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run(args.checkpoint, legacy_shim=args.legacy_shim)
    output = write_json(args.out or default_output(args.checkpoint, "value_probe"), result)
    print(scorecard(result))
    print(f"json={output}")


if __name__ == "__main__":
    main()
