"""Sample replayable mid-game probe states from low-search mirror self-play.

Each emitted state is a replay recipe rather than a native ``Game`` snapshot:
``seed`` and ``kingdom_def_ids`` recreate the initial game, while ``actions``
up to ``ply_index`` recreate the sampled decision exactly.  That keeps the
files portable across Python processes and independent of pybind internals.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import dominion_v2_py as dz

from _common import checked_step, load_checkpoint, seed_everything, write_json
from src.v2.arena.bot.policy import choose_nnmcts_action
from src.v2.web.server.defs import kingdom_def_ids, load_defs


DEFAULT_SEED = 0x5A7E5
DEFAULT_GAMES = 384
DEFAULT_SIMS = 64
DEFAULT_TARGET_PER_BUCKET = 40
MIN_TURN_COUNTER = 10
MAX_TURN_COUNTER = 30


def _definition_data() -> dict[int, dict[str, Any]]:
    return {int(card["id"]): card for card in load_defs()["defs"]}


def _slot_def_ids(game: Any) -> list[int]:
    """Return the engine slot-to-definition order via the public supply view.

    New games add one slot per supply pile before any card can move.  The
    binding exposes pile definitions in that same stable order, which lets us
    translate the own-zone count fields in a v3 observation without inspecting
    native state memory.
    """
    return [int(def_id) for def_id, _count in game.supply()]


def _deck_composition(game: Any, seat: int) -> Counter[int]:
    """Count all cards owned by ``seat`` from its five v3 own-zone fields."""
    observation = game.encode(seat, 3)
    slots = _slot_def_ids(game)
    zone_size = int(dz.MAX_SLOTS)
    own_zones = observation[4 : 4 + 5 * zone_size].reshape(5, zone_size)
    counts_by_slot = own_zones.sum(axis=0)
    if len(slots) > zone_size:
        raise RuntimeError("engine exposed more supply slots than observation supports")
    composition: Counter[int] = Counter()
    for slot, count in enumerate(counts_by_slot[: len(slots)]):
        rounded = int(round(float(count)))
        if rounded:
            composition[slots[slot]] = rounded
    return composition


def _bucket_for(composition: Counter[int], definitions: dict[int, dict[str, Any]]) -> str | None:
    total = sum(composition.values())
    if total <= 0:
        return None
    treasures = sum(
        count
        for def_id, count in composition.items()
        if "Treasure" in definitions[def_id].get("types", [])
    )
    actions = sum(
        count
        for def_id, count in composition.items()
        if "Action" in definitions[def_id].get("types", [])
    )
    curses = int(composition.get(int(dz.DEF_CURSE), 0))
    # Keep the bad-deck bucket first: a cursed money deck is useful junk data,
    # not a substitute for an ordinary money state.
    if curses >= 3:
        return "junk"
    if actions >= 4 and 100 * treasures <= 40 * total:
        return "engine"
    if 100 * treasures >= 60 * total and actions <= 2:
        return "money"
    return None


def _composition_summary(
    composition: Counter[int], definitions: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    total = sum(composition.values())
    treasures = sum(
        count
        for def_id, count in composition.items()
        if "Treasure" in definitions[def_id].get("types", [])
    )
    actions = sum(
        count
        for def_id, count in composition.items()
        if "Action" in definitions[def_id].get("types", [])
    )
    return {
        "cards": {
            str(definitions[def_id]["name"]): int(count)
            for def_id, count in sorted(composition.items())
        },
        "total_cards": int(total),
        "treasure_cards": int(treasures),
        "action_cards": int(actions),
        "curse_cards": int(composition.get(int(dz.DEF_CURSE), 0)),
        "treasure_fraction": float(treasures / total) if total else 0.0,
    }


def _is_post_treasure_duchy_buy(game: Any, legal: np.ndarray) -> bool:
    """Limit samples to real purchase nodes where P(Buy Duchy) is meaningful."""
    decision = game.current_decision()
    duchy_action = int(dz.A_BUY_BASE) + int(dz.DEF_DUCHY)
    if int(decision["kind"]) != 2 or not bool(legal[duchy_action]):  # PhaseBuy
        return False
    for def_id in (int(dz.DEF_GOLD), int(dz.DEF_SILVER), int(dz.DEF_COPPER)):
        if bool(legal[int(dz.A_PLAY_BASE) + def_id]):
            return False
    return True


def _record_state(
    *,
    game: Any,
    seat: int,
    history: list[int],
    kingdom: list[int],
    seed: int,
    game_index: int,
    mirror_index: int,
    bucket: str,
    composition: Counter[int],
    definitions: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "bucket": bucket,
        "game_index": int(game_index),
        "mirror_index": int(mirror_index),
        "acting_seat": int(seat),
        "turn_counter": int(game.turn()),
        "state_hash": f"0x{int(game.state_hash()):016x}",
        "deck_composition": _composition_summary(composition, definitions),
        "replay": {
            "seed": int(seed),
            "kingdom_def_ids": [int(def_id) for def_id in kingdom],
            "actions": [int(action) for action in history],
            "ply_index": len(history),
        },
    }


def _play_game(
    *,
    policy: Any,
    kingdom: list[int],
    seed: int,
    sims: int,
    game_index: int,
    mirror_index: int,
    buckets: dict[str, list[dict[str, Any]]],
    target_per_bucket: int,
    definitions: dict[int, dict[str, Any]],
) -> None:
    game = dz.new_game(dz.Setup(players=2, kingdom=kingdom), seed)
    history: list[int] = []
    sampled_turns: set[tuple[int, int]] = set()
    guard = 0
    while not game.game_over():
        guard += 1
        if guard > 4000:
            raise RuntimeError(f"sample game {game_index} exceeded its decision guard")
        legal = game.legal_mask()
        legal_actions = np.flatnonzero(legal)
        if legal_actions.size == 0:
            raise RuntimeError(f"sample game {game_index} has no legal action")
        decision = game.current_decision()
        seat = int(decision["player"])
        turn_counter = int(game.turn())
        key = (turn_counter, seat)
        if (
            MIN_TURN_COUNTER <= turn_counter <= MAX_TURN_COUNTER
            and key not in sampled_turns
            and _is_post_treasure_duchy_buy(game, legal)
        ):
            composition = _deck_composition(game, seat)
            bucket = _bucket_for(composition, definitions)
            if bucket is not None and len(buckets[bucket]) < target_per_bucket:
                buckets[bucket].append(
                    _record_state(
                        game=game,
                        seat=seat,
                        history=history,
                        kingdom=kingdom,
                        seed=seed,
                        game_index=game_index,
                        mirror_index=mirror_index,
                        bucket=bucket,
                        composition=composition,
                        definitions=definitions,
                    )
                )
            sampled_turns.add(key)

        action = int(choose_nnmcts_action(game, seat, policy, sims=sims, determinizations=1))
        checked_step(game, action, context=f"sample game {game_index} ply {len(history)}")
        history.append(action)


def sample(
    checkpoint: str | Path,
    *,
    games: int = DEFAULT_GAMES,
    sims: int = DEFAULT_SIMS,
    target_per_bucket: int = DEFAULT_TARGET_PER_BUCKET,
    seed: int = DEFAULT_SEED,
    legacy_shim: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Collect all three buckets, pairing games on each random kingdom."""
    if games <= 0 or games % 2:
        raise ValueError("--games must be a positive even number for mirrored pairs")
    if sims <= 0 or target_per_bucket <= 0:
        raise ValueError("--sims and --target-per-bucket must be positive")

    seed_everything(seed)
    policy = load_checkpoint(checkpoint, legacy_shim=legacy_shim)
    definitions = _definition_data()
    pool = kingdom_def_ids()
    if len(pool) < 10:
        raise RuntimeError("kingdom definition pool contains fewer than ten cards")
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {"money": [], "engine": [], "junk": []}

    for pair_index in range(games // 2):
        kingdom = sorted(int(def_id) for def_id in rng.sample(pool, 10))
        # The two independently shuffled deals on this board are the mirrored
        # self-play pair.  Both seats use the same policy and both may supply
        # sampled decisions, avoiding a seat-specific source distribution.
        for mirror_index in range(2):
            game_index = 2 * pair_index + mirror_index
            game_seed = rng.randrange(1, 2**63)
            _play_game(
                policy=policy,
                kingdom=kingdom,
                seed=game_seed,
                sims=sims,
                game_index=game_index,
                mirror_index=mirror_index,
                buckets=buckets,
                target_per_bucket=target_per_bucket,
                definitions=definitions,
            )
        sizes = {label: len(states) for label, states in buckets.items()}
        print(f"sample_states pairs={pair_index + 1} buckets={sizes}", flush=True)
        if all(len(states) >= target_per_bucket for states in buckets.values()):
            break
    return buckets


def _payload(
    *,
    label: str,
    checkpoint: Path,
    seed: int,
    games: int,
    sims: int,
    target_per_bucket: int,
    states: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format": "dominionzero.probe_states.v1",
        "label": label,
        "checkpoint": str(checkpoint),
        "states": states,
        "protocol": {
            "source": "low-search mirrored NN self-play",
            "games_budget": int(games),
            "mcts_sims": int(sims),
            "turn_counter_window": [MIN_TURN_COUNTER, MAX_TURN_COUNTER],
            "buy_node": "post-treasure PhaseBuy with Duchy legal",
            "target_per_bucket": int(target_per_bucket),
            "replay": "new_game(seed, kingdom_def_ids), then step actions[:ply_index]",
        },
        "bucket_size": len(states),
        "seed": int(seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--target-per-bucket", type=int, default=DEFAULT_TARGET_PER_BUCKET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--legacy-shim", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("bench"))
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write partial files instead of failing if a game budget misses a bucket",
    )
    args = parser.parse_args()

    buckets = sample(
        args.checkpoint,
        games=args.games,
        sims=args.sims,
        target_per_bucket=args.target_per_bucket,
        seed=args.seed,
        legacy_shim=args.legacy_shim,
    )
    missing = {
        label: args.target_per_bucket - len(states)
        for label, states in buckets.items()
        if len(states) < args.target_per_bucket
    }
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"sample budget did not fill requested buckets: {missing}; increase --games"
        )
    for label, states in buckets.items():
        output = args.out_dir / f"probe_states_{label}.json"
        write_json(
            output,
            _payload(
                label=label,
                checkpoint=args.checkpoint,
                seed=args.seed,
                games=args.games,
                sims=args.sims,
                target_per_bucket=args.target_per_bucket,
                states=states,
            ),
        )
        print(f"sample_states {label}={len(states)} json={output}", flush=True)


if __name__ == "__main__":
    main()
