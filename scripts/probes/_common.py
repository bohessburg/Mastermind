"""Small shared utilities for the standing policy-behaviour probes.

The helpers deliberately stay close to the native Python surface: checkpoint
loading goes through the arena's factory-aware loader and fixtures use the
engine's validated snapshot builder.  That lets a probe change only the cards
it names while retaining a genuine legal Dominion decision.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

# Invoking ``python scripts/probes/name.py`` puts the probe directory, rather
# than the repository root, on sys.path.  Keep the documented
# ``PYTHONPATH=build`` invocation sufficient for the ``src`` package too.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import dominion_v2_py as dz
from src.v2.arena.bot.policy import NNPolicy, load_policy


ENGINE_KINGDOM = (
    "Village",
    "Smithy",
    "Laboratory",
    "Market",
    "Chapel",
    "Festival",
    "Council Room",
    "Moat",
    "Militia",
    "Witch",
)

BASIC_TOTALS = {
    "Copper": 60,
    "Silver": 40,
    "Gold": 30,
    "Estate": 14,
    "Duchy": 8,
    "Province": 8,
    "Curse": 10,
}


def seed_everything(seed: int) -> None:
    """Fix the Python/NumPy RNGs used while generating probe fixtures."""
    random.seed(seed)
    np.random.seed(seed)


def load_checkpoint(path: str | Path, *, legacy_shim: bool = False) -> NNPolicy:
    """Use the arena loader without overriding its checkpoint layout inference."""
    return load_policy(Path(path), device="cpu", legacy_shim=legacy_shim)


def ids(counts: Mapping[str, int]) -> dict[int, int]:
    return {
        int(dz.def_id(name)): int(count)
        for name, count in counts.items()
        if int(count) > 0
    }


def names(counts: Mapping[int, int]) -> dict[str, int]:
    """Readable counter with stable names; probes supply their own name map."""
    return {str(int(card)): int(count) for card, count in sorted(counts.items()) if count}


def add_counts(*groups: Mapping[str, int]) -> Counter[str]:
    total: Counter[str] = Counter()
    for group in groups:
        total.update({name: int(count) for name, count in group.items()})
    return total


def _player(
    *,
    hand: Mapping[str, int] | None = None,
    deck: Mapping[str, int] | None = None,
    discard: Mapping[str, int] | None = None,
    in_play: Mapping[str, int] | None = None,
    set_aside: Mapping[str, int] | None = None,
    actions: int = 0,
    buys: int = 1,
    coins: int = 0,
    hidden_hand_count: int | None = None,
) -> tuple[dict[str, Any], Counter[str]]:
    """Make one validated native-snapshot player and its total card counter."""
    hand_counter = Counter(hand or {})
    deck_counter = Counter(deck or {})
    discard_counter = Counter(discard or {})
    in_play_counter = Counter(in_play or {})
    set_aside_counter = Counter(set_aside or {})
    hand_deck = hand_counter + deck_counter
    total = hand_deck + discard_counter + in_play_counter + set_aside_counter
    return (
        {
            "hand": ids(hand_counter),
            "hand_count": int(sum(hand_counter.values()) if hidden_hand_count is None else hidden_hand_count),
            "hand_deck": ids(hand_deck),
            "deck_count": int(sum(deck_counter.values()) if hidden_hand_count is None else sum(hand_deck.values()) - hidden_hand_count),
            "discard": ids(discard_counter),
            "in_play": ids(in_play_counter),
            "set_aside": ids(set_aside_counter),
            "actions": int(actions),
            "buys": int(buys),
            "coins": int(coins),
        },
        total,
    )


def snapshot_game(
    *,
    kingdom: Iterable[str],
    player0: Mapping[str, Any],
    player1: Mapping[str, Any],
    turn: int,
    phase: str,
    current_player: int,
    our_player: int = 0,
    interrupt: str | Mapping[str, Any] = "none",
    trash: Mapping[str, int] | None = None,
) -> Any:
    """Build a legal two-player snapshot, deriving supply from all zones.

    The snapshot builder requires exact card conservation.  Probe callers give
    zones only; this function subtracts them from the normal two-player supply
    totals, including the ten one-pile copies of each kingdom card.
    """
    ordered_kingdom = tuple(kingdom)
    first, first_total = _player(**player0)
    second, second_total = _player(**player1)
    trash_counter = Counter(trash or {})
    totals: Counter[str] = Counter(BASIC_TOTALS)
    totals.update({name: 10 for name in ordered_kingdom})
    occupied = first_total + second_total + trash_counter
    supply = totals - occupied
    missing = occupied - totals
    if missing:
        raise ValueError(f"fixture uses cards beyond a normal supply: {dict(missing)}")
    snapshot = {
        "num_players": 2,
        "our_player": int(our_player),
        "supply": ids(supply),
        "kingdom_order": [int(dz.def_id(name)) for name in ordered_kingdom],
        "players": [first, second],
        "trash": ids(trash_counter),
        "card_totals": ids(totals),
        "turn_number": int(turn),
        "phase": phase,
        "current_player": int(current_player),
        "interrupt": interrupt,
    }
    game = dz.game_from_snapshot(snapshot)
    game.validate()
    return game


def evaluate(policy: NNPolicy, game: Any, seat: int) -> tuple[np.ndarray, float]:
    """Return legal-action probabilities and the value head for one decision."""
    return evaluate_observation(policy, game.encode(seat, policy.obs_version), game.legal_mask())


def evaluate_observation(
    policy: NNPolicy,
    observation: np.ndarray,
    legal_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Evaluate an already encoded counterfactual using the original legal mask."""
    torch = policy.torch
    observation_tensor = torch.as_tensor(
        observation, dtype=torch.float32, device=policy.device
    ).unsqueeze(0)
    mask = torch.as_tensor(
        legal_mask, dtype=torch.bool, device=policy.device
    ).unsqueeze(0)
    with torch.no_grad():
        logits, values = policy.evaluate(observation_tensor, mask)
        probabilities = torch.softmax(logits, dim=-1)
    return (
        probabilities.squeeze(0).detach().cpu().numpy().astype(np.float64, copy=False),
        float(values.squeeze().detach().cpu().item()),
    )


def fixed_buy_states(
    *,
    kingdom: Iterable[str],
    seeds: Iterable[int],
    min_player_turn: int,
    max_player_turn: int | None,
    wanted: int,
) -> list[tuple[Any, int]]:
    """Collect one post-treasure buy node per turn from seeded EngineV3 games.

    This is a fixed, scripted-play source of genuine engine states.  Treasure
    plays are advanced in the same deterministic Gold/Silver/Copper order the
    arena treats as forced, leaving a policy-relevant purchase decision.
    """
    states: list[tuple[Any, int]] = []
    treasure_defs = (int(dz.DEF_GOLD), int(dz.DEF_SILVER), int(dz.DEF_COPPER))
    ordered_kingdom = list(kingdom)
    for seed in seeds:
        game = dz.new_game(dz.Setup(players=2, kingdom=ordered_kingdom), int(seed))
        bots = (dz.ScriptedBot("engine3"), dz.ScriptedBot("engine3"))
        recorded_turns: set[tuple[int, int]] = set()
        guard = 0
        while not game.game_over() and len(states) < wanted:
            guard += 1
            if guard > 4000:
                raise RuntimeError("scripted fixture game exceeded its decision guard")
            decision = game.current_decision()
            seat = int(decision["player"])
            if int(decision["kind"]) == 2:  # PhaseBuy
                legal = game.legal_mask()
                treasure = next(
                    (
                        int(dz.A_PLAY_BASE) + def_id
                        for def_id in treasure_defs
                        if bool(legal[int(dz.A_PLAY_BASE) + def_id])
                    ),
                    None,
                )
                if treasure is not None:
                    game.step(treasure)
                    continue
                player_turn = int(game.turn()) // 2 + 1
                key = (int(game.turn()), seat)
                if (
                    player_turn >= min_player_turn
                    and (max_player_turn is None or player_turn <= max_player_turn)
                    and key not in recorded_turns
                ):
                    states.append((game.clone(), seat))
                    recorded_turns.add(key)
                    if len(states) >= wanted:
                        break
            action = int(bots[seat].choose(game))
            if not bool(game.legal_mask()[action]):
                raise RuntimeError("EngineV3 supplied an illegal fixture action")
            game.step(action)
        if len(states) >= wanted:
            break
    if len(states) != wanted:
        raise RuntimeError(f"only collected {len(states)} of {wanted} requested buy states")
    return states


def card_name(def_id: int) -> str:
    """Card names are exposed by the web definitions table, not the extension."""
    from src.v2.web.server.defs import def_name

    return str(def_name(int(def_id)))


def buy_action(card: str) -> int:
    return int(dz.A_BUY_BASE) + int(dz.def_id(card))


def select_action(card: str) -> int:
    return int(dz.A_SELECT_BASE) + int(dz.def_id(card))


def action_card(action: int, *, base: int) -> str | None:
    def_id = int(action) - int(base)
    if 0 <= def_id < int(dz.ACTION_DEF_COUNT):
        return card_name(def_id)
    return None


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def checked_step(game: Any, action: int, *, context: str = "probe") -> None:
    """Advance one native game action after checking the public legal mask.

    The native binding validates this too, but probes should fail at the
    decision that produced a bad action rather than silently recording a
    replay recipe which can never be reconstructed.
    """
    action = int(action)
    legal = game.legal_mask()
    if action < 0 or action >= int(dz.ACTION_SPACE_SIZE) or not bool(legal[action]):
        raise RuntimeError(f"{context} selected illegal action {action}")
    game.step(action)


def replay_recipe(recipe: Mapping[str, Any]) -> Any:
    """Reconstruct a sampled state from its deterministic action history.

    The pybind module deliberately does not expose a persistent GameState
    serializer.  Probe fixtures therefore use this portable recipe instead of
    depending on ``Game.clone()`` or any in-memory state representation.
    """
    try:
        seed = int(recipe["seed"])
        kingdom = [int(def_id) for def_id in recipe["kingdom_def_ids"]]
        actions = [int(action) for action in recipe["actions"]]
        ply_index = int(recipe["ply_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid replay recipe") from error
    if ply_index < 0 or ply_index > len(actions):
        raise ValueError("replay recipe ply_index is outside its action list")

    game = dz.new_game(dz.Setup(players=2, kingdom=kingdom), seed)
    for ply, action in enumerate(actions[:ply_index]):
        checked_step(game, action, context=f"replay ply {ply}")
    return game


def read_sampled_states(path: str | Path) -> list[dict[str, Any]]:
    """Read either a per-bucket or combined sampled-probe JSON document."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise RuntimeError(f"could not read sampled states: {source}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"sampled states JSON is invalid: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError("sampled states JSON must contain an object")

    states = payload.get("states")
    if states is None and isinstance(payload.get("buckets"), dict):
        states = [
            state
            for bucket_states in payload["buckets"].values()
            if isinstance(bucket_states, list)
            for state in bucket_states
        ]
    if not isinstance(states, list) or not states:
        raise ValueError("sampled states JSON contains no states")

    normalized: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        if not isinstance(state, dict) or not isinstance(state.get("replay"), dict):
            raise ValueError(f"sampled state {index} has no replay recipe")
        normalized.append(state)
    return normalized


def default_output(checkpoint: str | Path, stem: str) -> Path:
    checkpoint_name = Path(checkpoint).stem
    return Path("artifacts") / "probes" / f"{stem}-{checkpoint_name}.json"
