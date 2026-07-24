"""Translate tracker public state into the native shadow-game snapshot."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import dominion_v2_py as dz

from .tracker import CardMultiset, SeatSnapshot, TrackerSnapshot


BASIC_SUPPLY = (
    "Copper",
    "Silver",
    "Gold",
    "Estate",
    "Duchy",
    "Province",
    "Curse",
)
PHASES = {
    "action": "action",
    "buy": "buy",
    "cleanup": "cleanup",
}


class BridgeError(RuntimeError):
    """A tracker snapshot cannot be represented by the engine surface."""


def _def_counts(cards: CardMultiset) -> dict[int, int]:
    result: dict[int, int] = {}
    for name, count in cards:
        try:
            def_id = int(dz.def_id(name))
        except (ValueError, TypeError) as error:
            raise BridgeError(f"engine has no card definition for {name!r}") from error
        if count < 0:
            raise BridgeError(f"negative count for {name!r}: {count}")
        result[def_id] = count
    return result


def _supply(snapshot: TrackerSnapshot) -> dict[int, int]:
    tracked = dict(snapshot.supply)
    names = dict.fromkeys((*BASIC_SUPPLY, *snapshot.kingdom))
    return {
        int(dz.def_id(name)): tracked.get(name, 0)
        for name in names
    }


def _take_arbitrary(
    available: Counter[str],
    count: int,
    *,
    context: str,
) -> Counter[str]:
    taken: Counter[str] = Counter()
    remaining = count
    for name in sorted(available):
        amount = min(available[name], remaining)
        if amount:
            taken[name] = amount
            available[name] -= amount
            remaining -= amount
        if not remaining:
            break
    if remaining:
        raise BridgeError(f"{context} needs {remaining} more resolvable cards")
    return taken


def _resolved_player_zones(
    snapshot: TrackerSnapshot,
    seat: SeatSnapshot,
) -> tuple[Counter[str], Counter[str], Counter[str], Counter[str]]:
    """Resolve anonymous public-zone slots without inventing card ownership."""
    hidden = Counter(dict(seat.hand_deck))
    if seat.seat == snapshot.our_seat:
        exact_hand = Counter(dict(seat.hand))
        available = hidden - exact_hand
    else:
        available = hidden.copy()

    discard = Counter(dict(seat.discard))
    in_play = Counter(dict(seat.in_play))
    set_aside = Counter(dict(seat.set_aside_revealed))
    for zone, anonymous, context in (
        (discard, seat.discard_anonymous, "discard"),
        (in_play, seat.in_play_anonymous, "in-play"),
        (set_aside, seat.set_aside_anonymous, "set-aside"),
    ):
        resolved = _take_arbitrary(
            available,
            anonymous,
            context=f"seat {seat.seat} {context}",
        )
        zone.update(resolved)
        hidden.subtract(resolved)
        hidden += Counter()

    if sum(hidden.values()) != seat.hand_deck_count:
        raise BridgeError(
            f"seat {seat.seat} resolved hand+deck has {sum(hidden.values())} "
            f"cards, expected {seat.hand_deck_count}"
        )
    expected_counts = (
        (discard, seat.discard_count, "discard"),
        (in_play, seat.in_play_count, "in-play"),
        (set_aside, seat.set_aside_revealed_count, "set-aside"),
    )
    for zone, expected, name in expected_counts:
        if sum(zone.values()) != expected:
            raise BridgeError(
                f"seat {seat.seat} {name} composition has "
                f"{sum(zone.values())} cards, expected {expected}"
            )
    return hidden, discard, in_play, set_aside


def _player(snapshot: TrackerSnapshot, seat: SeatSnapshot) -> dict[str, object]:
    hidden, discard, in_play, set_aside = _resolved_player_zones(snapshot, seat)

    if seat.seat == snapshot.our_seat:
        if seat.hand_anonymous:
            raise BridgeError(
                f"our hand still has {seat.hand_anonymous} anonymous cards"
            )
        if sum(count for _, count in seat.hand) != seat.hand_count:
            raise BridgeError("our exact hand does not match hand_count")
        hand = _def_counts(seat.hand)
    else:
        # Opponent identities are deliberately represented only by the
        # combined hidden pool. The native builder deals an arbitrary hand of
        # the right size, and determinize() re-deals it for every search world.
        hand = {}

    return {
        "hand": hand,
        "hand_count": seat.hand_count,
        "hand_deck": _def_counts(tuple(sorted(hidden.items()))),
        "deck_count": seat.deck_count,
        "discard": _def_counts(tuple(sorted(discard.items()))),
        "in_play": _def_counts(tuple(sorted(in_play.items()))),
        "set_aside": _def_counts(tuple(sorted(set_aside.items()))),
        "actions": seat.actions,
        "buys": seat.buys,
        "coins": seat.coins,
    }


def _seeded_interrupt(snapshot: TrackerSnapshot) -> dict[str, object] | str:
    pending = snapshot.pending_decision
    if pending is None or snapshot.turn_owner == snapshot.our_seat:
        return "none"
    if snapshot.our_seat is None or snapshot.turn_owner is None:
        raise BridgeError("interrupt decision has no attacker/defender seats")

    question = pending.question_id.upper()
    association = (pending.association or "").upper()
    offered = {name.upper() for name in pending.offered}
    if question == "GAME_MAY_REACT_WITH" and "MOAT" in offered:
        kind = "moat_reaction"
    elif "MILITIA" in question or association == "MILITIA":
        kind = "militia_discard"
    elif "BUREAUCRAT" in question or association == "BUREAUCRAT":
        kind = "bureaucrat_topdeck"
    elif "BANDIT" in question or association == "BANDIT":
        kind = "bandit_trash"
    else:
        raise BridgeError(
            "opponent-turn decision is outside the base-set interrupt surface: "
            f"{pending.question_id}"
        )
    return {
        "kind": kind,
        "attacker": snapshot.turn_owner,
        "defender": snapshot.our_seat,
    }


def engine_snapshot(snapshot: TrackerSnapshot) -> dict[str, object]:
    """Return the ergonomic native snapshot dict for a tracker snapshot."""
    if snapshot.our_seat is None:
        raise BridgeError("tracker snapshot has no known local seat")
    if snapshot.turn_owner is None:
        raise BridgeError("tracker snapshot has no current turn owner")
    if snapshot.turn_number is None:
        raise BridgeError("tracker snapshot has no turn number")
    if snapshot.phase not in PHASES:
        raise BridgeError(f"unsupported tracker phase: {snapshot.phase!r}")
    if len(snapshot.seats) != len(snapshot.players):
        raise BridgeError("seat and player counts differ")
    if snapshot.trash_anonymous:
        raise BridgeError("trash contains anonymous cards")
    if sum(count for _, count in snapshot.trash) != snapshot.trash_count:
        raise BridgeError("trash composition does not match trash_count")

    return {
        "num_players": len(snapshot.players),
        "our_player": snapshot.our_seat,
        "supply": _supply(snapshot),
        "players": [_player(snapshot, seat) for seat in snapshot.seats],
        "trash": _def_counts(snapshot.trash),
        "card_totals": _def_counts(snapshot.card_totals),
        "turn_number": snapshot.turn_number,
        "phase": PHASES[snapshot.phase],
        "current_player": snapshot.turn_owner,
        "interrupt": _seeded_interrupt(snapshot),
    }


def game_from_snapshot(snapshot: TrackerSnapshot) -> dz.Game:
    """Build and validate a native shadow game from tracked public state."""
    game = dz.game_from_snapshot(engine_snapshot(snapshot))
    game.validate()
    return game


def set_deck_order(
    game: dz.Game,
    player: int,
    display_names: Iterable[str],
) -> None:
    """Rig a full deck in draw order, mapping tracker display names to defs."""
    game.set_deck_order(player, [dz.def_id(name) for name in display_names])
    game.validate()
