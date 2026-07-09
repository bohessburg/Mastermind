from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

import dominion_v2_py as dz

from .defs import def_by_id, def_name


DECISION_KIND = {
    0: "None",
    1: "PhaseAction",
    2: "PhaseBuy",
    3: "PhaseNight",
    4: "Choose",
    5: "ChooseGain",
    6: "ChooseOption",
    7: "ChooseOrder",
    8: "ReactWindow",
    9: "OrderTriggers",
}


def decision_kind_name(decision: dict[str, Any]) -> str:
    return DECISION_KIND.get(int(decision.get("kind", 0)), "Unknown")


def action_def(action: int, base: int) -> int:
    return int(action) - int(base)


def action_category(action: int) -> str:
    if action == dz.A_PASS:
        return "pass"
    if dz.A_PLAY_BASE <= action < dz.A_PLAY_BASE + dz.ACTION_DEF_COUNT:
        return "play"
    if dz.A_BUY_BASE <= action < dz.A_BUY_BASE + dz.ACTION_DEF_COUNT:
        return "buy"
    if dz.A_SELECT_BASE <= action < dz.A_OPTION_BASE:
        return "select"
    if dz.A_OPTION_BASE <= action < dz.A_CALL_BASE:
        return "option"
    return "other"


def prompt_for(decision: dict[str, Any]) -> str:
    kind = decision_kind_name(decision)
    source = int(decision.get("source", 0))
    source_name = def_name(source) if source >= 0 else ""

    if kind == "PhaseAction":
        return "Action phase"
    if kind == "PhaseBuy":
        return "Buy phase"
    if kind == "PhaseNight":
        return "Night phase"
    if kind == "ReactWindow":
        return f"{source_name}: reveal a Reaction?"
    if kind == "OrderTriggers":
        return "Choose the next trigger to resolve"
    if kind == "ChooseGain":
        if source_name == "Workshop":
            return "Workshop: gain a card costing up to 4"
        if source_name == "Remodel":
            return "Remodel: gain a card costing up to 2 more"
        if source_name == "Mine":
            return "Mine: gain a Treasure to your hand"
        if source_name == "Artisan":
            return "Artisan: gain a card to your hand costing up to 5"
        return f"{source_name}: gain a card"
    if kind == "Choose":
        if source_name == "Militia":
            return "Militia: discard down to 3 cards"
        if source_name == "Chapel":
            return "Chapel: trash up to 4 cards"
        if source_name == "Cellar":
            return "Cellar: discard any number of cards"
        if source_name == "Poacher":
            return "Poacher: discard for empty Supply piles"
        if source_name == "Bandit":
            return "Bandit: trash one revealed Treasure"
        if source_name == "Bureaucrat":
            return "Bureaucrat: put a Victory card onto your deck"
        if source_name == "Harbinger":
            return "Harbinger: put a discard card onto your deck"
        if source_name == "Artisan":
            return "Artisan: put a card from your hand onto your deck"
        if source_name == "Throne Room":
            return "Throne Room: choose an Action to play twice"
        return f"{source_name}: choose cards"
    if kind == "ChooseOption":
        if source_name == "Library":
            return "Library: keep or set aside the Action"
        if source_name == "Sentry":
            return "Sentry: choose what to do with the looked-at card"
        if source_name == "Vassal":
            return "Vassal: play the discarded Action?"
        return f"{source_name}: choose an option"
    if kind == "ChooseOrder":
        if source_name == "Sentry":
            return "Sentry: choose card order"
        return f"{source_name}: choose an order"
    return "Waiting"


def _select_verb(decision: dict[str, Any]) -> str:
    kind = decision_kind_name(decision)
    source_name = def_name(int(decision.get("source", 0)))
    if kind == "ChooseGain":
        return "Gain"
    if kind == "ReactWindow":
        return "Reveal"
    if source_name in {"Chapel", "Remodel", "Mine", "Moneylender", "Bandit", "ExactTwoTest"}:
        return "Trash"
    if source_name in {"Cellar", "Poacher"}:
        return "Discard"
    if source_name == "Militia":
        return "Keep"
    if source_name in {"Bureaucrat", "Harbinger", "Artisan"}:
        return "Topdeck"
    if source_name == "Throne Room":
        return "Play"
    return "Select"


def _option_label(decision: dict[str, Any], option: int) -> str:
    kind = decision_kind_name(decision)
    source_name = def_name(int(decision.get("source", 0)))
    if kind == "OrderTriggers":
        return f"Resolve trigger {option + 1}"
    if kind == "ChooseOrder":
        return f"Position {option + 1}"
    if source_name == "Library":
        return "Keep Action" if option == 0 else "Set aside Action"
    if source_name == "Sentry":
        return ["Trash", "Discard", "Keep"][option] if option < 3 else f"Option {option + 1}"
    if source_name == "Vassal":
        return "Decline" if option == 0 else "Play Action"
    return f"Option {option + 1}"


def label_for_action(action: int, decision: dict[str, Any]) -> tuple[str, int | None]:
    category = action_category(action)
    if category == "pass":
        kind = decision_kind_name(decision)
        return ("Done" if kind in {"Choose", "ChooseGain"} else "Pass", None)
    if category == "play":
        def_id = action_def(action, dz.A_PLAY_BASE)
        return f"Play {def_name(def_id)}", def_id
    if category == "buy":
        def_id = action_def(action, dz.A_BUY_BASE)
        return f"Buy {def_name(def_id)}", def_id
    if category == "select":
        def_id = action_def(action, dz.A_SELECT_BASE)
        return f"{_select_verb(decision)} {def_name(def_id)}", def_id
    if category == "option":
        option = action_def(action, dz.A_OPTION_BASE)
        return _option_label(decision, option), None
    return f"Action {action}", None


def legal_options(mask: np.ndarray, decision: dict[str, Any]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for action in np.flatnonzero(mask):
        action_int = int(action)
        label, def_id = label_for_action(action_int, decision)
        option: dict[str, Any] = {"action": action_int, "label": label}
        if def_id is not None:
            option["def"] = def_id
            option["name"] = def_by_id(def_id)["name"]
        options.append(option)
    return options


def log_line(seat: int, action: int, decision: dict[str, Any]) -> str:
    label, _ = label_for_action(action, decision)
    words = label.split(maxsplit=1)
    if not words:
        return f"P{seat + 1} acts"
    verb = words[0].lower()
    rest = f" {words[1]}" if len(words) > 1 else ""
    if verb == "pass":
        return f"P{seat + 1} passes"
    if verb == "done":
        return f"P{seat + 1} finishes choosing"
    return f"P{seat + 1} {verb}s{rest}"


@dataclass(frozen=True)
class PlayerPublicSnapshot:
    hand_count: int
    deck_count: int
    discard: tuple[int, ...]
    discard_top: int | None
    set_aside_count: int


@dataclass(frozen=True)
class PublicSnapshot:
    players: tuple[PlayerPublicSnapshot, ...]
    trash: dict[int, int]


@dataclass(frozen=True)
class LogContext:
    seat: int
    action: int
    decision: dict[str, Any]
    after_decision: dict[str, Any] | None
    before: PublicSnapshot
    after: PublicSnapshot
    generic: str
    source_name: str


def capture_public_snapshot(game: Any) -> PublicSnapshot:
    players: list[PlayerPublicSnapshot] = []
    for player in range(int(game.num_players())):
        discard = tuple(int(def_value) for def_value in game.discard(player))
        players.append(
            PlayerPublicSnapshot(
                hand_count=int(game.hand_count(player)),
                deck_count=int(game.deck_count(player)),
                discard=discard,
                discard_top=int(game.discard_top(player)) if game.discard_top(player) is not None else None,
                set_aside_count=len(game.set_aside(player)),
            )
        )
    return PublicSnapshot(
        players=tuple(players),
        trash={int(def_value): int(count) for def_value, count in game.trash().items()},
    )


def _card_list(defs: list[int]) -> str:
    if not defs:
        return "nothing"
    names = [def_name(def_value) for def_value in defs]
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _discard_added(before: PlayerPublicSnapshot, after: PlayerPublicSnapshot) -> list[int]:
    if len(after.discard) >= len(before.discard) and after.discard[: len(before.discard)] == before.discard:
        return list(after.discard[len(before.discard) :])

    before_counts = Counter(before.discard)
    added: list[int] = []
    for def_value in after.discard:
        if before_counts[def_value] > 0:
            before_counts[def_value] -= 1
        else:
            added.append(def_value)
    return added


def _trash_added(before: PublicSnapshot, after: PublicSnapshot) -> list[int]:
    added: list[int] = []
    for def_value, count in sorted(after.trash.items()):
        delta = count - before.trash.get(def_value, 0)
        added.extend([def_value] * max(0, delta))
    return added


def _source_name(action: int, decision: dict[str, Any]) -> str:
    if action_category(action) == "play":
        return def_name(action_def(action, dz.A_PLAY_BASE))
    source = int(decision.get("source", -1))
    return def_name(source) if source >= 0 else ""


def _format_bandit(context: LogContext) -> list[str]:
    trash_added = _trash_added(context.before, context.after)
    category = action_category(context.action)
    lines: list[str] = []
    for player, (before, after) in enumerate(zip(context.before.players, context.after.players)):
        if category == "play" and player == context.seat:
            continue
        discard_added = _discard_added(before, after)
        if not discard_added and not trash_added:
            continue
        revealed = trash_added + discard_added
        trashed = _card_list(trash_added) if trash_added else "nothing"
        lines.append(f"P{player + 1} reveals {_card_list(revealed)}; trashes {trashed}")
    return lines


def _format_militia(context: LogContext) -> list[str]:
    lines: list[str] = []
    for player, (before, after) in enumerate(zip(context.before.players, context.after.players)):
        discarded_count = max(0, len(after.discard) - len(before.discard))
        if discarded_count == 0:
            continue
        top = def_name(after.discard_top) if after.discard_top is not None else "nothing"
        plural = "card" if discarded_count == 1 else "cards"
        lines.append(f"P{player + 1} discards {discarded_count} {plural}; discard top is now {top}")
    return lines


def _format_witch(context: LogContext) -> list[str]:
    lines: list[str] = []
    curse = int(dz.DEF_CURSE)
    for player, (before, after) in enumerate(zip(context.before.players, context.after.players)):
        if player == context.seat:
            continue
        if curse in _discard_added(before, after):
            lines.append(f"P{player + 1} gains a Curse")
    return lines


def _format_bureaucrat(context: LogContext) -> list[str]:
    category = action_category(context.action)
    after_kind = decision_kind_name(context.after_decision or {})
    after_source = int((context.after_decision or {}).get("source", -1))
    attack_still_pending = after_source == int(dz.DEF_BUREAUCRAT) and after_kind in {
        "ReactWindow",
        "Choose",
    }
    lines: list[str] = []
    for player, (before, after) in enumerate(zip(context.before.players, context.after.players)):
        if category == "play" and player == context.seat:
            continue
        if after.hand_count < before.hand_count and after.deck_count > before.deck_count:
            lines.append(f"P{player + 1} topdecks a Victory card")
        elif category == "play" and not attack_still_pending and after.hand_count == before.hand_count:
            lines.append(f"P{player + 1} reveals a hand with no Victory cards")
    return lines


def _format_sentry(context: LogContext) -> list[str]:
    before = context.before.players[context.seat]
    after = context.after.players[context.seat]
    trash_added = _trash_added(context.before, context.after)
    discard_added = _discard_added(before, after)
    parts: list[str] = []
    if trash_added:
        parts.append(f"trashes {_card_list(trash_added)}")
    if discard_added:
        parts.append(f"discards {_card_list(discard_added)}")
    if parts:
        return [f"P{context.seat + 1} {' and '.join(parts)} (Sentry)"]

    put_back = max(0, after.deck_count - before.deck_count)
    if put_back > 0:
        plural = "card" if put_back == 1 else "cards"
        return [f"P{context.seat + 1} puts {put_back} {plural} back (Sentry)"]
    return []


PUBLIC_FORMATTERS = {
    "Bandit": _format_bandit,
    "Militia": _format_militia,
    "Witch": _format_witch,
    "Bureaucrat": _format_bureaucrat,
    "Sentry": _format_sentry,
}


def _generic_line_is_private(context: LogContext) -> bool:
    category = action_category(context.action)
    return (
        (context.source_name == "Militia" and category == "select")
        or (context.source_name == "Bureaucrat" and category == "select")
    )


def public_log_lines(
    seat: int,
    action: int,
    decision: dict[str, Any],
    before: PublicSnapshot,
    after: PublicSnapshot,
    after_decision: dict[str, Any] | None = None,
) -> list[str]:
    generic = log_line(seat, action, decision)
    source_name = _source_name(action, decision)
    context = LogContext(seat, action, decision, after_decision, before, after, generic, source_name)
    formatter = PUBLIC_FORMATTERS.get(source_name)
    details = formatter(context) if formatter is not None else []
    if _generic_line_is_private(context):
        return details
    return [generic, *details] if details else [generic]
