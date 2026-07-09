from __future__ import annotations

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
