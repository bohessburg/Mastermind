from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
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


def _context_subjects(context: dict[str, Any] | None) -> list[int]:
    if not context:
        return []
    return [int(def_value) for def_value in context.get("subject_defs", [])]


def _context_subject_index(context: dict[str, Any] | None) -> int | None:
    if not context or context.get("subject_index") is None:
        return None
    return int(context["subject_index"])


def _current_subject_name(context: dict[str, Any] | None) -> str:
    subjects = _context_subjects(context)
    index = _context_subject_index(context)
    if index is None or index < 0 or index >= len(subjects):
        return ""
    return def_name(subjects[index])


def prompt_for(decision: dict[str, Any], context: dict[str, Any] | None = None) -> str:
    kind = decision_kind_name(decision)
    source = int(decision.get("source", 0))
    source_name = def_name(source) if source >= 0 else ""
    subjects = _context_subjects(context)
    current_name = _current_subject_name(context)

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
        if source_name == "Library" and current_name:
            return f"Library: drew {current_name} - set it aside?"
        if source_name == "Sentry" and subjects:
            return f"Sentry: you look at {_card_list(subjects)}"
        if source_name == "Vassal" and current_name:
            return f"Vassal: discarded {current_name} - play it?"
        if source_name == "Library":
            return "Library: keep or set aside the Action"
        if source_name == "Sentry":
            return "Sentry: choose what to do with the looked-at card"
        if source_name == "Vassal":
            return "Vassal: play the discarded Action?"
        return f"{source_name}: choose an option"
    if kind == "ChooseOrder":
        if source_name == "Sentry" and subjects:
            return f"Sentry: put {_card_list(subjects)} back in order"
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


def _option_label(decision: dict[str, Any], option: int, context: dict[str, Any] | None = None) -> str:
    kind = decision_kind_name(decision)
    source_name = def_name(int(decision.get("source", 0)))
    subjects = _context_subjects(context)
    current_name = _current_subject_name(context)
    if kind == "OrderTriggers":
        return f"Resolve trigger {option + 1}"
    if kind == "ChooseOrder":
        if source_name == "Sentry" and option < len(subjects):
            return f"Put {def_name(subjects[option])} on top (drawn next)"
        return f"Position {option + 1}"
    if source_name == "Library" and current_name:
        return f"Keep {current_name}" if option == 0 else f"Set aside {current_name}"
    if source_name == "Sentry" and current_name:
        return [f"Trash {current_name}", f"Discard {current_name}", f"Keep {current_name}"][option] if option < 3 else f"Option {option + 1}"
    if source_name == "Vassal" and current_name:
        return f"Do not play {current_name}" if option == 0 else f"Play {current_name}"
    if source_name == "Library":
        return "Keep Action" if option == 0 else "Set aside Action"
    if source_name == "Sentry":
        return ["Trash", "Discard", "Keep"][option] if option < 3 else f"Option {option + 1}"
    if source_name == "Vassal":
        return "Decline" if option == 0 else "Play Action"
    return f"Option {option + 1}"


def label_for_action(action: int, decision: dict[str, Any], context: dict[str, Any] | None = None) -> tuple[str, int | None]:
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
        return _option_label(decision, option, context), None
    return f"Action {action}", None


def legal_options(mask: np.ndarray, decision: dict[str, Any], context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for action in np.flatnonzero(mask):
        action_int = int(action)
        label, def_id = label_for_action(action_int, decision, context)
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
    suffix = "es" if verb.endswith(("s", "x", "z", "ch", "sh")) else "s"
    return f"P{seat + 1} {verb}{suffix}{rest}"


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
    decision_context: dict[str, Any] | None
    after_decision: dict[str, Any] | None
    before: PublicSnapshot
    after: PublicSnapshot
    generic: str
    source_name: str


@dataclass
class SentryLogState:
    looked: list[int] = field(default_factory=list)
    trashed: list[int] = field(default_factory=list)
    discarded: list[int] = field(default_factory=list)
    kept: list[int] = field(default_factory=list)
    kept_on_top: int | None = None


@dataclass
class BanditLogState:
    hits_by_attacker: dict[int, int] = field(default_factory=dict)


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


def _bandit_revealed_discards(before: PlayerPublicSnapshot, after: PlayerPublicSnapshot) -> list[int]:
    if len(after.discard) >= len(before.discard) and after.discard[: len(before.discard)] == before.discard:
        return list(after.discard[len(before.discard) :])
    return list(after.discard)


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
    return []


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
    return []


def _format_library(context: LogContext) -> list[str]:
    if action_category(context.action) != "option":
        return []
    option = action_def(context.action, dz.A_OPTION_BASE)
    current = _current_subject_from_log_context(context)
    if current is None or option != 1:
        return []
    return [f"P{context.seat + 1} sets aside {def_name(current)} (Library)"]


def _format_vassal(context: LogContext) -> list[str]:
    if action_category(context.action) != "option":
        return []
    option = action_def(context.action, dz.A_OPTION_BASE)
    current = _current_subject_from_log_context(context)
    if current is None:
        return []
    if option == 1:
        return [f"P{context.seat + 1} plays {def_name(current)} (Vassal)"]
    return [f"P{context.seat + 1} declines to play {def_name(current)} (Vassal)"]


def _current_subject_from_log_context(context: LogContext) -> int | None:
    subject_defs = _context_subjects(context.decision_context)
    subject_index = _context_subject_index(context.decision_context)
    if subject_index is None or subject_index < 0 or subject_index >= len(subject_defs):
        return None
    return int(subject_defs[subject_index])


PUBLIC_FORMATTERS = {
    "Bandit": _format_bandit,
    "Militia": _format_militia,
    "Witch": _format_witch,
    "Bureaucrat": _format_bureaucrat,
    "Sentry": _format_sentry,
    "Library": _format_library,
    "Vassal": _format_vassal,
}


def _generic_line_is_private(context: LogContext) -> bool:
    category = action_category(context.action)
    return (
        (context.source_name == "Militia" and category == "select")
        or (context.source_name == "Bandit" and category == "select")
        or (context.source_name == "Bureaucrat" and category == "select")
        or (context.source_name in {"Sentry", "Library", "Vassal"} and category == "option")
    )


def public_log_lines(
    seat: int,
    action: int,
    decision: dict[str, Any],
    before: PublicSnapshot,
    after: PublicSnapshot,
    decision_context: dict[str, Any] | None = None,
    after_decision: dict[str, Any] | None = None,
) -> list[str]:
    generic = log_line(seat, action, decision)
    source_name = _source_name(action, decision)
    context = LogContext(seat, action, decision, decision_context, after_decision, before, after, generic, source_name)
    formatter = PUBLIC_FORMATTERS.get(source_name)
    details = formatter(context) if formatter is not None else []
    if _generic_line_is_private(context):
        return details
    return [generic, *details] if details else [generic]


def _sentry_completion_pending(
    decision: dict[str, Any],
    decision_context: dict[str, Any] | None,
    after_decision: dict[str, Any] | None,
) -> bool:
    kind = decision_kind_name(decision)
    if kind == "ChooseOrder":
        return True
    if kind != "ChooseOption":
        return False

    subjects = _context_subjects(decision_context)
    index = _context_subject_index(decision_context)
    if index is None or index + 1 < len(subjects):
        return False

    after_kind = decision_kind_name(after_decision or {})
    after_source = int((after_decision or {}).get("source", -1))
    return not (after_source == int(dz.DEF_SENTRY) and after_kind == "ChooseOrder")


def _sentry_public_summary(seat: int, state: SentryLogState) -> list[str]:
    parts: list[str] = []
    if state.trashed:
        parts.append(f"trashes {_card_list(state.trashed)}")
    if state.discarded:
        parts.append(f"discards {_card_list(state.discarded)}")
    if state.kept:
        plural = "card" if len(state.kept) == 1 else "cards"
        parts.append(f"keeps {len(state.kept)} {plural} on top")
    if not parts:
        return []
    return [f"P{seat + 1} {' and '.join(parts)} (Sentry)"]


def _sentry_private_summary(state: SentryLogState) -> list[str]:
    parts: list[str] = [f"You looked at {_card_list(state.looked)}"]
    actions: list[str] = []
    if state.trashed:
        actions.append(f"trashed {_card_list(state.trashed)}")
    if state.discarded:
        actions.append(f"discarded {_card_list(state.discarded)}")
    if state.kept:
        top = state.kept_on_top if state.kept_on_top is not None else state.kept[-1]
        if len(state.kept) == 1:
            actions.append(f"kept {def_name(top)} on top")
        else:
            others = [def_value for def_value in state.kept if def_value != top]
            if others:
                actions.append(f"kept {def_name(top)} on top over {_card_list(others)}")
            else:
                actions.append(f"kept {def_name(top)} on top")
    if actions:
        parts.append("; " + "; ".join(actions))
    return ["".join(parts)]


def sentry_resolution_logs(
    states: dict[int, SentryLogState],
    seat: int,
    action: int,
    decision: dict[str, Any],
    decision_context: dict[str, Any] | None,
    after_decision: dict[str, Any] | None,
) -> tuple[list[str], dict[int, list[str]]]:
    if int(decision.get("source", -1)) != int(dz.DEF_SENTRY) or action_category(action) != "option":
        return [], {}

    state = states.setdefault(seat, SentryLogState())
    subjects = _context_subjects(decision_context)
    if subjects and not state.looked:
        state.looked = list(subjects)

    option = action_def(action, dz.A_OPTION_BASE)
    kind = decision_kind_name(decision)
    if kind == "ChooseOption":
        index = _context_subject_index(decision_context)
        if index is not None and 0 <= index < len(subjects):
            subject = subjects[index]
            if option == 0:
                state.trashed.append(subject)
            elif option == 1:
                state.discarded.append(subject)
            elif option == 2:
                state.kept.append(subject)
                state.kept_on_top = subject
    elif kind == "ChooseOrder":
        if subjects and not state.kept:
            state.kept = list(subjects)
        if 0 <= option < len(subjects):
            state.kept_on_top = subjects[option]

    if not _sentry_completion_pending(decision, decision_context, after_decision):
        return [], {}

    public = _sentry_public_summary(seat, state)
    private = {seat: _sentry_private_summary(state)}
    states.pop(seat, None)
    return public, private


def _remove_one(values: list[int], value: int) -> bool:
    try:
        values.remove(value)
    except ValueError:
        return False
    return True


def _bandit_sequence_start(action: int, decision: dict[str, Any]) -> bool:
    category = action_category(action)
    source_name = def_name(int(decision.get("source", -1)))
    if category == "play" and action_def(action, dz.A_PLAY_BASE) == int(dz.DEF_BANDIT):
        return True
    return (
        category == "select"
        and source_name == "Throne Room"
        and action_def(action, dz.A_SELECT_BASE) == int(dz.DEF_BANDIT)
    )


def _bandit_resolution_candidate(action: int, decision: dict[str, Any]) -> bool:
    if _bandit_sequence_start(action, decision):
        return True
    return _source_name(action, decision) == "Bandit"


def _bandit_emit_hit(
    state: BanditLogState,
    attacker: int | None,
    victim: int,
    revealed: list[int],
    trashed: list[int],
) -> list[str]:
    lines: list[str] = []
    if attacker is not None:
        prior_hits = state.hits_by_attacker.get(attacker, 0)
        if prior_hits > 0:
            lines.append(f"P{attacker + 1} plays Bandit (again)")
        state.hits_by_attacker[attacker] = prior_hits + 1
    trashed_text = _card_list(trashed) if trashed else "nothing"
    lines.append(f"P{victim + 1} reveals {_card_list(revealed)}; trashes {trashed_text}")
    return lines


def _split_bandit_hits(discarded: list[int], trashed: list[int]) -> list[tuple[list[int], list[int]]]:
    discard_remaining = list(discarded)
    trash_remaining = list(trashed)
    hits: list[tuple[list[int], list[int]]] = []
    while discard_remaining or trash_remaining:
        hit_trash: list[int] = []
        revealed: list[int] = []
        if trash_remaining:
            card = trash_remaining.pop(0)
            hit_trash.append(card)
            revealed.append(card)
        while len(revealed) < 2 and discard_remaining:
            revealed.append(discard_remaining.pop(0))
        hits.append((revealed, hit_trash))
    return hits


def _fallback_bandit_attacker(
    seat: int,
    category: str,
    decision_context: dict[str, Any] | None,
    victim: int | None,
    player_count: int,
) -> int | None:
    if decision_context and decision_context.get("attacker_player") is not None:
        return int(decision_context["attacker_player"])
    if category == "play":
        return seat
    if player_count == 2 and victim is not None:
        return 1 - victim
    return None


def bandit_resolution_logs(
    state: BanditLogState,
    seat: int,
    action: int,
    decision: dict[str, Any],
    before: PublicSnapshot,
    after: PublicSnapshot,
    decision_context: dict[str, Any] | None,
) -> list[str]:
    started = _bandit_sequence_start(action, decision)
    if started:
        state.hits_by_attacker[seat] = 0

    if not _bandit_resolution_candidate(action, decision):
        return []

    category = action_category(action)
    source_name = _source_name(action, decision)
    player_count = len(before.players)
    trash_remaining = _trash_added(before, after)
    discard_remaining = [
        _bandit_revealed_discards(before_player, after_player)
        for before_player, after_player in zip(before.players, after.players)
    ]
    lines: list[str] = []

    if category == "select" and source_name == "Bandit":
        victim = int((decision_context or {}).get("victim_player", decision.get("player", seat)))
        attacker = _fallback_bandit_attacker(seat, category, decision_context, victim, player_count)
        revealed = _context_subjects(decision_context)
        selected = action_def(action, dz.A_SELECT_BASE)
        if revealed:
            lines.extend(_bandit_emit_hit(state, attacker, victim, revealed, [selected]))
            _remove_one(trash_remaining, selected)
            for card in revealed:
                if card != selected and 0 <= victim < len(discard_remaining):
                    _remove_one(discard_remaining[victim], card)

    victim_indices = list(range(player_count))
    attacker_for_auto = seat if started else _fallback_bandit_attacker(seat, category, decision_context, None, player_count)
    if attacker_for_auto is not None:
        victim_indices = [player for player in victim_indices if player != attacker_for_auto]

    for player in victim_indices:
        discarded = discard_remaining[player]
        if not discarded and not trash_remaining:
            continue
        if trash_remaining and not discarded and player_count != 2:
            before_player = before.players[player]
            after_player = after.players[player]
            if before_player.deck_count == after_player.deck_count:
                continue
        attacker = _fallback_bandit_attacker(seat, category, decision_context, player, player_count)
        for revealed, trashed in _split_bandit_hits(discarded, trash_remaining):
            if not revealed:
                continue
            lines.extend(_bandit_emit_hit(state, attacker, player, revealed, trashed))
            for card in trashed:
                _remove_one(trash_remaining, card)
        discard_remaining[player] = []

    return lines
