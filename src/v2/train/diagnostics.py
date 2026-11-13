"""Measurement tooling for the 2026-11 drawing-board diagnostics.

The training gate (``gating.run_gate_match``) turned out to mis-measure in
four ways, all of which this module removes:

1. Cross-routed leaves. A leaf was scored by the model of the player deciding
   AT the leaf, so each side's search used the opponent's network at opponent
   nodes. ``independent_search`` routes a whole tree to the searching seat.
2. Shared tree. With ``tree_reuse`` the two models searched one carried-over
   tree. Independent search requires reuse off.
3. Opening temperature. ``gate_temp_moves = 0`` is ignored under
   ``temp_mode = per_seat_buy``; openings were sampled at T=1. Matches here
   force greedy play.
4. First-N selection. The continuously-reset runner returned the first N
   finished games, over-representing short ones. Matches here run a slot
   manifest: every prescribed game is played exactly once.

Games are also PAIRED: the same game index (kingdom, shuffles) is played in
both seat orientations, and statistics are computed over pairs.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch

import dominion_v2_py as dz

from .config import SelfPlayConfig
from .observation import (
    model_encoder_generation,
    model_observation_version,
    observations_for_model,
    obs_version_for_width,
)
from .selfplay import make_runner_config, route_leaf_evaluations

# Two simulations = the root evaluation plus one visit to the highest-prior
# child, so the move played is exactly the policy head's argmax.
RAW_POLICY_SIMS = 2


def pick_device(name: str | None = None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def zero_value_copy(model: torch.nn.Module) -> torch.nn.Module:
    """A distinct copy whose value output is always zero.

    Search with it is guided by the policy prior alone, so ``model`` versus
    this copy measures what the value head contributes to search.
    """
    clone = copy.deepcopy(model)
    clone.eval()
    clone.register_forward_hook(
        lambda _module, _inputs, out: (out[0], torch.zeros_like(out[1])) + tuple(out[2:])
    )
    return clone


def match_selfplay_config(
    base: SelfPlayConfig,
    *,
    sims: int,
    determinize: str | None = None,
    greedy: bool = True,
    tree_reuse: bool = False,
) -> SelfPlayConfig:
    """Evaluation settings derived from a training self-play config.

    The defaults are the clean measurement: greedy play and a fresh tree per
    decision. ``greedy=False`` keeps the training temperature schedule and
    ``tree_reuse=True`` the carried-over tree, which is what ``run_gate_match``
    actually ran; they exist so the measurement itself can be A/B tested.
    """
    config = copy.deepcopy(base)
    config.sims_per_move = int(sims)
    config.deep_slice_fraction = 0.0
    config.deep_slice_sims = 0
    config.dirichlet_frac = 0.0
    config.temp_moves = 0
    config.kingdom_mode = "random"
    config.opening_templates_enabled = False
    config.tree_reuse = bool(tree_reuse)
    if determinize is not None:
        config.determinize = determinize
    if greedy:
        config.temp_mode = "legacy"
        config.temp_final = 0.0
    return config


def play_manifest_games(
    seat_models: Sequence[torch.nn.Module],
    config: SelfPlayConfig,
    *,
    seed: int,
    device: torch.device,
    game_indices: Sequence[int],
    seat_sims: tuple[int, int] = (0, 0),
    independent_search: bool = True,
    record_states: bool = False,
    on_finished: Callable[[list[dict]], None] | None = None,
    tick: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], int]:
    """Play each listed game index exactly once; returns (records, nn_evals).

    Records come back sorted by game index. The game seed depends only on
    ``seed`` and the index, so the same indices with the seat models swapped
    replay the same kingdoms and shuffles. ``tick(finished, nn_evals)`` is
    called after every inference batch so callers can print a heartbeat.
    """
    if len(seat_models) != 2:
        raise ValueError("two seat models are required")
    indices = [int(index) for index in game_indices]
    if not indices:
        return [], 0
    runner_config = copy.deepcopy(config)
    per_seat = any(int(s) > 0 for s in seat_sims)
    if runner_config.tree_reuse and (independent_search or per_seat):
        raise ValueError("independent search and per-seat budgets need tree_reuse off")
    max_sims = max(int(runner_config.sims_per_move), *(int(s) for s in seat_sims))
    if not runner_config.tree_reuse:
        # Each node owns a 16 KB GameState and is allocated eagerly; a fresh
        # tree needs about one node per simulation, so the training-sized slab
        # (sized for reuse) would only waste memory per concurrent game.
        runner_config.max_tree_nodes = 2 * max_sims + 64
    native = make_runner_config(
        runner_config,
        seed,
        slot_manifest=[{"game_index": index} for index in indices],
    )
    native.seat_sims = (int(seat_sims[0]), int(seat_sims[1]))
    native.independent_search = bool(independent_search)
    native.record_states = bool(record_states)
    runner = dz.SelfPlayRunner(native)
    for model in seat_models:
        model.eval()
    same = seat_models[0] is seat_models[1]
    records: list[dict] = []
    evals = 0
    while len(records) < len(indices):
        obs, masks = runner.collect_leaves(runner_config.max_batch)
        batch = int(obs.shape[0])
        if batch == 0:
            finished = runner.finished_games()
            if not finished:
                raise RuntimeError("manifest runner stalled before finishing every game")
        else:
            route = None if same else runner.leaf_route_seats()
            logits, values = route_leaf_evaluations(
                seat_models, obs, masks, route, device, same_model_fast_path=same
            )
            runner.provide_evaluations(values, logits)
            evals += batch
            finished = runner.finished_games()
        if finished:
            records.extend(finished)
            if on_finished is not None:
                on_finished(finished)
        if tick is not None:
            tick(len(records), evals)
    records.sort(key=lambda record: int(record["game_index"]))
    return records, evals


def _score_for(record: dict, player: int) -> float:
    winner = record.get("winner")
    if winner is None:
        return 0.5
    return 1.0 if int(winner) == player else 0.0


def raw_policy_agreement(
    model: torch.nn.Module,
    records: Sequence[dict],
    seat: int,
    device: torch.device,
) -> tuple[int, int]:
    """(decisions where the played move is the prior argmax, decisions).

    Sanity check for RAW_POLICY_SIMS: a two-simulation search must play the
    policy head's top LEGAL move at every recorded decision of ``seat``.
    """
    agree = total = 0
    for record in records:
        players = np.asarray(record["players"])
        rows = np.flatnonzero(players == seat)
        rows = rows[rows < len(record["sampled_actions"])]
        if rows.size == 0:
            continue
        obs = np.asarray(record["observations"], dtype=np.float32)[rows]
        mask = np.asarray(record["legal_mask"], dtype=bool)[rows]
        source = obs_version_for_width(int(obs.shape[-1]))
        adapted = observations_for_model(
            obs, source, model_observation_version(model, source), model_encoder_generation(model)
        )
        with torch.no_grad():
            logits, _ = model.evaluate(
                torch.as_tensor(adapted, dtype=torch.float32, device=device),
                torch.as_tensor(mask, dtype=torch.bool, device=device),
            )
        top = logits.argmax(dim=-1).cpu().numpy()
        played = np.asarray(record["sampled_actions"])[rows]
        agree += int((top == played).sum())
        total += int(rows.size)
    return agree, total


def greedy_stats(records: Sequence[dict]) -> dict[str, float]:
    """How often the played move was NOT the most-visited root move.

    Zero under greedy play. The training gate was meant to be greedy; this is
    the direct read of how far it was not.
    """
    decisions = off = 0
    for record in records:
        policy = np.asarray(record["policy_targets"], dtype=np.float32)
        played = np.asarray(record["sampled_actions"]).astype(np.int64)
        # max_recorded_moves caps the recorded targets but not the move list.
        count = min(policy.shape[0], played.shape[0])
        if count == 0:
            continue
        policy, played = policy[:count], played[:count]
        rows = np.arange(count)
        # A move tied for most visits still counts as greedy.
        off += int((policy[rows, played] < policy.max(axis=-1) - 1.0e-6).sum())
        decisions += count
    return {
        "decisions": decisions,
        "non_greedy": off,
        "non_greedy_pct": (100.0 * off / decisions) if decisions else 0.0,
    }


@dataclass
class PairedMatch:
    """A versus B over paired games; scores are A's (1 win, 0.5 tie, 0 loss)."""

    a_as_seat0: list[float] = field(default_factory=list)
    a_as_seat1: list[float] = field(default_factory=list)
    turns: list[int] = field(default_factory=list)
    truncated: int = 0
    nn_evals: int = 0
    wall_time: float = 0.0
    non_greedy: int = 0
    decisions: int = 0
    raw_agree: dict[str, list[int]] = field(default_factory=dict)

    @property
    def pairs(self) -> int:
        return min(len(self.a_as_seat0), len(self.a_as_seat1))

    def summary(self) -> dict[str, Any]:
        n = self.pairs
        s0 = np.asarray(self.a_as_seat0[:n], dtype=np.float64)
        s1 = np.asarray(self.a_as_seat1[:n], dtype=np.float64)
        both = np.concatenate([s0, s1])
        games = int(both.size)
        if games == 0:
            return {"pairs": 0, "games": 0}
        pair_means = (s0 + s1) / 2.0
        mean = float(pair_means.mean())
        # Pairs are the independent unit: both games of a pair share a board.
        paired_se = float(pair_means.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        naive_se = float(both.std(ddof=1) / math.sqrt(games)) if games > 1 else float("nan")
        wins = int((both == 1.0).sum())
        losses = int((both == 0.0).sum())
        ties = games - wins - losses
        clipped = min(max(mean, 1.0e-6), 1.0 - 1.0e-6)
        out: dict[str, Any] = {
            "pairs": n,
            "games": games,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "a_score": mean,
            "a_win_pct_decisive": (wins / (wins + losses)) if (wins + losses) else 0.5,
            "paired_se": paired_se,
            "naive_se": naive_se,
            "ci95": [mean - 1.96 * paired_se, mean + 1.96 * paired_se],
            "elo": 400.0 * math.log10(clipped / (1.0 - clipped)),
            "a_score_as_seat0": float(s0.mean()),
            "a_score_as_seat1": float(s1.mean()),
            "seat0_score": float((s0.mean() + (1.0 - s1.mean())) / 2.0),
            "mean_turns": float(np.mean(self.turns)) if self.turns else 0.0,
            "truncated": int(self.truncated),
            "non_greedy_pct": (100.0 * self.non_greedy / self.decisions) if self.decisions else 0.0,
            "nn_evals": int(self.nn_evals),
            "wall_s": round(self.wall_time, 1),
        }
        for side, (agree, total) in self.raw_agree.items():
            out[f"raw_policy_agreement_{side}"] = (agree / total) if total else None
        return out


def run_paired_match(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    config: SelfPlayConfig,
    *,
    pairs: int,
    seed: int,
    device: torch.device,
    sims_a: int = 0,
    sims_b: int = 0,
    independent_search: bool = True,
    chunk: int = 64,
    progress: Callable[[PairedMatch], None] | None = None,
    tick: Callable[[int, int], None] | None = None,
) -> PairedMatch:
    """Play ``pairs`` boards twice each (A as seat 0, then A as seat 1).

    ``sims_a`` / ``sims_b`` of zero inherit ``config.sims_per_move``. Work is
    done in chunks of ``chunk`` concurrent games so memory stays bounded and
    a long match reports progress: ``progress(match)`` after each chunk and
    ``tick(games_finished, nn_evals)`` after every inference batch.
    """
    if pairs <= 0:
        raise ValueError("pairs must be positive")
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    match = PairedMatch()
    started = time.perf_counter()
    for first in range(0, int(pairs), int(chunk)):
        indices = list(range(first, min(first + int(chunk), int(pairs))))
        for a_seat in (0, 1):
            seat_models = (model_a, model_b) if a_seat == 0 else (model_b, model_a)
            seat_sims = (int(sims_a), int(sims_b)) if a_seat == 0 else (int(sims_b), int(sims_a))
            games_before, evals_before = len(match.turns), match.nn_evals
            records, evals = play_manifest_games(
                seat_models,
                config,
                seed=seed,
                device=device,
                game_indices=indices,
                seat_sims=seat_sims,
                independent_search=independent_search,
                tick=None if tick is None else (
                    lambda done, used, g=games_before, e=evals_before: tick(g + done, e + used)
                ),
            )
            scores = match.a_as_seat0 if a_seat == 0 else match.a_as_seat1
            for record in records:
                scores.append(_score_for(record, a_seat))
                match.turns.append(int(record["turn_counter"]))
                match.truncated += int(bool(record["truncated"]))
            greedy = greedy_stats(records)
            match.non_greedy += int(greedy["non_greedy"])
            match.decisions += int(greedy["decisions"])
            match.nn_evals += evals
            for side, model, sims, seat in (("a", model_a, sims_a, a_seat), ("b", model_b, sims_b, 1 - a_seat)):
                if int(sims) == RAW_POLICY_SIMS:
                    agree, total = raw_policy_agreement(model, records, seat, device)
                    tally = match.raw_agree.setdefault(side, [0, 0])
                    tally[0] += agree
                    tally[1] += total
            match.wall_time = time.perf_counter() - started
        if progress is not None:
            progress(match)
    return match
