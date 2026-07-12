from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

import dominion_v2_py as dz

from .config import SelfPlayConfig, validate_deep_slice_config
from .replay import ReplayBuffer


@dataclass
class SelfPlayStats:
    games: int = 0
    positions: int = 0
    leaves: int = 0
    nn_evals: int = 0
    wall_time: float = 0.0
    inference_time: float = 0.0
    plumbing_time: float = 0.0
    # Only routed self-play increments these. They distinguish normal
    # best-vs-best routed segments from genuine two-model league segments.
    routed_fast_path_batches: int = 0
    routed_split_batches: int = 0
    scripted_games: int = 0
    scripted_wins: int = 0
    # Deep-search games retain ordinary replay handling, but are surfaced in
    # generation metrics so their data contribution can be monitored.
    deep_games: int = 0
    deep_positions: int = 0
    # Maps scripted opponent kind to (games, neural-network wins).
    scripted_by_kind: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def games_per_hour(self) -> float:
        return 3600.0 * self.games / self.wall_time if self.wall_time > 0 else 0.0

    @property
    def leaves_per_sec(self) -> float:
        return self.leaves / self.wall_time if self.wall_time > 0 else 0.0

    @property
    def nn_evals_per_sec(self) -> float:
        return self.nn_evals / self.inference_time if self.inference_time > 0 else 0.0

    @property
    def inference_pct(self) -> float:
        return 100.0 * self.inference_time / self.wall_time if self.wall_time > 0 else 0.0

    @property
    def plumbing_pct(self) -> float:
        return 100.0 * self.plumbing_time / self.wall_time if self.wall_time > 0 else 0.0


def _kingdom_mode(mode: str):
    normalized = mode.lower()
    if normalized == "fixed":
        return dz.SelfPlayKingdomMode.Fixed
    if normalized == "random":
        return dz.SelfPlayKingdomMode.Random
    raise ValueError(f"unknown kingdom mode: {mode}")


def _scripted_bot_kind(kind: str | None):
    if kind is None:
        return dz.SelfPlayScriptedBotKind.None_
    normalized = kind.lower()
    if normalized == "bigmoney":
        return dz.SelfPlayScriptedBotKind.BigMoney
    if normalized == "engine":
        return dz.SelfPlayScriptedBotKind.Engine
    if normalized == "random":
        return dz.SelfPlayScriptedBotKind.Random
    if normalized == "scaffold":
        return dz.SelfPlayScriptedBotKind.Scaffold
    raise ValueError(f"unknown scripted opponent: {kind}")


def _value_target(value_target: str):
    normalized = value_target.lower()
    if normalized == "outcome":
        return dz.SelfPlayValueTarget.Outcome
    if normalized == "margin":
        return dz.SelfPlayValueTarget.Margin
    raise ValueError(f"unknown value target: {value_target}")


def make_runner_config(
    config: SelfPlayConfig,
    seed: int,
    *,
    scripted_kind: str | None = None,
    scripted_nn_player: int = 0,
    kingdom_pool: Sequence[int] | None = None,
    sims_override: int = 0,
):
    validate_deep_slice_config(config)
    if not math.isfinite(config.margin_scale) or config.margin_scale <= 0.0:
        raise ValueError("margin_scale must be finite and positive")
    if not isinstance(sims_override, int) or isinstance(sims_override, bool) or sims_override < 0:
        raise ValueError("sims_override must be a non-negative integer")
    runner_sims = int(sims_override) if sims_override else int(config.sims_per_move)
    runner_max_tree_nodes = int(config.max_tree_nodes)
    if sims_override:
        # C13 used 4,096 nodes for 512 sims (an 8x margin), but multiplying
        # that full margin for every rare 8-16x deep run is needlessly large.
        # Two nodes per simulation safely grows capacity with the budget while
        # retaining the configured cap whenever it is already larger.
        runner_max_tree_nodes = max(runner_max_tree_nodes, runner_sims * 2)
    runner_config = dz.SelfPlayConfig(
        n_games=config.n_games,
        sims_per_move=runner_sims,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_frac=config.dirichlet_frac,
        temp_moves=config.temp_moves,
        max_batch=config.max_batch,
        seed=seed,
        obs_version=int(config.obs_version),
        kingdom_mode=_kingdom_mode(config.kingdom_mode),
        kingdom=config.fixed_kingdom,
        kingdom_pool=list(kingdom_pool) if kingdom_pool is not None else None,
        max_recorded_moves=config.max_recorded_moves,
        max_tree_nodes=runner_max_tree_nodes,
        scaffold_sims=config.scaffold_sims,
        scaffold_sims_opening=config.scaffold_sims_opening,
        scaffold_determinizations=config.scaffold_determinizations,
        scripted_threads=config.scripted_threads,
        scripted_bot=_scripted_bot_kind(scripted_kind),
        scripted_nn_player=int(scripted_nn_player),
        auto_play_treasures=config.auto_play_treasures,
        prune_treasure_plays=config.prune_treasure_plays,
        tree_reuse=config.tree_reuse,
        expand_top_k=config.expand_top_k,
        value_target=_value_target(config.value_target),
        margin_scale=config.margin_scale,
    )
    return runner_config


def _records_to_replay(records: list[dict], replay: ReplayBuffer) -> tuple[int, int]:
    games = 0
    positions = 0
    for record in records:
        obs = np.asarray(record["observations"], dtype=np.float32)
        policy = np.asarray(record["policy_targets"], dtype=np.float32)
        value = np.asarray(record["values"], dtype=np.float32)
        if obs.shape[0] == 0:
            continue
        legal_mask = policy > 0.0
        replay.add(obs, policy, value, legal_mask)
        games += 1
        positions += obs.shape[0]
    return games, positions


def _record_scripted_outcomes(
    stats: SelfPlayStats,
    records: list[dict],
    scripted_kind: str | None,
) -> None:
    if scripted_kind is None:
        return
    games = len(records)
    wins = sum(
        1
        for record in records
        if record.get("winner") is not None
        and record.get("scripted_nn_player") is not None
        and int(record["winner"]) == int(record["scripted_nn_player"])
    )
    stats.scripted_games += games
    stats.scripted_wins += wins
    previous_games, previous_wins = stats.scripted_by_kind.get(scripted_kind, (0, 0))
    stats.scripted_by_kind[scripted_kind] = (previous_games + games, previous_wins + wins)


def run_self_play_generation(
    model: torch.nn.Module,
    replay: ReplayBuffer,
    config: SelfPlayConfig,
    seed: int,
    device: torch.device,
) -> SelfPlayStats:
    runner = dz.SelfPlayRunner(make_runner_config(config, seed))
    model.eval()
    stats = SelfPlayStats()
    start = time.perf_counter()

    with torch.no_grad():
        while stats.games < config.games_per_generation:
            plumbing_start = time.perf_counter()
            obs, masks = runner.collect_leaves(config.max_batch)
            stats.plumbing_time += time.perf_counter() - plumbing_start
            batch = int(obs.shape[0])
            if batch == 0:
                continue

            inference_start = time.perf_counter()
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=device)
            logits, values = model.evaluate(obs_tensor, mask_tensor)
            logits_np = logits.detach().cpu().numpy().astype(np.float32, copy=False)
            values_np = values.detach().cpu().numpy().astype(np.float32, copy=False)
            stats.inference_time += time.perf_counter() - inference_start

            plumbing_start = time.perf_counter()
            runner.provide_evaluations(values_np, logits_np)
            games, positions = _records_to_replay(runner.finished_games(), replay)
            stats.games += games
            stats.positions += positions
            stats.plumbing_time += time.perf_counter() - plumbing_start
            stats.leaves += batch
            stats.nn_evals += batch

    stats.wall_time = time.perf_counter() - start
    return stats


def route_leaf_evaluations(
    seat_models: Sequence[torch.nn.Module],
    obs: np.ndarray,
    masks: np.ndarray,
    leaf_players: np.ndarray | None,
    device: torch.device,
    *,
    same_model_fast_path: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a mixed self-play leaf batch with the model for its seat.

    ``SelfPlayRunner.leaf_players()`` is aligned with ``collect_leaves``.  The
    split/scatter keeps the engine's original batch order intact for
    ``provide_evaluations`` while allowing a historical opponent on one seat.
    When both seats reference the same model table entry, callers select the
    full-batch fast path and deliberately do not fetch player attribution.
    """
    if len(seat_models) != 2:
        raise ValueError("two seat models are required")
    models_are_identical = seat_models[0] is seat_models[1]
    if same_model_fast_path is True and not models_are_identical:
        raise ValueError("same-model fast path requires both seat models to be the same object")
    use_fast_path = models_are_identical and same_model_fast_path is not False
    if use_fast_path:
        model = seat_models[0]
        model.eval()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=device)
            logits, values = model.evaluate(obs_tensor, mask_tensor)
        return (
            logits.detach().cpu().numpy().astype(np.float32, copy=False),
            values.detach().cpu().numpy().astype(np.float32, copy=False),
        )

    if leaf_players is None:
        raise ValueError("mixed-model routing requires leaf player attribution")
    players = np.asarray(leaf_players, dtype=np.uint8)
    if players.ndim != 1 or players.shape[0] != obs.shape[0]:
        raise ValueError("leaf player attribution must have one entry per observation")
    if models_are_identical:
        # This explicit compatibility mode is used only by the fixed-seed
        # equivalence test.  A GEMM over a full batch versus two differently
        # shaped seat sub-batches can differ by one ULP, despite the model
        # being mathematically batch-independent.  Preserve the routing
        # machinery (attribution and scatter) while retaining the canonical
        # full-batch numerical result that production's fast path emits.
        model = seat_models[0]
        model.eval()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=device)
            logits, values = model.evaluate(obs_tensor, mask_tensor)
        full_logits = logits.detach().cpu().numpy().astype(np.float32, copy=False)
        full_values = values.detach().cpu().numpy().astype(np.float32, copy=False)
        logits_out = np.empty_like(full_logits)
        values_out = np.empty_like(full_values)
        for player in np.unique(players):
            player_index = int(player)
            if player_index not in (0, 1):
                raise ValueError("SelfPlayRunner emitted an invalid player id")
            indices = np.flatnonzero(players == player)
            logits_out[indices] = full_logits[indices]
            values_out[indices] = full_values[indices]
        return logits_out, values_out

    logits_out = np.empty((obs.shape[0], dz.ACTION_SPACE_SIZE), dtype=np.float32)
    values_out = np.empty((obs.shape[0],), dtype=np.float32)
    with torch.no_grad():
        for player in np.unique(players):
            player_index = int(player)
            if player_index not in (0, 1):
                raise ValueError("SelfPlayRunner emitted an invalid player id")
            indices = np.flatnonzero(players == player)
            model = seat_models[player_index]
            model.eval()
            obs_tensor = torch.as_tensor(obs[indices], dtype=torch.float32, device=device)
            mask_tensor = torch.as_tensor(masks[indices], dtype=torch.bool, device=device)
            logits, values = model.evaluate(obs_tensor, mask_tensor)
            logits_out[indices] = logits.detach().cpu().numpy().astype(np.float32, copy=False)
            values_out[indices] = values.detach().cpu().numpy().astype(np.float32, copy=False)
    return logits_out, values_out


def play_routed_games(
    seat_models: Sequence[torch.nn.Module],
    config: SelfPlayConfig,
    *,
    seed: int,
    device: torch.device,
    target_games: int,
    same_model_fast_path: bool | None = None,
    scripted_kind: str | None = None,
    scripted_nn_player: int = 0,
    kingdom_pool: Sequence[int] | None = None,
    kingdom_mode: str | None = None,
    sims_override: int = 0,
) -> tuple[SelfPlayStats, list[dict]]:
    """Generate an exact number of games while routing every leaf by seat."""
    if target_games < 0:
        raise ValueError("target_games cannot be negative")
    stats = SelfPlayStats()
    if target_games == 0:
        return stats, []
    runner_config = SelfPlayConfig(**config.__dict__)
    runner_config.n_games = max(1, min(int(config.n_games), int(target_games)))
    if kingdom_mode is not None:
        runner_config.kingdom_mode = kingdom_mode
    runner = dz.SelfPlayRunner(
        make_runner_config(
            runner_config,
            seed,
            scripted_kind=scripted_kind,
            scripted_nn_player=scripted_nn_player,
            kingdom_pool=kingdom_pool,
            sims_override=sims_override,
        )
    )
    for model in seat_models:
        model.eval()
    models_are_identical = seat_models[0] is seat_models[1]
    if same_model_fast_path is True and not models_are_identical:
        raise ValueError("same-model fast path requires both seat models to be the same object")
    use_fast_path = models_are_identical and same_model_fast_path is not False

    records: list[dict] = []
    start = time.perf_counter()
    while len(records) < target_games:
        plumbing_start = time.perf_counter()
        obs, masks = runner.collect_leaves(runner_config.max_batch)
        players = None if use_fast_path else runner.leaf_players()
        stats.plumbing_time += time.perf_counter() - plumbing_start
        batch = int(obs.shape[0])
        if batch == 0:
            continue
        inference_start = time.perf_counter()
        logits_np, values_np = route_leaf_evaluations(
            seat_models,
            obs,
            masks,
            players,
            device,
            same_model_fast_path=use_fast_path,
        )
        stats.inference_time += time.perf_counter() - inference_start
        if use_fast_path:
            stats.routed_fast_path_batches += 1
        else:
            stats.routed_split_batches += 1
        plumbing_start = time.perf_counter()
        runner.provide_evaluations(values_np, logits_np)
        finished = runner.finished_games()
        stats.plumbing_time += time.perf_counter() - plumbing_start
        stats.leaves += batch
        stats.nn_evals += batch
        remaining = target_games - len(records)
        records.extend(finished[:remaining])
    stats.games, stats.positions = _records_to_replay(records, _DiscardReplay())
    _record_scripted_outcomes(stats, records, scripted_kind)
    stats.wall_time = time.perf_counter() - start
    return stats, records


class _DiscardReplay:
    """Counts records through the existing helper without retaining them twice."""

    def add(self, *_args: object) -> None:
        return None


def run_routed_self_play_generation(
    seat_models: Sequence[torch.nn.Module],
    replay: ReplayBuffer,
    config: SelfPlayConfig,
    *,
    seed: int,
    device: torch.device,
    target_games: int,
    same_model_fast_path: bool | None = None,
    scripted_kind: str | None = None,
    scripted_nn_player: int = 0,
    kingdom_pool: Sequence[int] | None = None,
    kingdom_mode: str | None = None,
    sims_override: int = 0,
) -> SelfPlayStats:
    stats, records = play_routed_games(
        seat_models,
        config,
        seed=seed,
        device=device,
        target_games=target_games,
        same_model_fast_path=same_model_fast_path,
        scripted_kind=scripted_kind,
        scripted_nn_player=scripted_nn_player,
        kingdom_pool=kingdom_pool,
        kingdom_mode=kingdom_mode,
        sims_override=sims_override,
    )
    games, positions = _records_to_replay(records, replay)
    stats.games = games
    stats.positions = positions
    return stats
