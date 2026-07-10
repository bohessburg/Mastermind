from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

import dominion_v2_py as dz

from .config import SelfPlayConfig
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


def make_runner_config(config: SelfPlayConfig, seed: int):
    return dz.SelfPlayConfig(
        n_games=config.n_games,
        sims_per_move=config.sims_per_move,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_frac=config.dirichlet_frac,
        temp_moves=config.temp_moves,
        max_batch=config.max_batch,
        seed=seed,
        kingdom_mode=_kingdom_mode(config.kingdom_mode),
        kingdom=config.fixed_kingdom,
        max_recorded_moves=config.max_recorded_moves,
        max_tree_nodes=config.max_tree_nodes,
    )


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
    leaf_players: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a mixed self-play leaf batch with the model for its seat.

    ``SelfPlayRunner.leaf_players()`` is aligned with ``collect_leaves``.  The
    split/scatter keeps the engine's original batch order intact for
    ``provide_evaluations`` while allowing a historical opponent on one seat.
    """
    players = np.asarray(leaf_players, dtype=np.uint8)
    if players.ndim != 1 or players.shape[0] != obs.shape[0]:
        raise ValueError("leaf player attribution must have one entry per observation")
    if len(seat_models) != 2:
        raise ValueError("two seat models are required")
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
) -> tuple[SelfPlayStats, list[dict]]:
    """Generate an exact number of games while routing every leaf by seat."""
    if target_games < 0:
        raise ValueError("target_games cannot be negative")
    stats = SelfPlayStats()
    if target_games == 0:
        return stats, []
    runner_config = SelfPlayConfig(**config.__dict__)
    runner_config.n_games = max(1, min(int(config.n_games), int(target_games)))
    runner = dz.SelfPlayRunner(make_runner_config(runner_config, seed))
    for model in seat_models:
        model.eval()

    records: list[dict] = []
    start = time.perf_counter()
    while len(records) < target_games:
        plumbing_start = time.perf_counter()
        obs, masks = runner.collect_leaves(runner_config.max_batch)
        players = runner.leaf_players()
        stats.plumbing_time += time.perf_counter() - plumbing_start
        batch = int(obs.shape[0])
        if batch == 0:
            continue
        inference_start = time.perf_counter()
        logits_np, values_np = route_leaf_evaluations(seat_models, obs, masks, players, device)
        stats.inference_time += time.perf_counter() - inference_start
        plumbing_start = time.perf_counter()
        runner.provide_evaluations(values_np, logits_np)
        finished = runner.finished_games()
        stats.plumbing_time += time.perf_counter() - plumbing_start
        stats.leaves += batch
        stats.nn_evals += batch
        remaining = target_games - len(records)
        records.extend(finished[:remaining])
    stats.games, stats.positions = _records_to_replay(records, _DiscardReplay())
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
) -> SelfPlayStats:
    stats, records = play_routed_games(seat_models, config, seed=seed, device=device, target_games=target_games)
    games, positions = _records_to_replay(records, replay)
    stats.games = games
    stats.positions = positions
    return stats
