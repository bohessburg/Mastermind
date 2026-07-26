"""Multiprocess self-play collection for the v2 trainer.

Parallel collection deliberately does not promise a deterministic insertion
order in the parent replay buffer: worker result messages arrive as soon as
they are ready.  Each worker itself is seed-pinned to ``config.seed + index``;
legacy work keeps a persistent runner, while segmented generations build one
heterogeneous slot-manifest runner for their full assigned quota.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
import queue
import random
import time
import traceback
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
import torch

import dominion_v2_py as dz

from .config import TrainConfig
from .gating import SelfPlaySegment
from .inference_server import (
    InferenceServer,
    InferenceServerEndpoints,
    WorkerSharedMemoryViews,
    deserialize_cpu_state_dict,
    serialize_cpu_state_dict,
)
from .model import build_model, model_config_dict
from .observation import (
    model_observation_version,
    observations_for_model,
    obs_size_for_config,
    obs_size_for_version,
    obs_version_for_width,
)
from .selfplay import (
    SelfPlayStats,
    make_runner_config,
    record_opening_template_telemetry,
    route_leaf_evaluations,
)


PackedGameRecords = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
ModelStatePayload = bytes | tuple[bytes, dict[str, Any]]


@dataclass(frozen=True)
class ParallelSelfPlayResult:
    stats: SelfPlayStats
    collection_wall_time: float
    league_games: int = 0
    seat_model_evals: dict[tuple[int, int], int] | None = None

    @property
    def aggregate_games_per_hour(self) -> float:
        if self.collection_wall_time <= 0.0:
            return 0.0
        return 3600.0 * self.stats.games / self.collection_wall_time


@dataclass(frozen=True)
class SelfPlaySlotManifest:
    """One independently live runner slot plus Python-only attribution tags."""

    game_index: int
    seat0_model_id: int
    seat1_model_id: int
    scripted_kind: str | None = None
    nn_player: int = 0
    kingdom_pool: list[int] | None = None
    kingdom_mode: str | None = None
    sims_override: int = 0
    league_opponent: str | None = None


def index_selfplay_segments(segments: list[SelfPlaySegment]) -> list[SelfPlaySegment]:
    """Assign stable global game ranges after the final plan is composed.

    Segment allocation may deliberately reorder reserved deep or league work.
    Each copied fragment carries this range forward, which keeps per-game
    seeds stable even when a different worker receives the segment.
    """
    indexed: list[SelfPlaySegment] = []
    next_index = 0
    for segment in segments:
        if segment.n_games <= 0:
            raise ValueError("self-play segments must have positive game counts")
        start = next_index if segment.game_index is None else int(segment.game_index)
        if start < 0:
            raise ValueError("self-play segment game_index must be non-negative")
        indexed.append(replace(segment, game_index=start))
        next_index = max(next_index, start + int(segment.n_games))
    return indexed


def build_slot_manifest(segments: list[SelfPlaySegment]) -> list[SelfPlaySlotManifest]:
    """Flatten exact segment work into independently configured game slots."""
    slots: list[SelfPlaySlotManifest] = []
    for segment in index_selfplay_segments(segments):
        assert segment.game_index is not None
        for offset in range(segment.n_games):
            slots.append(
                SelfPlaySlotManifest(
                    game_index=int(segment.game_index) + offset,
                    seat0_model_id=int(segment.seat0_model_id),
                    seat1_model_id=int(segment.seat1_model_id),
                    scripted_kind=segment.scripted_kind,
                    nn_player=int(segment.nn_player),
                    kingdom_pool=(list(segment.kingdom_pool) if segment.kingdom_pool is not None else None),
                    kingdom_mode=segment.kingdom_mode,
                    sims_override=int(segment.sims_override),
                    league_opponent=segment.league_opponent,
                )
            )
    if len({slot.game_index for slot in slots}) != len(slots):
        raise ValueError("self-play slot manifest contains duplicate game indices")
    return slots


def _unpack_model_state_payload(
    payload: ModelStatePayload,
    fallback_model_config: object,
) -> tuple[bytes, dict[str, Any]]:
    """Accept legacy state-only payloads and architecture-aware new payloads."""
    if isinstance(payload, bytes):
        return payload, model_config_dict(fallback_model_config)
    if (
        isinstance(payload, tuple)
        and len(payload) == 2
        and isinstance(payload[0], bytes)
        and isinstance(payload[1], dict)
    ):
        return payload[0], dict(payload[1])
    raise TypeError("model state payload must be bytes or (bytes, model_config)")


def game_quotas(total_games: int, workers: int) -> list[int]:
    """Split a generation exactly, giving the first workers one extra game."""
    if total_games <= 0:
        raise ValueError("games_per_generation must be positive in parallel mode")
    if workers <= 0:
        raise ValueError("parallel_workers must be positive")
    if workers > total_games:
        raise ValueError("parallel_workers cannot exceed games_per_generation")
    base, remainder = divmod(total_games, workers)
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def split_segments_by_quotas(
    segments: list[SelfPlaySegment],
    quotas: list[int],
) -> list[list[SelfPlaySegment]]:
    """Split exact game segments across workers while honoring exact quotas.

    Deep-search pieces and a small number of league-model pieces are each
    independently runnable.  Reserve those pieces on distinct workers before
    filling quota slack with the remaining contiguous segments.  This avoids
    concentrating a high-cost deep slice on the last worker merely because it
    follows the normal segment in planner order.
    """
    segments = index_selfplay_segments(segments)
    if sum(segment.n_games for segment in segments) != sum(quotas):
        raise ValueError("self-play segments must cover exactly one generation")
    if not segments:
        return [[] for _ in quotas]

    def copy_with_games(source: SelfPlaySegment, n_games: int, game_index: int) -> SelfPlaySegment:
        return SelfPlaySegment(
            n_games,
            source.seat0_model_id,
            source.seat1_model_id,
            scripted_kind=source.scripted_kind,
            nn_player=source.nn_player,
            kingdom_pool=source.kingdom_pool,
            kingdom_mode=source.kingdom_mode,
            sims_override=source.sims_override,
            league_opponent=source.league_opponent,
            game_index=game_index,
        )

    per_worker: list[list[SelfPlaySegment]] = [[] for _ in quotas]
    reserved: list[list[SelfPlaySegment]] = [[] for _ in quotas]
    remaining_quotas = list(quotas)
    assigned_indices: set[int] = set()

    def reserve_on_distinct_workers(groups: list[list[int]]) -> bool:
        """Reserve contiguous segment groups on separate workers, largest first."""
        if len(groups) > sum(not worker_reserved for worker_reserved in reserved):
            return False
        tentative_remaining = list(remaining_quotas)
        unassigned_workers = {
            worker_index
            for worker_index in range(len(quotas))
            if not reserved[worker_index]
        }
        assignments: list[tuple[int, list[int]]] = []
        for group in sorted(
            groups,
            key=lambda indices: (-sum(segments[index].n_games for index in indices), min(indices)),
        ):
            group_games = sum(segments[index].n_games for index in group)
            eligible = [
                worker_index
                for worker_index in unassigned_workers
                if tentative_remaining[worker_index] >= group_games
            ]
            if not eligible:
                return False
            worker_index = max(eligible, key=lambda index: (tentative_remaining[index], -index))
            assignments.append((worker_index, group))
            unassigned_workers.remove(worker_index)
            # Keep tentative capacity local until all priority segments fit.
            tentative_remaining[worker_index] -= group_games
        remaining_quotas[:] = tentative_remaining
        for worker_index, group in assignments:
            for segment_index in group:
                reserved[worker_index].append(
                    copy_with_games(
                        segments[segment_index],
                        segments[segment_index].n_games,
                        int(segments[segment_index].game_index),
                    )
                )
                assigned_indices.add(segment_index)
        return True

    # The planner emits no more than one deep piece per worker.  Give those
    # pieces first claim on workers; they are substantially more expensive
    # than the normal-mirror work that fills the rest of each quota.
    deep_indices = [index for index, segment in enumerate(segments) if segment.sims_override > 0]
    if len(quotas) > 1 and deep_indices:
        reserve_on_distinct_workers([[index] for index in deep_indices])

    # Kingdom curriculum can split one opponent's work into multiple adjacent
    # segments. Reserve those same-model pieces as one block so the worker
    # retains fat per-model inference batches; if a block cannot fit, the
    # contiguous fill below splits it across only the required workers.
    league_groups: dict[tuple[int, int], list[int]] = {}
    for index, segment in enumerate(segments):
        if index not in assigned_indices and segment.is_league:
            league_groups.setdefault((segment.seat0_model_id, segment.seat1_model_id), []).append(index)
    if len(quotas) > 1 and league_groups:
        reserve_on_distinct_workers(list(league_groups.values()))

    remaining_segments = [
        segment for index, segment in enumerate(segments) if index not in assigned_indices
    ]
    segment_index = 0
    remaining = remaining_segments[0].n_games if remaining_segments else 0
    next_game_index = int(remaining_segments[0].game_index) if remaining_segments else 0
    for worker_index, needed in enumerate(remaining_quotas):
        while needed > 0:
            if segment_index >= len(remaining_segments):
                raise ValueError("self-play segment allocation ended early")
            source = remaining_segments[segment_index]
            take = min(needed, remaining)
            per_worker[worker_index].append(copy_with_games(source, take, next_game_index))
            needed -= take
            remaining -= take
            next_game_index += take
            if remaining == 0:
                segment_index += 1
                if segment_index < len(remaining_segments):
                    remaining = remaining_segments[segment_index].n_games
                    next_game_index = int(remaining_segments[segment_index].game_index)
    if segment_index != len(remaining_segments):
        raise ValueError("self-play segment allocation left unassigned games")
    for worker_segments, worker_reserved in zip(per_worker, reserved, strict=True):
        worker_segments.extend(worker_reserved)
    return per_worker


def _empty_packed_records(obs_size: int) -> PackedGameRecords:
    return (
        np.empty((0,), dtype=np.int32),
        np.empty((0, int(obs_size)), dtype=np.float32),
        np.empty((0, dz.ACTION_SPACE_SIZE), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )


def _pack_records(records: list[dict[str, Any]], obs_size: int) -> PackedGameRecords:
    """Flatten finished-game dictionaries into four raw NumPy buffers.

    Queue messages carry only these contiguous arrays and a tiny tuple header,
    never the binding's large list-of-dicts game-record representation.
    """
    expected_obs_size = int(obs_size)
    if expected_obs_size <= 0:
        raise ValueError("packed self-play observation width must be positive")
    if not records:
        return _empty_packed_records(expected_obs_size)
    observations = [np.asarray(record["observations"], dtype=np.float32) for record in records]
    policies = [np.asarray(record["policy_targets"], dtype=np.float32) for record in records]
    values = [np.asarray(record["values"], dtype=np.float32) for record in records]
    for record_index, (obs, policy, value) in enumerate(zip(observations, policies, values, strict=True)):
        if obs.ndim != 2 or obs.shape[1] != expected_obs_size:
            actual_width = obs.shape[1] if obs.ndim == 2 else None
            raise ValueError(
                f"packed self-play record {record_index} observation width {actual_width} "
                f"does not match configured width {expected_obs_size}"
            )
        if policy.shape != (obs.shape[0], dz.ACTION_SPACE_SIZE):
            raise ValueError(f"packed self-play record {record_index} policy shape mismatch")
        if value.shape != (obs.shape[0],):
            raise ValueError(f"packed self-play record {record_index} value shape mismatch")
    lengths = np.asarray([obs.shape[0] for obs in observations], dtype=np.int32)
    nonempty = lengths > 0
    if not np.any(nonempty):
        empty = _empty_packed_records(expected_obs_size)
        return lengths, empty[1], empty[2], empty[3]
    return (
        lengths,
        np.ascontiguousarray(np.concatenate(observations, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(policies, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(values, axis=0), dtype=np.float32),
    )


def add_packed_records(replay: Any, packed: PackedGameRecords) -> tuple[int, int]:
    """Insert a worker message into the parent replay buffer."""
    lengths, obs, policy, value = packed
    lengths = np.asarray(lengths, dtype=np.int32)
    obs = np.asarray(obs, dtype=np.float32)
    policy = np.asarray(policy, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    if lengths.ndim != 1 or obs.ndim != 2 or policy.ndim != 2 or value.ndim != 1:
        raise ValueError("packed self-play record buffers have invalid dimensions")
    replay_obs_size = getattr(replay, "obs_size", None)
    if replay_obs_size is not None and obs.shape[1] != int(replay_obs_size):
        raise ValueError(
            f"packed self-play observation width {obs.shape[1]} does not match replay width {int(replay_obs_size)}"
        )
    positions = int(lengths.sum())
    if positions != int(obs.shape[0]) or positions != int(policy.shape[0]) or positions != int(value.shape[0]):
        raise ValueError("packed self-play record lengths do not match their buffers")
    if positions > 0:
        replay.add(obs, policy, value, policy > 0.0)
    return int(lengths.shape[0]), positions


def _worker_device(name: str) -> torch.device:
    requested = name.lower()
    if requested not in {"cpu", "cuda"}:
        raise ValueError("worker_device must be 'cpu' or 'cuda'")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("worker_device=cuda requested but unavailable")
    return torch.device(requested)


def _server_selfplay_enabled(config: TrainConfig) -> bool:
    enabled = getattr(config, "server_selfplay", False)
    if not isinstance(enabled, bool):
        raise ValueError("server_selfplay must be a boolean")
    return enabled or config.worker_device.lower() == "server"


def _seed_worker(seed: int, device: torch.device | None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if device is None:
        # Server-mode workers must not touch CUDA at all; the dedicated server
        # is the sole owner of that context and of Torch model state.
        return
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    else:
        # CPU smoke runs should exercise process parallelism, not oversubscribe
        # every core with two independent tiny Torch inference sessions.
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=True)


def _local_evaluator(model: torch.nn.Module, device: torch.device) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    def evaluate(obs: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=device)
            logits, values = model.evaluate(obs_tensor, mask_tensor)
        return (
            logits.detach().cpu().numpy().astype(np.float32, copy=False),
            values.detach().cpu().numpy().astype(np.float32, copy=False),
        )

    return evaluate


def _server_evaluator(
    endpoints: InferenceServerEndpoints,
    worker_index: int,
) -> tuple[Callable[..., tuple[np.ndarray, np.ndarray]], WorkerSharedMemoryViews | None]:
    response_queue = endpoints.response_queues[worker_index]
    request_id = 0
    shared_views = (
        WorkerSharedMemoryViews(endpoints.shared_memory_specs[worker_index])
        if endpoints.transport == "shm"
        else None
    )

    def evaluate(
        obs: np.ndarray,
        masks: np.ndarray,
        model_id: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal request_id
        request_id += 1
        deadline = time.monotonic() + endpoints.response_timeout_s
        count = int(obs.shape[0])
        if count <= 0 or count > endpoints.request_batch_size:
            raise RuntimeError("worker attempted an invalid inference-server request batch")
        if not isinstance(model_id, (int, np.integer)) or int(model_id) < 0:
            raise RuntimeError("worker attempted an invalid inference-server model id")
        model_id = int(model_id)
        if shared_views is not None:
            slot = request_id % shared_views.spec.slots
            np.copyto(shared_views.request_obs[slot, :count], np.asarray(obs, dtype=np.float32))
            np.copyto(shared_views.request_masks[slot, :count], np.asarray(masks, dtype=np.uint8))
            submitted_ns = time.perf_counter_ns()
            if endpoints.poll == "spin":
                shared_views.request_counts[slot] = count
                shared_views.request_model_ids[slot] = model_id
                shared_views.request_submitted_ns[slot] = submitted_ns
                shared_views.request_sequences[slot] = request_id
                request = ()
            else:
                request = (worker_index, slot, count, request_id, model_id, submitted_ns)
        else:
            request_obs = np.ascontiguousarray(obs, dtype=np.float32)
            request_masks = np.ascontiguousarray(masks, dtype=np.uint8)
            submitted_ns = time.perf_counter_ns()
            request = (
                worker_index,
                request_id,
                model_id,
                submitted_ns,
                request_obs,
                request_masks,
            )
        if shared_views is None or endpoints.poll != "spin":
            while True:
                if not endpoints.alive_event.is_set():
                    raise RuntimeError("inference server is not alive while submitting a leaf batch")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError("timed out submitting a leaf batch to the inference server")
                try:
                    endpoints.request_queue.put(request, timeout=min(0.25, remaining))
                    break
                except queue.Full:
                    continue

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError("timed out waiting for an inference-server response")
            if shared_views is not None and endpoints.poll == "spin":
                if int(shared_views.response_sequences[slot]) == request_id:
                    return (
                        shared_views.response_policies[slot, :count],
                        shared_views.response_values[slot, :count],
                    )
                if not endpoints.alive_event.is_set():
                    raise RuntimeError("inference server died while a worker awaited a response")
                time.sleep(0)
                continue
            try:
                response = response_queue.get(timeout=min(0.25, remaining))
            except queue.Empty:
                if not endpoints.alive_event.is_set():
                    raise RuntimeError("inference server died while a worker awaited a response")
                continue
            if shared_views is not None:
                response_slot, response_count, response_id = response
                if response_slot != slot or response_count != count or response_id != request_id:
                    raise RuntimeError("inference server routed a shared-memory response to the wrong request")
                return (
                    shared_views.response_policies[slot, :count],
                    shared_views.response_values[slot, :count],
                )
            kind, response_id, values, policies = response
            if kind == "error":
                raise RuntimeError(f"inference server rejected request {request_id}: {values}")
            if kind != "response" or response_id != request_id:
                raise RuntimeError("inference server routed a response to the wrong request")
            return (
                np.asarray(policies, dtype=np.float32),
                np.asarray(values, dtype=np.float32),
            )

    return evaluate, shared_views


def _generate_games(
    runner: Any,
    evaluate: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    collect_max_batch: int,
    target_games: int,
    deferred: list[dict[str, Any]],
    result_queue: Any,
    worker_index: int,
    generation: int,
    obs_size: int,
) -> tuple[SelfPlayStats, list[dict[str, Any]]]:
    stats = SelfPlayStats()
    start = time.perf_counter()
    sent = 0

    def emit(records: list[dict[str, Any]]) -> None:
        nonlocal sent
        if not records:
            return
        record_opening_template_telemetry(stats, records)
        packed = _pack_records(records, obs_size)
        games, positions = int(packed[0].shape[0]), int(packed[0].sum())
        # NumPy buffers keep queue serialization to raw array payloads instead
        # of pickling a large Python object graph for every game record.
        result_queue.put(("records", worker_index, generation, packed))
        sent += games
        stats.games += games
        stats.positions += positions

    if deferred:
        ready = deferred[:target_games]
        deferred = deferred[target_games:]
        emit(ready)

    while sent < target_games:
        plumbing_start = time.perf_counter()
        obs, masks = runner.collect_leaves(collect_max_batch)
        stats.plumbing_time += time.perf_counter() - plumbing_start
        batch = int(obs.shape[0])
        if batch == 0:
            continue

        inference_start = time.perf_counter()
        logits_np, values_np = evaluate(obs, masks)
        stats.inference_time += time.perf_counter() - inference_start

        plumbing_start = time.perf_counter()
        runner.provide_evaluations(values_np, logits_np)
        finished = runner.finished_games()
        stats.plumbing_time += time.perf_counter() - plumbing_start
        stats.leaves += batch
        stats.nn_evals += batch
        if not finished:
            continue

        remaining = target_games - sent
        emit(finished[:remaining])
        deferred.extend(finished[remaining:])

    stats.wall_time = time.perf_counter() - start
    return stats, deferred


def _record_scripted_outcomes(
    stats: SelfPlayStats,
    records: list[dict[str, Any]],
    scripted_kind: str | None,
) -> None:
    if scripted_kind is None:
        return
    games = len(records)
    wins = 0
    for record in records:
        winner = record.get("winner")
        nn_player = record.get("scripted_nn_player")
        if winner is not None and nn_player is not None and int(winner) == int(nn_player):
            wins += 1
    stats.scripted_games += games
    stats.scripted_wins += wins
    previous_games, previous_wins = stats.scripted_by_kind.get(scripted_kind, (0, 0))
    stats.scripted_by_kind[scripted_kind] = (previous_games + games, previous_wins + wins)


def _record_league_outcomes(
    stats: SelfPlayStats,
    records: list[dict[str, Any]],
    league_opponent: str | None,
) -> None:
    if league_opponent is None:
        return
    # This matches the single-process route: replay keeps both seats, while
    # league win counters mean the current model's fixed seat-zero result.
    games = len(records)
    wins = sum(1 for record in records if record.get("winner") is not None and int(record["winner"]) == 0)
    previous_games, previous_wins = stats.league_by_opponent.get(league_opponent, (0, 0))
    stats.league_by_opponent[league_opponent] = (previous_games + games, previous_wins + wins)


def _generate_games_exact(
    runner: Any,
    evaluate: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    collect_max_batch: int,
    target_games: int,
    result_queue: Any,
    worker_index: int,
    generation: int,
    scripted_kind: str | None = None,
    *,
    obs_size: int,
) -> SelfPlayStats:
    """Use the persistent runner but never carry records across a segment."""
    stats = SelfPlayStats()
    start = time.perf_counter()
    sent = 0
    while sent < target_games:
        plumbing_start = time.perf_counter()
        obs, masks = runner.collect_leaves(collect_max_batch)
        stats.plumbing_time += time.perf_counter() - plumbing_start
        batch = int(obs.shape[0])
        if batch == 0:
            continue
        inference_start = time.perf_counter()
        logits_np, values_np = evaluate(obs, masks)
        stats.inference_time += time.perf_counter() - inference_start
        plumbing_start = time.perf_counter()
        runner.provide_evaluations(values_np, logits_np)
        finished = runner.finished_games()
        stats.plumbing_time += time.perf_counter() - plumbing_start
        stats.leaves += batch
        stats.nn_evals += batch
        if not finished:
            continue
        records = finished[: target_games - sent]
        record_opening_template_telemetry(stats, records)
        _record_scripted_outcomes(stats, records, scripted_kind)
        packed = _pack_records(records, obs_size)
        games, positions = int(packed[0].shape[0]), int(packed[0].sum())
        if games:
            result_queue.put(("records", worker_index, generation, packed))
            sent += games
            stats.games += games
            stats.positions += positions
    stats.wall_time = time.perf_counter() - start
    return stats


def _generate_routed_games(
    seat_models: tuple[torch.nn.Module, torch.nn.Module],
    selfplay_config: Any,
    seed: int,
    device: torch.device,
    target_games: int,
    result_queue: Any,
    worker_index: int,
    generation: int,
    seat_model_ids: tuple[int, int] | None = None,
    route_audit: dict[tuple[int, int], int] | None = None,
    same_model_fast_path: bool | None = None,
    scripted_kind: str | None = None,
    scripted_nn_player: int = 0,
    kingdom_pool: list[int] | None = None,
    kingdom_mode: str | None = None,
    sims_override: int = 0,
    league_opponent: str | None = None,
) -> SelfPlayStats:
    """Generate a task with fixed models for player zero and player one."""
    if target_games <= 0:
        return SelfPlayStats()
    task_config = copy.deepcopy(selfplay_config)
    task_config.n_games = max(1, min(int(task_config.n_games), int(target_games)))
    if kingdom_mode is not None:
        task_config.kingdom_mode = kingdom_mode
    obs_size = obs_size_for_version(int(task_config.obs_version))
    runner = dz.SelfPlayRunner(
        make_runner_config(
            task_config,
            seed,
            scripted_kind=scripted_kind,
            scripted_nn_player=scripted_nn_player,
            kingdom_pool=kingdom_pool,
            sims_override=sims_override,
        )
    )
    for model in seat_models:
        model.eval()
    same_model = (
        seat_model_ids[0] == seat_model_ids[1]
        if same_model_fast_path is None and seat_model_ids is not None
        else (seat_models[0] is seat_models[1] if same_model_fast_path is None else same_model_fast_path)
    )
    stats = SelfPlayStats()
    sent = 0
    start = time.perf_counter()
    while sent < target_games:
        plumbing_start = time.perf_counter()
        obs, masks = runner.collect_leaves(task_config.max_batch)
        players = None if same_model else runner.leaf_players()
        stats.plumbing_time += time.perf_counter() - plumbing_start
        batch = int(obs.shape[0])
        if batch == 0:
            continue
        if route_audit is not None and seat_model_ids is not None and players is not None:
            for player in np.unique(players):
                player_index = int(player)
                key = (player_index, seat_model_ids[player_index])
                route_audit[key] = route_audit.get(key, 0) + int(np.count_nonzero(players == player))
        inference_start = time.perf_counter()
        logits_np, values_np = route_leaf_evaluations(
            seat_models,
            obs,
            masks,
            players,
            device,
            same_model_fast_path=same_model,
        )
        stats.inference_time += time.perf_counter() - inference_start
        if same_model:
            stats.routed_fast_path_batches += 1
        else:
            stats.routed_split_batches += 1
        plumbing_start = time.perf_counter()
        runner.provide_evaluations(values_np, logits_np)
        finished = runner.finished_games()
        stats.plumbing_time += time.perf_counter() - plumbing_start
        stats.leaves += batch
        stats.nn_evals += batch
        if not finished:
            continue
        records = finished[: target_games - sent]
        record_opening_template_telemetry(stats, records)
        _record_scripted_outcomes(stats, records, scripted_kind)
        _record_league_outcomes(stats, records, league_opponent)
        packed = _pack_records(records, obs_size)
        games, positions = int(packed[0].shape[0]), int(packed[0].sum())
        if games:
            result_queue.put(("records", worker_index, generation, packed))
            sent += games
            stats.games += games
            stats.positions += positions
    stats.wall_time = time.perf_counter() - start
    return stats


def _evaluate_manifest_model_groups(
    model_table: list[torch.nn.Module],
    obs: np.ndarray,
    masks: np.ndarray,
    model_ids: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate each required model once and restore runner leaf order."""
    ids = np.asarray(model_ids, dtype=np.uint32)
    if ids.ndim != 1 or ids.shape[0] != obs.shape[0]:
        raise ValueError("leaf model attribution must have one entry per observation")
    logits_out = np.empty((obs.shape[0], dz.ACTION_SPACE_SIZE), dtype=np.float32)
    values_out = np.empty((obs.shape[0],), dtype=np.float32)
    source_version = obs_version_for_width(int(obs.shape[-1]))
    with torch.no_grad():
        for model_id in np.unique(ids):
            index = int(model_id)
            if not 0 <= index < len(model_table):
                raise RuntimeError("self-play slot references an unknown model id")
            rows = np.flatnonzero(ids == model_id)
            model = model_table[index]
            model.eval()
            adapted_obs = observations_for_model(
                obs[rows],
                source_version,
                model_observation_version(model, source_version),
            )
            obs_tensor = torch.as_tensor(adapted_obs, dtype=torch.float32, device=device)
            mask_tensor = torch.as_tensor(masks[rows], dtype=torch.bool, device=device)
            logits, values = model.evaluate(obs_tensor, mask_tensor)
            logits_out[rows] = logits.detach().cpu().numpy().astype(np.float32, copy=False)
            values_out[rows] = values.detach().cpu().numpy().astype(np.float32, copy=False)
    return logits_out, values_out


def _evaluate_manifest_server_groups(
    evaluate: Callable[..., tuple[np.ndarray, np.ndarray]],
    obs: np.ndarray,
    masks: np.ndarray,
    model_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Submit homogeneous per-model requests and restore native leaf order."""

    ids = np.asarray(model_ids, dtype=np.uint32)
    if ids.ndim != 1 or ids.shape[0] != obs.shape[0]:
        raise ValueError("leaf model attribution must have one entry per observation")
    logits_out = np.empty((obs.shape[0], dz.ACTION_SPACE_SIZE), dtype=np.float32)
    values_out = np.empty((obs.shape[0],), dtype=np.float32)
    for model_id in np.unique(ids):
        rows = np.flatnonzero(ids == model_id)
        logits, values = evaluate(obs[rows], masks[rows], int(model_id))
        logits_array = np.asarray(logits, dtype=np.float32)
        values_array = np.asarray(values, dtype=np.float32)
        if logits_array.shape != (rows.shape[0], dz.ACTION_SPACE_SIZE):
            raise RuntimeError("inference server returned a policy batch with the wrong shape")
        if values_array.shape != (rows.shape[0],):
            raise RuntimeError("inference server returned a value batch with the wrong shape")
        logits_out[rows] = logits_array
        values_out[rows] = values_array
    return logits_out, values_out


def _record_manifest_outcomes(
    stats: SelfPlayStats,
    records: list[dict[str, Any]],
    slots_by_game_index: dict[int, SelfPlaySlotManifest],
) -> None:
    record_opening_template_telemetry(stats, records)
    scripted: dict[str, list[dict[str, Any]]] = {}
    league: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        game_index = int(record.get("game_index", -1))
        try:
            slot = slots_by_game_index[game_index]
        except KeyError as exc:
            raise RuntimeError(f"self-play runner finished an unknown slot game_index {game_index}") from exc
        if slot.scripted_kind is not None:
            scripted.setdefault(slot.scripted_kind, []).append(record)
        elif slot.seat0_model_id != slot.seat1_model_id and slot.league_opponent is not None:
            league.setdefault(slot.league_opponent, []).append(record)
        if slot.sims_override:
            stats.deep_games += 1
            stats.deep_positions += int(np.asarray(record["observations"]).shape[0])
    for kind, grouped in scripted.items():
        _record_scripted_outcomes(stats, grouped, kind)
    for opponent, grouped in league.items():
        _record_league_outcomes(stats, grouped, opponent)


def _generate_manifest_games(
    runner: Any,
    slots: list[SelfPlaySlotManifest],
    evaluate: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None,
    model_table: list[torch.nn.Module] | None,
    device: torch.device | None,
    collect_max_batch: int,
    result_queue: Any,
    worker_index: int,
    generation: int,
    *,
    obs_size: int,
    route_audit: dict[tuple[int, int], int],
) -> SelfPlayStats:
    """Run all assigned segment games concurrently through one slot manifest."""
    if not slots:
        return SelfPlayStats()
    if len({slot.game_index for slot in slots}) != len(slots):
        raise RuntimeError("self-play slot manifest contains duplicate game indices")
    if (model_table is None) == (device is not None):
        # Local evaluation needs both a table and device; server evaluation
        # needs neither because its closure owns the transport.
        raise RuntimeError("invalid self-play manifest evaluator configuration")
    if model_table is None and evaluate is None:
        raise RuntimeError("self-play manifest has no evaluator")

    slots_by_game_index = {slot.game_index: slot for slot in slots}
    seen_game_indices: set[int] = set()
    stats = SelfPlayStats()
    start = time.perf_counter()

    def emit(records: list[dict[str, Any]]) -> None:
        if not records:
            return
        duplicate = [int(record["game_index"]) for record in records if int(record["game_index"]) in seen_game_indices]
        if duplicate:
            raise RuntimeError(f"self-play manifest emitted duplicate game indices: {duplicate}")
        _record_manifest_outcomes(stats, records, slots_by_game_index)
        seen_game_indices.update(int(record["game_index"]) for record in records)
        packed = _pack_records(records, obs_size)
        games, positions = int(packed[0].shape[0]), int(packed[0].sum())
        if games:
            result_queue.put(("records", worker_index, generation, packed))
            stats.games += games
            stats.positions += positions

    while len(seen_game_indices) < len(slots):
        plumbing_start = time.perf_counter()
        obs, masks = runner.collect_leaves(collect_max_batch)
        players = runner.leaf_players()
        model_ids = runner.leaf_model_ids()
        game_indices = runner.leaf_game_indices()
        stats.plumbing_time += time.perf_counter() - plumbing_start
        batch = int(obs.shape[0])
        if batch == 0:
            # A fully scripted slot can finish while collect has no NN leaves;
            # consume those records rather than spinning forever.  Scaffold
            # workers may temporarily own every remaining slot, so yield too.
            emit(runner.finished_games())
            if len(seen_game_indices) < len(slots):
                time.sleep(0)
            continue

        players_array = np.asarray(players, dtype=np.uint8)
        model_ids_array = np.asarray(model_ids, dtype=np.uint32)
        game_indices_array = np.asarray(game_indices, dtype=np.uint64)
        if (
            players_array.shape != model_ids_array.shape
            or players_array.shape != game_indices_array.shape
            or players_array.shape[0] != batch
        ):
            raise RuntimeError("self-play runner returned misaligned leaf attribution")
        for player, model_id in zip(players_array, model_ids_array, strict=True):
            key = (int(player), int(model_id))
            route_audit[key] = route_audit.get(key, 0) + 1

        inference_start = time.perf_counter()
        if model_table is None:
            assert evaluate is not None
            logits_np, values_np = _evaluate_manifest_server_groups(
                evaluate,
                obs,
                masks,
                model_ids_array,
            )
        else:
            assert device is not None
            logits_np, values_np = _evaluate_manifest_model_groups(
                model_table,
                obs,
                masks,
                model_ids_array,
                device,
            )
        stats.inference_time += time.perf_counter() - inference_start
        # Preserve the established metric definition: a leaf batch from a
        # two-model league slot is "split" even when this particular pass has
        # only one seat's leaves. The new execution still evaluates it once by
        # model id, but the counters remain comparable with segment runners.
        if any(
            slots_by_game_index[int(game_index)].seat0_model_id
            != slots_by_game_index[int(game_index)].seat1_model_id
            for game_index in game_indices_array
        ):
            stats.routed_split_batches += 1
        else:
            stats.routed_fast_path_batches += 1

        plumbing_start = time.perf_counter()
        runner.provide_evaluations(values_np, logits_np)
        emit(runner.finished_games())
        stats.plumbing_time += time.perf_counter() - plumbing_start
        stats.leaves += batch
        stats.nn_evals += batch

    if stats.games != len(slots):
        raise RuntimeError(
            f"self-play manifest completed {stats.games} records for {len(slots)} slots"
        )
    stats.wall_time = time.perf_counter() - start
    return stats


def _accumulate_stats(total: SelfPlayStats, update: SelfPlayStats) -> None:
    total.games += update.games
    total.positions += update.positions
    total.leaves += update.leaves
    total.nn_evals += update.nn_evals
    total.wall_time += update.wall_time
    total.inference_time += update.inference_time
    total.plumbing_time += update.plumbing_time
    total.routed_fast_path_batches += update.routed_fast_path_batches
    total.routed_split_batches += update.routed_split_batches
    total.scripted_games += update.scripted_games
    total.scripted_wins += update.scripted_wins
    total.deep_games += update.deep_games
    total.deep_positions += update.deep_positions
    total.cards_trashed += update.cards_trashed
    for index in range(7):
        total.opening_template_games[index] += update.opening_template_games[index]
        total.opening_template_vs_unconstrained_games[index] += (
            update.opening_template_vs_unconstrained_games[index]
        )
        total.opening_template_vs_unconstrained_wins[index] += (
            update.opening_template_vs_unconstrained_wins[index]
        )
    for index in range(4):
        total.unconstrained_buy_counts[index] += update.unconstrained_buy_counts[index]
    for kind, (games, wins) in update.scripted_by_kind.items():
        previous_games, previous_wins = total.scripted_by_kind.get(kind, (0, 0))
        total.scripted_by_kind[kind] = (previous_games + games, previous_wins + wins)
    for opponent, (games, wins) in update.league_by_opponent.items():
        previous_games, previous_wins = total.league_by_opponent.get(opponent, (0, 0))
        total.league_by_opponent[opponent] = (previous_games + games, previous_wins + wins)


def _worker_main(
    config: TrainConfig,
    worker_index: int,
    runner_games: int,
    command_queue: Any,
    result_queue: Any,
    inference_endpoints: InferenceServerEndpoints | None,
) -> None:
    """Worker entry point. Kept module-level for the spawn start method."""
    parent_pid = os.getppid()
    generation = -1
    shared_views: WorkerSharedMemoryViews | None = None
    try:
        worker_seed = int(config.seed) + worker_index
        server_mode = _server_selfplay_enabled(config)
        if server_mode and inference_endpoints is None:
            raise RuntimeError("shared server self-play requires inference-server endpoints")
        device = None if server_mode else _worker_device(config.worker_device)
        _seed_worker(worker_seed, device)
        worker_selfplay = copy.deepcopy(config.selfplay)
        # A process needs only its generation quota of in-flight games. This
        # avoids allocating the full global n_games pipeline in every worker.
        worker_selfplay.n_games = min(worker_selfplay.n_games, runner_games)
        # The legacy stream is allocated lazily. Segmented generations instead
        # get exactly one manifest-backed runner for their full quota.
        runner: Any | None = None
        obs_size = obs_size_for_config(config)
        if server_mode:
            assert inference_endpoints is not None
            evaluate, shared_views = _server_evaluator(inference_endpoints, worker_index)
            collect_max_batch = inference_endpoints.request_batch_size
            model: torch.nn.Module | None = None
        else:
            assert device is not None
            model = build_model(config.model, obs_size, dz.ACTION_SPACE_SIZE).to(device)
            model.eval()
            evaluate = _local_evaluator(model, device)
            collect_max_batch = config.selfplay.max_batch
        # A collect/provide batch can complete a few games past its quota.
        # They are emitted first next generation, giving the documented bound
        # of at most one generation of model-weight staleness.
        deferred: list[dict[str, Any]] = []

        while True:
            try:
                command = command_queue.get(timeout=0.5)
            except queue.Empty:
                if os.getppid() != parent_pid:
                    return
                continue
            if command[0] == "stop":
                return
            (
                _,
                generation,
                target_games,
                state_payload,
                segments,
                model_state_payloads,
                opening_settings,
            ) = command
            (
                opening_enabled,
                opening_lambda,
                opening_turn_window,
                opening_weights,
            ) = opening_settings
            normalized_opening_weights = list(opening_weights)
            previous_opening_settings = (
                bool(worker_selfplay.opening_templates_enabled),
                float(worker_selfplay.opening_lambda),
                int(worker_selfplay.opening_turn_window),
                list(worker_selfplay.template_weights),
            )
            next_opening_settings = (
                bool(opening_enabled),
                float(opening_lambda),
                int(opening_turn_window),
                normalized_opening_weights,
            )
            if previous_opening_settings != next_opening_settings:
                worker_selfplay.opening_templates_enabled = next_opening_settings[0]
                worker_selfplay.opening_lambda = next_opening_settings[1]
                worker_selfplay.opening_turn_window = next_opening_settings[2]
                worker_selfplay.template_weights = next_opening_settings[3]
                # A legacy stream owns the native config for its full life;
                # rebuild only when a generation changes opening forcing.
                runner = None
            model_table: list[torch.nn.Module] = []
            if model is not None:
                primary_payload = model_state_payloads[0] if model_state_payloads else state_payload
                if primary_payload is None:
                    raise RuntimeError("local self-play is missing a model state payload")
                primary_state, _ = _unpack_model_state_payload(primary_payload, config.model)
                model.load_state_dict(deserialize_cpu_state_dict(primary_state))
                model.eval()
                model_table.append(model)
                for payload in model_state_payloads[1:]:
                    opponent_state, opponent_config = _unpack_model_state_payload(payload, config.model)
                    configured_version = opponent_config.get("obs_version")
                    opponent_version = int(
                        config.selfplay.obs_version if configured_version is None else configured_version
                    )
                    opponent = build_model(
                        opponent_config,
                        obs_size_for_version(opponent_version),
                        dz.ACTION_SPACE_SIZE,
                    ).to(device)
                    opponent.load_state_dict(deserialize_cpu_state_dict(opponent_state))
                    opponent.eval()
                    model_table.append(opponent)
            if segments:
                indexed_segments = index_selfplay_segments(segments)
                slots = build_slot_manifest(indexed_segments)
                if len(slots) != int(target_games):
                    raise RuntimeError("self-play slot manifest does not match the worker game quota")
                if model is None or device is None:
                    manifest_model_table: list[torch.nn.Module] | None = None
                    manifest_device: torch.device | None = None
                    manifest_evaluate = evaluate
                else:
                    if any(
                        not (
                            0 <= slot.seat0_model_id < len(model_table)
                            and 0 <= slot.seat1_model_id < len(model_table)
                        )
                        for slot in slots
                    ):
                        raise RuntimeError("self-play slot references an unknown model id")
                    manifest_model_table = list(model_table)
                    manifest_device = device
                    manifest_evaluate = None
                manifest_runner = dz.SelfPlayRunner(
                    make_runner_config(
                        worker_selfplay,
                        worker_seed ^ (int(generation) * 0x9E37),
                        slot_manifest=slots,
                    )
                )
                route_audit: dict[tuple[int, int], int] = {}
                stats = _generate_manifest_games(
                    manifest_runner,
                    slots,
                    manifest_evaluate,
                    manifest_model_table,
                    manifest_device,
                    collect_max_batch,
                    result_queue,
                    worker_index,
                    int(generation),
                    obs_size=obs_size,
                    route_audit=route_audit,
                )
                league_games = sum(
                    1
                    for slot in slots
                    if slot.scripted_kind is None and slot.seat0_model_id != slot.seat1_model_id
                )
                deferred = []
            else:
                if runner is None:
                    runner = dz.SelfPlayRunner(make_runner_config(worker_selfplay, worker_seed))
                stats, deferred = _generate_games(
                    runner,
                    evaluate,
                    collect_max_batch,
                    int(target_games),
                    deferred,
                    result_queue,
                    worker_index,
                    int(generation),
                    obs_size,
                )
                league_games = 0
                route_audit = {}
            result_queue.put(
                (
                    "done",
                    worker_index,
                    int(generation),
                    (
                        stats.games,
                        stats.positions,
                        stats.leaves,
                        stats.nn_evals,
                        stats.wall_time,
                        stats.inference_time,
                        stats.plumbing_time,
                        stats.routed_fast_path_batches,
                        stats.routed_split_batches,
                        stats.scripted_games,
                        stats.scripted_wins,
                        stats.scripted_by_kind,
                        stats.deep_games,
                        stats.deep_positions,
                        league_games,
                        stats.league_by_opponent,
                        route_audit,
                        stats.opening_template_games,
                        stats.opening_template_vs_unconstrained_games,
                        stats.opening_template_vs_unconstrained_wins,
                        stats.cards_trashed,
                        stats.unconstrained_buy_counts,
                    ),
                )
            )
    except BaseException:
        # Re-raise after attempting to notify the parent so an unrecoverable
        # queue failure is still visible through the worker's non-zero exit.
        result_queue.put(("error", worker_index, generation, traceback.format_exc()))
        raise
    finally:
        if shared_views is not None:
            shared_views.close()


class ParallelSelfPlayPool:
    """A persistent spawn-process pool that synchronizes weights per generation."""

    def __init__(self, config: TrainConfig, inference_server: InferenceServer | None = None):
        self.config = copy.deepcopy(config)
        self.quotas = game_quotas(config.selfplay.games_per_generation, config.parallel_workers)
        server_mode = _server_selfplay_enabled(config)
        if server_mode != (inference_server is not None):
            raise ValueError("shared server self-play requires exactly one inference server")
        self.inference_server = inference_server
        endpoints = inference_server.endpoints if inference_server is not None else None
        context = mp.get_context("spawn")
        self.result_queue = context.Queue(maxsize=max(2, config.parallel_workers * 2))
        self.command_queues = [context.Queue(maxsize=1) for _ in self.quotas]
        self.processes = [
            context.Process(
                target=_worker_main,
                args=(self.config, index, quota, command_queue, self.result_queue, endpoints),
                name=f"dominion-selfplay-{index}",
            )
            for index, (quota, command_queue) in enumerate(zip(self.quotas, self.command_queues))
        ]
        for process in self.processes:
            process.start()

    def generate(
        self,
        model: torch.nn.Module,
        replay: Any,
        generation: int,
        segments: list[SelfPlaySegment] | None = None,
        model_state_payloads: list[ModelStatePayload] | None = None,
        selfplay_config: Any | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ParallelSelfPlayResult:
        effective_selfplay = self.config.selfplay if selfplay_config is None else selfplay_config
        worker_model_payloads = model_state_payloads
        if self.inference_server is not None:
            self.inference_server.ensure_alive()
            state_payload = None
        elif segments is not None:
            # Segmented (gated/league) work carries a deduplicated model table
            # below.  Do not also pickle the primary model in the legacy slot.
            state_payload = None
        else:
            state_payload = serialize_cpu_state_dict(model)
        if segments is not None:
            # Index once before quota splitting. Reserved deep/league blocks
            # may move between workers, but their slot seeds keep this global
            # plan position rather than acquiring a worker-local index.
            segments = index_selfplay_segments(segments)
            if sum(segment.n_games for segment in segments) != self.config.selfplay.games_per_generation:
                raise ValueError("self-play segments must cover exactly one generation")
            if self.inference_server is None:
                worker_model_payloads = worker_model_payloads or [serialize_cpu_state_dict(model)]
                if not worker_model_payloads:
                    raise ValueError("local self-play segments require a best-model payload")
            per_worker_segments = split_segments_by_quotas(segments, self.quotas)
        else:
            per_worker_segments = [None] * len(self.quotas)

        if self.inference_server is not None:
            server_payloads = worker_model_payloads or [
                (
                    serialize_cpu_state_dict(model),
                    model_config_dict(getattr(model, "_dominion_model_config", self.config.model)),
                )
            ]
            if segments is not None:
                highest_model_id = max(
                    max(segment.seat0_model_id, segment.seat1_model_id)
                    for segment in segments
                )
                if highest_model_id >= len(server_payloads):
                    raise ValueError("self-play segment references a missing server model payload")
            self.inference_server.sync_models(server_payloads, generation)
            # Model bytes are installed once in the server, never copied into
            # the worker command queues.
            worker_model_payloads = []
        elif segments is None:
            worker_model_payloads = []

        for command_queue, quota, worker_segments in zip(self.command_queues, self.quotas, per_worker_segments):
            opening_settings = (
                bool(effective_selfplay.opening_templates_enabled),
                float(effective_selfplay.opening_lambda),
                int(effective_selfplay.opening_turn_window),
                list(effective_selfplay.template_weights),
            )
            command_queue.put(
                (
                    "generate",
                    generation,
                    quota,
                    state_payload,
                    worker_segments,
                    worker_model_payloads,
                    opening_settings,
                )
            )

        stats = SelfPlayStats()
        received_by_worker = [0 for _ in self.quotas]
        league_games = 0
        seat_model_evals: dict[tuple[int, int], int] = {}
        completed: set[int] = set()
        start = time.perf_counter()
        if progress_callback is not None:
            progress_callback(stats.games, stats.positions)
        while len(completed) < len(self.quotas):
            try:
                message = self.result_queue.get(timeout=1.0)
            except queue.Empty:
                if self.inference_server is not None:
                    self.inference_server.ensure_alive()
                failed = [process.name for process in self.processes if process.exitcode not in (None, 0)]
                if failed:
                    raise RuntimeError(f"self-play worker exited unexpectedly: {', '.join(failed)}")
                if progress_callback is not None:
                    progress_callback(stats.games, stats.positions)
                continue

            kind, worker_index, message_generation, payload = message
            if kind == "error":
                raise RuntimeError(f"self-play worker {worker_index} failed:\n{payload}")
            if message_generation != generation:
                raise RuntimeError("received a self-play result for the wrong generation")
            if kind == "records":
                games, positions = add_packed_records(replay, payload)
                received_by_worker[worker_index] += games
                stats.games += games
                stats.positions += positions
                # Records are the established worker-to-trainer transport, so
                # this incremental count adds no worker IPC on the hot path.
                if progress_callback is not None:
                    progress_callback(stats.games, stats.positions)
                continue
            if kind != "done":
                raise RuntimeError(f"unknown self-play worker message: {kind}")
            if worker_index in completed:
                raise RuntimeError("self-play worker sent duplicate completion")
            (
                worker_games,
                worker_positions,
                leaves,
                nn_evals,
                wall,
                inference,
                plumbing,
                worker_fast_path_batches,
                worker_split_batches,
                worker_scripted_games,
                worker_scripted_wins,
                worker_scripted_by_kind,
                worker_deep_games,
                worker_deep_positions,
                worker_league_games,
                worker_league_by_opponent,
                worker_route_audit,
                worker_opening_template_games,
                worker_opening_template_vs_unconstrained_games,
                worker_opening_template_vs_unconstrained_wins,
                worker_cards_trashed,
                worker_unconstrained_buy_counts,
            ) = payload
            if received_by_worker[worker_index] != self.quotas[worker_index]:
                raise RuntimeError("self-play worker completed before delivering its full game quota")
            if worker_games != self.quotas[worker_index]:
                raise RuntimeError("self-play worker reported an incorrect game quota")
            # Positions and games were counted as records arrived; timing work
            # is summed across workers for the existing detailed metrics.
            if worker_positions < 0:
                raise RuntimeError("self-play worker reported invalid positions")
            stats.leaves += leaves
            stats.nn_evals += nn_evals
            stats.wall_time += wall
            stats.inference_time += inference
            stats.plumbing_time += plumbing
            stats.routed_fast_path_batches += worker_fast_path_batches
            stats.routed_split_batches += worker_split_batches
            stats.scripted_games += worker_scripted_games
            stats.scripted_wins += worker_scripted_wins
            stats.deep_games += worker_deep_games
            stats.deep_positions += worker_deep_positions
            stats.cards_trashed += worker_cards_trashed
            for index in range(7):
                stats.opening_template_games[index] += int(worker_opening_template_games[index])
                stats.opening_template_vs_unconstrained_games[index] += int(
                    worker_opening_template_vs_unconstrained_games[index]
                )
                stats.opening_template_vs_unconstrained_wins[index] += int(
                    worker_opening_template_vs_unconstrained_wins[index]
                )
            for index in range(4):
                stats.unconstrained_buy_counts[index] += int(worker_unconstrained_buy_counts[index])
            for kind, (games, wins) in worker_scripted_by_kind.items():
                previous_games, previous_wins = stats.scripted_by_kind.get(kind, (0, 0))
                stats.scripted_by_kind[kind] = (previous_games + games, previous_wins + wins)
            for opponent, (games, wins) in worker_league_by_opponent.items():
                previous_games, previous_wins = stats.league_by_opponent.get(opponent, (0, 0))
                stats.league_by_opponent[opponent] = (previous_games + games, previous_wins + wins)
            league_games += worker_league_games
            for key, count in worker_route_audit.items():
                normalized = (int(key[0]), int(key[1]))
                seat_model_evals[normalized] = seat_model_evals.get(normalized, 0) + int(count)
            completed.add(worker_index)

        expected = self.config.selfplay.games_per_generation
        if stats.games != expected:
            raise RuntimeError(f"parallel self-play collected {stats.games} games, expected {expected}")
        if segments is not None:
            expected_league_games = sum(segment.n_games for segment in segments if segment.is_league)
            if league_games != expected_league_games:
                raise RuntimeError(
                    f"parallel league self-play collected {league_games} league games, "
                    f"expected {expected_league_games}"
                )
            expected_scripted_games = sum(segment.n_games for segment in segments if segment.is_scripted)
            if stats.scripted_games != expected_scripted_games:
                raise RuntimeError(
                    f"parallel scripted self-play collected {stats.scripted_games} scripted games, "
                    f"expected {expected_scripted_games}"
                )
            expected_deep_games = sum(segment.n_games for segment in segments if segment.sims_override > 0)
            if stats.deep_games != expected_deep_games:
                raise RuntimeError(
                    f"parallel deep self-play collected {stats.deep_games} deep games, "
                    f"expected {expected_deep_games}"
                )
        if self.inference_server is not None:
            self.inference_server.ensure_alive()
        return ParallelSelfPlayResult(
            stats=stats,
            collection_wall_time=time.perf_counter() - start,
            league_games=league_games,
            seat_model_evals=seat_model_evals,
        )

    def close(self) -> None:
        for command_queue in self.command_queues:
            try:
                command_queue.put(("stop",), timeout=1.0)
            except (queue.Full, ValueError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10.0)
        for command_queue in self.command_queues:
            command_queue.close()
        self.result_queue.close()
