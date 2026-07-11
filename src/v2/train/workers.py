"""Multiprocess self-play collection for the v2 trainer.

Parallel collection deliberately does not promise a deterministic insertion
order in the parent replay buffer: worker result messages arrive as soon as
they are ready.  Each worker itself is seed-pinned to ``config.seed + index``
and owns one persistent ``SelfPlayRunner``, so its game stream is reproducible.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import queue
import random
import time
import traceback
from dataclasses import dataclass
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
from .model import DominionNet
from .selfplay import SelfPlayStats, make_runner_config, route_leaf_evaluations


PackedGameRecords = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


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
    """Split exact game segments across workers without duplicating a model id."""
    if sum(segment.n_games for segment in segments) != sum(quotas):
        raise ValueError("self-play segments must cover exactly one generation")
    per_worker: list[list[SelfPlaySegment]] = []
    segment_index = 0
    remaining = segments[0].n_games if segments else 0
    for quota in quotas:
        worker_segments: list[SelfPlaySegment] = []
        needed = quota
        while needed > 0:
            if segment_index >= len(segments):
                raise ValueError("self-play segment allocation ended early")
            source = segments[segment_index]
            take = min(needed, remaining)
            worker_segments.append(
                SelfPlaySegment(
                    take,
                    source.seat0_model_id,
                    source.seat1_model_id,
                    scripted_kind=source.scripted_kind,
                    nn_player=source.nn_player,
                )
            )
            needed -= take
            remaining -= take
            if remaining == 0:
                segment_index += 1
                if segment_index < len(segments):
                    remaining = segments[segment_index].n_games
        per_worker.append(worker_segments)
    if segment_index != len(segments):
        raise ValueError("self-play segment allocation left unassigned games")
    return per_worker


def _empty_packed_records() -> PackedGameRecords:
    return (
        np.empty((0,), dtype=np.int32),
        np.empty((0, dz.OBS_SIZE), dtype=np.float32),
        np.empty((0, dz.ACTION_SPACE_SIZE), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )


def _pack_records(records: list[dict[str, Any]]) -> PackedGameRecords:
    """Flatten finished-game dictionaries into four raw NumPy buffers.

    Queue messages carry only these contiguous arrays and a tiny tuple header,
    never the binding's large list-of-dicts game-record representation.
    """
    if not records:
        return _empty_packed_records()
    observations = [np.asarray(record["observations"], dtype=np.float32) for record in records]
    policies = [np.asarray(record["policy_targets"], dtype=np.float32) for record in records]
    values = [np.asarray(record["values"], dtype=np.float32) for record in records]
    lengths = np.asarray([obs.shape[0] for obs in observations], dtype=np.int32)
    nonempty = lengths > 0
    if not np.any(nonempty):
        empty = _empty_packed_records()
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
) -> tuple[Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]], WorkerSharedMemoryViews | None]:
    response_queue = endpoints.response_queues[worker_index]
    request_id = 0
    shared_views = (
        WorkerSharedMemoryViews(endpoints.shared_memory_specs[worker_index])
        if endpoints.transport == "shm"
        else None
    )

    def evaluate(obs: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal request_id
        request_id += 1
        deadline = time.monotonic() + endpoints.response_timeout_s
        count = int(obs.shape[0])
        if count <= 0 or count > endpoints.request_batch_size:
            raise RuntimeError("worker attempted an invalid inference-server request batch")
        if shared_views is not None:
            slot = request_id % shared_views.spec.slots
            np.copyto(shared_views.request_obs[slot, :count], np.asarray(obs, dtype=np.float32))
            np.copyto(shared_views.request_masks[slot, :count], np.asarray(masks, dtype=np.uint8))
            if endpoints.poll == "spin":
                shared_views.request_counts[slot] = count
                shared_views.request_sequences[slot] = request_id
                request = ()
            else:
                request = (worker_index, slot, count, request_id)
        else:
            request = (
                worker_index,
                request_id,
                np.ascontiguousarray(obs, dtype=np.float32),
                np.ascontiguousarray(masks, dtype=np.uint8),
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
) -> tuple[SelfPlayStats, list[dict[str, Any]]]:
    stats = SelfPlayStats()
    start = time.perf_counter()
    sent = 0

    def emit(records: list[dict[str, Any]]) -> None:
        nonlocal sent
        if not records:
            return
        packed = _pack_records(records)
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


def _generate_games_exact(
    runner: Any,
    evaluate: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    collect_max_batch: int,
    target_games: int,
    result_queue: Any,
    worker_index: int,
    generation: int,
    scripted_kind: str | None = None,
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
        _record_scripted_outcomes(stats, records, scripted_kind)
        packed = _pack_records(records)
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
) -> SelfPlayStats:
    """Generate a task with fixed models for player zero and player one."""
    if target_games <= 0:
        return SelfPlayStats()
    task_config = copy.deepcopy(selfplay_config)
    task_config.n_games = max(1, min(int(task_config.n_games), int(target_games)))
    runner = dz.SelfPlayRunner(
        make_runner_config(
            task_config,
            seed,
            scripted_kind=scripted_kind,
            scripted_nn_player=scripted_nn_player,
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
        _record_scripted_outcomes(stats, records, scripted_kind)
        packed = _pack_records(records)
        games, positions = int(packed[0].shape[0]), int(packed[0].sum())
        if games:
            result_queue.put(("records", worker_index, generation, packed))
            sent += games
            stats.games += games
            stats.positions += positions
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
    for kind, (games, wins) in update.scripted_by_kind.items():
        previous_games, previous_wins = total.scripted_by_kind.get(kind, (0, 0))
        total.scripted_by_kind[kind] = (previous_games + games, previous_wins + wins)


def _worker_main(
    config: TrainConfig,
    worker_index: int,
    runner_games: int,
    command_queue: Any,
    result_queue: Any,
    inference_endpoints: InferenceServerEndpoints | None,
) -> None:
    """Worker entry point. Kept module-level for the spawn start method."""
    generation = -1
    shared_views: WorkerSharedMemoryViews | None = None
    try:
        worker_seed = int(config.seed) + worker_index
        server_mode = config.worker_device.lower() == "server"
        if server_mode and inference_endpoints is None:
            raise RuntimeError("worker_device=server requires inference-server endpoints")
        device = None if server_mode else _worker_device(config.worker_device)
        _seed_worker(worker_seed, device)
        worker_selfplay = copy.deepcopy(config.selfplay)
        # A process needs only its generation quota of in-flight games. This
        # avoids allocating the full global n_games pipeline in every worker.
        worker_selfplay.n_games = min(worker_selfplay.n_games, runner_games)
        runner = dz.SelfPlayRunner(make_runner_config(worker_selfplay, worker_seed))
        if server_mode:
            assert inference_endpoints is not None
            evaluate, shared_views = _server_evaluator(inference_endpoints, worker_index)
            collect_max_batch = inference_endpoints.request_batch_size
            model: DominionNet | None = None
        else:
            assert device is not None
            model = DominionNet(dz.OBS_SIZE, dz.ACTION_SPACE_SIZE, config.model.hidden_sizes).to(device)
            model.eval()
            evaluate = _local_evaluator(model, device)
            collect_max_batch = config.selfplay.max_batch
        # A collect/provide batch can complete a few games past its quota.
        # They are emitted first next generation, giving the documented bound
        # of at most one generation of model-weight staleness.
        deferred: list[dict[str, Any]] = []

        while True:
            command = command_queue.get()
            if command[0] == "stop":
                return
            _, generation, target_games, state_payload, segments, model_state_payloads = command
            model_table: list[DominionNet] = []
            if model is not None:
                primary_payload = model_state_payloads[0] if model_state_payloads else state_payload
                model.load_state_dict(deserialize_cpu_state_dict(primary_payload))
                model.eval()
                model_table.append(model)
                for payload in model_state_payloads[1:]:
                    opponent = DominionNet(dz.OBS_SIZE, dz.ACTION_SPACE_SIZE, config.model.hidden_sizes).to(device)
                    opponent.load_state_dict(deserialize_cpu_state_dict(payload))
                    opponent.eval()
                    model_table.append(opponent)
            if segments:
                stats = SelfPlayStats()
                league_games = 0
                route_audit: dict[tuple[int, int], int] = {}
                for task_index, segment in enumerate(segments):
                    if segment.n_games <= 0:
                        raise RuntimeError("self-play segment has a non-positive game count")
                    if model is None or device is None:
                        if (segment.seat0_model_id, segment.seat1_model_id) != (0, 0):
                            raise RuntimeError("mini-league self-play requires worker_device='cpu' or 'cuda'")
                        task_runner = runner
                        if segment.is_scripted:
                            task_config = copy.deepcopy(worker_selfplay)
                            task_config.n_games = max(1, min(int(task_config.n_games), int(segment.n_games)))
                            task_runner = dz.SelfPlayRunner(
                                make_runner_config(
                                    task_config,
                                    worker_seed ^ (int(generation) * 0x9E37) ^ (task_index * 0x10001),
                                    scripted_kind=segment.scripted_kind,
                                    scripted_nn_player=segment.nn_player,
                                )
                            )
                        task_stats = _generate_games_exact(
                            task_runner,
                            evaluate,
                            collect_max_batch,
                            segment.n_games,
                            result_queue,
                            worker_index,
                            int(generation),
                            segment.scripted_kind,
                        )
                    else:
                        if not (
                            0 <= segment.seat0_model_id < len(model_table)
                            and 0 <= segment.seat1_model_id < len(model_table)
                        ):
                            raise RuntimeError("self-play segment references an unknown model id")
                        # A segment can be shorter than this worker's total
                        # quota.  Use a fresh exact-sized runner rather than
                        # stopping the persistent quota-sized runner midway
                        # through its in-flight games and accidentally carrying
                        # best-vs-best state into a league segment.
                        task_stats = _generate_routed_games(
                            (model_table[segment.seat0_model_id], model_table[segment.seat1_model_id]),
                            worker_selfplay,
                            worker_seed ^ (int(generation) * 0x9E37) ^ (task_index * 0x10001),
                            device,
                            segment.n_games,
                            result_queue,
                            worker_index,
                            int(generation),
                            (segment.seat0_model_id, segment.seat1_model_id),
                            route_audit,
                            segment.seat0_model_id == segment.seat1_model_id,
                            segment.scripted_kind,
                            segment.nn_player,
                        )
                        if segment.is_league:
                            league_games += task_stats.games
                    _accumulate_stats(stats, task_stats)
                deferred = []
            else:
                stats, deferred = _generate_games(
                    runner,
                    evaluate,
                    collect_max_batch,
                    int(target_games),
                    deferred,
                    result_queue,
                    worker_index,
                    int(generation),
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
                        league_games,
                        route_audit,
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
        server_mode = config.worker_device.lower() == "server"
        if server_mode != (inference_server is not None):
            raise ValueError("worker_device=server requires exactly one inference server")
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
        model_state_payloads: list[bytes] | None = None,
    ) -> ParallelSelfPlayResult:
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
            if sum(segment.n_games for segment in segments) != self.config.selfplay.games_per_generation:
                raise ValueError("self-play segments must cover exactly one generation")
            if self.inference_server is not None and any(segment.is_league for segment in segments):
                raise ValueError("mini-league self-play is unavailable with worker_device='server'")
            if self.inference_server is None:
                model_state_payloads = model_state_payloads or [serialize_cpu_state_dict(model)]
                if not model_state_payloads:
                    raise ValueError("local self-play segments require a best-model payload")
            else:
                model_state_payloads = []
            per_worker_segments = split_segments_by_quotas(segments, self.quotas)
        else:
            model_state_payloads = []
            per_worker_segments = [None] * len(self.quotas)
        for command_queue, quota, worker_segments in zip(self.command_queues, self.quotas, per_worker_segments):
            command_queue.put(
                ("generate", generation, quota, state_payload, worker_segments, model_state_payloads)
            )

        stats = SelfPlayStats()
        received_by_worker = [0 for _ in self.quotas]
        league_games = 0
        seat_model_evals: dict[tuple[int, int], int] = {}
        completed: set[int] = set()
        start = time.perf_counter()
        while len(completed) < len(self.quotas):
            try:
                message = self.result_queue.get(timeout=1.0)
            except queue.Empty:
                if self.inference_server is not None:
                    self.inference_server.ensure_alive()
                failed = [process.name for process in self.processes if process.exitcode not in (None, 0)]
                if failed:
                    raise RuntimeError(f"self-play worker exited unexpectedly: {', '.join(failed)}")
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
                worker_league_games,
                worker_route_audit,
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
            for kind, (games, wins) in worker_scripted_by_kind.items():
                previous_games, previous_wins = stats.scripted_by_kind.get(kind, (0, 0))
                stats.scripted_by_kind[kind] = (previous_games + games, previous_wins + wins)
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
