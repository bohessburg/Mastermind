"""Multiprocess self-play collection for the v2 trainer.

Parallel collection deliberately does not promise a deterministic insertion
order in the parent replay buffer: worker result messages arrive as soon as
they are ready.  Each worker itself is seed-pinned to ``config.seed + index``
and owns one persistent ``SelfPlayRunner``, so its game stream is reproducible.
"""

from __future__ import annotations

import copy
import io
import multiprocessing as mp
import queue
import random
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

import dominion_v2_py as dz

from .config import TrainConfig
from .model import DominionNet
from .selfplay import SelfPlayStats, make_runner_config


PackedGameRecords = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class ParallelSelfPlayResult:
    stats: SelfPlayStats
    collection_wall_time: float

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


def _seed_worker(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    else:
        # CPU smoke runs should exercise process parallelism, not oversubscribe
        # every core with two independent tiny Torch inference sessions.
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=True)


def _load_cpu_weights(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    # load_state_dict performs the one device-local copy required by this
    # worker's inference session.
    model.load_state_dict(state_dict)


def _deserialize_cpu_state_dict(payload: bytes) -> dict[str, torch.Tensor]:
    """Restore the CPU tensor state sent through a regular process queue."""
    buffer = io.BytesIO(payload)
    try:
        return torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older supported Torch versions
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu")


def _generate_games(
    runner: Any,
    model: torch.nn.Module,
    device: torch.device,
    config: TrainConfig,
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

    model.eval()
    with torch.no_grad():
        while sent < target_games:
            plumbing_start = time.perf_counter()
            obs, masks = runner.collect_leaves(config.selfplay.max_batch)
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


def _worker_main(
    config: TrainConfig,
    worker_index: int,
    runner_games: int,
    command_queue: Any,
    result_queue: Any,
) -> None:
    """Worker entry point. Kept module-level for the spawn start method."""
    generation = -1
    try:
        worker_seed = int(config.seed) + worker_index
        device = _worker_device(config.worker_device)
        _seed_worker(worker_seed, device)
        worker_selfplay = copy.deepcopy(config.selfplay)
        # A process needs only its generation quota of in-flight games. This
        # avoids allocating the full global n_games pipeline in every worker.
        worker_selfplay.n_games = min(worker_selfplay.n_games, runner_games)
        runner = dz.SelfPlayRunner(make_runner_config(worker_selfplay, worker_seed))
        model = DominionNet(dz.OBS_SIZE, dz.ACTION_SPACE_SIZE, config.model.hidden_sizes).to(device)
        # A collect/provide batch can complete a few games past its quota.
        # They are emitted first next generation, giving the documented bound
        # of at most one generation of model-weight staleness.
        deferred: list[dict[str, Any]] = []

        while True:
            command = command_queue.get()
            if command[0] == "stop":
                return
            _, generation, target_games, state_payload = command
            state_dict = _deserialize_cpu_state_dict(state_payload)
            _load_cpu_weights(model, state_dict)
            stats, deferred = _generate_games(
                runner,
                model,
                device,
                config,
                int(target_games),
                deferred,
                result_queue,
                worker_index,
                int(generation),
            )
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
                    ),
                )
            )
    except BaseException:
        # Re-raise after attempting to notify the parent so an unrecoverable
        # queue failure is still visible through the worker's non-zero exit.
        result_queue.put(("error", worker_index, generation, traceback.format_exc()))
        raise


class ParallelSelfPlayPool:
    """A persistent spawn-process pool that synchronizes weights per generation."""

    def __init__(self, config: TrainConfig):
        self.config = copy.deepcopy(config)
        self.quotas = game_quotas(config.selfplay.games_per_generation, config.parallel_workers)
        context = mp.get_context("spawn")
        self.result_queue = context.Queue(maxsize=max(2, config.parallel_workers * 2))
        self.command_queues = [context.Queue(maxsize=1) for _ in self.quotas]
        self.processes = [
            context.Process(
                target=_worker_main,
                args=(self.config, index, quota, command_queue, self.result_queue),
                name=f"dominion-selfplay-{index}",
            )
            for index, (quota, command_queue) in enumerate(zip(self.quotas, self.command_queues))
        ]
        for process in self.processes:
            process.start()

    @staticmethod
    def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    @classmethod
    def _serialized_cpu_state_dict(cls, model: torch.nn.Module) -> bytes:
        # Sending Tensor objects directly activates Torch's shared-memory
        # reducer. A torch.save archive still contains only CPU tensors, while
        # travelling as ordinary queue bytes on platforms without that service.
        buffer = io.BytesIO()
        torch.save(cls._cpu_state_dict(model), buffer)
        return buffer.getvalue()

    def generate(
        self,
        model: torch.nn.Module,
        replay: Any,
        generation: int,
    ) -> ParallelSelfPlayResult:
        state_payload = self._serialized_cpu_state_dict(model)
        for command_queue, quota in zip(self.command_queues, self.quotas):
            command_queue.put(("generate", generation, quota, state_payload))

        stats = SelfPlayStats()
        received_by_worker = [0 for _ in self.quotas]
        completed: set[int] = set()
        start = time.perf_counter()
        while len(completed) < len(self.quotas):
            try:
                message = self.result_queue.get(timeout=1.0)
            except queue.Empty:
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
            worker_games, worker_positions, leaves, nn_evals, wall, inference, plumbing = payload
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
            completed.add(worker_index)

        expected = self.config.selfplay.games_per_generation
        if stats.games != expected:
            raise RuntimeError(f"parallel self-play collected {stats.games} games, expected {expected}")
        return ParallelSelfPlayResult(stats=stats, collection_wall_time=time.perf_counter() - start)

    def close(self) -> None:
        for command_queue in self.command_queues:
            try:
                command_queue.put(("stop",), timeout=1.0)
            except (queue.Full, ValueError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10.0)
        for command_queue in self.command_queues:
            command_queue.close()
        self.result_queue.close()
