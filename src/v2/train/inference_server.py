"""One-process batched inference service with shared-memory worker rings.

In ``shm`` mode, queues carry only small request/response headers.  Leaf
arrays live in a two-slot request/response ring owned by each worker, avoiding
the large pickle copies that otherwise dominate small-batch self-play.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import io
import json
import multiprocessing as mp
import queue
import time
import traceback
import uuid
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

try:
    from multiprocessing import shared_memory
except ImportError:  # pragma: no cover - supported CPython versions provide it
    shared_memory = None  # type: ignore[assignment]

import dominion_v2_py as dz

from .config import TrainConfig
from .model import DominionNet
from .observation import obs_size_for_config


def _align(offset: int, alignment: int = 8) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _request_layout(spec: WorkerSharedMemorySpec) -> tuple[int, int, int, int]:
    obs_bytes = spec.slots * spec.max_request * spec.obs_size * np.dtype(np.float32).itemsize
    masks_offset = obs_bytes
    masks_bytes = spec.slots * spec.max_request * spec.action_size * np.dtype(np.uint8).itemsize
    counts_offset = _align(masks_offset + masks_bytes, np.dtype(np.uint64).itemsize)
    sequences_offset = _align(counts_offset + spec.slots * np.dtype(np.uint32).itemsize, np.dtype(np.uint64).itemsize)
    return masks_offset, counts_offset, sequences_offset, sequences_offset + spec.slots * np.dtype(np.uint64).itemsize


def _response_layout(spec: WorkerSharedMemorySpec) -> tuple[int, int, int]:
    values_bytes = spec.slots * spec.max_request * np.dtype(np.float32).itemsize
    policies_offset = values_bytes
    policies_bytes = spec.slots * spec.max_request * spec.action_size * np.dtype(np.float32).itemsize
    sequences_offset = _align(policies_offset + policies_bytes, np.dtype(np.uint64).itemsize)
    return policies_offset, sequences_offset, sequences_offset + spec.slots * np.dtype(np.uint64).itemsize


def serialize_cpu_state_dict(model: torch.nn.Module) -> bytes:
    """Make a portable queue payload containing a CPU tensor state_dict."""
    state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def deserialize_cpu_state_dict(payload: bytes) -> dict[str, torch.Tensor]:
    """Restore a CPU tensor state_dict sent by :func:`serialize_cpu_state_dict`."""
    buffer = io.BytesIO(payload)
    try:
        return torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older supported Torch versions
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu")


@dataclass(frozen=True)
class WorkerSharedMemorySpec:
    request_name: str
    response_name: str
    slots: int
    max_request: int
    obs_size: int
    action_size: int


class WorkerSharedMemoryViews:
    """Attached NumPy views for one worker's fixed-size request/response rings."""

    def __init__(self, spec: WorkerSharedMemorySpec):
        if shared_memory is None:  # pragma: no cover - guarded by allocator
            raise RuntimeError("multiprocessing.shared_memory is unavailable")
        self.spec = spec
        self.request_block = shared_memory.SharedMemory(name=spec.request_name)
        self.response_block = shared_memory.SharedMemory(name=spec.response_name)
        request_masks_offset, request_counts_offset, request_sequences_offset, _ = _request_layout(spec)
        response_policies_offset, response_sequences_offset, _ = _response_layout(spec)
        self.request_obs = np.ndarray(
            (spec.slots, spec.max_request, spec.obs_size),
            dtype=np.float32,
            buffer=self.request_block.buf,
        )
        self.request_masks = np.ndarray(
            (spec.slots, spec.max_request, spec.action_size),
            dtype=np.uint8,
            buffer=self.request_block.buf,
            offset=request_masks_offset,
        )
        self.request_counts = np.ndarray(
            (spec.slots,),
            dtype=np.uint32,
            buffer=self.request_block.buf,
            offset=request_counts_offset,
        )
        self.request_sequences = np.ndarray(
            (spec.slots,),
            dtype=np.uint64,
            buffer=self.request_block.buf,
            offset=request_sequences_offset,
        )
        self.response_values = np.ndarray(
            (spec.slots, spec.max_request),
            dtype=np.float32,
            buffer=self.response_block.buf,
        )
        self.response_policies = np.ndarray(
            (spec.slots, spec.max_request, spec.action_size),
            dtype=np.float32,
            buffer=self.response_block.buf,
            offset=response_policies_offset,
        )
        self.response_sequences = np.ndarray(
            (spec.slots,),
            dtype=np.uint64,
            buffer=self.response_block.buf,
            offset=response_sequences_offset,
        )

    def close(self) -> None:
        self.request_block.close()
        self.response_block.close()


class SharedMemoryTransport:
    """Parent-owned shared-memory allocation with idempotent cleanup."""

    def __init__(self, specs: list[WorkerSharedMemorySpec], blocks: list[Any]):
        self.specs = specs
        self._blocks = blocks
        self._closed = False
        atexit.register(self.close)

    @classmethod
    def create(
        cls,
        worker_count: int,
        slots: int,
        max_request: int,
        obs_size: int = dz.OBS_SIZE,
    ) -> SharedMemoryTransport:
        if shared_memory is None:
            raise RuntimeError("multiprocessing.shared_memory is unavailable")
        if worker_count <= 0 or slots < 2 or max_request <= 0:
            raise ValueError("invalid shared-memory ring dimensions")
        # macOS limits POSIX shared-memory names far more tightly than Linux.
        # A 12-hex token remains unique for a training invocation while
        # leaving room for the request/response and worker suffixes.
        token = f"dz{uuid.uuid4().hex[:12]}"
        layout_spec = WorkerSharedMemorySpec("", "", slots, max_request, int(obs_size), dz.ACTION_SPACE_SIZE)
        request_bytes = _request_layout(layout_spec)[3]
        response_bytes = _response_layout(layout_spec)[2]
        specs: list[WorkerSharedMemorySpec] = []
        blocks: list[Any] = []
        try:
            for worker_id in range(worker_count):
                request = shared_memory.SharedMemory(
                    create=True,
                    size=request_bytes,
                    name=f"{token}r{worker_id}",
                )
                response = shared_memory.SharedMemory(
                    create=True,
                    size=response_bytes,
                    name=f"{token}p{worker_id}",
                )
                blocks.extend((request, response))
                specs.append(
                    WorkerSharedMemorySpec(
                        request_name=request.name,
                        response_name=response.name,
                        slots=slots,
                        max_request=max_request,
                        obs_size=int(obs_size),
                        action_size=dz.ACTION_SPACE_SIZE,
                    )
                )
        except BaseException:
            for block in blocks:
                block.close()
                try:
                    block.unlink()
                except FileNotFoundError:
                    pass
            raise
        return cls(specs, blocks)

    @property
    def names(self) -> list[str]:
        return [block.name for block in self._blocks]

    def __enter__(self) -> SharedMemoryTransport:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for block in self._blocks:
            block.close()
            try:
                block.unlink()
            except FileNotFoundError:
                pass


@dataclass(frozen=True)
class InferenceServerEndpoints:
    request_queue: Any
    response_queues: list[Any]
    alive_event: Any
    response_timeout_s: float
    request_batch_size: int
    obs_size: int
    transport: str
    shared_memory_specs: list[WorkerSharedMemorySpec] | None
    poll: str


@dataclass(frozen=True)
class _Request:
    worker_id: int
    request_id: int
    slot: int | None
    count: int
    obs: np.ndarray
    masks: np.ndarray


@dataclass
class _GenerationMetrics:
    evals: int = 0
    batches: int = 0
    inference_time: float = 0.0
    batch_waits_s: list[float] | None = None

    def __post_init__(self) -> None:
        if self.batch_waits_s is None:
            self.batch_waits_s = []

    def snapshot(self, include_totals: bool = False) -> dict[str, float]:
        waits_ms = np.asarray(self.batch_waits_s, dtype=np.float64)
        metrics = {
            "server_evals_per_sec": self.evals / self.inference_time if self.inference_time > 0.0 else 0.0,
            "server_mean_batch_size": self.evals / self.batches if self.batches > 0 else 0.0,
            "server_batch_wait_p50_ms": float(np.percentile(waits_ms, 50.0)) if waits_ms.size else 0.0,
            "server_batch_wait_p99_ms": float(np.percentile(waits_ms, 99.0)) if waits_ms.size else 0.0,
        }
        if include_totals:
            metrics["_server_total_evals"] = float(self.evals)
            metrics["_server_total_batches"] = float(self.batches)
        return metrics


class _PinnedStaging:
    """Persistent host staging for a complete server batch and its responses."""

    def __init__(self, device: torch.device, max_batch: int, obs_size: int):
        pinned = device.type == "cuda"
        self.device = device
        self.obs = torch.empty((max_batch, int(obs_size)), dtype=torch.float32, pin_memory=pinned)
        self.masks = torch.empty((max_batch, dz.ACTION_SPACE_SIZE), dtype=torch.bool, pin_memory=pinned)
        self.values = torch.empty((max_batch,), dtype=torch.float32, pin_memory=pinned)
        self.policies = torch.empty((max_batch, dz.ACTION_SPACE_SIZE), dtype=torch.float32, pin_memory=pinned)
        self.obs_np = self.obs.numpy()
        # Bool and uint8 both occupy one byte. Workers intentionally write the
        # compact uint8 legal mask directly into this persistent Bool tensor.
        self.masks_np = self.masks.numpy().view(np.uint8)
        self.values_np = self.values.numpy()
        self.policies_np = self.policies.numpy()

    def load_requests(self, requests: list[_Request]) -> int:
        offset = 0
        for request in requests:
            end = offset + request.count
            self.obs_np[offset:end] = request.obs
            self.masks_np[offset:end] = request.masks
            offset = end
        return offset

    def forward(self, model: DominionNet, count: int, use_fp16: bool) -> float:
        start = time.perf_counter()
        if self.device.type == "cuda":
            obs = self.obs[:count].to(self.device, non_blocking=True)
            masks = self.masks[:count].to(self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                policies, values = model.evaluate(obs, masks)
            # Persistent pinned response tensors receive each aggregate output
            # once; individual worker slices are then copied into their rings.
            self.values[:count].copy_(values.float(), non_blocking=True)
            self.policies[:count].copy_(policies.float(), non_blocking=True)
            torch.cuda.current_stream(self.device).synchronize()
        else:
            policies, values = model.evaluate(self.obs[:count], self.masks[:count])
            self.values[:count].copy_(values.float())
            self.policies[:count].copy_(policies.float())
        return time.perf_counter() - start


def _server_device(name: str) -> torch.device:
    requested = name.lower()
    if requested not in {"cpu", "cuda"}:
        raise ValueError("server_device must be 'cpu' or 'cuda'")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("server_device=cuda requested but unavailable")
    return torch.device(requested)


def _server_main(
    config: TrainConfig,
    endpoints: InferenceServerEndpoints,
    command_queue: Any,
    status_queue: Any,
) -> None:
    """Process entry point; all model updates occur between complete batches."""
    carry: _Request | None = None
    running = True
    metrics = _GenerationMetrics()
    views: list[WorkerSharedMemoryViews] = []
    try:
        device = _server_device(config.server_device)
        obs_size = obs_size_for_config(config)
        model = DominionNet(obs_size, dz.ACTION_SPACE_SIZE, config.model.hidden_sizes).to(device)
        model.eval()
        staging = _PinnedStaging(device, int(config.server_max_batch), obs_size)
        if endpoints.transport == "shm":
            assert endpoints.shared_memory_specs is not None
            views = [WorkerSharedMemoryViews(spec) for spec in endpoints.shared_memory_specs]
        endpoints.alive_event.set()
        last_request_sequences = [np.zeros(view.spec.slots, dtype=np.uint64) for view in views]
        # Flatten this once.  The spin path executes this scan for every
        # request, so repeated worker/slot arithmetic would otherwise become
        # another per-batch Python cost at high worker counts.
        spin_slots = [
            (worker_id, slot)
            for worker_id, view in enumerate(views)
            for slot in range(view.spec.slots)
        ]
        spin_cursor = 0

        def handle_commands() -> None:
            nonlocal running, metrics
            while True:
                try:
                    command = command_queue.get_nowait()
                except queue.Empty:
                    return
                kind = command[0]
                if kind == "stop":
                    running = False
                    return
                if kind == "weights":
                    _, generation, payload = command
                    model.load_state_dict(deserialize_cpu_state_dict(payload))
                    model.eval()
                    status_queue.put(("weights", generation, None))
                    continue
                if kind == "metrics":
                    _, generation, include_totals = command
                    status_queue.put(("metrics", generation, metrics.snapshot(bool(include_totals))))
                    metrics = _GenerationMetrics()
                    continue
                raise RuntimeError(f"unknown inference-server command: {kind}")

        def shared_request_from_header(worker_id: int, slot: int, count: int, sequence: int) -> _Request:
            if not (0 <= worker_id < len(views)):
                raise ValueError("shared-memory request has invalid worker id")
            view = views[worker_id]
            if not (0 <= slot < view.spec.slots and 0 < count <= view.spec.max_request):
                raise ValueError("shared-memory request has invalid slot or count")
            if slot != sequence % view.spec.slots:
                raise ValueError("shared-memory request sequence does not match its slot")
            return _Request(
                worker_id=worker_id,
                request_id=sequence,
                slot=slot,
                count=count,
                obs=view.request_obs[slot, :count],
                masks=view.request_masks[slot, :count],
            )

        def poll_shared_sequences() -> _Request | None:
            nonlocal spin_cursor
            total_slots = len(spin_slots)
            if total_slots == 0:
                return None
            for scanned in range(total_slots):
                flat_slot = (spin_cursor + scanned) % total_slots
                worker_id, slot = spin_slots[flat_slot]
                view = views[worker_id]
                sequence = int(view.request_sequences[slot])
                if sequence <= int(last_request_sequences[worker_id][slot]):
                    continue
                count = int(view.request_counts[slot])
                request = shared_request_from_header(worker_id, slot, count, sequence)
                last_request_sequences[worker_id][slot] = sequence
                spin_cursor = (flat_slot + 1) % total_slots
                return request
            return None

        def dequeue_request(timeout: float | None = None) -> _Request:
            if endpoints.poll == "spin":
                deadline = None if timeout is None else time.perf_counter() + max(0.0, timeout)
                while True:
                    request = poll_shared_sequences()
                    if request is not None:
                        return request
                    if deadline is not None and time.perf_counter() >= deadline:
                        raise queue.Empty
                    # Yield without a kernel queue wakeup; request/response
                    # sequence counters provide the producer notification.
                    time.sleep(0)
            if endpoints.transport == "shm":
                if timeout is None:
                    worker_id, slot, count, sequence = endpoints.request_queue.get()
                elif timeout <= 0.0:
                    worker_id, slot, count, sequence = endpoints.request_queue.get_nowait()
                else:
                    worker_id, slot, count, sequence = endpoints.request_queue.get(timeout=timeout)
                worker_id, slot, count, sequence = int(worker_id), int(slot), int(count), int(sequence)
                return shared_request_from_header(worker_id, slot, count, sequence)
            if timeout is None:
                worker_id, request_id, obs, masks = endpoints.request_queue.get()
            elif timeout <= 0.0:
                worker_id, request_id, obs, masks = endpoints.request_queue.get_nowait()
            else:
                worker_id, request_id, obs, masks = endpoints.request_queue.get(timeout=timeout)
            obs = np.ascontiguousarray(obs, dtype=np.float32)
            masks = np.ascontiguousarray(masks, dtype=np.uint8)
            if (
                obs.ndim != 2
                or obs.shape[1] != endpoints.obs_size
                or masks.shape != (obs.shape[0], dz.ACTION_SPACE_SIZE)
            ):
                raise ValueError("queue inference request shape mismatch")
            return _Request(int(worker_id), int(request_id), None, int(obs.shape[0]), obs, masks)

        while running:
            handle_commands()
            if not running:
                break
            if carry is None:
                try:
                    request = dequeue_request(timeout=0.01)
                except queue.Empty:
                    continue
            else:
                request, carry = carry, None
            if request.count > config.server_max_batch:
                raise ValueError("inference request batch is outside server_max_batch")

            requests = [request]
            batch_size = request.count
            wait_start = time.perf_counter()
            deadline = wait_start + (float(config.server_max_wait_ms) / 1000.0)
            while batch_size < config.server_max_batch:
                try:
                    candidate = dequeue_request(timeout=0.0)
                except queue.Empty:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        break
                    try:
                        candidate = dequeue_request(timeout=remaining)
                    except queue.Empty:
                        break
                if candidate.count > config.server_max_batch:
                    raise ValueError("inference request batch is outside server_max_batch")
                if batch_size + candidate.count > config.server_max_batch:
                    carry = candidate
                    break
                requests.append(candidate)
                batch_size += candidate.count

            # Generation barriers ensure any newly received state is installed
            # only between complete batches, never halfway through a forward.
            handle_commands()
            if not running:
                break
            batch_wait = time.perf_counter() - wait_start
            assert staging.load_requests(requests) == batch_size
            inference_time = staging.forward(model, batch_size, bool(config.server_fp16))
            metrics.evals += batch_size
            metrics.batches += 1
            metrics.inference_time += inference_time
            metrics.batch_waits_s.append(batch_wait)

            offset = 0
            for request in requests:
                end = offset + request.count
                if endpoints.transport == "shm":
                    assert request.slot is not None
                    view = views[request.worker_id]
                    np.copyto(view.response_values[request.slot, : request.count], staging.values_np[offset:end])
                    np.copyto(view.response_policies[request.slot, : request.count], staging.policies_np[offset:end])
                    if endpoints.poll == "spin":
                        view.response_sequences[request.slot] = request.request_id
                    else:
                        endpoints.response_queues[request.worker_id].put((request.slot, request.count, request.request_id))
                else:
                    endpoints.response_queues[request.worker_id].put(
                        (
                            "response",
                            request.request_id,
                            staging.values_np[offset:end].copy(),
                            staging.policies_np[offset:end].copy(),
                        )
                    )
                offset = end
    except BaseException:
        endpoints.alive_event.clear()
        status_queue.put(("error", -1, traceback.format_exc()))
        raise
    finally:
        endpoints.alive_event.clear()
        for view in views:
            view.close()


class InferenceServer:
    """Parent-side lifecycle, transport allocation, and generation barriers."""

    def __init__(self, config: TrainConfig, worker_count: int):
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        self.config = copy.deepcopy(config)
        requested_transport = config.server_transport.lower()
        if requested_transport not in {"shm", "queue"}:
            raise ValueError("server_transport must be 'shm' or 'queue'")
        requested_poll = config.server_poll.lower()
        if requested_poll not in {"queue", "spin"}:
            raise ValueError("server_poll must be 'queue' or 'spin'")
        context = mp.get_context("spawn")
        self.request_queue = context.Queue(maxsize=max(4, worker_count * 4))
        self.response_queues = [context.Queue(maxsize=2) for _ in range(worker_count)]
        self.command_queue = context.Queue()
        self.status_queue = context.Queue()
        self.alive_event = context.Event()
        request_batch_size = max(1, min(config.selfplay.max_batch, config.server_max_batch // worker_count))
        obs_size = obs_size_for_config(config)
        self.shared_transport: SharedMemoryTransport | None = None
        self.transport = requested_transport
        if requested_transport == "shm":
            try:
                self.shared_transport = SharedMemoryTransport.create(
                    worker_count,
                    int(config.server_shm_slots),
                    request_batch_size,
                    obs_size,
                )
            except Exception as exc:
                self.transport = "queue"
                warnings.warn(
                    f"shared-memory inference transport unavailable ({exc}); falling back to queue transport",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self.poll = requested_poll
        if self.poll == "spin" and self.transport != "shm":
            self.poll = "queue"
            warnings.warn(
                "server_poll=spin requires shared-memory transport; falling back to queue polling",
                RuntimeWarning,
                stacklevel=2,
            )
        self.endpoints = InferenceServerEndpoints(
            request_queue=self.request_queue,
            response_queues=self.response_queues,
            alive_event=self.alive_event,
            response_timeout_s=float(config.server_response_timeout_s),
            request_batch_size=request_batch_size,
            obs_size=obs_size,
            transport=self.transport,
            shared_memory_specs=self.shared_transport.specs if self.shared_transport is not None else None,
            poll=self.poll,
        )
        self.process = context.Process(
            target=_server_main,
            args=(self.config, self.endpoints, self.command_queue, self.status_queue),
            name="dominion-inference-server",
        )
        try:
            self.process.start()
        except BaseException:
            self.request_queue.close()
            self.command_queue.close()
            self.status_queue.close()
            for response_queue in self.response_queues:
                response_queue.close()
            if self.shared_transport is not None:
                self.shared_transport.close()
            raise

    @property
    def shared_memory_names(self) -> list[str]:
        return self.shared_transport.names if self.shared_transport is not None else []

    def __enter__(self) -> InferenceServer:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def ensure_alive(self) -> None:
        if self.process.is_alive():
            return
        detail = ""
        try:
            kind, _, payload = self.status_queue.get_nowait()
            if kind == "error":
                detail = f"\n{payload}"
        except queue.Empty:
            pass
        raise RuntimeError(f"inference server exited unexpectedly (exitcode={self.process.exitcode}){detail}")

    def _wait_for_status(self, expected_kind: str, generation: int) -> Any:
        deadline = time.monotonic() + max(5.0, float(self.config.server_response_timeout_s))
        while True:
            self.ensure_alive()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError(f"timed out waiting for inference-server {expected_kind} acknowledgement")
            try:
                kind, message_generation, payload = self.status_queue.get(timeout=min(0.25, remaining))
            except queue.Empty:
                continue
            if kind == "error":
                raise RuntimeError(f"inference server failed:\n{payload}")
            if kind == expected_kind and message_generation == generation:
                return payload
            raise RuntimeError(f"unexpected inference-server status: {kind} for generation {message_generation}")

    def sync_weights(self, model: torch.nn.Module, generation: int) -> None:
        self.ensure_alive()
        self.command_queue.put(("weights", generation, serialize_cpu_state_dict(model)))
        self._wait_for_status("weights", generation)

    def collect_metrics(self, generation: int, include_totals: bool = False) -> dict[str, float]:
        self.ensure_alive()
        self.command_queue.put(("metrics", generation, include_totals))
        return self._wait_for_status("metrics", generation)

    def close(self) -> None:
        if self.process.is_alive():
            try:
                self.command_queue.put(("stop",), timeout=1.0)
            except (queue.Full, ValueError, OSError):
                pass
            self.process.join(timeout=10.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=10.0)
        self.alive_event.clear()
        self.request_queue.close()
        self.command_queue.close()
        self.status_queue.close()
        for response_queue in self.response_queues:
            response_queue.close()
        if self.shared_transport is not None:
            self.shared_transport.close()


def _bench_worker_main(
    endpoints: InferenceServerEndpoints,
    worker_id: int,
    start_event: Any,
    stop_event: Any,
    ready_queue: Any,
    result_queue: Any,
) -> None:
    """Synthetic max-rate requester used by ``--bench-server``."""
    shared_views: WorkerSharedMemoryViews | None = None
    try:
        if endpoints.transport == "shm":
            assert endpoints.shared_memory_specs is not None
            shared_views = WorkerSharedMemoryViews(endpoints.shared_memory_specs[worker_id])
        count = endpoints.request_batch_size
        obs = np.full((count, endpoints.obs_size), float(worker_id), dtype=np.float32)
        masks = np.ones((count, dz.ACTION_SPACE_SIZE), dtype=np.uint8)
        response_queue = endpoints.response_queues[worker_id]
        sequence = 0
        batches = 0
        evals = 0
        ready_queue.put(worker_id)
        start_event.wait()
        while not stop_event.is_set():
            sequence += 1
            deadline = time.monotonic() + endpoints.response_timeout_s
            if shared_views is not None:
                slot = sequence % shared_views.spec.slots
                np.copyto(shared_views.request_obs[slot, :count], obs)
                np.copyto(shared_views.request_masks[slot, :count], masks)
                if endpoints.poll == "spin":
                    shared_views.request_counts[slot] = count
                    shared_views.request_sequences[slot] = sequence
                    request: tuple[Any, ...] = ()
                else:
                    request = (worker_id, slot, count, sequence)
            else:
                request = (worker_id, sequence, obs, masks)
            if shared_views is None or endpoints.poll != "spin":
                while True:
                    if not endpoints.alive_event.is_set():
                        raise RuntimeError("inference server died during benchmark request submission")
                    try:
                        endpoints.request_queue.put(request, timeout=0.1)
                        break
                    except queue.Full:
                        if time.monotonic() >= deadline:
                            raise RuntimeError("benchmark request submission timed out")
            while True:
                if time.monotonic() >= deadline:
                    raise RuntimeError("benchmark response wait timed out")
                if shared_views is not None and endpoints.poll == "spin":
                    if int(shared_views.response_sequences[slot]) == sequence:
                        break
                    if not endpoints.alive_event.is_set():
                        raise RuntimeError("inference server died during benchmark response wait")
                    time.sleep(0)
                    continue
                try:
                    response = response_queue.get(timeout=0.1)
                except queue.Empty:
                    if not endpoints.alive_event.is_set():
                        raise RuntimeError("inference server died during benchmark response wait")
                    continue
                if shared_views is not None:
                    slot, response_count, response_sequence = response
                    if slot != sequence % shared_views.spec.slots or response_count != count or response_sequence != sequence:
                        raise RuntimeError("benchmark shared-memory response routing mismatch")
                else:
                    kind, response_sequence, _, _ = response
                    if kind != "response" or response_sequence != sequence:
                        raise RuntimeError("benchmark queue response routing mismatch")
                break
            batches += 1
            evals += count
        result_queue.put((worker_id, batches, evals, None))
    except BaseException as exc:
        result_queue.put((worker_id, 0, 0, repr(exc)))
    finally:
        if shared_views is not None:
            shared_views.close()


def _header_echo_main(request_queue: Any, response_queues: list[Any], stop_event: Any) -> None:
    while True:
        try:
            worker_id, sequence = request_queue.get(timeout=0.01)
        except queue.Empty:
            if stop_event.is_set():
                return
            continue
        response_queues[worker_id].put(sequence)


def _header_echo_worker(
    worker_id: int,
    request_queue: Any,
    response_queue: Any,
    start_event: Any,
    stop_event: Any,
    ready_queue: Any,
    result_queue: Any,
) -> None:
    ready_queue.put(worker_id)
    start_event.wait()
    sequence = 0
    elapsed_ns = 0
    rounds = 0
    while not stop_event.is_set():
        sequence += 1
        started = time.perf_counter_ns()
        request_queue.put((worker_id, sequence))
        if response_queue.get(timeout=5.0) != sequence:
            result_queue.put((worker_id, 0, "header response routing mismatch"))
            return
        elapsed_ns += time.perf_counter_ns() - started
        rounds += 1
    result_queue.put((worker_id, (elapsed_ns, rounds), None))


def _bench_queue_header_round_trip(context: Any, worker_count: int, duration_s: float) -> float:
    """Measure global-queue header ping-pong without model or array payloads."""
    request_queue = context.Queue()
    response_queues = [context.Queue() for _ in range(worker_count)]
    start_event = context.Event()
    stop_event = context.Event()
    ready_queue = context.Queue()
    result_queue = context.Queue()
    echo = context.Process(target=_header_echo_main, args=(request_queue, response_queues, stop_event))
    workers = [
        context.Process(
            target=_header_echo_worker,
            args=(worker_id, request_queue, response_queues[worker_id], start_event, stop_event, ready_queue, result_queue),
        )
        for worker_id in range(worker_count)
    ]
    try:
        echo.start()
        for worker in workers:
            worker.start()
        for _ in workers:
            ready_queue.get(timeout=10.0)
        start_event.set()
        time.sleep(duration_s)
        stop_event.set()
        results = [result_queue.get(timeout=10.0) for _ in workers]
        errors = [result[2] for result in results if result[2] is not None]
        if errors:
            raise RuntimeError(errors[0])
        elapsed_ns = sum(result[1][0] for result in results)
        rounds = sum(result[1][1] for result in results)
        return elapsed_ns / rounds / 1000.0 if rounds else 0.0
    finally:
        stop_event.set()
        for process in [*workers, echo]:
            if process.is_alive():
                process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        request_queue.close()
        ready_queue.close()
        result_queue.close()
        for response_queue in response_queues:
            response_queue.close()


def bench_server(
    *,
    device: str,
    worker_counts: list[int],
    duration_s: float,
    max_batch: int,
    transport: str,
    poll: str,
) -> list[dict[str, float | int | str]]:
    """Run standalone synthetic transport/server measurements for each worker count."""
    if duration_s <= 0.0 or max_batch <= 0:
        raise ValueError("benchmark duration and max batch must be positive")
    context = mp.get_context("spawn")
    rows: list[dict[str, float | int | str]] = []
    for worker_count in worker_counts:
        cfg = TrainConfig()
        cfg.seed = 123
        cfg.server_device = device
        cfg.server_transport = transport
        cfg.server_poll = poll
        cfg.server_max_batch = max_batch
        cfg.server_max_wait_ms = 2.0
        cfg.server_response_timeout_s = 10.0
        cfg.selfplay.max_batch = max_batch
        cfg.model.hidden_sizes = [32]
        server = InferenceServer(cfg, worker_count)
        start_event = context.Event()
        stop_event = context.Event()
        ready_queue = context.Queue()
        result_queue = context.Queue()
        workers = [
            context.Process(
                target=_bench_worker_main,
                args=(server.endpoints, worker_id, start_event, stop_event, ready_queue, result_queue),
                name=f"inference-bench-{worker_id}",
            )
            for worker_id in range(worker_count)
        ]
        try:
            for worker in workers:
                worker.start()
            for _ in workers:
                ready_queue.get(timeout=15.0)
            start = time.perf_counter()
            start_event.set()
            time.sleep(duration_s)
            stop_event.set()
            results = []
            for _ in workers:
                results.append(result_queue.get(timeout=15.0))
            wall_time = time.perf_counter() - start
            for worker in workers:
                worker.join(timeout=5.0)
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5.0)
            errors = [result[3] for result in results if result[3] is not None]
            if errors:
                raise RuntimeError(f"benchmark synthetic worker failure: {errors[0]}")
            metrics = server.collect_metrics(generation=0, include_totals=True)
            header_round_trip_us = _bench_queue_header_round_trip(context, worker_count, min(1.0, duration_s))
            total_evals = int(metrics["_server_total_evals"])
            total_batches = int(metrics["_server_total_batches"])
            rows.append(
                {
                    "device": device,
                    "transport": server.transport,
                    "poll": server.poll,
                    "workers": worker_count,
                    "wall_time_s": wall_time,
                    "batches_per_sec": total_batches / wall_time if wall_time > 0.0 else 0.0,
                    "evals_per_sec": total_evals / wall_time if wall_time > 0.0 else 0.0,
                    "mean_batch_size": float(metrics["server_mean_batch_size"]),
                    "queue_header_round_trip_us": header_round_trip_us,
                }
            )
        finally:
            stop_event.set()
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5.0)
            ready_queue.close()
            result_queue.close()
            server.close()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-server", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--workers", default="8,16,24")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--max-batch", type=int, default=1024)
    parser.add_argument("--transport", choices=["shm", "queue"], default="shm")
    parser.add_argument("--poll", choices=["queue", "spin"], default="queue")
    args = parser.parse_args(argv)
    if not args.bench_server:
        parser.error("--bench-server is required when invoking this module directly")
    worker_counts = [int(value) for value in args.workers.split(",") if value]
    for row in bench_server(
        device=args.device,
        worker_counts=worker_counts,
        duration_s=args.duration,
        max_batch=args.max_batch,
        transport=args.transport,
        poll=args.poll,
    ):
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
