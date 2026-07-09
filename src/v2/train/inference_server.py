"""One-process batched inference service with shared-memory worker rings.

In ``shm`` mode, queues carry only small request/response headers.  Leaf
arrays live in a two-slot request/response ring owned by each worker, avoiding
the large pickle copies that otherwise dominate small-batch self-play.
"""

from __future__ import annotations

import atexit
import copy
import io
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
        request_obs_bytes = spec.slots * spec.max_request * spec.obs_size * np.dtype(np.float32).itemsize
        response_values_bytes = spec.slots * spec.max_request * np.dtype(np.float32).itemsize
        self.request_obs = np.ndarray(
            (spec.slots, spec.max_request, spec.obs_size),
            dtype=np.float32,
            buffer=self.request_block.buf,
        )
        self.request_masks = np.ndarray(
            (spec.slots, spec.max_request, spec.action_size),
            dtype=np.uint8,
            buffer=self.request_block.buf,
            offset=request_obs_bytes,
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
            offset=response_values_bytes,
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
    def create(cls, worker_count: int, slots: int, max_request: int) -> SharedMemoryTransport:
        if shared_memory is None:
            raise RuntimeError("multiprocessing.shared_memory is unavailable")
        if worker_count <= 0 or slots < 2 or max_request <= 0:
            raise ValueError("invalid shared-memory ring dimensions")
        # macOS limits POSIX shared-memory names far more tightly than Linux.
        # A 12-hex token remains unique for a training invocation while
        # leaving room for the request/response and worker suffixes.
        token = f"dz{uuid.uuid4().hex[:12]}"
        request_bytes = slots * max_request * (
            dz.OBS_SIZE * np.dtype(np.float32).itemsize + dz.ACTION_SPACE_SIZE * np.dtype(np.uint8).itemsize
        )
        response_bytes = slots * max_request * (
            np.dtype(np.float32).itemsize + dz.ACTION_SPACE_SIZE * np.dtype(np.float32).itemsize
        )
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
                        obs_size=dz.OBS_SIZE,
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
    transport: str
    shared_memory_specs: list[WorkerSharedMemorySpec] | None


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

    def snapshot(self) -> dict[str, float]:
        waits_ms = np.asarray(self.batch_waits_s, dtype=np.float64)
        return {
            "server_evals_per_sec": self.evals / self.inference_time if self.inference_time > 0.0 else 0.0,
            "server_mean_batch_size": self.evals / self.batches if self.batches > 0 else 0.0,
            "server_batch_wait_p50_ms": float(np.percentile(waits_ms, 50.0)) if waits_ms.size else 0.0,
            "server_batch_wait_p99_ms": float(np.percentile(waits_ms, 99.0)) if waits_ms.size else 0.0,
        }


class _PinnedStaging:
    """Persistent host staging for a complete server batch and its responses."""

    def __init__(self, device: torch.device, max_batch: int):
        pinned = device.type == "cuda"
        self.device = device
        self.obs = torch.empty((max_batch, dz.OBS_SIZE), dtype=torch.float32, pin_memory=pinned)
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
        model = DominionNet(dz.OBS_SIZE, dz.ACTION_SPACE_SIZE, config.model.hidden_sizes).to(device)
        model.eval()
        staging = _PinnedStaging(device, int(config.server_max_batch))
        if endpoints.transport == "shm":
            assert endpoints.shared_memory_specs is not None
            views = [WorkerSharedMemoryViews(spec) for spec in endpoints.shared_memory_specs]
        endpoints.alive_event.set()

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
                    _, generation = command
                    status_queue.put(("metrics", generation, metrics.snapshot()))
                    metrics = _GenerationMetrics()
                    continue
                raise RuntimeError(f"unknown inference-server command: {kind}")

        def dequeue_request(timeout: float | None = None) -> _Request:
            if endpoints.transport == "shm":
                if timeout is None:
                    worker_id, slot, count, sequence = endpoints.request_queue.get()
                else:
                    worker_id, slot, count, sequence = endpoints.request_queue.get(timeout=timeout)
                worker_id, slot, count, sequence = int(worker_id), int(slot), int(count), int(sequence)
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
            if timeout is None:
                worker_id, request_id, obs, masks = endpoints.request_queue.get()
            else:
                worker_id, request_id, obs, masks = endpoints.request_queue.get(timeout=timeout)
            obs = np.ascontiguousarray(obs, dtype=np.float32)
            masks = np.ascontiguousarray(masks, dtype=np.uint8)
            if obs.ndim != 2 or masks.shape != (obs.shape[0], dz.ACTION_SPACE_SIZE):
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
        context = mp.get_context("spawn")
        self.request_queue = context.Queue(maxsize=max(4, worker_count * 4))
        self.response_queues = [context.Queue(maxsize=2) for _ in range(worker_count)]
        self.command_queue = context.Queue()
        self.status_queue = context.Queue()
        self.alive_event = context.Event()
        request_batch_size = max(1, min(config.selfplay.max_batch, config.server_max_batch // worker_count))
        self.shared_transport: SharedMemoryTransport | None = None
        self.transport = requested_transport
        if requested_transport == "shm":
            try:
                self.shared_transport = SharedMemoryTransport.create(
                    worker_count,
                    int(config.server_shm_slots),
                    request_batch_size,
                )
            except Exception as exc:
                self.transport = "queue"
                warnings.warn(
                    f"shared-memory inference transport unavailable ({exc}); falling back to queue transport",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self.endpoints = InferenceServerEndpoints(
            request_queue=self.request_queue,
            response_queues=self.response_queues,
            alive_event=self.alive_event,
            response_timeout_s=float(config.server_response_timeout_s),
            request_batch_size=request_batch_size,
            transport=self.transport,
            shared_memory_specs=self.shared_transport.specs if self.shared_transport is not None else None,
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

    def collect_metrics(self, generation: int) -> dict[str, float]:
        self.ensure_alive()
        self.command_queue.put(("metrics", generation))
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
