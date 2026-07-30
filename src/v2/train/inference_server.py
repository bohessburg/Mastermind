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
import logging
import math
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
import uuid
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

try:
    from multiprocessing import shared_memory
except ImportError:  # pragma: no cover - supported CPython versions provide it
    shared_memory = None  # type: ignore[assignment]

import dominion_v2_py as dz

from .config import TrainConfig
from .model import build_model, model_config_dict
from .observation import observations_for_model, obs_size_for_config, obs_size_for_version


logger = logging.getLogger(__name__)


def _align(offset: int, alignment: int = 8) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _request_layout(spec: WorkerSharedMemorySpec) -> tuple[int, int, int, int, int, int]:
    obs_bytes = spec.slots * spec.max_request * spec.obs_size * np.dtype(np.float32).itemsize
    masks_offset = obs_bytes
    masks_bytes = spec.slots * spec.max_request * spec.action_size * np.dtype(np.uint8).itemsize
    counts_offset = _align(masks_offset + masks_bytes, np.dtype(np.uint64).itemsize)
    model_ids_offset = counts_offset + spec.slots * np.dtype(np.uint32).itemsize
    submitted_ns_offset = _align(
        model_ids_offset + spec.slots * np.dtype(np.uint32).itemsize,
        np.dtype(np.uint64).itemsize,
    )
    sequences_offset = _align(
        submitted_ns_offset + spec.slots * np.dtype(np.uint64).itemsize,
        np.dtype(np.uint64).itemsize,
    )
    return (
        masks_offset,
        counts_offset,
        model_ids_offset,
        submitted_ns_offset,
        sequences_offset,
        sequences_offset + spec.slots * np.dtype(np.uint64).itemsize,
    )


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
        (
            request_masks_offset,
            request_counts_offset,
            request_model_ids_offset,
            request_submitted_ns_offset,
            request_sequences_offset,
            _,
        ) = _request_layout(spec)
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
        self.request_model_ids = np.ndarray(
            (spec.slots,),
            dtype=np.uint32,
            buffer=self.request_block.buf,
            offset=request_model_ids_offset,
        )
        self.request_submitted_ns = np.ndarray(
            (spec.slots,),
            dtype=np.uint64,
            buffer=self.request_block.buf,
            offset=request_submitted_ns_offset,
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
        obs_size: int,
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
        request_bytes = _request_layout(layout_spec)[5]
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
    model_id: int
    submitted_at_s: float
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


@dataclass(frozen=True)
class _ReadyModel:
    model_id: int
    trigger: str


class _PerModelRequestQueues:
    """Persistent per-model accumulators with independent fire deadlines."""

    def __init__(
        self,
        *,
        worker_count: int,
        target_rows: int,
        max_batch: int,
        coalesce_s: float,
    ) -> None:
        if worker_count <= 0:
            raise ValueError("coalescing worker_count must be positive")
        if target_rows <= 0 or max_batch <= 0:
            raise ValueError("coalescing row limits must be positive")
        if not math.isfinite(coalesce_s) or coalesce_s < 0.0:
            raise ValueError("coalescing deadline must be finite and non-negative")
        self.worker_count = int(worker_count)
        self.target_rows = min(int(target_rows), int(max_batch))
        self.max_batch = int(max_batch)
        self.coalesce_s = float(coalesce_s)
        self._requests: dict[int, deque[_Request]] = {}
        self._rows: dict[int, int] = {}
        self._worker_counts: dict[int, int] = {}

    def add(self, request: _Request) -> None:
        if request.count <= 0 or request.count > self.max_batch:
            raise ValueError("inference request batch is outside server_max_batch")
        self._requests.setdefault(request.model_id, deque()).append(request)
        self._rows[request.model_id] = self._rows.get(request.model_id, 0) + request.count
        self._worker_counts[request.worker_id] = self._worker_counts.get(request.worker_id, 0) + 1

    def rows_for(self, model_id: int) -> int:
        return self._rows.get(int(model_id), 0)

    def request_count_for(self, model_id: int) -> int:
        return len(self._requests.get(int(model_id), ()))

    def has_requests(self) -> bool:
        return bool(self._requests)

    def _deadline(self, model_id: int) -> float:
        requests = self._requests[model_id]
        return min(request.submitted_at_s for request in requests) + self.coalesce_s

    def ready_model(self, now: float) -> _ReadyModel | None:
        """Select one fireable model without starving expired model queues."""

        now = float(now)
        all_workers_pending = len(self._worker_counts) >= self.worker_count
        expired: list[_ReadyModel] = []
        candidates: list[_ReadyModel] = []
        for model_id in self._requests:
            deadline = self._deadline(model_id)
            if now >= deadline:
                expired.append(_ReadyModel(model_id, "deadline"))
                continue
            elif self._rows[model_id] >= self.target_rows:
                trigger = "fill_target"
            elif all_workers_pending:
                trigger = "all_workers_pending"
            else:
                continue
            candidates.append(_ReadyModel(model_id, trigger))
        if expired:
            return min(
                expired,
                key=lambda candidate: (
                    self._deadline(candidate.model_id),
                    candidate.model_id,
                ),
            )
        if not candidates:
            return None
        for candidate in candidates:
            if candidate.model_id == 0:
                return candidate
        return min(
            candidates,
            key=lambda candidate: (
                self._deadline(candidate.model_id),
                candidate.model_id,
            ),
        )

    def seconds_until_deadline(self, now: float) -> float | None:
        if not self._requests:
            return None
        return max(0.0, min(self._deadline(model_id) for model_id in self._requests) - float(now))

    def take_batch(self, model_id: int) -> list[_Request]:
        """Remove one request-aligned batch without splitting worker requests."""

        model_id = int(model_id)
        requests = self._requests.get(model_id)
        if not requests:
            raise ValueError("cannot take a batch for a model with no pending requests")
        batch: list[_Request] = []
        rows = 0
        while requests:
            request = requests[0]
            if batch and rows + request.count > self.max_batch:
                break
            requests.popleft()
            batch.append(request)
            rows += request.count
            remaining_for_worker = self._worker_counts[request.worker_id] - 1
            if remaining_for_worker:
                self._worker_counts[request.worker_id] = remaining_for_worker
            else:
                del self._worker_counts[request.worker_id]
        self._rows[model_id] -= rows
        if not requests:
            del self._requests[model_id]
            del self._rows[model_id]
        return batch


def _drain_during_flight(
    in_flight: Any,
    dequeue: Callable[[float], _Request],
    accept: Callable[[_Request], None],
    *,
    poll_interval_s: float = 0.0005,
) -> float:
    """Drain request headers while an aggregate device result is pending."""

    while not in_flight.ready():
        try:
            accept(dequeue(poll_interval_s))
        except queue.Empty:
            continue
    return float(in_flight.finish())


def _normalized_server_batch_buckets(buckets: list[int] | None) -> tuple[int, ...]:
    """Validate and normalize configured static server forward sizes."""

    if buckets is None:
        return ()
    if not isinstance(buckets, list):
        raise ValueError("server_batch_buckets must be a list of positive integers or null")
    if any(
        not isinstance(bucket, int) or isinstance(bucket, bool) or bucket <= 0
        for bucket in buckets
    ):
        raise ValueError("server_batch_buckets must contain only positive integers")
    return tuple(sorted(set(buckets)))


def _bucketed_batch_size(count: int, buckets: tuple[int, ...] | list[int] | None) -> int:
    """Return the smallest configured static shape that can hold ``count``."""

    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("batch count must be a positive integer")
    if not buckets:
        return count
    return min((bucket for bucket in buckets if bucket >= count), default=count)


class _ServerEvaluator(torch.nn.Module):
    """Make each architecture's existing ``evaluate`` path compilable as one call."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, obs: torch.Tensor, legal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.evaluate(obs, legal_mask)


def compile_server_evaluator(model: torch.nn.Module) -> torch.nn.Module:
    """Compile the exact server evaluation path without changing weight ownership.

    ``fullgraph=True`` intentionally rejects graph breaks during startup warmup
    rather than silently partitioning a serving forward into a slower eager
    fallback.  Static bucket inputs then let ``reduce-overhead`` capture each
    compiled CUDA graph outside the request path.
    """

    if not hasattr(torch, "compile"):
        raise RuntimeError("server_compile requires a PyTorch build with torch.compile")
    return torch.compile(
        _ServerEvaluator(model).eval(),
        mode="reduce-overhead",
        fullgraph=True,
    )


def _maybe_compile_server_evaluator(
    model: torch.nn.Module,
    device: torch.device,
    enabled: bool,
) -> torch.nn.Module | None:
    if not enabled:
        return None
    if device.type != "cuda":
        logger.warning(
            "server_compile=true is ignored on %s; compilation is enabled only for CUDA servers",
            device.type,
        )
        return None
    return compile_server_evaluator(model)


@dataclass
class _InFlightForward:
    """One aggregate result whose single device-to-host copy may still run."""

    started_at_s: float
    completion_event: Any | None
    retained_tensors: tuple[Any, ...] = ()
    _elapsed_s: float | None = None

    def ready(self) -> bool:
        return self.completion_event is None or bool(self.completion_event.query())

    def finish(self) -> float:
        if self._elapsed_s is None:
            if self.completion_event is not None:
                self.completion_event.synchronize()
            self._elapsed_s = time.perf_counter() - self.started_at_s
            self.retained_tensors = ()
        return self._elapsed_s


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

    def load_requests(
        self,
        requests: list[_Request],
        source_obs_version: int,
        model_obs_version: int,
        encoder_generation: int,
    ) -> int:
        offset = 0
        for request in requests:
            end = offset + request.count
            adapted_obs = observations_for_model(
                request.obs,
                source_obs_version,
                model_obs_version,
                encoder_generation,
            )
            self.obs_np[offset:end] = adapted_obs
            self.masks_np[offset:end] = request.masks
            offset = end
        return offset

    def zero_inputs(self, count: int) -> None:
        """Prepare an all-zero static input shape for warmup or bucket padding."""

        self.obs[:count].zero_()
        self.masks[:count].zero_()

    def launch(
        self,
        model: torch.nn.Module,
        count: int,
        use_fp16: bool,
        *,
        evaluator: torch.nn.Module | None = None,
        use_bf16: bool = False,
        batch_buckets: tuple[int, ...] | list[int] | None = None,
    ) -> _InFlightForward:
        forward_count = _bucketed_batch_size(count, batch_buckets)
        if forward_count > self.obs.shape[0]:
            raise ValueError("bucketed server batch exceeds staging capacity")
        if forward_count > count:
            # The zero legal mask is immaterial to real rows, and keeps the
            # padded rows harmless for both architecture evaluation paths.
            self.zero_inputs_slice(count, forward_count)
        started_at_s = time.perf_counter()
        if self.device.type == "cuda":
            with torch.inference_mode():
                obs = self.obs[:forward_count].to(self.device, non_blocking=True)
                masks = self.masks[:forward_count].to(self.device, non_blocking=True)
                if use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        if evaluator is None:
                            policies, values = model.evaluate(obs, masks)
                        else:
                            policies, values = evaluator(obs, masks)
                elif evaluator is None:
                    # Keep the legacy CUDA eager path structurally unchanged
                    # when every optional precision/compiler knob is disabled.
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                        policies, values = model.evaluate(obs, masks)
                else:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                        policies, values = evaluator(obs, masks)
                real_values = values if forward_count == count else values[:count]
                real_policies = policies if forward_count == count else policies[:count]
                values_float = real_values.float()
                policies_float = real_policies.float()
                # Exactly one aggregate D2H copy per output tensor. The CUDA
                # event is recorded after both copies, so host scatter cannot
                # observe partially transferred response buffers.
                self.values[:count].copy_(values_float, non_blocking=True)
                self.policies[:count].copy_(policies_float, non_blocking=True)
                completion = torch.cuda.Event(enable_timing=False)
                completion.record(torch.cuda.current_stream(self.device))
            return _InFlightForward(
                started_at_s,
                completion,
                (obs, masks, policies, values, values_float, policies_float),
            )
        else:
            with torch.inference_mode():
                if evaluator is None:
                    policies, values = model.evaluate(self.obs[:forward_count], self.masks[:forward_count])
                else:
                    policies, values = evaluator(self.obs[:forward_count], self.masks[:forward_count])
                if forward_count == count:
                    self.values[:count].copy_(values.float())
                    self.policies[:count].copy_(policies.float())
                else:
                    self.values[:count].copy_(values[:count].float())
                    self.policies[:count].copy_(policies[:count].float())
            return _InFlightForward(started_at_s, None)

    def forward(
        self,
        model: torch.nn.Module,
        count: int,
        use_fp16: bool,
        *,
        evaluator: torch.nn.Module | None = None,
        use_bf16: bool = False,
        batch_buckets: tuple[int, ...] | list[int] | None = None,
    ) -> float:
        """Synchronous compatibility wrapper used only during warmup."""

        return self.launch(
            model,
            count,
            use_fp16,
            evaluator=evaluator,
            use_bf16=use_bf16,
            batch_buckets=batch_buckets,
        ).finish()

    def zero_inputs_slice(self, start: int, end: int) -> None:
        self.obs[start:end].zero_()
        self.masks[start:end].zero_()


def _warm_server_evaluator(
    staging: _PinnedStaging,
    model: torch.nn.Module,
    evaluator: torch.nn.Module | None,
    use_fp16: bool,
    use_bf16: bool,
    batch_buckets: tuple[int, ...],
    server_max_batch: int,
) -> None:
    """Compile/capture static serving shapes before the first worker request."""

    if evaluator is None and not batch_buckets:
        return
    warmup_sizes = batch_buckets or (int(server_max_batch),)
    for bucket in sorted(warmup_sizes, reverse=True):
        staging.zero_inputs(bucket)
        with torch.no_grad():
            elapsed = staging.forward(
                model,
                bucket,
                use_fp16,
                evaluator=evaluator,
                use_bf16=use_bf16,
                batch_buckets=batch_buckets,
            )
        logger.info("inference server warmup bucket=%d elapsed_ms=%.3f", bucket, elapsed * 1000.0)


@dataclass
class _ResidentModel:
    """One server-owned model, compiled evaluator, and shape-specific staging."""

    config: dict[str, Any]
    obs_version: int
    encoder_generation: int
    model: torch.nn.Module
    evaluator: torch.nn.Module | None
    staging: _PinnedStaging


def _payload_parts(
    payload: bytes | tuple[bytes, dict[str, Any]],
    fallback_model_config: object,
) -> tuple[bytes, dict[str, Any]]:
    if isinstance(payload, bytes):
        return payload, model_config_dict(fallback_model_config)
    if (
        isinstance(payload, tuple)
        and len(payload) == 2
        and isinstance(payload[0], bytes)
        and isinstance(payload[1], dict)
    ):
        return payload[0], dict(payload[1])
    raise TypeError("server model payload must be bytes or (bytes, model_config)")


def _build_resident_model(
    model_config: dict[str, Any],
    device: torch.device,
    config: TrainConfig,
    batch_buckets: tuple[int, ...],
    use_bf16: bool,
) -> _ResidentModel:
    configured_version = model_config.get("obs_version")
    obs_version = int(config.selfplay.obs_version if configured_version is None else configured_version)
    if obs_version not in (1, 2, 3):
        raise ValueError(f"server model has invalid obs_version {obs_version!r}")
    encoder_generation = int(model_config.get("encoder_generation", 2))
    if encoder_generation < 1:
        raise ValueError(f"server model has invalid encoder_generation {encoder_generation!r}")
    model = build_model(
        model_config,
        obs_size_for_version(obs_version),
        dz.ACTION_SPACE_SIZE,
    ).to(device)
    model.eval()
    normalized_config = model_config_dict(getattr(model, "_dominion_model_config", model_config))
    normalized_config["encoder_generation"] = encoder_generation
    staging = _PinnedStaging(
        device,
        max(int(config.server_max_batch), max(batch_buckets, default=0)),
        obs_size_for_version(obs_version),
    )
    evaluator = _maybe_compile_server_evaluator(model, device, bool(config.server_compile))
    _warm_server_evaluator(
        staging,
        model,
        evaluator,
        bool(config.server_fp16),
        use_bf16,
        batch_buckets,
        int(config.server_max_batch),
    )
    return _ResidentModel(normalized_config, obs_version, encoder_generation, model, evaluator, staging)


def _server_device(name: str) -> torch.device:
    requested = name.lower()
    if requested not in {"cpu", "cuda"}:
        raise ValueError("server_device must be 'cpu' or 'cuda'")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("server_device=cuda requested but unavailable")
    return torch.device(requested)


def _scatter_responses(
    requests: list[_Request],
    values: np.ndarray,
    policies: np.ndarray,
    endpoints: InferenceServerEndpoints,
    views: list[WorkerSharedMemoryViews],
) -> None:
    """Scatter contiguous aggregate result slices with one memcpy per ring."""

    expected_rows = sum(request.count for request in requests)
    if values.shape[0] < expected_rows or policies.shape[0] < expected_rows:
        raise ValueError("aggregate inference result is smaller than its request batch")
    offset = 0
    for request in requests:
        end = offset + request.count
        if endpoints.transport == "shm":
            assert request.slot is not None
            view = views[request.worker_id]
            np.copyto(view.response_values[request.slot, : request.count], values[offset:end])
            np.copyto(view.response_policies[request.slot, : request.count], policies[offset:end])
            if endpoints.poll == "spin":
                view.response_sequences[request.slot] = request.request_id
            else:
                endpoints.response_queues[request.worker_id].put(
                    (request.slot, request.count, request.request_id)
                )
        else:
            endpoints.response_queues[request.worker_id].put(
                (
                    "response",
                    request.request_id,
                    values[offset:end].copy(),
                    policies[offset:end].copy(),
                )
            )
        offset = end


def _server_main(
    config: TrainConfig,
    endpoints: InferenceServerEndpoints,
    command_queue: Any,
    status_queue: Any,
    telemetry_queue: Any,
) -> None:
    """Process entry point; all model updates occur between complete batches."""
    parent_pid = os.getppid()
    running = True
    metrics = _GenerationMetrics()
    heartbeat_started = time.monotonic()
    heartbeat_requests = 0
    heartbeat_evals = 0
    heartbeat_batches = 0
    heartbeat_wait_start = 0
    views: list[WorkerSharedMemoryViews] = []
    try:
        device = _server_device(config.server_device)
        source_obs_version = int(config.selfplay.obs_version)
        batch_buckets = _normalized_server_batch_buckets(config.server_batch_buckets)
        use_bf16 = bool(config.server_autocast_bf16)
        if use_bf16 and device.type != "cuda":
            logger.warning("server_autocast_bf16=true is ignored on %s; CUDA is required", device.type)
            use_bf16 = False
        if use_bf16 and config.server_fp16:
            logger.warning("server_autocast_bf16=true takes precedence over server_fp16=true")
        # With compile enabled, a merged batch above the largest warmed bucket
        # would run at its exact size and trigger a mid-serve torch.compile
        # stall (30s+), starving workers into response timeouts. Cap merges at
        # the largest bucket so every forward hits a pre-warmed shape.
        merge_cap = int(config.server_max_batch)
        if bool(config.server_compile) and batch_buckets:
            merge_cap = min(merge_cap, max(batch_buckets))
        pending = _PerModelRequestQueues(
            worker_count=len(endpoints.response_queues),
            target_rows=int(config.server_coalesce_target_rows),
            max_batch=merge_cap,
            coalesce_s=float(config.server_coalesce_ms) / 1000.0,
        )
        try:
            resident_models = [
                _build_resident_model(
                    model_config_dict(config.model),
                    device,
                    config,
                    batch_buckets,
                    use_bf16,
                )
            ]
        except Exception as exc:
            if bool(config.server_compile) and device.type == "cuda":
                raise RuntimeError(
                    "server_compile warmup failed; refusing to fall back to eager execution. "
                    "Check the full-graph compiler error above."
                ) from exc
            raise
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

        def emit_heartbeat_if_due() -> None:
            """Report server work without adding a request-path round trip."""

            nonlocal heartbeat_started, heartbeat_requests, heartbeat_evals, heartbeat_batches, heartbeat_wait_start
            now = time.monotonic()
            if heartbeat_requests == 0 or now - heartbeat_started < 60.0:
                return
            elapsed = now - heartbeat_started
            assert metrics.batch_waits_s is not None
            waits_ms = np.asarray(metrics.batch_waits_s[heartbeat_wait_start:], dtype=np.float64) * 1000.0
            summary = {
                "requests_per_sec": float(heartbeat_requests / elapsed),
                "evals_per_sec": float(heartbeat_evals / elapsed),
                "mean_batch_size": float(heartbeat_evals / heartbeat_batches) if heartbeat_batches else 0.0,
                "wait_p50_ms": float(np.percentile(waits_ms, 50.0)) if waits_ms.size else 0.0,
                "wait_p99_ms": float(np.percentile(waits_ms, 99.0)) if waits_ms.size else 0.0,
            }
            print(
                "inference server: "
                f"{summary['requests_per_sec']:.1f} requests/s, "
                f"mean batch {summary['mean_batch_size']:.1f}, "
                f"wait p50/p99 {summary['wait_p50_ms']:.2f}/{summary['wait_p99_ms']:.2f} ms",
                file=sys.stderr,
                flush=True,
            )
            try:
                telemetry_queue.put_nowait(summary)
            except queue.Full:
                # Telemetry is deliberately best-effort: trainer progress
                # never gets to delay serving work.
                pass
            heartbeat_started = now
            heartbeat_requests = 0
            heartbeat_evals = 0
            heartbeat_batches = 0
            heartbeat_wait_start = len(metrics.batch_waits_s)

        def install_models(
            payloads: list[bytes | tuple[bytes, dict[str, Any]]],
        ) -> None:
            """Install one copy per table entry, reusing compiled shapes."""

            nonlocal resident_models
            if not payloads:
                raise ValueError("inference server requires at least one resident model")
            installed: list[_ResidentModel] = []
            for model_id, payload in enumerate(payloads):
                state_payload, requested_config = _payload_parts(payload, config.model)
                configured_version = requested_config.get("obs_version")
                requested_version = int(
                    source_obs_version if configured_version is None else configured_version
                )
                requested_config = dict(requested_config)
                requested_config["obs_version"] = requested_version
                reusable = resident_models[model_id] if model_id < len(resident_models) else None
                if reusable is not None and reusable.config == requested_config:
                    resident = reusable
                else:
                    try:
                        resident = _build_resident_model(
                            requested_config,
                            device,
                            config,
                            batch_buckets,
                            use_bf16,
                        )
                    except Exception as exc:
                        if bool(config.server_compile) and device.type == "cuda":
                            raise RuntimeError(
                                f"server_compile warmup failed for model {model_id}; "
                                "refusing to fall back to eager execution"
                            ) from exc
                        raise
                resident.model.load_state_dict(deserialize_cpu_state_dict(state_payload))
                resident.model.eval()
                installed.append(resident)
            resident_models = installed

        def handle_commands() -> None:
            nonlocal running, metrics, heartbeat_started, heartbeat_requests, heartbeat_evals, heartbeat_batches, heartbeat_wait_start
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
                    resident_models[0].model.load_state_dict(deserialize_cpu_state_dict(payload))
                    resident_models[0].model.eval()
                    status_queue.put(("weights", generation, None))
                    continue
                if kind == "models":
                    _, generation, payloads = command
                    install_models(list(payloads))
                    status_queue.put(("models", generation, len(resident_models)))
                    continue
                if kind == "metrics":
                    _, generation, include_totals = command
                    status_queue.put(("metrics", generation, metrics.snapshot(bool(include_totals))))
                    metrics = _GenerationMetrics()
                    heartbeat_started = time.monotonic()
                    heartbeat_requests = 0
                    heartbeat_evals = 0
                    heartbeat_batches = 0
                    heartbeat_wait_start = 0
                    continue
                raise RuntimeError(f"unknown inference-server command: {kind}")

        def shared_request_from_header(
            worker_id: int,
            slot: int,
            count: int,
            sequence: int,
            model_id: int,
            submitted_ns: int,
        ) -> _Request:
            if not (0 <= worker_id < len(views)):
                raise ValueError("shared-memory request has invalid worker id")
            view = views[worker_id]
            if not (0 <= slot < view.spec.slots and 0 < count <= view.spec.max_request):
                raise ValueError("shared-memory request has invalid slot or count")
            if slot != sequence % view.spec.slots:
                raise ValueError("shared-memory request sequence does not match its slot")
            if not 0 <= model_id < len(resident_models):
                raise ValueError(f"inference request references unknown model id {model_id}")
            if submitted_ns <= 0:
                raise ValueError("shared-memory request is missing its submission timestamp")
            return _Request(
                worker_id=worker_id,
                request_id=sequence,
                slot=slot,
                count=count,
                model_id=model_id,
                submitted_at_s=float(submitted_ns) / 1.0e9,
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
                model_id = int(view.request_model_ids[slot])
                submitted_ns = int(view.request_submitted_ns[slot])
                request = shared_request_from_header(
                    worker_id,
                    slot,
                    count,
                    sequence,
                    model_id,
                    submitted_ns,
                )
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
                    worker_id, slot, count, sequence, model_id, submitted_ns = endpoints.request_queue.get()
                elif timeout <= 0.0:
                    worker_id, slot, count, sequence, model_id, submitted_ns = (
                        endpoints.request_queue.get_nowait()
                    )
                else:
                    worker_id, slot, count, sequence, model_id, submitted_ns = (
                        endpoints.request_queue.get(timeout=timeout)
                    )
                worker_id, slot, count, sequence, model_id, submitted_ns = (
                    int(worker_id),
                    int(slot),
                    int(count),
                    int(sequence),
                    int(model_id),
                    int(submitted_ns),
                )
                return shared_request_from_header(
                    worker_id,
                    slot,
                    count,
                    sequence,
                    model_id,
                    submitted_ns,
                )
            if timeout is None:
                worker_id, request_id, model_id, submitted_ns, obs, masks = endpoints.request_queue.get()
            elif timeout <= 0.0:
                worker_id, request_id, model_id, submitted_ns, obs, masks = (
                    endpoints.request_queue.get_nowait()
                )
            else:
                worker_id, request_id, model_id, submitted_ns, obs, masks = (
                    endpoints.request_queue.get(timeout=timeout)
                )
            obs = np.ascontiguousarray(obs, dtype=np.float32)
            masks = np.ascontiguousarray(masks, dtype=np.uint8)
            if (
                obs.ndim != 2
                or obs.shape[1] != endpoints.obs_size
                or masks.shape != (obs.shape[0], dz.ACTION_SPACE_SIZE)
            ):
                raise ValueError("queue inference request shape mismatch")
            model_id = int(model_id)
            if not 0 <= model_id < len(resident_models):
                raise ValueError(f"inference request references unknown model id {model_id}")
            submitted_ns = int(submitted_ns)
            if submitted_ns <= 0:
                raise ValueError("queue inference request is missing its submission timestamp")
            return _Request(
                int(worker_id),
                int(request_id),
                None,
                int(obs.shape[0]),
                model_id,
                float(submitted_ns) / 1.0e9,
                obs,
                masks,
            )

        while running:
            if os.getppid() != parent_pid:
                # Do not leave the sole GPU owner orphaned if the trainer is
                # killed before its normal finally block can send ``stop``.
                break
            handle_commands()
            if not running:
                break
            ready = pending.ready_model(time.perf_counter())
            if ready is None:
                until_deadline = pending.seconds_until_deadline(time.perf_counter())
                timeout = 0.01 if until_deadline is None else min(0.01, until_deadline)
                try:
                    pending.add(dequeue_request(timeout=timeout))
                except queue.Empty:
                    continue
                continue

            requests = pending.take_batch(ready.model_id)
            batch_size = sum(request.count for request in requests)
            batch_fire = time.perf_counter()
            resident = resident_models[ready.model_id]
            assert resident.staging.load_requests(
                requests,
                source_obs_version,
                resident.obs_version,
                resident.encoder_generation,
            ) == batch_size
            in_flight = resident.staging.launch(
                resident.model,
                batch_size,
                bool(config.server_fp16),
                evaluator=resident.evaluator,
                use_bf16=use_bf16,
                batch_buckets=batch_buckets,
            )
            inference_time = _drain_during_flight(
                in_flight,
                dequeue_request,
                pending.add,
            )
            _scatter_responses(
                requests,
                resident.staging.values_np[:batch_size],
                resident.staging.policies_np[:batch_size],
                endpoints,
                views,
            )

            metrics.evals += batch_size
            metrics.batches += 1
            metrics.inference_time += inference_time
            assert metrics.batch_waits_s is not None
            batch_waits_s = [max(0.0, batch_fire - request.submitted_at_s) for request in requests]
            metrics.batch_waits_s.extend(batch_waits_s)
            heartbeat_requests += len(requests)
            heartbeat_evals += batch_size
            heartbeat_batches += 1
            emit_heartbeat_if_due()
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
        if (
            not isinstance(config.server_coalesce_target_rows, int)
            or isinstance(config.server_coalesce_target_rows, bool)
            or config.server_coalesce_target_rows <= 0
        ):
            raise ValueError("server_coalesce_target_rows must be a positive integer")
        if (
            isinstance(config.server_coalesce_ms, bool)
            or not isinstance(config.server_coalesce_ms, (int, float))
            or not math.isfinite(float(config.server_coalesce_ms))
            or float(config.server_coalesce_ms) < 0.0
        ):
            raise ValueError("server_coalesce_ms must be finite and non-negative")
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
        # A best-effort once-per-minute channel gives the trainer its latest
        # serving rate without querying the server during the inference loop.
        self.telemetry_queue = context.Queue(maxsize=2)
        self.latest_evals_per_sec = 0.0
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
            args=(self.config, self.endpoints, self.command_queue, self.status_queue, self.telemetry_queue),
            name="dominion-inference-server",
        )
        try:
            self.process.start()
        except BaseException:
            self.request_queue.close()
            self.command_queue.close()
            self.status_queue.close()
            self.telemetry_queue.close()
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

    def _wait_for_status(self, expected_kind: str, generation: int, timeout_s: float | None = None) -> Any:
        effective = float(self.config.server_response_timeout_s) if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + max(5.0, effective)
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

    def sync_models(
        self,
        payloads: list[bytes | tuple[bytes, dict[str, Any]]],
        generation: int,
    ) -> None:
        """Atomically replace the generation's resident model table."""

        if not payloads:
            raise ValueError("inference server requires at least one model payload")
        self.ensure_alive()
        self.command_queue.put(("models", generation, list(payloads)))
        installed = self._wait_for_status(
            "models", generation, timeout_s=getattr(self.config, "server_install_timeout_s", 900.0)
        )
        if int(installed) != len(payloads):
            raise RuntimeError("inference server installed an incomplete model table")

    def collect_metrics(self, generation: int, include_totals: bool = False) -> dict[str, float]:
        self.ensure_alive()
        self.command_queue.put(("metrics", generation, include_totals))
        return self._wait_for_status("metrics", generation)

    def drain_telemetry(self) -> float:
        """Return the latest server rate without sending it a command."""
        while True:
            try:
                summary = self.telemetry_queue.get_nowait()
            except queue.Empty:
                return self.latest_evals_per_sec
            if isinstance(summary, dict):
                self.latest_evals_per_sec = float(summary.get("evals_per_sec", 0.0))

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
        self.telemetry_queue.close()
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
                submitted_ns = time.perf_counter_ns()
                if endpoints.poll == "spin":
                    shared_views.request_counts[slot] = count
                    shared_views.request_model_ids[slot] = 0
                    shared_views.request_submitted_ns[slot] = submitted_ns
                    shared_views.request_sequences[slot] = sequence
                    request: tuple[Any, ...] = ()
                else:
                    request = (worker_id, slot, count, sequence, 0, submitted_ns)
            else:
                request = (worker_id, sequence, 0, time.perf_counter_ns(), obs, masks)
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
        cfg.server_coalesce_target_rows = min(512, max_batch)
        cfg.server_coalesce_ms = 4.0
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
