"""Torch-free inference-server transport protocol used by self-play workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from multiprocessing import shared_memory
except ImportError:  # pragma: no cover - supported CPython versions provide it
    shared_memory = None  # type: ignore[assignment]


def _align(offset: int, alignment: int = 8) -> int:
    return (offset + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class WorkerSharedMemorySpec:
    request_name: str
    response_name: str
    slots: int
    max_request: int
    obs_size: int
    action_size: int


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
