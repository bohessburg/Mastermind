"""One-process batched inference service for parallel self-play.

Workers submit raw NumPy leaf buffers and wait for their own response queue.
The server is the only process that creates a CUDA context in ``server`` mode,
so it can combine work from many MCTS runners into large, efficient batches.
"""

from __future__ import annotations

import copy
import io
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

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
class InferenceServerEndpoints:
    request_queue: Any
    response_queues: list[Any]
    alive_event: Any
    response_timeout_s: float
    request_batch_size: int


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


def _server_device(name: str) -> torch.device:
    requested = name.lower()
    if requested not in {"cpu", "cuda"}:
        raise ValueError("server_device must be 'cpu' or 'cuda'")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("server_device=cuda requested but unavailable")
    return torch.device(requested)


def _forward_batch(
    model: DominionNet,
    device: torch.device,
    use_fp16: bool,
    obs: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluate one aggregate batch and return policy logits plus values."""
    obs_cpu = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32))
    masks_cpu = torch.from_numpy(np.ascontiguousarray(masks, dtype=np.bool_))
    start = time.perf_counter()
    if device.type == "cuda":
        # Pinning makes the queued H2D copies eligible for non-blocking DMA;
        # the single server owns stream/context synchronization.
        obs_tensor = obs_cpu.pin_memory().to(device, non_blocking=True)
        masks_tensor = masks_cpu.pin_memory().to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            logits, values = model.evaluate(obs_tensor, masks_tensor)
    else:
        logits, values = model.evaluate(obs_cpu, masks_cpu)
    # Moving to NumPy synchronizes only this batch before routing responses.
    policies = logits.detach().float().cpu().numpy().astype(np.float32, copy=False)
    values_np = values.detach().float().cpu().numpy().astype(np.float32, copy=False)
    return policies, values_np, time.perf_counter() - start


def _server_main(
    config: TrainConfig,
    request_queue: Any,
    response_queues: list[Any],
    command_queue: Any,
    status_queue: Any,
    alive_event: Any,
) -> None:
    """Process entry point; all model updates occur between complete batches."""
    carry: tuple[int, int, np.ndarray, np.ndarray] | None = None
    running = True
    metrics = _GenerationMetrics()
    try:
        device = _server_device(config.server_device)
        model = DominionNet(dz.OBS_SIZE, dz.ACTION_SPACE_SIZE, config.model.hidden_sizes).to(device)
        model.eval()
        alive_event.set()

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
                    # This runs only before dequeuing the next aggregate batch,
                    # so a forward pass never observes a partially updated model.
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

        while running:
            handle_commands()
            if not running:
                break
            if carry is None:
                try:
                    request = request_queue.get(timeout=0.01)
                except queue.Empty:
                    continue
            else:
                request, carry = carry, None

            worker_id, request_id, obs, masks = request
            obs = np.ascontiguousarray(obs, dtype=np.float32)
            masks = np.ascontiguousarray(masks, dtype=np.bool_)
            if obs.ndim != 2 or masks.shape != (obs.shape[0], dz.ACTION_SPACE_SIZE):
                raise ValueError("inference request shape mismatch")
            if obs.shape[0] == 0 or obs.shape[0] > config.server_max_batch:
                raise ValueError("inference request batch is outside server_max_batch")

            requests = [(int(worker_id), int(request_id), obs, masks)]
            batch_size = int(obs.shape[0])
            wait_start = time.perf_counter()
            deadline = wait_start + (float(config.server_max_wait_ms) / 1000.0)
            while batch_size < config.server_max_batch:
                try:
                    candidate = request_queue.get_nowait()
                except queue.Empty:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        break
                    try:
                        candidate = request_queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                candidate_worker, candidate_request, candidate_obs, candidate_masks = candidate
                candidate_obs = np.ascontiguousarray(candidate_obs, dtype=np.float32)
                candidate_masks = np.ascontiguousarray(candidate_masks, dtype=np.bool_)
                if candidate_obs.shape[0] == 0 or candidate_obs.shape[0] > config.server_max_batch:
                    raise ValueError("inference request batch is outside server_max_batch")
                if candidate_masks.shape != (candidate_obs.shape[0], dz.ACTION_SPACE_SIZE):
                    raise ValueError("inference request mask shape mismatch")
                if batch_size + candidate_obs.shape[0] > config.server_max_batch:
                    carry = (int(candidate_worker), int(candidate_request), candidate_obs, candidate_masks)
                    break
                requests.append((int(candidate_worker), int(candidate_request), candidate_obs, candidate_masks))
                batch_size += int(candidate_obs.shape[0])

            # A just-arrived update is applied before this next whole batch;
            # parent generation barriers guarantee no prior-generation callers.
            handle_commands()
            if not running:
                break
            batch_obs = np.concatenate([request[2] for request in requests], axis=0)
            batch_masks = np.concatenate([request[3] for request in requests], axis=0)
            batch_wait = time.perf_counter() - wait_start
            policies, values, inference_time = _forward_batch(
                model,
                device,
                bool(config.server_fp16),
                batch_obs,
                batch_masks,
            )
            metrics.evals += batch_size
            metrics.batches += 1
            metrics.inference_time += inference_time
            metrics.batch_waits_s.append(batch_wait)

            offset = 0
            for request_worker, request_id, request_obs, _ in requests:
                count = int(request_obs.shape[0])
                response_queues[request_worker].put(
                    (
                        "response",
                        request_id,
                        values[offset : offset + count].copy(),
                        policies[offset : offset + count].copy(),
                    )
                )
                offset += count
    except BaseException:
        alive_event.clear()
        status_queue.put(("error", -1, traceback.format_exc()))
        raise
    finally:
        alive_event.clear()


class InferenceServer:
    """Parent-side lifecycle and generation-boundary controls for the server."""

    def __init__(self, config: TrainConfig, worker_count: int):
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        self.config = copy.deepcopy(config)
        context = mp.get_context("spawn")
        self.request_queue = context.Queue(maxsize=max(4, worker_count * 4))
        self.response_queues = [context.Queue(maxsize=2) for _ in range(worker_count)]
        self.command_queue = context.Queue()
        self.status_queue = context.Queue()
        self.alive_event = context.Event()
        # Keep each request bounded so enough workers can contribute to a fat
        # server batch instead of one runner monopolizing all 8192 slots.
        request_batch_size = max(1, min(config.selfplay.max_batch, config.server_max_batch // worker_count))
        self.endpoints = InferenceServerEndpoints(
            request_queue=self.request_queue,
            response_queues=self.response_queues,
            alive_event=self.alive_event,
            response_timeout_s=float(config.server_response_timeout_s),
            request_batch_size=request_batch_size,
        )
        self.process = context.Process(
            target=_server_main,
            args=(self.config, self.request_queue, self.response_queues, self.command_queue, self.status_queue, self.alive_event),
            name="dominion-inference-server",
        )
        self.process.start()

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
