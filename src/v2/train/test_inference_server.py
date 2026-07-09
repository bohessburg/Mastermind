from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from .inference_server import InferenceServer
from .test_train_smoke import read_metrics, tiny_config
from .train import build_objects, run_training
from .workers import ParallelSelfPlayPool


def server_config(tmp_path: Path, *, generations: int = 2):
    cfg = tiny_config(tmp_path, seed=8484, generations=generations)
    cfg.parallel_workers = 2
    cfg.worker_device = "server"
    cfg.server_device = "cpu"
    cfg.server_max_batch = 16
    cfg.server_max_wait_ms = 2.0
    cfg.server_fp16 = False
    cfg.server_response_timeout_s = 3.0
    cfg.model.hidden_sizes = [16]
    cfg.selfplay.n_games = 2
    cfg.selfplay.games_per_generation = 2
    cfg.selfplay.sims_per_move = 2
    cfg.selfplay.max_batch = 8
    cfg.selfplay.max_recorded_moves = 64
    cfg.selfplay.max_tree_nodes = 256
    cfg.optim.batch_size = 8
    cfg.optim.train_steps_per_generation = 1
    cfg.replay.capacity = 512
    return cfg


def test_shared_cpu_inference_server_selfplay_smoke(tmp_path: Path) -> None:
    cfg = server_config(tmp_path)
    start = time.monotonic()
    result = run_training(cfg)
    elapsed = time.monotonic() - start

    assert elapsed < 180.0
    assert [row["games"] for row in result["metrics"]] == [2, 2]
    rows = read_metrics(Path(cfg.metrics_csv))
    assert len(rows) == 2
    assert all(float(row["server_evals_per_sec"]) > 0.0 for row in rows)
    assert all(float(row["server_mean_batch_size"]) > 0.0 for row in rows)
    assert all(float(row["server_batch_wait_p99_ms"]) >= float(row["server_batch_wait_p50_ms"]) for row in rows)


def test_inference_server_routes_fixed_weight_requests_under_concurrent_load(tmp_path: Path) -> None:
    cfg = server_config(tmp_path, generations=1)
    cfg.server_max_batch = 32
    cfg.server_max_wait_ms = 10.0
    torch.manual_seed(333)
    model, _, _ = build_objects(cfg, torch.device("cpu"))
    model.eval()
    obs = np.linspace(-1.0, 1.0, 3 * model.trunk[0].in_features, dtype=np.float32).reshape(
        3, model.trunk[0].in_features
    )
    masks = np.ones((obs.shape[0], model.policy_head.out_features), dtype=np.bool_)
    with torch.no_grad():
        expected_policy, expected_value = model.evaluate(
            torch.from_numpy(obs),
            torch.from_numpy(masks),
        )
    expected_policy_np = expected_policy.numpy()
    expected_value_np = expected_value.numpy()

    server = InferenceServer(cfg, worker_count=2)
    try:
        server.sync_weights(model, generation=1)
        # Submit from separate callers at once. Per-worker queues must still
        # receive only their matching request id and equal fixed outputs.
        barrier = threading.Barrier(3)

        def submit(worker_id: int, request_id: int) -> None:
            barrier.wait()
            server.request_queue.put((worker_id, request_id, obs, masks))

        submitters = [
            threading.Thread(target=submit, args=(0, 11)),
            threading.Thread(target=submit, args=(1, 22)),
        ]
        for submitter in submitters:
            submitter.start()
        barrier.wait()
        for submitter in submitters:
            submitter.join(timeout=5.0)
            assert not submitter.is_alive()
        first = server.endpoints.response_queues[0].get(timeout=5.0)
        second = server.endpoints.response_queues[1].get(timeout=5.0)
        for expected_request_id, (kind, request_id, values, policies) in zip((11, 22), (first, second)):
            assert kind == "response"
            assert request_id == expected_request_id
            # GEMM row tiling may differ between a 3-row direct call and the
            # server's concatenated batch by a few fp32 ulps.
            np.testing.assert_allclose(values, expected_value_np, rtol=1.0e-6, atol=1.0e-6)
            np.testing.assert_allclose(policies, expected_policy_np, rtol=1.0e-6, atol=1.0e-6)
        metrics = server.collect_metrics(generation=1)
        assert metrics["server_mean_batch_size"] >= float(obs.shape[0])
    finally:
        server.close()


def test_server_death_propagates_to_parent_without_hanging(tmp_path: Path) -> None:
    cfg = server_config(tmp_path, generations=1)
    # Give workers a clear in-flight response wait before the test terminates
    # the server, rather than racing a tiny CPU batch to completion.
    cfg.server_max_wait_ms = 1000.0
    cfg.server_response_timeout_s = 2.0
    model, _, replay = build_objects(cfg, torch.device("cpu"))
    server = InferenceServer(cfg, worker_count=2)
    pool = ParallelSelfPlayPool(cfg, server)
    error: list[BaseException] = []
    try:
        server.sync_weights(model, generation=1)

        def generate() -> None:
            try:
                pool.generate(model, replay, generation=1)
            except BaseException as exc:  # asserted on the parent thread
                error.append(exc)

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
        time.sleep(0.5)
        server.process.terminate()
        server.process.join(timeout=5.0)
        thread.join(timeout=10.0)

        assert not thread.is_alive(), "parent remained blocked after inference-server death"
        assert error
        assert "inference server" in str(error[0]).lower()
    finally:
        pool.close()
        server.close()
