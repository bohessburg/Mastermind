from __future__ import annotations

import copy
import queue
import threading
import time
from collections import deque
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .gating import SelfPlaySegment
from .inference_server import (
    InferenceServer,
    WorkerSharedMemoryViews,
    _PerModelRequestQueues,
    _Request,
    _drain_during_flight,
    _scatter_responses,
    serialize_cpu_state_dict,
)
from .model import build_model, model_config_dict
from .observation import downgrade_v3_observations, obs_size_for_version
from .test_train_smoke import read_metrics, tiny_config
from .train import build_objects, run_training
from .workers import (
    ParallelSelfPlayPool,
    _evaluate_manifest_server_groups,
    _server_evaluator,
)


def _fake_request(
    worker_id: int,
    model_id: int,
    count: int,
    submitted_at_s: float,
) -> _Request:
    return _Request(
        worker_id=worker_id,
        request_id=worker_id + 1,
        slot=None,
        count=count,
        model_id=model_id,
        submitted_at_s=submitted_at_s,
        obs=np.zeros((count, 1), dtype=np.float32),
        masks=np.ones((count, 1), dtype=np.uint8),
    )


def test_per_model_firing_is_independent_and_prioritizes_current_model() -> None:
    pending = _PerModelRequestQueues(
        worker_count=16,
        target_rows=10,
        max_batch=32,
        coalesce_s=0.004,
    )
    pending.add(_fake_request(0, 1, 2, 0.000))
    pending.add(_fake_request(1, 0, 6, 0.001))
    pending.add(_fake_request(2, 0, 5, 0.002))

    ready = pending.ready_model(0.002)
    assert ready is not None
    assert (ready.model_id, ready.trigger) == (0, "fill_target")
    assert sum(request.count for request in pending.take_batch(0)) == 11
    assert pending.rows_for(1) == 2
    assert pending.ready_model(0.003) is None

    # A second current-model batch still fires first even after the league
    # trickle's independent deadline expires.
    pending.add(_fake_request(1, 0, 7, 0.004))
    pending.add(_fake_request(2, 0, 4, 0.005))
    ready = pending.ready_model(0.006)
    assert ready is not None
    assert (ready.model_id, ready.trigger) == (0, "fill_target")
    pending.take_batch(0)
    ready = pending.ready_model(0.006)
    assert ready is not None
    assert (ready.model_id, ready.trigger) == (1, "deadline")


def test_all_workers_pending_fires_without_cross_model_row_merging() -> None:
    pending = _PerModelRequestQueues(
        worker_count=3,
        target_rows=512,
        max_batch=8192,
        coalesce_s=0.004,
    )
    pending.add(_fake_request(0, 0, 20, 0.000))
    pending.add(_fake_request(1, 1, 20, 0.001))
    pending.add(_fake_request(2, 0, 20, 0.002))

    ready = pending.ready_model(0.002)
    assert ready is not None
    assert (ready.model_id, ready.trigger) == (0, "all_workers_pending")
    assert sum(request.count for request in pending.take_batch(0)) == 40
    assert pending.rows_for(1) == 20


def test_requests_are_drained_while_fake_evaluator_is_in_flight() -> None:
    incoming = deque(
        [
            _fake_request(1, 0, 3, 0.001),
            _fake_request(2, 1, 2, 0.002),
            _fake_request(3, 0, 4, 0.003),
        ]
    )
    accepted: list[_Request] = []

    class _FakeEvaluatorFlight:
        def ready(self) -> bool:
            return len(accepted) == 3

        def finish(self) -> float:
            assert len(accepted) == 3
            return 0.007

    def dequeue(_timeout: float) -> _Request:
        if not incoming:
            raise queue.Empty
        return incoming.popleft()

    elapsed = _drain_during_flight(_FakeEvaluatorFlight(), dequeue, accepted.append)

    assert elapsed == pytest.approx(0.007)
    assert [(request.worker_id, request.model_id) for request in accepted] == [
        (1, 0),
        (2, 1),
        (3, 0),
    ]


def test_scatter_is_correct_for_interleaved_multi_model_completions() -> None:
    class _FakeView:
        def __init__(self) -> None:
            self.response_values = np.full((2, 4), -1.0, dtype=np.float32)
            self.response_policies = np.full((2, 4, 3), -1.0, dtype=np.float32)
            self.response_sequences = np.zeros((2,), dtype=np.uint64)

    response_queues = [queue.Queue() for _ in range(3)]
    endpoints = SimpleNamespace(transport="shm", poll="queue", response_queues=response_queues)
    views = [_FakeView() for _ in range(3)]

    league_requests = [
        _Request(1, 101, 0, 2, 1, 0.0, np.empty((2, 1)), np.empty((2, 1))),
        _Request(0, 102, 1, 1, 1, 0.0, np.empty((1, 1)), np.empty((1, 1))),
    ]
    league_values = np.asarray([10.0, 11.0, 12.0], dtype=np.float32)
    league_policies = np.arange(9, dtype=np.float32).reshape(3, 3)
    _scatter_responses(league_requests, league_values, league_policies, endpoints, views)

    current_requests = [
        _Request(0, 201, 0, 2, 0, 0.0, np.empty((2, 1)), np.empty((2, 1))),
        _Request(2, 202, 1, 1, 0, 0.0, np.empty((1, 1)), np.empty((1, 1))),
    ]
    current_values = np.asarray([20.0, 21.0, 22.0], dtype=np.float32)
    current_policies = np.arange(30, 39, dtype=np.float32).reshape(3, 3)
    _scatter_responses(current_requests, current_values, current_policies, endpoints, views)

    np.testing.assert_array_equal(views[1].response_values[0, :2], [10.0, 11.0])
    np.testing.assert_array_equal(views[1].response_policies[0, :2], league_policies[:2])
    np.testing.assert_array_equal(views[0].response_values[1, :1], [12.0])
    np.testing.assert_array_equal(views[0].response_policies[1, :1], league_policies[2:3])
    np.testing.assert_array_equal(views[0].response_values[0, :2], [20.0, 21.0])
    np.testing.assert_array_equal(views[0].response_policies[0, :2], current_policies[:2])
    np.testing.assert_array_equal(views[2].response_values[1, :1], [22.0])
    np.testing.assert_array_equal(views[2].response_policies[1, :1], current_policies[2:3])
    assert response_queues[0].get_nowait() == (1, 1, 102)
    assert response_queues[0].get_nowait() == (0, 2, 201)
    assert response_queues[1].get_nowait() == (0, 2, 101)
    assert response_queues[2].get_nowait() == (1, 1, 202)


def server_config(tmp_path: Path, *, generations: int = 2):
    cfg = tiny_config(tmp_path, seed=8484, generations=generations)
    cfg.parallel_workers = 2
    cfg.worker_device = "cpu"
    cfg.server_selfplay = True
    cfg.server_device = "cpu"
    cfg.server_max_batch = 16
    cfg.server_coalesce_target_rows = 16
    cfg.server_coalesce_ms = 4.0
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


@pytest.mark.parametrize("poll", ["queue", "spin"])
def test_inference_server_routes_fixed_weight_requests_under_concurrent_load(
    tmp_path: Path,
    poll: str,
) -> None:
    cfg = server_config(tmp_path, generations=1)
    cfg.server_max_batch = 32
    cfg.server_coalesce_target_rows = 32
    cfg.server_coalesce_ms = 10.0
    cfg.server_poll = poll
    cfg.selfplay.obs_version = 2
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
    views: list[WorkerSharedMemoryViews] = []
    names: list[str] = []
    try:
        if server.transport != "shm":
            pytest.skip("POSIX shared memory is unavailable on this host")
        names = server.shared_memory_names
        views = [WorkerSharedMemoryViews(spec) for spec in server.endpoints.shared_memory_specs or []]
        server.sync_weights(model, generation=1)
        # Submit from separate callers at once. Per-worker queues must still
        # receive only their matching request id and equal fixed outputs.
        barrier = threading.Barrier(3)

        def submit(worker_id: int, request_id: int) -> None:
            barrier.wait()
            view = views[worker_id]
            slot = request_id % view.spec.slots
            np.copyto(view.request_obs[slot, : obs.shape[0]], obs)
            np.copyto(view.request_masks[slot, : obs.shape[0]], masks.astype(np.uint8, copy=False))
            submitted_ns = time.perf_counter_ns()
            if poll == "spin":
                view.request_counts[slot] = obs.shape[0]
                view.request_model_ids[slot] = 0
                view.request_submitted_ns[slot] = submitted_ns
                view.request_sequences[slot] = request_id
            else:
                server.request_queue.put(
                    (worker_id, slot, obs.shape[0], request_id, 0, submitted_ns)
                )

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
        responses: list[tuple[int, int, int]] = []
        for worker_id, request_id in ((0, 11), (1, 22)):
            if poll == "spin":
                slot = request_id % views[worker_id].spec.slots
                deadline = time.monotonic() + 5.0
                while int(views[worker_id].response_sequences[slot]) != request_id:
                    assert time.monotonic() < deadline, "spin response did not arrive"
                    time.sleep(0)
                responses.append((slot, obs.shape[0], request_id))
            else:
                responses.append(server.endpoints.response_queues[worker_id].get(timeout=5.0))
        for worker_id, expected_request_id, response in ((0, 11, responses[0]), (1, 22, responses[1])):
            slot, count, request_id = response
            assert slot == expected_request_id % views[worker_id].spec.slots
            assert count == obs.shape[0]
            assert request_id == expected_request_id
            values = views[worker_id].response_values[slot, :count]
            policies = views[worker_id].response_policies[slot, :count]
            # GEMM row tiling may differ between a 3-row direct call and the
            # server's concatenated batch by a few fp32 ulps.
            np.testing.assert_allclose(values, expected_value_np, rtol=1.0e-4, atol=1.0e-4)
            np.testing.assert_allclose(policies, expected_policy_np, rtol=1.0e-4, atol=1.0e-4)
        metrics = server.collect_metrics(generation=1)
        assert metrics["server_mean_batch_size"] >= float(obs.shape[0])
    finally:
        for view in views:
            view.close()
        server.close()
    for name in names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)


def test_multi_model_server_routes_downgrades_and_scatters(tmp_path: Path) -> None:
    cfg = server_config(tmp_path, generations=1)
    cfg.server_transport = "queue"
    cfg.server_max_batch = 32
    cfg.server_coalesce_target_rows = 32
    cfg.server_coalesce_ms = 10.0
    cfg.selfplay.obs_version = 3
    torch.manual_seed(1919)
    current, _, _ = build_objects(cfg, torch.device("cpu"))
    v2_config = copy.deepcopy(cfg.model)
    v2_config.obs_version = 2
    historical = build_model(v2_config, obs_size_for_version(2), current.policy_head.out_features)
    with torch.no_grad():
        for parameter in historical.parameters():
            parameter.add_(0.25)
    current.eval()
    historical.eval()

    rng = np.random.default_rng(8181)
    worker_obs = [
        rng.normal(size=(5, obs_size_for_version(3))).astype(np.float32),
        rng.normal(size=(4, obs_size_for_version(3))).astype(np.float32),
    ]
    for obs in worker_obs:
        obs[:, 0] = 3.0
        obs[:, 1] = float(obs_size_for_version(3))
    worker_ids = [
        np.asarray([1, 0, 1, 0, 1], dtype=np.uint32),
        np.asarray([0, 1, 0, 1], dtype=np.uint32),
    ]
    worker_masks = [
        np.ones((obs.shape[0], current.policy_head.out_features), dtype=np.bool_)
        for obs in worker_obs
    ]

    server = InferenceServer(cfg, worker_count=2)
    evaluators: list[tuple[object, WorkerSharedMemoryViews | None]] = []
    try:
        server.sync_models(
            [
                (
                    serialize_cpu_state_dict(current),
                    model_config_dict(current._dominion_model_config),
                ),
                (
                    serialize_cpu_state_dict(historical),
                    model_config_dict(historical._dominion_model_config),
                ),
            ],
            generation=1,
        )
        evaluators = [_server_evaluator(server.endpoints, worker_id) for worker_id in range(2)]
        barrier = threading.Barrier(3)
        results: list[tuple[np.ndarray, np.ndarray] | None] = [None, None]

        def evaluate_worker(worker_id: int) -> None:
            barrier.wait()
            results[worker_id] = _evaluate_manifest_server_groups(
                evaluators[worker_id][0],
                worker_obs[worker_id],
                worker_masks[worker_id],
                worker_ids[worker_id],
            )

        threads = [threading.Thread(target=evaluate_worker, args=(worker_id,)) for worker_id in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=10.0)
            assert not thread.is_alive()

        for obs, masks, ids, result in zip(
            worker_obs,
            worker_masks,
            worker_ids,
            results,
            strict=True,
        ):
            assert result is not None
            policies, values = result
            expected_policies = np.empty_like(policies)
            expected_values = np.empty_like(values)
            with torch.no_grad():
                for model_id, model in enumerate((current, historical)):
                    rows = np.flatnonzero(ids == model_id)
                    adapted = obs[rows] if model_id == 0 else downgrade_v3_observations(obs[rows])
                    expected_policy, expected_value = model.evaluate(
                        torch.from_numpy(adapted),
                        torch.from_numpy(masks[rows]),
                    )
                    expected_policies[rows] = expected_policy.numpy()
                    expected_values[rows] = expected_value.numpy()
            np.testing.assert_allclose(policies, expected_policies, rtol=1.0e-4, atol=1.0e-4)
            np.testing.assert_allclose(values, expected_values, rtol=1.0e-4, atol=1.0e-4)
        metrics = server.collect_metrics(generation=1, include_totals=True)
        assert metrics["_server_total_evals"] == 9.0
        assert metrics["_server_total_batches"] == 2.0
        assert metrics["server_mean_batch_size"] == pytest.approx(4.5)
        assert metrics["server_batch_wait_p99_ms"] >= metrics["server_batch_wait_p50_ms"]
    finally:
        for _, views in evaluators:
            if views is not None:
                views.close()
        server.close()


def test_server_bucket_padding_matches_unpadded_evaluation(tmp_path: Path) -> None:
    cfg = server_config(tmp_path, generations=1)
    cfg.server_transport = "queue"
    cfg.server_batch_buckets = [8]
    cfg.server_max_batch = 8
    cfg.selfplay.obs_version = 3
    torch.manual_seed(4242)
    model, _, _ = build_objects(cfg, torch.device("cpu"))
    model.eval()
    obs = np.linspace(
        -0.5,
        0.5,
        3 * obs_size_for_version(3),
        dtype=np.float32,
    ).reshape(3, obs_size_for_version(3))
    obs[:, 0] = 3.0
    obs[:, 1] = float(obs_size_for_version(3))
    masks = np.ones((3, model.policy_head.out_features), dtype=np.bool_)
    with torch.no_grad():
        expected_policy, expected_value = model.evaluate(
            torch.from_numpy(obs),
            torch.from_numpy(masks),
        )

    server = InferenceServer(cfg, worker_count=1)
    evaluate, views = _server_evaluator(server.endpoints, 0)
    try:
        server.sync_models(
            [
                (
                    serialize_cpu_state_dict(model),
                    model_config_dict(model._dominion_model_config),
                )
            ],
            generation=1,
        )
        policy, value = evaluate(obs, masks, 0)
        # CPU GEMM row tiling changes between the live 3-row batch and padded
        # 8-row batch, so allow the same few-ulp tolerance used by CUDA checks.
        np.testing.assert_allclose(policy, expected_policy.numpy(), rtol=1.0e-3, atol=1.0e-3)
        np.testing.assert_allclose(value, expected_value.numpy(), rtol=1.0e-3, atol=1.0e-3)
    finally:
        if views is not None:
            views.close()
        server.close()


def test_abandoned_worker_response_does_not_wedge_server(tmp_path: Path) -> None:
    cfg = server_config(tmp_path, generations=1)
    cfg.server_transport = "queue"
    cfg.selfplay.obs_version = 3
    model, _, _ = build_objects(cfg, torch.device("cpu"))
    model.eval()
    obs = np.zeros((2, obs_size_for_version(3)), dtype=np.float32)
    obs[:, 0] = 3.0
    obs[:, 1] = float(obs_size_for_version(3))
    masks = np.ones((2, model.policy_head.out_features), dtype=np.bool_)
    with torch.no_grad():
        expected_policy, expected_value = model.evaluate(
            torch.from_numpy(obs),
            torch.from_numpy(masks),
        )

    server = InferenceServer(cfg, worker_count=2)
    evaluate, views = _server_evaluator(server.endpoints, 1)
    try:
        server.sync_models(
            [
                (
                    serialize_cpu_state_dict(model),
                    model_config_dict(model._dominion_model_config),
                )
            ],
            generation=1,
        )
        # Simulate worker zero disappearing after submission: its response is
        # intentionally never consumed. Worker one's later request must still
        # complete normally.
        server.request_queue.put(
            (0, 1, 0, time.perf_counter_ns(), obs.copy(), masks.astype(np.uint8))
        )
        policy, value = evaluate(obs, masks, 0)
        np.testing.assert_allclose(policy, expected_policy.numpy(), rtol=1.0e-4, atol=1.0e-4)
        np.testing.assert_allclose(value, expected_value.numpy(), rtol=1.0e-4, atol=1.0e-4)
    finally:
        if views is not None:
            views.close()
        server.close()


def test_queue_transport_fallback_smoke(tmp_path: Path) -> None:
    cfg = server_config(tmp_path, generations=1)
    cfg.server_transport = "queue"
    cfg.selfplay.obs_version = 2
    result = run_training(cfg)
    assert result["metrics"][0]["games"] == 2
    rows = read_metrics(Path(cfg.metrics_csv))
    assert float(rows[0]["server_evals_per_sec"]) > 0.0


def test_spin_poll_falls_back_when_shared_memory_is_unavailable(tmp_path: Path) -> None:
    cfg = server_config(tmp_path, generations=1)
    cfg.server_transport = "queue"
    cfg.server_poll = "spin"
    with pytest.warns(RuntimeWarning, match="server_poll=spin requires shared-memory"):
        server = InferenceServer(cfg, worker_count=2)
    try:
        assert server.transport == "queue"
        assert server.poll == "queue"
    finally:
        server.close()


def test_server_death_propagates_to_parent_without_hanging(tmp_path: Path) -> None:
    cfg = server_config(tmp_path, generations=1)
    # Keep the generation active long enough to terminate the service while
    # workers are collecting or awaiting repeated leaf responses.
    cfg.selfplay.sims_per_move = 64
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


def test_parallel_v3_selfplay_with_v2_league_model_matches_eager_aggregates(
    tmp_path: Path,
) -> None:
    base = server_config(tmp_path / "base", generations=1)
    base.selfplay.obs_version = 3
    base.server_transport = "queue"
    base.selfplay.games_per_generation = 2
    base.selfplay.n_games = 1
    base.selfplay.sims_per_move = 2
    base.selfplay.max_batch = 4
    base.selfplay.max_recorded_moves = 64
    base.selfplay.max_tree_nodes = 256
    torch.manual_seed(7373)
    source_current, _, _ = build_objects(base, torch.device("cpu"))
    v2_config = copy.deepcopy(base.model)
    v2_config.obs_version = 2
    source_historical = build_model(
        v2_config,
        obs_size_for_version(2),
        source_current.policy_head.out_features,
    )
    with torch.no_grad():
        for parameter in source_historical.parameters():
            parameter.mul_(0.75)
    payloads = [
        (
            serialize_cpu_state_dict(source_current),
            model_config_dict(source_current._dominion_model_config),
        ),
        (
            serialize_cpu_state_dict(source_historical),
            model_config_dict(source_historical._dominion_model_config),
        ),
    ]
    segments = [SelfPlaySegment(2, 0, 1, league_opponent="v2-seed.pt")]
    runs: dict[bool, tuple[object, object, dict[str, float] | None]] = {}

    for server_enabled in (False, True):
        cfg = copy.deepcopy(base)
        cfg.server_selfplay = server_enabled
        cfg.worker_device = "cpu"
        current, _, replay = build_objects(cfg, torch.device("cpu"))
        current.load_state_dict(source_current.state_dict())
        server = InferenceServer(cfg, cfg.parallel_workers) if server_enabled else None
        pool = ParallelSelfPlayPool(cfg, server)
        try:
            result = pool.generate(
                current,
                replay,
                generation=1,
                segments=segments,
                model_state_payloads=payloads,
            )
            metrics = server.collect_metrics(1) if server is not None else None
        finally:
            pool.close()
            if server is not None:
                server.close()
        runs[server_enabled] = (result, replay, metrics)

    for server_enabled, (result, replay, metrics) in runs.items():
        assert result.stats.games == 2
        assert result.league_games == 2
        assert result.stats.positions > 0
        assert result.stats.nn_evals > 0
        assert len(replay) == result.stats.positions
        assert np.all(np.isfinite(replay.obs[: len(replay)]))
        assert np.all(np.isfinite(replay.policy[: len(replay)]))
        assert np.all(np.isfinite(replay.value[: len(replay)]))
        np.testing.assert_allclose(
            replay.policy[: len(replay)].sum(axis=1),
            1.0,
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        assert np.all(np.abs(replay.value[: len(replay)]) <= 1.0)
        assert (result.seat_model_evals or {}).get((0, 0), 0) > 0
        assert (result.seat_model_evals or {}).get((1, 1), 0) > 0
        if server_enabled:
            assert metrics is not None
            assert metrics["server_evals_per_sec"] > 0.0
            assert metrics["server_mean_batch_size"] > 0.0
        else:
            assert metrics is None

    eager_result = runs[False][0]
    server_result = runs[True][0]
    position_ratio = server_result.stats.positions / eager_result.stats.positions
    eval_ratio = server_result.stats.nn_evals / eager_result.stats.nn_evals
    assert 0.25 <= position_ratio <= 4.0
    assert 0.25 <= eval_ratio <= 4.0
