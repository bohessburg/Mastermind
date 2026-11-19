from __future__ import annotations

import pytest

from scripts.infra.providers.common import MissingApiKeyError
from scripts.infra.providers.runpod import RunPodProvider
from scripts.infra.providers.vastai import VastAIProvider
from scripts.infra.remote_run import main


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self) -> None:
        self.gets: list[tuple[str, dict | None]] = []
        self.posts: list[tuple[str, dict | None]] = []
        self.deletes: list[str] = []
        self.next_get = {}
        self.next_post = {}
        self.next_delete = {}

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs.get("params")))
        return FakeResponse(self.next_get)

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs.get("json")))
        return FakeResponse(self.next_post)

    def delete(self, url, **kwargs):
        self.deletes.append(url)
        return FakeResponse(self.next_delete)


def test_vastai_provider_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        VastAIProvider(session=FakeSession())


def test_vastai_search_filters_and_sorts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAST_API_KEY", "test-key")
    session = FakeSession()
    session.next_get = {
        "offers": [
            {"id": 2, "gpu_name": "RTX 4090", "dph_total": 0.72, "num_cpus": 16, "inet_up": 850},
            {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.62, "num_cpus": 8, "inet_up": 900},
            {"id": 3, "gpu_name": "RTX 4090", "dph_total": 0.68, "num_cpus": 20, "inet_up": 700},
        ]
    }
    offers = VastAIProvider(session=session).search_offers("RTX 4090", 16, 0.70)
    assert [offer.id for offer in offers] == ["3"]
    assert session.gets


def test_vastai_create_status_destroy_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAST_API_KEY", "test-key")
    session = FakeSession()
    provider = VastAIProvider(session=session)

    session.next_post = {"new_contract": 42, "status": "creating"}
    instance = provider.create_instance("offer-1", "nvidia/cuda")
    assert instance.id == "42"
    assert session.posts and session.posts[0][1]["image"] == "nvidia/cuda"
    assert "onstart" not in session.posts[0][1]

    session.next_get = {
        "instance": {
            "id": 42,
            "actual_status": "running",
            "ssh_host": "203.0.113.10",
            "ssh_port": 2222,
            "ssh_user": "root",
            "dph_total": 0.55,
        }
    }
    status = provider.instance_status("42")
    assert status.status == "running"
    assert status.ssh_target() == "root@203.0.113.10"
    assert status.ssh_port == 2222

    session.next_delete = {"success": True}
    provider.destroy_instance("42")
    assert session.deletes[-1].endswith("/instances/42/")


def test_runpod_search_uses_graphql_and_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    session = FakeSession()
    session.next_post = {
        "data": {
            "gpuTypes": [
                {"id": "NVIDIA GeForce RTX 4090", "displayName": "RTX 4090", "lowestPrice": 0.49},
                {"id": "A100", "displayName": "A100", "lowestPrice": 1.50},
            ]
        }
    }
    offers = RunPodProvider(session=session).search_offers("RTX 4090", 16, 1.0)
    assert len(offers) == 1
    assert offers[0].gpu == "RTX 4090"
    assert session.posts and "query" in session.posts[0][1]


def test_runpod_create_status_destroy_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    session = FakeSession()
    provider = RunPodProvider(session=session)

    session.next_post = {"data": {"podRentInterruptable": {"id": "pod-1", "desiredStatus": "RUNNING"}}}
    instance = provider.create_instance("RTX4090", "dominion-v2-train")
    assert instance.id == "pod-1"
    assert "podRentInterruptable" in session.posts[-1][1]["query"]

    session.next_post = {
        "data": {
            "pod": {
                "id": "pod-1",
                "desiredStatus": "RUNNING",
                "costPerHr": 0.44,
                "runtime": {
                    "ports": [
                        {
                            "ip": "203.0.113.20",
                            "isIpPublic": True,
                            "privatePort": 22,
                            "publicPort": 30022,
                            "type": "tcp",
                        }
                    ]
                },
            }
        }
    }
    status = provider.instance_status("pod-1")
    assert status.ssh_target() == "root@203.0.113.20"
    assert status.ssh_port == 30022

    session.next_post = {"data": {"podStop": {"id": "pod-1", "desiredStatus": "EXITED"}}}
    provider.destroy_instance("pod-1")
    assert "podStop" in session.posts[-1][1]["query"]


def test_remote_run_launch_dry_run_prints_commands(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--dry-run",
            "--provider",
            "runpod",
            "--ssh-target",
            "root@example.invalid",
            "--run",
            "ci",
            "launch",
            "--config",
            "src/v2/train/configs/smoke.json",
            "--resume-latest",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "scp src/v2/train/configs/smoke.json" in out
    assert "docker run -d --restart on-failure --gpus all" in out
    assert "--resume latest" in out
    assert "WATCHDOG" in out


def test_remote_run_native_up_bootstrap_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--dry-run",
            "--provider",
            "vastai",
            "--ssh-target",
            "root@example.invalid",
            "--run",
            "nativeci",
            "up",
            "--offer-id",
            "offer-1",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "git archive --format=tar HEAD" in out
    assert "tar -xzf /tmp/dominion-src.tar.gz" in out
    assert "apt-get install -y cmake build-essential" in out
    assert "python3 -m pip install --upgrade pybind11 numpy" in out
    assert "torch.cuda.is_available()" in out
    assert "cu128" in out
    assert "cmake --build build --target dominion_v2_py" in out


def test_remote_run_native_smoke_launch_status_sync_down_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    common = [
        "--dry-run",
        "--provider",
        "vastai",
        "--mode",
        "native",
        "--instance-id",
        "123",
        "--ssh-target",
        "root@example.invalid",
        "--run",
        "nativeci",
    ]
    assert main([*common, "smoke"]) == 0
    assert main([*common, "launch", "--config", "src/v2/train/configs/smoke.json", "--resume-latest"]) == 0
    assert main([*common, "status"]) == 0
    assert main([*common, "sync"]) == 0
    assert main([*common, "down"]) == 0
    out = capsys.readouterr().out
    assert "PYTHONPATH=build python3 -m src.v2.train.train --config src/v2/train/configs/smoke.json --smoke --device auto" in out
    assert '"device": "cuda"' in out
    assert "nohup bash -lc" in out
    assert "console.log" in out
    assert "train.pid" in out
    assert "watchdog.marker" in out
    assert "native_pid=" in out
    assert "rsync -az --partial root@example.invalid:/root/dominion/checkpoints/nativeci/" in out
    assert out.rfind("rsync") < out.rfind("destroy vastai instance 123")


def test_remote_run_down_dry_run_syncs_before_destroy(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--dry-run",
            "--provider",
            "runpod",
            "--mode",
            "docker",
            "--instance-id",
            "123",
            "--ssh-target",
            "root@example.invalid",
            "--run",
            "ci",
            "down",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert out.index("rsync") < out.index("destroy runpod instance 123")
