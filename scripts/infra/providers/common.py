from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


class ProviderError(RuntimeError):
    pass


class MissingApiKeyError(ProviderError):
    pass


@dataclass(frozen=True)
class Offer:
    id: str
    gpu: str
    price_per_hour: float
    vcpus: int
    upload_mbps: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Instance:
    id: str
    status: str
    ssh_host: str = ""
    ssh_port: int = 22
    ssh_user: str = "root"
    cost_per_hour: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def ssh_target(self) -> str:
        if not self.ssh_host:
            raise ProviderError(f"instance {self.id} has no SSH host yet")
        return f"{self.ssh_user}@{self.ssh_host}" if self.ssh_user else self.ssh_host


def require_api_key(name: str, explicit: str | None = None) -> str:
    value = explicit or os.environ.get(name)
    if not value:
        raise MissingApiKeyError(f"{name} is required for live provider calls")
    return value


def requests_session():
    try:
        import requests  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on host env
        raise ProviderError("requests is required for live provider calls") from exc
    return requests.Session()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default
