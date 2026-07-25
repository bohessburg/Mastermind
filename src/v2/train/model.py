from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import torch
from torch import nn


class DominionNet(nn.Module):
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_sizes: list[int] | tuple[int, ...] = (1024, 1024, 512),
        input_scale: float = 1.0,
    ):
        super().__init__()
        # Deliberately not a buffer: checkpoints persist this through their
        # config payload, keeping existing state_dicts byte-for-byte stable.
        self.input_scale = float(input_scale)
        layers: list[nn.Module] = []
        last = obs_size
        for width in hidden_sizes:
            layers.append(nn.Linear(last, int(width)))
            layers.append(nn.ReLU())
            last = int(width)
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, action_size)
        self.value_head = nn.Sequential(nn.Linear(last, 1), nn.Tanh())

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Preserve the legacy path exactly, including avoiding an unnecessary
        # tensor operation for the many existing input_scale=1.0 checkpoints.
        if self.input_scale != 1.0:
            obs = obs / self.input_scale
        x = self.trunk(obs)
        return self.policy_head(x), self.value_head(x).squeeze(-1)

    @torch.no_grad()
    def evaluate(self, obs: torch.Tensor, legal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = self(obs)
        masked_logits = logits.masked_fill(~legal_mask, -1.0e9)
        return masked_logits, values


def model_config_dict(model_config: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Return portable model metadata from a config mapping or dataclass."""
    if isinstance(model_config, Mapping):
        return dict(model_config)
    if is_dataclass(model_config) and not isinstance(model_config, type):
        return asdict(model_config)
    raise TypeError("model config must be a mapping or dataclass instance")


def build_model(model_config: Mapping[str, Any] | Any, obs_size: int, action_size: int) -> nn.Module:
    """Build a policy/value network described by persisted model metadata.

    ``arch`` intentionally defaults to ``mlp`` so checkpoints written before
    architecture metadata existed reconstruct byte-for-byte-compatible
    :class:`DominionNet` instances.
    """
    config = model_config_dict(model_config)
    arch = config.get("arch", "mlp")
    if not isinstance(arch, str):
        raise ValueError("model.arch must be a string")
    config["arch"] = arch
    # Payloads sent to self-play workers must describe the model's actual
    # input ABI.  Old checkpoints omitted this transformer field, but the
    # input width remains authoritative and lets v3 runners downgrade only
    # those historical v2 opponents.
    observed_version = {1141: 1, 1717: 2, 1788: 3}.get(int(obs_size))
    if observed_version is not None:
        configured_version = config.get("obs_version")
        if arch == "card_transformer" and configured_version is not None and configured_version != observed_version:
            raise ValueError(
                f"model.obs_version {configured_version} does not match observation size {obs_size}"
            )
        config["obs_version"] = observed_version

    if arch == "mlp":
        model: nn.Module = DominionNet(
            obs_size,
            action_size,
            hidden_sizes=config.get("hidden_sizes", (1024, 1024, 512)),
            input_scale=config.get("input_scale", 1.0),
        )
    elif arch == "card_transformer":
        from .card_transformer import CardTokenNet

        model = CardTokenNet(
            obs_size,
            action_size,
            d_model=config.get("d_model", 192),
            n_layers=config.get("n_layers", 3),
            n_heads=config.get("n_heads", 4),
            ffn_multiplier=config.get("ffn_multiplier", 4),
            dropout=config.get("dropout", 0.0),
            obs_version=config.get("obs_version"),
        )
    else:
        raise ValueError(f"unknown model.arch {arch!r}; expected 'mlp' or 'card_transformer'")
    # This non-state-dict metadata lets worker processes reconstruct a mixed
    # league model table after its weights have been serialized independently.
    model._dominion_model_config = config  # type: ignore[attr-defined]
    return model


def masked_policy_loss(logits: torch.Tensor, legal_mask: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    masked = logits.masked_fill(~legal_mask, -1.0e9)
    log_probs = torch.log_softmax(masked, dim=-1)
    loss = -(target * log_probs).sum(dim=-1).mean()
    probs = torch.softmax(masked, dim=-1)
    entropy = -(probs * log_probs).masked_fill(~legal_mask, 0.0).sum(dim=-1).mean()
    return loss, entropy


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
