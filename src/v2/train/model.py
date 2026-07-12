from __future__ import annotations

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


def masked_policy_loss(logits: torch.Tensor, legal_mask: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    masked = logits.masked_fill(~legal_mask, -1.0e9)
    log_probs = torch.log_softmax(masked, dim=-1)
    loss = -(target * log_probs).sum(dim=-1).mean()
    probs = torch.softmax(masked, dim=-1)
    entropy = -(probs * log_probs).masked_fill(~legal_mask, 0.0).sum(dim=-1).mean()
    return loss, entropy


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
