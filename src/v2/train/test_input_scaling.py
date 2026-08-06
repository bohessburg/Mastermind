from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

import dominion_v2_py as dz

from .config import SelfPlayConfig, TrainConfig
from .model import DominionNet
from .replay import ReplayBuffer
from .selfplay import run_self_play_generation
from .train import build_objects, load_checkpoint, load_full_checkpoint, save_checkpoint


class _DivisionTracker(TorchDispatchMode):
    """Record tensor division while executing a model forward pass."""

    def __init__(self) -> None:
        self.operations: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        if func._schema.name == "aten::div":
            self.operations.append(str(func))
        return func(*args, **(kwargs or {}))


def test_default_input_scale_preserves_legacy_forward_without_division() -> None:
    torch.manual_seed(20260711)
    model = DominionNet(obs_size=7, action_size=5, hidden_sizes=[9])
    observations = torch.randn(3, 7)

    with torch.no_grad():
        legacy_hidden = model.trunk(observations)
        expected_logits = model.policy_head(legacy_hidden)
        expected_values = model.value_head(legacy_hidden).squeeze(-1)

    tracker = _DivisionTracker()
    with tracker, torch.no_grad():
        logits, values = model(observations)

    assert model.input_scale == 1.0
    assert tracker.operations == []
    assert "input_scale" not in model.state_dict()
    torch.testing.assert_close(logits, expected_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(values, expected_values, rtol=0.0, atol=0.0)


def test_input_scale_round_trips_in_checkpoint_config_and_defaults_for_legacy_payload(tmp_path) -> None:
    config = TrainConfig()
    config.device = "cpu"
    config.checkpoint_dir = str(tmp_path / "checkpoints")
    config.metrics_csv = str(tmp_path / "checkpoints" / "metrics.csv")
    config.model.hidden_sizes = [8]
    config.model.input_scale = 16.0
    model, optimizer, replay = build_objects(config, torch.device("cpu"))

    checkpoint = save_checkpoint(config, 1, model, optimizer, replay)
    payload = load_full_checkpoint(checkpoint, "cpu")
    assert payload["config"]["model"]["input_scale"] == 16.0
    assert "input_scale" not in payload["model"]

    _, generation, restored, _, _ = load_checkpoint(checkpoint, torch.device("cpu"))
    assert generation == 1
    assert restored.input_scale == 16.0

    del payload["config"]["model"]["input_scale"]
    legacy_checkpoint = checkpoint.with_name("legacy_without_input_scale.pt")
    torch.save(payload, legacy_checkpoint)
    _, _, legacy_model, _, _ = load_checkpoint(legacy_checkpoint, torch.device("cpu"))
    assert legacy_model.input_scale == 1.0


def test_scaled_v2_selfplay_values_overfit_with_adam() -> None:
    """A small real v2 generation retains the observed scaled-input fit."""
    torch.manual_seed(20260711)
    selfplay = SelfPlayConfig(
        n_games=1,
        games_per_generation=1,
        sims_per_move=2,
        max_batch=16,
        dirichlet_frac=0.25,
        temp_moves=4,
        obs_version=2,
        kingdom_mode="fixed",
        max_recorded_moves=192,
        max_tree_nodes=512,
    )
    model = DominionNet(
        dz.OBS_SIZE_V2,
        dz.ACTION_SPACE_SIZE,
        hidden_sizes=[48, 48],
        input_scale=16.0,
    )
    replay = ReplayBuffer(512, dz.OBS_SIZE_V2, dz.ACTION_SPACE_SIZE, seed=20260711)
    stats = run_self_play_generation(
        model,
        replay,
        selfplay,
        seed=20260711 ^ 0xA11CE,
        device=torch.device("cpu"),
    )
    assert stats.games == 1
    assert len(replay) > 0

    observations = torch.as_tensor(replay.obs[: len(replay)])
    targets = torch.as_tensor(replay.value[: len(replay)])
    assert set(targets.tolist()) == {-1.0, 1.0}
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    for _ in range(200):
        _, values = model(observations)
        value_loss = F.mse_loss(values, targets)
        optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        optimizer.step()

    final_mse = F.mse_loss(model(observations)[1], targets)
    # The raw-input companion remained above MSE 1.0 in the offline overfit
    # experiment; do not run it here because it would double test runtime.
    assert final_mse.item() < 0.4
