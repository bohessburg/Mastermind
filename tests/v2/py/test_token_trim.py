"""Token trimming and the Apple-GPU serving backends.

Run with:
PYTHONPATH=build ./.venv/bin/python -m pytest tests/v2/py/test_token_trim.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dominion_v2_py as dz
from src.v2.train import apple_backend
from src.v2.train.apple_backend import (
    MlxBackend,
    TorchHostBackend,
    active_pile_count_np,
    build_host_backend,
)
from src.v2.train.card_transformer import MAX_PILES, CardTokenNet
from src.v2.train.config import TrainConfig, validate_server_shards
from src.v2.train.model import build_model

KINGDOMS = [
    ["Village", "Smithy", "Market", "Witch", "Moat", "Cellar", "Chapel", "Workshop", "Remodel", "Mine"],
    ["Gardens", "Militia", "Library", "Throne Room", "Festival", "Laboratory", "Bandit", "Sentry", "Artisan", "Harbinger"],
    ["Council Room", "Bureaucrat", "Moneylender", "Poacher", "Vassal", "Merchant", "Gardens", "Witch", "Chapel", "Market"],
]
# The c31 architecture at a width small enough for a unit test.
MODEL_CONFIG = {
    "arch": "card_transformer",
    "obs_version": 3,
    "d_model": 48,
    "n_layers": 2,
    "n_heads": 4,
    "derived_features": True,
    "final_norm": True,
    "aux_margin_buckets": 21,
}
BASE_SET_PILES = 17  # 7 basic piles + 10 kingdom piles in a two-player game


def _observations(rows: int = 192, seed: int = 31) -> tuple[torch.Tensor, torch.Tensor]:
    """Observations and legal masks from random play across several kingdoms."""

    rng = np.random.default_rng(seed)
    obs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    game_index = 0
    while len(obs) < rows:
        kingdom = KINGDOMS[game_index % len(KINGDOMS)]
        game = dz.new_game(dz.Setup(players=2, kingdom=kingdom), 1000 + game_index)
        game_index += 1
        for _ in range(400):
            legal = np.asarray(game.legal_mask(), dtype=bool)
            actions = np.flatnonzero(legal)
            if actions.size == 0 or len(obs) >= rows:
                break
            player = int(game.current_decision()["player"])
            obs.append(np.asarray(game.encode(player, 3), dtype=np.float32).copy())
            masks.append(legal.copy())
            game.step(int(rng.choice(actions)))
    return torch.from_numpy(np.stack(obs)), torch.from_numpy(np.stack(masks))


def _model(seed: int = 7, **overrides: object) -> CardTokenNet:
    torch.manual_seed(seed)
    model = build_model({**MODEL_CONFIG, **overrides}, dz.OBS_SIZE_V3, dz.ACTION_SPACE_SIZE)
    assert isinstance(model, CardTokenNet)
    # Zero-initialised heads would make every comparison below vacuous.
    with torch.no_grad():
        for parameter in model.parameters():
            if float(parameter.abs().max()) == 0.0:
                parameter.normal_(std=0.05)
    return model.eval()


def _pair() -> tuple[CardTokenNet, CardTokenNet]:
    trimmed = _model()
    full = _model(token_trim=False)
    full.load_state_dict(trimmed.state_dict())
    return trimmed, full


def test_trim_is_on_by_default_and_is_not_part_of_the_state_dict() -> None:
    trimmed, full = _pair()
    assert trimmed.token_trim is True
    assert full.token_trim is False
    assert trimmed.state_dict().keys() == full.state_dict().keys()
    assert TrainConfig().model.token_trim is True


def test_active_pile_count_covers_exactly_the_supply() -> None:
    obs, _ = _observations()
    model = _model()
    assert model.active_pile_count(obs) == BASE_SET_PILES
    assert active_pile_count_np(obs.numpy(), model.num_card_defs) == BASE_SET_PILES
    assert model.active_pile_count(torch.zeros_like(obs)) == 1
    assert active_pile_count_np(np.zeros_like(obs.numpy()), model.num_card_defs) == 1
    # Every active pile lies inside the trimmed prefix, none outside it.
    card_mask = model.tokenize(obs).card_mask
    assert bool(card_mask[:, :BASE_SET_PILES].all())
    assert not bool(card_mask[:, BASE_SET_PILES:].any())


def test_tokenize_keeps_the_full_inspection_shape_by_default() -> None:
    obs, _ = _observations(rows=8)
    model = _model()
    full = model.tokenize(obs)
    trimmed = model.tokenize(obs, BASE_SET_PILES)
    assert full.card_tokens.shape[1] == MAX_PILES
    assert trimmed.card_tokens.shape[1] == BASE_SET_PILES
    torch.testing.assert_close(trimmed.card_tokens, full.card_tokens[:, :BASE_SET_PILES])
    torch.testing.assert_close(trimmed.global_token, full.global_token)
    with pytest.raises(ValueError):
        model.tokenize(obs, 0)
    with pytest.raises(ValueError):
        model.tokenize(obs, MAX_PILES + 1)


def test_trimmed_outputs_match_the_untrimmed_network() -> None:
    obs, masks = _observations()
    trimmed, full = _pair()
    assert trimmed._forward_pile_count(obs) == BASE_SET_PILES
    assert full._forward_pile_count(obs) is None
    with torch.no_grad():
        for got, want in zip(trimmed(obs), full(obs)):
            torch.testing.assert_close(got, want, rtol=0, atol=2e-5)
        for got, want in zip(trimmed.forward_with_aux(obs), full.forward_with_aux(obs)):
            torch.testing.assert_close(got, want, rtol=0, atol=2e-5)
    got_logits, got_values = trimmed.evaluate(obs, masks)
    want_logits, want_values = full.evaluate(obs, masks)
    torch.testing.assert_close(got_logits, want_logits, rtol=0, atol=2e-5)
    torch.testing.assert_close(got_values, want_values, rtol=0, atol=2e-5)
    assert bool((got_logits.argmax(1) == want_logits.argmax(1)).all())


def test_trimmed_gradients_match_the_untrimmed_network() -> None:
    obs, masks = _observations(rows=64)
    trimmed, full = _pair()
    gradients = []
    for model in (trimmed, full):
        model.train()
        model.zero_grad()
        logits, values, aux = model.forward_with_aux(obs)
        log_probs = torch.log_softmax(logits.masked_fill(~masks, -1.0e9), dim=1)
        loss = -log_probs.max(dim=1).values.mean() + (values**2).mean() + torch.logsumexp(aux, dim=1).mean()
        loss.backward()
        gradients.append({name: p.grad for name, p in model.named_parameters()})
    for name, grad in gradients[0].items():
        other = gradients[1][name]
        assert (grad is None) == (other is None), name
        if grad is not None:
            torch.testing.assert_close(grad, other, rtol=0, atol=2e-5, msg=name)


def test_trim_handles_empty_and_mixed_batches() -> None:
    obs, masks = _observations(rows=32)
    trimmed, full = _pair()
    zeros = torch.zeros_like(obs)
    with torch.no_grad():
        logits, values = trimmed(zeros)
        assert logits.shape == (obs.shape[0], dz.ACTION_SPACE_SIZE)
        assert bool(torch.isfinite(values).all())
        # Padded rows (the server zero-pads batches) do not disturb real rows.
        mixed = torch.cat((obs, zeros), dim=0)
        mixed_logits, mixed_values = trimmed(mixed)
        want_logits, want_values = full(obs)
    torch.testing.assert_close(mixed_logits[: obs.shape[0]], want_logits, rtol=0, atol=2e-5)
    torch.testing.assert_close(mixed_values[: obs.shape[0]], want_values, rtol=0, atol=2e-5)


def test_compiled_graphs_keep_the_full_width(monkeypatch: pytest.MonkeyPatch) -> None:
    obs, _ = _observations(rows=8)
    trimmed, _ = _pair()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert trimmed._forward_pile_count(obs) is None


def test_full_graph_server_compile_is_unaffected_by_the_trim() -> None:
    """The CUDA serving graph must stay one static 49-token graph."""

    from src.v2.train.inference_server import _ServerEvaluator

    obs, masks = _observations(rows=16)
    trimmed, full = _pair()
    widths: list[int] = []
    encoder_forward = trimmed.transformer.forward

    def recording_forward(tokens: torch.Tensor, **kwargs: object) -> torch.Tensor:
        widths.append(int(tokens.shape[1]))
        return encoder_forward(tokens, **kwargs)

    trimmed.transformer.forward = recording_forward  # type: ignore[method-assign]
    with torch.no_grad():
        trimmed.evaluate(obs, masks)
        assert widths == [1 + BASE_SET_PILES]
        trimmed.transformer.forward = encoder_forward  # type: ignore[method-assign]
        try:
            compiled = torch.compile(_ServerEvaluator(trimmed).eval(), backend="eager", fullgraph=True)
            got_logits, got_values = compiled(obs, masks)
            explanation = torch._dynamo.explain(_ServerEvaluator(trimmed).eval())(obs, masks)
        except RuntimeError as exc:  # pragma: no cover - platforms without dynamo
            pytest.skip(f"torch.compile is unavailable: {exc}")
        want_logits, want_values = full.evaluate(obs, masks)
    torch.testing.assert_close(got_logits, want_logits, rtol=0, atol=2e-5)
    torch.testing.assert_close(got_values, want_values, rtol=0, atol=2e-5)
    assert explanation.graph_count == 1 and explanation.graph_break_count == 0
    token_widths = {
        int(node.meta["example_value"].shape[1])
        for graph in explanation.graphs
        for node in graph.graph.nodes
        if node.op == "call_function"
        and "cat" in str(node.target)
        and getattr(node.meta.get("example_value"), "ndim", 0) == 3
        and int(node.meta["example_value"].shape[2]) == trimmed.d_model
    }
    assert token_widths == {1 + MAX_PILES}


def test_float16_weights_run_and_stay_close() -> None:
    obs, masks = _observations()
    reference = _model()
    half = _model().half()
    with torch.no_grad():
        want_logits, want_values = reference.evaluate(obs, masks)
        try:
            got_logits, got_values = half.evaluate(obs.half(), masks)
        except RuntimeError as exc:  # pragma: no cover - CPU builds without half kernels
            pytest.skip(f"float16 is not supported on this CPU build: {exc}")
    assert got_logits.dtype == torch.float32
    assert bool((got_logits[~masks] == reference.POINTER_FILL).all())
    torch.testing.assert_close(got_values.float(), want_values, rtol=0, atol=2e-2)
    torch.testing.assert_close(got_logits[masks], want_logits[masks], rtol=0, atol=5e-2)


# -- Apple-GPU serving backends ---------------------------------------------

HAS_MPS = bool(torch.backends.mps.is_available())
needs_mps = pytest.mark.skipif(not HAS_MPS, reason="requires an Apple GPU")


def _reference(model: CardTokenNet, obs: torch.Tensor, masks: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    logits, values = model.evaluate(obs, masks)
    return logits.numpy(), values.numpy()


def _assert_backend_matches(
    backend: object,
    model: CardTokenNet,
    obs: torch.Tensor,
    masks: torch.Tensor,
    *,
    atol: float,
) -> None:
    want_logits, want_values = _reference(model, obs, masks)
    got_logits, got_values = backend.evaluate(obs.numpy(), masks.numpy())  # type: ignore[attr-defined]
    assert got_logits.dtype == np.float32 and got_values.dtype == np.float32
    assert got_logits.shape == want_logits.shape and got_values.shape == want_values.shape
    legal = masks.numpy()
    np.testing.assert_allclose(got_values, want_values, rtol=0, atol=atol)
    np.testing.assert_allclose(got_logits[legal], want_logits[legal], rtol=0, atol=atol * 3)
    assert bool((got_logits[~legal] <= -1.0e8).all())
    agreement = float((got_logits.argmax(1) == want_logits.argmax(1)).mean())
    assert agreement >= (1.0 if atol <= 1e-3 else 0.97)


def test_torch_host_backend_matches_the_model_on_cpu() -> None:
    obs, masks = _observations()
    model = _model()
    backend = TorchHostBackend(model, torch.device("cpu"))
    assert backend.padded_rows(100, 4096) == 100
    _assert_backend_matches(backend, model, obs, masks, atol=1e-5)


@needs_mps
@pytest.mark.parametrize("half", [False, True])
def test_torch_host_backend_on_mps(half: bool) -> None:
    obs, masks = _observations()
    reference = _model()
    served = _model().to("mps")
    backend = TorchHostBackend(served, torch.device("mps"), half=half)
    _assert_backend_matches(backend, reference, obs, masks, atol=2e-2 if half else 1e-4)
    # A weight update on the resident model must reach the half-precision copy.
    updated = _model(seed=99)
    served.load_state_dict(updated.state_dict())
    backend.refresh()
    _assert_backend_matches(backend, updated, obs, masks, atol=2e-2 if half else 1e-4)


@needs_mps
@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("token_trim", [True, False])
def test_mlx_backend_matches_pytorch(half: bool, token_trim: bool) -> None:
    pytest.importorskip("mlx.core")
    obs, masks = _observations()
    model = _model(token_trim=token_trim)
    backend = MlxBackend(model, half=half)
    _assert_backend_matches(backend, model, obs, masks, atol=2e-2 if half else 1e-4)
    # Zero rows (server padding) and a second batch shape reuse the same weights.
    padded = torch.cat((obs, torch.zeros_like(obs[:16])), dim=0)
    padded_masks = torch.cat((masks, torch.zeros_like(masks[:16])), dim=0)
    got_logits, got_values = backend.evaluate(padded.numpy(), padded_masks.numpy())
    want_logits, want_values = _reference(model, obs, masks)
    np.testing.assert_allclose(got_values[: obs.shape[0]], want_values, rtol=0, atol=2e-2 if half else 1e-4)
    updated = _model(seed=99, token_trim=token_trim)
    model.load_state_dict(updated.state_dict())
    backend.refresh()
    _assert_backend_matches(backend, updated, obs, masks, atol=2e-2 if half else 1e-4)


@needs_mps
def test_mlx_backend_rejects_other_architectures() -> None:
    pytest.importorskip("mlx.core")
    mlp = build_model({"arch": "mlp", "hidden_sizes": [16]}, dz.OBS_SIZE_V3, dz.ACTION_SPACE_SIZE)
    with pytest.raises(ValueError, match="card_transformer"):
        MlxBackend(mlp)


def test_mlx_padding_limits_distinct_batch_shapes() -> None:
    padded = {MlxBackend.padded_rows(None, count, 8192) for count in range(1, 8193)}  # type: ignore[arg-type]
    assert len(padded) <= 32
    assert all(MlxBackend.padded_rows(None, count, 8192) >= count for count in (1, 63, 64, 65, 4097, 8192))  # type: ignore[arg-type]
    assert MlxBackend.padded_rows(None, 8190, 8192) == 8192  # type: ignore[arg-type]
    assert MlxBackend.padded_rows(None, 100, 100) == 100  # type: ignore[arg-type]


def test_backend_selection_rules() -> None:
    model = _model()
    assert build_host_backend(model, torch.device("cpu"), "torch", half=False) is None
    assert build_host_backend(model, torch.device("cuda"), "torch", half=True) is None
    with pytest.raises(ValueError, match="mps"):
        build_host_backend(model, torch.device("cpu"), "mlx", half=False)
    with pytest.raises(ValueError, match="server_backend"):
        build_host_backend(model, torch.device("cpu"), "tensorflow", half=False)

    config = TrainConfig()
    assert config.server_backend == "torch"
    config.server_backend = "mlx"
    config.server_device = "cuda"
    with pytest.raises(ValueError, match="mps"):
        validate_server_shards(config)
    config.server_device = "mps"
    validate_server_shards(config)
    config.server_backend = "jax"
    with pytest.raises(ValueError, match="server_backend"):
        validate_server_shards(config)


@needs_mps
@pytest.mark.parametrize(
    ("backend_name", "half"),
    [("torch", False), ("torch", True), ("mlx", False), ("mlx", True)],
)
def test_server_resident_model_serves_through_the_selected_backend(backend_name: str, half: bool) -> None:
    if backend_name == "mlx":
        pytest.importorskip("mlx.core")
    from src.v2.train.inference_server import _build_resident_model, _server_device
    from src.v2.train.model import model_config_dict

    obs, masks = _observations(rows=100)
    config = TrainConfig()
    for key, value in MODEL_CONFIG.items():
        setattr(config.model, key, value)
    config.selfplay.obs_version = 3
    config.server_device = "mps"
    config.server_backend = backend_name
    config.server_fp16 = half
    config.server_max_batch = 256
    device = _server_device(config.server_device)
    resident = _build_resident_model(model_config_dict(config.model), device, config, (), False)
    assert resident.host_backend is not None and resident.host_backend.name == backend_name
    assert next(resident.model.parameters()).device.type == ("cpu" if backend_name == "mlx" else "mps")

    source = _model(seed=5)
    resident.load_weights(source.state_dict())
    count = obs.shape[0]
    resident.staging.obs_np[:count] = obs.numpy()
    resident.staging.masks_np[:count] = masks.numpy().astype(np.uint8)
    in_flight = resident.staging.launch(
        resident.model, count, half, evaluator=None, host_backend=resident.host_backend
    )
    assert in_flight.ready()
    assert in_flight.finish() > 0.0
    want_logits, want_values = _reference(source, obs, masks)
    atol = 2e-2 if half else 1e-4
    np.testing.assert_allclose(resident.staging.values_np[:count], want_values, rtol=0, atol=atol)
    legal = masks.numpy()
    np.testing.assert_allclose(
        resident.staging.policies_np[:count][legal], want_logits[legal], rtol=0, atol=atol * 3
    )


def test_module_lists_both_backends() -> None:
    assert apple_backend.SERVER_BACKENDS == ("torch", "mlx")
