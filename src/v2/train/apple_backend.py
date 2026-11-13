"""Host-buffer inference backends for Apple-silicon inference servers.

The CUDA serving path in :mod:`inference_server` is built around pinned
staging, asynchronous device copies, and compiled graphs.  None of that
applies to a Metal GPU, so an ``mps`` server evaluates through one of the
small synchronous backends here instead.  Both read the server's NumPy staging
rows and return float32 NumPy ``(policies, values)``:

``TorchHostBackend``
    The resident PyTorch model on ``mps``.  With ``half=True`` it serves from a
    float16 copy of the weights (MPS fp16 autocast is slower than fp32, a
    plain half-precision copy is not).

``MlxBackend``
    An MLX reimplementation of :class:`CardTokenNet` inference.  MLX is an
    optional dependency (``pip install mlx``) imported only when this backend
    is requested.  The PyTorch model stays the owner of the weights; they are
    converted again after every weight update (:meth:`refresh`).

Measured on an M1 Max with the c31 net at batch 4096 (evals/s): PyTorch fp32
46k, PyTorch half 52k, MLX fp32 48k, MLX half 62k, all with token trimming.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch

from .card_transformer import (
    ACTION_DEF_COUNT,
    DEF_GARDENS,
    MAX_LANDSCAPES,
    MAX_OPPONENTS,
    MAX_PILES,
    MAX_SLOTS,
    OBS_DECISION_OFFSET,
    OBS_LANDSCAPE_OFFSET,
    OBS_META_OFFSET,
    OBS_OPPONENT_OFFSET,
    OBS_OWN_OFFSET,
    OBS_RESOURCE_OFFSET,
    OBS_SUPPLY_OFFSET,
    OBS_TURN_OFFSET,
    OBS_V3_SELECT_SEMANTIC_OFFSET,
    OBS_V3_TRASH_OFFSET,
    OPPONENT_BLOCK_SIZE_V2,
    OPTION_ACTION_COUNT,
    SELECT_SEMANTIC_COUNT,
    SPEND_ACTION_COUNT,
    SUPPLY_BLOCK_SIZE,
    CardTokenNet,
)

SERVER_BACKENDS = ("torch", "mlx")
MASK_FILL = -1.0e9


def active_pile_count_np(obs: np.ndarray, num_card_defs: int = ACTION_DEF_COUNT) -> int:
    """NumPy twin of :meth:`CardTokenNet.active_pile_count` for host buffers."""

    base_ids_plus_one = np.rint(
        obs[:, OBS_SUPPLY_OFFSET + 2 : OBS_SUPPLY_OFFSET + (MAX_PILES * SUPPLY_BLOCK_SIZE) : SUPPLY_BLOCK_SIZE]
    )
    used = ((base_ids_plus_one >= 1) & (base_ids_plus_one <= num_card_defs)).any(axis=0)
    indices = np.flatnonzero(used)
    return max(1, int(indices[-1]) + 1) if indices.size else 1


class TorchHostBackend:
    """Serve a resident PyTorch model on ``device`` from host NumPy rows."""

    name = "torch"

    def __init__(self, model: torch.nn.Module, device: torch.device, *, half: bool = False):
        self.model = model
        self.device = device
        self.half = bool(half)
        self._half_model: torch.nn.Module | None = None
        self.refresh()

    def refresh(self) -> None:
        """Re-derive serving weights after the resident model was updated."""

        if not self.half:
            return
        if self._half_model is None:
            self._half_model = copy.deepcopy(self.model).half().eval()
        else:
            # load_state_dict copies into the existing float16 parameters.
            self._half_model.load_state_dict(self.model.state_dict())
            self._half_model.eval()

    def padded_rows(self, count: int, capacity: int) -> int:
        return int(count)

    def evaluate(self, obs: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            obs_t = torch.from_numpy(obs).to(self.device)
            masks_t = torch.from_numpy(masks).to(self.device)
            if self._half_model is None:
                policies, values = self.model.evaluate(obs_t, masks_t)
            else:
                logits, values = self._half_model(obs_t.half())
                # Neither architecture's public fill fits in float16.
                policies = logits.float().masked_fill(~masks_t, MASK_FILL)
            return policies.float().cpu().numpy(), values.float().cpu().numpy()


class MlxBackend:
    """MLX inference for a :class:`CardTokenNet` whose weights PyTorch owns."""

    name = "mlx"

    def __init__(self, model: torch.nn.Module, *, half: bool = False):
        if not isinstance(model, CardTokenNet):
            raise ValueError("server_backend='mlx' supports only model.arch='card_transformer'")
        try:
            import mlx.core as mx
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError(
                "server_backend='mlx' requires the optional 'mlx' package (pip install mlx)"
            ) from exc
        self._mx = mx
        self.model = model
        self.half = bool(half)
        self._dtype = mx.float16 if self.half else mx.float32
        self._params: dict[str, Any] = {}
        self._compiled: dict[int, Any] = {}
        self.refresh()

    def refresh(self) -> None:
        """Convert the PyTorch weights again; compiled graphs capture them."""

        mx = self._mx
        model = self.model
        state = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
        params = {name: mx.array(value).astype(self._dtype) for name, value in state.items()}
        pointer_heads = ["play_head", *(f"way_heads.{i}" for i in range(MAX_LANDSCAPES)), "buy_head", "select_head", "call_head"]
        params["_pointer.weight"] = mx.concatenate([params[f"{name}.weight"] for name in pointer_heads], axis=0)
        params["_pointer.bias"] = mx.concatenate([params[f"{name}.bias"] for name in pointer_heads], axis=0)
        # Derived features are computed at float32 like the other tokenizer
        # arithmetic, then cast once at the projection.
        params["_def_vp"] = mx.array(model._def_vp.detach().cpu().numpy()).astype(mx.float32)
        mx.eval(params)
        self._params = params
        self._compiled = {}

    def padded_rows(self, count: int, capacity: int) -> int:
        """Round a batch up so the number of distinct compiled shapes stays small."""

        step = 64 if count <= 1024 else 256 if count <= 4096 else 1024
        return min(int(capacity), ((int(count) + step - 1) // step) * step)

    def evaluate(self, obs: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mx = self._mx
        n_piles = (
            active_pile_count_np(obs, self.model.num_card_defs) if self.model.token_trim else MAX_PILES
        )
        fn = self._compiled.get(n_piles)
        if fn is None:
            fn = mx.compile(lambda o, k, _n=n_piles: self._forward(o, k, _n))
            self._compiled[n_piles] = fn
        logits, values = fn(mx.array(obs), mx.array(masks))
        mx.eval(logits, values)
        return np.array(logits), np.array(values)

    # -- the network -------------------------------------------------------

    def _linear(self, name: str, x: Any) -> Any:
        return x @ self._params[f"{name}.weight"].T + self._params[f"{name}.bias"]

    def _layer_norm(self, name: str, x: Any) -> Any:
        mx = self._mx
        mean = x.mean(-1, keepdims=True)
        var = ((x - mean) ** 2).mean(-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + 1e-5) * self._params[f"{name}.weight"] + self._params[f"{name}.bias"]

    def _gelu(self, x: Any) -> Any:
        return x * 0.5 * (1.0 + self._mx.erf(x * (1.0 / math.sqrt(2.0))))

    def _signed_log1p(self, x: Any) -> Any:
        mx = self._mx
        return mx.sign(x) * mx.log1p(mx.abs(x))

    def _log_columns(self, x: Any, columns: list[int]) -> Any:
        """Signed-log1p the listed columns; the PyTorch code edits them in place."""

        selected = np.zeros(x.shape[1], dtype=bool)
        selected[columns] = True
        return self._mx.where(self._mx.array(selected), self._signed_log1p(x), x)

    def _forward(self, obs: Any, legal: Any, piles: int) -> tuple[Any, Any]:
        mx = self._mx
        model = self.model
        P = self._params
        dtype = self._dtype
        v3 = model.obs_version == 3
        d_model = model.d_model
        num_defs = model.num_card_defs
        batch = obs.shape[0]

        # Tokenizer: mirrors CardTokenNet.tokenize, at float32.
        supply = obs[:, OBS_SUPPLY_OFFSET : OBS_SUPPLY_OFFSET + MAX_PILES * SUPPLY_BLOCK_SIZE]
        supply = supply.reshape(batch, MAX_PILES, SUPPLY_BLOCK_SIZE)[:, :piles]
        base_ids_plus_one = mx.round(supply[:, :, 2]).astype(mx.int32)
        def_ids = base_ids_plus_one - 1
        card_mask = (def_ids >= 0) & (def_ids < num_defs)
        embedding_ids = mx.clip(def_ids, 0, num_defs - 1)

        own = obs[:, OBS_OWN_OFFSET : OBS_OWN_OFFSET + 5 * MAX_SLOTS]
        own = own.reshape(batch, 5, MAX_SLOTS)[:, :, :piles].transpose(0, 2, 1)
        opponents = obs[:, OBS_OPPONENT_OFFSET : OBS_OPPONENT_OFFSET + MAX_OPPONENTS * OPPONENT_BLOCK_SIZE_V2]
        opponents = opponents.reshape(batch, MAX_OPPONENTS, OPPONENT_BLOCK_SIZE_V2)
        opponent_slots = mx.stack(
            (opponents[:, :, 6:70], opponents[:, :, 75:139], opponents[:, :, 139:203], opponents[:, :, 203:267]),
            axis=2,
        ).reshape(batch, MAX_OPPONENTS * 4, MAX_SLOTS)[:, :, :piles].transpose(0, 2, 1)
        supply_remaining = supply[:, :, 0:1]
        top_ids_plus_one = mx.round(supply[:, :, 1]).astype(mx.int32)
        top_is_base = ((top_ids_plus_one == base_ids_plus_one) & card_mask).astype(mx.float32)[..., None]
        counts = mx.log1p(mx.maximum(mx.concatenate((own, opponent_slots, supply_remaining), axis=-1), 0))
        card_parts = [counts, top_is_base]
        if v3:
            trash_raw = obs[:, OBS_V3_TRASH_OFFSET : OBS_V3_TRASH_OFFSET + MAX_SLOTS]
            card_parts.append(mx.log1p(mx.maximum(trash_raw[:, :piles], 0))[..., None])
        card_features = mx.concatenate(card_parts, axis=-1).astype(dtype)
        card_tokens = P["def_embedding.weight"][embedding_ids] + self._linear("card_projection", card_features)

        meta = self._log_columns(obs[:, OBS_META_OFFSET : OBS_META_OFFSET + 4], [1, 3])
        resources = self._log_columns(obs[:, OBS_RESOURCE_OFFSET : OBS_RESOURCE_OFFSET + 12], list(range(9)))
        turn = self._log_columns(obs[:, OBS_TURN_OFFSET : OBS_TURN_OFFSET + 10], [7, 9])
        decision = self._log_columns(obs[:, OBS_DECISION_OFFSET : OBS_DECISION_OFFSET + 15], [13, 14])
        landscapes = self._log_columns(obs[:, OBS_LANDSCAPE_OFFSET : OBS_LANDSCAPE_OFFSET + 27], [21])
        global_parts = [meta, resources, turn, decision, landscapes]
        if v3:
            global_parts.append(self._signed_log1p(trash_raw.sum(-1, keepdims=True)))
            global_parts.append(
                obs[:, OBS_V3_SELECT_SEMANTIC_OFFSET : OBS_V3_SELECT_SEMANTIC_OFFSET + SELECT_SEMANTIC_COUNT]
            )
        global_token = self._linear("global_projection", mx.concatenate(global_parts, axis=-1).astype(dtype))
        if v3:
            source_plus_one = mx.round(obs[:, OBS_DECISION_OFFSET + 12]).astype(mx.int32)
            phase_decision = (obs[:, OBS_DECISION_OFFSET + 1 : OBS_DECISION_OFFSET + 4] != 0).any(-1)
            source_present = (source_plus_one != 0) & ~phase_decision
            source_ids = mx.clip(source_plus_one - 1, 0, num_defs - 1)
            global_token = global_token + P["def_embedding.weight"][source_ids] * source_present[:, None].astype(dtype)
        if model.derived_projection is not None:
            mask = card_mask.astype(mx.float32)
            pile_vp = P["_def_vp"][embedding_ids] * mask
            gardens = ((def_ids == DEF_GARDENS) & card_mask).astype(mx.float32)
            own_counts = mx.maximum(own, 0).sum(-1) * mask
            opp_collection = mx.maximum(opponents[:, 0, 75:139][:, :piles], 0) * mask
            own_total = own_counts.sum(-1)
            opp_total = opp_collection.sum(-1)
            own_vp = (own_counts * pile_vp).sum(-1) + (own_counts * gardens).sum(-1) * mx.floor(own_total / 10.0)
            opp_vp = (
                (opp_collection * pile_vp).sum(-1)
                + (opp_collection * gardens).sum(-1) * mx.floor(opp_total / 10.0)
                + opponents[:, 0, 70]
            )
            zone_sizes = (mx.maximum(own, 0) * mask[..., None]).sum(1)
            opp_sizes = mx.maximum(
                mx.stack((opponents[:, 0, 2], opponents[:, 0, 3], opponents[:, 0, 5]), axis=-1), 0
            )
            empty_piles = ((supply[:, :, 0] <= 0) & card_mask).astype(mx.float32).sum(-1)
            derived = mx.concatenate(
                (
                    (own_vp / 20.0)[:, None],
                    (opp_vp / 20.0)[:, None],
                    ((own_vp - opp_vp) / 20.0)[:, None],
                    mx.log1p(own_total)[:, None],
                    mx.log1p(opp_total)[:, None],
                    mx.log1p(zone_sizes),
                    mx.log1p(opp_sizes),
                    (empty_piles / 3.0)[:, None],
                ),
                axis=-1,
            ).astype(dtype)
            global_token = global_token + self._linear("derived_projection", derived)

        # Pre-LN encoder over [global, cards].
        x = mx.concatenate((global_token[:, None, :], card_tokens), axis=1)
        tokens = x.shape[1]
        n_heads = int(model.transformer.layers[0].self_attn.num_heads)
        head_dim = d_model // n_heads
        valid = mx.concatenate((mx.ones((batch, 1), dtype=mx.bool_), card_mask), axis=1)
        # Additive key-padding mask; -3e4 is representable in float16.
        attention_mask = mx.where(valid, 0.0, -3.0e4).astype(dtype)[:, None, None, :]
        for layer in range(len(model.transformer.layers)):
            prefix = f"transformer.layers.{layer}"
            hidden = self._layer_norm(f"{prefix}.norm1", x)
            qkv = hidden @ P[f"{prefix}.self_attn.in_proj_weight"].T + P[f"{prefix}.self_attn.in_proj_bias"]
            q, k, v = (
                part.reshape(batch, tokens, n_heads, head_dim).transpose(0, 2, 1, 3)
                for part in mx.split(qkv, 3, axis=-1)
            )
            attended = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=1.0 / math.sqrt(head_dim), mask=attention_mask
            )
            attended = attended.transpose(0, 2, 1, 3).reshape(batch, tokens, d_model)
            x = x + self._linear(f"{prefix}.self_attn.out_proj", attended)
            hidden = self._layer_norm(f"{prefix}.norm2", x)
            x = x + self._linear(f"{prefix}.linear2", self._gelu(self._linear(f"{prefix}.linear1", hidden)))
        if model.transformer.norm is not None:
            x = self._layer_norm("transformer.norm", x)
        global_out = x[:, 0]
        card_out = x[:, 1:]

        # Policy: eight pointer heads scattered by DefId, plus the flat bands.
        scores = self._linear("_pointer", card_out).astype(mx.float32)  # [B, piles, 8]
        valid_card = card_mask & (def_ids < ACTION_DEF_COUNT)
        # Invalid tokens write into a discarded extra row instead of DefId 0.
        target = mx.where(valid_card, def_ids, ACTION_DEF_COUNT)
        by_def = mx.full((batch, ACTION_DEF_COUNT + 1, scores.shape[-1]), MASK_FILL, dtype=mx.float32)
        by_def = mx.put_along_axis(by_def, mx.broadcast_to(target[..., None], scores.shape), scores, axis=1)
        by_def = by_def[:, :ACTION_DEF_COUNT]
        non_def = self._linear("non_def_head", global_out).astype(mx.float32)
        events_end = 1 + MAX_LANDSCAPES
        options_end = events_end + OPTION_ACTION_COUNT
        ways = [by_def[:, :, 1 + i] for i in range(MAX_LANDSCAPES)]
        logits = mx.concatenate(
            (
                non_def[:, 0:1],
                by_def[:, :, 0],
                *ways,
                by_def[:, :, 1 + MAX_LANDSCAPES],
                non_def[:, 1:events_end],
                by_def[:, :, 2 + MAX_LANDSCAPES],
                non_def[:, events_end:options_end],
                by_def[:, :, 3 + MAX_LANDSCAPES],
                non_def[:, options_end : options_end + SPEND_ACTION_COUNT],
            ),
            axis=-1,
        )
        logits = mx.where(legal, logits, MASK_FILL)

        hidden = self._layer_norm("value_head.0", global_out)
        hidden = self._gelu(self._linear("value_head.1", hidden))
        hidden = self._gelu(self._linear("value_head.3", hidden))
        values = mx.tanh(self._linear("value_head.5", hidden)).astype(mx.float32)[:, 0]
        return logits, values


def build_host_backend(
    model: torch.nn.Module,
    device: torch.device,
    backend: str,
    *,
    half: bool,
) -> TorchHostBackend | MlxBackend | None:
    """Backend for one resident server model, or ``None`` for the built-in paths.

    CUDA and plain CPU serving keep their existing code paths.  ``mps`` always
    goes through a host backend; ``mlx`` additionally keeps the PyTorch model
    on the CPU, where it only holds the weights.
    """

    if backend not in SERVER_BACKENDS:
        raise ValueError(f"server_backend must be one of {SERVER_BACKENDS}, got {backend!r}")
    if backend == "mlx":
        if device.type != "mps":
            raise ValueError("server_backend='mlx' requires server_device='mps'")
        return MlxBackend(model, half=half)
    if device.type == "mps":
        return TorchHostBackend(model, device, half=half)
    return None
