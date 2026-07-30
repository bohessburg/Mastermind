"""Card-token policy/value network for v2/v3 Dominion observations.

The versioned encoder deliberately remains a fixed-width C++ ABI. This module
turns a flat observation back into per-supply-card tokens in PyTorch, using
the invariant that a base-set pile's index is also its Slot. The base DefId
stored in each supply block is therefore the per-sample Slot -> DefId mapping;
it must not be assumed to be a fixed Slot -> DefId table.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


# These mirror the current C++ ABI in encode/{encoder.h,layout.md} and
# core/actions.h.  Keeping them here avoids making model-only code depend on
# the optional native Python bindings at import time.
OBS_SIZE_V2 = 1717
OBS_SIZE_V3 = 1788
MAX_SLOTS = 64
MAX_PILES = 48
MAX_OPPONENTS = 3
OPPONENT_BLOCK_SIZE_V2 = 267
SUPPLY_BLOCK_SIZE = 11
ACTION_DEF_COUNT = 41
MAX_LANDSCAPES = 4
OPTION_ACTION_COUNT = 16
SPEND_ACTION_COUNT = 8

OBS_META_OFFSET = 0
OBS_OWN_OFFSET = 4
OBS_OPPONENT_OFFSET = 324
OBS_SUPPLY_OFFSET = 1125
OBS_LANDSCAPE_OFFSET = 1653
OBS_RESOURCE_OFFSET = 1680
OBS_TURN_OFFSET = 1692
OBS_DECISION_OFFSET = 1702
OBS_V3_TRASH_OFFSET = OBS_SIZE_V2
OBS_V3_SELECT_SEMANTIC_OFFSET = OBS_V3_TRASH_OFFSET + MAX_SLOTS
SELECT_SEMANTIC_COUNT = 7

A_PASS = 0
A_PLAY_BASE = 1
A_WAY_BASE = A_PLAY_BASE + ACTION_DEF_COUNT
A_BUY_BASE = A_WAY_BASE + (MAX_LANDSCAPES * ACTION_DEF_COUNT)
A_EVENT_BASE = A_BUY_BASE + ACTION_DEF_COUNT
A_SELECT_BASE = A_EVENT_BASE + MAX_LANDSCAPES
A_OPTION_BASE = A_SELECT_BASE + ACTION_DEF_COUNT
A_CALL_BASE = A_OPTION_BASE + OPTION_ACTION_COUNT
A_SPEND_BASE = A_CALL_BASE + ACTION_DEF_COUNT
ACTION_SPACE_SIZE = A_SPEND_BASE + SPEND_ACTION_COUNT

# The auxiliary score-margin distribution covers the same clipped score range
# as the established scalar margin target. For the default 21 heads, bucket i
# is the lower-edge pair [-20 + 2*i, -19 + 2*i], with the final +20 endpoint
# retained in bucket 20 after clamping. Thus -20/-1/0/+1/+20 map to
# 0/9/10/10/20 respectively.
AUX_MARGIN_MIN = -20
AUX_MARGIN_MAX = 20
DEFAULT_AUX_MARGIN_BUCKETS = 21


def margin_bucket_ids(margins: torch.Tensor, n_buckets: int = DEFAULT_AUX_MARGIN_BUCKETS) -> torch.Tensor:
    """Map signed terminal margins to clamped lower-edge distribution bins.

    The default 21-bin geometry has edges ``[-20, -18, ..., 20]``. Integer
    margins are clamped to ``[-20, 20]`` then assigned by lower edge, so the
    central bucket covers ``[0, 1]`` and the final endpoint remains bucket 20.
    Other configured counts retain the same inclusive range with evenly
    scaled lower-edge IDs.
    """

    if isinstance(n_buckets, bool) or not isinstance(n_buckets, int) or n_buckets < 2:
        raise ValueError("aux_margin_buckets must be an integer of at least two")
    clipped = margins.to(dtype=torch.long).clamp(min=AUX_MARGIN_MIN, max=AUX_MARGIN_MAX)
    return ((clipped - AUX_MARGIN_MIN) * (n_buckets - 1)) // (AUX_MARGIN_MAX - AUX_MARGIN_MIN)


@dataclass(frozen=True)
class TokenizedCards:
    """Intermediate tokenizer output, kept public for model-side inspection.

    ``card_features`` contains the values actually sent through the card
    feature projection: count fields are log1p transformed and the
    top-is-base field is binary. In v3, the appended final scalar is the
    log1p trash count. ``def_ids`` contains native zero-based IDs;
    inactive supply blocks have ``-1`` and are masked from attention.
    """

    card_tokens: torch.Tensor
    global_token: torch.Tensor
    card_features: torch.Tensor
    global_features: torch.Tensor
    def_ids: torch.Tensor
    card_mask: torch.Tensor


class CardTokenNet(nn.Module):
    """A pre-LN transformer over active supply-card tokens plus one global token.

    Its ``forward`` and ``evaluate`` methods intentionally match
    :class:`DominionNet`: observations are batched float tensors and outputs
    are ``([B, ACTION_SPACE_SIZE], [B])`` policy/value tensors on the same
    device.  No observation-wide input scaling is used; count fields are
    featurized locally with log1p instead.
    """

    # Public indices make tokenizer assertions and diagnostics unambiguous.
    OWN_HAND = 0
    OWN_DECK = 1
    OWN_DISCARD = 2
    OWN_IN_PLAY = 3
    OWN_SET_ASIDE = 4
    OPPONENT_FEATURE_OFFSET = 5
    SUPPLY_REMAINING = OWN_SET_ASIDE + 1 + (MAX_OPPONENTS * 4)
    TOP_IS_BASE = SUPPLY_REMAINING + 1
    # Keep all v2 feature indices and widths fixed so v2 state dicts retain
    # their exact projection shapes. v3 appends its trash scalar after them.
    CARD_FEATURE_SIZE_V2 = TOP_IS_BASE + 1
    TRASH_COUNT = CARD_FEATURE_SIZE_V2
    CARD_FEATURE_SIZE_V3 = TRASH_COUNT + 1
    CARD_FEATURE_SIZE = CARD_FEATURE_SIZE_V2
    GLOBAL_FEATURE_SIZE_V2 = 4 + 12 + 10 + 15 + 27
    GLOBAL_TRASH_TOTAL = GLOBAL_FEATURE_SIZE_V2
    GLOBAL_SELECT_SEMANTIC_OFFSET = GLOBAL_TRASH_TOTAL + 1
    GLOBAL_FEATURE_SIZE_V3 = GLOBAL_SELECT_SEMANTIC_OFFSET + SELECT_SEMANTIC_COUNT
    GLOBAL_FEATURE_SIZE = GLOBAL_FEATURE_SIZE_V2
    POINTER_FILL = -1.0e9

    def __init__(
        self,
        obs_size: int = OBS_SIZE_V2,
        action_size: int = ACTION_SPACE_SIZE,
        *,
        d_model: int = 192,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        num_card_defs: int = ACTION_DEF_COUNT,
        action_def_count: int = ACTION_DEF_COUNT,
        obs_version: int | None = None,
        aux_margin_buckets: int | None = None,
    ):
        super().__init__()
        if int(obs_size) == OBS_SIZE_V2:
            inferred_obs_version = 2
        elif int(obs_size) == OBS_SIZE_V3:
            inferred_obs_version = 3
        else:
            raise ValueError(
                f"CardTokenNet requires v2 ({OBS_SIZE_V2}) or v3 ({OBS_SIZE_V3}) observations, got {obs_size}"
            )
        if obs_version is not None and int(obs_version) != inferred_obs_version:
            raise ValueError(
                f"CardTokenNet obs_version {obs_version} does not match observation size {obs_size}"
            )
        if d_model <= 0 or n_layers <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads; n_layers/n_heads must be positive")
        if ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if num_card_defs <= 0 or action_def_count <= 0:
            raise ValueError("card definition counts must be positive")
        if action_def_count != ACTION_DEF_COUNT:
            raise ValueError(
                f"current action ABI has {ACTION_DEF_COUNT} per-def actions, got {action_def_count}"
            )
        if int(action_size) != ACTION_SPACE_SIZE:
            raise ValueError(f"CardTokenNet requires action size {ACTION_SPACE_SIZE}, got {action_size}")
        if aux_margin_buckets is not None and (
            isinstance(aux_margin_buckets, bool)
            or not isinstance(aux_margin_buckets, int)
            or aux_margin_buckets < 2
        ):
            raise ValueError("aux_margin_buckets must be null or an integer of at least two")

        self.obs_size = int(obs_size)
        self.obs_version = inferred_obs_version
        self.action_size = int(action_size)
        self.d_model = int(d_model)
        self.num_card_defs = int(num_card_defs)
        self.action_def_count = int(action_def_count)
        self.aux_margin_buckets = None if aux_margin_buckets is None else int(aux_margin_buckets)

        self.card_feature_size = (
            self.CARD_FEATURE_SIZE_V3 if self.obs_version == 3 else self.CARD_FEATURE_SIZE_V2
        )
        self.global_feature_size = (
            self.GLOBAL_FEATURE_SIZE_V3 if self.obs_version == 3 else self.GLOBAL_FEATURE_SIZE_V2
        )

        self.def_embedding = nn.Embedding(self.num_card_defs, self.d_model)
        self.card_projection = nn.Linear(self.card_feature_size, self.d_model)
        self.global_projection = nn.Linear(self.global_feature_size, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=self.d_model * int(ffn_multiplier),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Nested-tensor conversion cannot use the pre-LN fast path and gives
        # no benefit for our fixed 49-token batches.  Disable it explicitly to
        # keep execution behaviour stable across PyTorch versions.
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(n_layers),
            enable_nested_tensor=False,
        )

        # One scalar head per card-indexed action family.  Ways have four
        # action bands in the flat action ABI, so they use four family heads.
        self.play_head = nn.Linear(self.d_model, 1)
        self.way_heads = nn.ModuleList(nn.Linear(self.d_model, 1) for _ in range(MAX_LANDSCAPES))
        self.buy_head = nn.Linear(self.d_model, 1)
        self.select_head = nn.Linear(self.d_model, 1)
        self.call_head = nn.Linear(self.d_model, 1)

        # The only flat non-def bands are PASS, events, options, and spends.
        self.non_def_head = nn.Linear(
            self.d_model,
            1 + MAX_LANDSCAPES + OPTION_ACTION_COUNT + SPEND_ACTION_COUNT,
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
            nn.Tanh(),
        )
        # Deliberately omit this module entirely for absent config so legacy
        # CardTokenNet state_dicts and forward behavior remain byte-identical.
        self.aux_margin_head = (
            nn.Linear(self.d_model, self.aux_margin_buckets)
            if self.aux_margin_buckets is not None
            else None
        )

    @staticmethod
    def _signed_log1p(values: torch.Tensor) -> torch.Tensor:
        """Log-scale potentially signed global counters without losing sign."""

        return values.sign() * torch.log1p(values.abs())

    def _global_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract the documented non-card sections with local featurization."""

        meta = obs[:, OBS_META_OFFSET : OBS_META_OFFSET + 4].clone()
        # Version and player ID are tiny categorical scalars.  Shape and slot
        # counts are scalar magnitudes and receive log1p conditioning.
        meta[:, 1] = self._signed_log1p(meta[:, 1])
        meta[:, 3] = self._signed_log1p(meta[:, 3])

        resources = obs[:, OBS_RESOURCE_OFFSET : OBS_RESOURCE_OFFSET + 12].clone()
        resources[:, :9] = self._signed_log1p(resources[:, :9])

        turn = obs[:, OBS_TURN_OFFSET : OBS_TURN_OFFSET + 10].clone()
        # phase[0:5], current-player-is-perspective, and truncated are already
        # one-hot/binary.  Turn count and effect stack depth are counts.
        turn[:, 7] = self._signed_log1p(turn[:, 7])
        turn[:, 9] = self._signed_log1p(turn[:, 9])

        decision = obs[:, OBS_DECISION_OFFSET : OBS_DECISION_OFFSET + 15].clone()
        # decision kind is a one-hot, source/player are IDs, and min/max are
        # the only count-like values in this section.
        decision[:, 13:15] = self._signed_log1p(decision[:, 13:15])

        landscapes = obs[:, OBS_LANDSCAPE_OFFSET : OBS_LANDSCAPE_OFFSET + 27].clone()
        # IDs and bought flags stay as encoded; only sun tokens are a count.
        landscapes[:, 21] = self._signed_log1p(landscapes[:, 21])

        features = torch.cat((meta, resources, turn, decision, landscapes), dim=-1)
        if self.obs_version == 2:
            return features

        trash = obs[:, OBS_V3_TRASH_OFFSET : OBS_V3_TRASH_OFFSET + MAX_SLOTS]
        trash_total = self._signed_log1p(trash.sum(dim=-1, keepdim=True))
        semantic = obs[
            :, OBS_V3_SELECT_SEMANTIC_OFFSET : OBS_V3_SELECT_SEMANTIC_OFFSET + SELECT_SEMANTIC_COUNT
        ]
        return torch.cat((features, trash_total, semantic), dim=-1)

    def _v3_decision_source_embedding(self, obs: torch.Tensor) -> torch.Tensor:
        """Embed the v3 decision source, with no-card represented by zero."""

        source_plus_one = obs[:, OBS_DECISION_OFFSET + 12].round().to(torch.long)
        source_present = source_plus_one != 0
        # The frozen v2 prefix predates source-less phase decisions and
        # serializes their default source value (0) as ``def + 1 == 1``.
        # Phase decisions intrinsically have no source card, so mask them
        # explicitly without altering any v2 feature values.
        phase_decision = obs[:, OBS_DECISION_OFFSET + 1 : OBS_DECISION_OFFSET + 4].bool().any(dim=-1)
        source_present = source_present & ~phase_decision
        source_ids = (source_plus_one - 1).clamp(min=0, max=self.num_card_defs - 1)
        embedded = self.def_embedding(source_ids)
        return embedded * source_present.unsqueeze(-1).to(dtype=embedded.dtype)

    def tokenize(self, obs: torch.Tensor) -> TokenizedCards:
        """Recover active card tokens directly from a batched v2/v3 observation.

        Supply block ``k`` provides the base DefId for Slot ``k``.  We gather
        all Slot-indexed own/opponent composition fields by that pile index,
        rather than relying on a global card-ID order.
        """

        if obs.ndim != 2 or obs.shape[1] != self.obs_size:
            raise ValueError(f"expected obs with shape [B, {self.obs_size}], got {tuple(obs.shape)}")

        batch_size = obs.shape[0]
        supply = obs[:, OBS_SUPPLY_OFFSET : OBS_SUPPLY_OFFSET + (MAX_PILES * SUPPLY_BLOCK_SIZE)]
        supply = supply.reshape(batch_size, MAX_PILES, SUPPLY_BLOCK_SIZE)

        # Encoder DefIds are stored as def + 1.  Round first so tokenization
        # remains robust to float32 transport while retaining the integer ABI.
        base_ids_plus_one = supply[:, :, 2].round().to(torch.long)
        def_ids = base_ids_plus_one - 1
        card_mask = (def_ids >= 0) & (def_ids < self.num_card_defs)
        embedding_ids = def_ids.clamp(min=0, max=self.num_card_defs - 1)

        own = obs[:, OBS_OWN_OFFSET : OBS_OWN_OFFSET + (5 * MAX_SLOTS)]
        own = own.reshape(batch_size, 5, MAX_SLOTS)[:, :, :MAX_PILES].transpose(1, 2)

        opponents = obs[
            :, OBS_OPPONENT_OFFSET : OBS_OPPONENT_OFFSET + (MAX_OPPONENTS * OPPONENT_BLOCK_SIZE_V2)
        ].reshape(batch_size, MAX_OPPONENTS, OPPONENT_BLOCK_SIZE_V2)
        # Four slot-indexed fields per opponent: in-play, known collection,
        # known discard, and known set-aside.  Preserve opponent order instead
        # of collapsing it, so a two-player game simply has zero padding.
        opponent_slots = torch.stack(
            (
                opponents[:, :, 6:70],
                opponents[:, :, 75:139],
                opponents[:, :, 139:203],
                opponents[:, :, 203:267],
            ),
            dim=2,
        ).reshape(batch_size, MAX_OPPONENTS * 4, MAX_SLOTS)[:, :, :MAX_PILES].transpose(1, 2)

        supply_remaining = supply[:, :, 0:1]
        top_ids_plus_one = supply[:, :, 1].round().to(torch.long)
        top_is_base = ((top_ids_plus_one == base_ids_plus_one) & card_mask).to(dtype=obs.dtype).unsqueeze(-1)

        raw_count_features = torch.cat((own, opponent_slots, supply_remaining), dim=-1)
        count_features = torch.log1p(raw_count_features.clamp_min(0))
        if self.obs_version == 2:
            # Retain the exact v2 tokenizer path and feature order for
            # existing checkpoints.
            card_features = torch.cat((count_features, top_is_base), dim=-1)
        else:
            # The trash composition is Slot-indexed like own-zone counts, so
            # supply token k gathers trash count for Slot k.
            trash = obs[:, OBS_V3_TRASH_OFFSET : OBS_V3_TRASH_OFFSET + MAX_SLOTS]
            trash = torch.log1p(trash[:, :MAX_PILES].clamp_min(0)).unsqueeze(-1)
            card_features = torch.cat((count_features, top_is_base, trash), dim=-1)

        card_tokens = self.def_embedding(embedding_ids) + self.card_projection(card_features)
        global_features = self._global_features(obs)
        global_token = self.global_projection(global_features)
        if self.obs_version == 3:
            global_token = global_token + self._v3_decision_source_embedding(obs)
        global_token = global_token.unsqueeze(1)
        return TokenizedCards(
            card_tokens=card_tokens,
            global_token=global_token,
            card_features=card_features,
            global_features=global_features,
            def_ids=def_ids,
            card_mask=card_mask,
        )

    def encode_tokens(self, tokenized: TokenizedCards) -> torch.Tensor:
        """Run global + card tokens through the pre-LN set transformer."""

        tokens = torch.cat((tokenized.global_token, tokenized.card_tokens), dim=1)
        global_mask = torch.ones(
            (tokenized.card_mask.shape[0], 1),
            dtype=torch.bool,
            device=tokenized.card_mask.device,
        )
        valid_tokens = torch.cat((global_mask, tokenized.card_mask), dim=1)
        return self.transformer(tokens, src_key_padding_mask=~valid_tokens)

    def _pointer_logits(
        self,
        scores: torch.Tensor,
        tokenized: TokenizedCards,
    ) -> torch.Tensor:
        """Scatter card-token scores into the current per-DefId action band."""

        batch_size = scores.shape[0]
        by_def = scores.new_full((batch_size, self.action_def_count), self.POINTER_FILL)
        action_def_ids = tokenized.def_ids.clamp(min=0, max=self.action_def_count - 1)
        valid = tokenized.card_mask & (tokenized.def_ids < self.action_def_count)
        source = scores.masked_fill(~valid, self.POINTER_FILL)
        # A normal game has one pile/token per DefId.  amax also defines a
        # deterministic outcome should a malformed observation repeat a DefId.
        by_def.scatter_reduce_(1, action_def_ids, source, reduce="amax", include_self=True)
        return by_def

    def policy_logits(self, encoded_tokens: torch.Tensor, tokenized: TokenizedCards) -> torch.Tensor:
        """Assemble pointer and global heads into the native flat action ABI."""

        global_token = encoded_tokens[:, 0]
        card_tokens = encoded_tokens[:, 1:]
        non_def = self.non_def_head(global_token)
        offset = 0
        pass_logits = non_def[:, offset : offset + 1]
        offset += 1
        event_logits = non_def[:, offset : offset + MAX_LANDSCAPES]
        offset += MAX_LANDSCAPES
        option_logits = non_def[:, offset : offset + OPTION_ACTION_COUNT]
        offset += OPTION_ACTION_COUNT
        spend_logits = non_def[:, offset : offset + SPEND_ACTION_COUNT]

        play_logits = self._pointer_logits(self.play_head(card_tokens).squeeze(-1), tokenized)
        way_logits = [
            self._pointer_logits(head(card_tokens).squeeze(-1), tokenized) for head in self.way_heads
        ]
        buy_logits = self._pointer_logits(self.buy_head(card_tokens).squeeze(-1), tokenized)
        select_logits = self._pointer_logits(self.select_head(card_tokens).squeeze(-1), tokenized)
        call_logits = self._pointer_logits(self.call_head(card_tokens).squeeze(-1), tokenized)

        logits = torch.cat(
            (
                pass_logits,
                play_logits,
                *way_logits,
                buy_logits,
                event_logits,
                select_logits,
                option_logits,
                call_logits,
                spend_logits,
            ),
            dim=-1,
        )
        if logits.shape[-1] != self.action_size:  # defensive ABI assertion
            raise RuntimeError(f"assembled {logits.shape[-1]} logits, expected {self.action_size}")
        return logits

    def forward_tokenized(self, tokenized: TokenizedCards) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a previously tokenized batch (useful for diagnostics/tests)."""

        encoded_tokens = self.encode_tokens(tokenized)
        return self.policy_logits(encoded_tokens, tokenized), self.value_head(encoded_tokens[:, 0]).squeeze(-1)

    def forward_tokenized_with_aux(
        self,
        tokenized: TokenizedCards,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate policy, scalar value, and configured margin logits together."""

        if self.aux_margin_head is None:
            raise RuntimeError("forward_with_aux requires configured aux_margin_buckets")
        encoded_tokens = self.encode_tokens(tokenized)
        global_token = encoded_tokens[:, 0]
        return (
            self.policy_logits(encoded_tokens, tokenized),
            self.value_head(global_token).squeeze(-1),
            self.aux_margin_head(global_token),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_tokenized(self.tokenize(obs))

    def forward_with_aux(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Opt-in auxiliary output without changing ``forward``'s ABI."""

        return self.forward_tokenized_with_aux(self.tokenize(obs))

    @torch.no_grad()
    def evaluate(self, obs: torch.Tensor, legal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = self(obs)
        masked_logits = logits.masked_fill(~legal_mask, self.POINTER_FILL)
        return masked_logits, values
