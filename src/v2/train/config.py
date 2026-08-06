from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


SCRIPTED_OPPONENT_KINDS = frozenset({"bigmoney", "engine", "engine3", "random", "scaffold"})
VALUE_TARGET_KINDS = frozenset({"outcome", "winloss", "margin", "margin_blend"})
C_PUCT_SCHEDULES = frozenset({"fixed", "visit_scaled"})
DETERMINIZE_MODES = frozenset({"off", "per_decision", "per_turn"})
TEMP_MODES = frozenset({"legacy", "per_seat_buy"})
OPTIMIZER_KINDS = frozenset({"adamw", "adam"})


@dataclass
class ModelConfig:
    # Explicit new configs self-describe their network while absent metadata
    # in historical checkpoints remains equivalent to this default.
    arch: str = "mlp"
    hidden_sizes: list[int] = field(default_factory=lambda: [1024, 1024, 512])
    input_scale: float = 1.0
    d_model: int = 192
    n_layers: int = 3
    n_heads: int = 4
    ffn_multiplier: int = 4
    dropout: float = 0.0
    # CardTokenNet v2 checkpoints omitted this because its layout was
    # implicit. New transformer checkpoints record it so v3 is unambiguous.
    obs_version: int | None = None
    # None keeps the auxiliary margin-distribution head absent. Enabling
    # aux_margin_weight promotes this to the documented 21-bin default.
    aux_margin_buckets: int | None = None


@dataclass
class SelfPlayConfig:
    n_games: int = 64
    sims_per_move: int = 64
    games_per_generation: int = 64
    # A fraction of mirror self-play games may use the higher search budget.
    # Zero preserves the historical all-normal generation behavior.
    deep_slice_fraction: float = 0.0
    deep_slice_sims: int = 0
    max_batch: int = 512
    c_puct: float = 1.25
    c_puct_schedule: str = "fixed"
    c_puct_init: float = 1.25
    c_puct_base: float = 19652.0
    dirichlet_alpha: float = 0.30
    dirichlet_frac: float = 0.25
    # KataGo-style root forced playouts and target pruning remain opt-in so
    # existing replay/checkpoint behavior is unchanged unless configured.
    forced_playouts: bool = False
    forced_playouts_k: float = 2.0
    # Legacy preserves the global self-play decision clock. Per-seat buy
    # scheduling decouples purchase exploration from effect/action choices.
    temp_moves: int = 20
    temp_mode: str = "legacy"
    temp_buy_turns: int = 14
    temp_action_plies: int = 10
    temp_effect_plies: int = 6
    temp_final: float = 0.0
    # Zero leaves the engine-wide 200-turn cap as the only limit. A positive
    # value caps training self-play only after a completed player turn.
    max_turns: int = 0
    # v1 remains the default so existing campaigns and checkpoints retain
    # their exact model input shape until a run explicitly opts into v2.
    obs_version: int = 1
    kingdom_mode: str = "random"
    fixed_kingdom: list[str] = field(
        default_factory=lambda: [
            "Sentry",
            "Library",
            "Throne Room",
            "Bandit",
            "Witch",
            "Moat",
            "Village",
            "Smithy",
            "Market",
            "Remodel",
        ]
    )
    max_recorded_moves: int = 512
    max_tree_nodes: int = 4096
    scaffold_sims: int = 400
    scaffold_sims_opening: int = 0
    scaffold_determinizations: int = 2
    scripted_threads: int = 2
    value_target: str = "outcome"
    margin_scale: float = 20.0
    margin_blend_alpha: float = 0.6
    auto_play_treasures: bool = False
    prune_treasure_plays: bool = False
    tree_reuse: bool = False
    # Honest hidden-information root sampling is opt-in so older campaign
    # configs and checkpoints retain their original search behavior.
    determinize: str = "off"
    min_new_sims: int = 64
    expand_top_k: int = 0
    # c18 opening-template curriculum. The train loop turns the schedule
    # fields below into the native per-generation lambda/weights values.
    opening_templates_enabled: bool = False
    opening_lambda: float = 0.6
    opening_turn_window: int = 8
    template_weights: list[float] = field(
        default_factory=lambda: [0.3, *(0.7 / 6.0 for _ in range(6))]
    )
    opening_lambda_initial: float = 0.6
    opening_lambda_final: float = 0.0
    opening_anneal_gens: int = 100
    opening_p_unconstrained_initial: float = 0.3
    opening_p_unconstrained_final: float = 1.0


@dataclass
class OptimConfig:
    # AdamW decouples weight decay from Adam's adaptive moments. Set this to
    # "adam" only when reproducing an older coupled-L2 campaign.
    optimizer: str = "adamw"
    lr: float = 1.0e-3
    lr_schedule: str = "constant"
    min_lr: float = 1.0e-5
    step_decay_every: int = 20
    step_decay_gamma: float = 0.5
    weight_decay: float = 1.0e-4
    batch_size: int = 256
    train_steps_per_generation: int = 100


@dataclass
class ImitationConfig:
    """Human-game behavior-cloning and self-play anchor settings.

    The anchor is deliberately a persistent training prior, not a warmup-only
    regularizer.  A nonzero floor is the intended steady-state policy for this
    project: anchor schedules should retain a nonzero final value and must not
    be annealed to zero once human behavior is in use.
    """

    # Directory containing tuple_manifest.json and tuples-*.npz shards.
    human_tuples: str = "exports/tuples"
    # Opt-in so existing campaigns retain their exact unweighted CE path.
    skill_weighting: bool = False
    # Optional manifest-backed filters. Empty lists retain every exported row.
    opponent_kinds: list[str] = field(default_factory=list)
    seat_indices: list[int] = field(default_factory=list)
    # Fresh runs only: these steps occur before self-play generation one.
    pretrain_steps: int = 0
    pretrain_batch_size: int = 256
    pretrain_lr: float | None = None
    # Zero keeps existing campaigns byte-for-byte on the self-play-only path.
    anchor_weight: float = 0.0
    anchor_batch_size: int = 256
    # [generation, weight] breakpoints, linearly interpolated like league
    # schedules. Keep a nonzero final floor; never anneal the anchor to zero.
    anchor_weight_schedule: list = field(default_factory=list)
    # beta=0 is uniform hard-label CE; positive beta enables lightweight AWR.
    anchor_awr_beta: float = 0.0


@dataclass
class ReplayConfig:
    capacity: int = 200_000
    # Self-imitation replay is deliberately dormant unless a campaign opts
    # in. A nonzero weight enables the priority pass and mixed sampler.
    sil_weight: float = 0.0
    sil_fraction: float = 0.25
    sil_alpha: float = 0.6


@dataclass
class EvalConfig:
    eval_every_n_generations: int = 0
    eval_games: int = 200
    eval_sims: int = 400
    eval_opponent: str = "engine"
    eval_kingdoms: str = "random"
    eval_n_games: int = 64
    eval_max_batch: int = 512
    # Keep periodic measurements clairvoyant by default; campaigns can opt
    # into the same per-decision root sampling as evaluate.py --honest.
    eval_honest: bool = False
    # These are deliberately small regression sentinels rather than another
    # training data source. MCTS uses the expensive scaffold, so keep its
    # default game count especially conservative.
    eval_sentinels: list[dict[str, object]] = field(
        default_factory=lambda: [
            {"opponent": "bigmoney", "games": 16},
            {"opponent": "mcts", "games": 4},
        ]
    )


@dataclass
class TrainConfig:
    seed: int = 12345
    generations: int = 10
    # Start a new campaign from model weights only. Unlike --resume, this
    # deliberately does not restore optimizer, replay, RNG, or generation
    # state from the source checkpoint.
    init_weights: str = ""
    device: str = "auto"
    # Parallel collection is opt-in so the legacy single-pipeline run remains
    # exactly deterministic for the default configuration.
    parallel_workers: int = 1
    worker_device: str = "cuda"
    # Route parallel workers through one shared inference process. False keeps
    # the established per-worker eager model replicas for deterministic
    # debugging. ``worker_device="server"`` remains a compatibility alias.
    server_selfplay: bool = False
    # Used only when shared server self-play is enabled. The server owns all
    # resident model inference contexts while workers remain CPU-only runners.
    server_device: str = "cuda"
    # Replicate the shared inference service across this many processes. Each
    # worker is pinned to one shard, so one remains the legacy topology.
    server_shards: int = 1
    server_max_batch: int = 8192
    # Cross-worker serving waits until one model reaches this many live rows,
    # every worker has submitted, or the bounded deadline below expires.
    server_coalesce_target_rows: int = 512
    server_coalesce_ms: float = 4.0
    # Retained for config/checkpoint compatibility; coalescing now uses the
    # explicit deadline above.
    server_max_wait_ms: float = 2.0
    server_fp16: bool = False
    # CUDA-only opt-in: compile the serving evaluation graph with
    # torch.compile(mode="reduce-overhead"). Non-CUDA devices keep eager mode.
    server_compile: bool = False
    # CUDA-only opt-in. If this and server_fp16 are both enabled, bf16 wins.
    # Server responses are always converted back to fp32 before transport.
    server_autocast_bf16: bool = False
    # Optional static forward sizes for CUDA graph-friendly serving. Each live
    # batch is zero-padded to the smallest fitting bucket; larger batches run
    # at their exact size.
    server_batch_buckets: list[int] | None = None
    server_response_timeout_s: float = 30.0
    # Model-table installs legitimately take minutes on first generation
    # (torch.compile + per-bucket warmup for every resident model); liveness
    # is still polled continuously, so a dead server fails fast regardless.
    server_install_timeout_s: float = 900.0
    # Shared memory is the fast path; queue is retained for unsupported hosts.
    server_transport: str = "shm"
    server_shm_slots: int = 2
    # queue blocks on the shared request-header queue; spin polls SHM counters.
    server_poll: str = "queue"
    # Set gate_games to a positive value to enable AlphaGo-style candidate
    # gating. Zero keeps the exact legacy single-network training behavior.
    gate_games: int = 0
    gate_sims: int = 64
    gate_threshold: float = 0.55
    # Gate matches normally use temperature-zero moves. A positive value
    # reuses self-play's seeded temperature sampling for this many moves per
    # player, while still disabling Dirichlet noise in the gate itself.
    gate_temp_moves: int = 0
    # Accept candidates unconditionally for the first N generations so the
    # data pool bootstraps past random play before strict gating engages
    # (strict gating from random init deadlocks — see docs/training-log.md).
    gate_warmup_generations: int = 0
    # Historical-opponent games may be sampled with or without candidate
    # gating. ``league_schedule`` overrides this fixed fraction per generation.
    league_fraction: float = 0.0
    league_pool_size: int = 8
    # Positive values cap a generation's historical opponents to this many
    # weighted, distinct checkpoints. Zero retains the uncapped legacy draw.
    league_opponents_per_gen: int = 3
    # External standard-format checkpoints copied into checkpoint_dir/league/
    # before generation one. They remain standing league opponents alongside
    # archived accepted bests.
    league_seed_checkpoints: list[str] = field(default_factory=list)
    # Every N generations, retain the newly saved candidate checkpoint as a
    # league member. Zero disables this periodic self-checkpoint stream.
    league_self_every: int = 0
    # [generation, fraction] breakpoints, linearly interpolated like the
    # scripted-opponent curriculum. A non-empty schedule overrides
    # ``league_fraction``.
    league_schedule: list = field(default_factory=list)
    # Fixed scripted opponents inject non-mirror game evidence into training.
    # Fractions are per generation and may be used with or without gating.
    scripted_opponents: dict[str, float] = field(default_factory=dict)
    # Per-opponent [generation, fraction] breakpoints override fixed fractions.
    scripted_opponent_schedule: dict[str, list] = field(default_factory=dict)
    # Ordered inclusive generation ranges that temporarily replace the base
    # self-play kingdom distribution. Later overlapping phases win.
    kingdom_curriculum: list = field(default_factory=list)
    # After this many consecutive rejected gated candidates, force-accept the
    # following candidate to refresh a stale self-play lineage. Zero disables
    # this pressure valve.
    gate_force_accept_every: int = 0
    checkpoint_dir: str = "checkpoints"
    metrics_csv: str = "checkpoints/metrics.csv"
    # KataGo-style decomposed terminal-margin supervision. Zero preserves the
    # established scalar-only objective and leaves the head absent by default.
    aux_margin_weight: float = 0.0
    model: ModelConfig = field(default_factory=ModelConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    imitation: ImitationConfig = field(default_factory=ImitationConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_scripted_opponent_kinds(config: TrainConfig) -> None:
    """Reject unknown scripted-opponent keys while loading a campaign config.

    Fraction and breakpoint validation remains generation-aware in
    ``gating.effective_scripted_fractions``. Keeping the vocabulary here makes
    malformed JSON fail at config-load time instead of after model setup.
    """
    for field_name in ("scripted_opponents", "scripted_opponent_schedule"):
        raw = getattr(config, field_name)
        if not isinstance(raw, dict):
            value_description = "fraction" if field_name == "scripted_opponents" else "breakpoints"
            raise ValueError(f"{field_name} must be an object mapping kind to {value_description}")
        for kind in raw:
            if not isinstance(kind, str) or kind not in SCRIPTED_OPPONENT_KINDS:
                allowed = ", ".join(sorted(SCRIPTED_OPPONENT_KINDS))
                raise ValueError(f"unknown scripted opponent {kind!r}; expected one of {allowed}")


def validate_value_target_config(config: SelfPlayConfig) -> None:
    """Validate value-target selection independently of the training path."""
    if not isinstance(config.value_target, str) or config.value_target.lower() not in VALUE_TARGET_KINDS:
        allowed = ", ".join(sorted(VALUE_TARGET_KINDS))
        raise ValueError(f"unknown value target {config.value_target!r}; expected one of {allowed}")
    alpha = config.margin_blend_alpha
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("margin_blend_alpha must be a finite number between zero and one")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("margin_blend_alpha must be a finite number between zero and one")


def validate_c_puct_config(config: SelfPlayConfig) -> None:
    """Validate the Python spelling and numeric parameters for PUCT."""
    if not isinstance(config.c_puct_schedule, str) or config.c_puct_schedule not in C_PUCT_SCHEDULES:
        allowed = ", ".join(sorted(C_PUCT_SCHEDULES))
        raise ValueError(f"unknown c_puct_schedule {config.c_puct_schedule!r}; expected one of {allowed}")
    for field_name in ("c_puct_init", "c_puct_base"):
        value = getattr(config, field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must be finite and positive")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{field_name} must be finite and positive")


def validate_determinize_config(config: SelfPlayConfig) -> None:
    """Validate the training-only self-play root-sampling policy."""
    if not isinstance(config.determinize, str) or config.determinize.lower() not in DETERMINIZE_MODES:
        allowed = ", ".join(sorted(DETERMINIZE_MODES))
        raise ValueError(f"unknown selfplay determinize mode {config.determinize!r}; expected one of {allowed}")


def validate_temperature_config(config: SelfPlayConfig) -> None:
    """Validate the decision-kind-aware self-play temperature schedule."""
    if not isinstance(config.temp_mode, str) or config.temp_mode.lower() not in TEMP_MODES:
        allowed = ", ".join(sorted(TEMP_MODES))
        raise ValueError(f"unknown selfplay temp_mode {config.temp_mode!r}; expected one of {allowed}")
    for field_name in (
        "temp_moves",
        "temp_buy_turns",
        "temp_action_plies",
        "temp_effect_plies",
    ):
        value = getattr(config, field_name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value > 65535
        ):
            raise ValueError(f"{field_name} must be a non-negative integer no greater than 65535")
    value = config.temp_final
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("temp_final must be a finite non-negative number")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError("temp_final must be a finite non-negative number")


def validate_selfplay_max_turns_config(config: SelfPlayConfig) -> None:
    """Validate the optional training-only self-play turn cap."""
    value = config.max_turns
    if not isinstance(value, int) or isinstance(value, bool) or (value != 0 and not 20 <= value <= 200):
        raise ValueError("selfplay.max_turns must be zero or an integer between 20 and 200")


def validate_forced_playouts_config(config: SelfPlayConfig) -> None:
    """Validate opt-in KataGo-style root forced-playout settings."""
    if not isinstance(config.forced_playouts, bool):
        raise ValueError("forced_playouts must be a boolean")
    value = config.forced_playouts_k
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("forced_playouts_k must be a finite positive number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError("forced_playouts_k must be a finite positive number")


def validate_optimizer_kind(optimizer: object) -> str:
    """Return a validated optimizer name used by the Python trainer."""
    if not isinstance(optimizer, str) or optimizer not in OPTIMIZER_KINDS:
        allowed = ", ".join(sorted(OPTIMIZER_KINDS))
        raise ValueError(f"unknown optimizer {optimizer!r}; expected one of {allowed}")
    return optimizer


def validate_optim_config(config: OptimConfig) -> None:
    validate_optimizer_kind(config.optimizer)


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return normalized


def validate_aux_margin_config(config: TrainConfig) -> None:
    """Validate and materialize opt-in margin-head configuration.

    The head is absent unless explicitly requested by a bucket count or a
    positive training weight. A positive weight without a count uses the
    documented default: 21 lower-edge buckets from -20 through +20 in steps
    of two.
    """

    weight = _finite_nonnegative(config.aux_margin_weight, "aux_margin_weight")
    buckets = config.model.aux_margin_buckets
    if buckets is None:
        if weight > 0.0:
            config.model.aux_margin_buckets = 21
            buckets = 21
    elif isinstance(buckets, bool) or not isinstance(buckets, int) or buckets < 2:
        raise ValueError("model.aux_margin_buckets must be null or an integer of at least two")
    if (weight > 0.0 or buckets is not None) and config.model.arch != "card_transformer":
        raise ValueError("auxiliary margin distribution is supported only by model.arch='card_transformer'")


def validate_replay_config(config: ReplayConfig) -> None:
    """Validate optional SIL-style prioritized replay without changing its off path."""
    if not isinstance(config.capacity, int) or isinstance(config.capacity, bool) or config.capacity <= 0:
        raise ValueError("replay.capacity must be a positive integer")
    weight = _finite_nonnegative(config.sil_weight, "replay.sil_weight")
    fraction = _opening_unit_interval(config.sil_fraction, "replay.sil_fraction")
    alpha = _finite_nonnegative(config.sil_alpha, "replay.sil_alpha")
    # Retain normalized values in case a JSON config supplied integral values.
    config.sil_weight = weight
    config.sil_fraction = fraction
    config.sil_alpha = alpha


def effective_anchor_weight(schedule: list, anchor_weight: float, generation: int) -> float:
    """Resolve imitation-anchor breakpoints with league-schedule semantics."""
    fixed_weight = _finite_nonnegative(anchor_weight, "anchor_weight")
    if not isinstance(schedule, list):
        raise ValueError("anchor_weight_schedule must be a list of [generation, weight] pairs")
    if not isinstance(generation, int) or isinstance(generation, bool):
        raise ValueError("generation must be an integer")
    if not schedule:
        return fixed_weight

    breakpoints: list[tuple[int, float]] = []
    previous_generation: int | None = None
    for raw_breakpoint in schedule:
        if not isinstance(raw_breakpoint, list) or len(raw_breakpoint) != 2:
            raise ValueError("anchor_weight_schedule must contain [generation, weight] pairs")
        raw_generation, raw_weight = raw_breakpoint
        if not isinstance(raw_generation, int) or isinstance(raw_generation, bool) or raw_generation < 0:
            raise ValueError("anchor_weight_schedule generation must be an integer at least zero")
        weight = _finite_nonnegative(raw_weight, "anchor_weight_schedule weight")
        if previous_generation is not None and raw_generation <= previous_generation:
            raise ValueError("anchor_weight_schedule generations must be strictly increasing")
        breakpoints.append((raw_generation, weight))
        previous_generation = raw_generation

    if generation <= breakpoints[0][0]:
        return breakpoints[0][1]
    if generation >= breakpoints[-1][0]:
        return breakpoints[-1][1]
    for (start_generation, start_weight), (end_generation, end_weight) in zip(
        breakpoints, breakpoints[1:]
    ):
        if generation <= end_generation:
            progress = (generation - start_generation) / (end_generation - start_generation)
            return start_weight + (end_weight - start_weight) * progress
    raise RuntimeError("anchor weight schedule did not cover its interpolation interval")


def anchor_weight_for_generation(config: ImitationConfig, generation: int) -> float:
    """Return the configured steady-state human-anchor weight for a generation."""
    return effective_anchor_weight(config.anchor_weight_schedule, config.anchor_weight, generation)


def validate_imitation_config(config: ImitationConfig) -> None:
    """Validate optional human imitation settings without touching tuple files."""
    if not isinstance(config.human_tuples, str) or not config.human_tuples:
        raise ValueError("imitation.human_tuples must be a non-empty string path")
    if not isinstance(config.skill_weighting, bool):
        raise ValueError("imitation.skill_weighting must be a boolean")
    if not isinstance(config.opponent_kinds, list) or not all(
        isinstance(kind, str) and kind for kind in config.opponent_kinds
    ):
        raise ValueError("imitation.opponent_kinds must be a list of non-empty strings")
    if not isinstance(config.seat_indices, list) or not all(
        isinstance(seat, int) and not isinstance(seat, bool) and seat >= 0
        for seat in config.seat_indices
    ):
        raise ValueError("imitation.seat_indices must be a list of non-negative integers")
    for field_name in ("pretrain_steps", "pretrain_batch_size", "anchor_batch_size"):
        value = getattr(config, field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if field_name == "pretrain_steps" else 1):
            qualifier = "non-negative" if field_name == "pretrain_steps" else "positive"
            raise ValueError(f"imitation.{field_name} must be a {qualifier} integer")
    if config.pretrain_lr is not None:
        if isinstance(config.pretrain_lr, bool) or not isinstance(config.pretrain_lr, (int, float)):
            raise ValueError("imitation.pretrain_lr must be null or a finite positive number")
        if not math.isfinite(float(config.pretrain_lr)) or float(config.pretrain_lr) <= 0.0:
            raise ValueError("imitation.pretrain_lr must be null or a finite positive number")
    _finite_nonnegative(config.anchor_weight, "anchor_weight")
    _finite_nonnegative(config.anchor_awr_beta, "anchor_awr_beta")
    # Evaluating at generation zero validates the whole breakpoint list.
    anchor_weight_for_generation(config, 0)


def _opening_unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number between zero and one")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be a finite number between zero and one")
    return normalized


def validate_opening_template_config(config: SelfPlayConfig) -> None:
    """Validate native opening fields plus the c18 annealing schedule."""
    if not isinstance(config.opening_templates_enabled, bool):
        raise ValueError("opening_templates_enabled must be a boolean")
    _opening_unit_interval(config.opening_lambda, "opening_lambda")
    if (
        not isinstance(config.opening_turn_window, int)
        or isinstance(config.opening_turn_window, bool)
        or config.opening_turn_window < 0
    ):
        raise ValueError("opening_turn_window must be a non-negative integer")
    weights = config.template_weights
    if not isinstance(weights, (list, tuple)) or len(weights) != 7:
        raise ValueError("template_weights must contain seven non-negative entries")
    total = 0.0
    for weight in weights:
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("template_weights must contain seven non-negative entries")
        normalized = float(weight)
        if not math.isfinite(normalized) or normalized < 0.0:
            raise ValueError("template_weights must contain seven non-negative entries")
        total += normalized
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("template_weights must have positive sum")
    for field_name in (
        "opening_lambda_initial",
        "opening_lambda_final",
        "opening_p_unconstrained_initial",
        "opening_p_unconstrained_final",
    ):
        _opening_unit_interval(getattr(config, field_name), field_name)
    if (
        not isinstance(config.opening_anneal_gens, int)
        or isinstance(config.opening_anneal_gens, bool)
        or config.opening_anneal_gens < 0
    ):
        raise ValueError("opening_anneal_gens must be a non-negative integer")


def opening_template_schedule(config: SelfPlayConfig, generation: int) -> tuple[float, float]:
    """Return (lambda, unconstrained probability) for a generation.

    Generation zero is the initial point. At and after
    ``opening_anneal_gens`` the result is clamped to the configured final
    values; a zero-length schedule is immediately final.
    """
    validate_opening_template_config(config)
    if not isinstance(generation, int) or isinstance(generation, bool):
        raise ValueError("generation must be an integer")
    duration = int(config.opening_anneal_gens)
    if duration == 0:
        progress = 1.0
    else:
        progress = min(1.0, max(0.0, float(generation) / float(duration)))
    opening_lambda = float(config.opening_lambda_initial) + progress * (
        float(config.opening_lambda_final) - float(config.opening_lambda_initial)
    )
    p_unconstrained = float(config.opening_p_unconstrained_initial) + progress * (
        float(config.opening_p_unconstrained_final)
        - float(config.opening_p_unconstrained_initial)
    )
    return opening_lambda, p_unconstrained


def scheduled_opening_selfplay_config(config: SelfPlayConfig, generation: int) -> SelfPlayConfig:
    """Copy a config with c18's generation-specific native settings applied."""
    scheduled = replace(config)
    if not scheduled.opening_templates_enabled:
        return scheduled
    opening_lambda, p_unconstrained = opening_template_schedule(scheduled, generation)
    scheduled.opening_lambda = opening_lambda
    archetype_weights = [float(weight) for weight in scheduled.template_weights[1:]]
    archetype_total = sum(archetype_weights)
    if archetype_total > 0.0:
        archetype_scale = (1.0 - p_unconstrained) / archetype_total
        scheduled.template_weights = [
            p_unconstrained,
            *(weight * archetype_scale for weight in archetype_weights),
        ]
    else:
        # A zeroed archetype distribution cannot be rescaled. Keep the
        # schedule usable by distributing its non-unconstrained mass evenly.
        scheduled.template_weights = [
            p_unconstrained,
            *((1.0 - p_unconstrained) / 6.0 for _ in range(6)),
        ]
    return scheduled


def _merge_dataclass(instance: Any, data: dict[str, Any]) -> Any:
    for key, value in data.items():
        if key.startswith("_comment"):
            continue
        if not hasattr(instance, key):
            raise ValueError(f"unknown config key: {key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def validate_deep_slice_config(config: SelfPlayConfig) -> None:
    """Validate the optional higher-budget self-play slice.

    Deep search is intentionally configured alongside the ordinary self-play
    search settings because it creates another runner of the same native type,
    rather than a separate data source.
    """
    fraction = config.deep_slice_fraction
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise ValueError("deep_slice_fraction must be numeric")
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("deep_slice_fraction must be between zero and one")
    if fraction == 0.0:
        return

    deep_sims = config.deep_slice_sims
    if not isinstance(deep_sims, int) or isinstance(deep_sims, bool):
        raise ValueError("deep_slice_sims must be an integer when deep_slice_fraction is positive")
    if deep_sims <= int(config.sims_per_move):
        raise ValueError("deep_slice_sims must exceed sims_per_move when deep_slice_fraction is positive")


def validate_server_shards(config: TrainConfig) -> None:
    """Validate the shared-server process count before process startup."""
    shards = config.server_shards
    if not isinstance(shards, int) or isinstance(shards, bool) or shards <= 0:
        raise ValueError("server_shards must be a positive integer")


def load_config(path: str | Path | None) -> TrainConfig:
    cfg = TrainConfig()
    if path is None:
        return cfg
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("config root must be an object")
    _merge_dataclass(cfg, data)
    validate_scripted_opponent_kinds(cfg)
    validate_value_target_config(cfg.selfplay)
    validate_c_puct_config(cfg.selfplay)
    validate_determinize_config(cfg.selfplay)
    validate_temperature_config(cfg.selfplay)
    validate_selfplay_max_turns_config(cfg.selfplay)
    validate_forced_playouts_config(cfg.selfplay)
    validate_optim_config(cfg.optim)
    validate_aux_margin_config(cfg)
    validate_replay_config(cfg.replay)
    validate_imitation_config(cfg.imitation)
    validate_deep_slice_config(cfg.selfplay)
    validate_opening_template_config(cfg.selfplay)
    validate_server_shards(cfg)
    return cfg


def save_config(config: TrainConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init-weights", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--server-shards", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--smoke", action="store_true")
