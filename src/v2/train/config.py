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
    temp_moves: int = 12
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
    lr: float = 1.0e-3
    lr_schedule: str = "constant"
    min_lr: float = 1.0e-5
    step_decay_every: int = 20
    step_decay_gamma: float = 0.5
    weight_decay: float = 1.0e-4
    batch_size: int = 256
    train_steps_per_generation: int = 100


@dataclass
class ReplayConfig:
    capacity: int = 200_000


@dataclass
class EvalConfig:
    eval_every_n_generations: int = 0
    eval_games: int = 200
    eval_sims: int = 400
    eval_opponent: str = "engine"
    eval_kingdoms: str = "random"
    eval_n_games: int = 64
    eval_max_batch: int = 512
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
    model: ModelConfig = field(default_factory=ModelConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
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
    validate_deep_slice_config(cfg.selfplay)
    validate_opening_template_config(cfg.selfplay)
    return cfg


def save_config(config: TrainConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init-weights", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--smoke", action="store_true")
