"""Checkpoint gating and deterministic mini-league bookkeeping.

The trainer keeps a candidate network (and its optimizer) separate from the
accepted best network.  League assignment is deliberately derived only from
the configured seed and generation: parallel workers may finish in a different
order, but the games assigned to each opponent remain reproducible.
"""

from __future__ import annotations

import copy
import math
import random
import re
import shutil
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import torch

from src.v2.encoder_compat import require_native_runner_encoder_compatibility

from .config import (
    SCRIPTED_OPPONENT_KINDS,
    SelfPlayConfig,
    TrainConfig,
    validate_deep_slice_config,
)
from .model import build_model, model_config_dict
from .observation import obs_size_for_config, obs_size_for_version, obs_version_for_checkpoint


BEST_FILENAME = "best.pt"
LEAGUE_DIRNAME = "league"
@dataclass(frozen=True)
class GateStats:
    wins: int
    losses: int
    ties: int

    @property
    def decisive(self) -> int:
        return self.wins + self.losses

    @property
    def win_pct(self) -> float:
        return self.wins / self.decisive if self.decisive else 0.0


@dataclass(frozen=True)
class SeatSwappedMatch:
    """Results and records from a deterministic two-seat NN match.

    ``records`` retain the original runner records so standalone consumers can
    inspect completed games without growing a second NN-vs-NN game loop.  The
    corresponding entry in ``a_players`` identifies the seat occupied by the
    first model for that record.
    """

    stats: GateStats
    records: tuple[dict[str, Any], ...]
    a_players: tuple[int, ...]
    wall_time: float


@dataclass(frozen=True)
class LeagueGame:
    """One self-play game using a historical opponent, or ``None`` if normal."""

    opponent_index: int
    best_player: int


@dataclass(frozen=True)
class SelfPlaySegment:
    """A contiguous pool work item for NN or scripted-opponent self-play."""

    n_games: int
    seat0_model_id: int
    seat1_model_id: int
    scripted_kind: str | None = None
    nn_player: int = 0
    # ``None`` means full random kingdom pool (or the configured fixed
    # kingdom); a non-empty list contains native DefIds for this segment.
    kingdom_pool: list[int] | None = None
    # ``None`` preserves SelfPlayConfig.kingdom_mode. Curriculum segments set
    # this explicitly so a random phase can override a fixed base campaign.
    kingdom_mode: str | None = None
    # Zero uses SelfPlayConfig.sims_per_move; a positive value creates a
    # higher-budget runner for this segment only.
    sims_override: int = 0
    # Checkpoint basename for metrics and strength-matched league sampling.
    # It is set only on true two-model league segments.
    league_opponent: str | None = None
    # Assigned after the final segment plan is composed.  It identifies the
    # first global game covered by this contiguous segment and survives worker
    # splitting, so a slot's seed never depends on which worker owns it.
    game_index: int | None = None

    @property
    def is_league(self) -> bool:
        return not self.is_scripted and self.seat0_model_id != self.seat1_model_id

    @property
    def is_scripted(self) -> bool:
        return self.scripted_kind is not None

    @property
    def is_normal_mirror(self) -> bool:
        """True only for ordinary current-best-versus-current-best games."""
        return not self.is_scripted and self.seat0_model_id == self.seat1_model_id == 0


@dataclass(frozen=True)
class KingdomCurriculumPhase:
    """The validated kingdom distribution active for one generation."""

    mode: str
    pool: tuple[int, ...] = ()
    pool_names: tuple[str, ...] = ()
    pool_fraction: float = 1.0
    start_generation: int | None = None
    end_generation: int | None = None

    @property
    def label(self) -> str:
        if self.start_generation is None:
            return f"base:{self.mode}"
        prefix = f"{self.start_generation}-{self.end_generation}:{self.mode}"
        if self.mode == "pool":
            return f"{prefix}:{self.pool_fraction:.6f}:{','.join(self.pool_names)}"
        return prefix


def gating_enabled(config: TrainConfig) -> bool:
    return int(config.gate_games) > 0


def gate_result(stats: GateStats, threshold: float) -> str:
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("gate_threshold must be between zero and one")
    return "accepted" if stats.decisive > 0 and stats.win_pct >= float(threshold) else "rejected"


def best_checkpoint_path(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / BEST_FILENAME


def league_directory(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / LEAGUE_DIRNAME


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _checkpoint_model_config(payload: object, path: str | Path) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"league checkpoint {path} must contain an object payload")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"league checkpoint {path} has a non-object 'config' payload")
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"league checkpoint {path} is missing config.model metadata")
    return dict(model)


def _load_checkpoint_payload(path: str | Path, device: torch.device) -> dict:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older supported Torch versions
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"league checkpoint {path} must contain an object payload")
    return payload


def _validate_league_checkpoint_observation(
    config: TrainConfig,
    payload: dict,
    path: str | Path,
) -> int:
    """Validate a league checkpoint and return its model observation version.

    The config's saved ``obs_version`` and the model's own layout are
    independent checks. MLPs expose that layout through their first-layer
    width; CardTokenNet records its v2/v3 layout in model metadata.  The one
    intentional cross-version pairing is a v2 model in a v3 runner: v3's
    byte-identical v2 prefix is sliced and its metadata is rewritten before
    that model evaluates it.  V1 remains incompatible with both v2 and v3.
    """
    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"league checkpoint {path} has a non-object 'config' payload")
    checkpoint_selfplay = checkpoint_config.get("selfplay")
    if not isinstance(checkpoint_selfplay, dict) or "obs_version" not in checkpoint_selfplay:
        raise ValueError(
            f"league checkpoint {path} is missing config.selfplay.obs_version; "
            "cannot validate its observation pipeline"
        )
    stored_version = checkpoint_selfplay["obs_version"]
    if not isinstance(stored_version, int) or isinstance(stored_version, bool):
        raise ValueError(f"league checkpoint {path} has an invalid stored obs_version {stored_version!r}")
    try:
        stored_width = obs_size_for_version(stored_version)
        model_config = _checkpoint_model_config(payload, path)
        arch = model_config.get("arch", "mlp")
        if arch == "card_transformer":
            configured_model_version = model_config.get("obs_version")
            model_version = 2 if configured_model_version is None else int(configured_model_version)
        elif arch == "mlp":
            model_version = obs_version_for_checkpoint(payload)
        else:
            raise ValueError(f"league checkpoint {path} has unknown model.arch {arch!r}")
        model_width = obs_size_for_version(model_version)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"league checkpoint {path} has an invalid observation input width: {exc}") from exc

    required_version = int(config.selfplay.obs_version)
    required_width = obs_size_for_config(config)
    if (
        stored_version != model_version
        or stored_width != model_width
        or not (
            stored_version == required_version
            or (required_version == 3 and stored_version == 2)
        )
    ):
        raise ValueError(
            f"league checkpoint {path} has stored obs_version {stored_version} and input width {model_width} "
            f"(model layout v{model_version}); current run requires obs_version {required_version} "
            f"and input width {required_width}. A v3 run may use only an obs-v2 opponent via the exact "
            "v3-to-v2 downgrade; no observation upgrade path exists and obs-v1 remains incompatible."
        )
    return model_version


def save_best_checkpoint(config: TrainConfig, generation: int, model: torch.nn.Module, path: str | Path) -> Path:
    """Persist only the accepted model and enough metadata to reconstruct it."""
    import dominion_v2_py as dz

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generation": int(generation),
            "config": config.to_dict(),
            "encoder_generation": int(dz.ENCODER_GENERATION),
            "model": _cpu_state_dict(model),
        },
        destination,
    )
    return destination


def load_best_checkpoint(config: TrainConfig, device: torch.device, path: str | Path) -> tuple[torch.nn.Module, int]:
    payload = _load_checkpoint_payload(path, device)
    require_native_runner_encoder_compatibility(payload, path)
    model_version = _validate_league_checkpoint_observation(config, payload, path)
    # Action-space dimensions come from the native protocol; import lazily to
    # keep this module usable by its pure persistence/sampling tests without
    # pybind. Observation width follows the selected training config.
    import dominion_v2_py as dz

    model = build_model(
        _checkpoint_model_config(payload, path),
        obs_size_for_version(model_version),
        dz.ACTION_SPACE_SIZE,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, int(payload["generation"])


def _load_league_seed_checkpoint(config: TrainConfig, device: torch.device, path: Path) -> None:
    """Validate that an external standard checkpoint exactly fits this run.

    Seed checkpoints are deliberately validated from their saved architecture
    metadata before ``load_state_dict``.  A partially compatible state dict is
    worse than a missing league member: it would silently change the intended
    external opponent.
    """
    payload = _load_checkpoint_payload(path, device)
    require_native_runner_encoder_compatibility(payload, path)
    if not isinstance(payload, dict) or "model" not in payload or "config" not in payload:
        raise ValueError(
            f"league seed checkpoint {path} must use the standard payload format "
            "with 'model' and 'config' keys"
        )
    checkpoint_config = payload["config"]
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"league seed checkpoint {path} has a non-object 'config' payload")
    checkpoint_model = _checkpoint_model_config(payload, path)
    checkpoint_arch = checkpoint_model.get("arch", "mlp")
    expected_model = model_config_dict(config.model)
    expected_arch = expected_model.get("arch", "mlp")
    # Preserve the historical same-MLP architecture guard. Different
    # architectures intentionally coexist in a league and are loaded from
    # their own payload metadata below.
    if checkpoint_arch == expected_arch == "mlp":
        checkpoint_hidden_sizes = checkpoint_model.get("hidden_sizes")
        if not isinstance(checkpoint_hidden_sizes, list):
            raise ValueError(
                f"league seed checkpoint {path} is missing config.model.hidden_sizes; "
                "cannot validate its architecture"
            )
        expected_hidden_sizes = list(config.model.hidden_sizes)
        if checkpoint_hidden_sizes != expected_hidden_sizes:
            raise ValueError(
                f"league seed checkpoint {path} has hidden_sizes {checkpoint_hidden_sizes}, "
                f"but the current run requires {expected_hidden_sizes}; refusing to truncate or pad weights"
            )

    model_version = _validate_league_checkpoint_observation(config, payload, path)

    # Action-space dimensions come from the native protocol; import lazily to
    # keep this module usable by its pure persistence/sampling tests without
    # pybind. Observation width follows the selected training config.
    import dominion_v2_py as dz

    model = build_model(
        checkpoint_model,
        obs_size_for_version(model_version),
        dz.ACTION_SPACE_SIZE,
    ).to(device)
    try:
        model.load_state_dict(payload["model"])
    except (RuntimeError, TypeError, KeyError) as exc:
        raise ValueError(
            f"league seed checkpoint {path} cannot be loaded with the current model architecture; "
            "refusing to truncate or pad weights"
        ) from exc


def seed_league_checkpoints(config: TrainConfig, device: torch.device) -> list[Path]:
    """Validate and copy configured external opponents into the league pool."""
    sources = config.league_seed_checkpoints
    if not isinstance(sources, list) or not all(isinstance(source, str) for source in sources):
        raise ValueError("league_seed_checkpoints must be a list of checkpoint paths")
    destination_dir = league_directory(config.checkpoint_dir)
    destinations: list[Path] = []
    for index, source_name in enumerate(sources):
        source = Path(source_name)
        if not source.is_file():
            raise FileNotFoundError(f"league seed checkpoint does not exist: {source}")
        _load_league_seed_checkpoint(config, device, source)
        destination = destination_dir / f"seed_{index}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        destinations.append(destination)
    return destinations


def clone_model(config: TrainConfig, source: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    import dominion_v2_py as dz

    clone = build_model(config.model, obs_size_for_config(config), dz.ACTION_SPACE_SIZE).to(device)
    clone.load_state_dict(source.state_dict())
    clone.eval()
    return clone


def initialize_best_checkpoint(
    config: TrainConfig,
    candidate: torch.nn.Module,
    device: torch.device,
    *,
    source_path: str | Path | None = None,
    start_generation: int = 0,
) -> tuple[torch.nn.Module, int, Path]:
    """Load a persisted best or seed the initial best from the candidate."""
    destination = best_checkpoint_path(config.checkpoint_dir)
    source = Path(source_path) if source_path is not None else destination
    if source.exists():
        best, generation = load_best_checkpoint(config, device, source)
        if source != destination:
            save_best_checkpoint(config, generation, best, destination)
        return best, generation, destination

    if start_generation > 0:
        warnings.warn(
            f"best checkpoint {source} is missing; seeding best from the resumed candidate",
            RuntimeWarning,
            stacklevel=2,
        )
    best = clone_model(config, candidate, device)
    generation = int(start_generation)
    save_best_checkpoint(config, generation, best, destination)
    return best, generation, destination


def archive_previous_best(
    config: TrainConfig,
    best_path: str | Path,
    best_generation: int,
) -> list[Path]:
    """Archive the outgoing best, retaining only the newest configured pool."""
    keep = int(config.league_pool_size)
    if keep <= 0:
        return []
    destination_dir = league_directory(config.checkpoint_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"best_{int(best_generation):04d}.pt"
    if not destination.exists():
        shutil.copy2(best_path, destination)
    _trim_self_league_checkpoints(config)
    return sorted(destination_dir.glob("best_*.pt"))


def _self_league_checkpoint_paths(config: TrainConfig) -> list[Path]:
    directory = league_directory(config.checkpoint_dir)
    paths = [*directory.glob("best_*.pt"), *directory.glob("self_*.pt")]

    def fifo_key(path: Path) -> tuple[int, int, str]:
        match = re.fullmatch(r"(best|self)_(\d+)\.pt", path.name)
        if match is None:  # pragma: no cover - glob patterns above enforce this
            return (0, 0, path.name)
        # A periodic self checkpoint is saved before a later gate archives the
        # same generation's accepted best, so it is the older FIFO member.
        kind, generation = match.groups()
        return (int(generation), 0 if kind == "self" else 1, path.name)

    return sorted(paths, key=fifo_key)


def _trim_self_league_checkpoints(config: TrainConfig) -> list[Path]:
    """Keep a FIFO cap over self-added opponents, never over standing seeds."""
    keep = int(config.league_pool_size)
    paths = _self_league_checkpoint_paths(config)
    while len(paths) > max(0, keep):
        paths.pop(0).unlink()
    return _self_league_checkpoint_paths(config)


def archive_self_checkpoint(
    config: TrainConfig,
    checkpoint_path: str | Path,
    generation: int,
) -> list[Path]:
    """Add a periodic candidate checkpoint to the league's self pool."""
    every = int(config.league_self_every)
    if every <= 0 or int(generation) % every != 0 or int(config.league_pool_size) <= 0:
        return _self_league_checkpoint_paths(config)
    destination_dir = league_directory(config.checkpoint_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"self_{int(generation):04d}.pt"
    source = Path(checkpoint_path)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return _trim_self_league_checkpoints(config)


def league_checkpoint_paths(config: TrainConfig) -> list[Path]:
    directory = league_directory(config.checkpoint_dir)
    # Seeds are standing external opponents; archive retention applies only to
    # self-added history, not to the explicitly requested seed list.
    seeds = sorted(directory.glob("seed_*.pt"))
    return [*seeds, *_trim_self_league_checkpoints(config)]


def sample_league_games(
    total_games: int,
    fraction: float,
    pool_size: int,
    seed: int,
    opponent_weights: list[float] | None = None,
    league_opponents_per_gen: int = 0,
) -> list[LeagueGame | None]:
    """Return an exact-fraction per-game plan with optional weighted opponents.

    A positive ``league_opponents_per_gen`` first chooses that many distinct
    historical opponents, weighted without replacement, then apportions all
    league games between only those opponents. Zero deliberately preserves the
    original independent per-game opponent draw.
    """
    if total_games < 0:
        raise ValueError("total_games cannot be negative")
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("league_fraction must be between zero and one")
    if (
        not isinstance(league_opponents_per_gen, int)
        or isinstance(league_opponents_per_gen, bool)
        or league_opponents_per_gen < 0
    ):
        raise ValueError("league_opponents_per_gen must be a non-negative integer")
    plan: list[LeagueGame | None] = [None] * total_games
    if total_games == 0 or pool_size <= 0 or fraction <= 0.0:
        return plan
    if opponent_weights is not None:
        if len(opponent_weights) != pool_size:
            raise ValueError("league opponent weights must match league pool size")
        if any(not math.isfinite(float(weight)) or float(weight) <= 0.0 for weight in opponent_weights):
            raise ValueError("league opponent weights must be finite and positive")
    league_games = min(total_games, int(round(total_games * float(fraction))))
    rng = random.Random(int(seed))
    indices = sorted(rng.sample(range(total_games), league_games))
    if league_opponents_per_gen == 0:
        # Keep the legacy per-game selection and RNG-consumption order exactly
        # intact for uncapped configurations and historical tests/checkpoints.
        for index in indices:
            plan[index] = LeagueGame(
                opponent_index=(
                    rng.randrange(pool_size)
                    if opponent_weights is None
                    else rng.choices(range(pool_size), weights=opponent_weights, k=1)[0]
                ),
                # Self-play always places current best at seat zero and the
                # sampled archived best at seat one. Gate matches separately
                # seat-swap candidate and best for unbiased promotion decisions.
                best_player=0,
            )
        return plan

    weights = [1.0] * pool_size if opponent_weights is None else list(opponent_weights)
    selected_count = min(league_opponents_per_gen, pool_size, league_games)
    remaining_opponents = list(range(pool_size))
    selected: list[int] = []
    while len(selected) < selected_count:
        selected_position = rng.choices(
            range(len(remaining_opponents)),
            weights=[weights[index] for index in remaining_opponents],
            k=1,
        )[0]
        selected.append(remaining_opponents.pop(selected_position))

    # Allocate exact integer counts proportionally while reserving one game
    # for every selected opponent. The reservation keeps the selected set and
    # planned set identical even for very small league slices.
    counts = [1] * selected_count
    remaining_games = league_games - selected_count
    if remaining_games:
        selected_weights = [weights[index] for index in selected]
        weight_total = sum(selected_weights)
        fractional_counts = [remaining_games * weight / weight_total for weight in selected_weights]
        extra_counts = [int(count) for count in fractional_counts]
        for position, count in enumerate(extra_counts):
            counts[position] += count
        leftover = remaining_games - sum(extra_counts)
        for position in sorted(
            range(selected_count),
            key=lambda index: (-(fractional_counts[index] - extra_counts[index]), index),
        )[:leftover]:
            counts[position] += 1

    opponent_draws = [
        opponent_index
        for opponent_index, count in zip(selected, counts, strict=True)
        for _ in range(count)
    ]
    rng.shuffle(opponent_draws)
    for index in indices:
        plan[index] = LeagueGame(
            opponent_index=opponent_draws.pop(),
            # Self-play always places current best at seat zero and the
            # sampled archived best at seat one. Gate matches separately
            # seat-swap candidate and best for unbiased promotion decisions.
            best_player=0,
        )
    return plan


def plan_selfplay_segments(
    total_games: int,
    fraction: float,
    pool_size: int,
    seed: int,
    opponent_weights: list[float] | None = None,
    opponent_names: list[str] | None = None,
    parallel_workers: int = 1,
    league_opponents_per_gen: int = 0,
) -> list[SelfPlaySegment]:
    """Collapse a deterministic league draw into deduplicated model segments.

    Model id zero is always the current best. Historical opponent ids are one
    plus their sampled-pool index. Grouping equal pairs means workers receive
    each model state once per generation instead of one payload per game.
    """
    if not isinstance(parallel_workers, int) or isinstance(parallel_workers, bool) or parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")
    if opponent_names is not None and len(opponent_names) != pool_size:
        raise ValueError("league opponent names must match league pool size")
    counts: dict[tuple[int, int], int] = {}
    for game in sample_league_games(
        total_games,
        fraction,
        pool_size,
        seed,
        opponent_weights,
        league_opponents_per_gen,
    ):
        pair = (0, 0) if game is None else (0, int(game.opponent_index) + 1)
        counts[pair] = counts.get(pair, 0) + 1
    segments = [
        SelfPlaySegment(
            n_games=count,
            seat0_model_id=pair[0],
            seat1_model_id=pair[1],
            league_opponent=(
                None
                if pair == (0, 0) or opponent_names is None
                else opponent_names[pair[1] - 1]
            ),
        )
        for pair, count in sorted(counts.items())
        if count > 0
    ]
    return segments


def carve_deep_slice_segments(
    segments: list[SelfPlaySegment],
    fraction: float,
    sims: int,
    base_sims: int,
    parallel_workers: int = 1,
) -> list[SelfPlaySegment]:
    """Replace an exact fraction of normal mirror games with deep-search work.

    The fraction is measured against the normal mirror pool, so a generation
    with scripted or league games keeps those data sources completely intact.
    Rounding intentionally matches ``scripted_opponent_game_counts``.
    """
    validate_deep_slice_config(
        SelfPlayConfig(
            sims_per_move=base_sims,
            deep_slice_fraction=fraction,
            deep_slice_sims=sims,
        )
    )
    if float(fraction) == 0.0:
        return segments
    if not isinstance(parallel_workers, int) or isinstance(parallel_workers, bool) or parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")

    normal_games = sum(segment.n_games for segment in segments if segment.is_normal_mirror)
    deep_games = min(normal_games, int(round(normal_games * float(fraction))))
    deep_segment_count = min(deep_games, parallel_workers)
    deep_base, deep_remainder = divmod(deep_games, deep_segment_count) if deep_segment_count else (0, 0)
    deep_piece_sizes = [
        deep_base + (1 if index < deep_remainder else 0)
        for index in range(deep_segment_count)
    ]
    remaining_deep_games = deep_games
    deep_piece_index = 0
    deep_piece_remaining = deep_piece_sizes[0] if deep_piece_sizes else 0
    carved: list[SelfPlaySegment] = []
    for segment in segments:
        if not segment.is_normal_mirror or remaining_deep_games == 0:
            carved.append(segment)
            continue
        segment_deep_games = min(segment.n_games, remaining_deep_games)
        segment_normal_games = segment.n_games - segment_deep_games
        if segment_normal_games:
            carved.append(
                SelfPlaySegment(
                    n_games=segment_normal_games,
                    seat0_model_id=segment.seat0_model_id,
                    seat1_model_id=segment.seat1_model_id,
                    scripted_kind=segment.scripted_kind,
                    nn_player=segment.nn_player,
                    kingdom_pool=segment.kingdom_pool,
                    kingdom_mode=segment.kingdom_mode,
                    sims_override=segment.sims_override,
                    league_opponent=segment.league_opponent,
                )
            )
        remaining_deep_games -= segment_deep_games
        while segment_deep_games:
            piece_games = min(segment_deep_games, deep_piece_remaining)
            carved.append(replace(segment, n_games=piece_games, sims_override=int(sims)))
            segment_deep_games -= piece_games
            deep_piece_remaining -= piece_games
            if deep_piece_remaining == 0:
                deep_piece_index += 1
                if deep_piece_index < len(deep_piece_sizes):
                    deep_piece_remaining = deep_piece_sizes[deep_piece_index]
    if remaining_deep_games != 0 or deep_piece_index != len(deep_piece_sizes):
        raise RuntimeError("deep self-play segments do not cover the requested game count")
    return carved


def scripted_opponent_game_counts(total_games: int, opponents: dict[str, float]) -> dict[str, int]:
    """Validate configured scripted fractions and turn them into exact counts."""
    if total_games < 0:
        raise ValueError("total_games cannot be negative")
    if not isinstance(opponents, dict):
        raise ValueError("scripted_opponents must be an object mapping kind to fraction")
    total_fraction = 0.0
    counts: dict[str, int] = {}
    for raw_kind, raw_fraction in sorted(opponents.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_kind, str) or raw_kind not in SCRIPTED_OPPONENT_KINDS:
            allowed = ", ".join(sorted(SCRIPTED_OPPONENT_KINDS))
            raise ValueError(f"unknown scripted opponent {raw_kind!r}; expected one of {allowed}")
        if not isinstance(raw_fraction, (int, float)):
            raise ValueError(f"scripted opponent fraction for {raw_kind} must be numeric")
        fraction = float(raw_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"scripted opponent fraction for {raw_kind} must be between zero and one")
        total_fraction += fraction
        count = min(total_games, int(round(total_games * fraction)))
        if count > 0:
            counts[raw_kind] = count
    if total_fraction > 1.0 + 1.0e-9:
        raise ValueError("scripted opponent fractions must sum to at most one")
    if sum(counts.values()) > total_games:
        raise ValueError("scripted opponent game counts exceed one generation")
    return counts


def effective_scripted_fractions(
    schedule: dict[str, list],
    scripted_opponents: dict[str, float],
    generation: int,
) -> dict[str, float]:
    """Resolve per-generation scripted-opponent fractions from breakpoints."""
    if not isinstance(schedule, dict):
        raise ValueError("scripted_opponent_schedule must be an object mapping kind to breakpoints")
    if not isinstance(scripted_opponents, dict):
        raise ValueError("scripted_opponents must be an object mapping kind to fraction")
    if not isinstance(generation, int) or isinstance(generation, bool):
        raise ValueError("generation must be an integer")

    fractions: dict[str, float] = {}
    for raw_kind, raw_fraction in sorted(scripted_opponents.items(), key=lambda item: str(item[0])):
        if raw_kind in schedule:
            continue
        if not isinstance(raw_kind, str) or raw_kind not in SCRIPTED_OPPONENT_KINDS:
            allowed = ", ".join(sorted(SCRIPTED_OPPONENT_KINDS))
            raise ValueError(f"unknown scripted opponent {raw_kind!r}; expected one of {allowed}")
        if not isinstance(raw_fraction, (int, float)) or isinstance(raw_fraction, bool):
            raise ValueError(f"scripted opponent fraction for {raw_kind} must be numeric")
        fraction = float(raw_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"scripted opponent fraction for {raw_kind} must be between zero and one")
        if fraction > 0.0:
            fractions[raw_kind] = fraction

    for raw_kind, raw_breakpoints in sorted(schedule.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_kind, str) or raw_kind not in SCRIPTED_OPPONENT_KINDS:
            allowed = ", ".join(sorted(SCRIPTED_OPPONENT_KINDS))
            raise ValueError(f"unknown scripted opponent {raw_kind!r}; expected one of {allowed}")
        if not isinstance(raw_breakpoints, list) or not raw_breakpoints:
            raise ValueError(f"scripted opponent schedule for {raw_kind} must be a non-empty list")

        breakpoints: list[tuple[int, float]] = []
        previous_generation: int | None = None
        for raw_breakpoint in raw_breakpoints:
            if not isinstance(raw_breakpoint, list) or len(raw_breakpoint) != 2:
                raise ValueError(f"scripted opponent schedule for {raw_kind} must contain [generation, fraction] pairs")
            raw_generation, raw_fraction = raw_breakpoint
            if not isinstance(raw_generation, int) or isinstance(raw_generation, bool) or raw_generation < 0:
                raise ValueError(
                    f"scripted opponent schedule generation for {raw_kind} must be an integer at least zero"
                )
            if not isinstance(raw_fraction, (int, float)) or isinstance(raw_fraction, bool):
                raise ValueError(f"scripted opponent schedule fraction for {raw_kind} must be numeric")
            fraction = float(raw_fraction)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"scripted opponent schedule fraction for {raw_kind} must be between zero and one")
            if previous_generation is not None and raw_generation <= previous_generation:
                raise ValueError(f"scripted opponent schedule generations for {raw_kind} must be strictly increasing")
            breakpoints.append((raw_generation, fraction))
            previous_generation = raw_generation

        if generation <= breakpoints[0][0]:
            effective_fraction = breakpoints[0][1]
        elif generation >= breakpoints[-1][0]:
            effective_fraction = breakpoints[-1][1]
        else:
            for (start_generation, start_fraction), (end_generation, end_fraction) in zip(
                breakpoints, breakpoints[1:]
            ):
                if generation <= end_generation:
                    progress = (generation - start_generation) / (end_generation - start_generation)
                    effective_fraction = start_fraction + (end_fraction - start_fraction) * progress
                    break
        if effective_fraction > 0.0:
            fractions[raw_kind] = effective_fraction

    return fractions


def effective_league_fraction(schedule: list, league_fraction: float, generation: int) -> float:
    """Resolve a league dose ramp with scripted-schedule breakpoint semantics."""
    if not isinstance(league_fraction, (int, float)) or isinstance(league_fraction, bool):
        raise ValueError("league_fraction must be numeric")
    fixed_fraction = float(league_fraction)
    if not math.isfinite(fixed_fraction) or not 0.0 <= fixed_fraction <= 1.0:
        raise ValueError("league_fraction must be between zero and one")
    if not isinstance(schedule, list):
        raise ValueError("league_schedule must be a list of [generation, fraction] pairs")
    if not isinstance(generation, int) or isinstance(generation, bool):
        raise ValueError("generation must be an integer")
    if not schedule:
        return fixed_fraction

    breakpoints: list[tuple[int, float]] = []
    previous_generation: int | None = None
    for raw_breakpoint in schedule:
        if not isinstance(raw_breakpoint, list) or len(raw_breakpoint) != 2:
            raise ValueError("league_schedule must contain [generation, fraction] pairs")
        raw_generation, raw_fraction = raw_breakpoint
        if not isinstance(raw_generation, int) or isinstance(raw_generation, bool) or raw_generation < 0:
            raise ValueError("league_schedule generation must be an integer at least zero")
        if not isinstance(raw_fraction, (int, float)) or isinstance(raw_fraction, bool):
            raise ValueError("league_schedule fraction must be numeric")
        fraction = float(raw_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("league_schedule fraction must be between zero and one")
        if previous_generation is not None and raw_generation <= previous_generation:
            raise ValueError("league_schedule generations must be strictly increasing")
        breakpoints.append((raw_generation, fraction))
        previous_generation = raw_generation

    if generation <= breakpoints[0][0]:
        return breakpoints[0][1]
    if generation >= breakpoints[-1][0]:
        return breakpoints[-1][1]
    for (start_generation, start_fraction), (end_generation, end_fraction) in zip(
        breakpoints, breakpoints[1:]
    ):
        if generation <= end_generation:
            progress = (generation - start_generation) / (end_generation - start_generation)
            return start_fraction + (end_fraction - start_fraction) * progress
    raise RuntimeError("league schedule did not cover its interpolation interval")


def league_opponent_weights(
    opponent_names: list[str],
    performance: dict[str, tuple[int, int]],
) -> list[float]:
    """Weight opponents toward lower observed current-seat win rates.

    Unseen opponents are treated as zero wins from zero games. The mandatory
    +0.1 floor leaves even a fully beaten opponent selectable.
    """
    weights: list[float] = []
    for name in opponent_names:
        games, wins = performance.get(name, (0, 0))
        if games < 0 or wins < 0 or wins > games:
            raise ValueError(f"league performance for {name!r} must satisfy 0 <= wins <= games")
        net_win_rate = wins / games if games else 0.0
        weights.append(1.0 - net_win_rate + 0.1)
    return weights


def _resolve_kingdom_pool(raw_pool: object, phase_index: int) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Resolve curriculum card names through the binding's canonical lookup."""
    if not isinstance(raw_pool, list) or not raw_pool:
        raise ValueError(f"kingdom curriculum phase {phase_index} pool must be a non-empty list of card names")
    if len(raw_pool) < 10:
        raise ValueError(f"kingdom curriculum phase {phase_index} pool must contain at least 10 cards")
    if not all(isinstance(name, str) for name in raw_pool):
        raise ValueError(f"kingdom curriculum phase {phase_index} pool card names must be strings")

    # ``def_id`` is also what Setup/fixed_kingdom accepts, so curriculum
    # names retain exactly the same spelling and lookup semantics. Assigning
    # the resolved ids to the native config additionally verifies that every
    # card is in SelfPlayRunner's implemented random-kingdom roster.
    import dominion_v2_py as dz

    names = tuple(raw_pool)
    defs: list[int] = []
    for name in names:
        try:
            defs.append(int(dz.def_id(name)))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"kingdom curriculum phase {phase_index} pool contains an unknown card name: {name!r}"
            ) from exc
    try:
        native_config = dz.SelfPlayConfig()
        native_config.kingdom_pool = defs
    except (TypeError, ValueError) as exc:
        raise ValueError(f"kingdom curriculum phase {phase_index} pool is invalid: {exc}") from exc
    return tuple(defs), names


def effective_kingdom_phase(
    kingdom_curriculum: list,
    base_kingdom_mode: str,
    generation: int,
) -> KingdomCurriculumPhase:
    """Resolve the final inclusive curriculum phase that contains generation.

    Later phases intentionally override earlier overlapping ranges. This makes
    a temporary schedule amendment append-only and mirrors the campaign's
    existing per-generation scheduling convention.
    """
    if not isinstance(kingdom_curriculum, list):
        raise ValueError("kingdom_curriculum must be a list of phases")
    if not isinstance(base_kingdom_mode, str) or base_kingdom_mode.lower() not in {"random", "fixed"}:
        raise ValueError("kingdom_mode must be 'random' or 'fixed'")
    if not isinstance(generation, int) or isinstance(generation, bool):
        raise ValueError("generation must be an integer")

    active: KingdomCurriculumPhase | None = None
    for phase_index, raw_phase in enumerate(kingdom_curriculum):
        if not isinstance(raw_phase, dict):
            raise ValueError(f"kingdom curriculum phase {phase_index} must be an object")
        raw_generations = raw_phase.get("generations")
        if not isinstance(raw_generations, list) or len(raw_generations) != 2:
            raise ValueError(
                f"kingdom curriculum phase {phase_index} generations must be a [start, end] pair"
            )
        start_generation, end_generation = raw_generations
        if (
            not isinstance(start_generation, int)
            or isinstance(start_generation, bool)
            or not isinstance(end_generation, int)
            or isinstance(end_generation, bool)
            or start_generation < 0
            or end_generation < start_generation
        ):
            raise ValueError(
                f"kingdom curriculum phase {phase_index} generations must be non-negative inclusive ranges"
            )
        raw_mode = raw_phase.get("mode")
        if not isinstance(raw_mode, str) or raw_mode.lower() not in {"random", "pool"}:
            raise ValueError(f"kingdom curriculum phase {phase_index} mode must be 'random' or 'pool'")
        mode = raw_mode.lower()
        raw_fraction = raw_phase.get("pool_fraction", 1.0)
        if not isinstance(raw_fraction, (int, float)) or isinstance(raw_fraction, bool):
            raise ValueError(f"kingdom curriculum phase {phase_index} pool_fraction must be numeric")
        pool_fraction = float(raw_fraction)
        if not math.isfinite(pool_fraction) or not 0.0 <= pool_fraction <= 1.0:
            raise ValueError(f"kingdom curriculum phase {phase_index} pool_fraction must be between zero and one")

        pool: tuple[int, ...] = ()
        pool_names: tuple[str, ...] = ()
        if mode == "pool":
            pool, pool_names = _resolve_kingdom_pool(raw_phase.get("pool"), phase_index)
        elif "pool" in raw_phase:
            # A supplied random-phase pool has no effect, but validate it so a
            # typo or unimplemented card cannot be silently ignored.
            _resolve_kingdom_pool(raw_phase["pool"], phase_index)

        if start_generation <= generation <= end_generation:
            active = KingdomCurriculumPhase(
                mode=mode,
                pool=pool,
                pool_names=pool_names,
                pool_fraction=pool_fraction,
                start_generation=start_generation,
                end_generation=end_generation,
            )

    if active is not None:
        return active
    return KingdomCurriculumPhase(mode=base_kingdom_mode.lower())


def assign_kingdom_phase_to_segments(
    segments: list[SelfPlaySegment],
    phase: KingdomCurriculumPhase,
    seed: int,
) -> list[SelfPlaySegment]:
    """Split segments into exact full-pool and curriculum-pool game counts."""
    total_games = sum(segment.n_games for segment in segments)
    if total_games < 0 or any(segment.n_games <= 0 for segment in segments):
        raise ValueError("self-play segments must have positive game counts")

    pool_games = (
        min(total_games, int(round(total_games * phase.pool_fraction)))
        if phase.mode == "pool"
        else 0
    )
    selected_pool_games = set(random.Random(int(seed)).sample(range(total_games), pool_games))
    assigned: list[SelfPlaySegment] = []
    game_offset = 0
    for segment in segments:
        selected = sum(
            game_offset <= game_index < game_offset + segment.n_games
            for game_index in selected_pool_games
        )

        def append_assignment(n_games: int, kingdom_pool: list[int] | None) -> None:
            if n_games <= 0:
                return
            assigned.append(
                SelfPlaySegment(
                    n_games=n_games,
                    seat0_model_id=segment.seat0_model_id,
                    seat1_model_id=segment.seat1_model_id,
                    scripted_kind=segment.scripted_kind,
                    nn_player=segment.nn_player,
                    kingdom_pool=kingdom_pool,
                    kingdom_mode="random" if phase.mode == "pool" else phase.mode,
                    sims_override=segment.sims_override,
                    league_opponent=segment.league_opponent,
                )
            )

        append_assignment(selected, list(phase.pool) if selected > 0 else None)
        append_assignment(segment.n_games - selected, None)
        game_offset += segment.n_games

    if sum(segment.n_games for segment in assigned) != total_games:
        raise RuntimeError("kingdom curriculum segments do not cover the generation")
    if sum(segment.n_games for segment in assigned if segment.kingdom_pool is not None) != pool_games:
        raise RuntimeError("kingdom curriculum pool game count mismatch")
    return assigned


def plan_training_selfplay_segments(
    total_games: int,
    league_fraction: float,
    league_pool_size: int,
    scripted_opponents: dict[str, float],
    seed: int,
    deep_slice_fraction: float = 0.0,
    deep_slice_sims: int = 0,
    sims_per_move: int = 64,
    league_opponent_weights: list[float] | None = None,
    league_opponent_names: list[str] | None = None,
    parallel_workers: int = 1,
    league_opponents_per_gen: int = 0,
) -> list[SelfPlaySegment]:
    """Compose normal, league, and seat-swapped scripted data segments.

    Each configured fraction is measured against the full generation. Scripted
    games are split between NN player zero and one, while historical-league
    games retain their existing current-best-at-player-zero behavior.
    """
    scripted_counts = scripted_opponent_game_counts(total_games, scripted_opponents)
    validate_deep_slice_config(
        SelfPlayConfig(
            sims_per_move=sims_per_move,
            deep_slice_fraction=deep_slice_fraction,
            deep_slice_sims=deep_slice_sims,
        )
    )
    if not scripted_counts:
        # Preserve the existing non-scripted planner byte-for-byte, including
        # its seed behavior and segment ordering.
        return carve_deep_slice_segments(
            plan_selfplay_segments(
                total_games,
                league_fraction,
                league_pool_size,
                seed,
                league_opponent_weights,
                league_opponent_names,
                parallel_workers,
                league_opponents_per_gen,
            ),
            deep_slice_fraction,
            deep_slice_sims,
            sims_per_move,
            parallel_workers,
        )

    if not 0.0 <= float(league_fraction) <= 1.0:
        raise ValueError("league_fraction must be between zero and one")
    league_games = (
        min(total_games, int(round(total_games * float(league_fraction))))
        if league_pool_size > 0 and league_fraction > 0.0
        else 0
    )
    scripted_games = sum(scripted_counts.values())
    if league_games + scripted_games > total_games:
        raise ValueError("league_fraction plus scripted_opponents exceeds one generation")

    segments: list[SelfPlaySegment] = []
    normal_games = total_games - league_games - scripted_games
    if normal_games > 0:
        segments.append(SelfPlaySegment(normal_games, 0, 0))
    if league_games > 0:
        segments.extend(
            plan_selfplay_segments(
                league_games,
                1.0,
                league_pool_size,
                seed,
                league_opponent_weights,
                league_opponent_names,
                parallel_workers,
                league_opponents_per_gen,
            )
        )
    for kind, count in scripted_counts.items():
        first_player_games = (count + 1) // 2
        second_player_games = count - first_player_games
        if first_player_games > 0:
            segments.append(SelfPlaySegment(first_player_games, 0, 0, scripted_kind=kind, nn_player=0))
        if second_player_games > 0:
            segments.append(SelfPlaySegment(second_player_games, 0, 0, scripted_kind=kind, nn_player=1))
    return carve_deep_slice_segments(
        segments,
        deep_slice_fraction,
        deep_slice_sims,
        sims_per_move,
        parallel_workers,
    )


def compact_selfplay_segments(
    segments: list[SelfPlaySegment],
) -> tuple[list[SelfPlaySegment], list[int]]:
    """Make a segment plan's historical model ids a dense worker model table.

    ``plan_selfplay_segments`` uses ``1 + opponent_index`` so its draw can be
    inspected independently of checkpoint paths.  A particular generation
    rarely uses every retained historical checkpoint, however.  Compacting
    here lets the parent broadcast only the referenced historical weights;
    returned indices select those weights from ``league_checkpoint_paths``.
    Model id zero remains the current best in all plans.
    """
    historical_ids = sorted(
        {
            model_id
            for segment in segments
            for model_id in (segment.seat0_model_id, segment.seat1_model_id)
            if model_id != 0
        }
    )
    remap = {0: 0, **{model_id: index + 1 for index, model_id in enumerate(historical_ids)}}
    return (
        [
            SelfPlaySegment(
                n_games=segment.n_games,
                seat0_model_id=remap[segment.seat0_model_id],
                seat1_model_id=remap[segment.seat1_model_id],
                scripted_kind=segment.scripted_kind,
                nn_player=segment.nn_player,
                kingdom_pool=segment.kingdom_pool,
                kingdom_mode=segment.kingdom_mode,
                sims_override=segment.sims_override,
                league_opponent=segment.league_opponent,
            )
            for segment in segments
        ],
        [model_id - 1 for model_id in historical_ids],
    )


def gate_match_seed(config: TrainConfig, generation: int) -> int:
    return int(config.seed) ^ (int(generation) * 0x6A09E667)


def run_seat_swapped_match(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    config: SelfPlayConfig,
    *,
    games: int,
    seed: int,
    device: torch.device,
    determinize: str | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> SeatSwappedMatch:
    """Run a deterministic, seat-swapped NN-MCTS match.

    This is the common game-driving path for training gates and offline
    checkpoint duels.  ``config`` is deliberately caller-owned: gates retain
    their historical copied self-play settings, while an offline evaluator can
    supply uniform, noise-free conditions for arbitrary checkpoints.  Mixed
    observation versions are handled by ``play_routed_games``' per-seat
    routing, including the exact v3-to-v2 downgrade.
    """
    from .selfplay import play_routed_games

    if games < 0:
        raise ValueError("games cannot be negative")
    if games == 0:
        return SeatSwappedMatch(GateStats(0, 0, 0), (), (), 0.0)

    first_half = (int(games) + 1) // 2
    second_half = int(games) - first_half
    wins = losses = ties = 0
    records: list[dict[str, Any]] = []
    a_players: list[int] = []
    start = time.perf_counter()
    progress_wins = progress_losses = progress_ties = 0
    for count, a_player, seat_models, offset in (
        (first_half, 0, (model_a, model_b), 0),
        (second_half, 1, (model_b, model_a), 1),
    ):
        if count == 0:
            continue

        def report_finished(completed: list[dict[str, Any]]) -> None:
            nonlocal progress_wins, progress_losses, progress_ties
            if progress_callback is None:
                return
            for record in completed:
                winner = record.get("winner")
                if winner is None:
                    progress_ties += 1
                elif int(winner) == a_player:
                    progress_wins += 1
                else:
                    progress_losses += 1
                progress_callback(progress_wins, progress_losses, progress_ties)

        route_kwargs: dict[str, Any] = {}
        if progress_callback is not None:
            route_kwargs["on_finished"] = report_finished
        if determinize is not None:
            route_kwargs["determinize"] = determinize
        _, completed = play_routed_games(
            seat_models,
            config,
            seed=int(seed) + (offset * 0x10001),
            device=device,
            target_games=count,
            **route_kwargs,
        )
        for record in completed:
            winner = record.get("winner")
            if winner is None:
                ties += 1
            elif int(winner) == a_player:
                wins += 1
            else:
                losses += 1
            records.append(record)
            a_players.append(a_player)
    return SeatSwappedMatch(
        stats=GateStats(wins=wins, losses=losses, ties=ties),
        records=tuple(records),
        a_players=tuple(a_players),
        wall_time=time.perf_counter() - start,
    )


def run_gate_match(
    candidate: torch.nn.Module,
    best: torch.nn.Module,
    config: TrainConfig,
    generation: int,
    device: torch.device,
) -> GateStats:
    """Run a deterministic, seat-swapped NN-MCTS candidate-versus-best match."""
    games = int(config.gate_games)
    if games <= 0:
        return GateStats(0, 0, 0)
    if int(config.gate_sims) <= 0:
        raise ValueError("gate_sims must be positive when gating is enabled")

    gate_config = copy.deepcopy(config.selfplay)
    gate_config.sims_per_move = int(config.gate_sims)
    # Gate matches remain a separate, uniform-budget evaluation; their copied
    # self-play config must not inherit a data-generation-only deep slice.
    gate_config.deep_slice_fraction = 0.0
    gate_config.deep_slice_sims = 0
    gate_config.dirichlet_frac = 0.0
    gate_config.temp_moves = int(config.gate_temp_moves)
    gate_config.kingdom_mode = "random"
    return run_seat_swapped_match(
        candidate,
        best,
        gate_config,
        games=games,
        seed=gate_match_seed(config, generation),
        device=device,
    ).stats
