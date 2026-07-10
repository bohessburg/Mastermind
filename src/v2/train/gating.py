"""Checkpoint gating and deterministic mini-league bookkeeping.

The trainer keeps a candidate network (and its optimizer) separate from the
accepted best network.  League assignment is deliberately derived only from
the configured seed and generation: parallel workers may finish in a different
order, but the games assigned to each opponent remain reproducible.
"""

from __future__ import annotations

import copy
import random
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import TrainConfig
from .model import DominionNet


BEST_FILENAME = "best.pt"
LEAGUE_DIRNAME = "league"
SCRIPTED_OPPONENT_KINDS = frozenset({"bigmoney", "engine", "random"})


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

    @property
    def is_league(self) -> bool:
        return not self.is_scripted and self.seat0_model_id != self.seat1_model_id

    @property
    def is_scripted(self) -> bool:
        return self.scripted_kind is not None


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


def save_best_checkpoint(config: TrainConfig, generation: int, model: torch.nn.Module, path: str | Path) -> Path:
    """Persist only the accepted model and enough metadata to reconstruct it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generation": int(generation),
            "config": config.to_dict(),
            "model": _cpu_state_dict(model),
        },
        destination,
    )
    return destination


def load_best_checkpoint(config: TrainConfig, device: torch.device, path: str | Path) -> tuple[DominionNet, int]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older supported Torch versions
        payload = torch.load(path, map_location=device)
    # Model dimensions are protocol constants; import lazily to keep this
    # module usable by its pure persistence/sampling tests without pybind.
    import dominion_v2_py as dz

    model = DominionNet(dz.OBS_SIZE, dz.ACTION_SPACE_SIZE, config.model.hidden_sizes).to(device)
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
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older supported Torch versions
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "model" not in payload or "config" not in payload:
        raise ValueError(
            f"league seed checkpoint {path} must use the standard payload format "
            "with 'model' and 'config' keys"
        )
    checkpoint_config = payload["config"]
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"league seed checkpoint {path} has a non-object 'config' payload")
    checkpoint_model = checkpoint_config.get("model")
    checkpoint_hidden_sizes = checkpoint_model.get("hidden_sizes") if isinstance(checkpoint_model, dict) else None
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

    # Model dimensions are protocol constants; import lazily to keep this
    # module usable by its pure persistence/sampling tests without pybind.
    import dominion_v2_py as dz

    model = DominionNet(dz.OBS_SIZE, dz.ACTION_SPACE_SIZE, expected_hidden_sizes).to(device)
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


def clone_model(config: TrainConfig, source: torch.nn.Module, device: torch.device) -> DominionNet:
    import dominion_v2_py as dz

    clone = DominionNet(dz.OBS_SIZE, dz.ACTION_SPACE_SIZE, config.model.hidden_sizes).to(device)
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
) -> tuple[DominionNet, int, Path]:
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
    paths = sorted(destination_dir.glob("best_*.pt"))
    while len(paths) > keep:
        paths.pop(0).unlink()
    return paths


def league_checkpoint_paths(config: TrainConfig) -> list[Path]:
    keep = int(config.league_pool_size)
    directory = league_directory(config.checkpoint_dir)
    # Seeds are standing external opponents; archive retention applies only to
    # accepted-best history, not to the explicitly requested seed list.
    seeds = sorted(directory.glob("seed_*.pt"))
    archived = sorted(directory.glob("best_*.pt"))
    return [*seeds, *archived[-keep:]] if keep > 0 else seeds


def sample_league_games(
    total_games: int,
    fraction: float,
    pool_size: int,
    seed: int,
) -> list[LeagueGame | None]:
    """Return an exact-fraction per-game plan, uniformly sampling opponents."""
    if total_games < 0:
        raise ValueError("total_games cannot be negative")
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("league_fraction must be between zero and one")
    plan: list[LeagueGame | None] = [None] * total_games
    if total_games == 0 or pool_size <= 0 or fraction <= 0.0:
        return plan
    league_games = min(total_games, int(round(total_games * float(fraction))))
    rng = random.Random(int(seed))
    indices = sorted(rng.sample(range(total_games), league_games))
    for index in indices:
        plan[index] = LeagueGame(
            opponent_index=rng.randrange(pool_size),
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
) -> list[SelfPlaySegment]:
    """Collapse a deterministic league draw into deduplicated model segments.

    Model id zero is always the current best. Historical opponent ids are one
    plus their sampled-pool index. Grouping equal pairs means workers receive
    each model state once per generation instead of one payload per game.
    """
    counts: dict[tuple[int, int], int] = {}
    for game in sample_league_games(total_games, fraction, pool_size, seed):
        pair = (0, 0) if game is None else (0, int(game.opponent_index) + 1)
        counts[pair] = counts.get(pair, 0) + 1
    return [
        SelfPlaySegment(n_games=count, seat0_model_id=pair[0], seat1_model_id=pair[1])
        for pair, count in sorted(counts.items())
        if count > 0
    ]


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


def plan_training_selfplay_segments(
    total_games: int,
    league_fraction: float,
    league_pool_size: int,
    scripted_opponents: dict[str, float],
    seed: int,
) -> list[SelfPlaySegment]:
    """Compose normal, league, and seat-swapped scripted data segments.

    Each configured fraction is measured against the full generation. Scripted
    games are split between NN player zero and one, while historical-league
    games retain their existing current-best-at-player-zero behavior.
    """
    scripted_counts = scripted_opponent_game_counts(total_games, scripted_opponents)
    if not scripted_counts:
        # Preserve the existing non-scripted planner byte-for-byte, including
        # its seed behavior and segment ordering.
        return plan_selfplay_segments(total_games, league_fraction, league_pool_size, seed)

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
        segments.extend(plan_selfplay_segments(league_games, 1.0, league_pool_size, seed))
    for kind, count in scripted_counts.items():
        first_player_games = (count + 1) // 2
        second_player_games = count - first_player_games
        if first_player_games > 0:
            segments.append(SelfPlaySegment(first_player_games, 0, 0, scripted_kind=kind, nn_player=0))
        if second_player_games > 0:
            segments.append(SelfPlaySegment(second_player_games, 0, 0, scripted_kind=kind, nn_player=1))
    return segments


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
            )
            for segment in segments
        ],
        [model_id - 1 for model_id in historical_ids],
    )


def gate_match_seed(config: TrainConfig, generation: int) -> int:
    return int(config.seed) ^ (int(generation) * 0x6A09E667)


def run_gate_match(
    candidate: torch.nn.Module,
    best: torch.nn.Module,
    config: TrainConfig,
    generation: int,
    device: torch.device,
) -> GateStats:
    """Run a deterministic, seat-swapped NN-MCTS candidate-versus-best match."""
    from .selfplay import play_routed_games

    games = int(config.gate_games)
    if games <= 0:
        return GateStats(0, 0, 0)
    if int(config.gate_sims) <= 0:
        raise ValueError("gate_sims must be positive when gating is enabled")

    gate_config = copy.deepcopy(config.selfplay)
    gate_config.sims_per_move = int(config.gate_sims)
    gate_config.dirichlet_frac = 0.0
    gate_config.temp_moves = int(config.gate_temp_moves)
    gate_config.kingdom_mode = "random"
    first_half = (games + 1) // 2
    second_half = games - first_half
    wins = losses = ties = 0
    seed = gate_match_seed(config, generation)
    for count, candidate_player, seat_models, offset in (
        (first_half, 0, (candidate, best), 0),
        (second_half, 1, (best, candidate), 1),
    ):
        if count == 0:
            continue
        _, records = play_routed_games(
            seat_models,
            gate_config,
            seed=seed + (offset * 0x10001),
            device=device,
            target_games=count,
        )
        for record in records:
            winner = record.get("winner")
            if winner is None:
                ties += 1
            elif int(winner) == candidate_player:
                wins += 1
            else:
                losses += 1
    return GateStats(wins=wins, losses=losses, ties=ties)
