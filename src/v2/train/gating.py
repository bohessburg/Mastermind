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
    if keep <= 0:
        return []
    return sorted(league_directory(config.checkpoint_dir).glob("best_*.pt"))[-keep:]


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
    first_best_player = rng.randrange(2)
    for order, index in enumerate(indices):
        plan[index] = LeagueGame(
            opponent_index=rng.randrange(pool_size),
            best_player=(first_best_player + order) % 2,
        )
    return plan


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
    gate_config.temp_moves = 0
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
