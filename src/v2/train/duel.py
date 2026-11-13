"""Offline, deterministic NN-versus-NN checkpoint duels.

Run from the repository root, for example::

    PYTHONPATH=build python src/v2/train/duel.py \
        --a checkpoints/remote/campaign18/gen_0005.pt \
        --b checkpoints/remote/campaign15/gen_0045.pt --games 200 --sims 400

The game runner always uses the newest observation protocol required by the
two checkpoints.  A v2 model in a v3 game is served through the same exact
v3-to-v2 downgrade used by league and gate games.

The native SelfPlayRunner owns its encoded leaf buffers. It cannot apply the
Python legacy shim, so generation-1 checkpoints must be duelled on a
pre-sentinel-fix encoder-generation-1 build.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

try:
    import dominion_v2_py as dz
except ModuleNotFoundError as exc:  # pragma: no cover - clearer standalone error
    raise SystemExit("dominion_v2_py not found; run with PYTHONPATH=build") from exc

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.v2.train.config import SelfPlayConfig, TrainConfig
    from src.v2.train.evaluate import load_model
    from src.v2.train.gating import SeatSwappedMatch, run_seat_swapped_match
    from src.v2.train.train import seed_everything, select_device
else:
    from .config import SelfPlayConfig, TrainConfig
    from .evaluate import load_model
    from .gating import SeatSwappedMatch, run_seat_swapped_match
    from .train import seed_everything, select_device


# These are the v2 encoder's stable supply-block layout constants from
# encode/encoder.h.  The native self-play record gives us the final decision
# observation, not the private terminal GameState available to EvalRunner.
# The game-over check happens at cleanup, so that final decision still has the
# terminal supply counts needed for the same Province/three-piles accounting.
_OBS_META_SIZE = 4
_OBS_OWN_ZONE_COUNT = 5
_OBS_OPPONENT_BLOCK_SIZE_V1 = 75
_OBS_PILE_BLOCK_SIZE = 11
_BASE_SUPPLY_PILES = 7  # Copper, Silver, Gold, Estate, Duchy, Province, Curse.
_MAX_PILES = 48
_OBS_LANDSCAPE_SIZE = 27
_OBS_RESOURCE_SIZE = 12
_OBS_PHASE_COUNT = 5
_MAX_TURNS = 200


@dataclass(frozen=True)
class DuelStats:
    games: int
    wins_a: int
    wins_b: int
    ties: int
    truncated: int
    end_province: int
    end_piles: int
    wall_time: float
    # (a_seat, turns, result) per completed game, result in {"W","L","T"} for a.
    per_game: tuple = ()

    @property
    def decisive(self) -> int:
        return self.wins_a + self.wins_b

    @property
    def win_pct_a_excl_ties(self) -> float:
        return 100.0 * self.wins_a / self.decisive if self.decisive else 0.0

    @property
    def games_per_hour(self) -> float:
        return 3600.0 * self.games / self.wall_time if self.wall_time > 0.0 else 0.0


@dataclass(frozen=True)
class LoadedDuelCheckpoint:
    path: str
    model: torch.nn.Module
    config: TrainConfig

    @property
    def arch(self) -> str:
        return str(self.config.model.arch)

    @property
    def obs_version(self) -> int:
        return int(self.config.selfplay.obs_version)


class _DuelProgress:
    """Emit a compact, flushed score line every ten completed duel games."""

    def __init__(self, games_total: int, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self.games_total = int(games_total)
        self.clock = clock
        self.started = float(clock())

    def record(self, wins: int, losses: int, ties: int) -> None:
        completed = int(wins) + int(losses) + int(ties)
        if completed == 0 or completed % 10 != 0:
            return
        elapsed = float(self.clock()) - self.started
        games_per_hour = 3600.0 * completed / elapsed if elapsed > 0.0 else 0.0
        print(
            f"{completed}/{self.games_total} games, a {wins}W-{losses}L-{ties}T, "
            f"{games_per_hour:.0f} games/hr",
            flush=True,
        )


def _require_servable_observations(a_version: int, b_version: int) -> int:
    """Return the runner version or raise before allocating a game runner."""
    versions = (int(a_version), int(b_version))
    if 1 in versions:
        raise ValueError(
            "obs-v1 checkpoints cannot be duelled: only equal obs-v2/obs-v3 "
            "pairs and the exact obs-v3 runner to obs-v2 model downgrade are supported"
        )
    if any(version not in (2, 3) for version in versions):
        raise ValueError(f"unsupported checkpoint observation versions: obs-v{versions[0]} vs obs-v{versions[1]}")
    runner_version = max(versions)
    if versions[0] != versions[1] and set(versions) != {2, 3}:
        raise ValueError(
            f"cannot serve obs-v{versions[0]} against obs-v{versions[1]}; "
            "only equal versions or v3-to-v2 are supported"
        )
    return runner_version


def make_duel_runner_config(
    base: SelfPlayConfig,
    *,
    games: int,
    sims: int,
    kingdoms: str,
    obs_version: int,
    n_games: int = 64,
    max_batch: int = 512,
    honest: bool = False,
) -> SelfPlayConfig:
    """Create uniform evaluation conditions without mutating checkpoint config."""
    if games <= 0:
        raise ValueError("games must be positive")
    if sims <= 0:
        raise ValueError("sims must be positive")
    if kingdoms not in {"random", "fixed"}:
        raise ValueError("kingdoms must be 'random' or 'fixed'")

    config = copy.deepcopy(base)
    config.n_games = max(1, min(int(n_games), int(games)))
    config.sims_per_move = int(sims)
    config.c_puct = 1.25
    config.c_puct_schedule = "fixed"
    config.c_puct_init = 1.25
    config.c_puct_base = 19652.0
    config.dirichlet_frac = 0.0
    config.temp_moves = 0
    config.obs_version = int(obs_version)
    config.kingdom_mode = kingdoms
    config.deep_slice_fraction = 0.0
    config.deep_slice_sims = 0
    config.opening_templates_enabled = False
    config.tree_reuse = False
    config.determinize = "per_decision" if honest else "off"
    # Keep treasure handling/search pruning a fixed game convention rather
    # than inheriting either checkpoint's training-time values.  These are the
    # standard campaign15/campaign18 evaluation settings.
    config.auto_play_treasures = True
    config.prune_treasure_plays = True
    config.max_batch = int(max_batch)
    config.max_tree_nodes = max(4096, int(sims) * 2)
    # Retain enough terminal decisions to classify every normal game end.
    # This remains a bounded record; unlike a replay run nothing is persisted.
    config.max_recorded_moves = max(2048, int(config.max_recorded_moves))
    return config


def _v2_supply_offset() -> int:
    return (
        _OBS_META_SIZE
        + (_OBS_OWN_ZONE_COUNT * int(dz.MAX_SLOTS))
        + ((int(dz.MAX_PLAYERS) - 1) * (_OBS_OPPONENT_BLOCK_SIZE_V1 + (3 * int(dz.MAX_SLOTS))))
    )


def _v2_turn_counter_offset() -> int:
    # turn = supply + all fixed supply blocks + landscape + resource;
    # then phase one-hot, active-player, perspective, and turn counter.
    return (
        _v2_supply_offset()
        + (_MAX_PILES * _OBS_PILE_BLOCK_SIZE)
        + _OBS_LANDSCAPE_SIZE
        + _OBS_RESOURCE_SIZE
        + _OBS_PHASE_COUNT
        + 2
    )


def _classify_record_end(record: dict[str, Any]) -> str:
    """Classify a self-play record's end as Province, piles, or turn cap."""
    observations = np.asarray(record.get("observations"), dtype=np.float32)
    if observations.ndim != 2 or observations.shape[0] == 0 or observations.shape[1] < int(dz.OBS_SIZE_V2):
        raise ValueError("duel record is missing the final obs-v2-compatible observation")
    final = observations[-1]
    # ``advance_turn_machinery`` checks the turn cap before ordinary game-end
    # conditions.  The recorded final decision therefore has turn 199 just
    # before cleanup raises it to the native 200-turn cap.
    if int(final[_v2_turn_counter_offset()]) >= _MAX_TURNS - 1:
        return "trunc"
    supply_offset = _v2_supply_offset()
    kingdom = record.get("kingdom", ())
    pile_count = _BASE_SUPPLY_PILES + len(kingdom)
    supply = final[supply_offset : supply_offset + (pile_count * _OBS_PILE_BLOCK_SIZE)]
    if supply.shape[0] != pile_count * _OBS_PILE_BLOCK_SIZE:
        raise ValueError("duel record has an incomplete supply encoding")
    blocks = supply.reshape(pile_count, _OBS_PILE_BLOCK_SIZE)
    province = blocks[np.isclose(blocks[:, 1], float(dz.DEF_PROVINCE) + 1.0)]
    if province.size and int(province[0, 0]) <= 1:
        return "province"
    if int(np.count_nonzero(blocks[:, 0] == 0.0)) >= 3:
        return "piles"
    return "trunc"


def _duel_stats(match: SeatSwappedMatch) -> DuelStats:
    end_province = 0
    end_piles = 0
    truncated = 0
    for record in match.records:
        outcome = _classify_record_end(record)
        if outcome == "province":
            end_province += 1
        elif outcome == "piles":
            end_piles += 1
        else:
            truncated += 1
    stats = match.stats
    if stats.wins + stats.losses + stats.ties != len(match.records):
        raise RuntimeError("seat-swapped duel did not retain every completed game")
    turn_offset = _v2_turn_counter_offset()
    per_game = []
    for record, a_player in zip(match.records, match.a_players):
        final = np.asarray(record.get("observations"), dtype=np.float32)[-1]
        turns = int(final[turn_offset])
        winner = record.get("winner")
        if winner is None or int(winner) < 0:
            result = "T"
        else:
            result = "W" if int(winner) == int(a_player) else "L"
        per_game.append((int(a_player), turns, result))
    return DuelStats(
        games=len(match.records),
        wins_a=stats.wins,
        wins_b=stats.losses,
        ties=stats.ties,
        truncated=truncated,
        end_province=end_province,
        end_piles=end_piles,
        wall_time=match.wall_time,
        per_game=tuple(per_game),
    )


def duel_loaded_checkpoints(
    a: LoadedDuelCheckpoint,
    b: LoadedDuelCheckpoint,
    *,
    games: int = 200,
    sims: int = 400,
    kingdoms: str = "random",
    seed: int = 0x4455454C,
    device: torch.device,
    n_games: int = 64,
    max_batch: int = 512,
    honest: bool = False,
) -> DuelStats:
    """Duel two already-loaded checkpoint models under fixed eval conditions."""
    runner_obs_version = _require_servable_observations(a.obs_version, b.obs_version)
    runner_config = make_duel_runner_config(
        a.config.selfplay,
        games=games,
        sims=sims,
        kingdoms=kingdoms,
        obs_version=runner_obs_version,
        n_games=n_games,
        max_batch=max_batch,
        honest=honest,
    )
    progress = _DuelProgress(games)
    match = run_seat_swapped_match(
        a.model,
        b.model,
        runner_config,
        games=int(games),
        seed=int(seed),
        device=device,
        determinize=runner_config.determinize,
        progress_callback=progress.record,
    )
    return _duel_stats(match)


def duel_checkpoints(
    checkpoint_a: str | Path,
    checkpoint_b: str | Path,
    *,
    games: int = 200,
    sims: int = 400,
    kingdoms: str = "random",
    seed: int = 0x4455454C,
    device_name: str = "auto",
    n_games: int = 64,
    max_batch: int = 512,
    honest: bool = False,
    legacy_shim: bool = False,
) -> tuple[DuelStats, LoadedDuelCheckpoint, LoadedDuelCheckpoint, torch.device]:
    """Load two self-describing checkpoints and run their seat-swapped duel."""
    device = select_device(device_name)
    seed_everything(int(seed), deterministic=True)
    if legacy_shim:
        model_a, config_a = load_model(checkpoint_a, device, legacy_shim=True)
        model_b, config_b = load_model(checkpoint_b, device, legacy_shim=True)
    else:
        model_a, config_a = load_model(checkpoint_a, device)
        model_b, config_b = load_model(checkpoint_b, device)
    a = LoadedDuelCheckpoint(str(checkpoint_a), model_a, config_a)
    b = LoadedDuelCheckpoint(str(checkpoint_b), model_b, config_b)
    stats = duel_loaded_checkpoints(
        a,
        b,
        games=games,
        sims=sims,
        kingdoms=kingdoms,
        seed=seed,
        device=device,
        n_games=n_games,
        max_batch=max_batch,
        honest=honest,
    )
    return stats, a, b, device


def print_summary(stats: DuelStats) -> None:
    print("games,wins_a,wins_b,ties,truncated,end_province,end_piles,win_pct_a_excl_ties,games_per_hour")
    print(
        f"{stats.games},{stats.wins_a},{stats.wins_b},{stats.ties},{stats.truncated},"
        f"{stats.end_province},{stats.end_piles},{stats.win_pct_a_excl_ties:.2f},{stats.games_per_hour:.2f}"
    )
    # Per-game telemetry for length/seat analysis; grep '^pg,' to extract.
    print("pg,index,a_seat,turns,result")
    for index, (a_seat, turns, result) in enumerate(stats.per_game):
        print(f"pg,{index},{a_seat},{turns},{result}", flush=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic NN-versus-NN checkpoint duel")
    parser.add_argument("--a", required=True, help="checkpoint for model A")
    parser.add_argument("--b", required=True, help="checkpoint for model B")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0x4455454C)
    parser.add_argument("--kingdoms", choices=["random", "fixed"], default="random")
    parser.add_argument("--honest", action="store_true", help="sample an honest hidden-information root per decision")
    parser.add_argument(
        "--legacy-shim",
        action="store_true",
        help="native SelfPlayRunner cannot apply this shim; legacy checkpoints require a generation-1 build",
    )
    args = parser.parse_args(argv)

    stats, a, b, device = duel_checkpoints(
        args.a,
        args.b,
        games=args.games,
        sims=args.sims,
        kingdoms=args.kingdoms,
        seed=args.seed,
        device_name=args.device,
        honest=args.honest,
        legacy_shim=args.legacy_shim,
    )
    print(
        json.dumps(
            {
                "a": {"arch": a.arch, "checkpoint": a.path, "obs_version": a.obs_version},
                "b": {"arch": b.arch, "checkpoint": b.path, "obs_version": b.obs_version},
                "device": device.type,
                "honest": bool(args.honest),
                "legacy_shim": bool(args.legacy_shim),
                "sims": int(args.sims),
            },
            sort_keys=True,
        )
    )
    print_summary(stats)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI smoke command
    raise SystemExit(main())
