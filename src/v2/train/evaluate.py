from __future__ import annotations

import argparse
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
except ModuleNotFoundError as exc:  # pragma: no cover - gives a clearer CLI error
    raise SystemExit("dominion_v2_py not found; run with PYTHONPATH=build") from exc

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.v2.encoder_compat import require_native_runner_encoder_compatibility
    from src.v2.train.config import TrainConfig, _merge_dataclass
    from src.v2.train.model import build_model, count_parameters
    from src.v2.train.observation import obs_size_for_version, obs_version_for_checkpoint
    from src.v2.train.train import load_full_checkpoint, seed_everything, select_device
else:
    from src.v2.encoder_compat import require_native_runner_encoder_compatibility
    from .config import TrainConfig, _merge_dataclass
    from .model import build_model, count_parameters
    from .observation import obs_size_for_version, obs_version_for_checkpoint
    from .train import load_full_checkpoint, seed_everything, select_device


@dataclass
class EvalStats:
    opponent: str
    games: int
    wins: int
    losses: int
    ties: int
    truncated: int
    end_province: int
    end_piles: int
    end_trunc: int
    wall_time: float

    @property
    def decisive(self) -> int:
        return self.wins + self.losses

    @property
    def win_pct_excl_ties(self) -> float:
        return 100.0 * self.wins / self.decisive if self.decisive > 0 else 0.0

    @property
    def games_per_hour(self) -> float:
        return 3600.0 * self.games / self.wall_time if self.wall_time > 0.0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "opponent": self.opponent,
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "truncated": self.truncated,
            "end_province": self.end_province,
            "end_piles": self.end_piles,
            "end_trunc": self.end_trunc,
            "win_pct_excl_ties": self.win_pct_excl_ties,
            "games_per_hour": self.games_per_hour,
            "wall_time": self.wall_time,
        }


class _EvalProgress:
    """Emit a compact, flushed score line every 25 completed evaluation games."""

    def __init__(self, games_total: int, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self.games_total = int(games_total)
        self.clock = clock
        self.started = float(clock())
        self.next_report = 25

    def record(self, completed: int, wins: int, losses: int, ties: int) -> None:
        completed = int(completed)
        if completed < self.next_report:
            return
        elapsed = float(self.clock()) - self.started
        games_per_hour = 3600.0 * completed / elapsed if elapsed > 0.0 else 0.0
        print(
            f"{completed}/{self.games_total} games, nn {wins}W-{losses}L-{ties}T, "
            f"{games_per_hour:.0f} games/hr",
            flush=True,
        )
        self.next_report = ((completed // 25) + 1) * 25


def _load_checkpoint_config(payload: dict[str, Any]) -> TrainConfig:
    cfg = TrainConfig()
    _merge_dataclass(cfg, payload["config"])
    return cfg


def load_model(
    checkpoint: str | Path,
    device: torch.device,
    *,
    legacy_shim: bool = False,
) -> tuple[torch.nn.Module, TrainConfig]:
    """Load a model for EvalRunner after enforcing native encoder compatibility.

    EvalRunner owns the encoded leaf buffer, so a Python legacy shim cannot
    run between encoding and model evaluation.  Generation-1 checkpoints must
    therefore be evaluated by a pre-sentinel-fix build.
    """
    payload = load_full_checkpoint(checkpoint, device)
    require_native_runner_encoder_compatibility(payload, checkpoint, legacy_shim=legacy_shim)
    cfg = _load_checkpoint_config(payload)
    # Legacy MLP checkpoints infer their layout from the first-layer width.
    # CardTokenNet records its v2/v3 tokenizer layout in model metadata; old
    # transformer checkpoints omitted that field and are v2 by definition.
    if cfg.model.arch == "card_transformer":
        model_obs_version = 2 if cfg.model.obs_version is None else int(cfg.model.obs_version)
        if int(cfg.selfplay.obs_version) != model_obs_version or model_obs_version not in (2, 3):
            raise ValueError("card_transformer checkpoint model.obs_version must match selfplay.obs_version (2 or 3)")
        cfg.model.obs_version = model_obs_version
    else:
        cfg.selfplay.obs_version = obs_version_for_checkpoint(payload)
    model = build_model(
        cfg.model,
        obs_size_for_version(cfg.selfplay.obs_version),
        dz.ACTION_SPACE_SIZE,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg


def _kingdom_mode(mode: str):
    normalized = mode.lower()
    if normalized == "fixed":
        return dz.SelfPlayKingdomMode.Fixed
    if normalized == "random":
        return dz.SelfPlayKingdomMode.Random
    raise ValueError(f"unknown kingdom mode: {mode}")


def _opponent_kind(name: str):
    normalized = name.lower()
    if normalized == "engine":
        return dz.EvalScriptedBotKind.Engine
    if normalized == "engine2":
        return dz.EvalScriptedBotKind.EngineV2
    if normalized == "engine3":
        return dz.EvalScriptedBotKind.EngineV3
    if normalized == "thinner":
        return dz.EvalScriptedBotKind.Thinner
    if normalized == "bigmoney":
        return dz.EvalScriptedBotKind.BigMoney
    if normalized == "heuristic":
        return dz.EvalScriptedBotKind.Heuristic
    if normalized == "random":
        return dz.EvalScriptedBotKind.Random
    if normalized == "mcts":
        return dz.EvalScriptedBotKind.Mcts
    raise ValueError(f"unknown opponent: {name}")


def _eval_determinize_mode(honest: bool):
    return (
        dz.SelfPlayDeterminizeMode.PerDecision
        if honest
        else dz.SelfPlayDeterminizeMode.Off
    )


def classify_game_end(game: Any) -> str:
    """Classify a completed binding Game without adding engine end-state APIs."""
    if game.truncated():
        return "trunc"
    supply = {int(def_id): int(count) for def_id, count in game.supply()}
    if supply.get(int(dz.DEF_PROVINCE), 1) == 0:
        return "province"
    if sum(count == 0 for count in supply.values()) >= 3:
        return "piles"
    raise ValueError("completed eval game has no recognized end condition")


def make_eval_runner_config(
    opponent: str,
    games: int,
    sims: int,
    kingdoms: str,
    seed: int,
    n_games: int,
    max_batch: int,
    c_puct: float,
    fixed_kingdom: list[str],
    max_tree_nodes: int,
    auto_play_treasures: bool = False,
    prune_treasure_plays: bool = False,
    obs_version: int = 1,
    c_puct_schedule: str = "fixed",
    c_puct_init: float = 1.25,
    c_puct_base: float = 19652.0,
    honest: bool = False,
):
    parallel_games = max(1, min(int(n_games), int(games)))
    return dz.EvalRunnerConfig(
        n_games=parallel_games,
        sims_per_move=int(sims),
        c_puct=float(c_puct),
        c_puct_schedule=c_puct_schedule,
        c_puct_init=float(c_puct_init),
        c_puct_base=float(c_puct_base),
        max_batch=int(max_batch),
        seed=int(seed),
        target_games=int(games),
        kingdom_mode=_kingdom_mode(kingdoms),
        kingdom=fixed_kingdom,
        max_tree_nodes=int(max_tree_nodes),
        opponent=_opponent_kind(opponent),
        retain_finished_games=True,
        auto_play_treasures=bool(auto_play_treasures),
        prune_treasure_plays=bool(prune_treasure_plays),
        obs_version=int(obs_version),
        determinize=_eval_determinize_mode(bool(honest)),
    )


def evaluate_model(
    model: torch.nn.Module,
    *,
    opponent: str = "engine",
    games: int = 200,
    sims: int = 400,
    kingdoms: str = "random",
    seed: int = 0x4556414C,
    device: torch.device,
    n_games: int = 64,
    max_batch: int = 512,
    c_puct: float = 1.25,
    c_puct_schedule: str = "fixed",
    c_puct_init: float = 1.25,
    c_puct_base: float = 19652.0,
    fixed_kingdom: list[str] | None = None,
    max_tree_nodes: int = 4096,
    auto_play_treasures: bool = False,
    prune_treasure_plays: bool = False,
    obs_version: int = 1,
    honest: bool = False,
) -> EvalStats:
    if fixed_kingdom is None:
        fixed_kingdom = [
            "Village",
            "Smithy",
            "Market",
            "Festival",
            "Laboratory",
            "Cellar",
            "Chapel",
            "Militia",
            "Witch",
            "Moat",
        ]
    runner = dz.EvalRunner(
        make_eval_runner_config(
            opponent=opponent,
            games=games,
            sims=sims,
            kingdoms=kingdoms,
            seed=seed,
            n_games=n_games,
            max_batch=max_batch,
            c_puct=c_puct,
            fixed_kingdom=fixed_kingdom,
            max_tree_nodes=max_tree_nodes,
            auto_play_treasures=auto_play_treasures,
            prune_treasure_plays=prune_treasure_plays,
            obs_version=obs_version,
            c_puct_schedule=c_puct_schedule,
            c_puct_init=c_puct_init,
            c_puct_base=c_puct_base,
            honest=honest,
        )
    )
    model.eval()
    start = time.perf_counter()
    progress = _EvalProgress(games)
    idle = 0
    with torch.no_grad():
        while runner.games_completed() < games:
            obs, masks = runner.collect_leaves(max_batch)
            batch = int(obs.shape[0])
            if batch == 0:
                idle += 1
                if idle > 10000:
                    raise RuntimeError("eval runner made no progress")
                continue
            idle = 0
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=device)
            logits, values = model.evaluate(obs_tensor, mask_tensor)
            logits_np = logits.detach().cpu().numpy().astype(np.float32, copy=False)
            values_np = values.detach().cpu().numpy().astype(np.float32, copy=False)
            runner.provide_evaluations(values_np, logits_np)
            completed = int(runner.games_completed())
            if completed >= progress.next_report:
                running = runner.result()
                progress.record(
                    completed,
                    int(running["nn_wins"]),
                    int(running["scripted_wins"]),
                    int(running["ties"]),
                )

    result = runner.result()
    finished_games = runner.finished_games()
    if len(finished_games) != int(result["games"]):
        raise RuntimeError("eval runner did not retain every completed game for end forensics")
    end_province = 0
    end_piles = 0
    end_trunc = 0
    for game in finished_games:
        outcome = classify_game_end(game)
        if outcome == "province":
            end_province += 1
        elif outcome == "piles":
            end_piles += 1
        else:
            end_trunc += 1
    wall = time.perf_counter() - start
    return EvalStats(
        opponent=opponent,
        games=int(result["games"]),
        wins=int(result["nn_wins"]),
        losses=int(result["scripted_wins"]),
        ties=int(result["ties"]),
        truncated=int(result["truncated"]),
        end_province=end_province,
        end_piles=end_piles,
        end_trunc=end_trunc,
        wall_time=wall,
    )


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    opponent: str = "engine",
    games: int = 200,
    sims: int = 400,
    kingdoms: str = "random",
    seed: int = 0x4556414C,
    device_name: str = "auto",
    n_games: int = 64,
    max_batch: int = 512,
    auto_play_treasures: bool | None = None,
    prune_treasure_plays: bool | None = None,
    honest: bool = False,
    legacy_shim: bool = False,
) -> EvalStats:
    device = select_device(device_name)
    seed_everything(seed, deterministic=device.type == "cpu")
    model, cfg = load_model(checkpoint, device, legacy_shim=legacy_shim)
    if auto_play_treasures is None:
        auto_play_treasures = cfg.selfplay.auto_play_treasures
    if prune_treasure_plays is None:
        prune_treasure_plays = cfg.selfplay.prune_treasure_plays
    return evaluate_model(
        model,
        opponent=opponent,
        games=games,
        sims=sims,
        kingdoms=kingdoms,
        seed=seed,
        device=device,
        n_games=n_games,
        max_batch=max_batch,
        c_puct=cfg.selfplay.c_puct,
        c_puct_schedule=cfg.selfplay.c_puct_schedule,
        c_puct_init=cfg.selfplay.c_puct_init,
        c_puct_base=cfg.selfplay.c_puct_base,
        fixed_kingdom=cfg.selfplay.fixed_kingdom,
        max_tree_nodes=cfg.selfplay.max_tree_nodes,
        auto_play_treasures=auto_play_treasures,
        prune_treasure_plays=prune_treasure_plays,
        obs_version=cfg.selfplay.obs_version,
        honest=honest,
    )


def print_table(rows: list[EvalStats]) -> None:
    print(
        "opponent,games,wins,losses,ties,truncated,end_province,end_piles,end_trunc,"
        "win_pct_excl_ties,games_per_hour"
    )
    for row in rows:
        print(
            f"{row.opponent},{row.games},{row.wins},{row.losses},{row.ties},{row.truncated},"
            f"{row.end_province},{row.end_piles},{row.end_trunc},"
            f"{row.win_pct_excl_ties:.2f},{row.games_per_hour:.2f}"
        )


def ladder_counts(args: argparse.Namespace) -> list[tuple[str, int]]:
    counts = [
        ("random", args.ladder_random_games),
        ("bigmoney", args.ladder_bigmoney_games),
        ("heuristic", args.ladder_heuristic_games),
        ("engine", args.ladder_engine_games),
    ]
    if args.ladder_mcts_games > 0:
        counts.append(("mcts", args.ladder_mcts_games))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--opponent",
        default="engine",
        choices=["engine", "engine2", "engine3", "thinner", "bigmoney", "heuristic", "random", "mcts"],
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--kingdoms", default="random", choices=["random", "fixed"])
    parser.add_argument("--seed", type=int, default=0x4556414C)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-games", type=int, default=64)
    parser.add_argument("--max-batch", type=int, default=512)
    parser.add_argument("--auto-play-treasures", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prune-treasure-plays", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--honest", action="store_true", help="sample an honest hidden-information root per NN decision")
    parser.add_argument(
        "--legacy-shim",
        action="store_true",
        help="native EvalRunner cannot apply this shim; legacy checkpoints require a generation-1 build",
    )
    parser.add_argument("--ladder", action="store_true")
    parser.add_argument("--ladder-random-games", type=int, default=40)
    parser.add_argument("--ladder-bigmoney-games", type=int, default=100)
    parser.add_argument("--ladder-heuristic-games", type=int, default=100)
    parser.add_argument("--ladder-engine-games", type=int, default=200)
    parser.add_argument("--ladder-mcts-games", type=int, default=0)
    args = parser.parse_args(argv)

    device = select_device(args.device)
    seed_everything(args.seed, deterministic=device.type == "cpu")
    model, cfg = load_model(args.checkpoint, device, legacy_shim=args.legacy_shim)
    auto_play_treasures = (
        cfg.selfplay.auto_play_treasures
        if args.auto_play_treasures is None
        else args.auto_play_treasures
    )
    prune_treasure_plays = (
        cfg.selfplay.prune_treasure_plays
        if args.prune_treasure_plays is None
        else args.prune_treasure_plays
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "device": device.type,
                "parameters": count_parameters(model),
                "sims": args.sims,
                "kingdoms": args.kingdoms,
                "honest": bool(args.honest),
                "legacy_shim": bool(args.legacy_shim),
            },
            sort_keys=True,
        )
    )

    rows: list[EvalStats] = []
    if args.ladder:
        for index, (opponent, games) in enumerate(ladder_counts(args)):
            rows.append(
                evaluate_model(
                    model,
                    opponent=opponent,
                    games=games,
                    sims=args.sims,
                    kingdoms=args.kingdoms,
                    seed=args.seed + (index * 0x10001),
                    device=device,
                    n_games=args.n_games,
                    max_batch=args.max_batch,
                    c_puct=cfg.selfplay.c_puct,
                    c_puct_schedule=cfg.selfplay.c_puct_schedule,
                    c_puct_init=cfg.selfplay.c_puct_init,
                    c_puct_base=cfg.selfplay.c_puct_base,
                    fixed_kingdom=cfg.selfplay.fixed_kingdom,
                    max_tree_nodes=cfg.selfplay.max_tree_nodes,
                    auto_play_treasures=auto_play_treasures,
                    prune_treasure_plays=prune_treasure_plays,
                    obs_version=cfg.selfplay.obs_version,
                    honest=args.honest,
                )
            )
    else:
        rows.append(
            evaluate_model(
                model,
                opponent=args.opponent,
                games=args.games,
                sims=args.sims,
                kingdoms=args.kingdoms,
                seed=args.seed,
                device=device,
                n_games=args.n_games,
                max_batch=args.max_batch,
                c_puct=cfg.selfplay.c_puct,
                c_puct_schedule=cfg.selfplay.c_puct_schedule,
                c_puct_init=cfg.selfplay.c_puct_init,
                c_puct_base=cfg.selfplay.c_puct_base,
                fixed_kingdom=cfg.selfplay.fixed_kingdom,
                max_tree_nodes=cfg.selfplay.max_tree_nodes,
                auto_play_treasures=auto_play_treasures,
                prune_treasure_plays=prune_treasure_plays,
                obs_version=cfg.selfplay.obs_version,
                honest=args.honest,
            )
        )
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
