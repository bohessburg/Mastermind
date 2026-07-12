from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import dominion_v2_py as dz
except ModuleNotFoundError as exc:  # pragma: no cover - gives a clearer CLI error
    raise SystemExit("dominion_v2_py not found; run with PYTHONPATH=build") from exc

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.v2.train.config import TrainConfig, _merge_dataclass
    from src.v2.train.model import DominionNet, count_parameters
    from src.v2.train.observation import obs_size_for_version, obs_version_for_checkpoint
    from src.v2.train.train import load_full_checkpoint, seed_everything, select_device
else:
    from .config import TrainConfig, _merge_dataclass
    from .model import DominionNet, count_parameters
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


def _load_checkpoint_config(payload: dict[str, Any]) -> TrainConfig:
    cfg = TrainConfig()
    _merge_dataclass(cfg, payload["config"])
    return cfg


def load_model(checkpoint: str | Path, device: torch.device) -> tuple[DominionNet, TrainConfig]:
    payload = load_full_checkpoint(checkpoint, device)
    cfg = _load_checkpoint_config(payload)
    # The saved first-layer width, rather than an optional config key, is the
    # source of truth for legacy checkpoints and future layout migrations.
    cfg.selfplay.obs_version = obs_version_for_checkpoint(payload)
    model = DominionNet(
        obs_size_for_version(cfg.selfplay.obs_version),
        dz.ACTION_SPACE_SIZE,
        cfg.model.hidden_sizes,
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
    if normalized == "bigmoney":
        return dz.EvalScriptedBotKind.BigMoney
    if normalized == "heuristic":
        return dz.EvalScriptedBotKind.Heuristic
    if normalized == "random":
        return dz.EvalScriptedBotKind.Random
    if normalized == "mcts":
        return dz.EvalScriptedBotKind.Mcts
    raise ValueError(f"unknown opponent: {name}")


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
):
    parallel_games = max(1, min(int(n_games), int(games)))
    return dz.EvalRunnerConfig(
        n_games=parallel_games,
        sims_per_move=int(sims),
        c_puct=float(c_puct),
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
    fixed_kingdom: list[str] | None = None,
    max_tree_nodes: int = 4096,
    auto_play_treasures: bool = False,
    prune_treasure_plays: bool = False,
    obs_version: int = 1,
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
            opponent,
            games,
            sims,
            kingdoms,
            seed,
            n_games,
            max_batch,
            c_puct,
            fixed_kingdom,
            max_tree_nodes,
            auto_play_treasures,
            prune_treasure_plays,
            obs_version,
        )
    )
    model.eval()
    start = time.perf_counter()
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
) -> EvalStats:
    device = select_device(device_name)
    seed_everything(seed, deterministic=device.type == "cpu")
    model, cfg = load_model(checkpoint, device)
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
        fixed_kingdom=cfg.selfplay.fixed_kingdom,
        max_tree_nodes=cfg.selfplay.max_tree_nodes,
        auto_play_treasures=auto_play_treasures,
        prune_treasure_plays=prune_treasure_plays,
        obs_version=cfg.selfplay.obs_version,
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
    parser.add_argument("--opponent", default="engine", choices=["engine", "bigmoney", "heuristic", "random", "mcts"])
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--kingdoms", default="random", choices=["random", "fixed"])
    parser.add_argument("--seed", type=int, default=0x4556414C)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-games", type=int, default=64)
    parser.add_argument("--max-batch", type=int, default=512)
    parser.add_argument("--auto-play-treasures", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prune-treasure-plays", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ladder", action="store_true")
    parser.add_argument("--ladder-random-games", type=int, default=40)
    parser.add_argument("--ladder-bigmoney-games", type=int, default=100)
    parser.add_argument("--ladder-heuristic-games", type=int, default=100)
    parser.add_argument("--ladder-engine-games", type=int, default=200)
    parser.add_argument("--ladder-mcts-games", type=int, default=0)
    args = parser.parse_args(argv)

    device = select_device(args.device)
    seed_everything(args.seed, deterministic=device.type == "cpu")
    model, cfg = load_model(args.checkpoint, device)
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
                    fixed_kingdom=cfg.selfplay.fixed_kingdom,
                    max_tree_nodes=cfg.selfplay.max_tree_nodes,
                    auto_play_treasures=auto_play_treasures,
                    prune_treasure_plays=prune_treasure_plays,
                    obs_version=cfg.selfplay.obs_version,
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
                fixed_kingdom=cfg.selfplay.fixed_kingdom,
                max_tree_nodes=cfg.selfplay.max_tree_nodes,
                auto_play_treasures=auto_play_treasures,
                prune_treasure_plays=prune_treasure_plays,
                obs_version=cfg.selfplay.obs_version,
            )
        )
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
