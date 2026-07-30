"""Deterministic honest-information checkpoint evaluation.

Unlike the native EvalRunner and SelfPlayRunner evaluators, this module drives
one visible game at a time.  Neural seats therefore make every decision with
the same determinized ``DecisionSearcher`` used by the serving stack.

Run from the repository root, for example::

    PYTHONPATH=build .venv/bin/python -m src.v2.train.honest_eval \
        --a checkpoints/remote/campaign15/gen_0045.pt --opponent engine3 \
        --games 200 --sims 400 --determinizations 2 --out results.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import time
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any

import numpy as np

try:
    import dominion_v2_py as dz
except ModuleNotFoundError as exc:  # pragma: no cover - clearer CLI failure
    raise SystemExit("dominion_v2_py not found; run with PYTHONPATH=build") from exc

from src.v2.arena.bot.policy import NNPolicy, choose_nnmcts_action, load_policy
from src.v2.train.evaluate import classify_game_end
from src.v2.web.server.defs import kingdom_def_ids


_UINT64_MASK = (1 << 64) - 1
_SCRIPTED_OPPONENTS = ("engine3", "engine2", "bigmoney", "thinner")


@dataclass(frozen=True)
class HonestEvalConfig:
    """Pickle-safe configuration shared by spawned game workers."""

    a: str
    b: str | None
    opponent: str | None
    games: int
    sims_a: int
    determinizations_a: int
    sims_b: int
    determinizations_b: int
    workers: int
    device: str
    seed: int
    torch_threads: int
    legacy_shim: bool

    @property
    def first_half(self) -> int:
        # Keep the duel.py convention: an odd game goes in A's first (seat-0)
        # half rather than silently dropping a game from either block.
        return (self.games + 1) // 2


@dataclass(frozen=True)
class GameResult:
    index: int
    game_seed: int
    a_seat: int
    turns: int
    result: str
    end_condition: str | None
    wall_time_seconds: float
    error: str | None = None
    a_obs_version: int | None = None
    b_obs_version: int | None = None


@dataclass
class _WorkerState:
    config: HonestEvalConfig
    policy_a: NNPolicy
    policy_b: NNPolicy | None


_worker_state: _WorkerState | None = None
_worker_init_error: str | None = None


def _game_seed(seed: int, index: int) -> int:
    """Derive one stable native-game seed from the CLI seed and game index."""
    return (int(seed) + int(index)) & _UINT64_MASK


def _sample_kingdom(game_seed: int) -> list[int]:
    """Use the web server's seeded random 10-card kingdom convention."""
    pool = kingdom_def_ids()
    if len(pool) < 10:
        raise RuntimeError("implemented kingdom pool contains fewer than ten cards")
    return sorted(random.Random(game_seed).sample(pool, 10))


def _ensure_worker_state(config: HonestEvalConfig) -> _WorkerState:
    """Load a worker's policies lazily once, after its spawn completes."""
    global _worker_state, _worker_init_error
    if _worker_state is not None:
        if _worker_state.config != config:
            raise RuntimeError("honest-eval worker received incompatible configurations")
        return _worker_state
    if _worker_init_error is not None:
        raise RuntimeError(_worker_init_error)

    try:
        # Do this before constructing the models so every CPU worker has the
        # requested, non-oversubscribed Torch thread pool.
        import torch

        torch.set_num_threads(config.torch_threads)
        policy_a = load_policy(
            Path(config.a), device=config.device, legacy_shim=config.legacy_shim
        )
        policy_b = (
            load_policy(Path(config.b), device=config.device, legacy_shim=config.legacy_shim)
            if config.b is not None
            else None
        )
        _worker_state = _WorkerState(config=config, policy_a=policy_a, policy_b=policy_b)
        return _worker_state
    except Exception as exc:
        _worker_init_error = f"worker policy initialization failed: {type(exc).__name__}: {exc}"
        raise RuntimeError(_worker_init_error) from exc


def _is_legal(game: Any, action: int) -> bool:
    return 0 <= int(action) < int(dz.ACTION_SPACE_SIZE) and bool(game.legal_mask()[int(action)])


def _a_seat(config: HonestEvalConfig, index: int) -> int:
    return 0 if index < config.first_half else 1


def _play_game(index: int, config: HonestEvalConfig) -> GameResult:
    """Play one entire game in a spawned worker, converting failures to errors."""
    started = time.perf_counter()
    game_seed = _game_seed(config.seed, index)
    a_seat = _a_seat(config, index)
    turns = 0
    state: _WorkerState | None = None
    game: Any | None = None
    try:
        state = _ensure_worker_state(config)
        kingdom = _sample_kingdom(game_seed)
        game = dz.new_game(dz.Setup(players=2, kingdom=kingdom), game_seed)
        # ScriptedBot instances keep their native decision machinery for one
        # game, exactly as the web Session does for its scripted seats.
        scripted_bot = dz.ScriptedBot(config.opponent) if config.opponent is not None else None

        while not game.game_over():
            seat = int(game.current_decision()["player"])
            if seat not in (0, 1):
                raise RuntimeError(f"game reported invalid active seat {seat}")

            if seat == a_seat:
                action = choose_nnmcts_action(
                    game,
                    seat,
                    state.policy_a,
                    sims=config.sims_a,
                    determinizations=config.determinizations_a,
                )
            elif state.policy_b is not None:
                action = choose_nnmcts_action(
                    game,
                    seat,
                    state.policy_b,
                    sims=config.sims_b,
                    determinizations=config.determinizations_b,
                )
            else:
                if scripted_bot is None:  # Defensive: argparse enforces a counterpart.
                    raise RuntimeError("missing opponent decision policy")
                action = int(scripted_bot.choose(game))

            action = int(action)
            if not _is_legal(game, action):
                raise RuntimeError(f"illegal action from seat {seat}: {action}")
            done = bool(game.step(action))
            if done:
                if not game.game_over():
                    raise RuntimeError("game.step returned terminal before game_over")
                break

        turns = int(game.turn())
        winner = game.winner()
        if winner is None:
            result = "T"
        elif int(winner) == a_seat:
            result = "W"
        else:
            result = "L"
        return GameResult(
            index=index,
            game_seed=game_seed,
            a_seat=a_seat,
            turns=turns,
            result=result,
            end_condition=classify_game_end(game),
            wall_time_seconds=time.perf_counter() - started,
            a_obs_version=state.policy_a.obs_version,
            b_obs_version=state.policy_b.obs_version if state.policy_b is not None else None,
        )
    except Exception as exc:
        if game is not None:
            turns = int(game.turn())
        return GameResult(
            index=index,
            game_seed=game_seed,
            a_seat=a_seat,
            turns=turns,
            result="E",
            end_condition=None,
            wall_time_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
            a_obs_version=state.policy_a.obs_version if state is not None else None,
            b_obs_version=state.policy_b.obs_version if state is not None and state.policy_b is not None else None,
        )


def _play_assigned_games(
    indices: list[int], config: HonestEvalConfig, result_connection: Connection
) -> None:
    """Run a worker's stable game-index assignment and stream each completion."""
    try:
        for index in indices:
            result_connection.send(_play_game(index, config))
    finally:
        result_connection.close()


def _empty_counts() -> dict[str, int]:
    return {"wins_a": 0, "losses_a": 0, "ties": 0, "errors": 0, "games": 0}


def _record_counts(counts: dict[str, int], result: GameResult) -> None:
    counts["games"] += 1
    if result.result == "W":
        counts["wins_a"] += 1
    elif result.result == "L":
        counts["losses_a"] += 1
    elif result.result == "T":
        counts["ties"] += 1
    else:
        counts["errors"] += 1


def _win_pct(counts: dict[str, int]) -> float:
    decisive = counts["wins_a"] + counts["losses_a"]
    return 100.0 * counts["wins_a"] / decisive if decisive else 0.0


def _progress_line(completed: int, config: HonestEvalConfig, overall: dict[str, int], started: float) -> str:
    elapsed = max(time.perf_counter() - started, 1e-9)
    games_per_hour = 3600.0 * completed / elapsed
    remaining_minutes = (config.games - completed) / games_per_hour * 60.0 if games_per_hour else 0.0
    return (
        f"progress: {completed}/{config.games} "
        f"{overall['wins_a']}-{overall['losses_a']}-{overall['ties']} "
        f"{games_per_hour:.1f} games/hr eta {remaining_minutes:.1f} min"
    )


def _result_payload(config: HonestEvalConfig, results: list[GameResult], wall_time: float) -> dict[str, Any]:
    overall = _empty_counts()
    per_seat_half = {"a_seat_0": _empty_counts(), "a_seat_1": _empty_counts()}
    end_conditions = {"province": 0, "piles": 0, "truncated": 0}
    completed_turns: list[int] = []
    for result in results:
        _record_counts(overall, result)
        _record_counts(per_seat_half[f"a_seat_{result.a_seat}"], result)
        if result.end_condition is not None:
            if result.end_condition == "trunc":
                end_conditions["truncated"] += 1
            else:
                end_conditions[result.end_condition] += 1
            completed_turns.append(result.turns)

    a_obs_versions = sorted({result.a_obs_version for result in results if result.a_obs_version is not None})
    b_obs_versions = sorted({result.b_obs_version for result in results if result.b_obs_version is not None})
    return {
        "config": {
            **asdict(config),
            "seat_blocked_first_half_games": config.first_half,
            "kingdom_sampling": "sorted(random.Random(game_seed).sample(kingdom_def_ids(), 10))",
            "a_obs_versions": a_obs_versions,
            "b_obs_versions": b_obs_versions,
        },
        "overall": {**overall, "win_pct_a_excl_ties": _win_pct(overall)},
        "per_seat_half": {
            key: {**counts, "win_pct_a_excl_ties": _win_pct(counts)}
            for key, counts in per_seat_half.items()
        },
        "end_conditions": end_conditions,
        "mean_turns": float(np.mean(completed_turns)) if completed_turns else 0.0,
        "wall_time_seconds": wall_time,
        "games_per_hour": 3600.0 * overall["games"] / wall_time if wall_time > 0.0 else 0.0,
        "games": [asdict(result) for result in results],
    }


def _print_final_summary(payload: dict[str, Any], out_path: Path) -> None:
    config = payload["config"]
    overall = payload["overall"]
    end_conditions = payload["end_conditions"]
    print(f"config: {json.dumps(config, sort_keys=True)}", flush=True)
    print(
        "overall: "
        f"W-L-T {overall['wins_a']}-{overall['losses_a']}-{overall['ties']} "
        f"errors {overall['errors']} win% excl ties {overall['win_pct_a_excl_ties']:.2f}",
        flush=True,
    )
    for half, counts in payload["per_seat_half"].items():
        print(
            f"{half}: W-L-T {counts['wins_a']}-{counts['losses_a']}-{counts['ties']} "
            f"errors {counts['errors']} win% excl ties {counts['win_pct_a_excl_ties']:.2f}",
            flush=True,
        )
    print(
        "end conditions: "
        f"province {end_conditions['province']} piles {end_conditions['piles']} "
        f"truncated {end_conditions['truncated']}",
        flush=True,
    )
    print(f"mean turns: {payload['mean_turns']:.2f}", flush=True)
    print(f"wall time: {payload['wall_time_seconds']:.2f} s", flush=True)
    print(f"games/hr: {payload['games_per_hour']:.2f}", flush=True)
    print(f"results: {out_path}", flush=True)


def run_honest_eval(config: HonestEvalConfig, out_path: str | Path) -> dict[str, Any]:
    """Run the game pool, emit heartbeat telemetry, write, and return JSON data."""
    out = Path(out_path)
    started = time.perf_counter()
    overall = _empty_counts()
    results: list[GameResult] = []
    context = multiprocessing.get_context("spawn")
    worker_count = min(config.workers, config.games)
    assignments = [list(range(worker, config.games, worker_count)) for worker in range(worker_count)]
    workers: list[multiprocessing.Process] = []
    connections: dict[Connection, list[int]] = {}
    for indices in assignments:
        parent_connection, child_connection = context.Pipe(duplex=False)
        worker = context.Process(
            target=_play_assigned_games,
            args=(indices, config, child_connection),
        )
        worker.start()
        child_connection.close()
        workers.append(worker)
        connections[parent_connection] = indices

    received_indices: set[int] = set()
    while connections:
        for connection in wait(connections):
            try:
                result = connection.recv()
            except EOFError:
                connection.close()
                connections.pop(connection)
                continue
            received_indices.add(result.index)
            results.append(result)
            _record_counts(overall, result)
            completed = len(results)
            print(f"pg,{result.index},{result.a_seat},{result.turns},{result.result}", flush=True)
            if completed % 10 == 0:
                print(_progress_line(completed, config, overall, started), flush=True)

    for worker in workers:
        worker.join()
    # A hard worker crash should not turn the run into a process-level crash:
    # preserve the one-result-per-requested-game accounting contract instead.
    for index in range(config.games):
        if index not in received_indices:
            result = GameResult(
                index=index,
                game_seed=_game_seed(config.seed, index),
                a_seat=_a_seat(config, index),
                turns=0,
                result="E",
                end_condition=None,
                wall_time_seconds=0.0,
                error="worker exited before reporting this game",
            )
            results.append(result)
            _record_counts(overall, result)
            completed = len(results)
            print(f"pg,{result.index},{result.a_seat},{result.turns},{result.result}", flush=True)
            if completed % 10 == 0:
                print(_progress_line(completed, config, overall, started), flush=True)

    results.sort(key=lambda result: result.index)
    payload = _result_payload(config, results, time.perf_counter() - started)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_final_summary(payload, out)
    return payload


def _parse_args(argv: list[str] | None = None) -> tuple[HonestEvalConfig, Path]:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint strength with serving-style hidden-information NN-MCTS."
    )
    parser.add_argument("--a", required=True, help="checkpoint for model A")
    counterpart = parser.add_mutually_exclusive_group(required=True)
    counterpart.add_argument("--b", help="checkpoint for model B")
    counterpart.add_argument("--opponent", choices=_SCRIPTED_OPPONENTS, help="native scripted opponent")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--sims", type=int, default=400, help="NN-MCTS simulations for A")
    parser.add_argument("--determinizations", type=int, default=2, help="determinizations for A")
    parser.add_argument("--sims-b", type=int, help="NN-MCTS simulations for B (NN-vs-NN only)")
    parser.add_argument(
        "--determinizations-b", type=int, help="determinizations for B (NN-vs-NN only)"
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--legacy-shim",
        action="store_true",
        help="restore pre-sentinel-fix inputs for encoder-generation-1 checkpoints",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    for name in ("games", "sims", "determinizations", "workers", "torch_threads"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.opponent is not None and (args.sims_b is not None or args.determinizations_b is not None):
        parser.error("--sims-b and --determinizations-b are only valid with --b")
    if args.sims_b is not None and args.sims_b <= 0:
        parser.error("--sims-b must be positive")
    if args.determinizations_b is not None and args.determinizations_b <= 0:
        parser.error("--determinizations-b must be positive")

    return (
        HonestEvalConfig(
            a=str(args.a),
            b=str(args.b) if args.b is not None else None,
            opponent=str(args.opponent) if args.opponent is not None else None,
            games=int(args.games),
            sims_a=int(args.sims),
            determinizations_a=int(args.determinizations),
            sims_b=int(args.sims if args.sims_b is None else args.sims_b),
            determinizations_b=int(
                args.determinizations if args.determinizations_b is None else args.determinizations_b
            ),
            workers=int(args.workers),
            device=str(args.device),
            seed=int(args.seed),
            torch_threads=int(args.torch_threads),
            legacy_shim=bool(args.legacy_shim),
        ),
        args.out,
    )


def main(argv: list[str] | None = None) -> int:
    config, out = _parse_args(argv)
    run_honest_eval(config, out)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke commands
    raise SystemExit(main())
