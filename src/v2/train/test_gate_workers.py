"""The paired gate can spread its boards over worker processes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from .config import validate_gate_config
from .diagnostics import match_selfplay_config, run_paired_match
from .gating import gate_board_slices, gate_match_seed, run_paired_gate, run_parallel_paired_match
from .test_gating import _paired_gate_config
from .train import build_objects


def test_board_slices_are_contiguous_balanced_and_complete() -> None:
    assert gate_board_slices(10, 3) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert gate_board_slices(4, 8) == [[0], [1], [2], [3]]
    assert gate_board_slices(5, 1) == [[0, 1, 2, 3, 4]]


@pytest.mark.parametrize("value", [0, 65, True, 2.0])
def test_gate_workers_validation(tmp_path: Path, value) -> None:
    cfg = _paired_gate_config(tmp_path)
    assert cfg.gate_workers == 1
    cfg.gate_workers = value
    with pytest.raises(ValueError, match="gate_workers"):
        validate_gate_config(cfg)


def _two_models(tmp_path: Path):
    cfg = _paired_gate_config(tmp_path)
    cfg.gate_games = 24  # 12 boards: several per worker
    device = torch.device("cpu")
    candidate, _, _ = build_objects(cfg, device)
    best, _, _ = build_objects(cfg, device)
    with torch.no_grad():
        for parameter in best.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    return cfg, device, candidate.eval(), best.eval()


def test_parallel_match_plays_the_same_games_as_the_serial_loop(tmp_path: Path) -> None:
    cfg, device, candidate, best = _two_models(tmp_path)
    match_config = match_selfplay_config(cfg.selfplay, sims=int(cfg.gate_sims), determinize=cfg.gate_determinize or None)
    pairs, seed = int(cfg.gate_games) // 2, gate_match_seed(cfg, 1)
    serial = run_paired_match(
        candidate, best, match_config, pairs=pairs, seed=seed, device=device,
        independent_search=True, chunk=int(cfg.gate_chunk),
    )
    ticks: list[tuple[int, int]] = []
    parallel = run_parallel_paired_match(
        candidate, best, match_config, pairs=pairs, seed=seed, device=device,
        workers=3, chunk=int(cfg.gate_chunk), config=cfg, tick=lambda g, e: ticks.append((g, e)),
    )
    # CPU evaluation of each row is independent of its batch here, so the
    # split reproduces the serial games exactly, board by board.
    assert parallel.a_as_seat0 == serial.a_as_seat0
    assert parallel.a_as_seat1 == serial.a_as_seat1
    assert sorted(parallel.turns) == sorted(serial.turns)
    assert parallel.truncated == serial.truncated
    assert parallel.decisions == serial.decisions and parallel.non_greedy == serial.non_greedy
    assert parallel.summary()["a_score"] == serial.summary()["a_score"]
    assert parallel.summary()["paired_se"] == serial.summary()["paired_se"]
    assert ticks and ticks[-1][0] == 2 * pairs


def test_gate_with_workers_reports_every_game(tmp_path: Path) -> None:
    cfg, device, candidate, best = _two_models(tmp_path)
    cfg.gate_workers = 2
    lines: list[str] = []
    outcome = run_paired_gate(candidate, best, cfg, generation=1, device=device, log=lines.append)
    assert outcome.stats.wins + outcome.stats.losses + outcome.stats.ties == cfg.gate_games
    assert any("2 worker processes" in line for line in lines)


def test_worker_failure_fails_the_match(tmp_path: Path) -> None:
    cfg, device, candidate, best = _two_models(tmp_path)
    match_config = match_selfplay_config(cfg.selfplay, sims=int(cfg.gate_sims), determinize=cfg.gate_determinize or None)
    with pytest.raises(Exception):
        run_parallel_paired_match(
            candidate, best, match_config, pairs=4, seed=1, device=torch.device("cpu"),
            workers=2, chunk=0, config=cfg,
        )


def test_resume_follows_the_requested_gate_workers(monkeypatch, tmp_path: Path) -> None:
    """gate_workers is a speed knob: a resumed run must use the requested value."""
    from . import gating
    from .train import run_training

    calls: list[int] = []
    original = gating.run_parallel_paired_match

    def recording(*args, **kwargs):
        calls.append(int(kwargs["workers"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(gating, "run_parallel_paired_match", recording)
    first = _paired_gate_config(tmp_path, generations=1)
    run_training(first)
    assert calls == []
    resumed = _paired_gate_config(tmp_path, generations=2)
    resumed.gate_workers = 2
    resumed.selfplay.generation_cutoff_fraction = 0.75
    run_training(resumed, resume=str(Path(first.checkpoint_dir) / "gen_0001.pt"))
    assert calls == [2]


def test_server_gate_plays_the_same_games_as_the_serial_loop(tmp_path: Path) -> None:
    cfg, device, candidate, best = _two_models(tmp_path)
    cfg.gate_server = True
    cfg.server_device = "cpu"
    cfg.server_compile = False
    cfg.server_autocast_bf16 = False
    cfg.server_shards = 2
    validate_gate_config(cfg)
    match_config = match_selfplay_config(cfg.selfplay, sims=int(cfg.gate_sims), determinize=cfg.gate_determinize or None)
    pairs, seed = int(cfg.gate_games) // 2, gate_match_seed(cfg, 1)
    serial = run_paired_match(
        candidate, best, match_config, pairs=pairs, seed=seed, device=device,
        independent_search=True, chunk=int(cfg.gate_chunk),
    )
    ticks: list[tuple[int, int]] = []
    served = run_parallel_paired_match(
        candidate, best, match_config, pairs=pairs, seed=seed, device=device,
        workers=3, chunk=int(cfg.gate_chunk), config=cfg, tick=lambda g, e: ticks.append((g, e)),
    )
    assert served.a_as_seat0 == serial.a_as_seat0
    assert served.a_as_seat1 == serial.a_as_seat1
    assert served.turns and sorted(served.turns) == sorted(serial.turns)
    assert served.nn_evals > 0 and ticks


@pytest.mark.parametrize("field", ["gate_server", "gate_bf16"])
def test_gate_server_flags_must_be_boolean(tmp_path: Path, field: str) -> None:
    cfg = _paired_gate_config(tmp_path)
    setattr(cfg, field, 1)
    with pytest.raises(ValueError, match=field):
        validate_gate_config(cfg)
