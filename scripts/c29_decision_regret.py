"""c29 decision regret: is the search's move better than the prior's move?

Self-play only improves a net if the search's visit targets are better than
the policy that produced them. This measures that directly, without a gate
and without the value head:

1. Play games with one net under a search config (greedy, no noise) and keep
   the live state at every decision.
2. At sampled decisions, take the search's move, the prior's move, and the
   top-K prior moves as candidates.
3. Score every candidate with policy playouts: sample a world the deciding
   player cannot distinguish from the real one, play the candidate, then let
   the net's policy play both seats to the end. All candidates of a position
   share the same sampled worlds (common random numbers), so differences are
   paired.

Reported, in win probability for the deciding player:
  search - prior   on decisions where the search overrode the prior. This is
                   the improvement the visit targets teach. <= 0 means
                   self-play targets cannot help.
  headroom         best candidate (picked on half the playouts, scored on the
                   other half) minus the prior's / the search's move. What a
                   one-ply honest-rollout improvement operator would add.
  peeking          the same search - prior difference scored in the TRUE world
                   (hidden cards as they really are). True-world gain without
                   honest gain means the search is exploiting hidden
                   information it should not see (determinize off).
  value fit        correlation of the value head with the playout value of the
                   prior's move.

Usage:
  python scripts/c29_decision_regret.py --tag bca_s128 \
      --model checkpoints/campaign25/bc_a.pt --sims 128 --games 64 \
      --positions 400 --playouts 256
Run it again with --determinize per_decision (honest search) or another
--sims to compare search settings.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
import dominion_v2_py as dz  # noqa: E402

from src.v2.train.config import load_config  # noqa: E402
from src.v2.train.diagnostics import match_selfplay_config, pick_device, play_manifest_games  # noqa: E402
from src.v2.train.evaluate import load_model  # noqa: E402
from src.v2.train.observation import (  # noqa: E402
    model_encoder_generation,
    model_observation_version,
    observations_for_model,
    obs_version_for_width,
)

HEARTBEAT_S = 20.0
# Positions have differing candidate counts; unused slots are NaN by design.
warnings.filterwarnings("ignore", message="Mean of empty slice")


def masked_logits(model, obs: np.ndarray, masks: np.ndarray, device, batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    source = obs_version_for_width(int(obs.shape[-1]))
    logits_out = np.empty((obs.shape[0], dz.ACTION_SPACE_SIZE), dtype=np.float32)
    values_out = np.empty((obs.shape[0],), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, obs.shape[0], batch):
            stop = min(start + batch, obs.shape[0])
            adapted = observations_for_model(
                obs[start:stop], source, model_observation_version(model, source), model_encoder_generation(model)
            )
            logits, values = model.evaluate(
                torch.as_tensor(adapted, dtype=torch.float32, device=device),
                torch.as_tensor(masks[start:stop], dtype=torch.bool, device=device),
            )
            logits_out[start:stop] = logits.float().cpu().numpy()
            values_out[start:stop] = values.float().cpu().numpy()
    return logits_out, values_out


def mean_se(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "se": float("nan")}
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": float(values.mean()), "se": se}


def fmt(stat: dict[str, float]) -> str:
    return f"{100.0 * stat['mean']:+.2f}±{100.0 * stat['se']:.2f} pts (n={stat['n']})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sims", type=int, default=128, help="search budget when generating positions")
    parser.add_argument("--determinize", choices=("off", "per_decision", "per_turn"), default=None,
                        help="search information model (default: the config's)")
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--positions", type=int, default=400)
    parser.add_argument("--overrides-only", action="store_true",
                        help="sample only decisions where the search overrode the prior")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--playouts", type=int, default=256, help="per candidate, per world mode")
    parser.add_argument("--playout-temperature", type=float, default=0.0, help="0 = argmax policy")
    parser.add_argument("--no-true-world", action="store_true", help="skip the peeking measurement")
    parser.add_argument("--slots", type=int, default=2048, help="concurrent playouts")
    parser.add_argument("--chunk", type=int, default=64, help="concurrent position-generation games")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--config", default="configs/run_c28.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="bench/c29_regret")
    args = parser.parse_args()
    if args.playouts < 2 or args.playouts % 2:
        parser.error("--playouts must be an even number (split-half headroom)")

    device = pick_device(args.device)
    config = load_config(args.config)
    model, _ = load_model(Path(args.model), device)
    search = match_selfplay_config(config.selfplay, sims=args.sims, determinize=args.determinize)
    search.max_recorded_moves = 4096
    print(f"[{args.tag}] device={device} model={args.model} sims={args.sims} determinize={search.determinize} "
          f"games={args.games} positions={args.positions} top_k={args.top_k} playouts={args.playouts}", flush=True)

    # 1. Positions.
    started = time.time()
    records: list[dict] = []
    for first in range(0, args.games, args.chunk):
        indices = list(range(first, min(first + args.chunk, args.games)))
        chunk_records, _ = play_manifest_games(
            (model, model), search, seed=args.seed, device=device, game_indices=indices,
            independent_search=False, record_states=True,
        )
        records.extend(chunk_records)
        print(f"[{args.tag}] positions: {len(records)}/{args.games} games {time.time() - started:.0f}s", flush=True)

    obs_rows, mask_rows, visit_rows, states, phase_rows = [], [], [], [], []
    for record in records:
        obs = np.asarray(record["observations"], dtype=np.float32)
        count = min(obs.shape[0], len(record["states"]))
        obs_rows.append(obs[:count])
        mask_rows.append(np.asarray(record["legal_mask"], dtype=bool)[:count])
        visit_rows.append(np.asarray(record["policy_targets"], dtype=np.float32)[:count])
        states.extend(record["states"][:count])
        phase_rows.append(np.arange(count, dtype=np.float32) / max(count, 1))
    all_obs = np.concatenate(obs_rows)
    all_masks = np.concatenate(mask_rows)
    all_visits = np.concatenate(visit_rows)
    all_phase = np.concatenate(phase_rows)
    logits, values = masked_logits(model, all_obs, all_masks, device)
    prior_move = logits.argmax(axis=-1)
    search_move = all_visits.argmax(axis=-1)
    overridden = prior_move != search_move
    decisions = int(all_obs.shape[0])
    print(f"[{args.tag}] {decisions} decisions, search overrides the prior on "
          f"{int(overridden.sum())} ({100.0 * overridden.mean():.1f}%)", flush=True)

    rng = np.random.default_rng(args.seed)
    pool = np.flatnonzero(overridden) if args.overrides_only else np.arange(decisions)
    chosen = np.sort(rng.choice(pool, size=min(args.positions, pool.size), replace=False))
    n_pos = int(chosen.size)

    # 2. Candidates: search move, prior move, then the remaining top-K priors.
    slots_per_pos = args.top_k + 2
    cand = np.full((n_pos, slots_per_pos), -1, dtype=np.int64)
    for row, index in enumerate(chosen):
        order = [int(search_move[index]), int(prior_move[index])]
        legal = np.flatnonzero(all_masks[index])
        ranked = legal[np.argsort(-logits[index, legal], kind="stable")]
        for action in ranked[: args.top_k]:
            if int(action) not in order:
                order.append(int(action))
        unique = list(dict.fromkeys(order))
        cand[row, : len(unique)] = unique
    n_cand = (cand >= 0).sum(axis=1)

    # 3. Playouts.
    runner = dz.RolloutRunner(
        n_slots=args.slots,
        obs_version=int(search.obs_version),
        auto_play_treasures=bool(search.auto_play_treasures),
        prune_treasure_plays=bool(search.prune_treasure_plays),
        max_turns=int(search.max_turns),
        temperature=float(args.playout_temperature),
    )
    position_ids = np.asarray([runner.add_position(states[index]) for index in chosen], dtype=np.uint32)
    modes = ("honest",) if args.no_true_world else ("honest", "true")
    world_seeds = rng.integers(1, 2**63 - 1, size=(n_pos, args.playouts), dtype=np.uint64)
    job_pos, job_cand, job_seed_index = [], [], []
    for row in range(n_pos):
        for c in range(int(n_cand[row])):
            job_pos.extend([row] * args.playouts)
            job_cand.extend([c] * args.playouts)
            job_seed_index.extend(range(args.playouts))
    job_pos = np.asarray(job_pos, dtype=np.int64)
    job_cand = np.asarray(job_cand, dtype=np.int64)
    job_seed_index = np.asarray(job_seed_index, dtype=np.int64)
    per_mode = int(job_pos.size)
    for mode_index, mode in enumerate(modes):
        runner.add_jobs(
            position_ids[job_pos],
            cand[job_pos, job_cand].astype(np.uint32),
            world_seeds[job_pos, job_seed_index],
            (np.arange(per_mode, dtype=np.uint64) + mode_index * per_mode).astype(np.uint32),
            determinize=(mode == "honest"),
        )
    total_jobs = per_mode * len(modes)
    print(f"[{args.tag}] {n_pos} positions, {total_jobs} playouts", flush=True)

    outcomes = np.full((len(modes), n_pos, slots_per_pos, args.playouts), np.nan, dtype=np.float32)
    margins = np.full_like(outcomes, np.nan)
    truncated = 0
    done = 0
    playout_started = time.time()
    last_beat = playout_started
    while True:
        obs, masks = runner.collect_leaves()
        if obs.shape[0] == 0:
            break
        batch_logits, _ = masked_logits(model, obs, masks, device)
        runner.provide_policies(batch_logits)
        if time.time() - last_beat >= HEARTBEAT_S:
            last_beat = time.time()
            finished = total_jobs - runner.queued_jobs() - runner.active_rollouts()
            elapsed = last_beat - playout_started
            print(f"[{args.tag}] playouts {finished}/{total_jobs} "
                  f"{runner.decisions_evaluated() / max(elapsed, 1e-9):.0f} evals/s {elapsed:.0f}s", flush=True)
    results = runner.take_results()
    tags = np.asarray(results["tags"], dtype=np.int64)
    mode_of = tags // per_mode
    job = tags % per_mode
    outcomes[mode_of, job_pos[job], job_cand[job], job_seed_index[job]] = results["outcomes"]
    margins[mode_of, job_pos[job], job_cand[job], job_seed_index[job]] = results["margins"]
    truncated = int(np.asarray(results["truncated"]).sum())
    done = int(tags.size)
    if done != total_jobs:
        raise RuntimeError(f"expected {total_jobs} playout results, got {done}")

    # 4. Statistics. Candidate 0 is the search move; the prior move is
    # candidate 1 unless the two coincide (then it is candidate 0).
    is_override = np.asarray([overridden[index] for index in chosen])
    prior_slot = np.where(is_override, 1, 0)
    rows = np.arange(n_pos)
    half = args.playouts // 2
    summary: dict[str, object] = {}
    for mode_index, mode in enumerate(modes):
        out = outcomes[mode_index]
        q = np.nanmean(out, axis=2)
        m = np.nanmean(margins[mode_index], axis=2)
        delta = q[rows, 0] - q[rows, prior_slot]
        summary[f"{mode}_search_minus_prior_overrides"] = mean_se(delta[is_override])
        summary[f"{mode}_search_minus_prior_all"] = mean_se(delta)
        summary[f"{mode}_margin_search_minus_prior_overrides"] = mean_se(
            (m[rows, 0] - m[rows, prior_slot])[is_override] / 100.0)
        if mode == "honest":
            head_prior, head_search = [], []
            for select, score in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
                q_select = np.nanmean(out[:, :, select], axis=2)
                q_score = np.nanmean(out[:, :, score], axis=2)
                best = np.nanargmax(np.where(np.isnan(q_select), -np.inf, q_select), axis=1)
                head_prior.append(q_score[rows, best] - q_score[rows, prior_slot])
                head_search.append(q_score[rows, best] - q_score[rows, 0])
            summary["headroom_over_prior"] = mean_se(np.mean(head_prior, axis=0))
            summary["headroom_over_search"] = mean_se(np.mean(head_search, axis=0))
            target = 2.0 * q[rows, prior_slot] - 1.0
            predicted = values[chosen]
            summary["value_fit"] = {
                "corr": float(np.corrcoef(predicted, target)[0, 1]) if n_pos > 2 else float("nan"),
                "mse": float(np.mean((predicted - target) ** 2)),
                "baseline_mse": float(np.var(target)),
                "mean_predicted": float(predicted.mean()),
                "mean_playout": float(target.mean()),
            }
            for label, keep in (("early", all_phase[chosen] < 1 / 3), ("mid", (all_phase[chosen] >= 1 / 3) & (all_phase[chosen] < 2 / 3)), ("late", all_phase[chosen] >= 2 / 3)):
                summary[f"honest_search_minus_prior_overrides_{label}"] = mean_se(delta[is_override & keep])
    if "true" in modes:
        gain_true = (np.nanmean(outcomes[1], axis=2)[rows, 0] - np.nanmean(outcomes[1], axis=2)[rows, prior_slot])
        gain_honest = (np.nanmean(outcomes[0], axis=2)[rows, 0] - np.nanmean(outcomes[0], axis=2)[rows, prior_slot])
        summary["peeking_true_minus_honest_overrides"] = mean_se((gain_true - gain_honest)[is_override])

    result = {
        "tag": args.tag, "model": args.model, "sims": args.sims, "determinize": search.determinize,
        "games": args.games, "decisions": decisions, "override_rate": float(overridden.mean()),
        "positions": n_pos, "sampled_overrides": int(is_override.sum()), "overrides_only": args.overrides_only,
        "top_k": args.top_k, "playouts": args.playouts, "playout_temperature": args.playout_temperature,
        "truncated_playouts": truncated, "total_playouts": total_jobs, "seed": args.seed, "config": args.config,
        "wall_s": round(time.time() - started), **summary,
        "per_position": [
            {
                "decision": int(index), "phase": float(all_phase[index]), "override": bool(overridden[index]),
                "value": float(values[index]),
                "candidates": [int(a) for a in cand[row, : n_cand[row]]],
                "visits": [float(all_visits[index, a]) for a in cand[row, : n_cand[row]]],
                "logits": [float(logits[index, a]) for a in cand[row, : n_cand[row]]],
                **{f"q_{mode}": [float(x) for x in np.nanmean(outcomes[i][row, : n_cand[row]], axis=1)]
                   for i, mode in enumerate(modes)},
            }
            for row, index in enumerate(chosen)
        ],
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag}.json").write_text(json.dumps(result, indent=1) + "\n")

    print(f"[{args.tag}] DONE {n_pos} positions ({int(is_override.sum())} overrides), "
          f"{total_jobs} playouts ({truncated} truncated), {result['wall_s']}s", flush=True)
    print(f"[{args.tag}] override rate over all decisions: {100.0 * overridden.mean():.1f}%", flush=True)
    print(f"[{args.tag}] search - prior on overrides (honest worlds): {fmt(summary['honest_search_minus_prior_overrides'])}", flush=True)
    for label in ("early", "mid", "late"):
        print(f"[{args.tag}]   {label:5s}: {fmt(summary[f'honest_search_minus_prior_overrides_{label}'])}", flush=True)
    if "true" in modes:
        print(f"[{args.tag}] search - prior on overrides (true world):    {fmt(summary['true_search_minus_prior_overrides'])}", flush=True)
        print(f"[{args.tag}] peeking (true - honest gain):                {fmt(summary['peeking_true_minus_honest_overrides'])}", flush=True)
    print(f"[{args.tag}] headroom over prior move:  {fmt(summary['headroom_over_prior'])}", flush=True)
    print(f"[{args.tag}] headroom over search move: {fmt(summary['headroom_over_search'])}", flush=True)
    fit = summary["value_fit"]
    print(f"[{args.tag}] value head vs playout value: corr={fit['corr']:.3f} mse={fit['mse']:.3f} "
          f"(predict-the-mean mse={fit['baseline_mse']:.3f})", flush=True)


if __name__ == "__main__":
    main()
