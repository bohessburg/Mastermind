"""c26 loss ablation: which training signal makes the BC net worse?

Every c26 candidate (BC fine-tuned on BC's own self-play) lost its gate to the
BC net (37.6 / 44.9 / 29.6%). Each arm here starts from the BC weights with a
fresh optimizer, trains STEPS steps on c26's saved replay with one subset of
the loss terms, then plays the standard gate (seat-swapped, gate_sims, same
boards for every arm) against the BC net.

  full      self-play policy + value + aux margin + human anchor (= c26 g3 loss)
  sp_policy self-play policy CE only
  sp_value  self-play value MSE only
  human     human anchor policy + value only (fine-tuning drift control)

Usage: python scripts/c26_ablation.py ARM [--steps 240]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.v2.train.gating import run_gate_match  # noqa: E402
from src.v2.train.model import masked_policy_loss  # noqa: E402
from src.v2.train.replay import load_replay_state  # noqa: E402
from src.v2.train.train import (  # noqa: E402
    build_objects,
    checkpoint_payload,
    human_imitation_losses,
    load_full_checkpoint,
    load_human_dataset_for_config,
    train_step,
)
from src.v2.train.config import load_config  # noqa: E402

RUN = Path("checkpoints/campaign26")
BC = Path("checkpoints/c25_bc_long/bc_a.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("full", "sp_policy", "sp_value", "human", "vhead", "value_anchor"))
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--out", default="bench/c26_ablation")
    parser.add_argument("--run", default=str(RUN))
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    device = torch.device("cuda")
    run = Path(args.run)
    config = load_config(run / "config.json") if (run / "config.json").exists() else load_config("configs/run_c26.json")
    torch.manual_seed(20261109)

    model, optimizer, replay = build_objects(config, device)
    best, _, _ = build_objects(config, device)
    bc_state = load_full_checkpoint(BC, device)["model"]
    model.load_state_dict(bc_state)
    best.load_state_dict(bc_state)
    best.eval()
    load_replay_state(replay, run / "replay_state.npz")
    if args.lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    tag = args.tag or args.arm
    if args.arm == "vhead":
        # Policy and trunk frozen: any gate change is purely search-value quality.
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith("value_head."))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=args.lr or config.optim.lr, weight_decay=config.optim.weight_decay)
    print(f"[{args.arm}] replay rows {len(replay)}, steps {args.steps}", flush=True)

    human = load_human_dataset_for_config(config).minibatches(
        config.imitation.anchor_batch_size, int(config.seed) ^ 0x414E_4348_4F52)
    started = time.time()
    for step in range(args.steps):
        if args.arm == "full":
            losses = train_step(
                model, optimizer, replay, config.optim.batch_size, device,
                human_batches=human,
                anchor_weight=config.imitation.anchor_weight,
                anchor_awr_beta=config.imitation.anchor_awr_beta,
                aux_margin_weight=config.aux_margin_weight,
                value_loss_weight=config.optim.value_loss_weight,
                anchor_value_weight=config.imitation.anchor_value_weight,
            )
        else:
            model.train()
            if args.arm == "value_anchor":
                batch = replay.sample(config.optim.batch_size)
                obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
                _, value = model(obs)
                sp_val = F.mse_loss(value, torch.as_tensor(batch.value, dtype=torch.float32, device=device))
                pol, hval = human_imitation_losses(model, next(human), device)
                loss = sp_val + pol + hval
                losses = {"value_loss": float(sp_val), "anchor_policy_loss": float(pol), "anchor_value_loss": float(hval)}
            elif args.arm == "human":
                pol, val = human_imitation_losses(model, next(human), device)
                loss = pol + val
                losses = {"anchor_policy_loss": float(pol), "anchor_value_loss": float(val)}
            else:
                batch = replay.sample(config.optim.batch_size)
                obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
                logits, value = model(obs)
                if args.arm == "vhead":
                    target = torch.as_tensor(batch.value, dtype=torch.float32, device=device)
                    loss = F.mse_loss(value, target)
                    losses = {"value_loss": float(loss)}
                elif args.arm == "sp_policy":
                    mask = torch.as_tensor(batch.legal_mask, dtype=torch.bool, device=device)
                    target = torch.as_tensor(batch.policy, dtype=torch.float32, device=device)
                    loss, _ = masked_policy_loss(logits, mask, target)
                    losses = {"policy_loss": float(loss)}
                else:
                    target = torch.as_tensor(batch.value, dtype=torch.float32, device=device)
                    loss = F.mse_loss(value, target)
                    losses = {"value_loss": float(loss)}
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if step % 40 == 0 or step == args.steps - 1:
            shown = {k: round(float(v), 4) for k, v in losses.items() if k != "loss"}
            print(f"[{args.arm}] step {step} {shown} {time.time() - started:.0f}s", flush=True)

    model.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(config, 0, model, optimizer), out / f"{tag}.pt")
    gate_started = time.time()
    stats = run_gate_match(model, best, config, 3, device)
    result = {"arm": args.arm, "tag": tag, "run": str(run), "lr": optimizer.param_groups[0]["lr"], "steps": args.steps, "wins": stats.wins, "losses": stats.losses,
              "ties": stats.ties, "win_pct": stats.win_pct, "gate_s": round(time.time() - gate_started),
              "gate_games": config.gate_games, "gate_sims": config.gate_sims}
    (out / f"{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[{args.arm}] GATE {stats.wins}W-{stats.losses}L-{stats.ties}T win_pct={stats.win_pct:.3f}", flush=True)


if __name__ == "__main__":
    main()
