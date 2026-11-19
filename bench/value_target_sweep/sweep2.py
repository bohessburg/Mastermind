"""Value-target geometry sweep v2 (task #15, replication-grade).

v1 pilot showed single-seed 1500-step fits are too noisy to rank alphas.
v2: 3 seeds per alpha, 3000 steps, and 13% human-tuple mixing per batch
(the only in-distribution engine states), human targets recomputed at
the arm's alpha. Probes run per fit on the (now-consistent) encoder;
report mean +/- sd of the money-engine gap, junk sanity, duchy deltas.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/Users/paisho/Projects/Mastermind")
sys.path.insert(0, str(REPO))

from src.v2.train.model import build_model  # noqa: E402
from src.v2.train.human_data import HumanBatchIterator, load_human_tuples  # noqa: E402

REPLAY = REPO / "checkpoints/remote/campaign19/replay_state.npz"
TUPLES = REPO / "exports/tuples"
OUT = REPO / "bench/value_target_sweep"
ALPHAS = [0.6, 0.4, 0.2, 0.0]
SEEDS = [1, 2, 3]
SCALE = 20.0
SLICE = 250_000
VAL = 10_000
STEPS = 3000
BATCH = 1024
HUMAN_ROWS = 128  # ~13% of each batch
LR = 1e-3
MODEL_CFG = {
    "arch": "card_transformer",
    "obs_version": 3,
    "d_model": 192,
    "n_layers": 3,
    "n_heads": 4,
    "ffn_multiplier": 4,
    "dropout": 0.0,
}


def invert_margin(value: np.ndarray, alpha0: float = 0.6) -> np.ndarray:
    sign = np.sign(value)
    g = (np.abs(value) - alpha0) / (1.0 - alpha0)
    margin = np.round((g - 0.5) * 2.0 * SCALE)
    margin = np.clip(margin, 0.0, SCALE) * sign
    margin[value == 0.0] = 0.0
    return margin


def blend(margin: np.ndarray, alpha: float) -> np.ndarray:
    sign = np.sign(margin)
    g = 0.5 + 0.5 * np.minimum(np.abs(margin), SCALE) / SCALE
    out = sign * (alpha + (1.0 - alpha) * g)
    out[margin == 0.0] = 0.0
    return out.astype(np.float32)


def masked_ce(logits, legal, target):
    masked = logits.masked_fill(~legal, -1e9)
    logp = torch.log_softmax(masked, dim=-1)
    return -(target * logp).sum(-1).mean()


def one_hot(actions: torch.Tensor, width: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(actions.long(), num_classes=width).float()


def fit_arm(alpha, seed, obs, policy, legal, margin, device):
    torch.manual_seed(seed * 7919 + 17)
    model = build_model(MODEL_CFG, obs.shape[1], policy.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    value = blend(margin, alpha)
    human = load_human_tuples(TUPLES, margin_blend_alpha=alpha)
    hiter = HumanBatchIterator(human, batch_size=HUMAN_ROWS, seed=seed * 104729 + 3)
    n = obs.shape[0] - VAL
    rng = np.random.default_rng(seed * 65537 + 9)
    t0 = time.time()
    model.train()
    for step in range(1, STEPS + 1):
        idx = rng.integers(0, n, size=BATCH - HUMAN_ROWS)
        hb = next(hiter)
        o = torch.cat([
            torch.as_tensor(obs[idx], dtype=torch.float32),
            torch.as_tensor(hb.obs, dtype=torch.float32),
        ]).to(device)
        p = torch.cat([
            torch.as_tensor(policy[idx], dtype=torch.float32),
            one_hot(torch.as_tensor(np.asarray(hb.action)), policy.shape[1]),
        ]).to(device)
        lg = torch.cat([
            torch.as_tensor(legal[idx]),
            torch.as_tensor(np.asarray(hb.legal)),
        ]).to(device)
        v = torch.cat([
            torch.as_tensor(value[idx], dtype=torch.float32),
            torch.as_tensor(np.asarray(hb.value), dtype=torch.float32),
        ]).to(device)
        logits, pred = model(o)
        loss = masked_ce(logits, lg, p) + torch.nn.functional.mse_loss(pred, v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 250 == 0:
            print(f"a={alpha} s={seed} step {step}/{STEPS} loss={loss.item():.4f} ({step/(time.time()-t0):.1f} st/s)", flush=True)
    ckpt = OUT / f"fit2_a{alpha:.1f}_s{seed}.pt"
    torch.save({"config": {"model": MODEL_CFG, "selfplay": {"obs_version": 3}}, "model": model.state_dict()}, ckpt)
    return ckpt


def run_probe(script, ckpt, out_json):
    cmd = [str(REPO / ".venv/bin/python"), str(REPO / "scripts/probes" / script), "--checkpoint", str(ckpt), "--out", str(out_json)]
    env = {"PYTHONPATH": str(REPO / "build"), "PATH": "/usr/bin:/bin", "PYTHONUNBUFFERED": "1", "HOME": str(Path.home())}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO), timeout=1800)
    if r.returncode != 0:
        print(f"PROBE FAIL {script} {ckpt}: {r.stderr[-300:]}", flush=True)
        return None
    return json.loads(Path(out_json).read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}", flush=True)
    d = np.load(REPLAY)
    obs = d["obs"][:SLICE]
    policy = d["policy"][:SLICE]
    legal = d["legal_mask"][:SLICE]
    margin = invert_margin(d["value"][:SLICE])
    results = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            ckpt = fit_arm(alpha, seed, obs, policy, legal, margin, device)
            vp = run_probe("value_probe.py", ckpt, OUT / f"vp2_a{alpha:.1f}_s{seed}.json")
            dp = run_probe("duchy_probe.py", ckpt, OUT / f"dp2_a{alpha:.1f}_s{seed}.json")
            results.append({"alpha": alpha, "seed": seed, "value_probe": vp, "duchy_probe": dp, "checkpoint": str(ckpt)})
            print(f"FIT DONE a={alpha} s={seed}", flush=True)
    (OUT / "sweep2_results.json").write_text(json.dumps(results, indent=2))
    print("SWEEP2 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
