"""Value-target geometry sweep (task #15).

Re-targets the banked c19 replay slice under margin_blend alphas
{0.6 (control), 0.4, 0.2, 0.0} via exact inversion of the alpha=0.6
labels (|v| = a + (1-a)*(0.5 + 0.5*min(|m|,20)/20) is bijective in the
recorded range), fits an identical CardTokenNet per arm from one init
seed, then runs the value/duchy probes on each fitted checkpoint.

Comparative read only: same data, same init, same steps — differences
are attributable to target geometry alone.
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

REPLAY = REPO / "checkpoints/remote/campaign19/replay_state.npz"
OUT = REPO / "bench/value_target_sweep"
ALPHAS = [0.6, 0.4, 0.2, 0.0]
SCALE = 20.0
SLICE = 250_000
VAL = 10_000
STEPS = 1500
BATCH = 1024
LR = 1e-3
SEED = 20260729
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


def fit_arm(alpha, obs, policy, legal, margin, device):
    torch.manual_seed(SEED)
    model = build_model(MODEL_CFG, obs.shape[1], policy.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    value = blend(margin, alpha)
    n = obs.shape[0] - VAL
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    model.train()
    for step in range(1, STEPS + 1):
        idx = rng.integers(0, n, size=BATCH)
        o = torch.as_tensor(obs[idx], dtype=torch.float32, device=device)
        p = torch.as_tensor(policy[idx], dtype=torch.float32, device=device)
        lg = torch.as_tensor(legal[idx], device=device)
        v = torch.as_tensor(value[idx], dtype=torch.float32, device=device)
        logits, pred = model(o)
        loss = masked_ce(logits, lg, p) + torch.nn.functional.mse_loss(pred, v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            rate = step / (time.time() - t0)
            print(f"alpha={alpha} step {step}/{STEPS} loss={loss.item():.4f} ({rate:.1f} st/s)", flush=True)
    model.eval()
    with torch.no_grad():
        vs = slice(n, obs.shape[0])
        losses = []
        for lo in range(n, obs.shape[0], 2048):
            hi = min(lo + 2048, obs.shape[0])
            o = torch.as_tensor(obs[lo:hi], dtype=torch.float32, device=device)
            p = torch.as_tensor(policy[lo:hi], dtype=torch.float32, device=device)
            lg = torch.as_tensor(legal[lo:hi], device=device)
            v = torch.as_tensor(value[lo:hi], dtype=torch.float32, device=device)
            logits, pred = model(o)
            losses.append((masked_ce(logits, lg, p).item(), torch.nn.functional.mse_loss(pred, v).item(), hi - lo))
        wce = sum(l[0] * l[2] for l in losses) / sum(l[2] for l in losses)
        wmse = sum(l[1] * l[2] for l in losses) / sum(l[2] for l in losses)
    ckpt = OUT / f"fit_alpha{alpha:.1f}.pt"
    torch.save({"config": {"model": MODEL_CFG, "selfplay": {"obs_version": 3}}, "model": model.state_dict()}, ckpt)
    return {"alpha": alpha, "val_ce": wce, "val_mse": wmse, "checkpoint": str(ckpt)}


def run_probe(script, ckpt, out_json):
    cmd = [str(REPO / ".venv/bin/python"), str(REPO / "scripts/probes" / script), "--checkpoint", str(ckpt), "--out", str(out_json)]
    env = {"PYTHONPATH": str(REPO / "build"), "PATH": "/usr/bin:/bin", "PYTHONUNBUFFERED": "1", "HOME": str(Path.home())}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO), timeout=1800)
    if r.returncode != 0:
        print(f"PROBE FAIL {script} alpha ckpt={ckpt}: {r.stderr[-500:]}", flush=True)
        return None
    return json.loads(Path(out_json).read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device={device}", flush=True)
    d = np.load(REPLAY)
    obs = d["obs"][:SLICE]
    policy = d["policy"][:SLICE]
    legal = d["legal_mask"][:SLICE]
    margin = invert_margin(d["value"][:SLICE])
    print(f"slice={SLICE} ties={(margin == 0).sum()} mean|m|={np.abs(margin).mean():.2f}", flush=True)
    results = []
    for alpha in ALPHAS:
        res = fit_arm(alpha, obs, policy, legal, margin, device)
        vp = run_probe("value_probe.py", res["checkpoint"], OUT / f"value_probe_a{alpha:.1f}.json")
        dp = run_probe("duchy_probe.py", res["checkpoint"], OUT / f"duchy_probe_a{alpha:.1f}.json")
        res["value_probe"] = vp
        res["duchy_probe"] = dp
        results.append(res)
        print(f"ARM DONE alpha={alpha}: val_ce={res['val_ce']:.4f} val_mse={res['val_mse']:.4f}", flush=True)
    (OUT / "sweep_results.json").write_text(json.dumps(results, indent=2))
    print("SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
