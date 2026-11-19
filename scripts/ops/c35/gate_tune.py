"""Time the paired gate on this Mac under several setups (c35 control gen 6 vs gen 5, 400 games)."""
import copy, json, sys, time
from pathlib import Path
sys.path.insert(0, "/Users/paisho/Projects/Mastermind")
import torch
from src.v2.train.config import load_config
from src.v2.train.evaluate import load_model
from src.v2.train.gating import run_paired_gate_match

base = load_config("configs/run_c35_control_mac.json")
device = torch.device("mps")
cand, _ = load_model(Path("checkpoints/remote/campaign35_control/gen_0006.pt"), device)
best, _ = load_model(Path("checkpoints/remote/campaign35_control/gen_0005.pt"), device)
variants = [
    ("local_w1", dict(gate_server=False, gate_workers=1)),
    ("server_w4_s1", dict(gate_server=True, gate_workers=4, server_shards=1)),
    ("server_w8_s1", dict(gate_server=True, gate_workers=8, server_shards=1)),
    ("server_w8_s2", dict(gate_server=True, gate_workers=8, server_shards=2)),
]
only = sys.argv[1:] or [v[0] for v in variants]
for name, over in variants:
    if name not in only:
        continue
    cfg = copy.deepcopy(base)
    cfg.gate_games = 400
    for k, v in over.items():
        setattr(cfg, k, v)
    t = time.perf_counter()
    s = run_paired_gate_match(cand, best, cfg, 6, device, log=lambda m: print(f"  [{name}] {m}", flush=True))
    dt = time.perf_counter() - t
    print(json.dumps({"variant": name, "seconds": round(dt, 1), "score": round(float(s["a_score"]), 4),
                      "wins": s["wins"], "losses": s["losses"], "ties": s["ties"],
                      "evals": s.get("nn_evals"), "evals_per_s": round(float(s.get("nn_evals", 0)) / dt)}), flush=True)
