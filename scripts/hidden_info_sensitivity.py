#!/usr/bin/env python3
"""Measure how much the converter's sampled hidden-zone values matter to a BC policy.

Two conversions of the same raw games (different DGAMES_SAMPLE_SALT) have
identical actions, legal masks and public features; they differ only where the
converter sampled unobserved zones.  This script aligns them by
(game id, seat, ply) and reports:

1. which observation blocks differ between the two samples, and how much;
2. how a trained BC checkpoint's predictions change when it sees sample B
   instead of sample A on the same validation rows;
3. a control that zeroes the acting player's own deck/discard (and hand) blocks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.v2.train.evaluate import load_model  # noqa: E402

OPP = 324
BLOCKS = {
    "own_hand": (4, 68),
    "own_deck": (68, 132),
    "own_discard": (132, 196),
    "own_inplay": (196, 260),
    "own_setaside": (260, 324),
    "opp_prefix": (OPP, OPP + 75),
    "opp_collection": (OPP + 75, OPP + 139),
    "opp_discard": (OPP + 139, OPP + 203),
    "opp_setaside": (OPP + 203, OPP + 267),
}
QUALITY_NAMES = {0: "partially_inferred", 1: "fully_observed"}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def make_key(game_id: np.ndarray, seat: np.ndarray, ply: np.ndarray) -> np.ndarray:
    if int(ply.max()) >= (1 << 20):
        raise ValueError("ply_index exceeds key width")
    return (game_id.astype(np.int64) << 24) | (seat.astype(np.int64) << 20) | ply.astype(np.int64)


def load_keys(root: Path) -> dict:
    manifest = json.loads((root / "tuple_manifest.json").read_text())
    game_ids = {int(g["index"]): int(g["id"]) for g in manifest["games"]}
    out = {k: [] for k in ("key", "dtype", "quality", "shard", "row")}
    for shard_index, entry in enumerate(manifest["shards"]):
        z = np.load(root / entry["path"])
        gid = np.vectorize(game_ids.__getitem__)(z["game_index"])
        out["key"].append(make_key(gid, z["seat_index"], z["ply_index"]))
        out["dtype"].append(z["decision_type"])
        out["quality"].append(z["observation_quality"])
        out["shard"].append(np.full(len(gid), shard_index, dtype=np.int32))
        out["row"].append(np.arange(len(gid), dtype=np.int64))
    result = {k: np.concatenate(v) for k, v in out.items()}
    result["manifest"] = manifest
    if len(np.unique(result["key"])) != len(result["key"]):
        raise ValueError(f"duplicate (game, seat, ply) keys in {root}")
    return result


def gather(root: Path, info: dict, positions: np.ndarray, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read selected global row positions (sorted) from shards, in order."""
    shards = info["shard"][positions]
    rows = info["row"][positions]
    out: dict[str, list[np.ndarray]] = {f: [] for f in fields}
    order = np.argsort(positions, kind="stable")
    result: dict[str, np.ndarray | None] = {f: None for f in fields}
    for shard_index in np.unique(shards):
        sel = np.nonzero(shards == shard_index)[0]
        z = np.load(root / info["manifest"]["shards"][int(shard_index)]["path"])
        for f in fields:
            chunk = z[f][rows[sel]]
            if result[f] is None:
                result[f] = np.empty((len(positions),) + chunk.shape[1:], dtype=chunk.dtype)
            result[f][sel] = chunk
    del order, out
    return {f: result[f] for f in fields}


def val_keys(val_root: Path) -> np.ndarray:
    manifest = json.loads((val_root / "tuple_manifest.json").read_text())
    game_ids = {int(g["index"]): int(g["id"]) for g in manifest["games"] if g.get("source") == "dgames"}
    keys = []
    for entry in manifest["shards"]:
        if entry.get("source") != "dgames":
            continue
        z = np.load(val_root / entry["path"])
        gid = np.vectorize(game_ids.__getitem__)(z["game_index"])
        keys.append(make_key(gid, z["seat_index"], z["ply_index"]))
    return np.concatenate(keys)


def block_diff_report(obs_a: np.ndarray, obs_b: np.ndarray, quality: np.ndarray, dtype: np.ndarray, names: dict[int, str]) -> dict:
    diff = np.abs(obs_a - obs_b)
    covered = np.zeros(obs_a.shape[1], dtype=bool)
    report: dict[str, dict] = {}

    def summarize(mask: np.ndarray) -> dict:
        res = {}
        for name, (lo, hi) in BLOCKS.items():
            d = diff[mask, lo:hi]
            res[name] = {
                "rows_differing_pct": round(100.0 * float((d.sum(axis=1) > 0).mean()), 2),
                "mean_cards_moved": round(float(d.sum(axis=1).mean() / 2.0), 3),
            }
        return res

    for name, (lo, hi) in BLOCKS.items():
        covered[lo:hi] = True
    other = diff[:, ~covered]
    report["all"] = summarize(np.ones(len(diff), dtype=bool))
    report["all"]["everything_else"] = {"rows_differing_pct": round(100.0 * float((other.sum(axis=1) > 0).mean()), 4)}
    for q, qn in QUALITY_NAMES.items():
        mask = quality == q
        if mask.any():
            report[qn] = summarize(mask)
    for t in np.unique(dtype):
        mask = dtype == t
        report[f"type:{names.get(int(t), t)}"] = summarize(mask)
    return report


@torch.no_grad()
def run_variants(model, variants: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], batch: int, device) -> dict:
    """variants maps name -> (obs, legal, action); the first entry is the reference."""
    n = len(next(iter(variants.values()))[2])
    out = {name: {"nll": np.zeros(n), "correct": np.zeros(n, dtype=bool), "pred": np.zeros(n, dtype=np.int64)} for name in variants}
    probs_ref = {}
    tv = {name: np.zeros(n) for name in variants}
    value = {name: np.zeros(n) for name in variants}
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        ref = None
        for name, (obs, legal, action) in variants.items():
            leg = torch.as_tensor(legal[start:stop], dtype=torch.bool, device=device)
            act = torch.as_tensor(action[start:stop], dtype=torch.long, device=device)
            x = torch.as_tensor(obs[start:stop], dtype=torch.float32, device=device)
            logits, val = model(x)
            logits = logits.masked_fill(~leg, -1.0e9)
            logp = torch.log_softmax(logits, dim=-1)
            p = logp.exp()
            out[name]["nll"][start:stop] = (-logp.gather(1, act[:, None]).squeeze(1)).cpu().numpy()
            out[name]["pred"][start:stop] = logits.argmax(dim=-1).cpu().numpy()
            out[name]["correct"][start:stop] = (logits.argmax(dim=-1) == act).cpu().numpy()
            value[name][start:stop] = val.reshape(len(p), -1)[:, 0].cpu().numpy()
            if ref is None:
                ref = p
            tv[name][start:stop] = (0.5 * (p - ref).abs().sum(dim=-1)).cpu().numpy()
        if (start // batch) % 20 == 0:
            log(f"  eval {stop}/{n}")
    for name in variants:
        out[name]["tv_vs_first"] = tv[name]
        out[name]["value"] = value[name]
    return out


def summarize_eval(results: dict, mask: np.ndarray, first: str) -> dict:
    row = {}
    for name, r in results.items():
        row[name] = {
            "acc": round(100.0 * float(r["correct"][mask].mean()), 2),
            "nll": round(float(r["nll"][mask].mean()), 4),
            "argmax_agree_with_first_pct": round(100.0 * float((r["pred"][mask] == results[first]["pred"][mask]).mean()), 2),
            "mean_tv_vs_first": round(float(r["tv_vs_first"][mask].mean()), 4),
            "mean_abs_value_diff": round(float(np.abs(r["value"][mask] - results[first]["value"][mask]).mean()), 4),
        }
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-a", type=Path, default=Path("exports/tuples_dgames"))
    parser.add_argument("--export-b", type=Path, default=Path("exports/tuples_dgames_saltB"))
    parser.add_argument("--val", type=Path, default=Path("exports/tuples_all_40plus/val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/c21_ablation/bc_a_973k.pt"))
    parser.add_argument("--diff-sample", type=int, default=150_000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--report", type=Path, default=Path("bench/hidden_info_sensitivity.json"))
    args = parser.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cpu")

    log("loading keys (A)")
    a = load_keys(args.export_a)
    log(f"A rows={len(a['key'])}")
    log("loading keys (B)")
    b = load_keys(args.export_b)
    log(f"B rows={len(b['key'])}")
    names = {int(v): k for k, v in a["manifest"]["row_provenance"]["decision_type_ids"].items()}

    common, ia, ib = np.intersect1d(a["key"], b["key"], return_indices=True)
    log(f"rows common to A and B: {len(common)}")
    sort_a = np.argsort(a["key"])
    sort_b = np.argsort(b["key"])
    del sort_a, sort_b

    vk = val_keys(args.val)
    if args.max_val_rows:
        vk = vk[: args.max_val_rows]
    pos_in_common = np.searchsorted(common, vk)
    pos_in_common = np.clip(pos_in_common, 0, len(common) - 1)
    found = common[pos_in_common] == vk
    log(f"val dgames rows={len(vk)} found in both exports={int(found.sum())}")
    vpos = pos_in_common[found]
    va, vb = ia[vpos], ib[vpos]

    fields = ("obs", "action", "legal")
    log("gathering val rows from A")
    ga = gather(args.export_a, a, va, fields)
    log("gathering val rows from B")
    gb = gather(args.export_b, b, vb, fields)
    dtype = a["dtype"][va]
    quality = a["quality"][va]
    same_label = (ga["action"] == gb["action"]) & (ga["legal"] == gb["legal"]).all(axis=1)
    log(f"rows whose chosen action AND legal mask are identical in A and B: {int(same_label.sum())}/{len(same_label)}")
    mismatch_by_type = {}
    for t in np.unique(dtype):
        m = dtype == t
        mismatch_by_type[names.get(int(t), str(t))] = {
            "rows": int(m.sum()),
            "label_or_mask_differs_pct": round(100.0 * float((~same_label[m]).mean()), 2),
        }
    log("label/mask mismatch by decision type: " + json.dumps(mismatch_by_type))

    # Integrity: A's rows must equal the rows the 40+ val corpus actually contains.
    val_manifest = json.loads((args.val / "tuple_manifest.json").read_text())
    probe_entry = next(e for e in val_manifest["shards"] if e.get("source") == "dgames")
    probe = np.load(args.val / probe_entry["path"])
    probe_keys = make_key(
        np.vectorize({int(g["index"]): int(g["id"]) for g in val_manifest["games"] if g.get("source") == "dgames"}.__getitem__)(probe["game_index"]),
        probe["seat_index"],
        probe["ply_index"],
    )
    probe_pos = np.searchsorted(vk, probe_keys[:2000])
    # vk is not sorted, so locate by direct match instead.
    lookup = {int(k): i for i, k in enumerate(vk[: len(vk)])}
    check_rows = [i for i, k in enumerate(probe_keys[:2000]) if int(k) in lookup and found[lookup[int(k)]]]
    del probe_pos
    mismatch = 0
    inv = np.full(len(vk), -1, dtype=np.int64)
    inv[np.nonzero(found)[0]] = np.arange(int(found.sum()))
    for i in check_rows[:500]:
        j = inv[lookup[int(probe_keys[i])]]
        if j >= 0 and not np.array_equal(probe["obs"][i], ga["obs"][j]):
            mismatch += 1
    log(f"integrity: checked {min(len(check_rows), 500)} val rows against the 40+ val corpus, obs mismatches={mismatch}")

    log("block-diff analysis on a random sample of common rows")
    rng = np.random.default_rng(0)
    sample_c = np.sort(rng.choice(len(common), size=min(args.diff_sample, len(common)), replace=False))
    sa = gather(args.export_a, a, ia[sample_c], ("obs",))["obs"]
    sb = gather(args.export_b, b, ib[sample_c], ("obs",))["obs"]
    diff_report = block_diff_report(sa, sb, a["quality"][ia[sample_c]], a["dtype"][ia[sample_c]], names)
    del sa, sb

    log(f"loading model {args.checkpoint}")
    model, _cfg = load_model(args.checkpoint, device)
    model.eval()
    obs_a = ga["obs"]
    obs_b = gb["obs"]
    obs_m1 = obs_a.copy()
    obs_m1[:, BLOCKS["own_deck"][0] : BLOCKS["own_discard"][1]] = 0.0
    obs_m2 = obs_a.copy()
    obs_m2[:, BLOCKS["own_hand"][0] : BLOCKS["own_discard"][1]] = 0.0
    variants = {
        "A_as_trained": (obs_a, ga["legal"], ga["action"]),
        "B_resampled": (obs_b, gb["legal"], gb["action"]),
        "A_zero_deck_discard": (obs_m1, ga["legal"], ga["action"]),
        "A_zero_hand_deck_discard": (obs_m2, ga["legal"], ga["action"]),
    }
    log("evaluating variants (each sample scored against its own labels and legal masks)")
    results = run_variants(model, variants, args.batch, device)

    everything = np.ones(len(dtype), dtype=bool)
    summary = {
        "overall": summarize_eval(results, everything, "A_as_trained"),
        "paired_rows_only_same_label_and_mask": {"rows": int(same_label.sum()), **summarize_eval(results, same_label, "A_as_trained")},
    }
    for q, qn in QUALITY_NAMES.items():
        mask = quality == q
        if mask.any():
            summary[qn] = {"rows": int(mask.sum()), **summarize_eval(results, mask, "A_as_trained")}
    for t in np.unique(dtype):
        mask = dtype == t
        if mask.sum() >= 200:
            summary[f"type:{names.get(int(t), t)}"] = {"rows": int(mask.sum()), **summarize_eval(results, mask, "A_as_trained")}

    report = {
        "checkpoint": str(args.checkpoint),
        "export_a": str(args.export_a),
        "export_b": str(args.export_b),
        "val_rows_evaluated": int(len(dtype)),
        "block_diff_sample_rows": int(len(sample_c)),
        "label_or_mask_mismatch_by_type": mismatch_by_type,
        "block_diff": diff_report,
        "eval": summary,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1))
    log(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
