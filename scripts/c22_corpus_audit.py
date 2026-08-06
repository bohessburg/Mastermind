#!/usr/bin/env python3
"""Report c22 raw and rating-weighted policy coverage for base-card buys."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.v2.train.card_transformer import A_BUY_BASE
from src.v2.train.human_data import (
    DEFAULT_RATINGS_SIDECAR,
    POLICY_WEIGHT_EPSILON,
    POLICY_WEIGHT_EXPERT,
    POLICY_WEIGHT_RAMP_END,
    POLICY_WEIGHT_RAMP_START,
    RATING_BAND_EXPERT,
    RATING_BAND_LOW,
    RATING_BAND_RAMP,
    RATING_BAND_UNRATED,
    RATING_DEVIATION_THRESHOLD,
    load_human_tuples,
)


DEFAULT_TUPLE_DIRS = (Path("exports/tuples_all/train"), Path("exports/tuples_all/val"))
DEFAULT_OUTPUT = Path("bench/c22_corpus_audit.json")

# These DefIds mirror src/v2/core/defs.h. The 33 card audit scope is the 26
# second-edition Base kingdom cards plus the seven purchasable base piles.
BASE_CARD_DEFS = (
    (0, "Copper"),
    (1, "Silver"),
    (2, "Gold"),
    (5, "Estate"),
    (6, "Duchy"),
    (7, "Province"),
    (9, "Curse"),
    (10, "Cellar"),
    (11, "Chapel"),
    (12, "Village"),
    (13, "Smithy"),
    (14, "Workshop"),
    (15, "Remodel"),
    (16, "Mine"),
    (19, "Merchant"),
    (20, "Militia"),
    (21, "Witch"),
    (22, "Moat"),
    (23, "Bureaucrat"),
    (27, "Market"),
    (28, "Festival"),
    (29, "Laboratory"),
    (30, "Gardens"),
    (31, "Moneylender"),
    (32, "Poacher"),
    (33, "Vassal"),
    (34, "Harbinger"),
    (35, "Throne Room"),
    (36, "Council Room"),
    (37, "Artisan"),
    (38, "Bandit"),
    (39, "Library"),
    (40, "Sentry"),
)
RATING_BANDS = (RATING_BAND_UNRATED, RATING_BAND_LOW, RATING_BAND_RAMP, RATING_BAND_EXPERT)


def audit_corpus(tuple_dirs: list[Path], ratings_sidecar: Path) -> dict[str, Any]:
    cards = {
        name: {"raw_positive_examples": 0, "policy_effective_positive_examples": 0.0}
        for _def_id, name in BASE_CARD_DEFS
    }
    bands = {
        band: {"games": set(), "tuples": 0, "policy_effective_tuples": 0.0}
        for band in RATING_BANDS
    }
    total_tuples = 0
    policy_effective_tuples = 0.0

    for tuple_dir in tuple_dirs:
        dataset = load_human_tuples(
            tuple_dir,
            skill_weighting=True,
            ratings_sidecar=ratings_sidecar,
        )
        total_tuples += len(dataset)
        policy_effective_tuples += float(dataset.policy_weight.sum(dtype=np.float64))
        corpus_name = str(tuple_dir)
        for band in RATING_BANDS:
            selected = dataset.rating_band == band
            bands[band]["tuples"] += int(selected.sum())
            bands[band]["policy_effective_tuples"] += float(
                dataset.policy_weight[selected].sum(dtype=np.float64)
            )
            bands[band]["games"].update(
                (corpus_name, int(game_index)) for game_index in dataset.game_index[selected]
            )
        for def_id, name in BASE_CARD_DEFS:
            selected = dataset.action == A_BUY_BASE + def_id
            cards[name]["raw_positive_examples"] += int(selected.sum())
            cards[name]["policy_effective_positive_examples"] += float(
                dataset.policy_weight[selected].sum(dtype=np.float64)
            )

    total_buys = sum(card["raw_positive_examples"] for card in cards.values())
    total_effective_buys = sum(card["policy_effective_positive_examples"] for card in cards.values())
    return {
        "tuple_dirs": [str(path) for path in tuple_dirs],
        "ratings_sidecar": str(ratings_sidecar),
        "curve": {
            "epsilon": POLICY_WEIGHT_EPSILON,
            "ramp_start_level": POLICY_WEIGHT_RAMP_START,
            "ramp_end_level": POLICY_WEIGHT_RAMP_END,
            "expert_policy_weight": POLICY_WEIGHT_EXPERT,
            "deviation_threshold": RATING_DEVIATION_THRESHOLD,
        },
        "totals": {
            "tuples": total_tuples,
            "policy_effective_tuples": policy_effective_tuples,
            "base_card_buy_positive_examples": total_buys,
            "base_card_buy_policy_effective_examples": total_effective_buys,
        },
        "rating_bands": {
            band: {
                "games": len(values["games"]),
                "tuples": values["tuples"],
                "policy_effective_tuples": values["policy_effective_tuples"],
            }
            for band, values in bands.items()
        },
        "cards": cards,
    }


def print_audit(report: dict[str, Any]) -> None:
    totals = report["totals"]
    print(
        "c22 corpus audit: "
        f"tuples={totals['tuples']:,} policy-effective={totals['policy_effective_tuples']:.2f} "
        f"base-buy-positives={totals['base_card_buy_positive_examples']:,} "
        f"base-buy-effective={totals['base_card_buy_policy_effective_examples']:.2f}"
    )
    print("\nRating band                 games     tuples  policy-effective")
    for band, values in report["rating_bands"].items():
        print(f"{band:<27} {values['games']:>5,} {values['tuples']:>10,} {values['policy_effective_tuples']:>17.2f}")
    print("\nCard                       raw buys  policy-effective buys")
    for name, values in report["cards"].items():
        print(f"{name:<27} {values['raw_positive_examples']:>8,} {values['policy_effective_positive_examples']:>22.2f}")
    print(
        f"{'TOTAL':<27} {totals['base_card_buy_positive_examples']:>8,} "
        f"{totals['base_card_buy_policy_effective_examples']:>22.2f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tuple-dir",
        type=Path,
        action="append",
        help="tuple corpus directory (repeatable; defaults to c22 train and validation)",
    )
    parser.add_argument("--ratings-sidecar", type=Path, default=DEFAULT_RATINGS_SIDECAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tuple_dirs = args.tuple_dir or list(DEFAULT_TUPLE_DIRS)
    report = audit_corpus(tuple_dirs, args.ratings_sidecar)
    print_audit(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nc22 corpus audit JSON: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
