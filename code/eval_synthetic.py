"""
evaluate a linkage strategy on the synthetic reference with known pairs.

Loads a synthetic SIM file + its truth table, runs the strategy against the
real current-year SINASC, and reports exact precision/recall/F1 plus the
per-pass profile (compared with the real-data profile from run_real_check,
which is the calibration target for the error model).

Recall is reported two ways:
  recall_overall     - denominator = all synthetic deaths (includes the
                       December/previous-year births, which v3.2 cannot match
                       by construction; the honest real-world figure)
  recall_within_year - denominator = deaths whose true birth is in the
                       current-year SINASC (isolates key quality from the
                       cross-year asymmetry)

Usage: python3 eval_synthetic.py <year> [tier]
"""

from __future__ import annotations

import json
import sys

import polars as pl

from common import OUTPUTS, load_sinasc
from strategies import v32_core


def evaluate(year: int, tier: str = "calibrated", strategy=v32_core, name: str = "v3.2"):
    synth = pl.read_csv(
        OUTPUTS / "synthetic" / f"synthetic_SIM_{year}_{tier}.csv",
        infer_schema_length=0,
        null_values=["", "NA", "NULL", "None", "null"],
    )
    truth = pl.read_csv(OUTPUTS / "synthetic" / f"truth_{year}_{tier}.csv")
    true_row = dict(zip(truth["SYNTH_ID"].to_list(), truth["TRUE_ROW"].to_list()))

    sinasc = load_sinasc(year)
    sim_rows = synth.to_dicts()
    matches, pass_counts = strategy(sinasc, sim_rows)

    n = len(sim_rows)
    n_within = sum(1 for v in true_row.values() if v >= 0)
    n_matched = len(matches)
    correct = sum(1 for i, (j, _) in matches.items() if true_row[int(sim_rows[i]["SYNTH_ID"])] == j)
    wrong = n_matched - correct

    precision = correct / n_matched if n_matched else 0.0
    recall_overall = correct / n
    recall_within = correct / n_within if n_within else 0.0
    f1 = (2 * precision * recall_within / (precision + recall_within)
          if precision + recall_within else 0.0)

    correct_by_pass = {p: 0 for p in (1, 2, 3, 4)}
    for i, (j, p) in matches.items():
        if true_row[int(sim_rows[i]["SYNTH_ID"])] == j:
            correct_by_pass[p] += 1

    print(f"\n=== {name} on synthetic {year} [{tier}] ===")
    print(f"deaths {n:,} (within-year {n_within:,}) | matched {n_matched:,} "
          f"| correct {correct:,} | wrong {wrong:,}")
    for p in (1, 2, 3, 4):
        c = pass_counts[p]
        print(f"  Pass {p}: {c:,} ({c/n*100:.1f}%) - precision {correct_by_pass[p]/c*100 if c else 0:.1f}%")
    print(f"  Unmatched: {n - n_matched:,} ({(n - n_matched)/n*100:.1f}%)")
    print(f"  PRECISION          {precision*100:.2f}%")
    print(f"  RECALL overall     {recall_overall*100:.2f}%")
    print(f"  RECALL within-year {recall_within*100:.2f}%")
    print(f"  F1 (within-year)   {f1*100:.2f}%")

    return {
        "strategy": name, "year": year, "tier": tier,
        "n": n, "n_within": n_within, "matched": n_matched,
        "correct": correct, "wrong": wrong,
        "precision": precision, "recall_overall": recall_overall,
        "recall_within_year": recall_within, "f1_within_year": f1,
        "pass_counts": pass_counts,
        "pass_precision": {p: (correct_by_pass[p] / pass_counts[p] if pass_counts[p] else None)
                           for p in (1, 2, 3, 4)},
    }


if __name__ == "__main__":
    year = int(sys.argv[1])
    tier = sys.argv[2] if len(sys.argv) > 2 else "calibrated"
    res = evaluate(year, tier)
    out = OUTPUTS / "synthetic" / f"eval_v32_{year}_{tier}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved {out}")
