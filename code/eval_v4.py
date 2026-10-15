"""
evaluate v4 stages on the synthetic reference with known pairs.

Same protocol as eval_synthetic.py (v3.2), with the v4 additions: the index
includes December of the previous year (so prev-December deaths are scorable,
truth encoded as 100_000_000 + row) and pass 5 is the tolerant unique-candidate
recovery. Reports precision/recall/F1, per-pass precision, and the ambiguity
rate left for the classifier/assignment stages.

Usage: python3 eval_v4.py <year> [tier] [stage]
"""

from __future__ import annotations

import json
import sys

import polars as pl

from common import OUTPUTS, load_sinasc
from strategies_v4 import v4_core


def evaluate_v4(year: int, tier: str = "calibrated", stage: str = "v4.0"):
    synth = pl.read_csv(
        OUTPUTS / "synthetic" / f"synthetic_SIM_{year}_{tier}.csv",
        infer_schema_length=0,
        null_values=["", "NA", "NULL", "None", "null"],
    )
    truth = pl.read_csv(OUTPUTS / "synthetic" / f"truth_{year}_{tier}.csv")
    true_row = dict(zip(truth["SYNTH_ID"].to_list(), truth["TRUE_ROW"].to_list()))
    no_pair = dict(zip(truth["SYNTH_ID"].to_list(), truth["NO_TRUE_PAIR"].to_list()))

    sinasc_cur = load_sinasc(year)
    sinasc_prev = load_sinasc(year - 1)
    prev_dec = (sinasc_prev.filter(pl.col("DTNASC").str.slice(2, 2) == "12")
                if sinasc_prev is not None else None)

    sim_rows = synth.to_dicts()
    matches, pass_counts, ambiguous = v4_core(sinasc_cur, prev_dec, sim_rows, stage=stage)

    n = len(sim_rows)
    n_pairable = sum(1 for sid, v in no_pair.items() if v == 0)
    n_matched = len(matches)

    def truth_of(i):
        return true_row[int(sim_rows[i]["SYNTH_ID"])]

    correct = sum(1 for i, (j, _) in matches.items() if truth_of(i) == j)
    precision = correct / n_matched if n_matched else 0.0
    recall_pairable = correct / n_pairable
    f1 = (2 * precision * recall_pairable / (precision + recall_pairable)
          if precision + recall_pairable else 0.0)

    correct_by_pass = {p: 0 for p in pass_counts}
    for i, (j, p) in matches.items():
        if truth_of(i) == j:
            correct_by_pass[p] += 1

    print(f"\n=== {stage} on synthetic {year} [{tier}] ===")
    print(f"deaths {n:,} (pairable {n_pairable:,}) | matched {n_matched:,} "
          f"| correct {correct:,} | wrong {n_matched - correct:,}")
    for p in sorted(pass_counts):
        c = pass_counts[p]
        print(f"  Pass {p}: {c:,} ({c/n*100:.1f}%) - precision {correct_by_pass[p]/c*100 if c else 0:.1f}%")
    print(f"  Ambiguous (unresolved): {len(ambiguous):,} ({len(ambiguous)/n*100:.1f}%)")
    print(f"  Unmatched: {n - n_matched:,} ({(n - n_matched)/n*100:.1f}%)")
    print(f"  PRECISION        {precision*100:.2f}%")
    print(f"  RECALL pairable  {recall_pairable*100:.2f}%")
    print(f"  F1               {f1*100:.2f}%")

    res = {
        "strategy": stage, "year": year, "tier": tier,
        "n": n, "n_pairable": n_pairable, "matched": n_matched,
        "correct": correct, "wrong": n_matched - correct,
        "precision": precision, "recall_pairable": recall_pairable, "f1": f1,
        "ambiguous": len(ambiguous),
        "pass_counts": pass_counts,
        "pass_precision": {p: (correct_by_pass[p] / pass_counts[p] if pass_counts[p] else None)
                           for p in pass_counts},
    }
    out = OUTPUTS / "synthetic" / f"eval_{stage.replace('.', '')}_{year}_{tier}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved {out}")
    return res


if __name__ == "__main__":
    year = int(sys.argv[1])
    tier = sys.argv[2] if len(sys.argv) > 2 else "calibrated"
    stage = sys.argv[3] if len(sys.argv) > 3 else "v4.0"
    evaluate_v4(year, tier, stage)
