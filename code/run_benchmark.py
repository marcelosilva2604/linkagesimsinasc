"""
Phase 3 driver: run benchmark strategies on the synthetic grid.

Produces the paper's central comparison table. S1-S3 are as-published M:N key
joins; S4-S5 are the internal v1/v2 ancestors; S7 is frozen v3.2; S8 is v4.2
(tau=0.3). S6 (splink Fellegi-Sunter) runs from bench_splink.py separately.

Usage: python3 run_benchmark.py <year> [tier]
Writes outputs/benchmark/benchmark_<year>_<tier>.json (+ prints table).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import xgboost as xgb

from common import OUTPUTS, load_sinasc
from strategies import v32_core
from strategies_bench import (match_metrics, pair_metrics, s1_labsus,
                              s2_fabiano, s3_strict, s4_v1, s5_v2)
from v4ml import (MODEL_PATH, assign, build_records, candidate_pairs,
                  exact_p13, load_synth)

TAU = 0.3


def run(year: int, tier: str = "calibrated"):
    sim_rows, true_row, no_pair = load_synth(year, tier)
    sinasc = load_sinasc(year)
    results = []

    for name, fn in (("S1 date+mun (labsus-style)", s1_labsus),
                     ("S2 date+sex+weight (no mun)", s2_fabiano),
                     ("S3 strict 6-field (JOSIF-style)", s3_strict)):
        pairs = fn(sinasc, sim_rows)
        results.append(pair_metrics(pairs, sim_rows, true_row, no_pair, name))
        print(f"done {name}", flush=True)

    for name, fn in (("S4 v1 (single key)", s4_v1),
                     ("S5 v2 (+mun fallback)", s5_v2)):
        matches = fn(sinasc, sim_rows)
        results.append(match_metrics(matches, sim_rows, true_row, no_pair, name))
        print(f"done {name}", flush=True)

    matches, _ = v32_core(sinasc, sim_rows)
    results.append(match_metrics(matches, sim_rows, true_row, no_pair, "S7 v3.2 (4-pass greedy)"))
    print("done S7", flush=True)

    records = build_records(year)
    m13, used, indexes = exact_p13(records, sim_rows)
    cpairs = candidate_pairs(records, sim_rows, m13, used, indexes)
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    scores = model.predict_proba(np.array([f for _, _, f in cpairs]))[:, 1]
    m13.update(assign(cpairs, scores, TAU))
    results.append(match_metrics(m13, sim_rows, true_row, no_pair,
                                 f"S8 v4.2 (ML+1:1, tau={TAU})"))
    print("done S8", flush=True)

    out_dir = OUTPUTS / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"benchmark_{year}_{tier}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Benchmark {year} [{tier}] ===")
    hdr = f"{'strategy':<34} {'pairs':>9} {'match%':>7} {'m2m%':>6} {'prec%':>7} {'rec%':>7} {'F1%':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['strategy']:<34} {r['pairs']:>9,} {r['match_rate']*100:>7.1f} "
              f"{r['m2m_rate']*100:>6.1f} {r['precision_pairs']*100:>7.2f} "
              f"{r['recall_pairable']*100:>7.2f} {r['f1']*100:>7.2f}")


if __name__ == "__main__":
    run(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "calibrated")
