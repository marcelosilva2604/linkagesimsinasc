"""
check the v3.2 reimplementation against the original pipeline outputs.

Reassembles the death set exactly like executar_linkage_ano (SIM of the year
+ January-window deaths from the following year born in the year), runs the
reimplemented passes, and prints the per-pass profile and match rate. Values
must reproduce the logged v3.2 results (match rate 88.7% in 2014, 92.5% in
2024, ~91% overall) for the harness to be usable in the benchmark.

Usage: python3 run_real_check.py [year ...]
"""

from __future__ import annotations

import sys

import polars as pl

from common import load_sinasc, load_sim_neonatal
from strategies import prepare_sim_rows, v32_core


def real_check(year: int) -> dict:
    sinasc = load_sinasc(year)
    sim_atual = load_sim_neonatal(year)
    sim_next = load_sim_neonatal(year + 1)

    # v3.2 casts to Utf8 (preparar_sim) BEFORE concatenating the two years
    sim_rows = prepare_sim_rows(sim_atual)
    n_jan = 0
    if sim_next is not None:
        sim_jan = sim_next.filter(pl.col("DTNASC").str.slice(4, 4) == str(year))
        n_jan = len(sim_jan)
        sim_rows = sim_rows + prepare_sim_rows(sim_jan)
    total = len(sim_rows)
    matches, pass_counts = v32_core(sinasc, sim_rows)
    matched = len(matches)

    print(f"\n=== {year} ===")
    print(f"SINASC {len(sinasc):,} births | SIM {len(sim_atual):,} (+{n_jan:,} jan/{year+1}) = {total:,} deaths")
    cum = 0
    for p in (1, 2, 3, 4):
        cum += pass_counts[p]
        print(f"  Pass {p}: {pass_counts[p]:,} (+{pass_counts[p]/total*100:.1f}%) cum {cum/total*100:.1f}%")
    print(f"  Unmatched: {total - matched:,} ({(total - matched)/total*100:.1f}%)")
    print(f"  MATCH RATE: {matched/total*100:.2f}%")
    return {"year": year, "total": total, "matched": matched, "pass_counts": pass_counts}


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or [2014, 2019, 2024]
    for y in years:
        real_check(y)
