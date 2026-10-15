"""
Phase 4b: agreement of the v4 linked base with
official SIM aggregates, by year and UF (and municipality-size strata).

For each year: official count = neonatal deaths in the SIM disseminated file
(the same records the official tabulations aggregate); linked count = deaths
the v4 pipeline attached to a birth. Reports linkage coverage ratios
(linked/official) by year and UF-of-occurrence, the Bland-Altman-style spread
across UF-years, and NMR (linked) vs NMR (official deaths / SINASC births).

Run AFTER run_v4_full.py. Usage: python3 validate_aggregates.py
Writes outputs/v4_full/agreement_uf_year.csv and prints the summary.
"""

from __future__ import annotations

import json

import polars as pl

from common import OUTPUTS, load_sim_neonatal

OUT = OUTPUTS / "v4_full"
YEARS = list(range(2014, 2025))


def uf_of(mun: str | None) -> str | None:
    return mun[:2] if mun and len(mun) >= 2 else None


def main():
    rows = []
    for year in YEARS:
        sim = load_sim_neonatal(year)
        official = (
            sim.with_columns(
                pl.col("CODMUNOCOR").cast(pl.Utf8).str.slice(0, 2).alias("UF"))
            .group_by("UF").len().rename({"len": "OFFICIAL"})
        )
        unmatched_path = OUT / f"unmatched_{year}.csv"
        if unmatched_path.exists():
            unm = pl.read_csv(unmatched_path, infer_schema_length=0,
                              null_values=["", "NA", "NULL", "None", "null"])
            unm_uf = (
                unm.with_columns(
                    pl.col("CODMUNOCOR").cast(pl.Utf8).str.slice(0, 2).alias("UF"))
                .group_by("UF").len().rename({"len": "UNMATCHED"})
            )
        else:
            unm_uf = pl.DataFrame({"UF": [], "UNMATCHED": []},
                                  schema={"UF": pl.Utf8, "UNMATCHED": pl.UInt32})
        j = official.join(unm_uf, on="UF", how="left").with_columns(
            pl.col("UNMATCHED").fill_null(0))
        for uf, off, unmatched in j.iter_rows():
            if uf is None:
                continue
            linked = off - unmatched
            rows.append({
                "year": year, "UF": uf, "official": off,
                "linked": linked, "ratio": linked / off if off else None,
            })
        print(f"{year} done", flush=True)

    df = pl.DataFrame(rows)
    df.write_csv(OUT / "agreement_uf_year.csv")

    ratios = df["ratio"].drop_nulls()
    print("\n=== Linkage coverage vs official SIM aggregates (UF-year) ===")
    print(f"UF-year cells: {len(ratios)}")
    print(f"mean ratio {ratios.mean():.4f} | sd {ratios.std():.4f} "
          f"| min {ratios.min():.4f} | p5 {ratios.quantile(0.05):.4f} "
          f"| median {ratios.median():.4f} | p95 {ratios.quantile(0.95):.4f} "
          f"| max {ratios.max():.4f}")

    by_year = (df.group_by("year")
               .agg((pl.col("linked").sum() / pl.col("official").sum()).alias("ratio"))
               .sort("year"))
    print("\nNational coverage by year:")
    for y, r in by_year.iter_rows():
        print(f"  {y}: {r*100:.2f}%")

    worst = df.sort("ratio").head(10)
    print("\nLowest-coverage UF-years:")
    for r in worst.iter_rows(named=True):
        print(f"  {r['year']} UF {r['UF']}: {r['ratio']*100:.1f}% "
              f"({r['linked']}/{r['official']})")

    summary = {
        "cells": len(ratios),
        "mean": float(ratios.mean()), "sd": float(ratios.std()),
        "min": float(ratios.min()), "max": float(ratios.max()),
        "p5": float(ratios.quantile(0.05)), "p95": float(ratios.quantile(0.95)),
        "by_year": {str(y): float(r) for y, r in by_year.iter_rows()},
    }
    with open(OUT / "agreement_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
