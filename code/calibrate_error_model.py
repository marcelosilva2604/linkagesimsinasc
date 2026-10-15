"""
Phase 1a: calibrate the synthetic error model from real data.

Measures, for each sentinel year, the field-quality profile of SIM neonatal
records vs SINASC records on every field used by the linkage keys:
missingness, invalid values, sex indeterminacy, weight rounding profile.
These measured rates parameterise make_synthetic_reference.py (calibrated tier);
optimistic/pessimistic tiers are derived as 0.5x / 2x the calibrated rates.

Reads raw/processed files from the frozen source project (read-only).
Writes outputs/calibration/error_model.json and a printable summary table.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from common import SOURCE
OUT = Path(__file__).resolve().parent.parent / "outputs" / "calibration"
SENTINEL_YEARS = [2014, 2019, 2024]

SIM_FIELDS = ["DTNASC", "SEXO", "PESO", "IDADEMAE", "CODESTAB", "CODMUNNATU", "CODMUNOCOR", "SEMAGESTAC", "ESCMAE"]
SINASC_FIELDS = ["DTNASC", "SEXO", "PESO", "IDADEMAE", "CODESTAB", "CODMUNNASC", "SEMAGESTAC", "ESCMAE"]


def detect_sep(path: Path) -> str:
    with open(path, encoding="latin1") as f:
        header = f.readline()
    return ";" if header.count(";") > header.count(",") else ","


def load_sinasc(year: int) -> pl.DataFrame:
    path = SOURCE / "raw" / "SINASC" / f"SINASC_{year}.csv"
    df = pl.read_csv(
        path, separator=detect_sep(path), encoding="latin1",
        infer_schema_length=0, ignore_errors=True, truncate_ragged_lines=True,
        null_values=["", "NA", "NULL", "None", "null"],
    )
    df = df.rename({c: c.strip('"').upper() for c in df.columns})
    return df.select([c for c in SINASC_FIELDS if c in df.columns])


def load_sim(year: int) -> pl.DataFrame:
    path = SOURCE / "processed" / f"SIM_neonatal_{year}.csv"
    df = pl.read_csv(
        path, infer_schema_length=0, ignore_errors=True,
        null_values=["", "NA", "NULL", "None", "null"],
    )
    df = df.rename({c: c.strip('"').upper() for c in df.columns})
    return df.select([c for c in SIM_FIELDS + ["IDADE_DIAS"] if c in df.columns])


def profile(df: pl.DataFrame, fields: list[str]) -> dict:
    n = len(df)
    out = {"n": n}
    for f in fields:
        if f not in df.columns:
            out[f] = None
            continue
        col = pl.col(f).cast(pl.Utf8)
        missing = df.select(col.is_null().sum()).item()
        stats = {"missing": missing / n}
        if f == "SEXO":
            norm = df.select(
                pl.when(col.str.to_uppercase().is_in(["1", "M", "MASCULINO"])).then(pl.lit("1"))
                .when(col.str.to_uppercase().is_in(["2", "F", "FEMININO"])).then(pl.lit("2"))
                .otherwise(pl.lit("other")).alias("s")
            )["s"]
            stats["indeterminate_or_other"] = (norm == "other").sum() / n
        if f == "PESO":
            w = df.select(col.cast(pl.Int64, strict=False).alias("w"))["w"]
            valid = w.filter((w >= 200) & (w <= 8000))
            stats["invalid_or_missing"] = 1 - len(valid) / n
            if len(valid):
                stats["pct_mult_100"] = ((valid % 100) == 0).sum() / len(valid)
                stats["pct_mult_50"] = ((valid % 50) == 0).sum() / len(valid)
                stats["pct_mult_10"] = ((valid % 10) == 0).sum() / len(valid)
        if f == "IDADEMAE":
            a = df.select(col.cast(pl.Int64, strict=False).alias("a"))["a"]
            stats["missing_or_invalid"] = 1 - ((a >= 8) & (a <= 60)).sum() / n
        if f == "DTNASC":
            d = df.select(col.str.zfill(8).alias("d"))["d"]
            stats["not_8digit"] = 1 - (d.str.len_chars() == 8).sum() / n
        out[f] = stats
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for year in SENTINEL_YEARS:
        print(f"=== {year} ===", flush=True)
        sim = load_sim(year)
        sinasc = load_sinasc(year)
        result[year] = {
            "SIM_neonatal": profile(sim, SIM_FIELDS),
            "SINASC": profile(sinasc, SINASC_FIELDS),
        }
        if "IDADE_DIAS" in sim.columns:
            d = sim["IDADE_DIAS"].cast(pl.Int64, strict=False)
            result[year]["SIM_neonatal"]["idade_dias_range"] = [int(d.min()), int(d.max())]
        print(json.dumps(result[year], indent=2, default=float), flush=True)

    with open(OUT / "error_model.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nSaved {OUT / 'error_model.json'}")


if __name__ == "__main__":
    main()
