"""
shared loaders. Replicas of the v3.2 baseline loaders (same parsing options, same
normalisation) so that strategy reimplementations behave identically.
Source data is read-only; nothing is ever written to the source project.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

SOURCE = Path(os.environ.get("LINKAGE_DATA_DIR", os.environ.get("MIM_DATA_DIR", "./data")))
RAW = SOURCE / "raw"
PROCESSED = SOURCE / "processed"

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"


def detect_sep(path: Path) -> str:
    with open(path, encoding="latin1") as f:
        header = f.readline()
    return ";" if header.count(";") > header.count(",") else ","


def normalize_sexo(df: pl.DataFrame, col: str = "SEXO") -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col(col).cast(pl.Utf8).str.to_uppercase().is_in(["1", "M", "MASCULINO"]))
        .then(pl.lit("1"))
        .when(pl.col(col).cast(pl.Utf8).str.to_uppercase().is_in(["2", "F", "FEMININO"]))
        .then(pl.lit("2"))
        .otherwise(pl.col(col).cast(pl.Utf8))
        .alias(col)
    )


def load_sinasc(year: int) -> pl.DataFrame | None:
    path = RAW / "SINASC" / f"SINASC_{year}.csv"
    if not path.exists():
        return None
    df = pl.read_csv(
        path, separator=detect_sep(path), encoding="latin1",
        infer_schema_length=10000, ignore_errors=True, truncate_ragged_lines=True,
        null_values=["", "NA", "NULL", "None", "null"],
    )
    df = df.rename({c: c.strip('"').upper() for c in df.columns})
    df = normalize_sexo(df)
    df = df.with_columns(pl.col("DTNASC").cast(pl.Utf8).str.zfill(8).alias("DTNASC"))
    return df


def load_sim_neonatal(year: int) -> pl.DataFrame | None:
    path = PROCESSED / f"SIM_neonatal_{year}.csv"
    if not path.exists():
        return None
    df = pl.read_csv(
        path, infer_schema_length=10000, ignore_errors=True,
        null_values=["", "NA", "NULL", "None", "null"],
    )
    df = df.rename({c: c.strip('"').upper() for c in df.columns})
    df = normalize_sexo(df)
    df = df.with_columns(pl.col("DTNASC").cast(pl.Utf8).str.zfill(8).alias("DTNASC"))
    return df
