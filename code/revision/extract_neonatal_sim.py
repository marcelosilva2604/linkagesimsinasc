"""
Reconstruction of the neonatal extraction from the general SIM files (Mortalidade_Geral_<year>.csv, DATASUS).
Rule: death records whose date of death minus date of birth is 0-27 completed days, kept in file order. The rebuilt
records are compared with the preserved extracts SIM_neonatal_<year>.csv (record count, and agreement by position on
eight fields, ignoring leading zeros that the preserved extracts dropped). Nothing in the data folders is modified; with --write-dir the rebuilt files are written elsewhere.
Output: outputs/revision/mim_final/extraction_check.csv (aggregate counts and file hashes; no record-level data).
Usage: python3 extract_neonatal_sim.py [--write-dir DIR]
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import polars as pl

import rv_common  # noqa: F401
from rv_common import REV
from common import PROCESSED, RAW

OUT = REV / "mim_final"
KEY = ["DTOBITO", "DTNASC", "SEXO", "CAUSABAS", "CODMUNOCOR", "CODMUNRES", "IDADEMAE", "SEMAGESTAC"]


def read_general(path: Path) -> pl.DataFrame:
    head = path.open(encoding="latin-1").readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    return pl.read_csv(path, separator=sep, infer_schema_length=0, encoding="utf8-lossy")


def neonatal(df: pl.DataFrame) -> pl.DataFrame:
    d = df.with_columns(
        pl.col("DTOBITO").str.zfill(8).str.to_date("%d%m%Y", strict=False).alias("_do"),
        pl.col("DTNASC").str.zfill(8).str.to_date("%d%m%Y", strict=False).alias("_dn"))
    return d.filter((pl.col("_do") - pl.col("_dn")).dt.total_days().is_between(0, 27)).drop(["_do", "_dn"])


def seq(df: pl.DataFrame) -> pl.Series:
    return df.select([pl.col(c).cast(pl.Utf8).str.strip_chars().str.strip_chars_start("0").fill_null("") for c in KEY]).with_columns(
        pl.concat_str(KEY, separator="|").alias("K"))["K"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    write_dir = Path(sys.argv[sys.argv.index("--write-dir") + 1]) if "--write-dir" in sys.argv else None
    rows = []
    for y in range(2014, 2025):
        src = RAW / "SIM" / f"Mortalidade_Geral_{y}.csv"
        pres = PROCESSED / f"SIM_neonatal_{y}.csv"
        g = read_general(src)
        e = neonatal(g)
        if pres.exists():
            p = pl.read_csv(pres, infer_schema_length=0)
            same = int((seq(e) == seq(p)).sum()) if e.height == p.height else None
            n_pres = p.height
        else:  # no preserved extract to compare with: only the rebuilt counts are reported
            same, n_pres = None, None
        rows.append({"year": y, "general_file_records": g.height, "general_file_sha256": sha(src),
                     "rebuilt_neonatal": e.height, "preserved_extract": n_pres,
                     "same_count": (e.height == n_pres) if n_pres is not None else None,
                     "positions_agreeing_on_8_fields": same})
        print(rows[-1]["year"], e.height, p.height, same, flush=True)
        if write_dir is not None:
            write_dir.mkdir(parents=True, exist_ok=True)
            e.write_csv(write_dir / f"SIM_neonatal_{y}.csv")
    df = pl.DataFrame(rows)
    df.write_csv(OUT / "extraction_check.csv")
    print("total rebuilt", int(df["rebuilt_neonatal"].sum()))


if __name__ == "__main__":
    main()
