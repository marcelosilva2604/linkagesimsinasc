"""
Downloads the public input files used in the study from the Brazilian Ministry of Health open data bucket and checks
each file against the SHA-256 recorded in ../input_hashes.csv.

  SIM    https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SIM/csv/Mortalidade_Geral_<year>_csv.zip     (2014-2024)
  SINASC https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINASC/csv/SINASC_<year>_csv.zip             (2013-2024)

Layout written under DEST (default: LINKAGE_DATA_DIR or ./data):
  raw/SIM/Mortalidade_Geral_<year>.csv   raw/SINASC/SINASC_<year>.csv

The publisher replaces files from time to time. ../input_hashes.csv records two hashes per file: the version used in the
study (sha256_study) and the version published on 2026-10-15 (sha256_published_20261015). Four files differ between the two
(SIM 2022, 2023, 2024 and SINASC 2024); results/replication_current_files/ gives the numbers obtained with the later
versions. A file matching neither hash is reported as DIFFERS and kept.

Usage: python3 download_inputs.py [--dest DIR] [--years 2014 2015 ...] [--system SIM|SINASC] [--check-only]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br"
HASHES = Path(__file__).resolve().parent.parent / "input_hashes.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def url(system: str, year: int) -> str:
    if system == "SIM":
        return f"{BASE}/SIM/csv/Mortalidade_Geral_{year}_csv.zip"
    return f"{BASE}/SINASC/csv/SINASC_{year}_csv.zip"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.environ.get("LINKAGE_DATA_DIR", os.environ.get("MIM_DATA_DIR", "./data")))
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--system", choices=["SIM", "SINASC"])
    ap.add_argument("--check-only", action="store_true", help="only hash files already in DEST")
    a = ap.parse_args()
    dest = Path(a.dest)
    todo = [r for r in csv.DictReader(HASHES.open())
            if (not a.years or int(r["year"]) in a.years) and (not a.system or r["system"] == a.system)]
    bad = 0
    for r in todo:
        sysname, year, fname = r["system"], int(r["year"]), r["file"]
        target = dest / "raw" / sysname / fname
        if not target.exists() and not a.check_only:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmp:
                z = Path(tmp) / "f.zip"
                print(f"download {url(sysname, year)}", flush=True)
                urllib.request.urlretrieve(url(sysname, year), z)
                with zipfile.ZipFile(z) as zf:
                    members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
                    with zf.open(members[0]) as src, target.open("wb") as out:
                        shutil.copyfileobj(src, out)
        if not target.exists():
            print(f"missing  {target}")
            bad += 1
            continue
        got = sha256(target)
        if got == r["sha256_study"]:
            label = "study version      "
        elif got == r["sha256_published_20261015"]:
            label = "published 2026-10-15"
        else:
            label = "DIFFERS            "
            bad += 1
        print(f"{label} {sysname} {year} {fname} sha256={got[:16]}...", flush=True)
    print("every file matches a recorded version" if not bad else f"{bad} file(s) missing or unrecognized")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
