"""Input manifest: input manifest: SHA-256, size and row counts of every input file used by linkage_final.py,
library versions and parameters. Writes outputs/revision/mim_final/input_manifest.json. Usage: python3 audit_mim_inputs.py"""
import hashlib, json, platform, sys
import polars as pl, xgboost, scipy
import rv_common  # noqa: F401
from rv_common import REV
from common import PROCESSED, RAW
from v4ml import MODEL_PATH

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

files = [PROCESSED / f"SIM_neonatal_{y}.csv" for y in range(2014, 2025)] + \
        [RAW / "SINASC" / f"SINASC_{y}.csv" for y in range(2013, 2025)] + [MODEL_PATH, REV.parent / 'v4_full' / 'stats_all.json']
out = {"python": sys.version, "platform": platform.platform(), "polars": pl.__version__, "xgboost": xgboost.__version__,
       "scipy": scipy.__version__, "tau_reference": 0.3, "seed": 42, "files": []}
for p in files:
    if not p.exists():
        raise SystemExit(f"required input missing: {p}")
    row = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}
    if p.suffix == ".csv":
        with open(p, "rb") as f:
            row["lines"] = sum(1 for _ in f)
    out["files"].append(row)
    print(p.name, flush=True)
json.dump(out, open(REV / "mim_final" / "input_manifest.json", "w"), indent=2)
