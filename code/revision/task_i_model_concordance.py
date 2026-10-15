"""
descriptive model of non-linkage and agreement of fields not used in linkage, both computed on the
consolidated C03 base (linkage_final.py): 260,219 deaths, one row each; accepted pairs one-to-one.

Model: logistic regression, outcome = death unlinked in C03; covariates region of occurrence (CODMUNOCOR), weight
category, race/color, place of death, SIM file year centered at 2014. References fixed in code. All categories kept,
including missing/ignored. Maximum likelihood (statsmodels), Wald 95% CI, convergence reported.
Agreement: fields CODMUNRES, SEMAGESTAC (20-45), ESCMAE (1-5), GRAVIDEZ (1-3), PARTO (1-2) between the death record and
the accepted birth record, by stage (exact P1-P3 vs classifier), against 50 permutations within stage and SIM file year, and within stage x SIM file year x state of occurrence.
CODMUNRES values ending in 0000 (state-level codes) are not comparable.
Descriptive only: does not estimate accuracy.

Outputs: outputs/revision/mim_final/nonlinkage_model_C03.csv, nonlinkage_model_metadata.json, concordance_C03.csv
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
import statsmodels.api as sm

import rv_common  # noqa: F401
from rv_common import REV
from common import load_sinasc

OUT = REV / "mim_final"
REGION = {"1": "North", "2": "Northeast", "3": "Southeast", "4": "South", "5": "Center-West"}
PLACE = {"1": "Hospital", "2": "Other health facility", "3": "Home", "4": "Public road", "5": "Other",
         "6": "Indigenous village", "9": "Ignored"}
FIELDS = ["CODMUNRES", "SEMAGESTAC", "ESCMAE", "GRAVIDEZ", "PARTO"]
VALID = {
    # six digits and not a state-level code (UF0000), which does not identify a municipality
    "CODMUNRES": lambda x: len(x) == 6 and x.isdigit() and not x.endswith("0000"),
    "SEMAGESTAC": lambda x: x.isdigit() and 20 <= int(x) <= 45,
    "ESCMAE": lambda x: x.isdigit() and 1 <= int(x) <= 5,
    "GRAVIDEZ": lambda x: x.isdigit() and 1 <= int(x) <= 3,
    "PARTO": lambda x: x.isdigit() and 1 <= int(x) <= 2,
}


def clean(v):
    if v is None:
        return None
    s = str(v).strip()
    return None if s == "" else s.split(".")[0]


def weight_cat(w):
    s = clean(w)
    if s is None:
        return "missing"
    try:
        g = float(s)
    except ValueError:
        return "invalid"
    if g < 200 or g > 8000:
        return "invalid"
    return "<1000" if g < 1000 else "1000-1499" if g < 1500 else "1500-2499" if g < 2500 else ">=2500"


def model():
    d = pl.read_parquet(OUT / "death_linkage_C03.parquet")
    rows = d.select(["death_file", "PESO", "CODMUNOCOR", "LOCOCOR", "RACACOR", "linked"]).to_dicts()
    cols = {"region": [], "weight": [], "race": [], "place": [], "year": [], "y": []}
    for r in rows:
        cols["region"].append(REGION.get((clean(r["CODMUNOCOR"]) or " ")[:1], "unknown"))
        cols["weight"].append(weight_cat(r["PESO"]))
        cols["race"].append("race_" + (clean(r["RACACOR"]) or "missing"))
        cols["place"].append(PLACE.get(clean(r["LOCOCOR"]) or "", "missing"))
        cols["year"].append(r["death_file"] - 2014)
        cols["y"].append(0 if r["linked"] else 1)
    refs = {"region": "Center-West", "weight": "1000-1499", "race": "race_1", "place": "Hospital"}
    names, mats = [], []
    for v, ref in refs.items():
        levels = sorted(set(cols[v]) - {ref})
        arr = np.array(cols[v])
        for lv in levels:
            names.append(f"{v}={lv}")
            mats.append((arr == lv).astype(float))
    names.append("year_per_calendar_year")
    mats.append(np.array(cols["year"], dtype=float))
    X = sm.add_constant(np.column_stack(mats))
    y = np.array(cols["y"])
    res = sm.Logit(y, X).fit(disp=0, maxiter=200, method="newton")
    ci = res.conf_int()
    out = []
    for k, n in enumerate(["const"] + names):
        b, lo, hi = res.params[k], ci[k][0], ci[k][1]
        v, _, lv = n.partition("=")
        cnt = int((np.array(cols[v]) == lv).sum()) if lv else None
        ev = int(((np.array(cols[v]) == lv) & (y == 1)).sum()) if lv else None
        out.append({"term": n, "n_in_category": cnt, "n_unlinked_in_category": ev, "OR": float(np.exp(b)),
                    "lo95": float(np.exp(lo)), "hi95": float(np.exp(hi)), "p_value": float(res.pvalues[k])})
    pl.DataFrame(out).write_csv(OUT / "nonlinkage_model_C03.csv")
    meta = {"n": int(len(y)), "events": int(y.sum()), "converged": bool(res.mle_retvals["converged"]),
            "iterations": int(res.mle_retvals.get("iterations", -1)), "references": refs,
            "weight_definition": "missing = null/blank; invalid = non-numeric or outside 200-8000 g",
            "region_definition": "region of occurrence of death (first digit of CODMUNOCOR)"}
    json.dump(meta, open(OUT / "nonlinkage_model_metadata.json", "w"), indent=2)
    print(meta)
    print(pl.DataFrame(out).to_pandas().round(3).to_string())


def concordance():
    acc = pl.read_parquet(OUT / "accepted_pairs_C03.parquet")
    deaths = pl.read_parquet(OUT / "death_linkage_C03.parquet").select(
        ["death_file", "death_row", "CODMUNOCOR"] + FIELDS).rename({f: f"S_{f}" for f in FIELDS})
    acc = acc.join(deaths, on=["death_file", "death_row"], how="left")
    parts = []
    for by in sorted(acc["birth_file"].unique().to_list()):
        sn = load_sinasc(by)
        if sn is None:
            continue
        sn = sn.with_row_index("birth_row").select(["birth_row"] + [f for f in FIELDS if f in sn.columns])
        sn = sn.with_columns([pl.col(f).cast(pl.Utf8) for f in FIELDS if f in sn.columns]).rename(
            {f: f"B_{f}" for f in FIELDS if f in sn.columns})
        sub = acc.filter(pl.col("birth_file") == by).join(
            sn.with_columns(pl.col("birth_row").cast(pl.Int32)), on="birth_row", how="left")
        parts.append(sub)
    pairs = pl.concat(parts, how="diagonal_relaxed")
    assert pairs.height == acc.height
    rng = np.random.default_rng(42)
    rows = []
    for stage_name, cond in [("exact", pl.col("stage") != "CLF"), ("classifier", pl.col("stage") == "CLF")]:
        sub = pairs.filter(cond)
        for f in FIELDS:
            if f"B_{f}" not in sub.columns:
                continue
            a = [clean(x) for x in sub[f"S_{f}"].to_list()]
            b = [clean(x) for x in sub[f"B_{f}"].to_list()]
            yr = np.array(sub["death_file"].to_list())
            ok = np.array([(x is not None and z is not None and VALID[f](x) and VALID[f](z)) for x, z in zip(a, b)])
            n = int(ok.sum())
            av = np.array(a, dtype=object)[ok]
            bv = np.array(b, dtype=object)[ok]
            ay = yr[ok]
            # block for the conditional comparator: SIM file year x state of occurrence of the death
            uf = np.array([(clean(x) or "")[:2] for x in sub["CODMUNOCOR"].to_list()])[ok]
            agree = float(np.mean(av == bv)) if n else float("nan")
            rnd = []
            for _ in range(50):
                m = 0
                for y in np.unique(ay):
                    idx = np.where(ay == y)[0]
                    m += int(np.sum(av[idx] == bv[idx][rng.permutation(len(idx))]))
                rnd.append(m / n)
            blk = np.array([f"{y}_{u}" for y, u in zip(ay, uf)])
            rnd2 = []
            for _ in range(50):
                m = 0
                for g in np.unique(blk):
                    idx = np.where(blk == g)[0]
                    m += int(np.sum(av[idx] == bv[idx][rng.permutation(len(idx))]))
                rnd2.append(m / n)
            rows.append({"stage": stage_name, "field": f, "n_pairs": sub.height, "n_compared": n,
                         "pct_comparable": 100 * n / sub.height, "agreement": agree,
                         "random_pairing_comparator": float(np.mean(rnd)),
                         "random_pairing_within_state_comparator": float(np.mean(rnd2))})
        # weight missing in the SIM death record
        w = [clean(x) is None for x in deaths.join(sub.select(["death_file", "death_row"]),
                                                  on=["death_file", "death_row"])["S_CODMUNRES"].to_list()]
    pl.DataFrame(rows).write_csv(OUT / "concordance_C03.csv")
    print(pl.DataFrame(rows).to_pandas().round(3).to_string())
    # weight missing in the death record by stage
    d = pl.read_parquet(OUT / "death_linkage_C03.parquet").filter(pl.col("linked"))
    d = d.with_columns((pl.col("PESO").is_null() | (pl.col("PESO").cast(pl.Utf8).str.strip_chars() == "")).alias("wm"),
                       (pl.col("stage") == "CLF").alias("clf"))
    print(d.group_by("clf").agg(pl.len(), pl.col("wm").mean()))


if __name__ == "__main__":
    import sys
    if "--concordance-only" not in sys.argv:
        model()
    concordance()
