"""
Revision Task F - concordance of fields NOT used in linkage, in real accepted pairs.

Rationale: no reference of true pairs exists, but SIM and SINASC share fields that the pipeline does not use
(mother's municipality of residence, plurality of pregnancy, type of delivery, gestational age in weeks,
maternal schooling, race/color, number of living children). If a stage links true pairs, its pairs should agree
on these fields at about the rate of true pairs; if it links wrong pairs, agreement falls toward the rate expected
for randomly paired records. This is indirect evidence, not a measure of accuracy.

For each year the in-memory v4.2 linkage of run_v4_full.py is repeated (stored classifier, tau 0.3; nothing is written
to outputs/v4_full). For accepted pairs with a current-year birth, agreement is computed by stage: exact passes
(1-3 pooled) and the classifier stage (6). Random-pairing agreement is the mean over 50 permutations of the SINASC
values within the same group and year. Agreement is descriptive: it does not identify the share of correct pairs, because the compared fields are associated
with the linkage keys and fill quality differs between stages (for example, no exact-pass pair has a missing weight).

Output: outputs/revision/taskF_concordance_<years>.csv
Usage: /usr/bin/python3 task_f_real_concordance.py [year ...]   (default 2014-2024)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

import rv_common  # noqa: F401
from rv_common import REV
from common import load_sim_neonatal, load_sinasc
from strategies import prepare_sim_rows
from strategies_v4 import PREV_OFFSET
from v4ml import MODEL_PATH, assign, build_records, candidate_pairs, exact_p13

TAU = 0.3
FIELDS = ["CODMUNRES", "SEMAGESTAC", "ESCMAE", "GRAVIDEZ", "PARTO"]
# Fields dropped after review: RACACOR (SIM refers to the deceased, SINASC to the mother/infant record) and
# QTDFILVIVO (the death certificate excludes the deceased from the count; coding of 99 = ignored).
VALID = {  # substantive codes only; ignored/invalid codes are excluded before comparison
    "CODMUNRES": lambda x: len(x) == 6 and x.isdigit(),
    "SEMAGESTAC": lambda x: x.isdigit() and 20 <= int(x) <= 45,
    "ESCMAE": lambda x: x.isdigit() and 1 <= int(x) <= 5,
    "GRAVIDEZ": lambda x: x.isdigit() and 1 <= int(x) <= 3,
    "PARTO": lambda x: x.isdigit() and 1 <= int(x) <= 2,
}
RNG = np.random.default_rng(42)


def vals(df: pl.DataFrame, col: str) -> np.ndarray:
    return df[col].cast(pl.Utf8).to_numpy() if col in df.columns else np.array([None] * len(df), dtype=object)


def link_year(year: int, model):
    sim_atual = load_sim_neonatal(year)
    sim_next = load_sim_neonatal(year + 1)
    sim_rows = prepare_sim_rows(sim_atual)
    n_atual = len(sim_rows)
    sim_all = sim_atual
    if sim_next is not None:
        sim_jan = sim_next.filter(pl.col("DTNASC").str.slice(4, 4) == str(year))
        sim_rows += prepare_sim_rows(sim_jan)
        sim_all = pl.concat([sim_atual, sim_jan], how="diagonal_relaxed")
    records = build_records(year)
    matches, used, indexes = exact_p13(records, sim_rows)
    pairs = candidate_pairs(records, sim_rows, matches, used, indexes)
    if pairs:
        sc = model.predict_proba(np.array([f for _, _, f in pairs]))[:, 1]
        matches.update(assign(pairs, sc, TAU))
    return matches, sim_all


def clean(v):
    return None if v is None or str(v).strip() == "" else str(v).split(".")[0].strip()


def agreement(a: np.ndarray, b: np.ndarray, field: str):
    ok_f = VALID[field]
    aa = [clean(x) for x in a]
    bb = [clean(x) for x in b]
    ok = np.array([(x is not None and y is not None and ok_f(x) and ok_f(y)) for x, y in zip(aa, bb)])
    if ok.sum() == 0:
        return np.nan, 0, ok, aa, bb
    eq = np.array([x == y for x, y, o in zip(aa, bb, ok) if o])
    return float(eq.mean()), int(ok.sum()), ok, aa, bb


def run(years):
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    rows = []
    for year in years:
        matches, sim_all = link_year(year, model)
        sinasc = load_sinasc(year)
        sim_ids = np.array([i for i, (j, p) in matches.items() if j < PREV_OFFSET], dtype=int)
        sn_ids = np.array([matches[i][0] for i in sim_ids], dtype=int)
        pas = np.array([matches[i][1] for i in sim_ids], dtype=int)
        grp = np.where(np.isin(pas, [1, 2, 3]), "exact", "classifier")
        peso_sim = vals(sim_all, "PESO")[sim_ids]
        for g in ("exact", "classifier"):
            m = grp == g
            miss = np.array([clean(x) is None for x in peso_sim[m]])
            rows.append({"year": year, "field": "PESO_missing_in_SIM", "stage": g, "n_pairs": int(m.sum()),
                         "n_compared": int(m.sum()), "agreement": float(miss.mean()) if m.sum() else np.nan,
                         "random_agreement": np.nan})
        for f in FIELDS:
            a_all = vals(sim_all, f)[sim_ids]
            b_all = vals(sinasc, f)[sn_ids]
            for g in ("exact", "classifier"):
                m = grp == g
                A, n, ok, aa, bb = agreement(a_all[m], b_all[m], f)
                if n == 0:
                    continue
                av = np.array(aa, dtype=object)[ok]; bv = np.array(bb, dtype=object)[ok]
                rnd = []
                for _ in range(50):
                    perm = RNG.permutation(len(bv))
                    rnd.append(float(np.mean(av == bv[perm])))
                rows.append({"year": year, "field": f, "stage": g, "n_pairs": int(m.sum()), "n_compared": n,
                             "agreement": A, "random_agreement": float(np.mean(rnd))})
        print(year, "done", flush=True)
    df = pd.DataFrame(rows)
    out = REV / f"taskF_concordance_{years[0]}_{years[-1]}.csv"
    df.to_csv(out, index=False)
    print("written", out)
    return df


if __name__ == "__main__":
    ys = [int(a) for a in sys.argv[1:]] or list(range(2014, 2025))
    df = run(ys)
    d = df[df["field"].isin(FIELDS)]
    for f in FIELDS:
        for g in ("exact", "classifier"):
            x = d[(d.field == f) & (d.stage == g)]
            if len(x):
                w = x["n_compared"]
                print(f"{f:11s} {g:10s} n={int(w.sum()):7d} agree={np.average(x['agreement'], weights=w):.3f} "
                      f"random={np.average(x['random_agreement'], weights=w):.3f}")
