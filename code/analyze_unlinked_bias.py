"""
Phase 5: bias profile of unlinked deaths.

Compares unmatched vs matched SIM neonatal deaths (2014-2024 pooled and per
year) on region, birth weight strata, place of death, race/colour, mother's
age and schooling, and fits a logistic model of P(unlinked | X) with 95% CIs
- the honest-limitations section of the paper and the IPW input for articles
5 and 6.

Run AFTER run_v4_full.py. Usage: python3 analyze_unlinked_bias.py
Writes outputs/v4_full/bias_profile.csv and bias_logit.csv.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from common import OUTPUTS, load_sim_neonatal

OUT = OUTPUTS / "v4_full"
YEARS = list(range(2014, 2025))

REGION = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"}


def peso_stratum(w) -> str:
    try:
        w = int(float(w))
    except (TypeError, ValueError):
        return "missing"
    if w < 1000:
        return "<1000"
    if w < 1500:
        return "1000-1499"
    if w < 2500:
        return "1500-2499"
    return ">=2500"


def build_frame() -> pl.DataFrame:
    frames = []
    for year in YEARS:
        sim = load_sim_neonatal(year)
        unmatched_path = OUT / f"unmatched_{year}.csv"
        unmatched_keys: set[tuple] = set()
        if unmatched_path.exists():
            unm = pl.read_csv(unmatched_path, infer_schema_length=0,
                              null_values=["", "NA", "NULL", "None", "null"])
            for r in unm.iter_rows(named=True):
                unmatched_keys.add((r.get("DTNASC"), r.get("SEXO"), r.get("DTOBITO"),
                                    r.get("CAUSABAS"), r.get("CODMUNOCOR")))
        cols = ["DTNASC", "SEXO", "DTOBITO", "CAUSABAS", "CODMUNOCOR",
                "PESO", "RACACOR", "IDADEMAE", "ESCMAE", "LOCOCOR", "IDADE_DIAS"]
        df = sim.select([pl.col(c).cast(pl.Utf8) for c in cols if c in sim.columns])
        rows = []
        for r in df.iter_rows(named=True):
            key = (r.get("DTNASC"), r.get("SEXO"), r.get("DTOBITO"),
                   r.get("CAUSABAS"), r.get("CODMUNOCOR"))
            rows.append({
                "year": year,
                "unlinked": int(key in unmatched_keys),
                "region": REGION.get((r.get("CODMUNOCOR") or " ")[:1], "unknown"),
                "peso_stratum": peso_stratum(r.get("PESO")),
                "racacor": r.get("RACACOR") or "missing",
                "escmae": r.get("ESCMAE") or "missing",
                "lococor": r.get("LOCOCOR") or "missing",
                "early_neonatal": int((r.get("IDADE_DIAS") or "99") in
                                      [str(d) for d in range(0, 7)]),
            })
        frames.append(pl.DataFrame(rows))
        print(f"{year} done", flush=True)
    return pl.concat(frames)


def main():
    df = build_frame()
    overall = df["unlinked"].mean()
    print(f"\nPooled unlinked rate: {overall*100:.2f}% of {len(df):,} deaths")

    profile_rows = []
    for var in ["region", "peso_stratum", "racacor", "escmae", "lococor", "year"]:
        g = (df.group_by(var).agg(pl.len().alias("n"),
                                  pl.col("unlinked").mean().alias("unlinked_rate"))
             .sort("unlinked_rate", descending=True))
        for cat, n, rate in g.iter_rows():
            profile_rows.append({"variable": var, "category": str(cat),
                                 "n": n, "unlinked_rate": rate})
        print(f"\n{var}:")
        for cat, n, rate in g.iter_rows():
            print(f"  {str(cat):<12} n={n:>7,}  unlinked {rate*100:5.2f}%")
    pl.DataFrame(profile_rows).write_csv(OUT / "bias_profile.csv")

    # logistic model with one-hot categories (statsmodels-free: sklearn + Wald CIs
    # via numpy on the observed information matrix)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    X_cats = df.select(["region", "peso_stratum", "racacor", "lococor"]).to_numpy()
    year_num = (df["year"].to_numpy().astype(float) - 2014.0).reshape(-1, 1)
    enc = OneHotEncoder(drop="first", sparse_output=False)
    X = np.hstack([enc.fit_transform(X_cats), year_num])
    y = df["unlinked"].to_numpy()

    lr = LogisticRegression(max_iter=2000, C=1e6)
    lr.fit(X, y)
    p = lr.predict_proba(X)[:, 1]
    W = p * (1 - p)
    Xd = np.hstack([np.ones((len(X), 1)), X])
    info = Xd.T @ (Xd * W[:, None])
    cov = np.linalg.pinv(info)
    se = np.sqrt(np.diag(cov))[1:]
    names = list(enc.get_feature_names_out(
        ["region", "peso_stratum", "racacor", "lococor"])) + ["year_trend"]
    out_rows = []
    print("\nLogistic model P(unlinked | X) - odds ratios (95% CI):")
    for name, b, s in zip(names, lr.coef_[0], se):
        or_, lo, hi = np.exp(b), np.exp(b - 1.96 * s), np.exp(b + 1.96 * s)
        out_rows.append({"term": name, "OR": or_, "lo95": lo, "hi95": hi})
        print(f"  {name:<28} OR {or_:6.2f} ({lo:5.2f}-{hi:5.2f})")
    pl.DataFrame(out_rows).write_csv(OUT / "bias_logit.csv")


if __name__ == "__main__":
    main()
