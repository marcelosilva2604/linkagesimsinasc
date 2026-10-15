"""
Revision Task D - unlinked-death profile flagged by RECORD IDENTITY.

The stored unmatched_<year>.csv files keep no row identifier, so identity is
recovered by re-running the deterministic v4.2 matching of run_v4_full.py
(same data assembly, same functions, same stored model, tau 0.3) in memory,
WITHOUT writing any linked base. For each SIM neonatal file of year y, row r is
LINKED iff
    r was matched in the run of year y  (own-file rows, i < n_atual), or
    r was matched as a January-window row in the run of year y-1
exactly the `unique_matched_file` accounting of run_v4_full.py (stats_<y>.json).
Everything else is UNLINKED. Checks: per-year linked counts must equal
stats_<y>.json unique_matched_file, and the per-run unmatched count must equal
the row count of the stored unmatched_<y>.csv.

Then the pooled profile and logistic model of analyze_unlinked_bias.py are
recomputed with the identity flag (same covariates, encoder, solver, Wald CIs)
and with the original key-based flag (to reproduce the published S2).

Usage: python3 task_d_unlinked_identity.py
Writes outputs/revision/taskD_*.csv/json.
"""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
import xgboost as xgb

import rv_common  # noqa: F401
from rv_common import REV
from common import OUTPUTS, load_sim_neonatal
from strategies import prepare_sim_rows
from v4ml import MODEL_PATH, assign, build_records, candidate_pairs, exact_p13

TAU = 0.3
V4 = OUTPUTS / "v4_full"
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


def rerun_identity(model):
    """Return {year: set(linked row positions in SIM_neonatal_<year>.csv)} and checks."""
    own: dict[int, set[int]] = {}
    jan_into: dict[int, set[int]] = {}   # year y+1 file rows matched in run y
    checks = []
    for year in YEARS:
        t = time.time()
        sim_atual = load_sim_neonatal(year)
        sim_next = load_sim_neonatal(year + 1)
        sim_rows = prepare_sim_rows(sim_atual)
        n_atual = len(sim_rows)
        jan_row_ids: list[int] = []
        if sim_next is not None:
            sim_jan = sim_next.with_row_index("RIDX").filter(
                pl.col("DTNASC").str.slice(4, 4) == str(year))
            jan_row_ids = sim_jan["RIDX"].to_list()
            sim_rows += prepare_sim_rows(sim_jan.drop("RIDX"))
        records = build_records(year)
        matches, used, indexes = exact_p13(records, sim_rows)
        pairs = candidate_pairs(records, sim_rows, matches, used, indexes)
        if pairs:
            sc = model.predict_proba(np.array([f for _, _, f in pairs]))[:, 1]
            matches.update(assign(pairs, sc, TAU))
        own[year] = {i for i in matches if i < n_atual}
        jan_into[year + 1] = {int(jan_row_ids[i - n_atual]) for i in matches if i >= n_atual}
        stats = json.loads((V4 / f"stats_{year}.json").read_text())
        unm_rows = pl.read_csv(V4 / f"unmatched_{year}.csv", infer_schema_length=0).height
        checks.append({"year": year, "matched_run": len(matches),
                       "matched_stats": stats["matched"],
                       "unmatched_run": len(sim_rows) - len(matches),
                       "unmatched_csv_rows": unm_rows,
                       "seconds": round(time.time() - t, 1)})
        print(checks[-1], flush=True)
    linked = {}
    for year in YEARS:
        linked[year] = own[year] | jan_into.get(year, set())
        stats = json.loads((V4 / f"stats_{year}.json").read_text())
        checks[YEARS.index(year)].update({
            "linked_identity": len(linked[year]),
            "unique_matched_file_stats": stats["unique_matched_file"],
            "deaths_file": stats["deaths_file"]})
    return linked, checks


def build_frame(linked: dict[int, set[int]]) -> pl.DataFrame:
    frames = []
    cols = ["DTNASC", "SEXO", "DTOBITO", "CAUSABAS", "CODMUNOCOR",
            "PESO", "RACACOR", "IDADEMAE", "ESCMAE", "LOCOCOR", "IDADE_DIAS"]
    for year in YEARS:
        sim = load_sim_neonatal(year)
        unm = pl.read_csv(V4 / f"unmatched_{year}.csv", infer_schema_length=0,
                          null_values=["", "NA", "NULL", "None", "null"])
        keys = {(r.get("DTNASC"), r.get("SEXO"), r.get("DTOBITO"),
                 r.get("CAUSABAS"), r.get("CODMUNOCOR")) for r in unm.iter_rows(named=True)}
        df = sim.select([pl.col(c).cast(pl.Utf8) for c in cols if c in sim.columns])
        rows = []
        for pos, r in enumerate(df.iter_rows(named=True)):
            key = (r.get("DTNASC"), r.get("SEXO"), r.get("DTOBITO"),
                   r.get("CAUSABAS"), r.get("CODMUNOCOR"))
            rows.append({
                "year": year, "pos": pos,
                "unlinked_key": int(key in keys),
                "unlinked_id": int(pos not in linked[year]),
                "region": REGION.get((r.get("CODMUNOCOR") or " ")[:1], "unknown"),
                "peso_stratum": peso_stratum(r.get("PESO")),
                "racacor": r.get("RACACOR") or "missing",
                "escmae": r.get("ESCMAE") or "missing",
                "lococor": r.get("LOCOCOR") or "missing",
            })
        frames.append(pl.DataFrame(rows))
    return pl.concat(frames)


def profile(df, ycol):
    out = []
    for var in ["region", "peso_stratum", "racacor", "escmae", "lococor", "year"]:
        g = df.group_by(var).agg(pl.len().alias("n"), pl.col(ycol).sum().alias("k"),
                                 pl.col(ycol).mean().alias("rate"))
        for cat, n, k, rate in g.iter_rows():
            out.append({"variable": var, "category": str(cat), "n": n,
                        "unlinked": k, "unlinked_rate": rate})
    return out


def logit(df, ycol):
    """Identical to analyze_unlinked_bias.main()'s model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    X_cats = df.select(["region", "peso_stratum", "racacor", "lococor"]).to_numpy()
    year_num = (df["year"].to_numpy().astype(float) - 2014.0).reshape(-1, 1)
    enc = OneHotEncoder(drop="first", sparse_output=False)
    X = np.hstack([enc.fit_transform(X_cats), year_num])
    y = df[ycol].to_numpy()
    lr = LogisticRegression(max_iter=2000, C=1e6)
    lr.fit(X, y)
    p = lr.predict_proba(X)[:, 1]
    W = p * (1 - p)
    Xd = np.hstack([np.ones((len(X), 1)), X])
    cov = np.linalg.pinv(Xd.T @ (Xd * W[:, None]))
    se = np.sqrt(np.diag(cov))[1:]
    names = list(enc.get_feature_names_out(
        ["region", "peso_stratum", "racacor", "lococor"])) + ["year_trend"]
    return {n: {"OR": float(np.exp(b)), "lo95": float(np.exp(b - 1.96 * s)),
                "hi95": float(np.exp(b + 1.96 * s))}
            for n, b, s in zip(names, lr.coef_[0], se)}


def main():
    t0 = time.time()
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    linked, checks = rerun_identity(model)
    t_rerun = time.time() - t0
    df = build_frame(linked)
    df.select(["year", "pos", "unlinked_key", "unlinked_id"]).write_csv(
        REV / "taskD_unlinked_flags_by_row.csv")

    n_key, n_id = int(df["unlinked_key"].sum()), int(df["unlinked_id"].sum())
    both = int((df["unlinked_key"] & df["unlinked_id"]).sum())
    print(f"deaths {len(df):,} | key-flagged {n_key:,} | identity-flagged {n_id:,} | both {both:,}")
    prof_old, prof_new = profile(df, "unlinked_key"), profile(df, "unlinked_id")
    lo, ln = logit(df, "unlinked_key"), logit(df, "unlinked_id")
    pl.DataFrame(prof_new).write_csv(REV / "taskD_bias_profile_identity.csv")
    pl.DataFrame(prof_old).write_csv(REV / "taskD_bias_profile_key_recomputed.csv")
    pl.DataFrame([{"term": k, **v} for k, v in ln.items()]).write_csv(REV / "taskD_bias_logit_identity.csv")
    pl.DataFrame([{"term": k, **v} for k, v in lo.items()]).write_csv(REV / "taskD_bias_logit_key_recomputed.csv")

    summary = {
        "deaths": len(df), "flagged_key": n_key, "flagged_identity": n_id,
        "flagged_both": both, "key_only": n_key - both, "identity_only": n_id - both,
        "pooled_rate_key": n_key / len(df), "pooled_rate_identity": n_id / len(df),
        "checks": checks, "seconds_rerun": t_rerun, "seconds_total": time.time() - t0,
    }
    (REV / "taskD_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
