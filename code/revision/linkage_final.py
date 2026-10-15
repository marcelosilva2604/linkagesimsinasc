"""
pair-level records with persistent identities and a single consolidated base.

Why: the stored v4_full outputs count deaths linked "in some run" (231,480) and births flagged in the current-year
export (231,211). The two differ by 269 because annual runs overlap (January-window deaths are proposed in the run of
year y and again, against December births, in the run of year y+1). To report one base, every proposed pair is
recorded with an identity on both sides and consolidated by a fixed rule.

Identities (never CONTADOR, which is not unique in the 2014-2017 SIM files):
  death_id = (SIM file year, row position in SIM_neonatal_<year>.csv)
  birth_id = (SINASC file year, row position in SINASC_<year>.csv)
  December births of the previous year are mapped back to their row in the previous year's SINASC file.

Procedure per annual run: exactly run_v4_full.py (same data assembly, same functions, same stored model). The Hungarian
assignment is run once without threshold; every classifier-stage pair keeps its score, so tau can be applied afterwards
without changing which pairs compete (assign() decides the assignment before applying tau).

Consolidation rule (fixed before looking at results):
  1. collapse repeated (death, birth) proposals, keeping every source run;
  2. order by stage (P1, P2, P3, classifier), then classifier score descending, then own-file run before
     January-window run, then older run, then ids;
  3. walk the order and accept a pair only if both its death and its birth are still free.
This is an operational rule, not evidence that the prioritized pair is correct.

Configurations: E (P1-P3 only), C02, C03 (reference), C06 (classifier at tau 0.2, 0.3, 0.6).

Writes only under outputs/revision/mim_final/. Usage: python3 linkage_final.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
import xgboost as xgb

import rv_common  # noqa: F401
from rv_common import REV
from common import load_sim_neonatal, load_sinasc
from strategies import prepare_sim_rows
from strategies_v4 import PREV_OFFSET, sinasc_records
from v4ml import MODEL_PATH, assign, candidate_pairs, exact_p13

YEARS = list(range(2014, 2025))
CONFIGS = {"E": None, "C02": 0.2, "C03": 0.3, "C06": 0.6}
STAGE_RANK = {"P1": 1, "P2": 2, "P3": 3, "CLF": 4}
OUT = REV / "mim_final"
DEATH_COLS = ["PESO", "CODMUNOCOR", "CODMUNRES", "LOCOCOR", "RACACOR", "DTNASC", "DTOBITO", "SEXO",
              "SEMAGESTAC", "ESCMAE", "GRAVIDEZ", "PARTO", "CODESTAB"]


def run_year(year: int, model) -> tuple[pl.DataFrame, int, int, int]:
    sim_atual = load_sim_neonatal(year)
    sim_next = load_sim_neonatal(year + 1) if year + 1 <= YEARS[-1] else None  # universe ends with the 2024 file
    sim_rows = prepare_sim_rows(sim_atual)
    n_atual = len(sim_rows)
    death_file = [year] * n_atual
    death_row = list(range(n_atual))
    if sim_next is not None:
        jan = sim_next.with_row_index("RIDX").filter(pl.col("DTNASC").str.slice(4, 4) == str(year))
        death_file += [year + 1] * len(jan)
        death_row += jan["RIDX"].to_list()
        sim_rows += prepare_sim_rows(jan.drop("RIDX"))

    cur = load_sinasc(year)
    prev = load_sinasc(year - 1)
    records = sinasc_records(cur)
    prev_rows: list[int] = []
    if prev is not None:
        prev_dec = prev.with_row_index("RIDX").filter(pl.col("DTNASC").str.slice(2, 2) == "12")
        prev_rows = prev_dec["RIDX"].to_list()
        records += sinasc_records(prev_dec.drop("RIDX"), offset=PREV_OFFSET)

    def birth_id(j: int) -> tuple[int, int]:
        return (year, j) if j < PREV_OFFSET else (year - 1, prev_rows[j - PREV_OFFSET])

    matches, used, indexes = exact_p13(records, sim_rows)
    pairs = candidate_pairs(records, sim_rows, matches, used, indexes)
    out = []
    for i, (j, pid) in matches.items():
        bf, br = birth_id(j)
        out.append((year, death_file[i], death_row[i], bf, br, f"P{pid}", None))
    n_pairs = len(pairs)
    if pairs:
        sc = model.predict_proba(np.array([f for _, _, f in pairs]))[:, 1]
        score_of = {(i, c): float(sc[k]) for k, (i, c, _) in enumerate(pairs)}
        for i, (c, _) in assign(pairs, sc, 0.0).items():
            bf, br = birth_id(c)
            out.append((year, death_file[i], death_row[i], bf, br, "CLF", score_of[(i, c)]))
    sch = {"run_year": pl.Int32, "death_file": pl.Int32, "death_row": pl.Int32, "birth_file": pl.Int32,
           "birth_row": pl.Int32, "stage": pl.Utf8, "score": pl.Float64}
    df = pl.DataFrame(out, schema=sch, orient="row", infer_schema_length=None)
    return df, len(sim_rows), n_atual, n_pairs


def consolidate(runs: pl.DataFrame, tau: float | None, own_first: bool = False) -> tuple[pl.DataFrame, dict]:
    p = runs.filter((pl.col("stage") != "CLF") | (tau is not None and pl.col("score") >= (tau or 0)))
    if tau is None:
        p = runs.filter(pl.col("stage") != "CLF")
    p = p.with_columns([
        pl.col("stage").replace_strict(STAGE_RANK, return_dtype=pl.Int8).alias("srank"),
        (pl.col("death_file") != pl.col("run_year")).cast(pl.Int8).alias("jan_run"),
    ])
    n_prop = len(p)
    key = ["death_file", "death_row", "birth_file", "birth_row"]
    order = (["jan_run", "srank", "neg_score", "run_year"] if own_first else ["srank", "neg_score", "jan_run", "run_year"]) + key
    p = p.with_columns((-pl.col("score").fill_null(0.0)).alias("neg_score")).sort(order)
    agg = p.group_by(key).agg(pl.len().alias("n_runs"), pl.col("run_year").sort().alias("source_runs").cast(pl.List(pl.Int32)))
    p = p.unique(subset=key, keep="first", maintain_order=True).join(agg, on=key).sort(order)
    n_collapsed = len(p)
    taken_d: set = set()
    taken_b: set = set()
    keep = []
    lost_birth = lost_death = 0
    for r in p.iter_rows(named=True):
        d = (r["death_file"], r["death_row"])
        b = (r["birth_file"], r["birth_row"])
        if d in taken_d:
            lost_death += 1
            keep.append(False)
        elif b in taken_b:
            lost_birth += 1
            keep.append(False)
        else:
            taken_d.add(d)
            taken_b.add(b)
            keep.append(True)
    acc = p.filter(pl.Series(keep))
    audit = {"proposals": n_prop, "after_collapse": n_collapsed, "accepted": len(acc),
             "dropped_death_already_linked": lost_death, "dropped_birth_already_taken": lost_birth,
             "deaths_proposed": p.select(["death_file", "death_row"]).unique().height}
    return acc.drop(["srank", "neg_score", "jan_run"]), audit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    legacy_path = REV.parent / "v4_full" / "stats_all.json"
    if not legacy_path.exists():
        raise SystemExit(f"required comparison input missing: {legacy_path} (copy results/legacy/stats_all.json there)")
    if not MODEL_PATH.exists():
        raise SystemExit(f"frozen classifier missing: {MODEL_PATH} (copy model/xgb_pairs.json there)")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    legacy = {s["year"]: s for s in json.load(open(legacy_path))}
    runs, checks = [], []
    for y in YEARS:
        t = time.time()
        df, n_run, n_file, n_pairs = run_year(y, model)
        runs.append(df)
        m = df.filter((pl.col("stage") != "CLF") | (pl.col("score") >= 0.3)).height
        ok = m == legacy[y]["matched"] and n_file == legacy[y]["deaths_file"]
        checks.append({"year": y, "run_deaths": n_run, "file_deaths": n_file, "matched_C03": m,
                       "legacy_matched": legacy[y]["matched"], "ok": bool(ok), "candidate_pairs": n_pairs})
        print(y, checks[-1], f"{time.time()-t:.0f}s", flush=True)
    runs = pl.concat(runs)
    runs.write_parquet(OUT / "run_pairs.parquet")
    n_diff = [c["year"] for c in checks if not c["ok"]]
    if n_diff:  # inputs differ from the files of the study (the publisher replaces files); results are then not the reported ones
        print("WARNING: per-run counts differ from the recorded counts for run years", n_diff,
              "- the input files may differ from those of the study (see download_inputs.py)", flush=True)

    # all deaths, with attributes for later strata
    parts = []
    for y in YEARS:
        d = load_sim_neonatal(y).with_row_index("death_row")
        keep = ["death_row"] + [c for c in DEATH_COLS if c in d.columns]
        d = d.select(keep).with_columns(pl.lit(y).cast(pl.Int32).alias("death_file"),
                                        pl.col("death_row").cast(pl.Int32))
        d = d.with_columns([pl.col(c).cast(pl.Utf8) for c in DEATH_COLS if c in d.columns])
        parts.append(d)
    deaths = pl.concat(parts, how="diagonal_relaxed")
    N = deaths.height

    flow = {"per_run_check": checks, "N_deaths": N, "legacy": {k: sum(s[k] for s in legacy.values()) for k in
                                     ("deaths_file", "matched", "unique_matched_file", "double_matched_across_runs",
                                      "prev_dec_matches", "linked_current_year", "deaths_jan_next")}, "configs": {}}
    conflict_rows = []
    for name, tau in CONFIGS.items():
        acc, audit = consolidate(runs, tau)
        acc = acc.with_columns(pl.lit(name).alias("config_id"))
        acc.write_parquet(OUT / f"accepted_pairs_{name}.parquet")
        dl = deaths.join(acc.select(["death_file", "death_row", "birth_file", "birth_row", "stage", "score", "n_runs"]),
                         on=["death_file", "death_row"], how="left").with_columns(
            pl.col("birth_row").is_not_null().alias("linked"))
        dl.write_parquet(OUT / f"death_linkage_{name}.parquet")
        L = int(dl["linked"].sum())
        assert acc.height == L, "accepted pairs and linked deaths differ"
        assert acc.select(["birth_file", "birth_row"]).unique().height == acc.height, "birth reused"
        by_stage = dict(acc.group_by("stage").len().iter_rows())
        flow["configs"][name] = {"tau": tau, "L": L, "U": N - L, "pct": 100 * L / N, "by_stage": by_stage, **audit}
        conflict_rows.append({"config": name, **flow["configs"][name], "by_stage": json.dumps(by_stage)})
        print(name, flow["configs"][name], flush=True)

    # alternative consolidation policy (own-file run before stage), reported only; the stage-first rule is the reference
    alt, alt_audit = consolidate(runs, 0.3, own_first=True)
    ref = pl.read_parquet(OUT / "accepted_pairs_C03.parquet")
    ref_d = ref.select(["death_file", "death_row", "birth_file", "birth_row"])
    alt_d = alt.select(["death_file", "death_row", "birth_file", "birth_row"])
    j = ref_d.join(alt_d, on=["death_file", "death_row"], how="full", suffix="_alt", coalesce=True)
    status_changed = j.filter(pl.col("birth_row").is_null() | pl.col("birth_row_alt").is_null()).height
    birth_changed = j.filter(pl.col("birth_row").is_not_null() & pl.col("birth_row_alt").is_not_null() &
                             ((pl.col("birth_row") != pl.col("birth_row_alt")) | (pl.col("birth_file") != pl.col("birth_file_alt")))).height
    flow["conflict_policy_audit_C03"] = {"alt_rule": "own-file run before stage", "L_reference": ref.height,
                                         "L_alternative": alt.height, "status_changes": status_changed,
                                         "birth_changes_among_linked_in_both": birth_changed}
    # overlap audit: deaths proposed in two runs under C03 (complete identities on both sides)
    c03 = runs.filter((pl.col("stage") != "CLF") | (pl.col("score") >= 0.3))
    dd = c03.group_by(["death_file", "death_row"]).agg(
        pl.struct(["birth_file", "birth_row"]).n_unique().alias("n_births"), pl.len().alias("n_props"))
    two = dd.filter(pl.col("n_props") > 1)
    flow["overlap_C03"] = {"deaths_in_two_runs": two.height,
                           "same_birth": two.filter(pl.col("n_births") == 1).height,
                           "different_birth": two.filter(pl.col("n_births") > 1).height}
    bb = c03.group_by(["birth_file", "birth_row"]).agg(pl.struct(["death_file", "death_row"]).n_unique().alias("nd"))
    flow["overlap_C03"]["births_proposed_to_more_than_one_death"] = bb.filter(pl.col("nd") > 1).height
    json.dump(flow, open(OUT / "flow_counts.json", "w"), indent=2, default=str)
    pl.DataFrame(conflict_rows).write_csv(OUT / "conflict_audit.csv")
    pl.DataFrame(checks).write_csv(OUT / "legacy_reconciliation.csv")
    print(json.dumps({k: v for k, v in flow.items() if k != "configs"}, indent=2, default=str))


if __name__ == "__main__":
    main()
