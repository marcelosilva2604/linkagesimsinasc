"""
Revision Task E - Fellegi-Sunter (splink) with the SAME information as the pipeline.

Why: the Task B comparator (bench_splink.py configuration) indexes only current-year births, compares a
single municipality field (CODMUNNATU or CODMUNOCOR), does not use the facility code, and requires exact
date in blocking. The proposed pipeline has the previous-December births, both SIM municipality fields,
the facility code, and tolerant dates. This script gives splink the same information so that a difference
in F1 is not attributable to information alone.

Design (all fields exist in both tables; link_only, seed 42):
  - SINASC side: current-year births + previous-December births (as v4ml.build_records).
  - Comparisons: SEXO exact; municipality (MUNA = birth municipality in SINASC vs CODMUNNATU in SIM;
    MUNB = birth municipality vs CODMUNOCOR in SIM); facility exact; date (exact, +/-1 day,
    day-digit transposition); weight (exact, <=10 g, rounded to 50 or 100 g); mother's age (exact, +/-1).
  - Blocking rules for predictions: (DTNASC,SEXO,MUNA), (DTNASC,SEXO,MUNB), (DTNASC,CODESTAB,PESO),
    (DTNASC,PESO), and date +/-1 day with SEXO and either municipality.
  - u by random sampling (seed 42); m by EM in two sessions (blocked on DTNASC+SEXO+MUNA, then DTNASC+PESO).
Output: outputs/revision/taskE_splink_equal_info_<year>_<tier>_seed42.csv (threshold, greedy and Hungarian).
No existing file is modified.

Usage: /usr/bin/python3 task_e_splink_equal_info.py <year> [xyear|xyear_estab|full]
"""
from __future__ import annotations

import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import rv_common  # noqa: F401
from rv_common import REV, metrics
from v4ml import assign, build_records, load_synth

THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def day_number(s):
    try:
        return datetime.strptime(s, "%d%m%Y").toordinal() if s else None
    except ValueError:
        return None


def swap_day(s):
    return (s[1] + s[0] + s[2:]) if s and len(s) == 8 else None


def r50(v, b):
    return None if v is None else int(round(v / b) * b)


def to_int(v):
    try:
        return int(float(v)) if v not in (None, "") else None
    except ValueError:
        return None


def build_frames(year, tier, variant="full"):
    records = build_records(year)
    sim_rows, true_row, no_pair = load_synth(year, tier)
    sn = []
    for (rid, dt, sx, mun, est, peso, idm) in records:
        p = to_int(peso)
        sn.append({"unique_id": rid, "DTNASC": dt, "DTNUM": day_number(dt), "DTSWAP": swap_day(dt),
                   "SEXO": sx, "MUNA": mun, "MUNB": mun, "CODESTAB": est,
                   "PESO": p, "PESO50": r50(p, 50), "PESO100": r50(p, 100), "IDADEMAE": to_int(idm)})
    sm = []
    for i, r in enumerate(sim_rows):
        dt = r.get("DTNASC")
        p = to_int(r.get("PESO"))
        sm.append({"unique_id": i, "DTNASC": dt, "DTNUM": day_number(dt), "DTSWAP": swap_day(dt),
                   "SEXO": r.get("SEXO"),
                   "MUNA": (r.get("CODMUNNATU") if variant == "full" else (r.get("CODMUNNATU") or r.get("CODMUNOCOR"))),
                   "MUNB": r.get("CODMUNOCOR"),
                   "CODESTAB": r.get("CODESTAB"), "PESO": p, "PESO50": r50(p, 50), "PESO100": r50(p, 100),
                   "IDADEMAE": to_int(r.get("IDADEMAE"))})
    cols = ["unique_id", "DTNASC", "DTNUM", "DTSWAP", "SEXO", "MUNA", "MUNB", "CODESTAB",
            "PESO", "PESO50", "PESO100", "IDADEMAE"]
    dfn, dfs = pd.DataFrame(sn)[cols], pd.DataFrame(sm)[cols]
    for c in ("DTNUM", "PESO", "PESO50", "PESO100", "IDADEMAE"):
        dfn[c] = dfn[c].astype("Int64"); dfs[c] = dfs[c].astype("Int64")
    return dfs, dfn, sim_rows, true_row, no_pair


def predict(year, tier, seed=42, variant="full"):
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on
    import splink.comparison_library as cl
    import splink.comparison_level_library as cll

    dfs, dfn, sim_rows, true_row, no_pair = build_frames(year, tier, variant)

    mun = cl.CustomComparison(output_column_name="MUN", comparison_levels=[
        cll.CustomLevel("MUNA_l IS NULL AND MUNB_l IS NULL OR MUNA_r IS NULL AND MUNB_r IS NULL",
                        "municipality missing"),
        cll.CustomLevel("MUNA_l = MUNA_r", "birth municipality = naturalidade"),
        cll.CustomLevel("MUNB_l = MUNB_r", "birth municipality = occurrence"),
        cll.ElseLevel()])
    date = cl.CustomComparison(output_column_name="DTNASC", comparison_levels=[
        cll.NullLevel("DTNASC"),
        cll.ExactMatchLevel("DTNASC"),
        cll.CustomLevel("abs(DTNUM_l - DTNUM_r) = 1", "date +/-1 day"),
        cll.CustomLevel("DTNASC_l = DTSWAP_r OR DTSWAP_l = DTNASC_r", "day digits transposed"),
        cll.ElseLevel()])
    weight = cl.CustomComparison(output_column_name="PESO", comparison_levels=[
        cll.NullLevel("PESO"),
        cll.ExactMatchLevel("PESO"),
        cll.CustomLevel("abs(PESO_l - PESO_r) <= 10", "weight <= 10 g"),
        cll.CustomLevel("PESO_l = PESO50_r OR PESO50_l = PESO_r OR PESO_l = PESO100_r OR PESO100_l = PESO_r",
                        "weight rounded to 50 or 100 g"),
        cll.ElseLevel()])
    age = cl.CustomComparison(output_column_name="IDADEMAE", comparison_levels=[
        cll.NullLevel("IDADEMAE"),
        cll.ExactMatchLevel("IDADEMAE"),
        cll.CustomLevel("abs(IDADEMAE_l - IDADEMAE_r) = 1", "age +/-1"),
        cll.ElseLevel()])
    def abs_diff(col, ths):
        lv = [cll.NullLevel(col), cll.ExactMatchLevel(col)]
        for t in ths:
            lv.append(cll.CustomLevel(f"abs({col}_l - {col}_r) <= {t}", f"abs diff <= {t}"))
        lv.append(cll.ElseLevel())
        return cl.CustomComparison(output_column_name=col, comparison_levels=lv)

    if variant in ("xyear", "xyear_estab"):
        comps = [cl.ExactMatch("SEXO"), cl.ExactMatch("MUNA"), abs_diff("PESO", [10, 100]),
                 abs_diff("IDADEMAE", [1])]
        rules = [block_on("DTNASC", "SEXO", "MUNA"), block_on("DTNASC", "PESO")]
        if variant == "xyear_estab":
            comps.append(cl.ExactMatch("CODESTAB"))
            rules.append(block_on("DTNASC", "CODESTAB", "PESO"))
        settings = SettingsCreator(link_type="link_only", blocking_rules_to_generate_predictions=rules,
                                   comparisons=comps, retain_intermediate_calculation_columns=False)
        linker = Linker([dfs, dfn], settings, db_api=DuckDBAPI(), input_table_aliases=["sim", "sinasc"])
        linker.training.estimate_u_using_random_sampling(max_pairs=2_000_000, seed=seed)
        linker.training.estimate_parameters_using_expectation_maximisation(block_on("DTNASC", "SEXO", "MUNA"))
        preds = linker.inference.predict(threshold_match_probability=min(THRESHOLDS)).as_pandas_dataframe()
        is_sim_l = preds["source_dataset_l"].astype(str).str.contains("sim")
        sim_id = np.where(is_sim_l, preds["unique_id_l"], preds["unique_id_r"]).astype(np.int64)
        sn_id = np.where(is_sim_l, preds["unique_id_r"], preds["unique_id_l"]).astype(np.int64)
        return sim_id, sn_id, preds["match_probability"].to_numpy(dtype=float), sim_rows, true_row, no_pair

    settings = SettingsCreator(
        link_type="link_only",
        blocking_rules_to_generate_predictions=[
            "l.DTNASC = r.DTNASC AND l.SEXO = r.SEXO AND l.MUNA = r.MUNA",
            "l.DTNASC = r.DTNASC AND l.SEXO = r.SEXO AND l.MUNB = r.MUNB",
            "l.DTNASC = r.DTNASC AND l.CODESTAB = r.CODESTAB AND l.PESO = r.PESO",
            "l.DTNASC = r.DTNASC AND l.PESO = r.PESO",
            "abs(l.DTNUM - r.DTNUM) = 1 AND l.SEXO = r.SEXO AND (l.MUNA = r.MUNA OR l.MUNB = r.MUNB)",
        ],
        comparisons=[cl.ExactMatch("SEXO"), mun, cl.ExactMatch("CODESTAB"), date, weight, age],
        retain_intermediate_calculation_columns=False,
    )
    linker = Linker([dfs, dfn], settings, db_api=DuckDBAPI(), input_table_aliases=["sim", "sinasc"])
    linker.training.estimate_u_using_random_sampling(max_pairs=2_000_000, seed=seed)
    # prior: deterministic rule with an assumed recall of 0.7 (documented splink practice)
    linker.training.estimate_probability_two_random_records_match(
        ["l.DTNASC = r.DTNASC AND l.SEXO = r.SEXO AND l.MUNA = r.MUNA AND l.PESO = r.PESO"], recall=0.7)
    # three EM sessions so that every comparison is estimated in at least one session
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("DTNASC", "SEXO", "MUNA"))      # trains weight, facility, age
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("DTNASC", "SEXO", "PESO"))      # trains municipality, facility, age
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("SEXO", "MUNA", "PESO"))        # trains date levels, facility, age
    preds = linker.inference.predict(threshold_match_probability=min(THRESHOLDS)).as_pandas_dataframe()
    is_sim_l = preds["source_dataset_l"].astype(str).str.contains("sim")
    sim_id = np.where(is_sim_l, preds["unique_id_l"], preds["unique_id_r"]).astype(np.int64)
    sn_id = np.where(is_sim_l, preds["unique_id_r"], preds["unique_id_l"]).astype(np.int64)
    prob = preds["match_probability"].to_numpy(dtype=float)
    return sim_id, sn_id, prob, sim_rows, true_row, no_pair


def greedy_by_prob(sim_id, sn_id, prob, t):
    order = np.argsort(-prob, kind="stable")
    used, matches = set(), {}
    for k in order:
        if prob[k] < t:
            break
        s, c = int(sim_id[k]), int(sn_id[k])
        if s in matches or c in used:
            continue
        matches[s] = (c, 9)
        used.add(c)
    return matches


def run(year, tier="calibrated", variant="full"):
    t0 = time.time()
    sim_id, sn_id, prob, sim_rows, true_row, no_pair = predict(year, tier, variant=variant)
    print(f"{len(prob):,} pairs >= {min(THRESHOLDS)} in {time.time()-t0:.0f}s", flush=True)
    rows = []
    for t in THRESHOLDS:
        m = metrics(greedy_by_prob(sim_id, sn_id, prob, t), sim_rows, true_row, no_pair)
        rows.append({"year": year, "tier": tier, "variant": variant, "threshold": t, "assignment": "greedy", **m})
        keep = prob >= t
        pairs = [(int(a), int(b), None) for a, b in zip(sim_id[keep], sn_id[keep])]
        hm = assign(pairs, prob[keep], 0.0) if len(pairs) else {}
        m2 = metrics(hm, sim_rows, true_row, no_pair) if hm else None
        if m2:
            rows.append({"year": year, "tier": tier, "variant": variant, "threshold": t, "assignment": "hungarian", **m2})
        g = rows[-2] if m2 else rows[-1]
        print(f"p>={t:<5} greedy P {g['precision']*100:.1f} R {g['recall']*100:.1f} F1 {g['f1']*100:.1f} "
              f"(matched {g['matched']:,}, wrong {g['wrong']:,})", flush=True)
    out = REV / f"taskE_splink_{variant}_{year}_{tier}_seed42.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("written", out)


if __name__ == "__main__":
    y = int(sys.argv[1]); variant = sys.argv[2] if len(sys.argv) > 2 else "full"
    run(y, "calibrated", variant)
