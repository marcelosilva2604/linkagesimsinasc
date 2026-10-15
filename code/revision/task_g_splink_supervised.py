"""
Revision Task G - Fellegi-Sunter (splink) with m probabilities estimated from the TRUE pairs of the 2014
synthetic reference (the same data on which the XGBoost classifier was trained), then applied unchanged to the
2019 and 2024 references. This removes the objection that the comparator was fitted without supervision while the
classifier was fitted with it.

Variants (all with the previous-December births, i.e. the same index as the pipeline):
  sup_xyear : SEXO, municipality (naturalidade or occurrence, single field), weight (<=10, <=100), mother's age (+/-1)
  sup_full  : sup_xyear + facility code, two municipality fields, date tolerance (+/-1, digit swap), weight rounding
u probabilities are estimated by random sampling on the training year (seed 42); m from the label column.
Output: outputs/revision/taskG_splink_<variant>_<year>_calibrated_seed42.csv
Usage: python3 task_g_splink_supervised.py <variant> <year> [year ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import rv_common  # noqa: F401
from rv_common import REV, metrics
from v4ml import assign
from task_e_splink_equal_info import build_frames, THRESHOLDS, greedy_by_prob

TRAIN_YEAR = 2014


def settings_for(variant, SettingsCreator, cl, cll):
    def abs_diff(col, ths):
        lv = [cll.NullLevel(col), cll.ExactMatchLevel(col)]
        for t in ths:
            lv.append(cll.CustomLevel(f"abs({col}_l - {col}_r) <= {t}", f"abs diff <= {t}"))
        lv.append(cll.ElseLevel())
        return cl.CustomComparison(output_column_name=col, comparison_levels=lv)

    if variant == "sup_xyear":
        comps = [cl.ExactMatch("SEXO"), cl.ExactMatch("MUNA"), abs_diff("PESO", [10, 100]), abs_diff("IDADEMAE", [1])]
        rules = ["l.DTNASC = r.DTNASC AND l.SEXO = r.SEXO AND l.MUNA = r.MUNA",
                 "l.DTNASC = r.DTNASC AND l.PESO = r.PESO"]
    else:
        mun = cl.CustomComparison(output_column_name="MUN", comparison_levels=[
            cll.CustomLevel("MUNA_l IS NULL AND MUNB_l IS NULL OR MUNA_r IS NULL AND MUNB_r IS NULL", "municipality missing"),
            cll.CustomLevel("MUNA_l = MUNA_r", "birth municipality = naturalidade"),
            cll.CustomLevel("MUNB_l = MUNB_r", "birth municipality = occurrence"),
            cll.ElseLevel()])
        date = cl.CustomComparison(output_column_name="DTNASC", comparison_levels=[
            cll.NullLevel("DTNASC"), cll.ExactMatchLevel("DTNASC"),
            cll.CustomLevel("abs(DTNUM_l - DTNUM_r) = 1", "date +/-1 day"),
            cll.CustomLevel("DTNASC_l = DTSWAP_r OR DTSWAP_l = DTNASC_r", "day digits transposed"),
            cll.ElseLevel()])
        weight = cl.CustomComparison(output_column_name="PESO", comparison_levels=[
            cll.NullLevel("PESO"), cll.ExactMatchLevel("PESO"),
            cll.CustomLevel("abs(PESO_l - PESO_r) <= 10", "weight <= 10 g"),
            cll.CustomLevel("PESO_l = PESO50_r OR PESO50_l = PESO_r OR PESO_l = PESO100_r OR PESO100_l = PESO_r",
                            "weight rounded to 50 or 100 g"),
            cll.ElseLevel()])
        comps = [cl.ExactMatch("SEXO"), mun, cl.ExactMatch("CODESTAB"), date, weight, abs_diff("IDADEMAE", [1])]
        rules = ["l.DTNASC = r.DTNASC AND l.SEXO = r.SEXO AND l.MUNA = r.MUNA",
                 "l.DTNASC = r.DTNASC AND l.SEXO = r.SEXO AND l.MUNB = r.MUNB",
                 "l.DTNASC = r.DTNASC AND l.CODESTAB = r.CODESTAB AND l.PESO = r.PESO",
                 "l.DTNASC = r.DTNASC AND l.PESO = r.PESO",
                 "abs(l.DTNUM - r.DTNUM) = 1 AND l.SEXO = r.SEXO AND (l.MUNA = r.MUNA OR l.MUNB = r.MUNB)"]
    return SettingsCreator(link_type="link_only", blocking_rules_to_generate_predictions=rules,
                           comparisons=comps, retain_intermediate_calculation_columns=False)


def add_cluster(dfs, dfn, true_row, no_pair, sim_rows):
    cl_sim, cl_sn = [], [f"sn_{int(r)}" for r in dfn["unique_id"]]
    for i, r in enumerate(sim_rows):
        sid = int(r["SYNTH_ID"])
        if no_pair.get(sid, 1) == 0 and true_row.get(sid) is not None:
            cl_sim.append(f"sn_{int(true_row[sid])}")
        else:
            cl_sim.append(f"sim_only_{i}")
    dfs = dfs.copy(); dfn = dfn.copy()
    dfs["cluster"] = cl_sim; dfn["cluster"] = cl_sn
    return dfs, dfn


def train_model(variant, model_path):
    from splink import DuckDBAPI, Linker, SettingsCreator
    import splink.comparison_library as cl
    import splink.comparison_level_library as cll
    dfs, dfn, sim_rows, true_row, no_pair = build_frames(TRAIN_YEAR, "calibrated", "full" if variant == "sup_full" else "xyear")
    dfs, dfn = add_cluster(dfs, dfn, true_row, no_pair, sim_rows)
    linker = Linker([dfs, dfn], settings_for(variant, SettingsCreator, cl, cll), db_api=DuckDBAPI(),
                    input_table_aliases=["sim", "sinasc"])
    linker.training.estimate_u_using_random_sampling(max_pairs=2_000_000, seed=42)
    linker.training.estimate_m_from_label_column("cluster")
    linker.misc.save_model_to_json(str(model_path), overwrite=True)
    return model_path


def evaluate(variant, year, model_path):
    from splink import DuckDBAPI, Linker
    dfs, dfn, sim_rows, true_row, no_pair = build_frames(year, "calibrated", "full" if variant == "sup_full" else "xyear")
    linker = Linker([dfs, dfn], str(model_path), db_api=DuckDBAPI(), input_table_aliases=["sim", "sinasc"])
    preds = linker.inference.predict(threshold_match_probability=min(THRESHOLDS)).as_pandas_dataframe()
    is_sim_l = preds["source_dataset_l"].astype(str).str.contains("sim")
    sim_id = np.where(is_sim_l, preds["unique_id_l"], preds["unique_id_r"]).astype(np.int64)
    sn_id = np.where(is_sim_l, preds["unique_id_r"], preds["unique_id_l"]).astype(np.int64)
    prob = preds["match_probability"].to_numpy(dtype=float)
    rows = []
    for t in THRESHOLDS:
        m = metrics(greedy_by_prob(sim_id, sn_id, prob, t), sim_rows, true_row, no_pair)
        rows.append({"year": year, "variant": variant, "threshold": t, "assignment": "greedy", **m})
        print(f"{variant} {year} p>={t:<5} P {m['precision']*100:.1f} R {m['recall']*100:.1f} F1 {m['f1']*100:.1f}", flush=True)
    out = REV / f"taskG_splink_{variant}_{year}_calibrated_seed42.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("written", out)


if __name__ == "__main__":
    variant = sys.argv[1]
    years = [int(a) for a in sys.argv[2:]]
    mp = REV / f"taskG_splink_model_{variant}_train{TRAIN_YEAR}.json"
    t0 = time.time()
    if not mp.exists():
        train_model(variant, mp)
        print(f"trained {variant} on {TRAIN_YEAR} in {time.time()-t0:.0f}s", flush=True)
    for y in years:
        evaluate(variant, y, mp)
