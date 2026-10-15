"""
Revision Task B - Fellegi-Sunter (splink) operating points.

Model configuration copied verbatim from ../bench_splink.py (blocking, comparisons,
u by random sampling, m by EM on DTNASC+SEXO+MUN). DEVIATION: bench_splink.py
does not seed the u random sampling, so its model (and the probability levels)
change from run to run; here the sampling is seeded (seed 42 convention) and
an unseeded replicate study quantifies the run-to-run variability.
Predictions are generated ONCE at the lowest threshold; each threshold t then keeps pairs with
match_probability >= t. Greedy 1:1 by descending probability over the kept
pairs is identical to running bench_splink.py at threshold t (greedy order is
by probability, so truncating the list = filtering). Hungarian 1:1 (v4ml.assign,
maximising total probability within connected blocks) is reported as well.

Usage: python3 task_b_splink_sweep.py <year> [tier] [--seed N] [--reps K]
  --reps K: K unseeded replicate fits, greedy metrics only (variability study)
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd
import polars as pl

import rv_common  # noqa: F401
from rv_common import REV, fmt, metrics
from common import load_sinasc
from v4ml import assign, load_synth

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
# supplementary low thresholds (beyond the requested grid) to locate the
# point where FS recall would reach v4.2's
SUPP_THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
N_TIE_DRAWS = 50  # random tie-break orders for greedy (rng seed 42)


def predict(year: int, tier: str, threshold: float, seed=42, model_json=None):
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on
    import splink.comparison_library as cl
    import splink.comparison_level_library as cll

    def abs_diff_comparison(col: str, thresholds: list[int]):
        levels = [cll.NullLevel(col), cll.ExactMatchLevel(col)]
        for t in thresholds:
            levels.append(cll.CustomLevel(
                f"abs({col}_l - {col}_r) <= {t}", f"abs diff <= {t}"))
        levels.append(cll.ElseLevel())
        return cl.CustomComparison(output_column_name=col, comparison_levels=levels)

    sim_rows, true_row, no_pair = load_synth(year, tier)
    sinasc = load_sinasc(year)
    sn = sinasc.select([
        pl.col("DTNASC").cast(pl.Utf8),
        pl.col("SEXO").cast(pl.Utf8),
        pl.col("CODMUNNASC").cast(pl.Utf8).alias("MUN"),
        pl.col("PESO").cast(pl.Int64, strict=False),
        pl.col("IDADEMAE").cast(pl.Int64, strict=False),
    ]).with_row_index("ROW")
    df_sinasc = sn.to_pandas()
    df_sinasc["unique_id"] = df_sinasc["ROW"]
    df_sim = pd.DataFrame([
        {
            "unique_id": i,
            "DTNASC": r.get("DTNASC"),
            "SEXO": r.get("SEXO"),
            "MUN": r.get("CODMUNNATU") or r.get("CODMUNOCOR"),
            "PESO": int(float(r["PESO"])) if r.get("PESO") else None,
            "IDADEMAE": int(float(r["IDADEMAE"])) if r.get("IDADEMAE") else None,
        }
        for i, r in enumerate(sim_rows)
    ])
    df_sim["PESO"] = df_sim["PESO"].astype("Int64")
    df_sim["IDADEMAE"] = df_sim["IDADEMAE"].astype("Int64")

    settings = SettingsCreator(
        link_type="link_only",
        blocking_rules_to_generate_predictions=[
            block_on("DTNASC", "SEXO", "MUN"),
            block_on("DTNASC", "PESO"),
        ],
        comparisons=[
            cl.ExactMatch("SEXO"),
            cl.ExactMatch("MUN"),
            abs_diff_comparison("PESO", [10, 100]),
            abs_diff_comparison("IDADEMAE", [1]),
        ],
        retain_intermediate_calculation_columns=False,
    )
    linker = Linker([df_sim, df_sinasc], settings, db_api=DuckDBAPI())
    linker.training.estimate_u_using_random_sampling(max_pairs=2_000_000, seed=seed)
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("DTNASC", "SEXO", "MUN"))
    preds = linker.inference.predict(
        threshold_match_probability=threshold).as_pandas_dataframe()
    if model_json is not None:
        linker.misc.save_model_to_json(str(model_json), overwrite=True)

    # orientation exactly as bench_splink.py
    out = []
    for _, row in preds.iterrows():
        a, b = int(row["unique_id_l"]), int(row["unique_id_r"])
        sim_id, sn_row = a, b
        if "source_dataset_l" in row.index:
            if "__splink__input_table_0" in str(row["source_dataset_l"]):
                sim_id, sn_row = a, b
            else:
                sim_id, sn_row = b, a
        out.append((sim_id, sn_row, float(row["match_probability"])))
    return out, sim_rows, true_row, no_pair


def greedy(pairs_sorted, t):
    used_rows, matches = set(), {}
    for sim_id, sn_row, p in pairs_sorted:
        if p < t:
            break
        if sim_id in matches or sn_row in used_rows:
            continue
        matches[sim_id] = (sn_row, 9)
        used_rows.add(sn_row)
    return matches


def run(year: int, tier: str = "calibrated", seed=42):
    t0 = time.time()
    preds, sim_rows, true_row, no_pair = predict(
        year, tier, min(SUPP_THRESHOLDS), seed=seed,
        model_json=REV / f"taskB_splink_model_{year}_{tier}_seed{seed}.json")
    t_pred = time.time() - t0
    print(f"{len(preds):,} pairs >= {min(SUPP_THRESHOLDS)} in {t_pred:.0f}s", flush=True)
    import numpy as np
    rng = np.random.default_rng(42)
    tie_keys = [rng.random(len(preds)) for _ in range(N_TIE_DRAWS)]
    n_levels = len({round(p, 12) for _, _, p in preds})
    # stable sort by descending probability (pandas sort in bench_splink is
    # quicksort on probability only; ties among equal probabilities are rare
    # but may be ordered differently - see caveat in REVISION_ANALYSES.md)
    preds_sorted = sorted(preds, key=lambda x: -x[2])
    rows = []
    for t in sorted(SUPP_THRESHOLDS + THRESHOLDS):
        m = metrics(greedy(preds_sorted, t), sim_rows, true_row, no_pair)
        draws = []
        for tk in tie_keys:
            order = sorted(range(len(preds)), key=lambda k: (-preds[k][2], tk[k]))
            draws.append(metrics(greedy([preds[k] for k in order], t),
                                 sim_rows, true_row, no_pair)["wrong"])
        m.update({"threshold": t, "assignment": "greedy", "year": year, "tier": tier,
                  "supplementary": t in SUPP_THRESHOLDS,
                  "wrong_min_random_ties": min(draws), "wrong_max_random_ties": max(draws),
                  "wrong_mean_random_ties": float(np.mean(draws))})
        rows.append(m); print(f"p>={t} greedy   {fmt(m)} wrong over random ties {min(draws)}-{max(draws)}", flush=True)
        kept = [(s, c, None) for s, c, p in preds if p >= t]
        sc = [p for s, c, p in preds if p >= t]
        m = metrics(assign(kept, sc, 0.0), sim_rows, true_row, no_pair)
        m.update({"threshold": t, "assignment": "hungarian", "year": year, "tier": tier,
                  "supplementary": t in SUPP_THRESHOLDS})
        rows.append(m); print(f"p>={t} hungarian {fmt(m)}", flush=True)

    # cross-check: bench_splink-style sort (pandas, descending) at 0.9
    df = pd.DataFrame(preds, columns=["s", "c", "p"]).sort_values("p", ascending=False)
    chk = metrics(greedy(list(df.itertuples(index=False, name=None)), 0.9),
                  sim_rows, true_row, no_pair)
    print(f"cross-check pandas-sorted greedy p>=0.9: {fmt(chk)}")

    elapsed = time.time() - t0
    (REV / f"taskB_splink_sweep_{year}_{tier}_seed{seed}.json").write_text(json.dumps(
        {"year": year, "tier": tier, "seed": seed, "seconds": elapsed, "seconds_predict": t_pred,
         "n_pairs_ge_min_threshold": len(preds), "distinct_probability_levels": n_levels, "rows": rows,
         "crosscheck_pandas_sort_p0.9": chk}, indent=2))
    pl.DataFrame(rows).write_csv(REV / f"taskB_splink_sweep_{year}_{tier}_seed{seed}.csv")
    print(f"done in {elapsed:.0f}s")


def replicates(year: int, tier: str, k: int):
    """k unseeded fits (as bench_splink.py does), greedy 1:1 per threshold."""
    rows = []
    for rep in range(k):
        preds, sim_rows, true_row, no_pair = predict(year, tier, min(THRESHOLDS), seed=None)
        ps = sorted(preds, key=lambda x: -x[2])
        for t in THRESHOLDS:
            m = metrics(greedy(ps, t), sim_rows, true_row, no_pair)
            m.update({"rep": rep, "threshold": t, "year": year, "tier": tier,
                      "n_levels": len({round(p, 12) for _, _, p in preds})})
            rows.append(m)
        print(f"rep {rep}: " + " | ".join(
            f"{r['threshold']}:{r['matched']}/{r['wrong']}" for r in rows[-len(THRESHOLDS):]), flush=True)
    pl.DataFrame(rows).write_csv(REV / f"taskB_splink_unseeded_reps_{year}_{tier}.csv")


if __name__ == "__main__":
    args = sys.argv[1:]
    year = int(args[0])
    tier = args[1] if len(args) > 1 and not args[1].startswith("--") else "calibrated"
    if "--reps" in args:
        replicates(year, tier, int(args[args.index("--reps") + 1]))
    else:
        seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 42
        run(year, tier, seed)
