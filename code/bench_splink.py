"""
S6: Fellegi-Sunter probabilistic linkage via splink (DuckDB).

Standard FS setup on the shared de-identified fields: blocking on
(DTNASC+SEXO+municipality) OR (DTNASC+PESO); comparisons SEXO, municipality,
PESO (exact / +/-10 g / +/-100 g), IDADEMAE (exact / +/-1); u estimated by
random sampling, m by EM. Pairs above the probability threshold are reduced
to 1:1 by descending match probability (greedy), as commonly practised.

Usage: python3 bench_splink.py <year> [tier] [threshold]
Appends its row to outputs/benchmark/benchmark_<year>_<tier>.json.
"""

from __future__ import annotations

import json
import sys

import pandas as pd
import polars as pl

from common import OUTPUTS, load_sinasc
from strategies_bench import match_metrics
from v4ml import load_synth


def run(year: int, tier: str = "calibrated", threshold: float = 0.9):
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
    linker.training.estimate_u_using_random_sampling(max_pairs=2_000_000)
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("DTNASC", "SEXO", "MUN"))
    preds = linker.inference.predict(
        threshold_match_probability=threshold).as_pandas_dataframe()

    # greedy 1:1 by descending probability; splink labels sides _l/_r
    preds = preds.sort_values("match_probability", ascending=False)
    used_rows: set[int] = set()
    matches: dict[int, tuple[int, int]] = {}
    # side with the smaller unique_id space is the SIM frame (first in list)
    for _, row in preds.iterrows():
        a, b = int(row["unique_id_l"]), int(row["unique_id_r"])
        src_l = str(row.get("source_dataset_l", ""))
        sim_id, sn_row = (a, b) if "_1" in src_l or src_l.endswith("_left") else (a, b)
        # identify by source dataset name when present
        if "source_dataset_l" in row.index:
            if "__splink__input_table_0" in str(row["source_dataset_l"]):
                sim_id, sn_row = a, b
            else:
                sim_id, sn_row = b, a
        if sim_id in matches or sn_row in used_rows:
            continue
        matches[sim_id] = (sn_row, 9)
        used_rows.add(sn_row)

    res = match_metrics(matches, sim_rows, true_row, no_pair,
                        f"S6 Fellegi-Sunter (splink, p>={threshold})")
    print(json.dumps(res, indent=2))

    path = OUTPUTS / "benchmark" / f"benchmark_{year}_{tier}.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    rows = [r for r in rows if not r["strategy"].startswith("S6 ")]
    rows.append(res)
    path.write_text(json.dumps(rows, indent=2))
    print(f"Appended to {path}")


if __name__ == "__main__":
    run(int(sys.argv[1]),
        sys.argv[2] if len(sys.argv) > 2 else "calibrated",
        float(sys.argv[3]) if len(sys.argv) > 3 else 0.9)
