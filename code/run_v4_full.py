"""
Phase 4: full national run of v4.2 on real data, 2014-2024.

Per year (mirroring v3.2's data assembly): deaths = SIM neonatal of the year
+ January-window deaths of the following year born in the year; index = the
year's SINASC + December of the previous year (cross-year fix). Pipeline =
exact P1-P3 + tolerant candidates + XGBoost scores + Hungarian 1:1 at tau.

Outputs under outputs/v4_full/:
  SINASC_SIM_v4_<year>.csv   full SINASC of the year + MORTE, DTOBITO,
                             IDADE_DIAS, CAUSABAS, MATCH_PASS, MATCH_SCORE
                             (December-of-previous-year matches are reported
                             in stats but flagged, not written into the
                             current-year output rows)
  unmatched_<year>.csv       unmatched SIM records (bias analysis, Phase 5)
  stats_<year>.json          per-pass counts, match rate, ambiguity, NMR

Usage: python3 run_v4_full.py [year ...]   (default 2014-2024)
"""

from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl
import xgboost as xgb

from common import OUTPUTS, load_sim_neonatal, load_sinasc
from strategies import prepare_sim_rows
from strategies_v4 import PREV_OFFSET
from v4ml import MODEL_PATH, assign, build_records, candidate_pairs, exact_p13

TAU = 0.3
OUT = OUTPUTS / "v4_full"


def run_year(year: int, model, prev_jan_matched: set) -> dict:
    print(f"\n=== {year} ===", flush=True)
    sim_atual = load_sim_neonatal(year)
    sim_next = load_sim_neonatal(year + 1)
    sim_rows = prepare_sim_rows(sim_atual)
    n_atual = len(sim_rows)
    # identity = row position within the source SIM file (CONTADOR is not
    # unique in the 2014-2017 files); jan-window rows carry their row index
    # in the NEXT year's file so they can be reconciled with that year's run
    jan_row_ids: list[int] = []
    n_jan = 0
    if sim_next is not None:
        sim_jan = sim_next.with_row_index("RIDX").filter(
            pl.col("DTNASC").str.slice(4, 4) == str(year))
        n_jan = len(sim_jan)
        jan_row_ids = sim_jan["RIDX"].to_list()
        sim_rows += prepare_sim_rows(sim_jan.drop("RIDX"))
    total = len(sim_rows)

    records = build_records(year)
    matches, used, indexes = exact_p13(records, sim_rows)
    pairs = candidate_pairs(records, sim_rows, matches, used, indexes)
    n_ambig_deaths = len({i for i, _, _ in pairs})
    if pairs:
        scores_arr = model.predict_proba(np.array([f for _, _, f in pairs]))[:, 1]
        ml = assign(pairs, scores_arr, TAU)
        score_of = {}
        for k, (i, c, _) in enumerate(pairs):
            if i in ml and ml[i][0] == c:
                score_of[i] = float(scores_arr[k])
        matches.update(ml)
    else:
        score_of = {}

    pass_counts: dict[int, int] = {}
    prev_dec_matches = 0
    for i, (j, p) in matches.items():
        pass_counts[p] = pass_counts.get(p, 0) + 1
        if j >= PREV_OFFSET:
            prev_dec_matches += 1

    # unique-death accounting: each SIM file record counted once nationally.
    # Own-file rows matched in this run, plus this file's records already
    # matched as the January window of the previous year's run; overlap
    # (matched in both runs) counted once and reported.
    own_matched = {i for i in matches if i < n_atual}
    jan_matched = {int(jan_row_ids[i - n_atual]) for i in matches if i >= n_atual}
    overlap = len(own_matched & prev_jan_matched)
    unique_matched = len(own_matched | prev_jan_matched)

    # --- write the linked base (current-year births only) ----------------
    sinasc_out = load_sinasc(year)
    n_sinasc = len(sinasc_out)
    morte = [0] * n_sinasc
    dtobito = [None] * n_sinasc
    idade_dias = [None] * n_sinasc
    causabas = [None] * n_sinasc
    match_pass = [None] * n_sinasc
    match_score = [None] * n_sinasc
    for i, (j, p) in matches.items():
        if j >= PREV_OFFSET:
            continue
        r = sim_rows[i]
        morte[j] = 1
        dtobito[j] = r.get("DTOBITO")
        idade_dias[j] = r.get("IDADE_DIAS")
        causabas[j] = r.get("CAUSABAS")
        match_pass[j] = p
        match_score[j] = score_of.get(i)

    sinasc_out = sinasc_out.with_columns([
        pl.Series("MORTE", morte),
        pl.Series("DTOBITO", dtobito, dtype=pl.Utf8),
        pl.Series("IDADE_DIAS", idade_dias, dtype=pl.Utf8),
        pl.Series("CAUSABAS", causabas, dtype=pl.Utf8),
        pl.Series("MATCH_PASS", match_pass, dtype=pl.Int8),
        pl.Series("MATCH_SCORE", match_score, dtype=pl.Float64),
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    sinasc_out.write_csv(OUT / f"SINASC_SIM_v4_{year}.csv")

    unmatched = [sim_rows[i] for i in range(total) if i not in matches]
    if unmatched:
        pl.DataFrame(unmatched).write_csv(OUT / f"unmatched_{year}.csv")

    linked = sum(morte)
    stats = {
        "year": year,
        "births": n_sinasc,
        "deaths_total": total,
        "deaths_file": n_atual,
        "deaths_jan_next": n_jan,
        "matched": len(matches),
        "match_rate": len(matches) / total,
        "unique_matched_file": unique_matched,
        "unique_match_rate_file": unique_matched / n_atual,
        "double_matched_across_runs": overlap,
        "jan_matched_next_file": len(jan_matched),
        "prev_dec_matches": prev_dec_matches,
        "ambiguous_deaths_scored": n_ambig_deaths,
        "pass_counts": pass_counts,
        "linked_current_year": linked,
        "nmr_per_1000": linked / n_sinasc * 1000,
        "tau": TAU,
    }
    with open(OUT / f"stats_{year}.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)
    return stats, jan_matched


def main():
    years = [int(a) for a in sys.argv[1:]] or list(range(2014, 2025))
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    all_stats = []
    prev_jan: set = set()
    for y in years:
        st, prev_jan = run_year(y, model, prev_jan)
        all_stats.append(st)
    with open(OUT / "stats_all.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n{'Year':<6}{'Births':>11}{'FileDeaths':>11}{'UniqMatch':>10}{'Rate%':>7}{'Dbl':>5}{'NMR':>7}")
    tot_d = tot_m = 0
    for s in all_stats:
        tot_d += s['deaths_file']; tot_m += s['unique_matched_file']
        print(f"{s['year']:<6}{s['births']:>11,}{s['deaths_file']:>11,}"
              f"{s['unique_matched_file']:>10,}{s['unique_match_rate_file']*100:>7.2f}"
              f"{s['double_matched_across_runs']:>5}{s['nmr_per_1000']:>7.2f}")
    print(f"TOTAL unique file deaths {tot_d:,} | unique matched {tot_m:,} "
          f"({tot_m/tot_d*100:.2f}%)")


if __name__ == "__main__":
    main()
