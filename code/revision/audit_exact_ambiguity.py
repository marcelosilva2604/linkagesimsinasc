"""
audit of ambiguity in the exact passes (P1-P3) of the final linkage, run year by year with the same inputs
as linkage_final.run_year. For every death matched in an exact pass it records how many unused birth records shared the
key at the moment of matching (ambiguous if > 1), and re-runs the passes with (a) reversed index order and (b) a
random permutation of the death order (seed 7) to count how many accepted pairs depend on the tie-breaking rule.
Descriptive: it does not say which tie-break choice is correct. Output: outputs/revision/mim_final/exact_ambiguity.csv
"""
from __future__ import annotations

import numpy as np
import polars as pl

import rv_common  # noqa: F401
from rv_common import REV
from common import load_sim_neonatal, load_sinasc
from strategies import prepare_sim_rows
from strategies_v4 import PREV_OFFSET, sinasc_records
from v4ml import build_indexes

OUT = REV / "mim_final"
YEARS = list(range(2014, 2025))


def keys(r):
    g = lambda f: "" if r.get(f) is None else str(r.get(f))  # noqa: E731
    return (f"{g('DTNASC')}_{g('SEXO')}_{g('CODMUNNATU')}_{g('PESO')}", 1, 0), \
           (f"{g('DTNASC')}_{g('SEXO')}_{g('CODMUNOCOR')}_{g('PESO')}", 2, 0), \
           (f"{g('DTNASC')}_{g('CODESTAB')}_{g('PESO')}", 3, 1)


def run(records, sim_rows, order, reverse=False):
    idx_mp, idx_ep, _ = build_indexes(records)
    idxs = [idx_mp, idx_ep]
    if reverse:
        idxs = [{k: list(reversed(v)) for k, v in ix.items()} for ix in idxs]
    used, match, amb = set(), {}, {}
    for pid in (1, 2, 3):
        for i in order:
            if i in match:
                continue
            key, _, which = keys(sim_rows[i])[pid - 1]
            lst = idxs[which].get(key)
            if not lst:
                continue
            free = [j for j in lst if j not in used]
            if free:
                used.add(free[0])
                match[i] = (free[0], pid)
                amb[i] = (len(free), pid)
    return match, amb


def main():
    rows = []
    rng = np.random.default_rng(7)
    flags = []
    for year in YEARS:
        sim_atual = load_sim_neonatal(year)
        sim_next = load_sim_neonatal(year + 1) if year + 1 <= YEARS[-1] else None
        sim_rows = prepare_sim_rows(sim_atual)
        death_file = [year] * len(sim_rows)
        death_row = list(range(len(sim_rows)))
        if sim_next is not None:
            jan = sim_next.with_row_index("RIDX").filter(pl.col("DTNASC").str.slice(4, 4) == str(year))
            death_file += [year + 1] * len(jan)
            death_row += jan["RIDX"].to_list()
            sim_rows += prepare_sim_rows(jan.drop("RIDX"))
        records = sinasc_records(load_sinasc(year))
        prev = load_sinasc(year - 1)
        if prev is not None:
            records += sinasc_records(prev.filter(pl.col("DTNASC").str.slice(2, 2) == "12"), offset=PREV_OFFSET)
        prev_rows = []
        if prev is not None:
            prev_rows = prev.with_row_index("RIDX").filter(pl.col("DTNASC").str.slice(2, 2) == "12")["RIDX"].to_list()
        base, amb = run(records, sim_rows, range(len(sim_rows)))
        rev, _ = run(records, sim_rows, range(len(sim_rows)), reverse=True)
        shuf, _ = run(records, sim_rows, rng.permutation(len(sim_rows)).tolist())
        def bid(j):
            return (year, j) if j < PREV_OFFSET else (year - 1, prev_rows[j - PREV_OFFSET])
        flags += [(year, death_file[i], death_row[i], v[1], v[0],
                   i in rev and rev[i][0] != base[i][0], i not in rev,
                   i in shuf and shuf[i][0] != base[i][0], i not in shuf,
                   bid(base[i][0])[0], bid(base[i][0])[1]) for i, v in amb.items()]
        n = len(base)
        n_amb = sum(1 for v in amb.values() if v[0] > 1)
        for pid in (1, 2, 3):
            tot = sum(1 for v in base.values() if v[1] == pid)
            a = sum(1 for v in amb.values() if v[1] == pid and v[0] > 1)
            rows.append({"run_year": year, "scope": f"P{pid}", "n_matched": tot, "n_ambiguous_at_match": a})
        same_rev = sum(1 for i, v in base.items() if rev.get(i, (None,))[0] == v[0])
        same_shuf = sum(1 for i, v in base.items() if shuf.get(i, (None,))[0] == v[0])
        rows.append({"run_year": year, "scope": "ALL", "n_matched": n, "n_ambiguous_at_match": n_amb,
                     "n_matched_reversed_index": len(rev), "n_same_birth_reversed_index": same_rev,
                     "n_matched_shuffled_deaths": len(shuf), "n_same_birth_shuffled_deaths": same_shuf})
        print(year, n, n_amb, len(rev), same_rev, len(shuf), same_shuf, flush=True)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "exact_ambiguity.csv")
    # record-level flags (local only, not released): joined to the accepted C03 pairs below
    fl = pl.DataFrame(flags, schema={"run_year": pl.Int32, "death_file": pl.Int32, "death_row": pl.Int32,
                                     "pass_id": pl.Int32, "n_free_candidates": pl.Int32,
                                     "birth_changes_reversed_index": pl.Boolean,
                                     "unmatched_reversed_index": pl.Boolean,
                                     "birth_changes_shuffled_deaths": pl.Boolean,
                                     "unmatched_shuffled_deaths": pl.Boolean,
                                     "audit_birth_file": pl.Int32, "audit_birth_row": pl.Int32}, orient="row")
    fl.write_parquet(OUT / "exact_ambiguity_flags.parquet")
    acc = pl.read_parquet(OUT / "accepted_pairs_C03.parquet").filter(pl.col("stage") != "CLF")
    j = acc.join(fl, on=["run_year", "death_file", "death_row"], how="left")
    n = j.height
    n_unmatched = int(j["pass_id"].is_null().sum())
    # the birth chosen by the audit must be the birth accepted in the base for the same death and run
    assert int(((j["audit_birth_file"] != j["birth_file"]) | (j["audit_birth_row"] != j["birth_row"])).sum()) == 0
    amb = j.filter(pl.col("n_free_candidates") > 1)
    print("final exact pairs", n, "not found in audit", n_unmatched)
    print("ambiguous at match", amb.height, round(100 * amb.height / n, 2))
    print("different birth if index reversed", int(j["birth_changes_reversed_index"].sum()), "unmatched",
          int(j["unmatched_reversed_index"].sum()), "; if death order shuffled",
          int(j["birth_changes_shuffled_deaths"].sum()), "unmatched", int(j["unmatched_shuffled_deaths"].sum()))
    tot = 260219
    print("yield if ambiguous exact pairs excluded", 231478 - amb.height, round(100 * (231478 - amb.height) / tot, 2))
    summ = {"final_exact_pairs": n, "not_found_in_audit": n_unmatched, "ambiguous_at_match": amb.height,
            "birth_changes_reversed_index": int(j["birth_changes_reversed_index"].sum()),
            "birth_changes_shuffled_deaths": int(j["birth_changes_shuffled_deaths"].sum()),
            "unmatched_reversed_index": int(j["unmatched_reversed_index"].sum()),
            "unmatched_shuffled_deaths": int(j["unmatched_shuffled_deaths"].sum()),
            "exact_matches_base_runs": int(sum(r["n_matched"] for r in rows if r["scope"] == "ALL")),
            "exact_matches_reversed_runs": int(sum(r["n_matched_reversed_index"] for r in rows if r["scope"] == "ALL")),
            "exact_matches_shuffled_runs": int(sum(r["n_matched_shuffled_deaths"] for r in rows if r["scope"] == "ALL")),
            "by_pass": {f"P{k}": [int((j["pass_id"] == k).sum()), int(((j["pass_id"] == k) & (j["n_free_candidates"] > 1)).sum())] for k in (1, 2, 3)}}
    # multiple pregnancy (GRAVIDEZ 2 or 3 in the SIM record) by stage; percentages over pairs with plurality recorded
    d = pl.read_parquet(OUT / "death_linkage_C03.parquet").select(["death_file", "death_row", "GRAVIDEZ"])
    allacc = pl.read_parquet(OUT / "accepted_pairs_C03.parquet").join(d, on=["death_file", "death_row"], how="left") \
        .join(fl.select(["run_year", "death_file", "death_row", "n_free_candidates"]),
              on=["run_year", "death_file", "death_row"], how="left")
    g = pl.col("GRAVIDEZ").cast(pl.Utf8).str.strip_chars()
    allacc = allacc.with_columns((g.is_not_null() & (g != "")).alias("rec"), g.is_in(["2", "3"]).alias("mult"),
                                 (pl.col("n_free_candidates") > 1).fill_null(False).alias("amb"))
    mrows = []
    for name, sub in [("P1", allacc.filter(pl.col("stage") == "P1")), ("P2", allacc.filter(pl.col("stage") == "P2")),
                      ("P3", allacc.filter(pl.col("stage") == "P3")), ("CLF", allacc.filter(pl.col("stage") == "CLF")),
                      ("exact_ambiguous", allacc.filter(pl.col("amb"))),
                      ("exact_not_ambiguous", allacc.filter((pl.col("stage") != "CLF") & ~pl.col("amb")))]:
        rec = sub.filter(pl.col("rec"))
        mrows.append({"group": name, "n_pairs": sub.height, "n_plurality_recorded": rec.height,
                      "n_multiple": int(rec["mult"].sum()), "pct_multiple_of_recorded": 100 * rec["mult"].mean(),
                      "n_ambiguous_at_match": int(sub["amb"].sum())})
    pl.DataFrame(mrows).write_csv(OUT / "multiple_births_by_stage.csv")
    import json
    json.dump(summ, open(OUT / "exact_ambiguity_final.json", "w"), indent=2)


if __name__ == "__main__":
    main()
