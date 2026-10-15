"""
sensitivity of linkage yield and composition to predefined decisions, in the real files.

Input: outputs/revision/mim_final/death_linkage_{E,C02,C03,C06}.parquet (from linkage_final.py; same 260,219 deaths).
Configurations: E (exact passes only), C02/C03/C06 (full procedure at tau 0.2/0.3/0.6); C03 is the reference.
Nothing is retrained or recalibrated; this does not estimate accuracy.

Outputs (outputs/revision/mim_final/): sensitivity_overall.csv, sensitivity_strata.csv,
sensitivity_transitions.csv, composition_C03.csv.
"""
from __future__ import annotations

import polars as pl

import rv_common  # noqa: F401
from rv_common import REV

OUT = REV / "mim_final"
CFG = ["E", "C02", "C03", "C06"]
REGION = {"1": "North", "2": "Northeast", "3": "Southeast", "4": "South", "5": "Center-West"}
PLACE = {"1": "Hospital", "2": "Other health facility", "3": "Home", "4": "Public road", "5": "Other",
         "6": "Indigenous village", "9": "Ignored"}


def prep(df: pl.DataFrame) -> pl.DataFrame:
    w = pl.col("PESO").cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)
    return df.with_columns([
        pl.when(pl.col("PESO").is_null() | (pl.col("PESO").cast(pl.Utf8).str.strip_chars() == "")).then(pl.lit("missing"))
        .when(w.is_null() | (w < 200) | (w > 8000)).then(pl.lit("invalid"))
        .otherwise(pl.lit("valid")).alias("weight_status"),
        pl.col("CODMUNOCOR").cast(pl.Utf8).str.slice(0, 1).replace_strict(REGION, default="Unknown").alias("region_occurrence"),
        pl.col("LOCOCOR").cast(pl.Utf8).str.split(".").list.first().replace_strict(PLACE, default="Missing").alias("place"),
        pl.col("death_file").cast(pl.Utf8).alias("sim_file_year"),
    ])


def main():
    d = {c: prep(pl.read_parquet(OUT / f"death_linkage_{c}.parquet")) for c in CFG}
    ref = d["C03"].sort(["death_file", "death_row"])
    N = ref.height
    for c in CFG:
        assert d[c].height == N
    key = lambda df: df.sort(["death_file", "death_row"])
    ref_ids = list(zip(ref["death_file"].to_list(), ref["death_row"].to_list()))
    over, trans = [], []
    for c in CFG:
        x = key(d[c])
        assert list(zip(x["death_file"].to_list(), x["death_row"].to_list())) == ref_ids, "death identities differ"
        L = int(x["linked"].sum())
        st = dict(x.filter(pl.col("linked")).group_by("stage").len().iter_rows())
        a, b = ref["linked"].to_numpy(), x["linked"].to_numpy()
        same_b = (ref["birth_file"].to_numpy() == x["birth_file"].to_numpy()) & (ref["birth_row"].to_numpy() == x["birth_row"].to_numpy())
        both = a & b
        n_same = int((both & same_b).sum())
        n_swap = int((both & ~same_b).sum())
        n_gain = int((~a & b).sum())
        n_loss = int((a & ~b).sum())
        n_00 = int((~a & ~b).sum())
        assert n_00 + n_gain + n_loss + n_same + n_swap == N
        over.append({"config": c, "N": N, "L": L, "U": N - L, "pct_linked": 100 * L / N,
                     "diff_pp_vs_C03": 100 * (L - int(ref["linked"].sum())) / N,
                     "P1": st.get("P1", 0), "P2": st.get("P2", 0), "P3": st.get("P3", 0), "CLF": st.get("CLF", 0)})
        trans.append({"config": c, "unlinked_in_both": n_00, "gains": n_gain, "losses": n_loss, "same_birth": n_same,
                      "different_birth": n_swap, "status_change_pct_of_N": 100 * (n_gain + n_loss) / N,
                      "birth_change_pct_of_both_linked": 100 * n_swap / max(n_same + n_swap, 1),
                      "any_change_pct_of_N": 100 * (n_gain + n_loss + n_swap) / N,
                      "check_L_diff": (int(x["linked"].sum()) - int(ref["linked"].sum())) == n_gain - n_loss})
    dec = []
    ws = ref["weight_status"].to_numpy()
    for c in CFG:
        x = key(d[c])
        a_, b_ = ref["linked"].to_numpy(), x["linked"].to_numpy()
        for kind, mask in (("gain_vs_C03", ~a_ & b_), ("loss_vs_C03", a_ & ~b_)):
            for w_ in ("missing", "invalid", "valid"):
                dec.append({"config": c, "change": kind, "weight_status": w_, "n": int((mask & (ws == w_)).sum()),
                            "n_total": int(mask.sum())})
    # recovery by the classifier stage among deaths left unlinked by exact passes only (E)
    e_l = key(d["E"])["linked"].to_numpy(); c3_l = ref["linked"].to_numpy()
    rec = []
    for w_ in ("missing", "invalid", "valid"):
        base = (~e_l) & (ws == w_)
        rec.append({"weight_status": w_, "unlinked_exact_only": int(base.sum()),
                    "linked_by_classifier_C03": int((base & c3_l).sum()),
                    "pct_recovered": 100 * int((base & c3_l).sum()) / max(int(base.sum()), 1)})
    pl.DataFrame(dec).write_csv(OUT / "sensitivity_changes_by_weight.csv")
    pl.DataFrame(rec).write_csv(OUT / "classifier_recovery_by_weight.csv")
    print(pl.DataFrame(rec))
    pl.DataFrame(over).write_csv(OUT / "sensitivity_overall.csv")
    pl.DataFrame(trans).write_csv(OUT / "sensitivity_transitions.csv")

    strata_rows, year_rows = [], []
    for c in CFG:
        x = d[c]
        for col, label in [("weight_status", "weight"), ("region_occurrence", "region_of_occurrence"),
                           ("place", "place_of_death"), ("sim_file_year", "year")]:
            g = x.group_by(col).agg(pl.len().alias("N_g"), pl.col("linked").sum().alias("L_g")).with_columns(
                (pl.col("N_g") - pl.col("L_g")).alias("U_g"))
            Lc, Uc = int(x["linked"].sum()), N - int(x["linked"].sum())
            g = g.with_columns([(100 * pl.col("U_g") / pl.col("N_g")).alias("pct_unlinked_in_stratum"),
                                (100 * pl.col("L_g") / Lc).alias("pct_of_all_linked"),
                                (100 * pl.col("U_g") / Uc).alias("pct_of_all_unlinked"),
                                pl.lit(c).alias("config"), pl.lit(label).alias("variable"),
                                pl.col(col).alias("category")]).drop(col).sort("category")
            strata_rows.append(g)
    st = pl.concat(strata_rows).select(["config", "variable", "category", "N_g", "L_g", "U_g",
                                         "pct_unlinked_in_stratum", "pct_of_all_linked", "pct_of_all_unlinked"])
    st.write_csv(OUT / "sensitivity_strata.csv")
    st.filter(pl.col("config") == "C03").write_csv(OUT / "composition_C03.csv")
    print(pl.DataFrame(over)); print(pl.DataFrame(trans))
    print(st.filter((pl.col("config").is_in(["C03"]))).to_pandas().to_string())

if __name__ == "__main__":
    main()
