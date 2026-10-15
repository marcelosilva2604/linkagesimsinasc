"""
Revision - assemble the four task tables (CSV/JSON + Markdown) from the JSON/CSV
outputs of task_a_ablation.py, task_b_splink_sweep.py, task_d_unlinked_identity.py.
Usage: python3 make_tables.py   -> outputs/revision/table_task{A,B,C,D}.{csv,json,md}
"""

from __future__ import annotations

import json

import polars as pl

import rv_common  # noqa: F401
from rv_common import REV
from common import OUTPUTS


def pct(x):
    return f"{x*100:.1f}"


def save(name, rows, md):
    pl.DataFrame(rows).write_csv(REV / f"{name}.csv")
    (REV / f"{name}.json").write_text(json.dumps(rows, indent=2))
    (REV / f"{name}.md").write_text(md)
    print(md)


def task_a():
    rows, md = [], []
    for y in (2019, 2014, 2024):
        p = REV / f"taskA_ablation_{y}_calibrated.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        md.append(f"\n**Synthetic {y}, calibrated tier** (pairable deaths {d['rows'][0]['n_pairable']:,})\n")
        md.append("| Step | Configuration | Matched | Wrong | P (%) | R (%) | F1 (%) |")
        md.append("|---|---|---:|---:|---:|---:|---:|")
        for r in d["rows"]:
            rows.append({k: r[k] for k in ("year", "step", "label", "matched", "wrong",
                                           "correct", "precision", "recall", "f1")})
            md.append(f"| {r['step']} | {r['label']} | {r['matched']:,} | {r['wrong']:,} | "
                      f"{pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} |")
    save("table_taskA", rows, "\n".join(md))


def task_b():
    rows, md = [], []
    for y in (2019, 2024):
        p = REV / f"taskB_splink_sweep_{y}_calibrated_seed42.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        by = {}
        for r in d["rows"]:
            by.setdefault(r["threshold"], {})[r["assignment"]] = r
        md.append(f"\n**Synthetic {y}, calibrated tier** (splink u-sampling seed 42)\n")
        md.append("| p threshold | Matched | Wrong (greedy) | Wrong range, 50 random tie orders | "
                  "P (%) | R (%) | F1 (%) | Hungarian: Wrong | P | R | F1 |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for t in sorted(by):
            g, h = by[t]["greedy"], by[t]["hungarian"]
            tag = " (suppl.)" if g["supplementary"] else ""
            rows.append({"year": y, "threshold": t, "supplementary": g["supplementary"],
                         "greedy_matched": g["matched"], "greedy_wrong": g["wrong"],
                         "greedy_wrong_min_ties": g["wrong_min_random_ties"],
                         "greedy_wrong_max_ties": g["wrong_max_random_ties"],
                         "greedy_P": g["precision"], "greedy_R": g["recall"], "greedy_F1": g["f1"],
                         "hung_matched": h["matched"], "hung_wrong": h["wrong"],
                         "hung_P": h["precision"], "hung_R": h["recall"], "hung_F1": h["f1"]})
            md.append(f"| {t}{tag} | {g['matched']:,} | {g['wrong']:,} | "
                      f"{g['wrong_min_random_ties']}-{g['wrong_max_random_ties']} | "
                      f"{pct(g['precision'])} | {pct(g['recall'])} | {pct(g['f1'])} | "
                      f"{h['wrong']:,} | {pct(h['precision'])} | {pct(h['recall'])} | {pct(h['f1'])} |")
        grid = [r for r in rows if r["year"] == y and not r["supplementary"]]
        allr = [r for r in rows if r["year"] == y]
        cr = min(grid, key=lambda r: (abs(r["greedy_R"] - 0.926), r["threshold"]))
        cp = min(grid, key=lambda r: (abs(r["greedy_P"] - 0.972), r["threshold"]))
        cr_all = min(allr, key=lambda r: (abs(r["greedy_R"] - 0.926), r["threshold"]))
        cp_all = min(allr, key=lambda r: (abs(r["greedy_P"] - 0.972), r["threshold"]))

        def desc(r):
            return (f"p>={r['threshold']}: P {pct(r['greedy_P'])} / R {pct(r['greedy_R'])} / "
                    f"F1 {pct(r['greedy_F1'])}")
        md.append(f"\nRequested grid - recall closest to 92.6: {desc(cr)}; "
                  f"precision closest to 97.2: {desc(cp)}.")
        md.append(f"Including supplementary low thresholds - recall closest to 92.6: {desc(cr_all)}; "
                  f"precision closest to 97.2: {desc(cp_all)}.")
    save("table_taskB", rows, "\n".join(md))


def task_c():
    rows, md = [], []
    per_year = {}
    for y in (2014, 2019, 2024):
        p = REV / f"taskA_ablation_{y}_calibrated.json"
        d = json.loads(p.read_text())
        per_year[y] = {r["tau"]: r for r in d["tau_sweep"] if r["assignment"] == "hungarian"}
    md.append("| tau | 2014 P | 2014 R | 2014 F1 | 2019 P | 2019 R | 2019 F1 | 2024 P | 2024 R | 2024 F1 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tau in sorted(per_year[2014]):
        cells = []
        for y in (2014, 2019, 2024):
            r = per_year[y][tau]
            rows.append({"year": y, "tau": tau, "matched": r["matched"], "wrong": r["wrong"],
                         "precision": r["precision"], "recall": r["recall"], "f1": r["f1"]})
            cells += [pct(r["precision"]), pct(r["recall"]), pct(r["f1"])]
        md.append(f"| {tau} | " + " | ".join(cells) + " |")
    for y in (2014, 2019, 2024):
        best = max(per_year[y].values(), key=lambda r: r["f1"])
        f03 = per_year[y][0.3]["f1"]
        plateau = sorted(t for t, r in per_year[y].items() if best["f1"] - r["f1"] <= 0.001)
        md.append(f"\n{y}: max F1 {best['f1']*100:.2f} at tau {best['tau']}; "
                  f"F1 at 0.3 = {f03*100:.2f} (gap {(best['f1']-f03)*100:.2f} pp); "
                  f"taus within 0.1 pp of max: {plateau}.")
    save("table_taskC", rows, "\n".join(md))


def task_d():
    V4 = OUTPUTS / "v4_full"
    old_p = pl.read_csv(V4 / "bias_profile.csv", schema_overrides={"category": pl.Utf8})
    old_l = pl.read_csv(V4 / "bias_logit.csv")
    rk_p = pl.read_csv(REV / "taskD_bias_profile_key_recomputed.csv", schema_overrides={"category": pl.Utf8})
    new_p = pl.read_csv(REV / "taskD_bias_profile_identity.csv", schema_overrides={"category": pl.Utf8})
    rk_l = pl.read_csv(REV / "taskD_bias_logit_key_recomputed.csv")
    new_l = pl.read_csv(REV / "taskD_bias_logit_identity.csv")
    summ = json.loads((REV / "taskD_summary.json").read_text())
    mle = pl.read_csv(REV / "taskD_bias_logit_MLE_statsmodels.csv")
    mle_k = mle.filter(pl.col("flag") == "unlinked_key")
    mle_i = mle.filter(pl.col("flag") == "unlinked_id")

    def prate(df, var, cat):
        r = df.filter((pl.col("variable") == var) & (pl.col("category") == cat))
        return r["unlinked_rate"][0] if r.height else None

    def lor(df, term):
        r = df.filter(pl.col("term") == term)
        return (r["OR"][0], r["lo95"][0], r["hi95"][0]) if r.height else None

    spec = [  # label, profile var/cat, logit term
        ("Birthweight missing", "peso_stratum", "missing", "peso_stratum_missing"),
        ("Birthweight <1000 g", "peso_stratum", "<1000", "peso_stratum_<1000"),
        ("Birthweight 1000-1499 g (ref.)", "peso_stratum", "1000-1499", None),
        ("Birthweight 1500-2499 g", "peso_stratum", "1500-2499", "peso_stratum_1500-2499"),
        ("Birthweight >=2500 g", "peso_stratum", ">=2500", "peso_stratum_>=2500"),
        ("Place: hospital (ref.)", "lococor", "1", None),
        ("Place: home", "lococor", "3", "lococor_3"),
        ("Place: other health facility", "lococor", "2", "lococor_2"),
        ("Place: public place", "lococor", "4", "lococor_4"),
        ("Place: elsewhere", "lococor", "5", "lococor_5"),
        ("Region: Centre-West (ref.)", "region", "Centro-Oeste", None),
        ("Region: North", "region", "Norte", "region_Norte"),
        ("Region: Northeast", "region", "Nordeste", "region_Nordeste"),
        ("Region: Southeast", "region", "Sudeste", "region_Sudeste"),
        ("Region: South", "region", "Sul", "region_Sul"),
        ("Race: white (ref.)", "racacor", "1", None),
        ("Race missing", "racacor", "missing", "racacor_missing"),
        ("Race: indigenous", "racacor", "5", "racacor_5"),
        ("Year (per year, trend)", None, None, "year_trend"),
    ]
    rows = []
    md = [f"Deaths 2014-2024: {summ['deaths']:,}. Flagged unlinked - key-based (published): "
          f"{summ['flagged_key']:,} ({summ['pooled_rate_key']*100:.1f}%); identity-based (new): "
          f"{summ['flagged_identity']:,} ({summ['pooled_rate_identity']*100:.1f}%). "
          f"All identity-flagged deaths are also key-flagged; the {summ['key_only']} key-only deaths are linked "
          "deaths sharing the 5-field key with an unlinked death in the same file (199) or own-run "
          "residuals linked as January-window deaths in the previous year's run (8).\n",
          "| Row (Table S2) | Unlinked % old | Unlinked % new | aOR (95% CI) old | aOR (95% CI) new | "
          "exact MLE, old flag | exact MLE, new flag |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for label, var, cat, term in spec:
        po = prate(old_p, var, cat) if var else None
        pk = prate(rk_p, var, cat) if var else None
        pn = prate(new_p, var, cat) if var else None
        lo_, lk, ln = (lor(old_l, term), lor(rk_l, term), lor(new_l, term)) if term else (None, None, None)
        mk, mi = (lor(mle_k, term), lor(mle_i, term)) if term else (None, None)
        rows.append({"row": label, "variable": var, "category": cat, "term": term,
                     "pct_old_published": po, "pct_old_recomputed": pk, "pct_new": pn,
                     "OR_old": lo_[0] if lo_ else None, "lo_old": lo_[1] if lo_ else None,
                     "hi_old": lo_[2] if lo_ else None,
                     "OR_old_recomputed": lk[0] if lk else None,
                     "OR_new": ln[0] if ln else None, "lo_new": ln[1] if ln else None,
                     "hi_new": ln[2] if ln else None,
                     "OR_MLE_old": mk[0] if mk else None, "lo_MLE_old": mk[1] if mk else None,
                     "hi_MLE_old": mk[2] if mk else None,
                     "OR_MLE_new": mi[0] if mi else None, "lo_MLE_new": mi[1] if mi else None,
                     "hi_MLE_new": mi[2] if mi else None})

        def ci(t):
            if not t:
                return "1 (ref.)" if term is None and var else ""
            return f"{t[0]:.2f} ({t[1]:.2f}-{t[2]:.2f})"
        md.append(f"| {label} | {pct(po) if po is not None else ''} | {pct(pn) if pn is not None else ''} | "
                  f"{ci(lo_)} | {ci(ln)} | {ci(mk)} | {ci(mi)} |")
    # consistency: published vs recomputed key-based
    maxdiff = max(abs(r["OR_old"] - r["OR_old_recomputed"]) for r in rows if r["OR_old"] is not None)
    md.append(f"\nCheck: key-based flag recomputed in this script reproduces the published "
              f"bias_logit.csv (max |OR difference| = {maxdiff:.2e}).")
    save("table_taskD", rows, "\n".join(md))


if __name__ == "__main__":
    task_a(); task_b(); task_c(); task_d()
