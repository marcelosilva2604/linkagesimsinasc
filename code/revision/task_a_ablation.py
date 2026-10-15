"""
Revision Task A - fine-grained cumulative ablation (+ Task C tau sweep, which
shares the candidate pairs and scores).

Rows:
  0  v3.2: four greedy exact passes, current-year SINASC index (strategies.v32_core)
  1  v3.2 passes 1-3 only (pass 4, mother's age, removed), current-year index
  2  + cross-year index (previous-December births): v4ml.exact_p13
  3  + tolerant unique-candidate recovery (strategies_v4 pass-5 rule: residual
       deaths processed in file order; accepted only if exactly one unused
       tolerant candidate; greedy, the accepted birth is consumed)
  4  P1-P3 cross-year + XGBoost scores on ALL tolerant candidate pairs
       (v4ml.candidate_pairs, the stored model), greedy best-score 1:1, tau 0.3
       (the unique-candidate rule of row 3 is subsumed: unique candidates are
       scored, not auto-accepted - this is how v4.2 is built)
  5  same pairs/scores, Hungarian 1:1 within blocks (v4ml.assign), tau 0.3 = v4.2
Supplementary rows:
  3+4  row 3 matches kept, XGB greedy on the remaining candidates
  v4.0 reference (strategies_v4.v4_core: exact P1-P4 cross-year + unique pass 5)

Usage: python3 task_a_ablation.py <year> [tier] [--sweep]
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

import rv_common  # noqa: F401  (sets sys.path)
from rv_common import REV, fmt, greedy_best_score, metrics
from common import load_sinasc
from strategies import v32_core
from strategies_v4 import gen_candidates, v4_core
from v4ml import (MODEL_PATH, assign, build_records, candidate_pairs,
                  exact_p13, load_synth)

TAU = 0.3


def run(year: int, tier: str = "calibrated", do_sweep: bool = False):
    t0 = time.time()
    sim_rows, true_row, no_pair = load_synth(year, tier)
    sinasc = load_sinasc(year)
    rows = []

    def add(step, label, matches):
        m = metrics(matches, sim_rows, true_row, no_pair)
        m.update({"step": step, "label": label, "year": year, "tier": tier})
        rows.append(m)
        print(f"[{step}] {label}: {fmt(m)}  ({time.time()-t0:.0f}s)", flush=True)

    # 0 / 1
    m32, _ = v32_core(sinasc, sim_rows)
    add("0", "v3.2: 4 greedy exact passes (current-year index)", m32)
    add("1", "Passes 1-3 only (mother's-age pass removed)",
        {i: v for i, v in m32.items() if v[1] in (1, 2, 3)})

    # 2
    records = build_records(year)
    m13, used, indexes = exact_p13(records, sim_rows)
    add("2", "+ cross-year index (previous-December births)", dict(m13))

    # 3: tolerant unique-candidate recovery after P1-P3 (pass-5 rule)
    idx_mp, idx_ep, idx_mi = indexes
    m3 = dict(m13)
    used3 = set(used)
    for i, r in enumerate(sim_rows):
        if i in m3:
            continue
        cands = {c for c in gen_candidates(r, idx_mp, idx_ep, idx_mi) if c not in used3}
        if len(cands) == 1:
            j = cands.pop()
            used3.add(j)
            m3[i] = (j, 5)
    add("3", "+ tolerant unique-candidate recovery (greedy)", m3)

    # 4 / 5
    pairs = candidate_pairs(records, sim_rows, m13, used, indexes)
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    scores = model.predict_proba(np.array([f for _, _, f in pairs]))[:, 1]
    print(f"candidate pairs {len(pairs):,} for {len({p[0] for p in pairs}):,} deaths", flush=True)

    m4 = dict(m13); m4.update(greedy_best_score(pairs, scores, TAU))
    add("4", "+ XGBoost scoring, greedy best-score 1:1 (tau 0.3)", m4)
    m5 = dict(m13); m5.update(assign(pairs, scores, TAU))
    add("5", "+ Hungarian optimal 1:1 (tau 0.3) = v4.2", m5)

    # supplementary
    keep = [k for k, (i, c, _) in enumerate(pairs) if i not in m3 and c not in used3]
    pairs_r = [pairs[k] for k in keep]
    scores_r = scores[keep]
    m34 = dict(m3); m34.update(greedy_best_score(pairs_r, scores_r, TAU))
    add("S1", "suppl.: row 3 kept, then XGB greedy on remaining (tau 0.3)", m34)
    m34h = dict(m3); m34h.update(assign(pairs_r, scores_r, TAU))
    add("S2", "suppl.: row 3 kept, then XGB Hungarian on remaining (tau 0.3)", m34h)

    sinasc_prev = load_sinasc(year - 1)
    prev_dec = (sinasc_prev.filter(pl.col("DTNASC").str.slice(2, 2) == "12")
                if sinasc_prev is not None else None)
    m40, _, _ = v4_core(sinasc, prev_dec, sim_rows)
    add("ref", "reference v4.0 (P1-P4 cross-year + unique pass 5)", m40)

    elapsed = time.time() - t0
    out = {"year": year, "tier": tier, "tau": TAU, "seconds": elapsed, "rows": rows}

    if do_sweep:
        sweep_rows = []
        for tau in [round(0.1 * k, 1) for k in range(1, 10)]:
            for method, fn in (("hungarian", assign), ("greedy", greedy_best_score)):
                m = dict(m13); m.update(fn(pairs, scores, tau))
                mm = metrics(m, sim_rows, true_row, no_pair)
                mm.update({"tau": tau, "assignment": method, "year": year, "tier": tier})
                sweep_rows.append(mm)
                print(f"tau {tau} {method}: {fmt(mm)}", flush=True)
        out["tau_sweep"] = sweep_rows
        pl.DataFrame(sweep_rows).write_csv(REV / f"taskC_tau_sweep_{year}_{tier}.csv")

    out["seconds_total"] = time.time() - t0
    (REV / f"taskA_ablation_{year}_{tier}.json").write_text(json.dumps(out, indent=2))
    pl.DataFrame(rows).write_csv(REV / f"taskA_ablation_{year}_{tier}.csv")
    print(f"done {year} in {out['seconds_total']:.0f}s")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(int(args[0]), args[1] if len(args) > 1 else "calibrated", "--sweep" in sys.argv)
