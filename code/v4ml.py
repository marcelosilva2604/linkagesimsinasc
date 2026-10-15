"""
v4.1/v4.2: ML-scored candidate pairs + globally optimal assignment.

Pipeline (stage "v4.2", the full method):
  1. Exact passes P1-P3 (high-precision, greedy) are accepted as-is.
  2. Every remaining SIM record gets a tolerant candidate set (strategies_v4
     variant probing) - this REPLACES v3.2's noisy greedy Pass 4.
  3. Each (death, candidate) pair is encoded as a comparison vector.
  4. An XGBoost classifier scores the pairs. It is trained on the synthetic
     reference of a DIFFERENT year (default 2014, calibrated tier), where the
     true pairs are known by construction; the reference is not independent of
     the generator, and nothing identified is ever needed.
  5. Within connected candidate blocks, a globally optimal 1:1 assignment
     (scipy Hungarian) maximises total score; pairs below the acceptance
     threshold tau are rejected.

CLI:
  python3 v4ml.py train [train_year] [tier]      -> outputs/models/xgb_pairs.json
  python3 v4ml.py eval  <year> [tier] [tau]      -> metrics + eval json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

import numpy as np
import polars as pl
import xgboost as xgb
from scipy.optimize import linear_sum_assignment

from common import OUTPUTS, load_sinasc
from strategies_v4 import (PREV_OFFSET, build_indexes, gen_candidates,
                           sinasc_records)

MODEL_PATH = OUTPUTS / "models" / "xgb_pairs.json"


# --------------------------------------------------------------------------
# comparison vector
# --------------------------------------------------------------------------

def _date_state(a: str | None, b: str | None) -> int:
    if not a or not b:
        return 3
    if a == b:
        return 0
    try:
        da = datetime.strptime(a, "%d%m%Y")
        db = datetime.strptime(b, "%d%m%Y")
        if abs((da - db).days) == 1:
            return 1
    except ValueError:
        return 3
    if a[1] + a[0] + a[2:] == b:
        return 2
    return 3


def _peso_state(a: str | None, b: str | None) -> int:
    if a is None or b is None:
        return 4
    try:
        wa, wb = int(float(a)), int(float(b))
    except ValueError:
        return 5
    d = abs(wa - wb)
    if d == 0:
        return 0
    if d <= 10:
        return 1
    if wa == int(round(wb / 50.0) * 50) or wb == int(round(wa / 50.0) * 50):
        return 2
    if wa == int(round(wb / 100.0) * 100) or wb == int(round(wa / 100.0) * 100):
        return 3
    return 5


def _idm_state(a: str | None, b: str | None) -> int:
    if a is None or b is None:
        return 2
    try:
        d = abs(int(float(a)) - int(float(b)))
    except ValueError:
        return 3
    return 0 if d == 0 else (1 if d == 1 else 3)


def features(sim_row: dict, rec: tuple, n_cands: int) -> list[float]:
    """rec = (row_id, dt, sexo, mun, estab, peso, idademae)."""
    _, dt, sx, mun, est, peso, idm = rec
    sim_sx = sim_row.get("SEXO")
    sexo_state = (2 if sim_sx in ("1", "2") and sx in ("1", "2") and sim_sx != sx
                  else (0 if sim_sx == sx else 1))
    natu, ocor = sim_row.get("CODMUNNATU"), sim_row.get("CODMUNOCOR")
    mun_state = 0 if (natu and natu == mun) else (1 if (ocor and ocor == mun) else 2)
    sim_est = sim_row.get("CODESTAB")
    estab_state = 2 if (not sim_est or not est) else (0 if sim_est == est else 1)
    return [
        _date_state(sim_row.get("DTNASC"), dt),
        sexo_state,
        mun_state,
        _peso_state(sim_row.get("PESO"), peso),
        _idm_state(sim_row.get("IDADEMAE"), idm),
        estab_state,
        float(min(n_cands, 50)),
    ]


# --------------------------------------------------------------------------
# shared pipeline pieces
# --------------------------------------------------------------------------

def load_synth(year: int, tier: str):
    synth = pl.read_csv(
        OUTPUTS / "synthetic" / f"synthetic_SIM_{year}_{tier}.csv",
        infer_schema_length=0,
        null_values=["", "NA", "NULL", "None", "null"],
    )
    truth = pl.read_csv(OUTPUTS / "synthetic" / f"truth_{year}_{tier}.csv")
    true_row = dict(zip(truth["SYNTH_ID"].to_list(), truth["TRUE_ROW"].to_list()))
    no_pair = dict(zip(truth["SYNTH_ID"].to_list(), truth["NO_TRUE_PAIR"].to_list()))
    return synth.to_dicts(), true_row, no_pair


def build_records(year: int):
    cur = load_sinasc(year)
    prev = load_sinasc(year - 1)
    records = sinasc_records(cur)
    if prev is not None:
        prev_dec = prev.filter(pl.col("DTNASC").str.slice(2, 2) == "12")
        records += sinasc_records(prev_dec, offset=PREV_OFFSET)
    return records


def exact_p13(records, sim_rows):
    """Greedy exact passes P1-P3 over the cross-year index."""
    idx_mp, idx_ep, idx_mi = build_indexes(records)
    used: set[int] = set()
    matches: dict[int, tuple[int, int]] = {}

    def g(row, field):
        v = row.get(field)
        return "" if v is None else str(v)

    def try_key(i, key, index, pid):
        if key in index:
            for j in index[key]:
                if j not in used:
                    used.add(j)
                    matches[i] = (j, pid)
                    return True
        return False

    for i, r in enumerate(sim_rows):
        try_key(i, f"{g(r,'DTNASC')}_{g(r,'SEXO')}_{g(r,'CODMUNNATU')}_{g(r,'PESO')}", idx_mp, 1)
    for i, r in enumerate(sim_rows):
        if i not in matches:
            try_key(i, f"{g(r,'DTNASC')}_{g(r,'SEXO')}_{g(r,'CODMUNOCOR')}_{g(r,'PESO')}", idx_mp, 2)
    for i, r in enumerate(sim_rows):
        if i not in matches:
            try_key(i, f"{g(r,'DTNASC')}_{g(r,'CODESTAB')}_{g(r,'PESO')}", idx_ep, 3)
    return matches, used, (idx_mp, idx_ep, idx_mi)


def candidate_pairs(records, sim_rows, matches, used, indexes):
    """(sim_idx, row_id, feature_vector) for every unmatched death's candidates."""
    idx_mp, idx_ep, idx_mi = indexes
    rec_by_id = {r[0]: r for r in records}
    pairs = []
    for i, r in enumerate(sim_rows):
        if i in matches:
            continue
        cands = {c for c in gen_candidates(r, idx_mp, idx_ep, idx_mi) if c not in used}
        for c in cands:
            pairs.append((i, c, features(r, rec_by_id[c], len(cands))))
    return pairs


# --------------------------------------------------------------------------
# train / eval
# --------------------------------------------------------------------------

def train(train_year: int = 2014, tier: str = "calibrated"):
    sim_rows, true_row, _ = load_synth(train_year, tier)
    records = build_records(train_year)
    matches, used, indexes = exact_p13(records, sim_rows)
    pairs = candidate_pairs(records, sim_rows, matches, used, indexes)

    X = np.array([f for _, _, f in pairs])
    y = np.array([
        1 if true_row[int(sim_rows[i]["SYNTH_ID"])] == c else 0
        for i, c, _ in pairs
    ])
    print(f"train {train_year} [{tier}]: {len(pairs):,} candidate pairs, "
          f"{y.sum():,} positives ({y.mean()*100:.1f}%)")

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", n_jobs=4, random_state=42,
    )
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


def assign(pairs, scores, tau: float):
    """Hungarian within connected blocks; accept score >= tau, 1:1 both sides."""
    by_sim: dict[int, list[int]] = {}
    for k, (i, c, _) in enumerate(pairs):
        by_sim.setdefault(i, []).append(k)

    # connected components over shared candidates
    comp_of: dict[int, int] = {}
    cand_comp: dict[int, int] = {}
    comp_members: dict[int, set[int]] = {}
    nc = 0
    for i, ks in by_sim.items():
        comps = {cand_comp[pairs[k][1]] for k in ks if pairs[k][1] in cand_comp}
        if comps:
            target = min(comps)
            for other in comps - {target}:
                for m in comp_members.pop(other):
                    comp_of[m] = target
                    comp_members.setdefault(target, set()).add(m)
                for cc, v in list(cand_comp.items()):
                    if v == other:
                        cand_comp[cc] = target
        else:
            target = nc
            nc += 1
        comp_of[i] = target
        comp_members.setdefault(target, set()).add(i)
        for k in ks:
            cand_comp[pairs[k][1]] = target

    accepted: dict[int, tuple[int, int]] = {}
    for members in comp_members.values():
        ks = [k for i in members for k in by_sim[i]]
        sims = sorted({pairs[k][0] for k in ks})
        cands = sorted({pairs[k][1] for k in ks})
        si = {s: a for a, s in enumerate(sims)}
        ci = {c: a for a, c in enumerate(cands)}
        cost = np.full((len(sims), len(cands)), 1e6)
        for k in ks:
            i, c, _ = pairs[k]
            cost[si[i], ci[c]] = -scores[k]
        rows, cols = linear_sum_assignment(cost)
        for a, b in zip(rows, cols):
            if -cost[a, b] >= tau:
                accepted[sims[a]] = (cands[b], 6)
    return accepted


def evaluate(year: int, tier: str = "calibrated", tau: float = 0.5,
             stage: str = "v4.2"):
    sim_rows, true_row, no_pair = load_synth(year, tier)
    records = build_records(year)
    matches, used, indexes = exact_p13(records, sim_rows)
    pairs = candidate_pairs(records, sim_rows, matches, used, indexes)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    scores = model.predict_proba(np.array([f for _, _, f in pairs]))[:, 1]

    matches.update(assign(pairs, scores, tau))

    n = len(sim_rows)
    n_pairable = sum(1 for v in no_pair.values() if v == 0)

    def truth_of(i):
        return true_row[int(sim_rows[i]["SYNTH_ID"])]

    correct = sum(1 for i, (j, _) in matches.items() if truth_of(i) == j)
    n_matched = len(matches)
    precision = correct / n_matched if n_matched else 0.0
    recall = correct / n_pairable
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    pass_counts: dict[int, int] = {}
    correct_by_pass: dict[int, int] = {}
    for i, (j, p) in matches.items():
        pass_counts[p] = pass_counts.get(p, 0) + 1
        if truth_of(i) == j:
            correct_by_pass[p] = correct_by_pass.get(p, 0) + 1

    print(f"\n=== {stage} (tau={tau}) on synthetic {year} [{tier}] ===")
    print(f"deaths {n:,} (pairable {n_pairable:,}) | matched {n_matched:,} "
          f"| correct {correct:,} | wrong {n_matched - correct:,}")
    for p in sorted(pass_counts):
        c = pass_counts[p]
        cb = correct_by_pass.get(p, 0)
        label = "ML+1:1" if p == 6 else f"Pass {p}"
        print(f"  {label}: {c:,} ({c/n*100:.1f}%) - precision {cb/c*100:.1f}%")
    print(f"  Unmatched: {n - n_matched:,} ({(n - n_matched)/n*100:.1f}%)")
    print(f"  PRECISION        {precision*100:.2f}%")
    print(f"  RECALL pairable  {recall*100:.2f}%")
    print(f"  F1               {f1*100:.2f}%")

    res = {
        "strategy": stage, "tau": tau, "year": year, "tier": tier,
        "n": n, "n_pairable": n_pairable, "matched": n_matched,
        "correct": correct, "wrong": n_matched - correct,
        "precision": precision, "recall_pairable": recall, "f1": f1,
        "pass_counts": pass_counts,
        "pass_precision": {p: correct_by_pass.get(p, 0) / c
                           for p, c in pass_counts.items()},
    }
    out = OUTPUTS / "synthetic" / f"eval_v42_{year}_{tier}_tau{tau}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved {out}")
    return res


def sweep(year: int, tier: str = "calibrated"):
    sim_rows, true_row, no_pair = load_synth(year, tier)
    records = build_records(year)
    matches0, used, indexes = exact_p13(records, sim_rows)
    pairs = candidate_pairs(records, sim_rows, matches0, used, indexes)
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    scores = model.predict_proba(np.array([f for _, _, f in pairs]))[:, 1]
    n_pairable = sum(1 for v in no_pair.values() if v == 0)

    def truth_of(i):
        return true_row[int(sim_rows[i]["SYNTH_ID"])]

    base_correct = sum(1 for i, (j, _) in matches0.items() if truth_of(i) == j)
    print(f"tau sweep {year} [{tier}] (P1-P3 fixed: {len(matches0):,} matched, "
          f"{base_correct:,} correct)")
    print(f"{'tau':>5} {'matched':>8} {'correct':>8} {'prec%':>7} {'rec%':>7} {'F1%':>7}")
    for tau in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        acc = assign(pairs, scores, tau)
        m = dict(matches0); m.update(acc)
        correct = sum(1 for i, (j, _) in m.items() if truth_of(i) == j)
        prec = correct / len(m)
        rec = correct / n_pairable
        f1 = 2 * prec * rec / (prec + rec)
        print(f"{tau:>5} {len(m):>8,} {correct:>8,} {prec*100:>7.2f} {rec*100:>7.2f} {f1*100:>7.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sweep":
        sweep(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "calibrated")
    elif cmd == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 2014,
              sys.argv[3] if len(sys.argv) > 3 else "calibrated")
    elif cmd == "eval":
        evaluate(int(sys.argv[2]),
                 sys.argv[3] if len(sys.argv) > 3 else "calibrated",
                 float(sys.argv[4]) if len(sys.argv) > 4 else 0.5)
