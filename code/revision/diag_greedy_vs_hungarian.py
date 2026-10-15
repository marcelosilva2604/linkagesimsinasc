"""Revision diagnostic: do greedy best-score and Hungarian (v4ml.assign) give
the same 1:1 match SET at each tau? Also counts contested candidate births.
Usage: python3 diag_greedy_vs_hungarian.py <year> [tier]"""
import json, sys
import numpy as np, xgboost as xgb
import rv_common  # noqa
from rv_common import REV, greedy_best_score
from v4ml import MODEL_PATH, assign, build_records, candidate_pairs, exact_p13, load_synth
year = int(sys.argv[1]); tier = sys.argv[2] if len(sys.argv) > 2 else "calibrated"
sim_rows, true_row, no_pair = load_synth(year, tier)
records = build_records(year)
m13, used, idx = exact_p13(records, sim_rows)
pairs = candidate_pairs(records, sim_rows, m13, used, idx)
model = xgb.XGBClassifier(); model.load_model(MODEL_PATH)
scores = model.predict_proba(np.array([f for _, _, f in pairs]))[:, 1]
out = {"year": year, "n_pairs": len(pairs), "distinct_scores": int(len(np.unique(np.round(scores, 12))))}
from collections import Counter
cc = Counter(c for _, c, _ in pairs)
out["births_in_candidate_sets"] = len(cc)
out["births_shared_by_2plus_deaths"] = sum(1 for v in cc.values() if v > 1)
for tau in [round(0.1*k, 1) for k in range(1, 10)]:
    g = greedy_best_score(pairs, scores, tau); h = assign(pairs, scores, tau)
    gs = {(i, c) for i, (c, _) in g.items()}; hs = {(i, c) for i, (c, _) in h.items()}
    out[f"tau{tau}"] = {"greedy": len(gs), "hungarian": len(hs), "sym_diff_pairs": len(gs ^ hs),
                        "sum_score_greedy": float(sum(scores[k] for k,(i,c,_) in enumerate(pairs) if (i,c) in gs)),
                        "sum_score_hungarian": float(sum(scores[k] for k,(i,c,_) in enumerate(pairs) if (i,c) in hs))}
print(json.dumps(out, indent=1))
(REV / f"diag_greedy_vs_hungarian_{year}_{tier}.json").write_text(json.dumps(out, indent=2))
