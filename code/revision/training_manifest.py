"""verifiable counts for the frozen classifier: reconstructs the training candidate pairs from the archived 2014
synthetic reference with the same functions as v4ml.train() (no refit) and records the number of rounds and the maximum tree depth observed in the frozen model file; the other hyperparameters come from the v4ml.train() source (the file does not store them).
Writes outputs/revision/mim_final/training_manifest.json."""
import hashlib, json
import numpy as np
import xgboost as xgb
import rv_common  # noqa: F401
from rv_common import REV
from v4ml import MODEL_PATH, build_records, candidate_pairs, exact_p13, load_synth

sim_rows, true_row, _ = load_synth(2014, "calibrated")
records = build_records(2014)
matches, used, indexes = exact_p13(records, sim_rows)
pairs = candidate_pairs(records, sim_rows, matches, used, indexes)
y = np.array([1 if true_row[int(sim_rows[i]["SYNTH_ID"])] == c else 0 for i, c, _ in pairs])
model = xgb.XGBClassifier(); model.load_model(MODEL_PATH)
def depth(node):
    return 1 + max((depth(c) for c in node.get("children", [])), default=0) if "children" in node else 0
trees = [json.loads(t) for t in model.get_booster().get_dump(dump_format="json")]
max_depth_observed = max(depth(t) for t in trees)
out = {"year": 2014, "tier": "calibrated", "synthetic_deaths": len(sim_rows), "exact_pass_matches_removed": len(matches),
       "candidate_pairs": len(pairs), "positives": int(y.sum()),
       "model_file": str(MODEL_PATH), "model_sha256": hashlib.sha256(open(MODEL_PATH, "rb").read()).hexdigest(),
       "boosted_rounds": model.get_booster().num_boosted_rounds(), "max_tree_depth_observed": max_depth_observed,
       "hyperparameters_source": "v4ml.train() source: n_estimators 300, max_depth 5, learning_rate 0.1, subsample 0.9, "
                                 "colsample_bytree 0.9, random_state 42; the saved model file does not store them (a reload shows defaults)",
       "note": "reconstruction with the same functions; the original training log was not saved"}
json.dump(out, open(REV / "mim_final" / "training_manifest.json", "w"), indent=2)
print(json.dumps(out, indent=2))
