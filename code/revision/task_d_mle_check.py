"""Revision Task D check: exact MLE (statsmodels Newton) of the same logistic
model, for the key-based and identity-based flags, to see whether the sklearn
lbfgs fit of analyze_unlinked_bias.py (tol 1e-4, C=1e6) is converged.
Rebuilds the covariate frame via task_d_unlinked_identity.build_frame using the
saved identity flags. Usage: python3 task_d_mle_check.py"""
import numpy as np, polars as pl, statsmodels.api as sm
import rv_common  # noqa
from rv_common import REV
from sklearn.preprocessing import OneHotEncoder
import task_d_unlinked_identity as T
flags = pl.read_csv(REV / "taskD_unlinked_flags_by_row.csv")
linked = {y: set(flags.filter((pl.col("year") == y) & (pl.col("unlinked_id") == 0))["pos"].to_list()) for y in T.YEARS}
df = T.build_frame(linked)
X_cats = df.select(["region", "peso_stratum", "racacor", "lococor"]).to_numpy()
enc = OneHotEncoder(drop="first", sparse_output=False)
X = np.hstack([enc.fit_transform(X_cats), (df["year"].to_numpy().astype(float) - 2014.0).reshape(-1, 1)])
names = list(enc.get_feature_names_out(["region", "peso_stratum", "racacor", "lococor"])) + ["year_trend"]
rows = []
for ycol in ("unlinked_key", "unlinked_id"):
    res = sm.Logit(df[ycol].to_numpy(), sm.add_constant(X)).fit(method="newton", maxiter=100, tol=1e-10, disp=0)
    ci = res.conf_int(); print(ycol, "converged", res.mle_retvals["converged"])
    for k, n in enumerate(names):
        rows.append({"flag": ycol, "term": n, "OR": float(np.exp(res.params[k+1])),
                     "lo95": float(np.exp(ci[k+1, 0])), "hi95": float(np.exp(ci[k+1, 1]))})
out = pl.DataFrame(rows); out.write_csv(REV / "taskD_bias_logit_MLE_statsmodels.csv")
w = out.pivot(on="flag", index="term", values="OR"); print(w)
