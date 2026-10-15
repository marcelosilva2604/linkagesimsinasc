# Reproducing the real-data results

No microdata are included. Inputs are public DATASUS files that you download yourself.

## Inputs

Set `LINKAGE_DATA_DIR` to a folder with:

```
raw/SINASC/SINASC_<year>.csv            live births, 2013-2024
raw/SIM/Mortalidade_Geral_<year>.csv    general mortality files, 2014-2024
processed/SIM_neonatal_<year>.csv       neonatal death extracts, 2014-2024 (rebuilt by extract_neonatal_sim.py)
```

`outputs/v4_full/stats_all.json` (per-year counts of the earlier baseline run) is used as a consistency check:
copy `results/legacy/stats_all.json` there. The frozen classifier is `model/xgb_pairs.json`; copy it to
`outputs/models/xgb_pairs.json`.

## Steps (run from `code/revision/`)

```
python3 extract_neonatal_sim.py --write-dir "$LINKAGE_DATA_DIR/processed"   # neonatal extraction from the general files
python3 run_real_analysis.py                                              # steps 1-6 listed in that file
```

Environment used: Apple M4 Max, 16 cores, 64 GB; Python 3.9, Polars 1.36.1, XGBoost 2.1.4, SciPy 1.13, statsmodels.
The eleven annual runs take about 90 seconds.

## What was checked

- The neonatal extract rebuilt from the general files (death minus birth date of 0-27 days, file order) had the same
  number of records in every year (260,219 in total) and the same order on eight fields as the extracts used in the study.
- Running `linkage_final.py` on the rebuilt extracts produced identical pair tables (run pairs, accepted pairs, and
  death-level linkage) for the four configurations. This was done by the author on the same machine; it is
  repeatability and a check of the extraction rule, not an independent reproduction.
- The original extraction script was not preserved; the rule above reproduces its output.

## Limits

- Row identity of a record is the file year plus its row position in the extract; it depends on the file order.
- The synthetic reference is not independent of the generator (see `PSEUDOCODE.md`, section 8).
- The classifier was reconstructed for its training-pair counts only (`training_manifest.py`); the original training
  log was not saved.
