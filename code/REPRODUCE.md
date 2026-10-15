# Reproducing the real-data results

No microdata are included. The inputs are public files that the script below downloads and checks.

## Steps

From the repository root, with Python 3.9+ and the packages in `requirements.txt`:

```
export LINKAGE_DATA_DIR=./data
cd code/revision
python3 download_inputs.py                                   # 23 files, about 2.6 GB zipped; SHA-256 checked
python3 extract_neonatal_sim.py --write-dir "$LINKAGE_DATA_DIR/processed"   # neonatal extraction from the general SIM files
mkdir -p ../../outputs/models ../../outputs/v4_full
cp ../../model/xgb_pairs.json ../../outputs/models/
cp ../../results/legacy/stats_all.json ../../outputs/v4_full/
python3 linkage_final.py                                     # annual runs, consolidation, four configurations
python3 task_h_real_sensitivity.py                           # yield, transitions, strata
python3 task_i_model_concordance.py                          # non-linkage model, agreement on other fields
python3 audit_exact_ambiguity.py                             # ambiguity of exact passes, multiple pregnancy by stage
```

Outputs are written under `outputs/revision/mim_final/` (ignored by git; they include record-level tables).
The eleven annual runs take about 90 seconds on an Apple M4 Max; the extraction takes a few minutes.

## Which input files, and what you will get

`code/input_hashes.csv` lists, for each of the 23 files, the SHA-256 of the version used in the study and of the version
published on 2026-10-15. Four files were replaced by the publisher after the study files were downloaded (SIM 2022, 2023, 2024
and SINASC 2024), and the publisher does not keep earlier versions. The study versions of those four files (public data,
unchanged) are attached to the release `inputs-2025-10` of this repository.

- Default (`download_inputs.py --version study`): 19 files from the publisher and the four study versions from the release.
  Every file matches `sha256_study`, and the pipeline gives the reported numbers: 231,478 linked deaths of 260,219 (88.96%),
  and pair tables identical to those of the study.
- `--version published`: the publisher's files of 2026-10-15. The pipeline then gives 231,494 linked deaths (88.96%);
  see `results/replication_current_files/`. The 16 additional pairs are in exact passes of the 2024 run.

`extract_neonatal_sim.py` reads the three layouts found in the general SIM files (header with ORIGEM first; no header, 87 or
86 fields; header with CONTADOR first) and writes dates as ddmmyyyy. The neonatal rule is date of death minus date of birth of
0-27 days, in file order; with the study files it reproduced the extracts used in the study record by record.

## What was checked

- Extraction: same record count in every year and same order on eight fields as the study extracts (study files), and the same
  260,219 records on the files of 2026-10-15.
- The whole procedure above was run from an empty folder with `--version study`: all 23 hashes matched, and the run pairs,
  accepted pairs of the four configurations, and death-level linkage were identical to the tables of the study.
- It was also run from an empty folder with `--version published` (results in `results/replication_current_files/`).
- Both runs were done by the author on one machine (Apple M4 Max).

## Limits

- Row identity of a record is the file year plus its row position in the extract; it depends on the file order.
- The synthetic reference is not independent of the generator (see `PSEUDOCODE.md`, section 8).
- The classifier is the frozen model of the study; `training_manifest.py` reconstructs its training-pair counts, and the
  original training log was not saved.
- The original extraction script was not preserved; the rule above reproduces its output.
