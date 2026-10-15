# linkagesimsinasc

Code, pseudocode, notebooks, and aggregate results for the study **"Linking Public Birth and Neonatal Death Records in Brazil Without Direct
Identifiers: Linkage Yield and Differential Non-linkage, 2014-2024"** (manuscript in preparation).

The study links the public, de-identified live birth records (SINASC) to neonatal death records (SIM) of Brazil, 2014-2024,
describes the linkage yield and the composition of the deaths left unlinked, and compares predefined linkage decisions.

## Contents

| File | What it shows |
|---|---|
| `01_procedure_toy_example.ipynb` | The procedure step by step on a small invented example: exact passes, tolerant candidates, comparison inputs, one-to-one assignment, consolidation of overlapping annual runs. Runs without the public data. |
| `02_results_2014_2024.ipynb` | Record flow, yield by year, composition of unlinked deaths, sensitivity to four predefined configurations, association with non-linkage, agreement on fields not used in linkage. Reads `results/real/`. |
| `03_synthetic_evaluation.ipynb` | Simulation-based comparison of strategies (baseline, procedure, Fellegi-Sunter variants) on a synthetic reference with pairs known by construction. Reads `results/synthetic/`. |
| `PSEUDOCODE.md` | The procedure and each analysis step in pseudocode, with the file that implements it. |
| `code/` | Python code: extraction of neonatal deaths from the general SIM files, linkage, consolidation, sensitivity analyses, audits, synthetic reference, and comparison with Fellegi-Sunter linkage. See `code/REPRODUCE.md`. |
| `results/` | Aggregate results only; `results/replication_current_files/` holds the results of running the code from the download step on the files published on 2026-10-15. |
| `model/xgb_pairs.json` | The frozen pair classifier used in the study. |

## What is not here

No record-level output of the study is included. The public input files are downloaded from their publisher; the four files that the publisher replaced after the study download are attached unchanged to the release `inputs-2025-10`. To run the code, follow `code/REPRODUCE.md`: `download_inputs.py` fetches the public files and checks their hashes.

## How to read the results

- **Linkage yield** is the proportion of death records assigned an accepted birth record. It does not measure the correctness
  of accepted pairs or the completeness of vital registration.
- No independent reference of true pairs was available for this study. The synthetic reference is not independent of the
  generator, and its results describe differences between strategies under the generator's assumptions, not accuracy on real pairs.
- The profile of unlinked deaths describes observed selection in the analyzed files and does not estimate the bias of any
  downstream analysis.

## Requirements

Notebooks: Python 3.9+, `numpy`, `pandas`, `scipy`, `matplotlib`; optional `xgboost` to load the frozen classifier in notebook 01. Code: see `requirements.txt` (adds `polars`, `xgboost`, `statsmodels`, `splink`).

## License

MIT (see `LICENSE`).
