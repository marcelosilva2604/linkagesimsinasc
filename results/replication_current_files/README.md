# Results with the input files published on 2026-10-15

The linkage of the study was run on the input files downloaded in October 2025. Between that date and 2026-10-15 the
publisher replaced four of the 23 files (SIM 2022, 2023, 2024 and SINASC 2024; see `code/input_hashes.csv`). The SIM files
of 2022 and 2023 now have no header line and use dd-mm-yyyy dates; `code/revision/extract_neonatal_sim.py` reads both layouts.

This folder holds aggregate results of running the code of this repository from the download step on the files as
published on 2026-10-15 (`download_inputs.py --version published`). With the default `--version study` the numbers are
those of the study (column 2).

| Quantity | Study files | Files of 2026-10-15 |
|---|---:|---:|
| Neonatal death records, 2014-2024 | 260,219 | 260,219 |
| Records identical on eight fields, by position | | 260,219 |
| Linked, reference configuration (tau 0.3) | 231,478 (88.96%) | 231,494 (88.96%) |
| Linked, exact passes only | 215,104 (82.67%) | 215,120 (82.67%) |
| Linked at tau 0.2 / 0.6 | 234,537 / 228,312 | 234,557 / 228,329 |
| Classifier-stage pairs | 16,374 | 16,374 |
| Exact pairs ambiguous at match | 4,723 | 4,720 |

The additional pairs are all in the run of 2024 (the births of SINASC 2024 changed) and in exact passes; every other run year
matched the recorded counts. Percentages round to the same values as in the manuscript.
