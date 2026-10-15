# Pseudocode of the procedure and analyses

This file describes what the code in `code/` does, in the order of execution. Names in `monospace` are the functions
and files that implement each step. It is a description of the study's procedure, not a specification of a recommended method.

## 0. Notation

- `D`: neonatal death records (SIM), one per row; `B`: live birth records (SINASC).
- A **death record** has `DTNASC` (date of birth), `SEXO`, `CODMUNNATU` (municipality of residence/naturalidade field used
  in linkage), `CODMUNOCOR` (municipality of occurrence), `CODESTAB`, `PESO`, `IDADEMAE`.
- A **birth record** has `DTNASC`, `SEXO`, `CODMUNNASC`, `CODESTAB`, `PESO`, `IDADEMAE`.
- `year`: the SIM file year; a record is identified by `(file year, row position)`.

## 1. Neonatal extraction (`revision/extract_neonatal_sim.py`)

```
for each year y in 2014..2024:
    G <- read general mortality file of y     # three published layouts are accepted; dates written as ddmmyyyy
    D_y <- records of G with 0 <= (date of death - date of birth) <= 27 days     # file order kept
    compare D_y with the preserved extract: record count, and agreement by position on eight fields
```

## 2. One annual run (`revision/linkage_final.py: run_year`)

```
inputs: D_y (deaths of file y) plus January deaths of file y+1 whose birth year is y   # "January window"
        B_y (births of y) plus December births of y-1                                  # cross-year index

index B by three exact keys:
    K1 = (DTNASC, SEXO, municipality, PESO)
    K2 = (DTNASC, CODESTAB, PESO)
    K3 = (DTNASC, SEXO, municipality, IDADEMAE)      # used only to generate tolerant candidates

# exact passes, each over all still-unmatched deaths, in file order
P1: for each death d:   key = (DTNASC, SEXO, CODMUNNATU, PESO)  -> first unused birth with that key
P2: for each unmatched d: key = (DTNASC, SEXO, CODMUNOCOR, PESO) -> first unused birth with that key
P3: for each unmatched d: key = (DTNASC, CODESTAB, PESO)         -> first unused birth with that key
(each accepted pair marks the birth as used)

# tolerant candidates for deaths without an exact match
for each unmatched d:
    candidates(d) = unused births found by probing K1, K2, K3 with:
        DTNASC in {date, date +1 day, date -1 day, date with day digits swapped}
        PESO   in {w, w+10, w-10, w rounded to 50, w rounded to 100}
        IDADEMAE in {a, a+1, a-1};  SEXO in {sex} or {1, 2} if missing/indeterminate
        municipality in {CODMUNNATU, CODMUNOCOR}
    for each (d, c) in candidates(d):
        x = comparison vector (seven inputs):
            date state   0 equal | 1 differ by one day | 2 digit swap | 3 other or missing
            sex state    0 equal | 1 missing | 2 differ
            municipality 0 birth municipality | 1 occurrence municipality | 2 none
            weight state 0 equal | 1 within 10 g | 2 equal after rounding to 50 | 3 rounding to 100 | 4 missing | 5 other
            age state    0 equal | 1 differ by one | 2 missing | 3 other
            facility     0 equal | 1 differ | 2 missing
            n_candidates min(|candidates(d)|, 50)
        score(d, c) = XGBoost(x)                      # frozen classifier, model/xgb_pairs.json

# one-to-one assignment within connected blocks of competing candidates
build connected components over shared candidates
for each component:
    solve Hungarian assignment maximizing the sum of scores   (scipy linear_sum_assignment)
record every assigned pair with its score; the threshold tau is NOT applied here

output: run_pairs = (run year, death id, birth id, stage in {P1, P2, P3, CLF}, score)
```

## 3. Consolidation across annual runs (`revision/linkage_final.py: consolidate`)

```
input: all run_pairs; threshold tau (None = exact passes only)
keep pairs with stage in {P1, P2, P3}, or stage = CLF and score >= tau
rank stage: P1 < P2 < P3 < CLF
sort pairs by (stage rank, score descending, January-window run last, run year)
collapse identical (death id, birth id) pairs, keeping the first and counting runs
accepted <- {}
for each pair in order:
    if death already assigned: drop (death already linked)
    else if birth already assigned: drop (birth already taken)
    else accept
output: one accepted birth at most per death and one death at most per birth

configurations:  E (exact only), C02 (tau 0.2), C03 (tau 0.3, reference), C06 (tau 0.6)
yield L / N = |accepted| / |D|
```

## 4. Sensitivity and composition (`revision/task_h_real_sensitivity.py`)

```
for each configuration c: yield, gains and losses relative to C03, changes of assigned birth among deaths linked in both
for strata (birthweight status, region of occurrence, place of death, file year):
    proportion unlinked within stratum and share of all unlinked deaths
classifier recovery: among deaths unlinked by exact passes, share linked by the classifier stage, by weight status
birthweight status: missing = blank; invalid = non-numeric or outside 200-8000 g; valid otherwise
```

## 5. Model of non-linkage and agreement on other fields (`revision/task_i_model_concordance.py`)

```
logistic regression  unlinked ~ region + weight category + race/color + place of death + (file year - 2014)
    maximum likelihood, references fixed in code, Wald 95% intervals

for stage in {exact passes, classifier}:
    for field in {mother's municipality of residence, gestational age 20-45, schooling 1-5, plurality 1-3, delivery type 1-2}:
        compare the death record with the accepted birth record where both values are substantive
        (state-level municipality codes ending 0000 are not comparable)
        agreement = share equal
        comparators: 50 permutations of birth values within (stage, file year), and within (stage, file year, state of occurrence)
descriptive only: no accuracy is estimated
```

## 6. Ambiguity of the exact passes (`revision/audit_exact_ambiguity.py`)

```
for each annual run:
    re-run P1-P3 as in section 2 and record, for each matched death, the number of unused births sharing the key
    re-run with the index lists reversed, and with the order of deaths shuffled (seed 7)
    count deaths whose assigned birth changes, and deaths left unmatched
join to the accepted C03 pairs by death identity; assert that the audited birth equals the accepted birth
report: ambiguous pairs per pass, changes, and multiple pregnancy (plurality code 2 or 3) by stage
```

## 7. Frozen classifier and training (`v4ml.py: train`, `training_manifest.py`)

```
training year 2014, calibrated synthetic reference (section 8):
    run exact passes; generate candidates for unmatched synthetic deaths
    label = 1 if the candidate is the known true pair, else 0
    fit XGBoost (300 trees, depth 5, learning rate 0.1, subsample 0.9, column sample 0.9, seed 42), no tuning
```

## 8. Synthetic reference with known pairs (`make_synthetic_reference.py`, `calibrate_error_model.py`)

```
measure missingness of each linking field in the real SIM neonatal file of the year
sample true birth records, stratified to the real death birthweight distribution,
    with the measured share born in December of the previous year, and some with no true pair
perturb linking fields with discrepancy rates (date +-1, weight rounding, municipality swaps, age +-1, ...)
    rates were set so that the earlier baseline procedure reproduces its pass profile on real data
    optimistic = 0.5x and pessimistic = 2x those rates
write synthetic deaths and the table synthetic id -> true birth row
```

The reference is conditional on this generator and shares linkage keys with the procedure; differences between strategies on it
are not estimates of accuracy on real pairs.

## 9. Comparison with Fellegi-Sunter linkage (`bench_splink.py`, `revision/task_b_*.py`, `task_e_*.py`, `task_g_*.py`)

```
for the synthetic reference of a year:
    fit a Fellegi-Sunter model (splink) with the same source fields; m probabilities estimated without labels,
        or from the known pairs of the 2014 reference (supervised variant)
    for each probability threshold: greedy one-to-one assignment by probability, precision, recall, F1
thresholds of the comparators are chosen on the evaluation data and are exploratory
```
