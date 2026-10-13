# LeukoGuard: Fixed Data-Cleaning Rules (Member 1, Part B)

**Dataset:** LeukoAlert public release, `leuk_vs_HC` (leukemia vs. healthy control), 11 CSV files, 446,663 records, 74 numeric columns.
**Implementation:** `src/leuko_common.py → clean_frame()`. Every team member imports this function, so all models use identical data.
**Result:** 446,637 records × 71 features (+ 6 metadata columns: `sample_id, source_file, split, site, cohort, y`).

## Principle
Fixed cleaning contains **only deterministic rules**: each acts on one record, or on one file's recording convention, and never uses statistics learned from labels or from other splits.
Everything that *learns* from data is done **inside the model pipeline on training folds only**:

| Not done here | Where it happens |
|---|---|
| Imputation of NaN (+ missing-indicator columns) | `SimpleImputer(add_indicator=True)` inside the CV pipeline |
| Scaling / log-transform / winsorising | inside the pipeline |
| SMOTE / class weights | inside training folds only |
| Outlier removal | **never**: extreme counts are the disease signal |

## Rules

| ID | Rule | Evidence from EDA | Records / cells affected |
|---|---|---|---|
| R1 | Rename raw headers to short team names; label → `y` (1 = leukemia, 0 = healthy) | — | all |
| R2 | **Drop** `Q-Flag(Blasts/Abn Lympho?)`, `Q-Flag(Blasts?)`, `Q-Flag(Abn Lympho?)` | Not in the paper's 72-feature list; 50–95% pre-filled, constant 0 at train-B and valid-D/G; the analyser's own blast message (label shortcut) | 3 columns |
| R3 | **Disguised missing → NaN**: in each file, values off the instrument precision grid (more than 4 decimals) | Every file/column has exactly **one** such value (e.g. `104.2533936651584`), a different one per file: per-file mean fill-in done before release | 105,552 cells |
| R4 | **HCT unit**: in files where median HCT < 1, multiply HCT by 100 | Sites B and C store HCT as a fraction (0.35), the others as a percent (35) | 167,532 records (sites B, C) |
| R5 | RBC, HGB, HCT, MCV, MCH, MCHC, RDW-SD, RDW-CV = 0 → NaN | Impossible in a living patient; failed measurement | 549 cells |
| R6 | MPV = 0 → PDW, MPV, P-LCR, PCT all NaN | The four platelet indices are always 0 together (not computed, usually very low PLT) | 5,390 records |
| R7 | Sum of NEUT%+LYMPH%+MONO%+EO%+BASO% = 0 → 23-column differential block NaN | No differential reported, while WBC is still measured in 96.6% of these records | 3,856 records |
| R8 | Cell-population parameters (NE-/LY-/MO- positions and widths) = 0 → NaN | A scattergram channel position or width cannot be 0 | 34,046 cells |
| R9 | MCHC > 500 g/L → MCHC and MCH NaN | Physically impossible (ceiling ~370); agglutination/optical artefact with falsely low RBC | 139 records |
| R10 | Exact duplicates (features + label) within a file: keep first; identical features with conflicting labels: drop all | 31 duplicate records (valid-D, valid-G); valid-G ones are fully mean-filled empty records; none cross splits | 22 records removed |
| R11 | Remove records with WBC, HGB and PLT all missing | Nothing to screen | 4 records removed |
| R12 | Types: features float64, `y` int8, site/split/cohort categorical | — | — |

## Deliberately NOT treated as errors
* **WBC = 0, PLT = 0**: below-detection readings; 47% of WBC = 0 records are leukemia (aplasia).
* **Extreme WBC (up to 727 ×10⁹/L), IG#, NRBC#, MONO%**: 50–93% of the >3×IQR outliers are leukemia.
* **NEUT% = 0 or 100 with a differential present**: possible when abnormal cells dominate.

## Known limitations
* **No patient identifier.** `sample_id` is per record; repeat CBCs of one person may fall into different CV folds of the development set, so CV may be slightly optimistic. Generalisation is judged on the external hospitals (D–G).
* **Missingness is informative and site-specific.** E.g. platelet indices are missing in 12–15% of development records but 4–9% at validation hospitals. Missing-indicators help the model but can carry site habits; monitor this in external validation.
* **Training files are down-sampled to 1:2.** Precision and PR-AUC must always be reported together with the cohort's prevalence.

## Cohorts in the cleaned data
| cohort | files | records | leukemia |
|---|---|---|---|
| development | train A, B, C | 26,922 | 8,974 (33.3%) |
| internal_validation | valid A, B, C | 211,611 | 7,818 (3.7%) |
| external_validation | valid D, E, F, G | 149,623 | 5,784 (3.9%) |
| real_world_test | test_site_true_world | 58,481 | 2,290 (3.9%) |
