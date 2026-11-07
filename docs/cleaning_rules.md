# LeukoGuard: Our Data Cleaning Rules

**Dataset:** LeukoAlert public data, folder `leuk_vs_HC` (leukemia vs healthy controls), 11 CSV files, 446,663 records, 74 numeric columns.
**Where the code is:** `src/leuko_common.py`, function `clean_frame()`. Everyone in the group uses this same function, so all our models get exactly the same clean data.
**Result:** 446,637 records and 71 features, plus 6 info columns (`sample_id, source_file, split, site, cohort, y`).

## The idea behind the rules

In cleaning we only used **fixed rules**. Each rule looks at one record at a time, or at one file's recording habit (like which unit it uses for HCT). No rule uses the label or any average calculated from the data. That way it is safe to clean all the files before we split anything.

The steps that *learn* numbers from the data are done later, on the training data only:

| Not done in cleaning | Where we do it instead |
|---|---|
| Filling missing values and adding missing flags | notebook 02b (and inside each CV training fold in Phase 2) |
| Log transform, clipping, scaling | notebook 02b (and inside each CV training fold in Phase 2) |
| SMOTE or class weights | inside the training folds only (Phase 2) |
| Removing outliers | **never**, because the extreme values are mostly leukemia |

## The rules

| ID | Rule | Why (what we saw in the EDA) | How much it changed |
|---|---|---|---|
| R1 | Rename the long column names to short ones (`WBC(10^9/L)` becomes `WBC`), and turn the label into `y` (1 = leukemia, 0 = healthy) | Easier to work with | all |
| R2 | **Drop** the 3 blast flags: `Q-Flag(Blasts/Abn Lympho?)`, `Q-Flag(Blasts?)`, `Q-Flag(Abn Lympho?)` | They are not in the original paper's 72 features, they are filled in for about half to almost all records, they are constant at train B and valid D and G, and they are the machine's own "blasts!" warning, so they basically give away the answer | 3 columns |
| R3 | **Hidden missing values become NaN**: any value with more than 4 decimals | Each file has exactly one such value per column (like `104.2533936651584`), different in every file. These are averages that were filled in before the data was published | 105,552 cells |
| R4 | **HCT unit**: if a file's median HCT is below 1, multiply its HCT by 100 | Sites B and C store HCT as a fraction (0.35), the others as a percent (35) | 167,532 records (sites B and C) |
| R5 | RBC, HGB, HCT, MCV, MCH, MCHC, RDW-SD or RDW-CV equal to 0 becomes NaN | These can't be 0 in a living person, so the measurement failed | 549 cells |
| R6 | If MPV = 0, set PDW, MPV, P-LCR and PCT to NaN | These four are always 0 together, which means the machine didn't calculate them (usually when platelets are very low) | 5,390 records |
| R7 | If NEUT% + LYMPH% + MONO% + EO% + BASO% = 0, set the whole differential block (23 columns) to NaN | No differential was reported, but WBC was still measured in 96.6% of these records, so we keep WBC | 3,856 records |
| R8 | Cell population values (NE-, LY-, MO- positions and widths) equal to 0 become NaN | A position or width on the machine's scatter plot can't be 0, so it means "not measured" | 34,046 cells |
| R9 | MCHC above 500 g/L: set MCHC and MCH to NaN | Physically impossible (the real upper limit is about 370). It happens when red cells stick together and the machine counts too few of them | 139 records |
| R10 | Exact duplicates inside a file: keep the first one. Same values but different labels: drop all copies | 31 duplicate records in valid D and valid G. The valid G ones were empty records filled with averages. No duplicate goes across two files | 22 records removed |
| R11 | Remove records where WBC, HGB and PLT are all missing | There is nothing left to screen | 4 records removed |
| R12 | Data types: features as float64, `y` as int8, site/split/cohort as categories | So NaN works everywhere | none |

## Things we checked but did NOT treat as errors

- **WBC = 0 and PLT = 0.** These are real "too low to measure" results. 47% of the WBC = 0 records are leukemia.
- **Very high WBC (up to 727), IG#, NRBC#, MONO%.** Between 50% and 93% of the extreme outliers (more than 3 × IQR) in these features are leukemia, so they are the disease and not mistakes.
- **NEUT% = 0 or 100 while a differential is there.** This can happen when abnormal cells take over.

## Limitations we know about

- **No patient ID.** `sample_id` is just a row number. If one person had several tests, they could end up in different CV folds, which may make CV scores a little too optimistic. That's why we judge the models on the unseen hospitals (D to G).
- **Missing values depend on the hospital.** For example the platelet indices are missing in 12 to 15% of the training records but only about 5 to 6% at the other hospitals. The missing flags help the model, but they could also teach it hospital habits, so we need to watch this.
- **The training files were down-sampled to 1:2.** So precision and PR-AUC must always be reported together with the prevalence of the cohort.

## Records in each cohort after cleaning

| Cohort | Files | Records | Leukemia |
|---|---|---|---|
| development | train A, B, C | 26,922 | 8,974 (33.3%) |
| internal_validation | valid A, B, C | 211,611 | 7,818 (3.7%) |
| external_validation | valid D, E, F, G | 149,623 | 5,784 (3.9%) |
| real_world_test | test_site_true_world | 58,481 | 2,290 (3.9%) |
