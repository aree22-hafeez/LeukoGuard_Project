# Phase 1 Insights: What We Learned From the Data

This file collects everything we learned in Phase 1 (notebooks 01, 02 and 02b) in one place. All numbers come from the saved notebook outputs and the tables in `reports/tables/`.
Unless we say otherwise, the EDA numbers are from the **development set** (the training files of sites A, B and C, 26,922 records), because we didn't want to look at the test hospitals before modelling.

## 1. The size of the data

- 446,663 raw records in 11 files from 7 hospitals. After cleaning, 446,637 records (26 removed).
- 74 numeric columns in the raw files. We dropped 3 blast flags, so 71 features are left. In notebook 02 we added 5 ratio features, so 76 in total.
- The target is `leukemia_label`, which we turned into `y` (1 = leukemia, 0 = healthy).

## 2. Class imbalance

- Overall only 5.6% of the records are leukemia (24,871 out of 446,663).
- But every training file is exactly 33.3% leukemia (1 leukemia for every 2 healthy). The dataset makers must have down-sampled them on purpose.
- The validation and test files have the real-life rates: 1.8% (valid B) to 4.9% (valid F), and site G has 18.7%.

**Why this matters for modelling:** the imbalance in training is mild, so class weights or SMOTE (only on training folds) should be enough. But precision and PR-AUC will drop a lot on the test hospitals just because leukemia is rarer there, so we always have to report the prevalence next to them. Recall should carry over better, so we pick the threshold based on recall (at least 90%), because missing a leukemia patient is worse than a false alarm.

## 3. Data quality problems

The data looked perfect (zero empty cells), but it had a lot of hidden problems:

| Problem | How many | What we did |
|---|---|---|
| Filled-in averages instead of real values | about 105,000 cells | turned back into NaN |
| "Not measured" written as 0 (differential, platelet indices, cell population blocks) | 3,856 records had no differential, 5,390 no platelet indices, and about 34,000 cell population values were 0 | turned into NaN |
| HCT stored as a fraction at sites B and C | all records of B and C | multiplied by 100 |
| MCHC above 500 g/L (impossible) | 139 records | MCHC and MCH set to NaN |
| RBC panel equal to 0 | 15 records | set to NaN |
| Duplicates and conflicting copies | 22 records | removed |
| No WBC, HGB and PLT at all | 4 records | removed |
| Blast flags mostly filled in or constant | 3 columns | dropped |

The biggest lesson for us: **"no missing values" does not mean the data is complete.** We only found the filled-in averages because one value had 13 decimal places, which a blood machine never gives.

## 4. Missing values are useful information

After cleaning, the missing values are honest NaN, and they are not random at all:

- When the white cell differential is missing, **92.4%** of those training records are leukemia, compared to 32.1% when it is there.
- When the platelet indices are missing, 71.4% are leukemia, compared to 27.4%.
- In the whole dataset, records without a differential are 31.4% leukemia, while the overall rate is 5.6%.

The reason is that the machine often can't sort abnormal cells into the normal groups, so it doesn't report a differential. For modelling this means we should not just fill the gaps. We added three 0/1 missing flags in notebook 02b so the model can use this.
But we also have to be careful, because missing values also depend on the hospital (platelet indices are missing in 12 to 15% of training records but only about 5 to 6% at the other hospitals), so a model could learn hospital habits instead of the disease.

## 5. Distributions

- The count features (WBC, NEUT#, LYMPH#, MONO#, EO#, BASO#, IG#, NRBC# and the ratios) are **very right-skewed**. The median skewness of the 34 features we log-transformed was about 21. After `log1p` it was about 2.
- 14 of the 34 logged features still have a skewness above 2. Most of them are tiny counts like BASO#, HFLC#, BA-N# and NRBC#: their normal values are around 0.02, and the log barely changes such small numbers. Clipping in notebook 02b takes care of their extreme values.
- HGB, RBC, HCT, MCV and MCHC look more like normal bell shapes and didn't need a log.

## 6. Outliers

- If we use the 3 × IQR rule, many features have "extreme outliers", for example 8.5% of IG# values and 7.5% of NRBC# values.
- But most of them are leukemia patients: 72.3% of the extreme WBC values, 82.9% of the extreme MONO# values and 93.3% of the extreme MONO% values are leukemia (while only 33.3% of the training records are leukemia).
- So we did **not** delete outliers. In notebook 02b we clipped at the 0.5% and 99.5% percentiles instead, which changed less than 1% of the values and kept every record. 72.6% of the clipped values were from leukemia records.

## 7. Leukemia vs healthy

Using Cliff's delta (how much the two groups overlap), 23 of the 71 features have a large difference. The biggest ones:

| Feature | Healthy median | Leukemia median | Direction in leukemia |
|---|---|---|---|
| PLT | 208 | 38 | much lower |
| RBC | 3.85 | 2.56 | lower |
| HCT | 35.2 | 23.9 | lower |
| HGB | 115 | 79 | lower |
| NEUT# | 5.01 | 1.05 | lower |
| WBC | 7.28 | 1.94 | lower |
| LYMPH# | 1.23 | 0.45 | lower |
| RDW-CV | 13.3 | 15.2 | higher |
| LYMPH% | 17.8 | 27.5 | higher |

This matches what medical papers say about leukemia: the bone marrow is filled with leukemia cells, so it makes fewer normal platelets, red cells and neutrophils.

One thing to remember: **WBC is lower on average, but it is abnormal in both directions.** Some leukemia patients have extremely high WBC (up to 727). A Logistic Regression can only push the risk in one direction for each feature, so it might not use those high-WBC cases well, while trees can.

## 8. Correlations

- 28 pairs of features have a Spearman correlation of 0.95 or more, and 46 pairs are above 0.90.
- Many are the same thing measured in different ways: WBC, TNC, WBC-N, TNC-N, WBC-D and TNC-D, or PLT and PLT-F. HGB, RBC and HCT are also very close (0.94 to 0.98).
- NEUT% and LYMPH% are strongly negative (−0.91): when one goes up the other goes down, because they are parts of the same 100%.

**For modelling:** Logistic Regression coefficients become unstable with so many near copies, so we need regularisation, and when we explain odds ratios we should use a smaller set of features. Random Forest and XGBoost are not really affected.

## 9. Ratio features (NLR, PLR, MLR, SII, dNLR)

- They are calculated correctly (NLR from counts and NLR from percentages agree with ρ = 1.0) and have no infinite or negative values.
- All five are **lower** in leukemia.
- None of them is better than its best part on its own. SII is the best ratio (AUC 0.81), but PLT alone has 0.86.
- NLR and dNLR are almost copies of LYMPH% and NEUT% (ρ = 0.98 and 0.998).
- After a log transform, 80 to 98% of each ratio can be rebuilt from the raw features with a simple linear regression. So we expect the ratios to help only a little in a log-based Logistic Regression. We will test this in Phase 2 (RQ3).
- Only 37 to 60% of our "healthy" records are inside the normal adult ranges for these ratios. That's because the healthy group is hospital patients without leukemia, not healthy volunteers, so many probably have infections or inflammation.

## 10. Differences between hospitals

- The healthy patients are different at each training site, for example median LYMPH% is 14.7 at site A but 22.2 at site B, and median HGB is 115 at A but 123 at B.
- Hospitals also record things differently (HCT units, which blocks are missing, how often the differential is skipped).
- After scaling, the mean of HGB and PLT in the other cohorts is about +0.3 to +0.5 instead of 0. Most of this is because those hospitals have far fewer leukemia patients (about 4% instead of 33%), and a smaller part is the hospital differences.

**For modelling:** a model trained on A, B and C has to deal with different patients and different recording habits at a new hospital. This is why we test on hospitals D, E, F and G, which are never used for training or tuning.

## 11. Things to keep in mind for Phase 2

1. Learn every preprocessing number (clip limits, medians, mean and std) on the training folds only.
2. Keep the missing flags, but check the model isn't just learning hospital habits.
3. Log-transform the count features for Logistic Regression.
4. Use regularisation because of the near-duplicate features.
5. Handle the 1:2 imbalance with class weights or SMOTE inside the training folds.
6. Choose the threshold for at least 90% recall and report precision together with the prevalence.
7. Judge the final model on the unseen hospitals, because CV scores might be a bit too optimistic (the data has no patient ID).
