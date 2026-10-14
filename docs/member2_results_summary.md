# LeukoGuard: Member 2 Results Summary (Random Forest, RQ1, model comparison)

Protocol: `docs/validation_protocol.md`. Split IDs: `data/splits/split_ids.csv`. Notebooks: `05`-`08` in `notebooks/`.

## A. Validation setup
- Development: train files A, B, C (26,922 records, 33.3% leukemia). Stratified 5-fold CV, seed 42, folds reused from Member 1.
- Held-out hospitals: D, E, F, G (149,623 records, 3.9% leukemia), never used for fitting, tuning or thresholds.
- Internal check: valid files A, B, C (211,611 records). Real-world test: 58,481 records.
- All leakage checks passed (`reports/tables/V1_leakage_checks.csv`). The release has **no patient ID**, so splits are record-level, as documented.

## B. Random Forest
- Input: the 71 raw CBC features (main model, as agreed). Balanced class weights (SMOTE was within 1 SE, so it was not chosen).
- Tuning: 18 configurations on 5-fold CV PR-AUC, 1-SE rule. Selected: `max_depth=16`, `min_samples_leaf=1`, `max_features='sqrt'`, 300 trees (`reports/tables/RF_tuning.csv`).
- Frozen threshold: **0.528** (recall >= 0.90 on pooled development OOF).

| Cohort | n | Prevalence | Recall | Precision | Specificity | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Development CV (OOF) | 26,922 | 33.3% | 0.900 | 0.919 | 0.961 | 0.982 | 0.960 |
| Internal A-C (pooled) | 211,611 | 3.7% | 0.880 | 0.383 | 0.946 | 0.970 | 0.559 |
| **Held-out D-G (pooled)** | 149,623 | 3.9% | 0.844 | 0.437 | 0.956 | **0.972** | **0.619** |
| Real-world test | 58,481 | 3.9% | 0.914 | 0.387 | 0.941 | 0.974 | 0.555 |

Per hospital (D-G): ROC-AUC 0.959 / 0.978 / 0.969 / 0.933; PR-AUC 0.586 / 0.631 / 0.572 / 0.727 (D / E / F / G).

Feature importance (`reports/figures/RF_feature_importance.png`): on held-out D-G, permutation importance ranks the white-cell counts first (WBC-N, TNC-N, NEUT#&, WBC, NEUT#), followed by PLT. Gini importance is in `RF_gini_importance.csv`.

## C. RQ1: generalization gap
Full table: `reports/tables/RQ1_generalization_gap.csv` (bootstrap 95% CIs).

- **ROC-AUC barely drops.** RF: 0.982 (CV) to 0.972 (held-out pooled), a drop of 0.010. LR: 0.968 to 0.960, a drop of 0.008.
- **PR-AUC drops a lot, and most of this is prevalence.** RF: 0.961 to 0.619. Matching the prevalence to 33% by subsampling non-leukemia records (`reports/tables/RQ1_prevalence_matched_pr.csv`) gives PR-AUC 0.938 at D-G pooled. That explains about **93%** of the drop. The remaining gap is 0.023 (LR 0.017).
- **Hospital G is the exception.** It has 18.6% leukemia, and its ROC-AUC is lowest (0.933 for RF, 0.912 for LR). Its prevalence-matched gap is still 0.114.
- **Feature shift is modest and mostly in healthy controls.** Within-class standardized differences (`reports/tables/RQ1_feature_shift_smd.csv`) are largest for MCV and MONO% at several hospitals, and for WBC and NEUT# at hospital G among non-leukemia patients (SMD about -0.3). The missing-value rates also differ by hospital: platelet indices are missing in 13.4% of development records but 3.8-8.4% at D-F (`RQ1_missingness_by_cohort.csv`).
- **Limitation of this explanation:** Member 1 noted that the "healthy" label covers non-leukemia hospital patients, so the case mix of negatives may differ across sites. The data cannot separate this from site effects.

**Answer to RQ1 (RF part):** routine CBC features discriminate well at an unseen hospital (ROC-AUC 0.97). Recall at the frozen threshold drops from 0.90 to 0.84, so a local threshold check is advisable before use at any new site.

## D. Final comparison (LR, RF; XGBoost pending)
Table: `reports/tables/FINAL_model_comparison_headline.csv`. Per-hospital: `FINAL_per_hospital_pivot.csv`. Figure: `reports/figures/FINAL_model_comparison.png`.

| Model | Held-out D-G recall | Specificity | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.883 | 0.916 | 0.960 | 0.529 |
| Random Forest | 0.844 | 0.956 | 0.972 | 0.619 |
| XGBoost | pending (Member 3) | | | |

Paired tests on D-G (`FINAL_pairwise_tests_heldout_DG.csv`):
- ROC-AUC: RF higher than LR, DeLong p < 0.001.
- PR-AUC: RF higher by 0.090 (95% bootstrap CI 0.078 to 0.100; p reported as 0.002, which is the floor for 500 resamples).
- Decisions at each model's own threshold (McNemar): RF is right where LR is wrong in 6,604 records, and the reverse happens in 1,031 (p < 0.001). RF makes fewer errors overall, but it misses more leukemias (lower recall). This trade-off should be reported explicitly.

## Limitations
- No patient identifiers: CV may be slightly optimistic (repeat CBCs can cross folds).
- Development prevalence is 33% and held-out prevalence is about 4%; PR-AUC is only directly comparable after prevalence matching.
- XGBoost rows and its comparison are missing until Member 3's results files are added.
- The RF was run with the installed scikit-learn 1.8.0, while `requirements.txt` pins 1.9.1. Re-run to reproduce exactly.
- Screening aid only, not a diagnostic tool.
