# LeukoGuard: Common Validation Protocol (Member 2, Part A)

Every model (Logistic Regression, Random Forest, XGBoost) follows this protocol, so their results are comparable.
The split file is `data/splits/split_ids.csv`, built by `notebooks/05_Validation_Splits.py`.

## Data and roles

| Role | Records | Leukemia | Hospitals | Used for |
|---|---|---|---|---|
| `development_cv` | 26,922 | 33.3% | A, B, C (train files) | Model fitting, tuning, CV, threshold |
| `internal_validation` | 211,611 | 3.7% | A, B, C (valid files) | Reported only |
| `held_out_hospital` | 149,623 | 3.9% | D, E, F, G | Generalization test (RQ1). Never used for tuning or thresholds |
| `real_world_test` | 58,481 | 3.9% | RW (test file) | Final check only |

Every record appears in exactly one role. Columns: `sample_id, role, cohort, hospital, source_file, y, cv_fold`.

## Patient grouping (limitation)

The public LeukoAlert release has **no patient identifier**. `sample_id` is a unique running row number (1 to 446,663). So the proposal's "keep each patient's records in the same split" can only be applied at the level the publishers did: **separate files per hospital and split**.

Leakage checks (`reports/tables/V1_leakage_checks.csv`), all passed:
- fold file IDs equal the development IDs (26,922);
- every record has exactly one role; no `sample_id` is shared between roles;
- no identical 71-feature CBC profile appears in more than one role, and none repeats within a role.

Repeat CBCs from one person that have different values cannot be detected, so they may sit in different CV folds. CV scores are therefore possibly slightly optimistic. The held-out hospitals (D-G) are the unbiased check.

## Cross-validation

- Stratified 5-fold, shuffled, seed 42. Folds come from `data/cv_folds_development.csv`, created by Member 1, so all models share the same folds.
- Fold balance: 33.32-33.34% leukemia per fold (`reports/tables/V4_development_fold_balance.csv`).
- Imputation, class weights, SMOTE and tuning are fitted inside each training fold only.

## Imbalance

- Class weights (`balanced`), SMOTE (training folds only) or none. Each model picks one by CV PR-AUC, with a 1-SE tie-break toward `balanced`.

## Tuning

- Grid search on mean CV PR-AUC. One-standard-error rule: pick the simplest model within 1 SE of the best.

## Threshold (frozen before any held-out use)

- Highest cut-off with recall >= 0.90 on pooled out-of-fold development predictions. This is the team's recall-first rule.

## Metrics

Recall, precision, F1, specificity, ROC-AUC, PR-AUC and confusion matrices. PR-AUC is always reported with the cohort prevalence, because it depends strongly on prevalence.

## Files each model must produce (for the final comparison)

| File | Contents |
|---|---|
| `reports/tables/<MODEL>_results_at_threshold.csv` | Member 1's `D5` layout (rows: dev CV, internal A/B/C/pooled, held-out D/E/F/G/pooled, RW) |
| `reports/predictions/<model>_oof_development.csv` | `sample_id, fold, y, p_oof` |
| `reports/predictions/<model>_eval_predictions.csv.gz` | `sample_id, cohort, site, y, p` for all held-out records |

Member 1's LR files: `D5_lr_results_at_threshold.csv`, `lr_raw71_oof_development.csv`, `lr_raw71_eval_predictions.csv.gz`.
Member 2's RF files: `RF_results_at_threshold.csv`, `rf_oof_development.csv`, `rf_eval_predictions.csv.gz`.
Member 3's XGBoost files should follow the same pattern as `xgb_*`. `notebooks/08_Final_Model_Comparison.py` picks them up automatically.

## Known environment note

The team's `requirements.txt` pins `scikit-learn==1.9.1`, but Member 2's results were run with scikit-learn 1.8.0. Re-run after installing the pinned version for exact reproduction.
