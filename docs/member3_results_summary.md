# Member 3 results and RQ2

## XGBoost evaluation

The model uses the 71 cleaned CBC measurements and a binary indicator for each measurement with missing training values. Median imputation, missing indicators, weighting and optional SMOTE were fitted inside each training fold. The five shared development folds were used for tuning, imbalance selection and threshold selection. Internal validation, held-out hospitals D–G and the real-world test set were evaluated after those choices were frozen. See the [common validation protocol](validation_protocol.md).

The untuned baseline had pooled out-of-fold ROC-AUC **0.9784** and PR-AUC **0.9540**. The 16-configuration sweep selected depth 5, 300 trees, a learning rate of 0.10 and a minimum child weight of 5 under the one-standard-error rule. At that configuration, the unweighted model had the highest mean fold PR-AUC (0.97319), with class weighting close enough (0.97291) to retain it under the team's tie-break rule. The frozen decision threshold was **0.75384992**, which gave 90.0% pooled development recall.

| Cohort | Records | Prevalence | ROC-AUC | PR-AUC | Recall | Precision | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development, pooled out-of-fold | 26,922 | 33.3% | 0.9874 | 0.9727 | 90.0% | 94.3% | 97.3% |
| Internal A–C, pooled | 211,611 | 3.7% | 0.9639 | 0.4923 | 85.9% | 34.2% | 93.6% |
| Held-out D–G, pooled | 149,623 | 3.9% | 0.9682 | 0.5798 | 82.9% | 43.1% | 95.6% |
| Real-world test | 58,481 | 3.9% | 0.9752 | 0.5590 | 90.0% | 40.4% | 94.6% |

On D–G, the frozen threshold flagged 4,794 leukemia records and missed 990. It also flagged 6,330 non-leukemia records. Recall ranged from 81.0% at site E to 85.2% at site G. The full per-site and per-fold results, confusion matrices and comparison with Logistic Regression and Random Forest are available below.

- [Baseline](../reports/tables/XGB_baseline_cv.csv), [tuning](../reports/tables/XGB_tuning.csv), [imbalance comparison](../reports/tables/XGB_imbalance_comparison.csv) and [CV folds](../reports/tables/XGB_cv_fold_results.csv)
- [Full evaluation table](../reports/tables/XGB_results_at_threshold.csv), [ROC and precision–recall curves](../reports/figures/XGB_roc_pr.png) and [confusion matrices](../reports/figures/XGB_confusion_matrices.png)
- [Three-model comparison](../notebooks/08_Final_Model_Comparison.ipynb) and [held-out paired tests](../reports/tables/FINAL_pairwise_tests_heldout_DG.csv)

On the same D–G records, XGBoost ranked between Logistic Regression and Random Forest on both ROC-AUC and PR-AUC. Its PR-AUC was 0.051 higher than Logistic Regression (paired bootstrap 95% CI 0.039 to 0.063) and 0.039 lower than Random Forest (95% CI 0.028 to 0.049 lower). XGBoost had the lowest recall at the models' frozen thresholds (82.9% versus 88.3% for Logistic Regression and 84.4% for Random Forest). These thresholds were selected separately on development data.

## RQ2 — Which CBC measurements drive predictions, and do they stay important across folds?

Tree SHAP explains the fitted model's output in log-odds of the leukemia class. A positive contribution raises the model score relative to its SHAP baseline. A negative contribution lowers it. Applying the logistic function to the baseline plus all signed contributions reconstructs the model score. The [global analysis](../reports/tables/XGB_shap_global.csv) uses a fixed random sample of 4,000 D–G records (159 leukemia). The [beeswarm](../reports/figures/XGB_shap_beeswarm.png) shows individual model inputs. The [CBC importance chart](../reports/figures/XGB_shap_global.png) and fold rankings add each measurement's missingness contribution to its measurement contribution before taking the mean absolute value.

| Measurement | Mean absolute SHAP, log-odds | Direction on the held-out sample |
|---|---:|---|
| Platelet count (PLT) | 0.750 | The lowest decile (median 47) averaged +0.753. The highest decile (median 391) averaged −0.762. |
| Microcytic red cell measure (MicroR) | 0.388 | The low and high observed quartiles averaged −0.822 and +0.162, respectively. |
| Red cell distribution width (RDW-CV) | 0.379 | The low and high observed quartiles averaged −0.669 and +0.286. |
| Lymphocyte analyser channel (LY-X) | 0.365 | The low and high observed quartiles averaged −0.502 and +0.094. |
| Neutrophil analyser channel (NEUT%&) | 0.358 | The low and high observed quartiles averaged +0.385 and −0.609. |
| Red blood cell count (RBC) | 0.316 | The low and high observed quartiles averaged +0.204 and −0.335. |

Platelets had almost twice the mean absolute contribution of the next measurement. Very low platelet counts raised the score on average, while higher counts generally lowered it. Low WBC also raised the score in the lowest decile (mean SHAP +0.276 at median WBC 2.44). The remaining deciles had negative mean contributions in this sample. The [direction profiles](../reports/figures/XGB_shap_direction.png) and [quartile table](../reports/tables/XGB_shap_direction.csv) summarize how contributions varied with observed values. They are descriptive averages over the records in each band.

### Individual predictions

The script chose cases near the median model score of each held-out confusion category. All four examples contain positive and negative contributions. Scores are compared with the frozen threshold of 0.7538.

| Example | Score | How the contributions support the prediction |
|---|---:|---|
| [True positive, E](../reports/figures/XGB_shap_waterfall_TP.png) | 0.983 | Low WBC (1.59, +0.478) and MacroR (+0.407) helped raise the score. PLT lowered it (−0.405). |
| [True negative, E](../reports/figures/XGB_shap_waterfall_TN.png) | 0.011 | NEUT#& (−0.707), NEUT%& (−0.691) and PLT (−0.678) pulled the score down despite a positive NE-WZ contribution (+0.497). |
| [False positive, E](../reports/figures/XGB_shap_waterfall_FP.png) | 0.919 | NE-WX (+0.649), RDW-CV (+0.500) and NEUT%& (+0.432) helped push the score above the threshold despite PLT (−0.564). |
| [False negative, F](../reports/figures/XGB_shap_waterfall_FN.png) | 0.407 | NE-WX strongly raised the score (+1.524). PLT (−0.896), NE-SSC (−0.511), LY-X (−0.432) and other negative effects kept the score below the threshold. |

[Case metadata](../reports/tables/XGB_shap_examples.csv) and [signed contributions](../reports/tables/XGB_shap_example_contributions.csv) contain the full decompositions.

### Stability and interpretation

PLT ranked first in all five fold models. Nine measurements appeared in every fold's top ten: **PLT, RDW-CV, MicroR, LY-X, NEUT%&, RBC, NE-WX, LY-WZ and Fragments**. Fold pairs shared 9–10 of their top ten measurements. Their Spearman correlations across all 71 ranks ranged from **0.913 to 0.961**. Comparing the final model with each fold gave correlations of **0.958–0.976**. WBC appeared in only one fold's top ten and ranked 20th in the final model, so the consistency is strongest for the leading measurements. See the [feature-level stability table](../reports/tables/XGB_shap_stability.csv) and [model-pair comparisons](../reports/tables/XGB_shap_stability_pairs.csv).

The fold models share training records, and the stability comparison uses the same D–G sample for every model. This measures sensitivity to the training fold. Stability at each hospital or on a different analyser would require a separate assessment. Correlated measurements can share attribution, and several highly ranked measurements are analyser-specific. The global and local results explain the fitted research model's behavior. Causal or clinical conclusions would require additional evidence.
