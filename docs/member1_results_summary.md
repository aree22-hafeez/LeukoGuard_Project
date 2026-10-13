# LeukoGuard: Member 1 Results Summary

## Evaluation procedure (defaults; change in notebook 03's settings cell if the team agreed otherwise)
- **Development:** train_site_A/B/C (26,922 records, 33.3% leukemia). Stratified 5-fold CV, seed 42; folds saved in `data/cv_folds_development.csv`.
- **Held-out hospitals (RQ1):** valid_site_D/E/F/G (149,623 records, 3.9% leukemia), never used for fitting, tuning or threshold choice.
- **Imbalance:** handled inside training folds only (class weights; SMOTE compared).
- **Threshold:** highest cut-off with recall ≥ 0.90 on pooled out-of-fold predictions, then frozen.

## A–B. Data and cleaning
- 446,663 raw records, 11 files, 74 numeric columns → **446,637 records × 71 features** after 12 deterministic rules (`docs/cleaning_rules.md`).
- Main issues found: **~105k disguised missing values** (per-file mean fill-ins); zero-coded missing blocks; **HCT stored as a fraction at sites B and C**; impossible MCHC > 500; 22 duplicates; 3 blast flags that are mostly missing and act as a label shortcut.
- Missingness is informative: no differential reported → 31% leukemia vs a 5.6% baseline.
- Strongest univariate differences: PLT ↓, RBC/HCT/HGB ↓, neutrophils and WBC ↓, LYMPH% ↑, RDW ↑ (Cliff's δ up to −0.73).

## C. Derived features
NLR, PLR, MLR, SII, dNLR (ε = 0.01 in denominators; no infinities). Count-based NLR = %-based NLR (ρ = 1.000).
On their own, none beats its best component (e.g. PLT AUC 0.86 vs SII 0.81). NLR and dNLR are near rank-copies of LYMPH% and NEUT% (|ρ| ≥ 0.98).

## D. Logistic Regression (final: L2, C = 0.01 by the 1-SE rule, class weights, 71 features + missing indicators, threshold 0.511)

| Cohort | n | Prevalence | Recall | Precision | Specificity | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Development 5-fold CV (mean ± SD) | 26,922 | 33.3% | 0.900 ± 0.006 | 0.868 | 0.931 | 0.968 ± 0.003 | 0.930 ± 0.007 |
| Internal validation A–C (pooled) | 211,611 | 3.7% | 0.898 | 0.287 | 0.915 | 0.962 | 0.491 |
| **Unseen hospital D** | 30,156 | 4.4% | 0.918 | 0.234 | 0.862 | 0.954 | 0.526 |
| **Unseen hospital E** | 94,318 | 2.7% | 0.878 | 0.266 | 0.933 | 0.965 | 0.521 |
| **Unseen hospital F** | 20,179 | 4.9% | 0.875 | 0.396 | 0.931 | 0.964 | 0.563 |
| **Unseen hospital G** | 4,970 | 18.6% | 0.853 | 0.557 | 0.845 | 0.912 | 0.638 |
| **Unseen hospitals pooled** | 149,623 | 3.9% | **0.883** | 0.297 | **0.916** | **0.960** | 0.529 |
| Real-world test | 58,481 | 3.9% | 0.935 | 0.315 | 0.917 | 0.967 | 0.478 |

- **RQ1 (LR part):** discrimination transfers well (ROC-AUC 0.95–0.97 at D/E/F, 0.91 at G). The frozen 90%-recall threshold gives 85–92% recall at the new hospitals, so a local threshold check is advisable.
  At real prevalence about 1 in 3–4 flags is leukemia, and about 12% of CBCs are flagged.
- Imbalance method barely matters for LR: none, class weights and SMOTE give the same ROC-AUC, PR-AUC and specificity@90% recall; they only shift the threshold.

## E. RQ3: do derived features help?
| Setting | ΔROC-AUC | ΔPR-AUC | ΔSpecificity @ 90% recall | Unseen hospitals |
|---|---|---|---|---|
| Full 71 features (main LR) | +0.0006* | +0.0011* | +0.0024* | ΔPR-AUC −0.0004 [−0.002, +0.002], p = 0.70 |
| Routine 24-value CBC | +0.0048* | +0.0089* | +0.0131* | ΔROC-AUC +0.0036 (DeLong p < 10⁻²⁵), ΔPR-AUC +0.017 [0.012, 0.020] |
| Routine CBC, no log transform | +0.0142* | +0.0271* | +0.0372* | — |

\* 5×5 repeated CV, Nadeau–Bengio corrected t-test, Holm-adjusted p < 0.05.

**Answer:** the derived ratios help **only when the model lacks the information they summarise**. There is a meaningful gain on a standard 5-part-differential CBC (relevant for typical Pakistani labs) and no practical gain with the full 71-parameter analyser panel or with log-transformed counts.

## E. LR error analysis (unseen hospitals)
- Missed leukemias have near-normal CBCs (median WBC 5.0, HGB 102, PLT 169). Recall is 97% when HGB, PLT and WBC are all abnormal vs 16% when none are.
- A linear model under-uses leukocytosis: recall 0.78 for WBC 11–30 and 0.66 for normal WBC, vs 0.94 for low WBC. This is the subgroup where tree models should be compared.
- False alarms are sick non-leukemia patients (median HGB 87, PLT 106).

## E. Odds ratios (readable 8-feature LR, per 1 SD, 95% bootstrap CI, all VIF ≤ 2.1)
PLT 0.28 (0.26–0.30) · HGB 0.50 (0.47–0.53) · WBC 0.51 (0.48–0.54) · MPV 0.71 (0.68–0.74) · LYMPH% 1.38 (1.30–1.45) · MCV 1.40 (1.34–1.46) · MONO% 1.63 (1.56–1.71) · RDW-CV 1.78 (1.71–1.86).
Adding NLR next to its components flips its sign (suppression, VIF ≈ 5), so ratio ORs must be read conditionally.

## Limitations
No patient ID (repeat CBCs may cross CV folds); the "healthy" class is non-leukemia hospital patients; prevalence differs between training (33%) and practice (2–5%); the readable model's ORs are associational, not causal.
