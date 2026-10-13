# LeukoGuard: Member 1 (Data Preparation + Logistic Regression + RQ3)

All five parts are complete. Open the notebooks **in order**; each one is saved with its outputs, so you can read them without running anything.

| Part | Notebook | What it contains |
|---|---|---|
| A + B | `notebooks/01_EDA_and_Cleaning.ipynb` | EDA, data-quality problems found, 12 fixed cleaning rules, cleaned data |
| C | `notebooks/02_Feature_Engineering.ipynb` | NLR, PLR, MLR, SII, dNLR + validity and usefulness checks |
| D | `notebooks/03_Logistic_Regression.ipynb` | LR: 5-fold CV, tuning, imbalance comparison, 90%-recall threshold, unseen hospitals D–G |
| E | `notebooks/04_RQ3_Derived_Features.ipynb` | RQ3 statistics, LR error analysis, coefficients and odds ratios |

**For the report:** `docs/member1_results_summary.md` (key numbers + answers) and `docs/cleaning_rules.md` (cleaning table). All figures are in `reports/figures/`, all tables in `reports/tables/`.

## Folder layout
```
data/raw/                       11 original CSVs (put them here; not included in the zip)
data/cleaned/                   created by notebook 01
data/features/                  created by notebook 02
data/cv_folds_development.csv   the shared 5 CV folds (sample_id -> fold) - give to Members 2 & 3
src/leuko_common.py             feature lists, cleaning rules, derived features (imported by all notebooks)
src/lr_pipeline.py              LR pipeline, SMOTE, metrics, threshold, DeLong / corrected t-test / bootstrap / McNemar
models/lr_final.joblib          final LR model + its frozen threshold
reports/predictions/            LR out-of-fold and validation predictions (for comparing with RF/XGBoost)
reports/figures, reports/tables all outputs
docs/                           cleaning rules + results summary
```

## Re-running (Google Colab)
1. Upload the whole `LeukoGuard` folder to Google Drive (`MyDrive/LeukoGuard`) and put the 11 CSVs in `MyDrive/LeukoGuard/data/raw/`.
2. Open a notebook in Colab and add this as the **first cell**:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   import os
   os.chdir("/content/drive/MyDrive/LeukoGuard")   # project root
   ```
3. Run notebooks 01 → 02 → 03 → 04 (about 3 + 2 + 5 + 9 minutes).
Requirements: pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, joblib (all preinstalled in Colab). No imbalanced-learn needed: SMOTE is implemented in `src/lr_pipeline.py`.

## For teammates (same data, same folds)
```python
import sys; sys.path.insert(0, "src")
import pandas as pd
from leuko_common import FEATURES, DERIVED, CORE_FEATURES
fe    = pd.read_csv("data/features/leukoalert_features.csv.gz")   # cleaned + derived features, all cohorts
dev   = fe[fe.cohort == "development"]                             # train A,B,C
folds = pd.read_csv("data/cv_folds_development.csv")               # identical 5 folds for every model
ext   = fe[fe.cohort == "external_validation"]                     # unseen hospitals D-G: never use for fitting
# NaN are left in on purpose -> impute inside your pipeline (XGBoost handles NaN natively)
```
