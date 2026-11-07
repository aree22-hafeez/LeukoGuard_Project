# LeukoGuard: Explainable Leukemia Risk Prediction

LeukoGuard is a machine learning-based screening aid that uses Complete Blood Count (CBC) parameters to estimate leukemia risk. It uses Logistic Regression and explainable AI techniques to help users understand which blood parameters influence a model's prediction.

**Note:** LeukoGuard is an academic project and is not intended to diagnose leukemia or replace professional medical advice.

## Problem Statement

Leukemia often changes the blood counts, for example very low platelets, low hemoglobin, or a WBC count that is far too low or far too high. A CBC is cheap and almost every hospital patient gets one, but to confirm leukemia you need a bone marrow test and other special tests. Many studies show that machine learning on CBC values can help with screening, but most of them only report accuracy and don't explain *why* a patient was flagged.

**Our question:** using only routine CBC values, can we predict if a patient has leukemia or is a healthy control, in a way that is explainable and still works at hospitals the model has never seen?

This is a **binary classification** problem: leukemia (1) vs healthy (0).

## Features

- CBC-based leukemia risk prediction
- Machine learning using Logistic Regression
- Explainable predictions to highlight influential blood parameters
- Data cleaning and feature engineering
- Model evaluation using multiple validation datasets
- Web interface for entering CBC values and viewing predictions

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- SHAP / model explainability techniques
- Flask or FastAPI (depending on the implementation in `app/main.py`)
- HTML, CSS, and JavaScript
- Jupyter Notebook

## Dataset

LeukoGuard uses the LeukoAlert dataset for CBC-based leukemia risk prediction research.
1. LeukoAlert Dataset
Dataset title: LeukoAlert-project: Transforming routine blood counts into a leukemia detection system with interpretable AI
Author: Shilong Liu
Source: Mendeley Data — LeukoAlert
DOI: 10.17632/vc7kwnyppz.1
Version: 1
License: Creative Commons Attribution 4.0 International (CC BY 4.0)

The dataset contains de-identified clinical records with Complete Blood Count (CBC) parameters. It is used for data cleaning, feature engineering, model training, and evaluation.

### Dataset details

| | |
|---|---|
| Download link | [doi.org/10.17632/vc7kwnyppz.1](https://doi.org/10.17632/vc7kwnyppz.1) |
| Folder we use | `leuk_vs_HC` (leukemia vs healthy controls) |
| Size | 446,663 records in 11 CSV files (about 180 MB), from 7 hospitals |
| Columns | `sample_id`, 74 numeric CBC features, and the label |
| Target variable | `leukemia_label` (`leukemia` / `healthy`), we turn it into `y` = 1 / 0 |
| Features after cleaning | 71 |

The raw files are in `data/raw/` in this repo. If they are ever missing, download them from the link above and put the 11 CSV files in `data/raw/`.

**How the files are split.** Each file is one hospital ("site") in one split:

| Files | Records | Leukemia | What we use them for |
|---|---|---|---|
| `train_site_A/B/C` | 26,922 | 33.3% | Development set: training, cross-validation, tuning |
| `valid_site_A/B/C` | 211,611 | 3.7% | Internal validation (same hospitals, new patients) |
| `valid_site_D/E/F/G` | 149,623 | 3.9% | External validation: hospitals never used in training |
| `test_site_true_world` | 58,481 | 3.9% | Real-world test set |

(These counts are after cleaning.)

**What the features are.** The 71 features come in 7 groups:
- Q-flags (8): warning scores from the analyser, like "Left Shift?" or "Iron Deficiency?", from 0 to 300
- WBC counts (8): WBC, NEUT#, LYMPH#, MONO#, EO#, BASO#, IG#, NRBC#
- Cell percentages (7): NEUT%, LYMPH%, MONO%, and so on
- Red cells (10): RBC, HGB, HCT, MCV, MCH, MCHC, RDW and others
- Platelets (6): PLT, PLT-F, PDW, MPV, P-LCR, PCT
- Extra WBC channels (15): other ways the machine counts white cells (TNC, WBC-N, WBC-D, ...)
- Cell population data (17): how the cells look to the machine's laser (NE-SSC, LY-X, MO-Y, ...)

We also made 5 new ratio features in notebook 02: NLR, PLR, MLR, SII and dNLR.

## Phase 1: Data Cleaning, EDA and Preprocessing

### 1. Data cleaning (notebook 01)

The data looked complete at first (zero empty cells), but it wasn't. These are the main problems we found and how we fixed them:

- **Hidden missing values.** About 105,000 values were not real measurements but averages that had been filled in before the data was published (values like `104.2533936651584`, which no blood machine reports). Each file had its own fill value. We turned them back into NaN.
- **Missing values written as 0.** Some hospitals wrote 0 for "not measured", in blocks (the whole differential, the platelet indices, the cell population values). We turned these zeros into NaN.
- **HCT in two units.** Sites B and C stored HCT as a fraction (0.35) and the others as a percent (35). We multiplied B and C by 100.
- **Impossible values.** 139 records had MCHC above 500 g/L, which is physically impossible, and 15 records had RBC/HGB/MCV = 0. These values became NaN (the rest of the record was kept).
- **Duplicates.** 22 duplicate or conflicting records were removed, and 4 records with no WBC, HGB and PLT at all were removed.
- **Useless columns.** 3 "blast" flag columns were dropped because they were mostly missing or constant, and they basically give away the answer.
- **Outliers were kept on purpose.** Very high or very low values (like WBC = 727) are mostly leukemia patients, so deleting them would delete the disease signal.

Result: **446,637 records and 71 features**, saved in `data/cleaned/`. All 12 rules are listed in `docs/cleaning_rules.md` and coded in `src/leuko_common.py`.

### 2. EDA (notebook 01, sections 8 to 12)

Histograms, box plots, correlation heatmaps, a feature-by-feature comparison of leukemia vs healthy, and a check of how different the hospitals are. The EDA only uses the training files, so we don't look at the test hospitals before modelling.

### 3. Feature engineering (notebook 02)

We added NLR, PLR, MLR, SII and dNLR, checked they are calculated correctly, and checked if they are useful.

### 4. Preprocessing and scaling (notebook 02b)

Yes, our data needed it. The features are on very different scales (PLT is in the hundreds, BASO# is about 0.02) and many are very skewed. In order:
1. log1p transform on 34 skewed count features
2. clip each feature to its 0.5% and 99.5% percentile (instead of deleting outliers)
3. add 3 missing flags (differential, platelet indices, cell population) and fill the remaining gaps with the median
4. standard scaling (z-score)

All the numbers in steps 2 to 4 are learned **only from the training files** and then applied to the other hospitals, so no information from the test data leaks in. The result is in `data/processed/`.

## Key Findings

1. **Class imbalance.** The training files are exactly 1 leukemia to 2 healthy (33.3%), so they were down-sampled. The other hospitals have only 1.8% to 4.9% leukemia (site G 18.7%). So precision and PR-AUC will look much lower on the test hospitals, and we must always report the prevalence with them.
2. **Missing values tell a story.** When the white cell differential is missing, 92% of those training records are leukemia (vs 32% when it is present). The machine often can't classify abnormal cells. That is why we kept missing flags.
3. **Strongest differences.** In leukemia, platelets (PLT), red cells (RBC, HGB, HCT), neutrophils and WBC are much **lower**, and LYMPH% and RDW are **higher**. Platelets are number 1 (median 38 in leukemia vs 208 in healthy).
4. **WBC goes both ways.** Leukemia patients have a lower WBC on average (median 1.9 vs 7.3), but some have extremely high WBC. A linear model can only push risk in one direction, so this is harder for Logistic Regression than for trees.
5. **Outliers are the disease.** 72% of the extreme WBC values and 93% of the extreme MONO% values belong to leukemia records.
6. **Lots of near-duplicate features.** 28 pairs of features have a correlation of 0.95 or more (for example WBC and its other channels, PLT and PLT-F, HGB and HCT). This matters for explaining Logistic Regression coefficients.
7. **Ratios are not better on their own.** None of the 5 ratios separates the classes better than its best part (SII is closest with AUC 0.81, PLT alone has 0.86). After a log transform they are mostly already "inside" the raw features. We test them properly in Phase 2 (RQ3).
8. **Hospitals are different.** Even the healthy patients look different from site to site (for example median LYMPH% 14.7 at site A vs 22.2 at site B), and the hospitals record missing values differently. This is why testing on the unseen hospitals D to G is important.
9. **Our "healthy" group is not fully healthy.** Only 37 to 60% of them have NLR, PLR, MLR, SII and dNLR in the normal adult range. They are hospital patients without leukemia, not healthy volunteers.

A longer version with all the numbers is in [`docs/phase1_insights.md`](docs/phase1_insights.md). Notes on every graph are in [`reports/figures/README.md`](reports/figures/README.md).

## Project Structure

```text
LeukoGuard/
├── app/
│   ├── main.py
│   └── static/
│       ├── index.html
│       ├── about.html
│       ├── style.css
│       └── script.js
├── data/
│   ├── cleaned/
│   ├── features/
│   ├── processed/
│   ├── raw/
│   └── splits/
├── docs/
├── models/
├── notebooks/
├── reports/
│   ├── figures/
│   └── tables/
├── src/
├── tools/
├── requirements.txt
├── train_web_model.py
└── README.md
```

**Phase 1 files:**
- `notebooks/01_EDA_and_Cleaning.ipynb`: cleaning + EDA (reads `data/raw/`, writes `data/cleaned/`)
- `notebooks/02_Feature_Engineering.ipynb`: ratio features (writes `data/features/`)
- `notebooks/02b_Preprocessing_and_Scaling.ipynb`: log, clip, missing flags, scaling (writes `data/processed/`)
- `reports/figures/`: all graphs, with a note for each one in `reports/figures/README.md`
- `docs/cleaning_rules.md` and `docs/phase1_insights.md`

Notebooks 03 to 08, `models/` and `app/` are our modelling work (Phase 2).

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/aree22-hafeez/LeukoGuard_Project.git
cd LeukoGuard_Project
```

### 2. Create and activate a virtual environment

**Windows:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

Check `app/main.py` for the application's entry point and run the command appropriate to its framework. For example, if it uses FastAPI:

```bash
uvicorn app.main:app --reload
```

Then open the local URL shown in the terminal.

### 5. Run the Phase 1 notebooks

```bash
jupyter notebook
```

Open the `notebooks/` folder and run them in this order:

1. `01_EDA_and_Cleaning.ipynb` (about 2 minutes)
2. `02_Feature_Engineering.ipynb` (about 1 minute)
3. `02b_Preprocessing_and_Scaling.ipynb` (about 1 minute)

All three notebooks are already saved with their outputs, so you can also just open them on GitHub and read them without running anything.

**Google Colab:** upload the project folder to Google Drive, then run this in the first cell:

```python
from google.colab import drive
drive.mount("/content/drive")
import os
os.chdir("/content/drive/MyDrive/LeukoGuard_Project/notebooks")
```

## Model and Data

The project includes data preprocessing, feature engineering, model training, and evaluation notebooks. Trained model files are stored in `models/`, while analysis results and visualizations are available in `reports/`.

Some large dataset files are intentionally excluded from the GitHub repository. They must be available locally if a particular notebook or workflow requires them.

These files are over 50 MB, so they are not uploaded but get created when you run the notebooks: `data/cleaned/cleaned_internal_validation.csv`, `data/cleaned/cleaned_external_validation.csv`, `data/features/leukoalert_features.csv.gz` and `data/processed/processed_all_cohorts.csv.gz`. The full cleaned dataset *is* uploaded as `data/cleaned/leukoalert_cleaned.csv.gz`.

## Limitations

- Predictions depend on the quality and representativeness of the training data.
- Performance may vary across datasets and healthcare settings.
- Model explanations describe factors influencing the model; they do not establish medical causation.
- The system is intended for educational and research purposes, not clinical diagnosis.
- The public data has no patient ID, so if one person had several blood tests, they could end up in different CV folds. This can make CV scores a little too optimistic, so we trust the results on the unseen hospitals more.
- The "healthy" group is hospital patients without leukemia, not healthy volunteers.
- The training files were down-sampled to 1:2 by the dataset makers, so precision in training is not the same as precision in real life.

## Contributors

- Areej Hafeez
- Anum Aamir
- Haadiyah Zafar

## Disclaimer

LeukoGuard is a student academic project. It is not a clinically validated diagnostic system. Any health concerns should be discussed with a qualified healthcare professional.
