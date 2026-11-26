# LeukoGuard: Explainable Leukemia Risk Prediction

LeukoGuard is a machine learning project that uses routine Complete Blood Count (CBC) values to flag patients who might have leukemia. We compare Logistic Regression, Random Forest and XGBoost, we test them on hospitals that were never used for training, and we show which blood values pushed each prediction up or down.

**Live website:** not deployed yet. See [Deploying the website](#deploying-the-website) and add the link here once it is live.

## Problem Statement

Leukemia changes the blood counts. For example the platelets and hemoglobin become low, and the WBC can be very low or very high. Almost every patient in a hospital gets a CBC test, but to confirm leukemia special tests like a bone marrow test are needed.

In our project we want to check if we can use only the normal CBC values to tell leukemia patients apart from healthy people. We also want the model to explain its answer and to still work in hospitals that were not used for training.

So this is a binary classification problem: leukemia (1) or healthy (0).

## Features

- CBC-based leukemia risk prediction with three models (Logistic Regression, Random Forest, XGBoost)
- Explanations: model coefficients (LR) and SHAP values (XGBoost) show which blood values affected a prediction
- Data cleaning and feature engineering
- Testing on hospitals that were never used for training, plus a real-world test set
- Web page where you enter CBC values and see the score and the main reasons for it

## Technologies Used

- Python, Pandas, NumPy, SciPy
- Scikit-learn, XGBoost, SHAP
- Matplotlib and Seaborn (graphs)
- Jupyter Notebook and Python scripts
- FastAPI, HTML, CSS and JavaScript (web app)

## Dataset

- **Name:** LeukoAlert-project: Transforming routine blood counts into a leukemia detection system with interpretable AI
- **Author:** Shilong Liu
- **Source:** Mendeley Data
- **Download link:** https://doi.org/10.17632/vc7kwnyppz.1 (Version 1)
- **License:** CC BY 4.0
- **Folder we used:** `leuk_vs_HC` (leukemia vs healthy)
- **Size:** 446,663 records in 11 CSV files (7 hospital sites, A to G, plus one real world test file)
- **Features:** 74 numeric CBC columns in the raw files, 71 after cleaning (for example WBC, HGB, PLT, NEUT%, LYMPH%, MCV, RDW, platelet indices, and the analyser flags)
- **Target variable:** `leukemia_label` (leukemia / healthy). We changed it to `y` (1 = leukemia, 0 = healthy)

The raw files are in `data/raw/`. If they are missing, download them from the link above and put the 11 CSV files in `data/raw/`.

Every file is one hospital ("site"):
- `train_site_A`, `B`, `C`: used for training (26,922 records, 33.3% leukemia)
- `valid_site_A`, `B`, `C`: validation from the same hospitals
- `valid_site_D`, `E`, `F`, `G`: hospitals that are never used in training
- `test_site_true_world`: real world test set

## Phase 1: Data Cleaning and EDA (Summary)

The data had no empty cells, but when we checked it carefully we found many problems:

1. About 105,000 values were not real. Someone had filled the gaps with the average before publishing the data (like `104.2533936651584`). We changed them back to missing (NaN).
2. Some hospitals wrote 0 when a test was not done. We changed these zeros to NaN.
3. Sites B and C wrote HCT as 0.35 and the other sites as 35. We multiplied B and C by 100 so all are the same.
4. Some values were impossible, like MCHC above 500 (139 records) and RBC = 0 (15 records). We made these values NaN.
5. We removed 22 duplicate records and 4 records that had no WBC, HGB and PLT.
6. We removed 3 "blast" flag columns because they were mostly empty and they almost give the answer directly.
7. We did not delete outliers, because most of the very high and very low values are leukemia patients.

After cleaning we have **446,637 records and 71 features**. All rules are written in `docs/cleaning_rules.md`.

For the EDA we made histograms, box plots, correlation heatmaps and a leukemia vs healthy comparison of every feature (notebook 01). All graphs are saved in `reports/figures/`.

Then we added 5 ratio features (NLR, PLR, MLR, SII, dNLR) and did scaling: log transform for skewed values, clipping of extreme values, filling missing values with the median (plus missing flags), and standard scaling. These numbers were calculated only from the training files.

### Phase 1 Key Findings

1. The training files have 1 leukemia for every 2 healthy (33.3%), but most of the other hospitals have only about 2 to 5% leukemia. So precision will be lower on the test hospitals.
2. Missing values are important. When the differential is missing, 92% of those records are leukemia.
3. Leukemia patients have much lower PLT, HGB, RBC, HCT, neutrophils and WBC, and higher LYMPH% and RDW. PLT is the strongest feature (median 38 vs 208).
4. WBC can be very low or very high in leukemia, so it is not just one direction.
5. Most outliers are leukemia patients (72% of the extreme WBC values), so we kept them.
6. Many features are almost the same (28 pairs have correlation above 0.95), like WBC and its other channels, or HGB and HCT.
7. The ratio features are not better than PLT on its own.
8. The hospitals are a bit different from each other, even for healthy patients.

More details are in `docs/phase1_insights.md`, and a note for every graph is in `reports/figures/README.md`.

## Phase 2: Modelling and Evaluation

### Protocol

- **Development:** hospitals A, B and C (training files). Stratified 5-fold cross-validation on the shared folds in `data/cv_folds_development.csv`. Tuning, class-imbalance choice and the decision threshold use only these folds.
- **Threshold:** for each model, the cut-off that gives 90% recall on the pooled out-of-fold predictions. It is frozen before any held-out data is used.
- **Held-out tests:** internal validation (A–C), hospitals D–G (never used before), and the real-world test set.
- Full protocol: `docs/validation_protocol.md`.

### Models

| Model | Inputs | Notes |
|---|---|---|
| Logistic Regression | 71 cleaned CBC features | Class-balanced, regularisation tuned by CV. Coefficients give readable odds ratios. |
| Random Forest | 71 cleaned CBC features | Depth, leaf size and features per split tuned by CV, one-standard-error rule. |
| XGBoost | 71 cleaned CBC features + missing indicators | 16-setting sweep, one-standard-error rule. SHAP explanations. |
| Web model (Logistic Regression) | 24 routine CBC values + 5 ratios | Only the values a user can type into the website. Used by `train_web_model.py`. |

### Final results

Threshold fixed on development data. Hospitals D–G are the main test, because those hospitals were never used for training.

**Held-out hospitals D–G (149,623 records, 3.9% leukemia)**

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.960 | 0.529 | 88.3% | 29.7% | 0.445 | 91.6% |
| Random Forest | **0.972** | **0.619** | 84.4% | **43.7%** | **0.576** | **95.6%** |
| XGBoost | 0.968 | 0.580 | 82.9% | 43.1% | 0.567 | 95.6% |

**Real-world test set (58,481 records, 3.9% leukemia)**

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.967 | 0.478 | 93.5% | 31.5% | 0.471 | 91.7% |
| Random Forest | 0.974 | 0.555 | 91.4% | 38.6% | 0.543 | 94.1% |
| XGBoost | 0.975 | 0.559 | 90.0% | 40.4% | 0.557 | 94.6% |

**Development (5-fold CV out-of-fold, 26,922 records, 33.3% leukemia)**

| Model | ROC-AUC | PR-AUC | Recall | Precision |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.968 | 0.930 | 90.0% | 86.8% |
| Random Forest | 0.982 | 0.961 | 90.0% | 91.9% |
| XGBoost | 0.987 | 0.973 | 90.0% | 94.3% |

Paired tests on hospitals D–G (`reports/tables/FINAL_pairwise_tests_heldout_DG.csv`):
- Random Forest vs Logistic Regression: PR-AUC +0.090 (95% CI 0.078 to 0.100), DeLong p < 0.001
- XGBoost vs Logistic Regression: PR-AUC +0.051 (95% CI 0.039 to 0.063), DeLong p < 0.001
- Random Forest vs XGBoost: PR-AUC +0.039 (95% CI 0.028 to 0.049), DeLong p < 0.001

Full tables: `reports/tables/FINAL_model_comparison_all.csv` (every site) and `reports/tables/FINAL_per_hospital_pivot.csv`.

### Comparison and analysis

- **Best model:** Random Forest ranks highest on the held-out hospitals (ROC-AUC 0.972, PR-AUC 0.619). XGBoost ranks in the middle on those hospitals but is best in development CV. Logistic Regression is the weakest on ranking, but it is the easiest to explain.
- **Generalisation:** every model is clearly weaker at precision on hospitals D–G than in development CV. The main reason is the much lower leukemia rate in those hospitals (3.9% vs 33.3%), so each flagged patient is more often a false alarm. The gap is largest for the ranking metric PR-AUC. ROC-AUC stays high (about 0.96 to 0.97). See `reports/figures/RQ1_prevalence_matched.png` and `RQ1_roc_pr_gap.png`.
- **Errors:** the error profiles (`reports/tables/E4_error_profiles.csv`) show that missed leukemia cases had higher platelet counts (169 vs 36 for true positives). False positives had more missing values (15% vs 8% for true positives).
- **Explanations (RQ2):** SHAP on XGBoost (`docs/member3_results_summary.md`) ranks platelets first in all five fold models. Low platelets raise the leukemia score. Nine measurements appear in every fold's top ten. The Logistic Regression platelet coefficient points the same way (lower platelets, higher risk).
- **Website model:** the web app uses a Logistic Regression model with only the 24 routine CBC values and 5 ratios, because users can only type routine values. On hospitals D–G it gets ROC-AUC 0.948, PR-AUC 0.510, recall 87.2% and precision 24.2% (`reports/tables/W1_web_model_results.csv`). It is less accurate than the 71-feature models.

### Figures

- Model comparison: `reports/figures/FINAL_model_comparison.png`
- ROC and PR curves: `reports/figures/XGB_roc_pr.png`, `reports/figures/RF_roc_pr_by_hospital.png`
- Confusion matrices: `reports/figures/D5_confusion_matrices.png`, `reports/figures/RF_confusion_matrices.png`, `reports/figures/XGB_confusion_matrices.png`
- Feature importance and SHAP: `reports/figures/RF_feature_importance.png`, `reports/figures/XGB_shap_global.png`, `reports/figures/XGB_shap_beeswarm.png`
- Logistic Regression odds ratios: `reports/figures/E5_readable_odds_ratios.png`

## Website

The website is a FastAPI app in `app/`. A user types CBC values (WBC, HGB and PLT are required; the rest are optional), and the page shows:
- the risk score and whether it is above the screening threshold (set for 90% recall on development data)
- the values that raised the score and the values that lowered it
- warnings when the differential is missing or few values are entered

The app is a screening aid, not a diagnosis.

## Project Structure

```text
LeukoGuard_Project/
├── app/                  Phase 2 website: FastAPI backend (main.py) and static pages
├── data/
│   ├── raw/              original 11 CSV files
│   ├── cleaned/          cleaned data
│   ├── features/         data with ratio features
│   ├── processed/        scaled data
│   └── splits/           shared split IDs
├── docs/                 cleaning rules, insights, validation protocol, results summaries
├── models/               saved models (LR, RF, XGBoost, website model)
├── notebooks/            Phase 1 and Phase 2 notebooks (.ipynb and .py)
├── reports/
│   ├── figures/          all graphs
│   ├── predictions/      out-of-fold and held-out predictions
│   └── tables/           all result tables
├── src/                  shared code (features, pipelines)
├── tools/                notebook runner
├── requirements.txt
├── train_web_model.py    trains the website model
└── README.md
```

Phase 1 notebooks:
- `01_EDA_and_Cleaning.ipynb`: cleaning and EDA
- `02_Feature_Engineering.ipynb`: ratio features
- `02b_Preprocessing_and_Scaling.ipynb`: scaling

Phase 2 notebooks and scripts:
- `03_Logistic_Regression.ipynb`, `04_RQ3_Derived_Features.ipynb`, `05_Validation_Splits.ipynb`
- `06_Random_Forest.ipynb`, `07_RQ1_Generalization.ipynb`
- `08_Final_Model_Comparison.ipynb`: the three-model comparison
- `09_XGBoost.py` and `10_XGBoost_SHAP.py`: XGBoost training and SHAP. These are scripts, not notebooks. Running them reproduces the saved tables.

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/aree22-hafeez/LeukoGuard_Project.git
cd LeukoGuard_Project
```

2. Make a virtual environment (Windows):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install the libraries:
```bash
pip install -r requirements.txt
```

4. Run the Phase 1 notebooks in this order: `01_EDA_and_Cleaning.ipynb`, then `02_Feature_Engineering.ipynb`, then `02b_Preprocessing_and_Scaling.ipynb`.
```bash
jupyter notebook
```
The notebooks are saved with outputs, so you can also open them on GitHub.

Some files are too big for GitHub, so they are not uploaded. They are made again when you run the notebooks (for example `data/processed/processed_all_cohorts.csv.gz`).

5. Reproduce the XGBoost results and SHAP explanations (from the repository root):
```bash
python notebooks/09_XGBoost.py
python notebooks/10_XGBoost_SHAP.py
python -X utf8 tools/run_notebook.py notebooks/08_Final_Model_Comparison.py notebooks/08_Final_Model_Comparison.ipynb
```
The XGBoost run takes about 3.5 minutes. The results are in `docs/member3_results_summary.md`.

6. Retrain the website model (optional, the trained file is in `models/web_model.joblib`):
```bash
python train_web_model.py
```

7. Run the website:
```bash
uvicorn app.main:app --reload
```
Then open http://127.0.0.1:8000 in your browser. The API docs are at http://127.0.0.1:8000/docs.

### Deploying the website

No deployment is set up yet. To put it online, for example on Render or Railway, use these settings:
- Install command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

The app needs `models/web_model.joblib`, `src/` and `app/` in the repository. It needs no API keys and no `.env` file.

## Conclusion

LeukoGuard shows that routine CBC values carry a strong leukemia signal. On hospitals never seen in training, the best models reach ROC-AUC of about 0.97, and platelet count is the top-ranked value in the XGBoost SHAP analysis in every fold. The results also show the main limit: a screening tool trained on hospitals with about 33% leukemia flags many healthy patients when it is used in hospitals with about 4% leukemia. On held-out hospitals D–G, about 44% of the patients flagged by the Random Forest have leukemia, and about 24% of those flagged by the routine-CBC website model do. The model is useful for ranking and for highlighting cases that need follow-up tests, but it is not a diagnostic tool. Before any clinical use, it would need prospective testing with the same analyser and local leukemia rates.

## Limitations
 
- The results depend on the data, and they can be different in other hospitals.
- The model explanations show what the model used, not what causes leukemia.
- The data has no patient ID, so one person's tests could be in both training and testing folds.
- The "healthy" group is hospital patients without leukemia, not fully healthy people.
- This project is only for learning and research, not for real diagnosis.
## Contributors
 
- Areej Hafeez
- Anum Aamir
- Haadiyah Zafar
## Disclaimer
 
LeukoGuard is a student project and is not tested for real medical use. If you have any health concerns, talk to a doctor.
