# LeukoGuard: Explainable Leukemia Risk Prediction
 
LeukoGuard is a machine learning project that uses Complete Blood Count (CBC) values to find patients who might have leukemia. We use Logistic Regression and other models, and we also show which blood values affected the prediction, so it is easy to understand.
 
## Problem Statement
 
Leukemia changes the blood counts. For example the platelets and hemoglobin become low, and the WBC can be very low or very high. Almost every patient in a hospital gets a CBC test, but to confirm leukemia special tests like a bone marrow test are needed.
 
In our project we want to check if we can use only the normal CBC values to tell leukemia patients apart from healthy people. We also want the model to explain its answer and to still work in hospitals that were not used for training.
 
So this is a binary classification problem: leukemia (1) or healthy (0).
 
## Features
 
- CBC-based leukemia risk prediction
- Logistic Regression and other models
- Explains which blood values affected the prediction
- Data cleaning and feature engineering
- Testing on data from different hospitals
- Web page where you can enter CBC values and see the result
## Technologies Used
 
- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib and Seaborn (graphs)
- Jupyter Notebook
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
## Data Cleaning (Summary)
 
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
 
## Key Findings
 
1. The training files have 1 leukemia for every 2 healthy (33.3%), but most of the other hospitals have only about 2 to 5% leukemia. So precision will be lower on the test hospitals.
2. Missing values are important. When the differential is missing, 92% of those records are leukemia.
3. Leukemia patients have much lower PLT, HGB, RBC, HCT, neutrophils and WBC, and higher LYMPH% and RDW. PLT is the strongest feature (median 38 vs 208).
4. WBC can be very low or very high in leukemia, so it is not just one direction.
5. Most outliers are leukemia patients (72% of the extreme WBC values), so we kept them.
6. Many features are almost the same (28 pairs have correlation above 0.95), like WBC and its other channels, or HGB and HCT.
7. The ratio features are not better than PLT on its own.
8. The hospitals are a bit different from each other, even for healthy patients.
More details are in `docs/phase1_insights.md`, and a note for every graph is in `reports/figures/README.md`.
 
## Project Structure
 
```text
LeukoGuard/
├── app/                
├── data/
│   ├── raw/              original 11 CSV files
│   ├── cleaned/          cleaned data
│   ├── features/         data with ratio features
│   ├── processed/        scaled data
│   └── splits/
├── docs/                 cleaning rules and insights
├── models/               saved models
├── notebooks/            all notebooks
├── reports/
│   ├── figures/          all graphs
│   └── tables/
├── src/                  shared code
├── tools/
├── requirements.txt
├── train_web_model.py
└── README.md
```
 
Phase 1 notebooks:
- `01_EDA_and_Cleaning.ipynb`: cleaning and EDA
- `02_Feature_Engineering.ipynb`: ratio features
- `02b_Preprocessing_and_Scaling.ipynb`: scaling
Notebooks 03 to 08 cover Logistic Regression, Random Forest, and the shared evaluation. The Member 3 scripts train XGBoost and generate its SHAP explanations.

### Reproduce XGBoost and SHAP

Run from the repository root after installing `requirements.txt`.

```bash
python notebooks/09_XGBoost.py
python notebooks/10_XGBoost_SHAP.py
python -X utf8 tools/run_notebook.py notebooks/08_Final_Model_Comparison.py notebooks/08_Final_Model_Comparison.ipynb
```

The XGBoost run uses the checked-in cleaned data and shared split IDs. It writes the model bundle to `models/xgb_final.joblib`, predictions to `reports/predictions/`, and evaluation and SHAP outputs to `reports/tables/` and `reports/figures/`. See the [Member 3 results and RQ2 discussion](docs/member3_results_summary.md).
 
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
 
4. Run the notebooks in this order: `01_EDA_and_Cleaning.ipynb`, then `02_Feature_Engineering.ipynb`, then `02b_Preprocessing_and_Scaling.ipynb`.
```bash
jupyter notebook
```
The notebooks are already saved with outputs, so you can also just open them on GitHub.
 
Some files are too big for GitHub, so they are not uploaded. They are made again when you run the notebooks (for example `data/processed/processed_all_cohorts.csv.gz`).
 
5. To run the web app:
```bash
uvicorn app.main:app --reload
```
Then open the link that shows in the terminal.
 
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
