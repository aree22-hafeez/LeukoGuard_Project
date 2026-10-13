# LeukoGuard: Explainable Leukemia Risk Prediction

LeukoGuard is a machine learning-based screening aid that uses Complete Blood Count (CBC) parameters to estimate leukemia risk. It uses Logistic Regression and explainable AI techniques to help users understand which blood parameters influence a model's prediction.

**Note:** LeukoGuard is an academic project and is not intended to diagnose leukemia or replace professional medical advice.

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
│   └── raw/
├── docs/
├── models/
├── notebooks/
├── reports/
├── src/
├── tools/
├── requirements.txt
├── train_web_model.py
└── README.md
```

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

## Model and Data

The project includes data preprocessing, feature engineering, model training, and evaluation notebooks. Trained model files are stored in `models/`, while analysis results and visualizations are available in `reports/`.

Some large dataset files are intentionally excluded from the GitHub repository. They must be available locally if a particular notebook or workflow requires them.

## Limitations

- Predictions depend on the quality and representativeness of the training data.
- Performance may vary across datasets and healthcare settings.
- Model explanations describe factors influencing the model; they do not establish medical causation.
- The system is intended for educational and research purposes, not clinical diagnosis.

## Contributors

- Areej Hafeez
- Anum Aamir
- Haadiyah Zafar

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

## Disclaimer

LeukoGuard is a student academic project. It is not a clinically validated diagnostic system. Any health concerns should be discussed with a qualified healthcare professional.
