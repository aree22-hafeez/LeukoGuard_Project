# Deploying LeukoGuard

Run commands from the repository root. The website and the four trained pipelines run in one FastAPI service. The routine screening form uses the routine CBC Logistic Regression model. **Compare models** accepts the 71 analyzer features and displays one score and flag each for Logistic Regression, Random Forest and XGBoost. The extended panel uses the cleaned dataset units, including g/L for HGB and MCHC and percent for HCT. Missing measurements pass through each model’s fitted imputer.

## Run locally

Use Python 3.14.7 with the serving dependencies.

```powershell
python -m pip install -r app/requirements.txt
python -m uvicorn app.main:app
```

Open http://127.0.0.1:8000 and try **Load typical example** followed by **Check results**. Then open **Compare models**, choose **Load illustrative panel** and select **Compare scores**.

## Render Web Service

Connect the GitHub repository, select the merged `main` branch and leave Root Directory blank. Set the following values.

| Setting | Value |
| --- | --- |
| Language | Python 3 |
| Environment variable | `PYTHON_VERSION=3.14.7` |
| Build command | `pip install -r app/requirements.txt` |
| Start command | `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/api/health` |

The service reads the four files in `models/`, their fitted pipeline code in `src/` and the RF development predictions in `reports/predictions/rf_oof_development.csv`. The latter reconstruct the RF threshold selected in training. LR and XGBoost thresholds are in the model bundles. The RF artifact was trained under scikit-learn 1.8.0 and emits a version warning under the shared 1.9.1 environment. Four sample RF scores matched the saved evaluation predictions in that environment.

After deployment, open the URL Render assigns, run both examples, and confirm `/api/health` lists four models. Add the working URL to the project README.

To check both prediction APIs from a terminal, run `python tools/check_web_models.py` locally or supply the deployed URL as its argument.
