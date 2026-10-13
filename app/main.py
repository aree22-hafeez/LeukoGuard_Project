"""
LeukoGuard - website backend (FastAPI).

Run from the project root:
    uvicorn app.main:app --reload
then open http://127.0.0.1:8000
"""
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from leuko_common import CORE_FEATURES, add_derived_features   # noqa: E402
import lr_pipeline                                              # noqa: E402,F401  (needed to unpickle)

BUNDLE = joblib.load(ROOT / "models" / "web_model.joblib")
MODEL = BUNDLE["model"]
THRESHOLD = BUNDLE["threshold"]

# form field -> (model feature name, multiplier to the dataset's unit, readable label)
# The dataset stores HGB and MCHC in g/L; most labs report g/dL, so the form uses g/dL (x10).
FIELDS = {
    "wbc":    ("WBC",    1,  "White blood cells (WBC)"),
    "neut":   ("NEUT#",  1,  "Neutrophils (absolute)"),
    "lymph":  ("LYMPH#", 1,  "Lymphocytes (absolute)"),
    "mono":   ("MONO#",  1,  "Monocytes (absolute)"),
    "eo":     ("EO#",    1,  "Eosinophils (absolute)"),
    "baso":   ("BASO#",  1,  "Basophils (absolute)"),
    "rbc":    ("RBC",    1,  "Red blood cells (RBC)"),
    "hgb":    ("HGB",    10, "Hemoglobin"),
    "hct":    ("HCT",    1,  "Hematocrit"),
    "mcv":    ("MCV",    1,  "MCV"),
    "mch":    ("MCH",    1,  "MCH"),
    "mchc":   ("MCHC",   10, "MCHC"),
    "rdw_sd": ("RDW-SD", 1,  "RDW-SD"),
    "rdw_cv": ("RDW-CV", 1,  "RDW-CV"),
    "plt":    ("PLT",    1,  "Platelets (PLT)"),
    "mpv":    ("MPV",    1,  "MPV"),
    "pdw":    ("PDW",    1,  "PDW"),
    "plcr":   ("P-LCR",  1,  "P-LCR"),
    "pct":    ("PCT",    1,  "Plateletcrit (PCT)"),
}
# generous upper limits, only to catch obvious unit / typing mistakes (value is in the form's unit)
MAX_VALUE = {"wbc": 1500, "neut": 1000, "lymph": 1000, "mono": 500, "eo": 100, "baso": 50,
             "rbc": 12, "hgb": 30, "hct": 80, "mcv": 160, "mch": 60, "mchc": 50,
             "rdw_sd": 150, "rdw_cv": 50, "plt": 3000, "mpv": 25, "pdw": 40, "plcr": 100, "pct": 5}

LABELS = {feat: label for feat, _, label in FIELDS.values()}
LABELS.update({"NLR": "Neutrophil/lymphocyte ratio (NLR)", "PLR": "Platelet/lymphocyte ratio (PLR)",
               "MLR": "Monocyte/lymphocyte ratio (MLR)", "SII": "Immune-inflammation index (SII)",
               "dNLR": "Derived NLR (dNLR)", "NEUT%": "Neutrophil %", "LYMPH%": "Lymphocyte %",
               "MONO%": "Monocyte %", "EO%": "Eosinophil %", "BASO%": "Basophil %"})
# the displayed value of a model feature, in the unit the user typed
DISPLAY_DIV = {"HGB": 10, "MCHC": 10}


class CBC(BaseModel):
    wbc: Optional[float] = Field(None, ge=0)
    neut: Optional[float] = Field(None, ge=0)
    lymph: Optional[float] = Field(None, ge=0)
    mono: Optional[float] = Field(None, ge=0)
    eo: Optional[float] = Field(None, ge=0)
    baso: Optional[float] = Field(None, ge=0)
    rbc: Optional[float] = Field(None, ge=0)
    hgb: Optional[float] = Field(None, ge=0)
    hct: Optional[float] = Field(None, ge=0)
    mcv: Optional[float] = Field(None, ge=0)
    mch: Optional[float] = Field(None, ge=0)
    mchc: Optional[float] = Field(None, ge=0)
    rdw_sd: Optional[float] = Field(None, ge=0)
    rdw_cv: Optional[float] = Field(None, ge=0)
    plt: Optional[float] = Field(None, ge=0)
    mpv: Optional[float] = Field(None, ge=0)
    pdw: Optional[float] = Field(None, ge=0)
    plcr: Optional[float] = Field(None, ge=0)
    pct: Optional[float] = Field(None, ge=0)


def build_row(cbc: CBC) -> pd.DataFrame:
    """User values -> one-row DataFrame with the model's columns (NaN where not entered)."""
    vals = cbc.model_dump()
    for k, mx in MAX_VALUE.items():
        if vals[k] is not None and vals[k] > mx:
            raise HTTPException(422, f"{LABELS[FIELDS[k][0]]} = {vals[k]} looks too large. "
                                     "Please check the unit.")
    row = {f: np.nan for f in CORE_FEATURES}
    for key, (feat, mult, _) in FIELDS.items():
        if vals[key] is not None:
            row[feat] = float(vals[key]) * mult
    # differential percentages are computed from the counts and WBC
    wbc = row["WBC"]
    if pd.notna(wbc) and wbc > 0:
        for c in ["NEUT", "LYMPH", "MONO", "EO", "BASO"]:
            if pd.notna(row[f"{c}#"]):
                row[f"{c}%"] = 100.0 * row[f"{c}#"] / wbc
    return add_derived_features(pd.DataFrame([row]))


def explain(X: pd.DataFrame, top: int = 4):
    """Per-feature contribution to the score (log-odds). For a linear model this equals the
    exact SHAP value in the scaled space: coefficient x (scaled value)."""
    z = MODEL.transform(X).ravel()
    contrib = MODEL.lr_.coef_.ravel() * z
    names = MODEL.feature_names_out_
    rows, missing_total = [], 0.0
    for n, c in zip(names, contrib):
        if n.startswith("MISSING_"):
            missing_total += c
            continue
        v = X.iloc[0].get(n, np.nan)
        if pd.isna(v):              # value not entered -> imputed; do not present it as a finding
            missing_total += c
            continue
        shown = round(float(v) / DISPLAY_DIV.get(n, 1), 2)
        rows.append({"feature": LABELS.get(n, n), "value": shown, "effect": float(c)})
    if abs(missing_total) > 1e-9:
        rows.append({"feature": "Values left blank (assumed typical)", "value": None,
                     "effect": float(missing_total)})
    up = sorted([r for r in rows if r["effect"] > 0], key=lambda r: -r["effect"])[:top]
    down = sorted([r for r in rows if r["effect"] < 0], key=lambda r: r["effect"])[:top]
    for r in up + down:
        r["effect"] = round(r["effect"], 3)
    return up, down


app = FastAPI(title="LeukoGuard", description="Explainable CBC screening aid (not a diagnosis).")


@app.get("/api/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD, "trained_on": BUNDLE["trained_on"]}


@app.get("/api/model-info")
def model_info():
    return {"threshold": THRESHOLD, "target_recall": BUNDLE["target_recall"],
            "features": BUNDLE["features"], "trained_on": BUNDLE["trained_on"],
            "results": BUNDLE["results"]}


@app.post("/api/predict")
def predict(cbc: CBC):
    entered = {k: v for k, v in cbc.model_dump().items() if v is not None}
    if cbc.wbc is None or cbc.hgb is None or cbc.plt is None:
        raise HTTPException(422, "WBC, hemoglobin and platelets are required.")
    X = build_row(cbc)
    score = float(MODEL.predict_proba(X[BUNDLE["features"]])[0, 1])
    up, down = explain(X[BUNDLE["features"]])

    warnings = []
    if cbc.neut is None and cbc.lymph is None and cbc.mono is None:
        warnings.append("No white-cell differential was entered. Blank values are treated as typical, "
                        "so any abnormality in neutrophils, lymphocytes or monocytes is invisible "
                        "to the model and the score may be too low. Enter them if you have them.")
    if len(entered) < 12:
        warnings.append(f"Only {len(entered)} of 19 values were entered; the result is less reliable.")
    return {
        "score": round(score, 4),
        "threshold": round(THRESHOLD, 4),
        "flagged": score >= THRESHOLD,
        "raises_score": up,
        "lowers_score": down,
        "warnings": warnings,
        "n_entered": len(entered),
    }


@app.get("/api/global-effects")
def global_effects(top: int = 12):
    """Strongest model weights (log-odds per 1 SD of the scaled value) - the model's 'global' explanation."""
    coef = pd.Series(MODEL.lr_.coef_.ravel(), index=MODEL.feature_names_out_)
    top_idx = coef.abs().sort_values(ascending=False).head(top).index
    return [{"feature": LABELS.get(n, n), "weight": round(float(coef[n]), 3)} for n in top_idx]


# static website (must be mounted last so /api/... routes win)
app.mount("/", StaticFiles(directory=ROOT / "app" / "static", html=True), name="static")
