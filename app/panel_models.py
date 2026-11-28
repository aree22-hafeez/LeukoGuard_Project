"""Serving logic for the three final models evaluated on the shared analyzer panel."""
import re

import joblib
import pandas as pd

from leuko_common import CORE_FEATURES, FEATURES, FEATURE_GROUPS, QFLAGS, RAW_TO_SHORT
from lr_pipeline import threshold_for_recall


def load_panel_models(root):
    models = root / "models"
    lr = joblib.load(models / "lr_final.joblib")
    rf = joblib.load(models / "rf_final.joblib")
    rf.rf_.n_jobs = 1  # A request scores a single row.
    xgb = joblib.load(models / "xgb_final.joblib")
    # Only the RF artifact omits its operating threshold. Recover it using the
    # same OOF probabilities and recall rule as notebook 06.
    oof = pd.read_csv(root / "reports/predictions/rf_oof_development.csv",
                      usecols=["y", "p_oof"], float_precision="round_trip")
    models_by_name = {
        "Logistic Regression": (lr["model"], float(lr["threshold"])),
        "Random Forest": (rf, threshold_for_recall(oof.y, oof.p_oof, target=0.90)),
        "XGBoost": (xgb["model"], float(xgb["threshold"])),
    }
    for name, (model, threshold) in models_by_name.items():
        if set(model.features) != set(FEATURES) or not 0 <= threshold <= 1:
            raise ValueError(f"Invalid analyzer schema or threshold for {name}")
    return models_by_name


def panel_schema():
    units = {}
    for header, feature in RAW_TO_SHORT.items():
        unit = re.search(r"\(([^()]*)\)\]?$", header)
        units[feature] = ("analyzer score (0–300)" if feature in QFLAGS
                          else unit.group(1) if unit else "analyzer units")
    return {group: [{"name": name, "unit": units[name]} for name in names]
            for group, names in FEATURE_GROUPS.items()}


class InvalidPanel(ValueError):
    """The supplied measurements cannot be scored as an extended panel."""


def compare_panel(values, models):
    unknown = values.keys() - set(FEATURES)
    if unknown:
        raise InvalidPanel(f"Unknown analyzer field: {min(unknown)}")
    entered = {name: value for name, value in values.items() if value is not None}
    if not entered.keys() & {"WBC", "HGB", "PLT"}:
        raise InvalidPanel("Enter WBC, HGB or PLT.")
    if not entered.keys() - set(CORE_FEATURES):
        raise InvalidPanel("Enter an extended analyzer measurement or use the routine screening tool.")
    # Reindex inserts NaNs for omitted fields. Each saved model runs its own
    # fitted imputer and preprocessing before prediction.
    row = pd.DataFrame([entered], dtype=float).reindex(columns=FEATURES)
    scores = []
    for name, (model, threshold) in models.items():
        score = float(model.predict_proba(row)[0, 1])
        scores.append({"model": name, "score": score, "threshold": threshold,
                       "flagged": score >= threshold})
    return {"n_entered": len(entered), "n_features": len(FEATURES), "results": scores}
