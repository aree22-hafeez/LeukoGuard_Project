"""
LeukoGuard - train the model that powers the website.

Uses the ROUTINE 24-value CBC (CORE_FEATURES) + the 5 ratio features, because a user
can type these in. The 71-feature model needs analyser-only values (scattergram channels etc.).

Run from the project root (the folder that contains src/ and data/):
    python train_web_model.py

Needs:  data/features/leukoalert_features.csv.gz   (made by notebook 02)
        data/cv_folds_development.csv               (shared 5 folds)
Creates: models/web_model.joblib   +   reports/tables/W1_web_model_results.csv
"""
import sys
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from leuko_common import CORE_FEATURES, DERIVED, LOG_FEATURES            # noqa: E402
from lr_pipeline import (LRScreen, oof_predict, threshold_for_recall,    # noqa: E402
                         metrics_at)
from sklearn.metrics import roc_auc_score                                # noqa: E402

FEAT_PATH = ROOT / "data" / "features" / "leukoalert_features.csv.gz"
FOLDS_PATH = ROOT / "data" / "cv_folds_development.csv"
MODEL_PATH = ROOT / "models" / "web_model.joblib"
TAB_DIR = ROOT / "reports" / "tables"
MODEL_PATH.parent.mkdir(exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

TARGET_RECALL = 0.90
FEATURES = CORE_FEATURES + DERIVED          # 24 + 5 = 29 model inputs
C_GRID = [0.01, 0.1, 1.0]


def main():
    print("Loading", FEAT_PATH)
    fe = pd.read_csv(FEAT_PATH)
    folds_df = pd.read_csv(FOLDS_PATH)[["sample_id", "fold"]]

    dev = fe[fe["cohort"] == "development"].merge(folds_df, on="sample_id", how="inner")
    dev = dev.reset_index(drop=True)
    assert len(dev) == len(folds_df), "development rows and fold file do not match"
    X, y = dev[FEATURES], dev["y"].astype(int)
    folds = [(np.where(dev["fold"] != k)[0], np.where(dev["fold"] == k)[0]) for k in range(5)]
    print(f"Development: {len(dev):,} records, {y.mean():.1%} leukemia")

    # ---- small tuning on the shared folds (pooled out-of-fold ROC-AUC) ----------
    scores = {}
    for C in C_GRID:
        m = LRScreen(features=FEATURES, log_features=LOG_FEATURES, C=C, l1_ratio=0.0,
                     imbalance="balanced", indicator=False)
        p = oof_predict(m, X, y, folds)
        scores[C] = (roc_auc_score(y, p), p)
        print(f"  C={C:<5} OOF ROC-AUC = {scores[C][0]:.4f}")
    best_auc = max(s[0] for s in scores.values())
    # prefer the strongest regularisation within 0.005 AUC of the best (same idea as the 1-SE rule):
    # it keeps correlated values (HGB / HCT / RBC) from getting large opposite weights, so the
    # per-patient explanations stay readable.
    C_best = min(C for C, s in scores.items() if s[0] >= best_auc - 0.005)
    p_oof = scores[C_best][1]
    thr = threshold_for_recall(y, p_oof, TARGET_RECALL)
    print(f"Chosen C = {C_best};  threshold for {TARGET_RECALL:.0%} recall = {thr:.4f}")

    # ---- final model on all development data ------------------------------------
    final = LRScreen(features=FEATURES, log_features=LOG_FEATURES, C=C_best, l1_ratio=0.0,
                     imbalance="balanced", indicator=False).fit(X, y)

    # ---- evaluation: CV + unseen hospitals + real-world test ---------------------
    rows = [{"cohort": "development (CV, OOF)", **metrics_at(y, p_oof, thr)}]
    for name, sub in [("internal_validation", fe[fe.cohort == "internal_validation"]),
                      ("external_validation (hospitals D-G)", fe[fe.cohort == "external_validation"]),
                      ("real_world_test", fe[fe.cohort == "real_world_test"])]:
        if len(sub):
            p = final.predict_proba(sub[FEATURES])[:, 1]
            rows.append({"cohort": name, **metrics_at(sub["y"].astype(int), p, thr)})
    res = pd.DataFrame(rows)
    res.to_csv(TAB_DIR / "W1_web_model_results.csv", index=False)
    show = ["cohort", "n", "prevalence_%", "recall", "precision", "specificity", "ROC_AUC", "PR_AUC"]
    print("\n", res[show].round(3).to_string(index=False))

    joblib.dump({
        "model": final,
        "threshold": float(thr),
        "target_recall": TARGET_RECALL,
        "features": FEATURES,
        "C": C_best,
        "sklearn_version": sklearn.__version__,
        "trained_on": dt.date.today().isoformat(),
        "results": res[show].round(4).to_dict("records"),
    }, MODEL_PATH)
    print("\nSaved", MODEL_PATH)


if __name__ == "__main__":
    main()
