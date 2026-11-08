"""Train and evaluate XGBoost. Run from the repository root."""
import argparse
from itertools import product
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    average_precision_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leuko_common import FEATURES
from lr_pipeline import metrics_at, threshold_for_recall
from xgb_pipeline import XGBScreen

TABLES = ROOT / "reports" / "tables"
PREDICTIONS = ROOT / "reports" / "predictions"
FIGURES = ROOT / "reports" / "figures"
MODEL = ROOT / "models" / "xgb_final.joblib"
TARGET_RECALL = 0.90
N_FOLDS = 5


def load_data():
    clean = pd.read_csv(ROOT / "data/cleaned/leukoalert_cleaned.csv.gz")
    split = pd.read_csv(ROOT / "data/splits/split_ids.csv")
    data = clean.merge(
        split[["sample_id", "role", "cv_fold", "y", "cohort", "hospital"]],
        on="sample_id", validate="one_to_one", suffixes=("", "_split"),
    )
    if (
        len(data) != len(clean) or len(data) != len(split)
        or not data["y"].eq(data["y_split"]).all()
        or not data["cohort"].eq(data["cohort_split"]).all()
        or not data["site"].eq(data["hospital"]).all()
    ):
        raise ValueError("The cleaned records do not match the shared split IDs.")
    dev = data[data["role"] == "development_cv"].reset_index(drop=True)
    if set(dev["cv_fold"]) != set(range(N_FOLDS)):
        raise ValueError("The shared split must contain development folds 0 through 4.")
    shared_folds = pd.read_csv(ROOT / "data/cv_folds_development.csv")
    actual = dev.set_index("sample_id")["cv_fold"].sort_index().astype(int)
    expected = shared_folds.set_index("sample_id")["fold"].sort_index().astype(int)
    if not actual.equals(expected):
        raise ValueError("The shared development fold files disagree.")
    held = data[data["role"] != "development_cv"].reset_index(drop=True)
    return dev, held


def cross_validate(dev, params, imbalance, jobs):
    y = dev["y"]
    oof = np.full(len(dev), np.nan)
    rows, models = [], []
    for fold in range(N_FOLDS):
        valid = dev["cv_fold"].eq(fold)
        model = XGBScreen(FEATURES, params, imbalance, jobs).fit(dev.loc[~valid], y[~valid])
        p = model.predict_proba(dev.loc[valid])[:, 1].astype(float)
        oof[valid] = p
        rows.append((roc_auc_score(y[valid], p), average_precision_score(y[valid], p)))
        models.append(model)
    scores = np.array(rows)
    summary = {
        "ROC_AUC": scores[:, 0].mean(), "ROC_AUC_sd": scores[:, 0].std(ddof=1),
        "PR_AUC": scores[:, 1].mean(), "PR_AUC_sd": scores[:, 1].std(ddof=1),
    }
    return summary, oof, models


def evaluate_baseline(dev, jobs):
    params = dict(max_depth=3, n_estimators=100, learning_rate=0.10, min_child_weight=1)
    summary, oof, _ = cross_validate(dev, params, "none", jobs)
    threshold = threshold_for_recall(dev["y"], oof, TARGET_RECALL)
    metrics = metrics_at(dev["y"], oof, threshold)
    pd.DataFrame([{**params, "imbalance": "none", "threshold": threshold,
                   "mean_fold_ROC_AUC": summary["ROC_AUC"], "mean_fold_PR_AUC": summary["PR_AUC"],
                   **metrics}]).to_csv(TABLES / "XGB_baseline_cv.csv", index=False)


def tune(dev, jobs):
    # Order from simplest to most complex for the one-standard-error rule.
    configs = [
        dict(max_depth=depth, n_estimators=trees, learning_rate=rate, min_child_weight=child)
        for depth, trees, rate, child in product((3, 5), (150, 300), (0.05, 0.10), (5, 1))
    ]
    rows = []
    for params in configs:
        summary, _, _ = cross_validate(dev, params, "balanced", jobs)
        rows.append({**params, **summary})
        print(f"{params}: mean PR-AUC {summary['PR_AUC']:.5f}", flush=True)

    best = max(rows, key=lambda row: row["PR_AUC"])
    cutoff = best["PR_AUC"] - best["PR_AUC_sd"] / np.sqrt(N_FOLDS)
    chosen = next(i for i, row in enumerate(rows) if row["PR_AUC"] >= cutoff)
    pd.DataFrame(rows).assign(selected=np.arange(len(rows)) == chosen).to_csv(
        TABLES / "XGB_tuning.csv", index=False
    )
    params = configs[chosen]

    runs = {
        option: cross_validate(dev, params, option, jobs)
        for option in ("balanced", "none", "smote")
    }
    top = max(runs, key=lambda option: runs[option][0]["PR_AUC"])
    balanced = runs["balanced"][0]
    if runs[top][0]["PR_AUC"] - balanced["PR_AUC"] <= balanced["PR_AUC_sd"] / np.sqrt(N_FOLDS):
        top = "balanced"
    pd.DataFrame([
        {"imbalance": option, **summary, "selected": option == top}
        for option, (summary, _, _) in runs.items()
    ]).to_csv(TABLES / "XGB_imbalance_comparison.csv", index=False)
    _, oof, models = runs[top]
    return params, top, oof, models


def evaluation_groups(held):
    for cohort, sites in (("internal_validation", "ABC"), ("external_validation", "DEFG")):
        group = held[held["cohort"].eq(cohort)]
        for site in sites:
            yield cohort, site, group[group["site"].eq(site)]
        yield cohort, "pooled", group
    yield "real_world_test", "RW", held[held["cohort"].eq("real_world_test")]


def plot_evaluation(groups, threshold):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for label, y, p in groups:
        RocCurveDisplay.from_predictions(y, p, name=label, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y, p, name=label, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--")
    axes[0].set_title("ROC")
    axes[1].set_title("Precision–recall")
    fig.savefig(FIGURES / "XGB_roc_pr.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for ax, (label, y, p) in zip(axes.flat, groups, strict=True):
        ConfusionMatrixDisplay.from_predictions(
            y, p >= threshold, labels=[0, 1], display_labels=["Non-leukemia", "Leukemia"],
            colorbar=False, values_format="d", ax=ax,
        )
        ax.set_title(label)
    fig.suptitle(f"XGBoost at frozen threshold {threshold:.4f}")
    fig.savefig(FIGURES / "XGB_confusion_matrices.png", dpi=160)
    plt.close(fig)


def main(jobs):
    for directory in (TABLES, PREDICTIONS, FIGURES, MODEL.parent):
        directory.mkdir(parents=True, exist_ok=True)
    dev, held = load_data()
    evaluate_baseline(dev, jobs)
    params, imbalance, oof, fold_models = tune(dev, jobs)
    threshold = threshold_for_recall(dev["y"], oof, TARGET_RECALL)
    pd.DataFrame([
        {"fold": fold, **metrics_at(dev.loc[dev["cv_fold"].eq(fold), "y"],
                                    oof[dev["cv_fold"].eq(fold)], threshold)}
        for fold in range(N_FOLDS)
    ]).to_csv(TABLES / "XGB_cv_fold_results.csv", index=False)

    dev[["sample_id", "cv_fold", "y"]].rename(columns={"cv_fold": "fold"}).assign(
        fold=dev["cv_fold"].astype(int), p_oof=oof
    ).to_csv(PREDICTIONS / "xgb_oof_development.csv", index=False)

    final = XGBScreen(FEATURES, params, imbalance, jobs).fit(dev, dev["y"])
    held["p"] = final.predict_proba(held)[:, 1].astype(float)
    held[["sample_id", "cohort", "site", "y", "p"]].to_csv(
        PREDICTIONS / "xgb_eval_predictions.csv.gz", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    rows = [{"cohort": "development (CV, OOF)", "site": "A+B+C",
             **metrics_at(dev["y"], oof, threshold)}]
    plot_groups = [("Development (OOF)", dev["y"], oof)]
    for cohort, site, group in evaluation_groups(held):
        rows.append({"cohort": cohort, "site": site, **metrics_at(group["y"], group["p"], threshold)})
        if site in ("pooled", "RW"):
            plot_groups.append((cohort.replace("_", " ").capitalize(), group["y"], group["p"]))
    results = pd.DataFrame(rows)
    results.to_csv(TABLES / "XGB_results_at_threshold.csv", index=False)
    plot_evaluation(plot_groups, threshold)
    joblib.dump(
        {"model": final, "fold_models": fold_models, "threshold": threshold,
         "target_recall": TARGET_RECALL, "params": params, "imbalance": imbalance},
        MODEL, compress=3,
    )
    print(f"Selected {params}, imbalance={imbalance}, frozen threshold={threshold:.8f}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4, help="XGBoost worker threads")
    main(parser.parse_args().jobs)
