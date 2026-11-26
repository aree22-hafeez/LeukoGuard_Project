"""Explain the saved XGBoost model with Tree SHAP on held-out hospitals D–G."""
import importlib.util
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
import shap

# Notebook 09 has a numeric filename. Import its shared data loader and paths.
spec = importlib.util.spec_from_file_location(
    "xgb_training", Path(__file__).with_name("09_XGBoost.py")
)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)
from leuko_common import FEATURES, SEED
import matplotlib.pyplot as plt

TOP_K = 10
SAMPLE_SIZE = 4000


def explain(model, rows):
    inputs = model.transform(rows)
    explainer = shap.TreeExplainer(
        model.model_, feature_perturbation="tree_path_dependent", model_output="raw",
        feature_names=model.feature_names_out_,
    )
    result = explainer(inputs)
    # Keep the raw CBC values in plots, so median imputation is visible as missing.
    result.data = np.column_stack([rows[FEATURES].to_numpy(), inputs[:, len(FEATURES):]])
    reconstructed = expit(result.base_values + result.values.sum(axis=1))
    np.testing.assert_allclose(reconstructed, model.predict_proba(rows)[:, 1], atol=2e-5, rtol=2e-5)
    return result


def grouped_attributions(result):
    # Combine each laboratory measurement with its optional missingness indicator.
    frame = pd.DataFrame(result.values, columns=result.feature_names)
    missing = frame.filter(regex="^MISSING_").rename(columns=lambda name: name.removeprefix("MISSING_"))
    return frame[FEATURES].add(missing.reindex(columns=FEATURES, fill_value=0.0))


def summarize_directions(rows, result):
    attributions = pd.DataFrame(result.values, columns=result.feature_names)
    output = []
    for name in FEATURES:
        values = rows[name].reset_index(drop=True)
        present = values.notna()
        phi = attributions[name]
        q25, q75 = values.quantile([0.25, 0.75])
        low, high = present & values.le(q25), present & values.ge(q75)
        # Tied quartiles have overlapping groups and no distinct directional comparison.
        distinct = q25 < q75
        output.append({
            "feature": name, "mean_abs_shap": phi.abs().mean(),
            "observed_n": int(present.sum()), "missing_n": int((~present).sum()),
            "q25": q25, "q75": q75,
            "low_n": int(low.sum()) if distinct else 0,
            "high_n": int(high.sum()) if distinct else 0,
            "low_mean_shap": phi[low].mean() if distinct else np.nan,
            "high_mean_shap": phi[high].mean() if distinct else np.nan,
            "observed_positive_fraction": (phi[present] > 0).mean(),
            "observed_negative_fraction": (phi[present] < 0).mean(),
        })
    return pd.DataFrame(output).sort_values("mean_abs_shap", ascending=False)


def direction_profiles(rows, result, directions):
    # Quantile bands can show reversals that a low/high comparison would miss.
    names = list(directions["feature"].head(4))
    if "WBC" not in names:
        names[-1] = "WBC"
    attributions = pd.DataFrame(result.values, columns=result.feature_names)
    records = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, name in zip(axes.flat, names, strict=True):
        values = rows[name]
        bands = pd.qcut(values, q=10, duplicates="drop")
        profile = pd.DataFrame({"value": values, "shap": attributions[name]}).groupby(
            bands, observed=True
        ).agg(n=("value", "size"), median_cbc=("value", "median"), mean_shap=("shap", "mean"))
        ax.axhline(0, color="grey", linewidth=1)
        ax.plot(profile["median_cbc"], profile["mean_shap"], marker="o", color="#336f9e")
        ax.set(xlabel=name, ylabel="Mean SHAP (log-odds)")
        records.extend({"feature": name, **row} for row in profile.to_dict("records"))
    fig.suptitle("Observed CBC values and their average SHAP contribution")
    fig.savefig(training.FIGURES / "XGB_shap_direction.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(records).to_csv(training.TABLES / "XGB_shap_direction_profiles.csv", index=False)


def stability(bundle, sample, final_scores):
    scores = {"final": final_scores}
    for fold, model in enumerate(bundle["fold_models"]):
        scores[f"fold_{fold}"] = grouped_attributions(explain(model, sample)).abs().mean()
    magnitudes = pd.DataFrame(scores)
    ranks = magnitudes.rank(ascending=False, method="min").astype(int)
    fold_ranks = ranks.drop(columns="final")
    summary = ranks.add_suffix("_rank").join(magnitudes.add_suffix("_mean_abs_shap"))
    summary["fold_top10_count"] = fold_ranks.le(TOP_K).sum(axis=1)
    summary["mean_fold_rank"] = fold_ranks.mean(axis=1)
    summary["fold_rank_sd"] = fold_ranks.std(axis=1)
    summary.sort_values(
        ["fold_top10_count", "mean_fold_rank", "final_rank"], ascending=[False, True, True]
    ).rename_axis("feature").to_csv(training.TABLES / "XGB_shap_stability.csv")

    pairs = []
    for a, b in combinations(ranks, 2):
        a_top = set(ranks.index[ranks[a] <= TOP_K])
        b_top = set(ranks.index[ranks[b] <= TOP_K])
        pairs.append({
            "model_a": a, "model_b": b, "top10_overlap": len(a_top & b_top),
            "top10_jaccard": len(a_top & b_top) / len(a_top | b_top),
            "rank_spearman": spearmanr(ranks[a], ranks[b]).statistic,
        })
    pd.DataFrame(pairs).to_csv(training.TABLES / "XGB_shap_stability_pairs.csv", index=False)


def explain_examples(bundle, held, probs):
    predictions = probs >= bundle["threshold"]
    categories = {"TP": (1, True), "TN": (0, False), "FP": (0, True), "FN": (1, False)}
    picks, contributions = [], []
    for label, (actual, predicted) in categories.items():
        positions = np.flatnonzero(held["y"].eq(actual).to_numpy() & (predictions == predicted))
        if not len(positions):
            print(f"No held-out {label} case available")
            continue
        midpoint = np.median(probs[positions])
        closest = positions[np.argsort(np.abs(probs[positions] - midpoint), kind="stable")[:30]]
        cases = held.iloc[closest]
        result = explain(bundle["model"], cases)
        both = np.flatnonzero((result.values > 0).any(axis=1) & (result.values < 0).any(axis=1))
        selected = int(both[0]) if len(both) else 0
        row = cases.iloc[selected]
        local = result[selected]
        probability = probs[closest[selected]]
        picks.append({
            "case": label, "sample_id": row["sample_id"], "site": row["site"], "y": actual,
            "p": probability, "threshold": bundle["threshold"], "base_log_odds": local.base_values,
            "model_log_odds": local.base_values + local.values.sum(),
        })
        for name, value, phi in zip(local.feature_names, local.data, local.values, strict=True):
            contributions.append({"case": label, "feature": name, "value": value, "shap_log_odds": phi})
        shap.plots.waterfall(local, max_display=12, show=False)
        plt.title(f"{label} example from site {row['site']} · model score {probability:.3f}", pad=38)
        plt.savefig(training.FIGURES / f"XGB_shap_waterfall_{label}.png", dpi=160, bbox_inches="tight")
        plt.close()
    pd.DataFrame(picks).to_csv(training.TABLES / "XGB_shap_examples.csv", index=False)
    pd.DataFrame(contributions).to_csv(training.TABLES / "XGB_shap_example_contributions.csv", index=False)


def main():
    bundle = joblib.load(training.MODEL)
    _, held = training.load_data()
    external = held[held["cohort"].eq("external_validation")]
    sample = external.sample(n=min(SAMPLE_SIZE, len(external)), random_state=SEED).reset_index(drop=True)
    sample[["sample_id", "site", "y"]].to_csv(training.TABLES / "XGB_shap_sample.csv", index=False)
    result = explain(bundle["model"], sample)
    pd.DataFrame({
        "feature": result.feature_names, "mean_abs_shap": np.abs(result.values).mean(axis=0),
        "mean_shap": result.values.mean(axis=0), "positive_fraction": (result.values > 0).mean(axis=0),
        "negative_fraction": (result.values < 0).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).to_csv(
        training.TABLES / "XGB_shap_global.csv", index=False
    )
    directions = summarize_directions(sample, result)
    directions.to_csv(training.TABLES / "XGB_shap_direction.csv", index=False)
    direction_profiles(sample, result, directions)
    np.random.seed(SEED)
    shap.plots.beeswarm(result, max_display=15, show=False, group_remaining_features=False)
    plt.xlabel("SHAP value (leukemia log-odds)")
    plt.savefig(training.FIGURES / "XGB_shap_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()
    grouped = grouped_attributions(result).abs().mean()
    top = grouped.nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.barh(top.index, top.to_numpy(), color="#336f9e")
    ax.set(xlabel="Mean absolute SHAP value (log-odds)", title="CBC feature importance · held-out hospitals D–G")
    fig.savefig(training.FIGURES / "XGB_shap_global.png", dpi=160)
    plt.close(fig)
    stability(bundle, sample, grouped)
    explain_examples(bundle, external, bundle["model"].predict_proba(external)[:, 1])


if __name__ == "__main__":
    main()
