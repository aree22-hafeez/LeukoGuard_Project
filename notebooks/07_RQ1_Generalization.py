# %% [markdown]
# # 07 - RQ1: internal cross-validation vs held-out hospitals (Member 2, Part C)
#
# **RQ1:** Can routine CBC features identify leukemia with high recall, and does the model still perform well at an unseen hospital?
#
# This notebook measures the **generalization gap** (development CV minus held-out hospital) for Random Forest, with Logistic Regression from Member 1 as a reference. It then checks which factors explain the gap:
# 1. **Prevalence shift:** development is 33% leukemia; hospitals are about 4%. PR-AUC depends strongly on prevalence, so the drop is partly expected.
# 2. **Feature shift:** do the CBC values look different at the new hospitals, within the same class?
# 3. **Missingness shift:** do the missing-value patterns differ by hospital?

# %%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("..").resolve() if Path.cwd().name == "notebooks" else Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
from leuko_common import FEATURES, KEY_FEATURES, SEED
from lr_pipeline import threshold_for_recall, metrics_at

FIG_DIR = ROOT / "reports" / "figures"; TAB_DIR = ROOT / "reports" / "tables"; PRED_DIR = ROOT / "reports" / "predictions"
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
C_RF, C_LR, C_DEV = "#2a78d6", "#eb6834", "black"
def savefig(name): plt.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
pd.set_option("display.width", 200)

# ---------------- SETTINGS ----------------
TARGET_RECALL = 0.90
N_BOOT = 500
N_SUBSAMPLE = 100          # repeats for prevalence-matched PR-AUC
SITES = ["D", "E", "F", "G"]

# %% [markdown]
# ## 1. Load predictions (development OOF + held-out) for RF and LR

# %%
MODELS = {
    "RF": {"oof": PRED_DIR / "rf_oof_development.csv", "eval": PRED_DIR / "rf_eval_predictions.csv.gz"},
    "LR": {"oof": PRED_DIR / "lr_raw71_oof_development.csv", "eval": PRED_DIR / "lr_raw71_eval_predictions.csv.gz"},
}
oof, held, thr, dev_fold = {}, {}, {}, {}
for name, paths in MODELS.items():
    o = pd.read_csv(paths["oof"])
    pcol = "p_oof" if "p_oof" in o.columns else "p"
    oof[name] = o
    held[name] = pd.read_csv(paths["eval"])
    thr[name] = threshold_for_recall(o["y"], o[pcol], TARGET_RECALL)
    dev_fold[name] = o.assign(p=o[pcol])
    print(f"{name}: development OOF {len(o):,}, held-out {len(held[name]):,}, frozen threshold {thr[name]:.4f}")

# Cross-check: Member 1 reported the LR threshold as 0.511
print(f"LR threshold recomputed = {thr['LR']:.4f} (Member 1 reported 0.511)")

# %% [markdown]
# ## 2. Generalization gap by cohort, with bootstrap 95% CIs

# %%
def boot_ci(y, p, n=N_BOOT, seed=SEED):
    """Stratified bootstrap of ROC-AUC and PR-AUC (resamples positives and negatives separately)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y).astype(int), np.asarray(p)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    roc, pr = np.empty(n), np.empty(n)
    for b in range(n):
        s = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        roc[b] = roc_auc_score(y[s], p[s]); pr[b] = average_precision_score(y[s], p[s])
    return np.percentile(roc, [2.5, 97.5]), np.percentile(pr, [2.5, 97.5])

gap_rows = []
for name in MODELS:
    o, h, t = dev_fold[name], held[name], thr[name]
    # development: per-fold mean +- SD (5 folds)
    fold_roc = [roc_auc_score(g["y"], g["p"]) for _, g in o.groupby("fold")]
    fold_pr = [average_precision_score(g["y"], g["p"]) for _, g in o.groupby("fold")]
    m = metrics_at(o["y"], o["p"], t)
    gap_rows.append({"model": name, "cohort": "development (CV, OOF)", "site": "A+B+C", "n": len(o),
                     "prevalence_%": m["prevalence_%"], "ROC_AUC": np.mean(fold_roc), "ROC_AUC_CI_lo": np.nan,
                     "ROC_AUC_CI_hi": np.nan, "PR_AUC": np.mean(fold_pr), "PR_AUC_CI_lo": np.nan, "PR_AUC_CI_hi": np.nan,
                     "fold_ROC_sd": np.std(fold_roc, ddof=1), "fold_PR_sd": np.std(fold_pr, ddof=1),
                     "recall": m["recall"], "specificity": m["specificity"]})
    targets = [("internal_validation", "A+B+C", h[h["cohort"] == "internal_validation"])]
    targets += [("external_validation", s, h[(h["cohort"] == "external_validation") & (h["site"] == s)]) for s in SITES]
    targets += [("external_validation", "pooled D-G", h[h["cohort"] == "external_validation"])]
    targets += [("real_world_test", "RW", h[h["cohort"] == "real_world_test"])]
    for cohort, site, g in targets:
        m = metrics_at(g["y"], g["p"], t)
        (rlo, rhi), (plo, phi) = boot_ci(g["y"], g["p"])
        gap_rows.append({"model": name, "cohort": cohort, "site": site, "n": len(g),
                         "prevalence_%": m["prevalence_%"], "ROC_AUC": m["ROC_AUC"], "ROC_AUC_CI_lo": rlo,
                         "ROC_AUC_CI_hi": rhi, "PR_AUC": m["PR_AUC"], "PR_AUC_CI_lo": plo, "PR_AUC_CI_hi": phi,
                         "fold_ROC_sd": np.nan, "fold_PR_sd": np.nan, "recall": m["recall"], "specificity": m["specificity"]})

gap = pd.DataFrame(gap_rows)
dev_ref = gap[gap["cohort"].str.startswith("development")].set_index("model")
gap["delta_ROC_vs_dev"] = gap.apply(lambda r: dev_ref.loc[r["model"], "ROC_AUC"] - r["ROC_AUC"], axis=1)
gap["delta_PR_vs_dev"] = gap.apply(lambda r: dev_ref.loc[r["model"], "PR_AUC"] - r["PR_AUC"], axis=1)
gap.to_csv(TAB_DIR / "RQ1_generalization_gap.csv", index=False)
display(gap.round(4))

# %% [markdown]
# ## 3. Prevalence-matched PR-AUC
#
# To separate prevalence from model quality: for each hospital, keep all leukemia records and randomly subsample non-leukemia records so prevalence matches development (33%). PR-AUC on the matched set is comparable with development PR-AUC.

# %%
def prevalence_matched_pr(y, p, target_prev=0.333, n=N_SUBSAMPLE, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y).astype(int), np.asarray(p)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    n0 = min(len(i0), int(round(len(i1) * (1 - target_prev) / target_prev)))
    vals = []
    for _ in range(n):
        s = np.concatenate([i1, rng.choice(i0, n0, replace=False)])
        vals.append(average_precision_score(y[s], p[s]))
    return np.mean(vals), np.std(vals, ddof=1), n0

pm_rows = []
for name in MODELS:
    h = held[name]
    dev_pr = gap[(gap["model"] == name) & gap["cohort"].str.startswith("development")]["PR_AUC"].item()
    for cohort, site, g in [("external_validation", s, h[(h["cohort"] == "external_validation") & (h["site"] == s)]) for s in SITES] + \
                           [("external_validation", "pooled D-G", h[h["cohort"] == "external_validation"])]:
        mean_pr, sd_pr, n0 = prevalence_matched_pr(g["y"], g["p"])
        raw_pr = average_precision_score(g["y"], g["p"])
        pm_rows.append({"model": name, "site": site, "prevalence_%": 100 * g["y"].mean(), "raw PR_AUC": raw_pr,
                        "PR_AUC at 33% prevalence": mean_pr, "sd (subsamples)": sd_pr, "dev CV PR_AUC": dev_pr,
                        "gap after prevalence matching": dev_pr - mean_pr, "non-leuk records kept": n0})
pm = pd.DataFrame(pm_rows)
pm.to_csv(TAB_DIR / "RQ1_prevalence_matched_pr.csv", index=False)
display(pm.round(4))

# %% [markdown]
# ## 4. Feature shift: within-class comparison
#
# Standardised mean difference (SMD, hospital minus development, divided by the development SD). Computed **within each class** so the shift is not confounded with prevalence. Missing values are excluded from the means.

# %%
clean = pd.read_csv(ROOT / "data" / "cleaned" / "leukoalert_cleaned.csv.gz",
                    usecols=["site", "cohort", "y"] + KEY_FEATURES)
dev_df = clean[clean["cohort"] == "development"]
smd_rows = []
for cls, cls_name in [(0, "non-leukemia"), (1, "leukemia")]:
    d = dev_df[dev_df["y"] == cls]
    for site in SITES:
        h = clean[(clean["cohort"] == "external_validation") & (clean["site"] == site) & (clean["y"] == cls)]
        for f in KEY_FEATURES:
            sd = d[f].std()
            smd_rows.append({"class": cls_name, "site": site, "feature": f,
                             "SMD (hospital - dev)": (h[f].mean() - d[f].mean()) / sd if sd > 0 else np.nan})
smd = pd.DataFrame(smd_rows)
smd_wide = smd.pivot_table(index="feature", columns=["class", "site"], values="SMD (hospital - dev)").reindex(KEY_FEATURES)
smd_wide.set_axis([f"{c[0]} | {c[1]}" for c in smd_wide.columns], axis=1).rename_axis("feature").to_csv(
    TAB_DIR / "RQ1_feature_shift_smd.csv")
display(smd_wide.round(2))

# %% [markdown]
# ## 5. Missingness by cohort
#
# Share of records with each key value missing. Platelet indices and the differential block follow the cleaning rules (R6, R7).

# %%
miss_cols = ["PDW", "MPV", "P-LCR", "PCT", "NEUT#", "LYMPH%", "MONO%", "IG#", "NE-SSC"]
miss_df = pd.read_csv(ROOT / "data" / "cleaned" / "leukoalert_cleaned.csv.gz", usecols=["cohort", "site"] + miss_cols)
miss_df["cohort_site"] = np.where(miss_df["cohort"] == "external_validation", miss_df["site"],
                                  miss_df["cohort"].map({"development": "dev A-C", "internal_validation": "int A-C",
                                                         "real_world_test": "RW"}))
miss = (miss_df.groupby("cohort_site")[miss_cols].apply(lambda g: 100 * g.isna().mean())).round(1)
miss = miss.loc[["dev A-C", "int A-C", "D", "E", "F", "G", "RW"]]
miss.to_csv(TAB_DIR / "RQ1_missingness_by_cohort.csv")
display(miss)

# %% [markdown]
# ## 6. Figures

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, metric in zip(axes, ["ROC_AUC", "PR_AUC"]):
    for k, name in enumerate(MODELS):
        g = gap[(gap["model"] == name) & gap["site"].isin(["A+B+C"] + SITES + ["pooled D-G"])]
        g = g[~((g["cohort"] == "internal_validation") & (g["site"] == "A+B+C"))]
        x = np.arange(len(g)) + (k - 0.5) * 0.38
        ax.bar(x, g[metric], width=0.38, color=C_RF if name == "RF" else C_LR, label=name)
        err_lo = g[metric] - g[f"{metric}_CI_lo"].fillna(g[metric]); err_hi = g[f"{metric}_CI_hi"].fillna(g[metric]) - g[metric]
        ax.errorbar(x, g[metric], yerr=[err_lo.clip(lower=0), err_hi.clip(lower=0)], fmt="none", ecolor="grey", lw=1)
    ax.set_xticks(np.arange(len(g)), g["site"].tolist())
    ax.set_title(f"{metric}: development CV vs held-out hospitals")
    ax.set_ylim(0.4 if metric == "ROC_AUC" else 0, 1.0)
axes[0].legend()
savefig("RQ1_roc_pr_gap")

fig, ax = plt.subplots(figsize=(9, 4.2))
pmr = pm[pm["model"] == "RF"]
x = np.arange(len(pmr))
ax.bar(x - 0.2, pmr["raw PR_AUC"], 0.4, color="#9ec3ee", label="raw PR-AUC (hospital prevalence)")
ax.bar(x + 0.2, pmr["PR_AUC at 33% prevalence"], 0.4, color=C_RF, label="PR-AUC at 33% prevalence")
ax.axhline(gap[(gap["model"] == "RF") & gap["cohort"].str.startswith("development")]["PR_AUC"].item(), color="black", ls="--", lw=1, label="dev CV PR-AUC")
ax.set_xticks(x, pmr["site"].tolist()); ax.set_title("RF: how much of the PR-AUC drop is prevalence?"); ax.legend(fontsize=8)
savefig("RQ1_prevalence_matched")

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(smd_wide.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(smd_wide.shape[1]), [f"{c[0][:4]} {c[1]}" for c in smd_wide.columns], rotation=45, ha="right")
ax.set_yticks(range(len(KEY_FEATURES)), KEY_FEATURES); ax.grid(False)
plt.colorbar(im, ax=ax, label="SMD (hospital - development), within class")
ax.set_title("Feature shift within class (0 = same as development)")
savefig("RQ1_feature_shift_heatmap")

# %% [markdown]
# ## 7. Summary

# %%
rf_dev = gap[(gap["model"] == "RF") & gap["cohort"].str.startswith("development")].iloc[0]
rf_pool = gap[(gap["model"] == "RF") & (gap["site"] == "pooled D-G")].iloc[0]
rf_pm = pm[(pm["model"] == "RF") & (pm["site"] == "pooled D-G")].iloc[0]
# PR-AUC drop split: part explained by prevalence (raw -> 33% matched) and part left after matching (matched -> dev)
explained = (rf_pm["PR_AUC at 33% prevalence"] - rf_pool["PR_AUC"]) / (rf_dev["PR_AUC"] - rf_pool["PR_AUC"])
print(f"RF development CV : ROC-AUC {rf_dev['ROC_AUC']:.4f}  PR-AUC {rf_dev['PR_AUC']:.4f}")
print(f"RF held-out D-G   : ROC-AUC {rf_pool['ROC_AUC']:.4f} [{rf_pool['ROC_AUC_CI_lo']:.4f}, {rf_pool['ROC_AUC_CI_hi']:.4f}]  "
      f"PR-AUC {rf_pool['PR_AUC']:.4f} (prevalence {rf_pool['prevalence_%']:.1f}%)  recall {rf_pool['recall']:.3f}")
print(f"RF PR-AUC at 33% prevalence (D-G pooled): {rf_pm['PR_AUC at 33% prevalence']:.4f}")
print(f"Share of the PR-AUC drop explained by prevalence: {100 * explained:.0f}%")
