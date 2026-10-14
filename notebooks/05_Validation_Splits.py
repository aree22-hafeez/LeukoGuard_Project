# %% [markdown]
# # 05 - Validation setup: split IDs, leakage checks, hospital and class balance (Member 2, Part A)
#
# | Item | Choice | Why |
# |---|---|---|
# | Development folds | Reuse `data/cv_folds_development.csv` (stratified 5-fold, seed 42) | Member 1 saved it for all members. Using the same folds keeps LR, RF and XGBoost comparable. |
# | Patient grouping | **Record-level**, not patient-level | The public release has no patient key. `sample_id` is a unique running row number (1 to 446,663). |
# | Held-out hospitals | D, E, F, G (never used for fitting, tuning or thresholds) | Member 1's protocol, so RQ1 uses the same held-out data as the LR results. |
# | Internal check | A, B, C validation files (not in development) | Member 1's internal validation cohort. Reported, never used for tuning. |
# | Real-world test | `test_site_true_world` | Member 1's extra test set. Touched once, at the end. |
#
# Output: `data/splits/split_ids.csv`, one row per record, with its role and CV fold.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("..").resolve() if Path.cwd().name == "notebooks" else Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
from leuko_common import FEATURES, INTERNAL_SITES, EXTERNAL_SITES, SEED

CLEAN_PATH = ROOT / "data" / "cleaned" / "leukoalert_cleaned.csv.gz"
FOLDS_PATH = ROOT / "data" / "cv_folds_development.csv"
SPLIT_DIR = ROOT / "data" / "splits"
TAB_DIR = ROOT / "reports" / "tables"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)

# %% [markdown]
# ## 1. Load the cleaned data and the shared development folds

# %%
df = pd.read_csv(CLEAN_PATH)
folds = pd.read_csv(FOLDS_PATH)
print(f"cleaned records: {len(df):,}  features: {len(FEATURES)}")
print(f"fold file rows : {len(folds):,}")

# %% [markdown]
# ## 2. Assign each record a role

# %%
ROLE_OF_COHORT = {"development": "development_cv", "internal_validation": "internal_validation",
                  "external_validation": "held_out_hospital", "real_world_test": "real_world_test"}
df["role"] = df["cohort"].map(ROLE_OF_COHORT)
assert df["role"].notna().all()
df = df.merge(folds[["sample_id", "fold"]], on="sample_id", how="left")
df["cv_fold"] = df["fold"].where(df["role"] == "development_cv").astype("Int64")
df = df.drop(columns="fold")

split = df[["sample_id", "role", "cohort", "site", "source_file", "y", "cv_fold"]].rename(columns={"site": "hospital"})
split.to_csv(SPLIT_DIR / "split_ids.csv", index=False)
print("saved data/splits/split_ids.csv", split.shape)

# %% [markdown]
# ## 3. Leakage checks

# %%
checks = []

def check(name, passed, detail=""):
    checks.append({"check": name, "passed": bool(passed), "detail": detail})

dev_ids = set(folds["sample_id"])
dev_split = split.loc[split["role"] == "development_cv", "sample_id"]
check("fold file ids == development ids", set(dev_split) == dev_ids, f"{len(dev_ids):,} ids")
check("every record has exactly one role", split["sample_id"].is_unique, f"{split['sample_id'].nunique():,} unique ids")

roles = split.groupby("role")["sample_id"].apply(set)
overlap = sum(len(roles[a] & roles[b]) for i, a in enumerate(roles.index) for b in roles.index[i + 1:])
check("no sample_id shared between roles", overlap == 0, f"{overlap} shared ids")

# Patient-key substitute: identical 71-feature CBC profiles appearing in more than one role.
key = pd.util.hash_pandas_object(df[FEATURES].round(6).fillna(-999.0), index=False).astype(str) + "|" + df["y"].astype(str)
df["_key"] = key.values
multi = df.groupby("_key")["role"].nunique()
cross = int((multi > 1).sum())
check("no identical CBC profile across roles", cross == 0, f"{cross} profiles appear in >1 role")

n_repeat = int((df.groupby(["_key", "role"]).size() > 1).sum())
check("no identical CBC profile repeated within a role", n_repeat == 0, f"{n_repeat} repeated profiles")

# Fold balance inside development
fold_stats = (split[split["role"] == "development_cv"].groupby("cv_fold")
              .agg(n=("y", "size"), leukemia=("y", "sum")).assign(prevalence_pct=lambda d: 100 * d["leukemia"] / d["n"]))
check("folds are balanced (max-min prevalence < 0.5 pt)",
      fold_stats["prevalence_pct"].max() - fold_stats["prevalence_pct"].min() < 0.5,
      f"{fold_stats['prevalence_pct'].min():.2f}-{fold_stats['prevalence_pct'].max():.2f}%")

checks_df = pd.DataFrame(checks)
checks_df.to_csv(TAB_DIR / "V1_leakage_checks.csv", index=False)
display(checks_df)

# %% [markdown]
# ## 4. Hospital distribution and class balance per role

# %%
summary = (split.groupby(["role", "hospital"])
           .agg(n=("y", "size"), leukemia=("y", "sum"))
           .assign(prevalence_pct=lambda d: 100 * d["leukemia"] / d["n"])
           .reset_index())
summary.to_csv(TAB_DIR / "V2_hospital_class_balance.csv", index=False)

role_summary = (split.groupby("role")
                .agg(n=("y", "size"), leukemia=("y", "sum"), hospitals=("hospital", lambda s: ",".join(sorted(s.unique()))))
                .assign(prevalence_pct=lambda d: 100 * d["leukemia"] / d["n"]))
role_summary.to_csv(TAB_DIR / "V3_role_summary.csv")
display(role_summary)
display(summary)

# %%
fold_stats.to_csv(TAB_DIR / "V4_development_fold_balance.csv")
display(fold_stats)
