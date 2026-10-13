"""
LeukoGuard - shared configuration, loading and FIXED (deterministic) cleaning.

Member 1 / Part B. Every rule here is applied record-by-record or file-by-file
and learns NOTHING from the label or from other splits, so it is safe to run on
the whole dataset before modelling. Imputation, scaling, outlier capping and
SMOTE/class weighting are NOT done here - they belong inside the model pipeline
so they are fitted on training folds only.

Usage (teammates):
    from leuko_common import load_clean
    df = load_clean("data/cleaned/leukoalert_cleaned.csv")      # cleaned file
    X, y = df[FEATURES], df["y"]
or rebuild from the raw CSVs:
    from leuko_common import build_cleaned_dataset
    df, log = build_cleaned_dataset("data/raw")
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# --------------------------------------------------------------------------- #
# 1. Column names: raw public header -> short name used by the whole team
#    (same short names as the group's code-leuk_vs_HC.py PAPER_FEATURES list)
# --------------------------------------------------------------------------- #
RAW_TO_SHORT = {
    # Q-flags (analyser suspicion scores, 0-300; >=100 = flag raised)
    "Q-Flag(Left Shift?)": "Left Shift",
    "Q-Flag(Atypical Lympho?)": "Atypical Lympho",
    "Q-Flag(RBC Agglutination?)": "RBC Agglutination",
    "Q-Flag(Turbidity/HGB Interf?)": "Turbidity/HGB Interf",
    "Q-Flag(Iron Deficiency?)": "Iron Deficiency",
    "Q-Flag(HGB Defect?)": "HGB Defect",
    "Q-Flag(Fragments?)": "Fragments",
    "Q-Flag(PLT Clumps?)": "PLT Clumps",
    # standard CBC
    "WBC(10^9/L)": "WBC", "RBC(10^12/L)": "RBC", "HGB(g/L)": "HGB", "HCT(%)": "HCT",
    "MCV(fL)": "MCV", "MCH(pg)": "MCH", "MCHC(g/L)": "MCHC", "PLT(10^9/L)": "PLT",
    "RDW-SD(fL)": "RDW-SD", "RDW-CV(%)": "RDW-CV", "PDW(fL)": "PDW", "MPV(fL)": "MPV",
    "P-LCR(%)": "P-LCR", "PCT(%)": "PCT", "NRBC#(10^9/L)": "NRBC#", "NRBC%(/100WBC)": "NRBC%",
    # 5-part differential
    "NEUT#(10^9/L)": "NEUT#", "LYMPH#(10^9/L)": "LYMPH#", "MONO#(10^9/L)": "MONO#",
    "EO#(10^9/L)": "EO#", "BASO#(10^9/L)": "BASO#",
    "NEUT%(%)": "NEUT%", "LYMPH%(%)": "LYMPH%", "MONO%(%)": "MONO%", "EO%(%)": "EO%", "BASO%(%)": "BASO%",
    # research / extended parameters (Sysmex XN)
    "[IG#(10^9/L)]": "IG#", "[IG%(%)]": "IG%", "[PLT-I(10^9/L)]": "PLT-F",
    "[MicroR(%)]": "MicroR", "[MacroR(%)]": "MacroR", "[TNC(10^9/L)]": "TNC",
    "[WBC-N(10^9/L)]": "WBC-N", "[TNC-N(10^9/L)]": "TNC-N", "[BA-N#(10^9/L)]": "BA-N#",
    "[BA-N%(%)]": "BA-N%", "[WBC-D(10^9/L)]": "WBC-D", "[TNC-D(10^9/L)]": "TNC-D",
    "[NEUT#&(10^9/L)]": "NEUT#&", "[NEUT%&(%)]": "NEUT%&", "[LYMP#&(10^9/L)]": "LYMPH#&",
    "[LYMP%&(%)]": "LYMPH%&", "[HFLC#(10^9/L)]": "HFLC#", "[HFLC%(%)]": "HFLC%",
    "[BA-D#(10^9/L)]": "BA-D#", "[BA-D%(%)]": "BA-D%",
    # cell-population (scattergram) parameters
    "[NE-SSC(ch)]": "NE-SSC", "[NE-SFL(ch)]": "NE-SFL", "[NE-FSC(ch)]": "NE-FSC",
    "[LY-X(ch)]": "LY-X", "[LY-Y(ch)]": "LY-Y", "[LY-Z(ch)]": "LY-Z",
    "[MO-X(ch)]": "MO-X", "[MO-Y(ch)]": "MO-Y", "[MO-Z(ch)]": "MO-Z",
    "[NE-WX]": "NE-WX", "[NE-WY]": "NE-WY", "[NE-WZ]": "NE-WZ",
    "[LY-WX]": "LY-WX", "[LY-WY]": "LY-WY", "[LY-WZ]": "LY-WZ",
    "[MO-WX]": "MO-WX", "[MO-WY]": "MO-WY",
}

# Blast-suspicion flags: in the public release, but NOT in the paper's 72-feature set.
DROPPED_BLAST_FLAGS = {
    "Q-Flag(Blasts/Abn Lympho?)": "Blasts/Abn Lympho",
    "Q-Flag(Blasts?)": "Blasts",
    "Q-Flag(Abn Lympho?)": "Abn Lympho",
}

# --------------------------------------------------------------------------- #
# 2. Feature groups (short names)
# --------------------------------------------------------------------------- #
QFLAGS = ["Left Shift", "Atypical Lympho", "RBC Agglutination", "Turbidity/HGB Interf",
          "Iron Deficiency", "HGB Defect", "Fragments", "PLT Clumps"]
WBC_GROUP = ["WBC", "NEUT#", "LYMPH#", "MONO#", "EO#", "BASO#", "IG#", "NRBC#"]
PCT_GROUP = ["NEUT%", "LYMPH%", "MONO%", "EO%", "BASO%", "IG%", "NRBC%"]
RBC_GROUP = ["RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDW-SD", "RDW-CV", "MicroR", "MacroR"]
PLT_GROUP = ["PLT", "PLT-F", "PDW", "MPV", "P-LCR", "PCT"]
RESEARCH_GROUP = ["TNC", "WBC-N", "TNC-N", "BA-N#", "BA-N%", "WBC-D", "TNC-D", "NEUT#&", "NEUT%&",
                  "LYMPH#&", "LYMPH%&", "HFLC#", "HFLC%", "BA-D#", "BA-D%"]
CPD_GROUP = ["NE-SSC", "NE-SFL", "NE-FSC", "LY-X", "LY-Y", "LY-Z", "MO-X", "MO-Y", "MO-Z",
             "NE-WX", "NE-WY", "NE-WZ", "LY-WX", "LY-WY", "LY-WZ", "MO-WX", "MO-WY"]

# 71 usable raw features (paper's 72 minus MO-WZ, which is absent from the public files)
FEATURES = [RAW_TO_SHORT[c] for c in RAW_TO_SHORT]
FEATURE_GROUPS = {"Q-flags": QFLAGS, "WBC counts": WBC_GROUP, "Cell percentages": PCT_GROUP,
                  "Red cells": RBC_GROUP, "Platelets": PLT_GROUP,
                  "Extended WBC": RESEARCH_GROUP, "Cell-population": CPD_GROUP}
assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(FEATURES)

# Key routine markers reported in the literature (used throughout the EDA / RQ2)
KEY_FEATURES = ["WBC", "NEUT#", "LYMPH#", "MONO#", "NEUT%", "LYMPH%", "MONO%",
                "HGB", "RBC", "HCT", "MCV", "RDW-CV", "PLT", "MPV", "IG#", "NRBC#"]

# --------------------------------------------------------------------------- #
# 3. Cohort definitions (file -> site / split / cohort)
# --------------------------------------------------------------------------- #
INTERNAL_SITES = {"A", "B", "C"}       # sites that also contributed training data
EXTERNAL_SITES = {"D", "E", "F", "G"}  # hospitals never seen in training


def file_meta(file_name: str) -> dict:
    stem = Path(file_name).stem
    split = stem.split("_")[0]  # train / valid / test
    m = re.search(r"site_([A-G])$", stem)
    site = m.group(1) if m else "RW"
    if split == "train":
        cohort = "development"
    elif split == "valid":
        cohort = "internal_validation" if site in INTERNAL_SITES else "external_validation"
    else:
        cohort = "real_world_test"
    return {"source_file": stem, "split": split, "site": site, "cohort": cohort}


def load_raw(raw_dir: str | Path) -> pd.DataFrame:
    """Load all raw site CSVs exactly as released, plus file metadata columns."""
    frames = []
    for f in sorted(Path(raw_dir).glob("*.csv")):
        df = pd.read_csv(f)
        for k, v in file_meta(f.name).items():
            df[k] = v
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 4. FIXED CLEANING RULES
# --------------------------------------------------------------------------- #
GRID = 1e4  # instruments report <=3 decimals (HCT as fraction uses 3) -> 4dp grid is safe
RBC_ZERO_INVALID = ["RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDW-SD", "RDW-CV"]
PLT_INDEX_BLOCK = ["PDW", "MPV", "P-LCR", "PCT"]
DIFF_PCT = ["NEUT%", "LYMPH%", "MONO%", "EO%", "BASO%"]
DIFF_BLOCK = ["NEUT#", "LYMPH#", "MONO#", "EO#", "BASO#", "NEUT%", "LYMPH%", "MONO%", "EO%", "BASO%",
              "IG#", "IG%", "NEUT#&", "NEUT%&", "LYMPH#&", "LYMPH%&", "HFLC#", "HFLC%",
              "BA-D#", "BA-D%", "NE-SSC", "NE-SFL", "NE-FSC"]
MCHC_MAX = 500.0  # g/L. Physiological ceiling ~370; >500 = optical/agglutination artefact
META_COLS = ["sample_id", "source_file", "split", "site", "cohort", "y"]

CLEANING_RULES = [
    ("R1", "Rename columns", "Raw headers -> short team names (e.g. 'WBC(10^9/L)' -> 'WBC'). Label 'leukemia'/'healthy' -> y = 1/0."),
    ("R2", "Drop blast-suspicion flags", "Q-Flag(Blasts/Abn Lympho?), Q-Flag(Blasts?), Q-Flag(Abn Lympho?) dropped: not in the paper's 72 features, 50-100% missing/constant at every site, and they are the analyser's own blast message (shortcut to the label)."),
    ("R3", "Disguised missing -> NaN", "In each file, each column contains exactly one value off the instrument's precision grid (e.g. 104.2533936651584). It is that file's column mean used as a fill-in. Every off-grid value is set back to NaN so imputation can be learned inside training folds."),
    ("R4", "HCT unit harmonisation", "Files whose median HCT < 1 (sites B and C) store HCT as a fraction (L/L). Multiply by 100 so HCT is in % everywhere."),
    ("R5", "RBC panel zero -> NaN", "RBC, HGB, HCT, MCV, MCH, MCHC, RDW-SD, RDW-CV equal to 0 are impossible in a living patient (failed measurement) -> NaN."),
    ("R6", "Platelet indices not computed -> NaN", "When MPV = 0 the analyser did not compute the platelet-index block; PDW, MPV, P-LCR, PCT are set to NaN for that record."),
    ("R7", "Differential not performed -> NaN", "When NEUT%+LYMPH%+MONO%+EO%+BASO% = 0 no differential was reported; the whole differential block (23 columns) is set to NaN. WBC is kept."),
    ("R8", "Cell-population zero -> NaN", "NE-/LY-/MO- scattergram positions and widths equal to 0 are 'not measured' (a channel position cannot be 0) -> NaN."),
    ("R9", "Impossible MCHC", f"MCHC > {MCHC_MAX:.0f} g/L is physically impossible (agglutination/optical artefact); MCHC and MCH (both derived from the corrupted RBC count) -> NaN."),
    ("R10", "Duplicates", "Exact duplicate records (all features + label) within a file: keep the first. Identical features with conflicting labels: drop all copies."),
    ("R11", "No core CBC", "Records where WBC, HGB and PLT are all missing carry no screening information and are removed."),
    ("R12", "Data types", "All features float64 (NaN-capable), y int8, site/split/cohort categorical. No rows are removed for being extreme - extreme counts are the disease signal."),
]


def _off_grid(s: pd.Series) -> pd.Series:
    v = s.to_numpy(dtype=float)
    return pd.Series(np.abs(v * GRID - np.round(v * GRID)) > 1e-3, index=s.index)


def clean_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply R1-R12. Returns (cleaned_df, log_df) where log_df counts changes per rule & file."""
    df = raw.copy()
    log = []

    def _log(rule, n, file="ALL", detail=""):
        log.append({"rule": rule, "file": file, "n_changed": int(n), "detail": detail})

    # R10 (done first, on raw values, per file)
    raw_feats = [c for c in df.columns if c in RAW_TO_SHORT or c in DROPPED_BLAST_FLAGS]
    keep = pd.Series(True, index=df.index)
    for f, g in df.groupby("source_file"):
        conflict = g.groupby(raw_feats)["leukemia_label"].transform("nunique") > 1
        dup = g.duplicated(raw_feats + ["leukemia_label"], keep="first")
        drop_idx = g.index[conflict | dup]
        keep.loc[drop_idx] = False
        if len(drop_idx):
            _log("R10", len(drop_idx), f, f"{int(dup.sum())} exact dups, {int(conflict.sum())} conflicting-label rows")
    df = df.loc[keep].copy()

    # R1 rename + label
    df["y"] = df["leukemia_label"].astype(str).str.strip().str.lower().map({"healthy": 0, "leukemia": 1})
    assert df["y"].notna().all(), "unknown label value"
    df = df.rename(columns={**RAW_TO_SHORT, **DROPPED_BLAST_FLAGS})
    # R2
    df = df.drop(columns=list(DROPPED_BLAST_FLAGS.values()) + ["leukemia_label"])
    _log("R2", 3, "ALL", "columns dropped")
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    for f, idx in df.groupby("source_file").groups.items():
        sub = df.loc[idx]
        # R3 disguised missing
        n3 = 0
        for c in FEATURES:
            m = _off_grid(sub[c])
            if m.any():
                df.loc[m[m].index, c] = np.nan
                n3 += int(m.sum())
        _log("R3", n3, f, "cells set to NaN")
        # R4 HCT unit
        if df.loc[idx, "HCT"].median() < 1:
            df.loc[idx, "HCT"] = (df.loc[idx, "HCT"] * 100).round(2)
            _log("R4", len(idx), f, "HCT x100 (fraction -> %)")

    # R5
    for c in RBC_ZERO_INVALID:
        m = df[c] == 0
        df.loc[m, c] = np.nan
        _log("R5", m.sum(), "ALL", c)
    # R6
    m = df["MPV"] == 0
    df.loc[m, PLT_INDEX_BLOCK] = np.nan
    _log("R6", m.sum(), "ALL", "records")
    # R7
    m = df[DIFF_PCT].sum(axis=1, min_count=1) == 0
    df.loc[m, DIFF_BLOCK] = np.nan
    _log("R7", m.sum(), "ALL", "records")
    # R8
    for c in CPD_GROUP:
        mm = df[c] == 0
        df.loc[mm, c] = np.nan
        _log("R8", mm.sum(), "ALL", c)
    # R9
    m = df["MCHC"] > MCHC_MAX
    df.loc[m, ["MCHC", "MCH"]] = np.nan
    _log("R9", m.sum(), "ALL", "records")

    # R11 records without any core CBC value
    m = df[["WBC", "HGB", "PLT"]].isna().all(axis=1)
    df = df.loc[~m]
    _log("R11", m.sum(), "ALL", "records removed")

    # R12 dtypes + column order
    df["y"] = df["y"].astype("int8")
    for c in ["site", "split", "cohort", "source_file"]:
        df[c] = df[c].astype("category")
    df = df[META_COLS + FEATURES].reset_index(drop=True)
    return df, pd.DataFrame(log)


def build_cleaned_dataset(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return clean_frame(load_raw(raw_dir))


def load_clean(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["y"] = df["y"].astype("int8")
    for c in ["site", "split", "cohort", "source_file"]:
        df[c] = df[c].astype("category")
    return df


# --------------------------------------------------------------------------- #
# 5. PART C - DERIVED (RATIO) FEATURES
#    Row-wise, deterministic formulas -> no information learned from data,
#    so they can be added to every cohort before modelling (no leakage).
# --------------------------------------------------------------------------- #
EPS = 0.01  # = analyser resolution for differential counts (10^9/L); avoids division by 0

DERIVED_DEFS = {
    # name: (formula text, function)
    "NLR":  ("NEUT# / (LYMPH# + 0.01)",            lambda d: d["NEUT#"] / (d["LYMPH#"] + EPS)),
    "PLR":  ("PLT / (LYMPH# + 0.01)",              lambda d: d["PLT"] / (d["LYMPH#"] + EPS)),
    "MLR":  ("MONO# / (LYMPH# + 0.01)",            lambda d: d["MONO#"] / (d["LYMPH#"] + EPS)),
    "SII":  ("PLT x NEUT# / (LYMPH# + 0.01)",      lambda d: d["PLT"] * d["NEUT#"] / (d["LYMPH#"] + EPS)),
    "dNLR": ("NEUT# / (WBC - NEUT# + 0.01)",       lambda d: d["NEUT#"] / ((d["WBC"] - d["NEUT#"]).clip(lower=0) + EPS)),
}
DERIVED = list(DERIVED_DEFS)
DERIVED_REFERENCE = {
    "NLR": "Neutrophil-to-lymphocyte ratio; normal adult range roughly 0.8-3.5",
    "PLR": "Platelet-to-lymphocyte ratio; normal roughly 50-200",
    "MLR": "Monocyte-to-lymphocyte ratio; normal roughly 0.1-0.35",
    "SII": "Systemic immune-inflammation index (PLT x NLR); normal roughly 200-700",
    "dNLR": "Derived NLR: neutrophils / (all other white cells)",
}

# Standard routine CBC (5-part differential + red cells + platelets) available in any lab
CORE_FEATURES = ["WBC", "NEUT#", "LYMPH#", "MONO#", "EO#", "BASO#",
                 "NEUT%", "LYMPH%", "MONO%", "EO%", "BASO%",
                 "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDW-SD", "RDW-CV",
                 "PLT", "MPV", "PDW", "P-LCR", "PCT"]

# Right-skewed, non-negative count-type features -> log1p inside the LR pipeline
LOG_FEATURES = (["WBC", "NEUT#", "LYMPH#", "MONO#", "EO#", "BASO#", "IG#", "NRBC#", "NRBC%", "IG%",
                 "PLT", "PLT-F", "PCT", "TNC", "WBC-N", "TNC-N", "WBC-D", "TNC-D", "BA-N#", "BA-D#",
                 "NEUT#&", "LYMPH#&", "HFLC#", "HFLC%", "BASO%", "EO%", "BA-N%", "BA-D%", "MicroR"]
                + DERIVED)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append NLR, PLR, MLR, SII, dNLR. NaN if any component is missing."""
    out = df.copy()
    for name, (_, fn) in DERIVED_DEFS.items():
        out[name] = fn(out).astype(float)
    return out
