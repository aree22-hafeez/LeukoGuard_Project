"""
LeukoGuard - Logistic Regression pipeline, evaluation helpers and statistical tests
(Member 1, Parts D & E).

Everything that LEARNS from data (clipping limits, imputation medians, scaling,
SMOTE neighbours, LR weights) lives inside `LRScreen.fit`, so when LRScreen is
used inside cross-validation it only ever sees the training fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_SK_NEW = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)


def make_lr(C: float, l1_ratio: float, class_weight=None, seed: int = 42, max_iter: int = 3000):
    """LogisticRegression that works on old (<1.8) and new (>=1.8) scikit-learn.
    l1_ratio = 0 -> L2 (ridge), 1 -> L1 (lasso), in between -> elastic net."""
    if l1_ratio == 0:      # L2: lbfgs is fast and exact
        kw = dict(solver="lbfgs") if _SK_NEW else dict(penalty="l2", solver="lbfgs")
    elif l1_ratio == 1:    # L1: liblinear (coordinate descent) is ~50x faster than saga here
        kw = dict(solver="liblinear", l1_ratio=1.0) if _SK_NEW else dict(penalty="l1", solver="liblinear")
    else:                  # elastic net needs saga (slow on this data; not used in the final grid)
        kw = dict(solver="saga", l1_ratio=l1_ratio) if _SK_NEW else dict(penalty="elasticnet", solver="saga", l1_ratio=l1_ratio)
    return LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter, random_state=seed, **kw)


def smote(X: np.ndarray, y: np.ndarray, k: int = 5, ratio: float = 1.0, seed: int = 42):
    """Minimal SMOTE (Chawla et al. 2002): synthetic minority samples on the line between a
    minority sample and one of its k nearest minority neighbours, until minority = ratio x majority.
    Same algorithm as imblearn.over_sampling.SMOTE; written here so the project has no extra dependency."""
    rng = np.random.default_rng(seed)
    X_min = X[y == 1]
    n_new = int(ratio * (y == 0).sum()) - len(X_min)
    if n_new <= 0:
        return X, y
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    neigh = nn.kneighbors(X_min, return_distance=False)[:, 1:]
    base = rng.integers(0, len(X_min), n_new)
    pick = neigh[base, rng.integers(0, k, n_new)]
    gap = rng.random((n_new, 1))
    X_new = X_min[base] + gap * (X_min[pick] - X_min[base])
    return np.vstack([X, X_new]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])


class LRScreen(BaseEstimator, ClassifierMixin):
    """Full LR screening pipeline (fit only on training data):
       1. log1p on right-skewed count features (fixed list, no learning)
       2. clip each feature to its training 0.5-99.5 percentiles (learned)
       3. median imputation + binary missing-indicator columns (learned)
       4. standard scaling (learned)
       5. class imbalance: 'none' | 'balanced' (class weights) | 'smote' (training data only)
       6. regularised logistic regression (C, l1_ratio)
    """

    def __init__(self, features=None, log_features=None, use_log=True, clip=True,
                 C=1.0, l1_ratio=0.0, imbalance="balanced", indicator=True, seed=42):
        self.features = features
        self.log_features = log_features
        self.use_log = use_log
        self.clip = clip
        self.C = C
        self.l1_ratio = l1_ratio
        self.imbalance = imbalance
        self.indicator = indicator
        self.seed = seed

    # ---- preprocessing --------------------------------------------------
    def _log(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.features].astype(float).copy()
        if self.use_log and self.log_features:
            cols = [c for c in self.log_features if c in X.columns]
            X[cols] = np.log1p(X[cols].clip(lower=0))
        return X

    def _prep_fit(self, X):
        Z = self._log(X)
        if self.clip:
            self.lo_, self.hi_ = Z.quantile(0.005), Z.quantile(0.995)
            Z = Z.clip(self.lo_, self.hi_, axis=1)
        self.imputer_ = SimpleImputer(strategy="median", add_indicator=self.indicator)
        A = self.imputer_.fit_transform(Z)
        self.scaler_ = StandardScaler().fit(A)
        names = list(self.imputer_.get_feature_names_out(Z.columns))
        self.feature_names_out_ = [n.replace("missingindicator_", "MISSING_") for n in names]
        return self.scaler_.transform(A)

    def transform(self, X):
        Z = self._log(X)
        if self.clip:
            Z = Z.clip(self.lo_, self.hi_, axis=1)
        return self.scaler_.transform(self.imputer_.transform(Z))

    # ---- model -----------------------------------------------------------
    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        A = self._prep_fit(X)
        cw = "balanced" if self.imbalance == "balanced" else None
        if self.imbalance == "smote":
            A, y = smote(A, y, seed=self.seed)
        self.lr_ = make_lr(self.C, self.l1_ratio, class_weight=cw, seed=self.seed).fit(A, y)
        self.classes_ = self.lr_.classes_
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba(self.transform(X))

    def decision_function(self, X):
        return self.lr_.decision_function(self.transform(X))

    def coefficients(self) -> pd.Series:
        """Coefficients on the standardised scale (log-odds per 1 SD of the transformed feature)."""
        return pd.Series(self.lr_.coef_.ravel(), index=self.feature_names_out_)


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------
def oof_predict(model, X, y, folds):
    """Out-of-fold probabilities for a list of (train_idx, test_idx)."""
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


# ---------------------------------------------------------------------------
# Metrics & thresholds
# ---------------------------------------------------------------------------
def threshold_for_recall(y, p, target=0.90):
    """Largest threshold whose recall >= target (= fewest false alarms at that recall)."""
    y, p = np.asarray(y), np.asarray(p)
    pos = np.sort(p[y == 1])[::-1]
    k = int(np.ceil(target * len(pos)))  # need at least k positives flagged
    return float(pos[k - 1])


def spec_at_recall(y, p, target=0.90):
    t = threshold_for_recall(y, p, target)
    y, p = np.asarray(y), np.asarray(p)
    return float(((p < t) & (y == 0)).sum() / (y == 0).sum())


def metrics_at(y, p, thr):
    y, p = np.asarray(y).astype(int), np.asarray(p)
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "n": len(y), "prevalence_%": 100 * y.mean(),
        "recall": recall_score(y, yhat, zero_division=0),
        "precision": precision_score(y, yhat, zero_division=0),
        "F1": f1_score(y, yhat, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ROC_AUC": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "PR_AUC": average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "flagged_%": 100 * yhat.mean(),
    }


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def corrected_resampled_ttest(diffs, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV.
    diffs: per-fold metric differences (model B - model A). Returns (mean, t, p)."""
    d = np.asarray(diffs, float)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return d.mean(), np.nan, 1.0
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return d.mean(), t, p


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong_test(y, p1, p2):
    """Paired DeLong test for two ROC-AUCs on the same patients (Sun & Xu 2014 fast version).
    Returns (auc1, auc2, z, p)."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")
    y = y[order]
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    m = int(y.sum())
    n = len(y) - m
    tx, ty, tz = (np.empty((2, m)), np.empty((2, n)), np.empty((2, m + n)))
    for r in range(2):
        tx[r] = _midrank(preds[r, :m])
        ty[r] = _midrank(preds[r, m:])
        tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[1] - aucs[0]) / np.sqrt(var) if var > 0 else 0.0
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))


def bootstrap_diff(y, p1, p2, metric, n_boot=1000, seed=42, strata=None):
    """Paired (stratified by class) bootstrap of metric(p2) - metric(p1). Returns (diff, lo, hi, p_two_sided)."""
    rng = np.random.default_rng(seed)
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
    obs = metric(y, p2) - metric(y, p1)
    ds = np.empty(n_boot)
    for b in range(n_boot):
        s = np.concatenate([rng.choice(idx1, len(idx1)), rng.choice(idx0, len(idx0))])
        ds[b] = metric(y[s], p2[s]) - metric(y[s], p1[s])
    lo, hi = np.percentile(ds, [2.5, 97.5])
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return obs, lo, hi, max(p, 1 / n_boot)


def mcnemar(y, yhat1, yhat2):
    """Exact McNemar test on the disagreements of two classifiers' correct/incorrect decisions."""
    y = np.asarray(y)
    c1, c2 = np.asarray(yhat1) == y, np.asarray(yhat2) == y
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p
