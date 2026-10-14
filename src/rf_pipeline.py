"""
LeukoGuard - Random Forest pipeline (Member 2, Part B).

Same contract as LRScreen in lr_pipeline.py: everything that LEARNS from data
(median imputation, missing indicators, SMOTE neighbours, trees) happens inside
`RFScreen.fit`, so inside cross-validation it only ever sees the training fold.

Trees are invariant to monotone transforms, so there is no log transform, no
clipping and no scaling here (unlike LRScreen).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from lr_pipeline import smote  # minimal SMOTE, fitted on training folds only


class RFScreen(BaseEstimator, ClassifierMixin):
    """Random Forest screening pipeline (fit only on training data):
       1. median imputation + binary missing-indicator columns (learned)
       2. class imbalance: 'none' | 'balanced' (class weights) | 'smote' (training data only)
       3. Random Forest (learned)
    """

    def __init__(self, features=None, n_estimators=300, max_depth=None, min_samples_leaf=1,
                 max_features="sqrt", imbalance="balanced", indicator=True, seed=42, n_jobs=-1):
        self.features = features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.imbalance = imbalance
        self.indicator = indicator
        self.seed = seed
        self.n_jobs = n_jobs

    def transform(self, X):
        return self.imputer_.transform(X[self.features].astype(float))

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.imputer_ = SimpleImputer(strategy="median", add_indicator=self.indicator)
        A = self.imputer_.fit_transform(X[self.features].astype(float))
        self.feature_names_out_ = [n.replace("missingindicator_", "MISSING_")
                                   for n in self.imputer_.get_feature_names_out(self.features)]
        if self.imbalance == "smote":
            A, y = smote(A, y, seed=self.seed)
        cw = "balanced" if self.imbalance == "balanced" else None
        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
            class_weight=cw, random_state=self.seed, n_jobs=self.n_jobs).fit(A, y)
        self.classes_ = self.rf_.classes_
        return self

    def predict_proba(self, X):
        return self.rf_.predict_proba(self.transform(X))

    def feature_importance(self) -> pd.Series:
        """Impurity-based (mean decrease in Gini) importance, one value per model input."""
        return pd.Series(self.rf_.feature_importances_, index=self.feature_names_out_)
