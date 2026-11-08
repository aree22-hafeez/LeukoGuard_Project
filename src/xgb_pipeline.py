"""Fold-local preprocessing and training for the Member 3 XGBoost model."""
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from leuko_common import SEED
from lr_pipeline import smote


class XGBScreen:
    def __init__(self, features, params, imbalance, jobs):
        self.features = features
        self.params = params
        self.imbalance = imbalance
        self.jobs = jobs

    def fit(self, X, y):
        y = y.to_numpy(dtype=int)
        self.imputer_ = SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )
        values = self.imputer_.fit_transform(X[self.features])
        self.feature_names_out_ = [
            name.replace("missingindicator_", "MISSING_", 1)
            for name in self.imputer_.get_feature_names_out()
        ]
        weight = 1.0
        if self.imbalance == "balanced":
            weight = (y == 0).sum() / (y == 1).sum()
        elif self.imbalance == "smote":
            # Interpolate only continuous CBC values. Restore binary indicators.
            values, y = smote(values, y, seed=SEED)
            values[:, len(self.features):] = values[:, len(self.features):] >= 0.5
        self.model_ = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            random_state=SEED, n_jobs=self.jobs, scale_pos_weight=weight,
            **self.params,
        ).fit(values, y)
        return self

    def transform(self, X):
        return self.imputer_.transform(X[self.features])

    def predict_proba(self, X):
        return self.model_.predict_proba(self.transform(X))
