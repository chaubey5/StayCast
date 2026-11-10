"""Chronological split, naive seasonal baseline and the candidate models."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.feature_engineering import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

BASELINE_COLUMN = "base_same_dow"
MODEL_NAMES = ["Linear Regression", "Random Forest", "HistGradientBoosting"]


# ------------------------------------------------------------------ split
@dataclass(frozen=True)
class SplitDates:
    """Inclusive end dates of each period, defined on TARGET dates."""
    first: pd.Timestamp
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict:
        return {k: str(v.date()) for k, v in asdict(self).items()}


def chronological_split(target_dates: pd.Series, fractions=config.SPLIT_FRACTIONS) -> SplitDates:
    """70/15/15 split by unique target date (never random)."""
    days = np.sort(pd.to_datetime(target_dates).unique())
    n = len(days)
    i_train = int(round(n * fractions[0])) - 1
    i_val = int(round(n * (fractions[0] + fractions[1]))) - 1
    return SplitDates(*(pd.Timestamp(x) for x in (days[0], days[i_train], days[i_val], days[-1])))


def assign_split(df: pd.DataFrame, split: SplitDates) -> pd.Series:
    t = df["target_date"]
    return pd.Series(
        np.where(t <= split.train_end, "train", np.where(t <= split.val_end, "validation", "test")),
        index=df.index,
    )


def split_of_date(date, split: SplitDates) -> str:
    d = pd.Timestamp(date)
    return "train" if d <= split.train_end else "validation" if d <= split.val_end else "test"


# --------------------------------------------------------------- baseline
def naive_seasonal_predict(df: pd.DataFrame) -> np.ndarray:
    """Demand on the same weekday of the most recent fully-observed week.

    For horizon h the reference date is d - 7*ceil(h/7), which is always <= the
    origin, so the baseline uses exactly the information the models get.
    """
    return df[BASELINE_COLUMN].to_numpy(dtype=float)


# ----------------------------------------------------------------- models
def _preprocessor(scale: bool, numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )


def make_estimator(name: str, params: dict | None = None):
    p = dict(params or {})
    if name == "Linear Regression":
        return LinearRegression(**p)
    if name == "Random Forest":
        return RandomForestRegressor(random_state=config.RANDOM_SEED, n_jobs=-1, **p)
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=config.RANDOM_SEED, **p)
    raise ValueError(f"Unknown model {name}")


class DemandModel(BaseEstimator, RegressorMixin):
    """Preprocessing + regressor, optionally learning the correction to the baseline.

    residual=True  -> fit on (y - naive_baseline); predict baseline + correction.
    residual=False -> fit on y directly.
    Output is clipped at 0: room-night demand cannot be negative.
    """

    def __init__(self, name: str = "Random Forest", params: dict | None = None, residual: bool = False,
                 exclude_features: tuple = ()):
        self.name, self.params, self.residual = name, params, residual
        self.exclude_features = exclude_features  # used only for the ablation study

    def _offset(self, X: pd.DataFrame) -> np.ndarray:
        return X[BASELINE_COLUMN].to_numpy(dtype=float) if self.residual else 0.0

    def fit(self, X: pd.DataFrame, y):
        ex = set(self.exclude_features)
        self.used_features_ = [c for c in FEATURE_COLUMNS if c not in ex]
        num = [c for c in NUMERIC_FEATURES if c not in ex]
        cat = [c for c in CATEGORICAL_FEATURES if c not in ex]
        self.pipeline_ = Pipeline(
            [
                ("prep", _preprocessor(self.name == "Linear Regression", num, cat)),
                ("model", make_estimator(self.name, self.params)),
            ]
        )
        self.pipeline_.fit(X[self.used_features_], np.asarray(y, dtype=float) - self._offset(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.pipeline_.predict(X[self.used_features_]) + self._offset(X)
        return np.clip(raw, 0.0, None)

    def describe(self) -> dict:
        return {"model": self.name, "params": self.params or {}, "residual_target": self.residual,
                "random_seed": config.RANDOM_SEED}


# Small, deliberately modest search spaces (selection is on validation only).
CANDIDATE_GRID: dict[str, list[dict]] = {
    "Linear Regression": [{}],
    "Random Forest": [
        {"n_estimators": 200, "min_samples_leaf": 5, "max_features": 0.5},
        {"n_estimators": 200, "min_samples_leaf": 20, "max_features": 0.5},
    ],
    "HistGradientBoosting": [
        {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 40, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_iter": 300, "max_depth": 4, "min_samples_leaf": 40, "l2_regularization": 1.0},
    ],
}
