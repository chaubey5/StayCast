"""Model-agnostic explanations (no SHAP dependency).

* Global: grouped permutation importance on held-out validation rows - how much
  does MAE worsen when a feature group is shuffled?
* Local: group "neutralisation" - replace a feature group with its typical
  training value and see how the prediction moves.

Both are sensitivity measures of THIS model. They are NOT causal effects:
shuffling "holiday" says how much the model relies on it, not what a holiday
does to real demand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.evaluate import mae
from src.feature_engineering import CATEGORICAL_FEATURES, FEATURE_GROUPS, NUMERIC_FEATURES


def reference_row(train_df: pd.DataFrame) -> dict:
    """Typical value per feature: median (numeric) or mode (categorical)."""
    ref = {c: float(train_df[c].median()) for c in NUMERIC_FEATURES}
    ref.update({c: str(train_df[c].mode().iloc[0]) for c in CATEGORICAL_FEATURES})
    return ref


def feature_statistics(df: pd.DataFrame) -> dict:
    """Training ranges used later for unusual-input detection."""
    return {
        c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0)),
            "min": float(df[c].min()), "max": float(df[c].max())}
        for c in NUMERIC_FEATURES
    }


def group_permutation_importance(model, df: pd.DataFrame, n_repeats: int = 5,
                                 seed: int = config.RANDOM_SEED) -> list[dict]:
    """Increase in MAE when each feature group is shuffled (rows permuted jointly)."""
    rng = np.random.default_rng(seed)
    base = mae(df["y"], model.predict(df))
    out = []
    for group, cols in FEATURE_GROUPS.items():
        deltas = []
        for _ in range(n_repeats):
            shuffled = df.copy()
            perm = rng.permutation(len(df))
            shuffled[cols] = df[cols].to_numpy()[perm]
            deltas.append(mae(df["y"], model.predict(shuffled)) - base)
        out.append({"group": group, "features": cols, "mae_increase": float(np.mean(deltas))})
    total = sum(max(o["mae_increase"], 0) for o in out) or 1.0
    for o in out:
        o["share_of_total"] = round(max(o["mae_increase"], 0) / total, 4)
        o["mae_increase"] = round(o["mae_increase"], 3)
    return sorted(out, key=lambda o: o["mae_increase"], reverse=True)


def local_explanation(model, row: pd.DataFrame, reference: dict) -> list[dict]:
    """Per-group sensitivity of one prediction (row = single-row feature frame)."""
    full = float(model.predict(row)[0])
    items = []
    for group, cols in FEATURE_GROUPS.items():
        neutral = row.copy()
        for c in cols:
            neutral[c] = reference[c]
        shift = full - float(model.predict(neutral)[0])
        values = {c: (row.iloc[0][c].item() if hasattr(row.iloc[0][c], "item") else row.iloc[0][c]) for c in cols}
        items.append({"group": group, "contribution": round(shift, 2), "values": values})
    return sorted(items, key=lambda i: abs(i["contribution"]), reverse=True)
