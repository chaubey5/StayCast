"""Inference pipeline: load the saved artifact and produce real predictions.

Flow for every request
  validate input -> build features with the SAME FeatureBuilder used in training
  -> saved model.predict -> clip at 0 -> residual-based interval -> sanity checks
  -> structured result.
If the artifact cannot be loaded we raise; we never fabricate a prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.explainability import local_explanation
from src.feature_engineering import FEATURE_COLUMNS, NUMERIC_FEATURES, FeatureBuilder, FeatureError
from src.forecasting import SplitDates, split_of_date
from src.preprocessing import load_clean

log = logging.getLogger("staycast.predict")
REQUIRED_KEYS = {"model", "model_name", "feature_columns", "interval_table", "reference_row",
                 "feature_stats", "split", "metadata", "category_max_demand"}


class ModelNotAvailableError(RuntimeError):
    """Artifact missing, corrupt or incompatible."""


class InputValidationError(ValueError):
    """The caller's request is invalid (bad date, category, horizon)."""


def _hgroup(h: int) -> str:
    return "1-7" if h <= 7 else "8-14"


class Forecaster:
    def __init__(self, artifact: dict, builder: FeatureBuilder, model_path: Path):
        self.artifact, self.builder, self.model_path = artifact, builder, Path(model_path)
        self.model = artifact["model"]
        s = artifact["split"]
        self.split = SplitDates(*(pd.Timestamp(s[k]) for k in ("first", "train_end", "val_end", "test_end")))
        self.categories = builder.cats

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_path: Path | str = config.MODEL_FILE, clean_path: Path | str = config.CLEAN_FILE) -> "Forecaster":
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelNotAvailableError(f"Model artifact not found at {model_path}. Run: python -m src.train")
        try:
            artifact = joblib.load(model_path)
        except Exception as exc:  # corrupt file, version mismatch...
            raise ModelNotAvailableError(f"Could not load model artifact: {exc}") from exc
        missing = REQUIRED_KEYS - set(artifact)
        if missing:
            raise ModelNotAvailableError(f"Artifact is missing keys: {sorted(missing)}")
        if list(artifact["feature_columns"]) != FEATURE_COLUMNS:
            raise ModelNotAvailableError("Artifact feature schema differs from current code - retrain the model.")
        try:
            builder = FeatureBuilder(load_clean(clean_path))
        except FileNotFoundError as exc:
            raise ModelNotAvailableError(str(exc)) from exc
        log.info("Loaded %s v%s", artifact["model_name"], artifact["metadata"]["version"])
        return cls(artifact, builder, model_path)

    # --------------------------------------------------------- validation
    def _check_category(self, cat: str) -> str:
        if cat not in self.categories:
            raise InputValidationError(f"Unknown room category '{cat}'. Valid: {self.categories}")
        return cat

    def _check_horizon(self, h: int) -> int:
        if not isinstance(h, (int, np.integer)) or h not in config.HORIZONS:
            raise InputValidationError(f"Horizon must be an integer between {config.HORIZONS[0]} and {config.HORIZONS[-1]} days.")
        return int(h)

    def _check_origin(self, origin: pd.Timestamp, horizon: int) -> None:
        lo, hi = self.builder.valid_origin_range(horizon)
        if not (lo <= origin <= hi):
            raise InputValidationError(
                f"For a {horizon}-day horizon the forecast origin must be between {lo.date()} and {hi.date()} "
                f"(history before {lo.date()} is incomplete; booking data ends {self.builder.last_date.date()})."
            )

    @staticmethod
    def _to_date(value, name: str) -> pd.Timestamp:
        try:
            return pd.Timestamp(value).normalize()
        except Exception as exc:
            raise InputValidationError(f"{name} '{value}' is not a valid date (use YYYY-MM-DD).") from exc

    # --------------------------------------------------------- core predict
    def _predict_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        pred = self.model.predict(rows[FEATURE_COLUMNS])
        keys = rows["horizon"].map(_hgroup) + "|" + rows["room_category"]
        table = self.artifact["interval_table"]
        lo = np.array([table[k][0] for k in keys])
        hi = np.array([table[k][1] for k in keys])
        out = rows[["origin_date", "target_date", "horizon", "room_category", "y"]].copy()
        out["prediction"] = pred
        out["lower"] = np.clip(pred + lo, 0, None)
        out["upper"] = np.clip(pred + hi, 0, None)
        out["split"] = [split_of_date(d, self.split) for d in out["target_date"]]
        out["otb"] = rows["otb"].to_numpy()
        return out

    def _warnings(self, row: pd.Series, feats: pd.Series) -> list[str]:
        w = []
        stats = self.artifact["feature_stats"]
        unusual = []
        for c in NUMERIC_FEATURES:
            s = stats[c]
            span = max(s["max"] - s["min"], 1e-9)
            if feats[c] > s["max"] + 0.25 * span or feats[c] < s["min"] - 0.25 * span:
                unusual.append(c)
        if unusual:
            w.append("Unusual input: " + ", ".join(unusual[:5]) + " lie outside the range seen in training; treat with extra caution.")
        cap = self.artifact["category_max_demand"][row["room_category"]]
        if row["prediction"] > 1.2 * cap:
            w.append("Prediction exceeds 120% of the highest demand ever recorded for this category.")
        if row["split"] != "test":
            w.append(f"This date lies in the {row['split']} period that the model was fitted on, so the error is "
                     "optimistic here. Only test-period dates are genuinely held-out.")
        if row["horizon"] > 7:
            w.append("Horizon above 7 days: fewer bookings are on the books, so uncertainty is larger.")
        if (row["upper"] - row["lower"]) > 0.5 * max(row["prediction"], 1):
            w.append("Prediction interval is wide relative to the estimate; consider a manual review.")
        return w

    # ------------------------------------------------------------- public
    def predict_one(self, target_date, room_category: str, horizon: int, explain: bool = True) -> dict:
        target = self._to_date(target_date, "target_date")
        cat, h = self._check_category(room_category), self._check_horizon(horizon)
        origin = target - pd.Timedelta(days=h)
        self._check_origin(origin, h)
        try:
            rows = self.builder.make_rows([origin], [h], [cat])
        except FeatureError as exc:
            raise InputValidationError(str(exc)) from exc
        res = self._predict_rows(rows).iloc[0]
        feats = rows.iloc[0]
        out = {
            "origin_date": str(origin.date()), "target_date": str(target.date()), "horizon": h,
            "room_category": cat, "prediction": round(float(res["prediction"]), 1),
            "lower": round(float(res["lower"]), 1), "upper": round(float(res["upper"]), 1),
            "interval_coverage": self.artifact["interval_coverage"],
            "data_split": res["split"], "bookings_on_the_books": int(res["otb"]),
            "recorded_actual": float(res["y"]),
            "model_name": self.artifact["model_name"], "model_version": self.artifact["metadata"]["version"],
            "warnings": self._warnings(res, feats),
            "feature_values": {c: (feats[c].item() if hasattr(feats[c], "item") else feats[c]) for c in FEATURE_COLUMNS},
        }
        if explain:
            out["explanation"] = local_explanation(self.model, rows[FEATURE_COLUMNS], self.artifact["reference_row"])
        out["history_context"] = self.history_context(target, cat)
        return out

    def forecast(self, origin_date, horizon: int, categories: list[str] | None = None) -> pd.DataFrame:
        """All target dates origin+1 .. origin+horizon for the chosen categories."""
        origin = self._to_date(origin_date, "origin_date")
        h_max = self._check_horizon(horizon)
        cats = [self._check_category(c) for c in (categories or self.categories)]
        self._check_origin(origin, h_max)
        rows = self.builder.make_rows([origin], list(range(1, h_max + 1)), cats)
        return self._predict_rows(rows).sort_values(["target_date", "room_category"]).reset_index(drop=True)

    def backtest(self, start, end, horizon: int, categories: list[str] | None = None) -> pd.DataFrame:
        """Rolling h-day-ahead predictions for every target date in [start, end]."""
        s, e = self._to_date(start, "start"), self._to_date(end, "end")
        h = self._check_horizon(horizon)
        if e < s:
            raise InputValidationError("end must be on/after start.")
        lo, hi = self.builder.valid_origin_range(h)
        s, e = max(s, lo + pd.Timedelta(days=h)), min(e, hi + pd.Timedelta(days=h))
        if e < s:
            raise InputValidationError("No valid target dates in that range for the chosen horizon.")
        cats = [self._check_category(c) for c in (categories or self.categories)]
        rows = self.builder.make_rows(pd.date_range(s - pd.Timedelta(days=h), e - pd.Timedelta(days=h)), [h], cats)
        return self._predict_rows(rows).sort_values(["target_date", "room_category"]).reset_index(drop=True)

    def history(self, start, end, categories: list[str] | None = None) -> pd.DataFrame:
        """Observed net room-night demand (the target series) in long format."""
        s, e = self._to_date(start, "start"), self._to_date(end, "end")
        lo, hi = pd.Timestamp(config.DEMAND_START), self.builder.last_date
        s, e = max(s, lo), min(e, hi)
        if e < s:
            raise InputValidationError(f"No history in range (available {lo.date()} .. {hi.date()}).")
        cats = [self._check_category(c) for c in (categories or self.categories)]
        frame = self.builder.demand_frame().loc[s:e, cats]
        gross = self.builder.demand_frame(gross=True).loc[s:e, cats]
        long = frame.stack().rename("room_nights").reset_index()
        long.columns = ["date", "room_category", "room_nights"]
        long["gross_room_nights"] = gross.stack().to_numpy()
        return long

    def history_context(self, target: pd.Timestamp, cat: str) -> dict:
        """Demand on the same weekday over previous weeks + category range."""
        series = self.builder.demand_frame()[cat]
        same = [(target - pd.Timedelta(days=7 * k)) for k in range(1, 7)]
        vals = {str(d.date()): float(series.get(d, np.nan)) for d in same if d >= pd.Timestamp(config.DEMAND_START)}
        recent = series.loc[:target - pd.Timedelta(days=1)].tail(28)
        return {"same_weekday_previous_weeks": vals, "mean_last_28d": round(float(recent.mean()), 1),
                "category_all_time_max": self.artifact["category_max_demand"][cat]}

    def info(self) -> dict:
        md = self.artifact["metadata"]
        return {"model_name": self.artifact["model_name"], "version": md["version"], "trained_at_utc": md["trained_at_utc"],
                "fit_on": md["fit_on"], "config": md["config"], "split": self.artifact["split"],
                "feature_columns": self.artifact["feature_columns"], "random_seed": md["random_seed"],
                "library_versions": md["library_versions"], "raw_data_sha256": md["raw_data_sha256"],
                "interval_method": "Residual quantiles from validation, per horizon group and room category",
                "interval_coverage": self.artifact["interval_coverage"],
                "categories": self.categories, "horizons": config.HORIZONS,
                "data_last_date": str(self.builder.last_date.date()),
                "first_valid_origin": str(self.builder.first_origin.date())}


def main() -> None:
    """CLI demo: python -m src.predict 2017-08-25 A 7"""
    import json
    import sys

    target, cat, h = (sys.argv[1:4] + ["2017-08-25", "A", "7"][len(sys.argv) - 1:])[:3]
    f = Forecaster.load()
    res = f.predict_one(target, cat, int(h))
    res.pop("feature_values")
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
