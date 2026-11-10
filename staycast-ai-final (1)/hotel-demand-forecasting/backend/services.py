"""Business logic behind the API: dashboards are built from model output + real data."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.predict import Forecaster, ModelNotAvailableError

log = logging.getLogger("staycast.services")
NOTICE = ("Estimates of booking room-night demand from a statistical model, not guaranteed outcomes and not "
          "physical occupancy. Use as decision support alongside human review.")

LIMITATIONS = [
    "Target is booking room-night demand (non-cancelled bookings), not physical occupancy: the dataset has no room inventory.",
    "Two hotels in Portugal, July 2015 - August 2017 only; patterns may not generalise to other markets or later years.",
    "Only two summers are available, so seasonality (and holidays) is learned from very few repetitions.",
    "Most predictive power comes from bookings already on the books; accuracy falls as the horizon grows.",
    "Exact duplicate rows were removed; if some were genuine bookings the absolute level is understated.",
    "1 booking is assumed to use 1 room; group blocks are not modelled separately.",
    "Importance scores describe what the model relies on, not causes of demand.",
    "Prediction intervals come from validation residuals; they were calibrated on one period and may under-cover in unusual times.",
]
REVIEW_GUIDANCE = [
    "Treat forecasts as an input to staffing and allocation decisions, never as an automatic trigger.",
    "Review any prediction that shows a warning, or whose interval is wide relative to the estimate.",
    "Dates in the training or validation period are in-sample; only test-period dates are genuinely held-out.",
    "Compare against current bookings, events and local knowledge before changing staffing.",
    "Log forecast vs actual each week and re-train if WAPE drifts materially above the reported test value.",
]


class AppState:
    """Holds the loaded model. A failed load is remembered, never faked."""

    def __init__(self, model_path: Path = config.MODEL_FILE, clean_path: Path = config.CLEAN_FILE,
                 metrics_path: Path = config.METRICS_FILE):
        self.model_path, self.clean_path, self.metrics_path = Path(model_path), Path(clean_path), Path(metrics_path)
        self.forecaster: Forecaster | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            try:
                self.forecaster = Forecaster.load(self.model_path, self.clean_path)
                self.error = None
            except ModelNotAvailableError as exc:
                self.forecaster, self.error = None, str(exc)
                log.error("Model unavailable: %s", exc)

    def require(self) -> Forecaster:
        if self.forecaster is None:
            raise ModelNotAvailableError(self.error or "Model not loaded")
        return self.forecaster

    def metrics(self) -> dict:
        if not self.metrics_path.exists():
            raise ModelNotAvailableError(f"{self.metrics_path} not found. Run: python -m src.train")
        return json.loads(self.metrics_path.read_text())


# ------------------------------------------------------------------ helpers
def points_from_frame(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df.itertuples():
        out.append({
            "origin_date": r.origin_date.date(), "target_date": r.target_date.date(), "horizon": int(r.horizon),
            "room_category": r.room_category, "prediction": round(float(r.prediction), 1),
            "lower": round(float(r.lower), 1), "upper": round(float(r.upper), 1), "data_split": r.split,
            "bookings_on_the_books": int(r.otb), "recorded_actual": None if pd.isna(r.y) else float(r.y),
        })
    return out


def overview(state: AppState) -> dict:
    f = state.require()
    metrics = state.metrics()
    h = 14
    origin = f.builder.last_date - pd.Timedelta(days=h)      # latest origin whose 14-day window is fully inside the data
    fc = f.forecast(origin, h)
    daily = fc.groupby("target_date").agg(prediction=("prediction", "sum"), actual=("y", "sum"), otb=("otb", "sum"))
    seven = daily.iloc[:7]
    peak, low = daily["prediction"].idxmax(), daily["prediction"].idxmin()
    prev7 = f.builder.demand_frame().loc[origin - pd.Timedelta(days=6):origin].sum(axis=1).sum()
    by_cat = fc.groupby("room_category")["prediction"].sum().sort_values(ascending=False)
    sel = metrics["selected_model"]
    ms, bs = metrics["models"][sel]["test"], metrics["baseline"]["test"]
    dow = daily["prediction"].groupby(daily.index.dayofweek >= 5).mean()
    wk_gap = (dow.get(True, np.nan) / dow.get(False, np.nan) - 1) if len(dow) == 2 else np.nan
    cover7 = seven["otb"].sum() / max(seven["prediction"].sum(), 1)
    cover14 = daily["otb"].sum() / max(daily["prediction"].sum(), 1)
    history = f.builder.demand_frame().loc[origin - pd.Timedelta(days=27):origin].sum(axis=1)
    insights = [
        f"Peak forecast demand is {daily['prediction'].max():.0f} room-nights on {peak:%a %d %b %Y}; the quietest day is "
        f"{low:%a %d %b} at {daily['prediction'].min():.0f}.",
        f"Category {by_cat.index[0]} carries {by_cat.iloc[0] / by_cat.sum():.0%} of forecast room-nights in the next 14 days.",
        f"The 7-day forecast is {seven['prediction'].sum() / prev7 - 1:+.1%} versus the 7 days just ended (recorded demand).",
        f"Room-nights already on the books at the origin equal {cover7:.0%} of the 7-day and {cover14:.0%} of the 14-day forecast; the model mainly adjusts them for expected cancellations and late bookings, which is why accuracy declines with horizon.",
        f"Selected model test WAPE {ms['overall_1_14d']['WAPE']:.1%} vs {bs['overall_1_14d']['WAPE']:.1%} for the naive seasonal baseline "
        f"(7-day: {ms['horizon_1_7d']['WAPE']:.1%} vs {bs['horizon_1_7d']['WAPE']:.1%}).",
    ]
    if not np.isnan(wk_gap):
        insights.append(f"Weekend nights in this window are forecast {wk_gap:+.1%} versus weekdays.")
    return {
        "origin_date": str(origin.date()), "model_name": sel, "model_version": f.artifact["metadata"]["version"],
        "note": ("The booking file ends on 2017-08-31, so the latest forecast window that can be fully checked starts "
                 f"after {origin.date()}. This is a historical (as-of) forecast, not a live feed."),
        "total_7d": round(float(seven["prediction"].sum()), 0), "total_14d": round(float(daily["prediction"].sum()), 0),
        "peak": {"date": str(peak.date()), "room_nights": round(float(daily.loc[peak, "prediction"]), 0)},
        "lowest": {"date": str(low.date()), "room_nights": round(float(daily.loc[low, "prediction"]), 0)},
        "daily": [{"date": str(d.date()), "prediction": round(float(r.prediction), 1), "recorded_actual": float(r.actual)}
                  for d, r in daily.iterrows()],
        "recent_history": [{"date": str(d.date()), "room_nights": float(v)} for d, v in history.items()],
        "by_category_14d": {k: round(float(v), 0) for k, v in by_cat.items()},
        "performance": {
            "selected_test": ms["overall_1_14d"], "baseline_test": bs["overall_1_14d"],
            "selected_test_7d": ms["horizon_1_7d"], "baseline_test_7d": bs["horizon_1_7d"],
            "interval_test_coverage": metrics["uncertainty"]["test_coverage"],
        },
        "insights": insights, "notice": NOTICE,
    }


def category_analytics(state: AppState) -> dict:
    f = state.require()
    metrics = state.metrics()
    d = f.builder.demand_frame().loc[config.DEMAND_START:]
    sel = metrics["selected_model"]
    totals = d.sum()
    monthly = d.groupby(d.index.to_period("M")).mean()
    last28, prev28 = d.iloc[-28:].mean(), d.iloc[-56:-28].mean()
    weekday = d.groupby(d.index.dayofweek).mean()
    cats = []
    for c in f.categories:
        cats.append({
            "room_category": c, "total_room_nights": int(totals[c]), "share": round(float(totals[c] / totals.sum()), 4),
            "avg_daily": round(float(d[c].mean()), 1), "peak_day": round(float(d[c].max()), 0),
            "last_28d_avg": round(float(last28[c]), 1), "change_vs_prior_28d": round(float(last28[c] / prev28[c] - 1), 4),
            "test_WAPE": metrics["models"][sel]["test"]["by_category"][c]["WAPE"],
            "test_MAE": metrics["models"][sel]["test"]["by_category"][c]["MAE"],
            "baseline_test_WAPE": metrics["baseline"]["test"]["by_category"][c]["WAPE"],
        })
    return {
        "highest_demand_category": totals.idxmax(), "categories": cats,
        "monthly_avg_daily": [{"month": str(p), "room_category": c, "avg_daily": round(float(v), 1)}
                              for p, row in monthly.iterrows() for c, v in row.items()],
        "weekday_avg_daily": [{"weekday": int(w), "room_category": c, "avg_daily": round(float(v), 1)}
                              for w, row in weekday.iterrows() for c, v in row.items()],
        "note": "Categories are RESERVED room types; rare types B, C, H, L, P are pooled as 'Other'.",
    }


def explain_global(state: AppState) -> dict:
    metrics = state.metrics()
    f = state.require()
    return {
        "model_name": metrics["selected_model"],
        "method": "Grouped permutation importance on held-out validation rows (MAE increase when a feature group is shuffled).",
        "importance": metrics["feature_importance"],
        "ablation_without_on_the_books": metrics.get("ablation_without_on_the_books"),
        "uncertainty": metrics["uncertainty"],
        "limitations": LIMITATIONS, "human_review": REVIEW_GUIDANCE,
        "caveat": "Importance is a property of this model, not proof of cause and effect.",
        "feature_columns": f.artifact["feature_columns"],
    }


def _sha12(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else None


def data_health(state: AppState) -> dict:
    prof = json.loads((config.REPORTS_DIR / "data_profile_raw.json").read_text()) if (config.REPORTS_DIR / "data_profile_raw.json").exists() else {}
    clean = json.loads(config.CLEAN_REPORT_FILE.read_text()) if config.CLEAN_REPORT_FILE.exists() else {}
    out = {
        "dataset": {
            "source": "Hotel Booking Demand (Antonio, Almeida & Nunes 2019), CC BY 4.0",
            "raw_rows": prof.get("rows"), "clean_rows": clean.get("clean_rows"),
            "exact_duplicates_dropped": clean.get("exact_duplicates_dropped"),
            "invalid_rows_dropped": clean.get("invalid_rows_dropped"),
            "arrival_date_min": prof.get("arrival_date_min"), "arrival_date_max": prof.get("arrival_date_max"),
            "missing_values_raw": {**prof.get("empty_cell_counts", {}),
                                   **{f"{k} (literal NULL)": v for k, v in prof.get("literal_NULL_string_counts", {}).items()}},
            "freshness": "Static historical file - last booking arrival 2017-08-31. No live data feed is connected.",
        },
        "monitoring": "Live drift/performance monitoring is not implemented; no live data is shown.",
        "artifact": {"path": str(config.MODEL_FILE.relative_to(config.PROJECT_ROOT)), "exists": state.model_path.exists(),
                     "size_kb": round(state.model_path.stat().st_size / 1024, 1) if state.model_path.exists() else None,
                     "sha256_12": _sha12(state.model_path), "loaded": state.forecaster is not None,
                     "load_error": state.error},
        "evaluation": {"metrics_file_present": state.metrics_path.exists()},
    }
    if state.forecaster:
        info = state.forecaster.info()
        m = state.metrics() if state.metrics_path.exists() else {}
        out["model"] = {"name": info["model_name"], "version": info["version"], "trained_at_utc": info["trained_at_utc"],
                        "fit_on": info["fit_on"], "split": info["split"]}
        if m:
            out["evaluation"].update({
                "selected_beats_baseline_on_validation": m["selected_beats_baseline_on_validation"],
                "selected_beats_baseline_on_test": m["selected_beats_baseline_on_test"],
                "rows": m["rows"]})
    return out
