"""Leakage-safe feature generation, shared by training AND inference.

Forecasting set-up
------------------
* Prediction timestamp ("origin") t = END of day t.
* We forecast net room-night demand for target date d = t + h, h = 1..14,
  for each room category c.
* Every feature for row (t, h, c) may only use information known at the end
  of day t:
    - realised demand for nights <= t                (lags, rolling means)
    - bookings on the books at t for night d          (booked on/before t and
      not yet cancelled by t)  <- "pickup" signal, the strongest legitimate input
    - arrivals up to t (cancel rate, mean lead time)
    - calendar facts about d (weekday, holiday): deterministic, known in advance.
* Never used: demand on/after t+1, bookings made after t, cancellations that
  happen after t, assigned room type, reservation_status.

`truncate_as_of` lets tests PROVE this: features computed from history
truncated at t must equal features computed from the full history.
"""
from __future__ import annotations

import math

import holidays
import numpy as np
import pandas as pd

from src import config
from src.preprocessing import expand_to_stay_nights

ID_COLUMNS = ["origin_date", "target_date", "horizon", "room_category", "y"]

LAG_FEATURES = [f"lag_{k}" for k in range(7)]
NUMERIC_FEATURES = (
    ["horizon"]
    + LAG_FEATURES
    + ["base_same_dow", "base_same_dow_prev"]
    + [f"roll_mean_{w}" for w in config.ROLL_WINDOWS]
    + ["trend_7_28", "otb", "otb_total", "otb_base", "otb_pace",
       "cancel_rate_28", "mean_lead_28", "is_weekend", "quarter", "is_holiday"]
)
CATEGORICAL_FEATURES = ["room_category", "dow", "month", "season"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Human-readable groups used by explainability (global + local).
FEATURE_GROUPS = {
    "On-the-books bookings (pickup)": ["otb", "otb_total", "otb_base", "otb_pace"],
    "Recent daily demand (lags)": LAG_FEATURES,
    "Same-weekday history": ["base_same_dow", "base_same_dow_prev"],
    "Rolling averages & trend": [f"roll_mean_{w}" for w in config.ROLL_WINDOWS] + ["trend_7_28"],
    "Calendar (weekday, month, season)": ["dow", "is_weekend", "month", "quarter", "season"],
    "Holiday calendar": ["is_holiday"],
    "Room category": ["room_category"],
    "Booking behaviour (cancel rate, lead time)": ["cancel_rate_28", "mean_lead_28"],
    "Forecast horizon": ["horizon"],
}


class FeatureError(ValueError):
    """Requested origin/horizon cannot be featurised from the available history."""


def season_of(month: int) -> str:
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer"}.get(month, "autumn")


def truncate_as_of(clean: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Return the booking history as it was known at the END of day `as_of`.

    Bookings made later are removed; a cancellation that happens after
    `as_of` is not yet known, so that booking is shown as not cancelled.
    """
    known = clean.loc[clean["booking_date"] <= as_of].copy()
    pending_cancel = (known["is_canceled"] == 1) & (known["status_date"] > as_of)
    known.loc[pending_cancel, "is_canceled"] = 0
    return known.loc[known["arrival_date"] <= as_of + pd.Timedelta(days=config.HORIZONS[-1])]


class FeatureBuilder:
    """Pre-computes demand / on-the-books arrays once, then serves feature rows."""

    def __init__(self, clean: pd.DataFrame):
        self.dates = pd.date_range(config.SERIES_START, config.DATA_END)
        self.d0 = self.dates[0]
        self.T, self.cats = len(self.dates), config.CATEGORIES
        self.C = len(self.cats)
        self._cat_idx = {c: i for i, c in enumerate(self.cats)}
        self.first_origin = pd.Timestamp(config.DEMAND_START) + pd.Timedelta(days=config.MAX_LAG - 1)
        self.last_date = self.dates[-1]
        self._holidays = holidays.country_holidays(
            config.HOLIDAY_COUNTRY, years=range(self.dates[0].year, self.dates[-1].year + 2)
        )

        nights = expand_to_stay_nights(clean)
        nights = nights.loc[nights["stay_date"] <= self.last_date]
        pos = (nights["stay_date"] - self.d0).dt.days.to_numpy()
        cidx = nights["room_category"].map(self._cat_idx).to_numpy()
        flat = pos * self.C + cidx
        size = self.T * self.C

        net = (nights["is_canceled"] == 0).to_numpy()
        self.demand = np.bincount(flat[net], minlength=size).reshape(self.T, self.C).astype(float)
        self.gross = np.bincount(flat, minlength=size).reshape(self.T, self.C).astype(float)

        # on-the-books table: otb[h-1, d, c] = room-nights for night d that were
        # booked and not yet cancelled at the end of day d-h.
        booked, cancelled = nights["booking_date"], nights["cancel_date"]
        self.otb = np.zeros((len(config.HORIZONS), self.T, self.C))
        for hi, h in enumerate(config.HORIZONS):
            origin = nights["stay_date"] - pd.Timedelta(days=h)
            on_books = ((booked <= origin) & (cancelled.isna() | (cancelled > origin))).to_numpy()
            self.otb[hi] = np.bincount(flat[on_books], minlength=size).reshape(self.T, self.C)

        # arrival-day statistics (global across categories)
        apos = (clean["arrival_date"] - self.d0).dt.days.to_numpy()
        ok = (apos >= 0) & (apos < self.T)
        self.arrivals = np.bincount(apos[ok], minlength=self.T).astype(float)
        self.arr_cancel = np.bincount(apos[ok], weights=clean["is_canceled"].to_numpy()[ok], minlength=self.T)
        self.arr_lead = np.bincount(apos[ok], weights=clean["lead_time"].to_numpy()[ok], minlength=self.T)

        self._cs_demand = np.vstack([np.zeros((1, self.C)), np.cumsum(self.demand, axis=0)])
        self._cs = {k: np.concatenate([[0.0], np.cumsum(v)])
                    for k, v in (("n", self.arrivals), ("c", self.arr_cancel), ("l", self.arr_lead))}

    # ------------------------------------------------------------- helpers
    def pos(self, date) -> int:
        return int((pd.Timestamp(date).normalize() - self.d0).days)

    def demand_frame(self, gross: bool = False) -> pd.DataFrame:
        arr = self.gross if gross else self.demand
        return pd.DataFrame(arr, index=self.dates, columns=self.cats)

    def valid_origin_range(self, horizon: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.first_origin, self.last_date - pd.Timedelta(days=horizon)

    def _rolling_mean(self, t: np.ndarray, c: np.ndarray, w: int) -> np.ndarray:
        return (self._cs_demand[t + 1, c] - self._cs_demand[t + 1 - w, c]) / w

    # ------------------------------------------------------------ features
    def make_rows(self, origins, horizons=None, categories=None) -> pd.DataFrame:
        """Feature rows for the cartesian product origins x horizons x categories."""
        origins = pd.DatetimeIndex(pd.to_datetime(list(origins))).normalize()
        horizons = list(horizons) if horizons is not None else config.HORIZONS
        categories = list(categories) if categories is not None else self.cats
        bad = [c for c in categories if c not in self._cat_idx]
        if bad:
            raise FeatureError(f"Unknown room categories: {bad}. Valid: {self.cats}")
        if any(h not in config.HORIZONS for h in horizons):
            raise FeatureError(f"Horizon must be within {config.HORIZONS[0]}..{config.HORIZONS[-1]} days")

        o_grid, h_grid, c_grid = np.meshgrid(
            np.arange(len(origins)), np.array(horizons), np.array([self._cat_idx[c] for c in categories]),
            indexing="ij",
        )
        o_dates = origins[o_grid.ravel()]
        h = h_grid.ravel()
        c = c_grid.ravel()
        t = np.array([self.pos(x) for x in o_dates]) if len(o_dates) else np.array([], dtype=int)
        d = t + h

        if len(t):
            if t.min() < self.pos(self.first_origin):
                raise FeatureError(
                    f"Origin must be on/after {self.first_origin.date()} (earlier history is incomplete)."
                )
            if d.max() > self.T - 1:
                raise FeatureError(
                    f"Target date must be on/before {self.last_date.date()}: booking data ends there, "
                    "so later nights would be under-counted."
                )

        m = np.ceil(h / 7).astype(int)
        base1_pos, base2_pos = d - 7 * m, d - 7 * (m + 1)
        out = pd.DataFrame(
            {
                "origin_date": o_dates,
                "target_date": self.dates[d] if len(d) else pd.DatetimeIndex([]),
                "horizon": h,
                "room_category": np.array(self.cats)[c],
                "y": self.demand[d, c] if len(d) else np.array([]),
            }
        )
        for k in range(7):
            out[f"lag_{k}"] = self.demand[t - k, c]
        out["base_same_dow"] = self.demand[base1_pos, c]
        out["base_same_dow_prev"] = self.demand[base2_pos, c]
        for w in config.ROLL_WINDOWS:
            out[f"roll_mean_{w}"] = self._rolling_mean(t, c, w)
        out["trend_7_28"] = out["roll_mean_7"] / out["roll_mean_28"].clip(lower=1e-6) - 1.0

        out["otb"] = self.otb[h - 1, d, c]
        out["otb_total"] = self.otb[h - 1, d, :].sum(axis=1)
        out["otb_base"] = self.otb[h - 1, base1_pos, c]          # same lead, same weekday last week(s)
        out["otb_pace"] = (out["otb"] + 1.0) / (out["otb_base"] + 1.0)

        n28 = self._cs["n"][t + 1] - self._cs["n"][t + 1 - 28]
        out["cancel_rate_28"] = (self._cs["c"][t + 1] - self._cs["c"][t + 1 - 28]) / np.maximum(n28, 1)
        out["mean_lead_28"] = (self._cs["l"][t + 1] - self._cs["l"][t + 1 - 28]) / np.maximum(n28, 1)

        tgt = pd.DatetimeIndex(out["target_date"])
        out["dow"] = tgt.dayofweek.astype(str)
        out["is_weekend"] = (tgt.dayofweek >= 5).astype(int)
        out["month"] = tgt.month.astype(str)
        out["quarter"] = tgt.quarter
        out["season"] = [season_of(mo) for mo in tgt.month]
        out["is_holiday"] = [int(x in self._holidays) for x in tgt]
        # `horizon` and `room_category` are both identifiers and features.
        return out[ID_COLUMNS + [c for c in FEATURE_COLUMNS if c not in ID_COLUMNS]]

    def training_origins(self) -> pd.DatetimeIndex:
        return pd.date_range(self.first_origin, self.last_date - pd.Timedelta(days=1))

    def all_rows(self) -> pd.DataFrame:
        """Every (origin, horizon, category) row whose target date is inside the data."""
        rows = []
        for h in config.HORIZONS:
            lo, hi = self.valid_origin_range(h)
            rows.append(self.make_rows(pd.date_range(lo, hi), [h]))
        return pd.concat(rows, ignore_index=True)
