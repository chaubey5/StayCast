"""Reproducible cleaning of the raw booking file.

Pipeline:  raw CSV --standardise--> typed frame --drop exact duplicates-->
           --drop invalid rows--> data/processed/bookings_clean.csv.gz

The raw file is never modified. Every decision is counted and written to
reports/cleaning_report.json so it can be quoted in the technical report.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data_loader import load_raw, validate_schema

CLEAN_COLUMNS = [
    "hotel", "is_canceled", "lead_time", "arrival_date", "booking_date",
    "status_date", "nights", "adults", "children", "babies", "room_category",
    "reserved_room_type", "adr", "market_segment", "customer_type",
    "deposit_type", "country", "reservation_status",
]


# ----------------------------------------------------------------- helpers
def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes and normalise categories. Returns a new frame."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype(str).str.strip()
    out["children"] = out["children"].fillna(0).astype(int)
    # The file encodes "unknown" as the literal string NULL.
    out["country"] = out["country"].replace({"NULL": "UNK"})
    out["agent"] = pd.to_numeric(out["agent"].replace({"NULL": np.nan}), errors="coerce")
    out["company"] = pd.to_numeric(out["company"].replace({"NULL": np.nan}), errors="coerce")
    out["reserved_room_type"] = out["reserved_room_type"].str.upper()
    out["assigned_room_type"] = out["assigned_room_type"].str.upper()
    return out


def add_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Derive arrival / booking / status dates and stay length."""
    out = df.copy()
    out["arrival_date"] = pd.to_datetime(
        {
            "year": out["arrival_date_year"],
            "month": out["arrival_date_month"].map(config.MONTH_TO_NUM),
            "day": out["arrival_date_day_of_month"],
        }
    )
    out["nights"] = out["stays_in_weekend_nights"] + out["stays_in_week_nights"]
    # The dataset has no booking timestamp; lead_time is days from booking
    # to arrival, so this reconstructs the booking date exactly.
    out["booking_date"] = out["arrival_date"] - pd.to_timedelta(out["lead_time"], unit="D")
    out["status_date"] = pd.to_datetime(out["reservation_status_date"])
    return out


def map_room_category(reserved: pd.Series) -> pd.Series:
    """Reserved room type -> forecasting category (rare types pooled)."""
    return reserved.where(reserved.isin(config.MAIN_CATEGORIES), config.OTHER_CATEGORY)


def invalid_row_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks (True = invalid) for each validation rule."""
    guests = df["adults"] + df["children"] + df["babies"]
    return {
        "zero_night_stay": df["nights"] <= 0,
        "zero_guests": guests <= 0,
        "negative_adr": df["adr"] < 0,
    }


# ------------------------------------------------------------------- clean
def clean_bookings(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Full cleaning pipeline. Returns (clean frame, decision report)."""
    report: dict = {"raw_rows": int(len(raw))}

    typed = add_dates(standardise(raw))

    # 1) Exact duplicates over ALL original columns. There is no booking ID, so
    #    we cannot prove they are the same booking; we keep the first copy
    #    (standard practice) and measure the sensitivity in the EDA.
    dup_mask = raw.pipe(standardise).duplicated(keep="first")
    report["exact_duplicates_dropped"] = int(dup_mask.sum())
    typed = typed.loc[~dup_mask]

    # 2) Invalid rows (sequential counts, so they do not double count).
    report["invalid_rows_dropped"] = {}
    for name in invalid_row_masks(typed):
        mask = invalid_row_masks(typed)[name]      # recomputed on surviving rows -> counts never overlap
        report["invalid_rows_dropped"][name] = int(mask.sum())
        typed = typed.loc[~mask]

    # 3) Informational checks that are flagged, not dropped.
    cancelled = typed["is_canceled"] == 1
    report["flags_not_dropped"] = {
        "cancel_date_before_booking_date": int(
            (cancelled & (typed["status_date"] < typed["booking_date"])).sum()
        ),
        "cancel_date_after_arrival": int(
            (cancelled & (typed["status_date"] > typed["arrival_date"])).sum()
        ),
        "adr_zero_rows": int((typed["adr"] == 0).sum()),
        "adr_above_1000": int((typed["adr"] > 1000).sum()),
        "reserved_ne_assigned": int((typed["reserved_room_type"] != typed["assigned_room_type"]).sum()),
        "missing_country_UNK": int((typed["country"] == "UNK").sum()),
    }

    typed["room_category"] = map_room_category(typed["reserved_room_type"])
    clean = typed[CLEAN_COLUMNS].reset_index(drop=True)

    report["clean_rows"] = int(len(clean))
    report["room_category_bookings"] = clean["room_category"].value_counts().to_dict()
    report["decisions"] = [
        "Raw file is never modified; processed data is written separately.",
        "Exact duplicates (all 32 columns) dropped, first copy kept.",
        "Rows with zero nights, zero guests or negative ADR dropped as invalid.",
        "children NA -> 0; country 'NULL' -> 'UNK'; agent/company 'NULL' -> NaN (not used as features).",
        "Cancelled / no-show rows are KEPT: they matter for on-the-books features and cancellation analysis.",
        "Room category = RESERVED room type (assigned type is only known at check-in).",
        "Rare room types B, C, H, L, P pooled into 'Other'.",
        "ADR outliers (max 5400) are not used by the model; they are only winsorised for EDA charts.",
    ]
    return clean, report


# ---------------------------------------------------------- stay expansion
def expand_to_stay_nights(clean: pd.DataFrame) -> pd.DataFrame:
    """One row per booked room-night (assumes 1 room per booking).

    The dataset has no room count; 1 booking = 1 room is the standard
    assumption and is documented as a limitation.
    """
    n = clean["nights"].to_numpy()
    idx = np.repeat(np.arange(len(clean)), n)
    offsets = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    base = clean.iloc[idx]
    out = pd.DataFrame(
        {
            "stay_date": base["arrival_date"].to_numpy() + pd.to_timedelta(offsets, unit="D").to_numpy(),
            "room_category": base["room_category"].to_numpy(),
            "hotel": base["hotel"].to_numpy(),
            "booking_date": base["booking_date"].to_numpy(),
            "is_canceled": base["is_canceled"].to_numpy(),
            # A cancelled booking leaves the books on its status date; a
            # booking that is never cancelled stays forever (NaT).
            "cancel_date": base["status_date"].where(base["is_canceled"] == 1).to_numpy(),
        }
    )
    out["stay_date"] = pd.to_datetime(out["stay_date"])
    out["booking_date"] = pd.to_datetime(out["booking_date"])
    out["cancel_date"] = pd.to_datetime(out["cancel_date"])
    return out


# --------------------------------------------------------------------- IO
def load_clean(path: Path | str = config.CLEAN_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.preprocessing")
    return pd.read_csv(path, parse_dates=["arrival_date", "booking_date", "status_date"])


def main() -> None:
    raw = load_raw()
    validate_schema(raw)
    clean, report = clean_bookings(raw)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(config.CLEAN_FILE, index=False)
    config.CLEAN_REPORT_FILE.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "decisions"}, indent=2))
    print(f"\nSaved {len(clean):,} rows -> {config.CLEAN_FILE.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
