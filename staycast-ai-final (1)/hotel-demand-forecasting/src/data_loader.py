"""Load the ORIGINAL hotel_bookings.csv and profile it (read-only).

Nothing here modifies the raw file. Cleaning lives in preprocessing.py
(Phase 2) and writes to data/processed/ only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config


class DataValidationError(ValueError):
    """Raised when the raw file does not look like the expected dataset."""


def load_raw(path: Path | str = config.RAW_FILE) -> pd.DataFrame:
    """Read the raw CSV exactly as shipped.

    keep_default_na=False because the file uses the literal string "NULL"
    for missing agent/company; we want to see that explicitly rather than
    let pandas silently convert it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download hotel_bookings.csv from Kaggle "
            "(jessemostipak/hotel-booking-demand) into data/raw/."
        )
    # low_memory=False: chunked parsing made `children` mixed-type and
    # silently changed the duplicate count (31,981 vs the true 31,994).
    # The 4 missing `children` values are the string "NA".
    return pd.read_csv(
        path,
        keep_default_na=False,
        na_values={"children": ["NA"]},
        low_memory=False,
    )


def validate_schema(df: pd.DataFrame) -> None:
    """Fail loudly if columns or size differ from the published dataset."""
    missing = [c for c in config.EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in config.EXPECTED_COLUMNS]
    if missing or extra:
        raise DataValidationError(f"Schema mismatch. Missing={missing} Extra={extra}")
    if len(df) != config.EXPECTED_ROWS:
        raise DataValidationError(
            f"Expected {config.EXPECTED_ROWS} rows, found {len(df)}."
        )


def arrival_dates(df: pd.DataFrame) -> pd.Series:
    """Build a real datetime from the year / month-name / day columns."""
    return pd.to_datetime(
        {
            "year": df["arrival_date_year"],
            "month": df["arrival_date_month"].map(config.MONTH_TO_NUM),
            "day": df["arrival_date_day_of_month"],
        }
    )


def profile(df: pd.DataFrame) -> dict:
    """Return a JSON-serialisable data-quality profile of the raw data."""
    arrival = arrival_dates(df)
    nights = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    children = pd.to_numeric(df["children"], errors="coerce").fillna(0)
    guests = df["adults"] + children + df["babies"]
    null_strings = {
        c: int((df[c].astype(str) == "NULL").sum()) for c in ("agent", "company", "country")
    }
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "arrival_date_min": str(arrival.min().date()),
        "arrival_date_max": str(arrival.max().date()),
        "distinct_arrival_days": int(arrival.nunique()),
        "calendar_days_in_span": int((arrival.max() - arrival.min()).days + 1),
        "hotels": df["hotel"].value_counts().to_dict(),
        "cancellation_rate": round(float(df["is_canceled"].mean()), 4),
        "reservation_status": df["reservation_status"].value_counts().to_dict(),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "empty_cell_counts": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "literal_NULL_string_counts": null_strings,
        "reserved_room_types": df["reserved_room_type"].str.strip().value_counts().to_dict(),
        "assigned_room_types": df["assigned_room_type"].str.strip().value_counts().to_dict(),
        "reserved_ne_assigned_share": round(
            float((df["reserved_room_type"].str.strip() != df["assigned_room_type"].str.strip()).mean()), 4
        ),
        "zero_night_stays": int((nights == 0).sum()),
        "max_nights": int(nights.max()),
        "zero_guest_rows": int((guests == 0).sum()),
        "adr_negative": int((df["adr"] < 0).sum()),
        "adr_zero": int((df["adr"] == 0).sum()),
        "adr_max": float(df["adr"].max()),
        "lead_time_max": int(df["lead_time"].max()),
    }


def main() -> None:
    df = load_raw()
    validate_schema(df)
    report = profile(df)
    out = config.REPORTS_DIR / "data_profile_raw.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSchema OK. Profile saved to {out.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
