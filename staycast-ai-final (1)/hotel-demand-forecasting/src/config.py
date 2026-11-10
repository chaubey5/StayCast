"""Central configuration: paths, seeds and forecasting constants.

Every other module imports from here so paths, the random seed and the
problem definition live in exactly one place.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILE = RAW_DIR / "hotel_bookings.csv"
CLEAN_FILE = PROCESSED_DIR / "bookings_clean.csv.gz"
CLEAN_REPORT_FILE = REPORTS_DIR / "cleaning_report.json"
MODEL_FILE = MODELS_DIR / "best_model.joblib"
METRICS_FILE = REPORTS_DIR / "metrics.json"

RANDOM_SEED = 42

# ---- dataset constants (original Kaggle file: 119,390 rows x 32 columns) ----
EXPECTED_ROWS = 119_390
EXPECTED_COLUMNS = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year",
    "arrival_date_month", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies", "meal",
    "country", "market_segment", "distribution_channel",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "reserved_room_type",
    "assigned_room_type", "booking_changes", "deposit_type", "agent",
    "company", "days_in_waiting_list", "customer_type", "adr",
    "required_car_parking_spaces", "total_of_special_requests",
    "reservation_status", "reservation_status_date",
]

MONTH_TO_NUM = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5,
    "June": 6, "July": 7, "August": 8, "September": 9, "October": 10,
    "November": 11, "December": 12,
}

# ---- forecasting problem definition ----
# Room categories = RESERVED room type (known at booking time). Rare types
# (B, C, H, L, P: <3% of bookings together) are pooled so every series has
# enough volume to forecast.
MAIN_CATEGORIES = ["A", "D", "E", "F", "G"]
OTHER_CATEGORY = "Other"
CATEGORIES = MAIN_CATEGORIES + [OTHER_CATEGORY]

HORIZONS = list(range(1, 15))      # forecast 1..14 days ahead of the origin
HORIZON_GROUPS = {7: range(1, 8), 14: range(1, 15)}

# Stays that began before 2015-07-01 are not in the file, so the first
# days of the series are under-counted. Skip a burn-in window.
DEMAND_START = "2015-07-22"
DATA_END = "2017-08-31"            # last arrival date; later nights are truncated

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # train / validation / test, by target date
INTERVAL_COVERAGE = 0.90               # nominal coverage of prediction intervals
ROLL_WINDOWS = (7, 14, 28)
MAX_LAG = 28                           # longest history window used by features
SERIES_START = "2015-07-01"            # first arrival date in the file
HOLIDAY_COUNTRY = "PT"                 # both hotels are in Portugal
