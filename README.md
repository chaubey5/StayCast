# STAYCAST AI
**Intelligent Hotel Demand Forecasting & Decision Intelligence**
ALGIONS AI LABS PRIVATE LIMITED - AI & ML internship - *Hotel Occupancy & Room-Demand Forecasting*

Forecasts **daily net room-night demand by reserved room category, 1-14 days ahead**, with prediction intervals and explanations, behind a FastAPI backend and a seven-screen dashboard.

> **Not occupancy.** The dataset has no room inventory, so the target is booking room-night demand (non-cancelled bookings). No occupancy percentage is ever computed. Forecasts are estimates for human decision support, not guarantees.

## Headline results (held-out test, 2017-05-12 to 2017-08-31)
| | WAPE 1-14d | WAPE 1-7d | MAE (room-nights/row) |
|---|---|---|---|
| Naive seasonal baseline | 10.7% | 10.4% | 6.31 |
| **HistGradientBoosting (selected on validation)** | **3.0%** | **2.4%** | **1.80** |
| Same model *without* on-the-books features (ablation) | 8.0% | - | 4.73 |

Most of the accuracy comes from bookings already on the books at the forecast origin; see `reports/technical_report.pdf` for the full, honest discussion.

## Quick start (fresh environment)
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 1. get the data (CC BY 4.0): https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
#    place hotel_bookings.csv in data/raw/
python scripts/verify_env.py                              # environment check
python -m src.data_loader                                 # schema check + raw profile
python -m src.preprocessing                               # data/processed/bookings_clean.csv.gz + cleaning_report.json
python -m src.eda                                         # EDA figures
python -m src.train                                       # trains, evaluates, saves models/best_model.joblib + reports/metrics.json (~2 min)
python -m src.predict 2017-08-25 A 7                      # CLI inference
pytest -q                                                 # 61 tests
uvicorn backend.api:app --port 8000                       # API  (docs: http://localhost:8000/docs)
streamlit run frontend/app.py                             # dashboard: http://localhost:8501
```
Optional: `python scripts/build_notebook.py` (executes `notebooks/hotel_eda.ipynb`), `python scripts/build_report.py` (report + PDF), `python scripts/take_screenshots.py`.

Environment variables (all optional) are listed in `.env.example`; `STAYCAST_API_URL` points the dashboard to the API.

## Project layout
```
data/raw, data/processed     original CSV (git-ignored) / cleaned data
src/                         data_loader, preprocessing, feature_engineering, forecasting, train, evaluate, explainability, predict, eda, config
models/best_model.joblib     fitted pipeline + interval table + stats + metadata
backend/                     api.py, schemas.py, services.py (FastAPI)
frontend/app.py              Streamlit + Plotly dashboard
tests/                       61 pytest tests
reports/                     technical_report.pdf/.md, metrics.json, figures/, screenshots/
docs/                        assessment evidence, submission form, video plan
```

## Design decisions (interview notes)
* **Target** = net room-nights per stay date x reserved category. Reserved type (known at booking) is used; assigned type (known at check-in) would leak.
* **Forecast origin** = end of day t, horizons 1-14. Features use only information available at t, including *on-the-books* bookings (booked <= t, not cancelled by t).
* **Leakage proof**: features computed from history truncated at t are identical to those from full history (`tests/test_forecasting.py`).
* **Split**: chronological 70/15/15 by target date; selection on validation; models refit on train+validation, test used once.
* **Baseline**: same weekday, latest observed week - same information as the models.
* **Intervals**: validation-residual quantiles per horizon group and category; 93% observed test coverage vs 90% nominal.
* **Honesty**: ablation shows reliance on pickup features; in-sample dates and unusual inputs are flagged; the file ends 2017-08-31 so dashboard "latest forecast" is an *as-of historical* forecast, not live.

## Dataset
Antonio, N., de Almeida, A., Nunes, L. (2019). *Hotel booking demand datasets.* Data in Brief 22, 41-49. License CC BY 4.0. The raw CSV is not committed; SHA-256 of the file used is stored in the model artifact metadata.

## Known limitations
See section 22 of the technical report and the Explainable AI screen.
