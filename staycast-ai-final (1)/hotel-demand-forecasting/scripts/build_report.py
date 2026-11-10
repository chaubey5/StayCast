"""Generate reports/technical_report.md (numbers read from metrics.json) and the PDF.

    python scripts/build_report.py
"""
import base64
import json
import re
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
m = json.loads((R / "metrics.json").read_text())
cl = json.loads((R / "cleaning_report.json").read_text())
prof = json.loads((R / "data_profile_raw.json").read_text())
eda = json.loads((R / "eda_interpretations.json").read_text())
sel = m["selected_model"]
pc = lambda x: f"{x * 100:.1f}%"  # noqa: E731
o = lambda mod, p, k="overall_1_14d": mod[p][k]  # noqa: E731
B, S = m["baseline"], m["models"][sel]
imp = m["feature_importance"]
ab = m["ablation_without_on_the_books"]
sp = m["split"]


def model_rows():
    rows = ["| Model | Config chosen on validation | Val MAE | Val RMSE | Val WAPE | Test MAE | Test RMSE | Test WAPE |", "|---|---|---|---|---|---|---|---|",
            f"| Naive seasonal baseline | same weekday, latest observed week | {o(B,'validation')['MAE']} | {o(B,'validation')['RMSE']} | {pc(o(B,'validation')['WAPE'])} | {o(B,'test')['MAE']} | {o(B,'test')['RMSE']} | {pc(o(B,'test')['WAPE'])} |"]
    for n, v in m["models"].items():
        rows.append(f"| {n}{' (selected)' if n == sel else ''} | residual target: {v['config']['residual_target']}; {v['config']['params'] or 'defaults'} | "
                    f"{o(v,'validation')['MAE']} | {o(v,'validation')['RMSE']} | {pc(o(v,'validation')['WAPE'])} | {o(v,'test')['MAE']} | {o(v,'test')['RMSE']} | {pc(o(v,'test')['WAPE'])} |")
    return "\n".join(rows)


def horizon_rows():
    rows = ["| Horizon (days) | Selected WAPE | Baseline WAPE |", "|---|---|---|"]
    for h in ("1", "3", "5", "7", "10", "14"):
        rows.append(f"| {h} | {pc(S['test']['by_horizon'][h]['WAPE'])} | {pc(B['test']['by_horizon'][h]['WAPE'])} |")
    return "\n".join(rows)


def cat_rows():
    rows = ["| Category | Test MAE (room-nights/day) | Selected WAPE | Baseline WAPE |", "|---|---|---|---|"]
    for c, v in S["test"]["by_category"].items():
        rows.append(f"| {c} | {v['MAE']} | {pc(v['WAPE'])} | {pc(B['test']['by_category'][c]['WAPE'])} |")
    return "\n".join(rows)


def imp_rows():
    rows = ["| Feature group | MAE increase when shuffled | Share |", "|---|---|---|"]
    rows += [f"| {i['group']} | {i['mae_increase']} | {pc(i['share_of_total'])} |" for i in imp]
    return "\n".join(rows)


md = f"""# STAYCAST AI - Technical Report
*Intelligent Hotel Demand Forecasting & Decision Intelligence* - ALGIONS AI LABS PRIVATE LIMITED, AI & ML internship (Hotel Occupancy & Room-Demand Forecasting).

## 1. Executive summary
STAYCAST AI forecasts **daily net room-night demand by reserved room category for the next 1-14 days** from the public Hotel Booking Demand dataset. On a chronologically held-out test period ({sp['val_end']} to {sp['test_end']}) the selected model, **{sel}**, reaches a WAPE of **{pc(o(S,'test')['WAPE'])}** (7-day: {pc(S['test']['horizon_1_7d']['WAPE'])}) against **{pc(o(B,'test')['WAPE'])}** for a naive seasonal baseline. The model was chosen on validation data only ({pc(o(S,'validation')['WAPE'])} vs {pc(o(B,'validation')['WAPE'])}).

Two honest qualifications: (1) about {pc(imp[0]['share_of_total'])} of the model's reliance is on *bookings already on the books* at the forecast origin - without those features the same model scores {pc(o(ab,'test')['WAPE'])} on test (baseline {pc(o(B,'test')['WAPE'])}); (2) the target is booking room-night demand, **not physical occupancy**, because the dataset contains no room inventory.

The system includes a leakage-safe feature pipeline, a saved joblib artifact, a FastAPI backend (real inference, validation, graceful errors), a seven-screen Streamlit dashboard and {61} automated tests.

## 2. Business problem
Managers need to anticipate room-night demand to plan staffing, housekeeping, room allocation and availability. The forecast supports human decisions; it never automates them.

## 3. Prediction target
`demand(d, c)` = number of booked room-nights on night *d*, for reserved room category *c*, from bookings that are **not cancelled** (1 booking = 1 room). Categories: A, D, E, F, G and Other (B, C, H, L, P pooled; each under 1% of bookings). Cancelled and no-show rows are kept in the data - they drive on-the-books features and the cancellation analysis - but are not counted in the target. Both hotels are pooled. **This is not occupancy**: no daily inventory exists, and no occupancy percentage is ever computed.

## 4. Forecasting horizon and prediction timestamp
The forecast origin is the **end of day t**; targets are t+1 ... t+14. 7-day results use horizons 1-7, 14-day results use 1-14. One model with horizon as a feature serves both.

## 5. Dataset source
Antonio, Almeida & Nunes (2019), *Hotel booking demand datasets*, Data in Brief 22, CC BY 4.0. Kaggle: jessemostipak/hotel-booking-demand. {prof['rows']:,} rows x {prof['columns']} columns; arrivals {prof['arrival_date_min']} to {prof['arrival_date_max']} ({prof['distinct_arrival_days']} consecutive days).

## 6. Dataset limitations
- Bookings, not occupancy; no inventory, no room counts (1 booking = 1 room is assumed).
- Two Portuguese hotels, about two years: limited representativeness; only two summers (2016, 2017) plus Jul-Aug 2015.
- No booking ID: {cl['exact_duplicates_dropped']:,} exact duplicate rows ({pc(cl['exact_duplicates_dropped']/prof['rows'])}) cannot be proven to be the same booking.
- Booking dates are reconstructed as arrival minus lead time; cancellation date = `reservation_status_date`.
- Anonymised room-type codes; reserved type differs from assigned type for {pc(prof['reserved_ne_assigned_share'])} of bookings.
- Arrivals end on {prof['arrival_date_max']}, so nights after that date are truncated and the file cannot support a true future forecast. All "latest forecasts" in the dashboard are **historical as-of forecasts**, not live ones.

## 7. Exploratory data analysis
Figures are in `reports/figures/eda_*.png` and in `notebooks/hotel_eda.ipynb`. Key findings (computed from the data):

- **Missing values.** {eda['missing_values']}
- **Duplicates.** {eda['duplicates_sensitivity']}
- **Cancellations.** {eda['cancellation']}
- **Weekday.** {eda['weekday_pattern']}
- **Seasonality.** {eda['seasonality_heatmap']}
- **Monthly trend.** {eda['monthly_trend']}
- **Room categories.** {eda['room_categories']}
- **Lead time.** {eda['lead_time']}
- **ADR.** {eda['adr']}
- **Outliers.** {eda['outliers']}

![Target series](figures/eda_target_series.png)

## 8. Preprocessing
Implemented in `src/preprocessing.py` (separate functions, raw file never modified, output `data/processed/bookings_clean.csv.gz`, decisions in `reports/cleaning_report.json`):
raw {cl['raw_rows']:,} rows -> exact duplicates dropped ({cl['exact_duplicates_dropped']:,}) -> invalid rows dropped sequentially (zero nights {cl['invalid_rows_dropped']['zero_night_stay']}, zero guests {cl['invalid_rows_dropped']['zero_guests']}, negative ADR {cl['invalid_rows_dropped']['negative_adr']}) -> **{cl['clean_rows']:,} clean bookings**. Type fixes: `children` NA -> 0, `country` NULL -> UNK, `agent`/`company` NULL -> NaN (unused). Consistency checks: no cancellation dated before booking or after arrival. After de-duplication the cancellation rate is 27.7% (37.0% raw).

## 9. Feature engineering
All features are computed by one class, `FeatureBuilder` (`src/feature_engineering.py`), used by both training and inference:
calendar of the target date (weekday, weekend, month, quarter, season, Portuguese public holiday via the `holidays` library); room category; horizon; demand lags 0-6 relative to the origin; same-weekday demand from the latest observed week(s); rolling means (7/14/28 days) and a 7/28 trend ratio; **on-the-books** room-nights for the target night at the origin (booked on/before t and not yet cancelled by t), their total across categories, the same quantity for the reference week and the resulting pace ratio; trailing 28-day cancellation rate and mean lead time of arrivals. The first usable origin is 2015-08-18 so every window is complete.

## 10. Leakage analysis (audit)
| Risk | Control |
|---|---|
| Demand after the origin in lags/rolling means | Lags are indexed from t; reference weeks use d - 7*ceil(h/7) <= t (tested for every row) |
| Bookings made after the origin | On-the-books counts only `booking_date <= t` |
| Future cancellations | A booking is removed only if `cancel_date <= t`; later cancellations are invisible at t |
| Outcome columns (`reservation_status`, `is_canceled` of future stays) | Not features; test asserts banned columns are absent |
| Assigned room type (known at check-in) | Not used; reserved type only |
| Random splitting | Chronological split by target date |
| Training/inference skew | Same `FeatureBuilder` for both |

**Executable proof:** `truncate_as_of(history, t)` rebuilds the booking history as known at the end of day t; `tests/test_forecasting.py` verifies that features for origin t are *identical* whether the full or the truncated history is used (four origins), and that distorting every booking made after t does not change any feature.

## 11. Baseline methodology
Naive seasonal: predict demand on the same weekday of the most recent fully observed week, d - 7*ceil(h/7) (one week back for h<=7, two for h 8-14). It uses no more information than the models and is evaluated on identical rows.

## 12. Model development
Candidates: Linear Regression (interpretable), Random Forest (nonlinear), HistGradientBoostingRegressor (nonlinear, efficient). Each was tried with and without a *residual target* (learning the correction to the baseline) and with a small hyper-parameter grid; selection used validation WAPE only. Seed {42} everywhere. Preprocessing (scaling for LR, one-hot for categoricals) is inside the saved pipeline. The search log is in `reports/metrics.json`.

![Pipeline](figures/model_pipeline.png)

## 13. Chronological split
By target date: train {sp['first']} to {sp['train_end']} ({m['rows']['train']:,} rows), validation to {sp['val_end']} ({m['rows']['validation']:,}), test to {sp['test_end']} ({m['rows']['test']:,}); roughly 70/15/15. Validation metrics come from train-only fits. After selection, the chosen configurations were **refitted on train+validation** and scored once on the untouched test period. Caveat: the test period is summer (high season) while validation is spring; both periods contain only one comparable season in training.

## 14. Evaluation results
{model_rows()}

![Model comparison](figures/model_comparison.png)
![Actual vs predicted](figures/actual_vs_predicted_test.png)

The selected model beats the baseline on both validation and test ({pc(o(S,'test')['WAPE'])} vs {pc(o(B,'test')['WAPE'])}). Validation errors are higher than test errors for every model, **including the baseline** (baseline {pc(o(B,'validation')['WAPE'])} vs {pc(o(B,'test')['WAPE'])}): the validation window (Jan-May 2017, with a moving Easter) is simply harder, so the gap should not be read as overfitting or luck in the test.

## 15. Error analysis
**By horizon (test)**

{horizon_rows()}

![Error by horizon](figures/error_by_horizon.png)

**By room category (test)**

{cat_rows()}

![Error by category](figures/error_by_category.png)
![Residuals](figures/residual_distribution.png)

*What this means operationally.* For the large categories the average miss is a few room-nights per day (category A MAE {S['test']['by_category']['A']['MAE']}), small relative to hundreds of room-nights - adequate for staffing and housekeeping rosters. For small categories (F, G, Other) the percentage error is larger because daily volumes are tiny, so allocation decisions for them should keep a manual buffer. Errors grow with horizon because fewer bookings are on the books.

## 16. Uncertainty and explainability
**Intervals.** Empirical quantiles of validation residuals per (horizon group x category) give a nominal {pc(m['uncertainty']['nominal_coverage'])} interval; observed test coverage was **{pc(m['uncertainty']['test_coverage'])}** (mean width {m['uncertainty']['mean_width_test']} room-nights). Slight over-coverage reflects the easier test period; coverage may be lower in unusual periods.

![Interval](figures/prediction_interval_category_A.png)

**Global importance** (grouped permutation importance on held-out validation rows):

{imp_rows()}

**Ablation.** Removing the on-the-books group, the same model scores {pc(o(ab,'validation')['WAPE'])} (validation) and {pc(o(ab,'test')['WAPE'])} (test) - still better than the baseline, but far worse than with pickup information. The headline accuracy therefore rests mostly on knowing what is already booked.

**Local explanations.** For each prediction the API reports, per feature group, how far the prediction moves if that group is replaced by a typical training value. These are sensitivities of the model, **not causal effects**.

**Sanity checks.** Output is clipped at zero; predictions are compared with the category's historical maximum; inputs outside training ranges are flagged; dates inside the training/validation period are marked in-sample.

## 17. Architecture
![Architecture](figures/architecture.png)

Modules: `src/` (data, preprocessing, features, models, train, evaluate, explainability, predict), `backend/` (FastAPI: api, schemas, services), `frontend/app.py` (Streamlit), `tests/`. The dashboard shows only API output; the API calls the saved model through `Forecaster`.

## 18. API instructions
`uvicorn backend.api:app --port 8000`, docs at `/docs`. Endpoints: `GET /api/health`, `/api/model/metrics`, `/api/model/info`, `/api/model/explain`, `/api/forecast`, `POST /api/forecast/predict`, `/api/forecast/history`, plus `/api/forecast/backtest`, `/api/overview`, `/api/analytics/categories`, `/api/data/health`. Invalid input returns 422 with a message; a missing or corrupt artifact returns 503 - no prediction is ever fabricated.

Example: `curl -X POST localhost:8000/api/forecast/predict -H "Content-Type: application/json" -d '{{"target_date":"2017-08-25","room_category":"A","horizon":7}}'`

## 19. Dashboard instructions
`streamlit run frontend/app.py` (API must be running; override with `STAYCAST_API_URL`). Screens: Executive Overview, Forecast Intelligence, Room Category Analytics, Model Performance Lab, Prediction Explorer, Explainable AI, Data & Model Health. Screenshots are in `reports/screenshots/`.

![Overview](screenshots/1_executive_overview.png)
![Explorer](screenshots/5_prediction_explorer.png)

## 20. Testing
`pytest -q` runs 61 tests: schema/data validation, preprocessing, target expansion, leakage proofs, chronological split, determinism of training, artifact loading (missing, corrupt, incomplete), inference equals direct model output, invalid inputs, and every API endpoint including the 503 path.

## 21. Responsible AI
Decision support only; estimates are not guarantees; no personal data is used or shown (guest country is aggregated only in EDA); the model must not trigger automated high-stakes actions; in-sample dates and unusual inputs are flagged; limits are displayed in the Explainable AI screen.

## 22. Known limitations
Booking demand is not occupancy; short history with two summers; strong dependence on on-the-books data (a hotel without reliable booking-pace data would see the weaker ablation numbers); duplicates handled by a documented assumption; one booking = one room; intervals calibrated on one validation period; the dataset ends 2017-08-31 so no live forecast beyond it is possible; no drift monitoring is implemented.

## 23. Future improvements
Add real inventory to compute occupancy; expose booking-pace curves by lead-time bucket; quantile-regression or conformal intervals; rolling-origin cross-validation with more years; per-hotel models; event calendars; scheduled retraining with forecast-vs-actual monitoring; replace the Streamlit MVP with React if multi-user deployment is needed.
"""
(R / "technical_report.md").write_text(md)

# ---- PDF
html_body = markdown.markdown(md, extensions=["tables", "fenced_code"])


def inline(match):
    p = R / match.group(1)
    return f'src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"' if p.exists() else match.group(0)


html_body = re.sub(r'src="((?:figures|screenshots)/[^"]+)"', inline, html_body)
css = """body{font-family:Georgia,'Times New Roman',serif;color:#1B2430;font-size:10.5pt;line-height:1.45;margin:0}
h1{color:#0B1F3A;border-bottom:3px solid #B8985A;padding-bottom:6px} h2{color:#0B1F3A;margin-top:22px;border-bottom:1px solid #E4DCC8}
table{border-collapse:collapse;width:100%;font-size:8.8pt;margin:8px 0} th{background:#0B1F3A;color:#F6F1E7;padding:4px 6px;text-align:left}
td{border:1px solid #E4DCC8;padding:3px 6px} img{max-width:100%;margin:6px 0} code{background:#EFE8D8;padding:0 3px;font-size:9pt}
p,li{orphans:3;widows:3}"""
html = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html_body}</body></html>"
(R / "_report.html").write_text(html)
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.set_content(html, wait_until="load")
    pg.pdf(path=str(R / "technical_report.pdf"), format="A4", margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
           print_background=True, display_header_footer=True,
           footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#777'>STAYCAST AI - Technical Report - page <span class='pageNumber'></span> / <span class='totalPages'></span></div>",
           header_template="<span></span>")
    b.close()
(R / "_report.html").unlink()
print("report written")
