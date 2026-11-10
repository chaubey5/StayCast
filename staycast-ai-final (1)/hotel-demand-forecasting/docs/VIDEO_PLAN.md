# 3-5 minute demonstration video plan
Record screen + voice. Run the API and dashboard first (README commands).

| Time | Show | Say (key point) |
|---|---|---|
| 0:00-0:25 | Executive Overview | Problem: hotels need 7/14-day room demand for staffing. Product name, what the screen shows. Target is booking room-nights, **not occupancy** |
| 0:25-1:05 | Notebook EDA / report figures | Data: 119,390 bookings, 2 hotels. Duplicates (31,994), cancellations (kept, 27.7% after dedupe), seasonality |
| 1:05-1:50 | Report sec. 9-10 or `tests/test_forecasting.py` | Features at end-of-day origin; on-the-books; **leakage proof**: truncated history gives identical features; chronological split |
| 1:50-2:40 | Model Performance Lab | Baseline vs 3 models, validation chooses, test once. 3.0% vs 10.7% WAPE. Be upfront: ablation shows 8.0% without on-the-books |
| 2:40-3:25 | Prediction Explorer | Pick 2017-08-25, category A, 7 days; show interval, warnings, feature sensitivity, raw API JSON; mention 422 on invalid input |
| 3:25-4:00 | Explainable AI + Data & Model Health | Importance is not causality; limitations; no live monitoring claimed |
| 4:00-4:30 | Terminal: `pytest -q` | 61 tests pass; reproducible from README |
| 4:30-5:00 | Wrap-up | Limitations and next steps (inventory for true occupancy, conformal intervals) |
