# Final submission form - fill in the bracketed fields yourself
(Nothing below is invented; items needing your personal details or links are left blank.)

| Field | Value |
|---|---|
| Candidate name | [YOUR NAME] |
| Registration number | [YOUR REG. NO.] |
| University email | [YOUR UNIVERSITY EMAIL] |
| Internship role | AI & Machine Learning - ALGIONS AI LABS PRIVATE LIMITED |
| Assigned project title | Hotel Occupancy & Room-Demand Forecasting (product name: STAYCAST AI) |
| GitHub repository | [PASTE URL AFTER YOU PUSH] |
| Primary branch | [e.g. main] |
| Final commit ID | [run `git rev-parse HEAD` after your last commit] |
| Video link | [PASTE URL AFTER RECORDING - see docs/VIDEO_PLAN.md] |
| Demo link | [none unless you deploy; dashboard runs locally with the commands in README] |
| Technical report link | [repo path `reports/technical_report.pdf` or hosted URL] |
| Dataset source | Hotel Booking Demand, Antonio, Almeida & Nunes (2019), CC BY 4.0 - https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand |
| Model artifact location | `models/best_model.joblib` (regenerate with `python -m src.train`) |
| Evaluation metrics (test, 1-14d) | HistGradientBoosting WAPE 3.0%, MAE 1.80, RMSE 3.15 vs naive seasonal baseline WAPE 10.7%, MAE 6.31, RMSE 11.63. Validation WAPE 5.3% vs 19.0%. Without on-the-books features: test WAPE 8.0% |
| Technology stack | Python, pandas, NumPy, scikit-learn, joblib, holidays, FastAPI, Pydantic, Uvicorn, Streamlit, Plotly, Matplotlib, pytest, Playwright (screenshots), Mermaid (diagrams) |
| Known limitations | Booking demand is not occupancy (no inventory); two Portuguese hotels, ~2 years, two summers; strong reliance on bookings on the books; exact duplicates dropped by assumption; 1 booking = 1 room; intervals calibrated on one validation window; data ends 2017-08-31 so no live forecast; no drift monitoring |
| AI tools and attribution | Claude (Anthropic) was used as a coding and drafting assistant for this project. [Edit this sentence to describe exactly how YOU used and reviewed it.] |
| Originality declaration | [Write and sign this yourself after you have understood and verified the code. Suggested: "I confirm that I understand, have run and can explain this work, and that sources and AI assistance are disclosed above."] |
