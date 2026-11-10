"""Builds and executes notebooks/hotel_eda.ipynb.   python scripts/build_notebook.py"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("""# STAYCAST AI - Data understanding, EDA and target construction
**Dataset:** Hotel Booking Demand (Antonio, Almeida & Nunes, 2019; CC BY 4.0) - 119,390 bookings, two Portuguese hotels, arrivals 2015-07-01 to 2017-08-31.

> **Important:** the file holds *bookings*, not daily occupancy snapshots and not room inventory. The target built here is **net booking room-night demand per stay date and reserved room category**. It is never described as occupancy %.

Every chart below is produced by `src/eda.py`; the interpretation under each chart is generated from the plotted numbers."""))
C.append(code("""import sys, json
sys.path.insert(0, '..')
import pandas as pd
from IPython.display import Markdown, display
from src import config, eda
pd.set_option('display.max_colwidth', 80, 'display.width', 160)
d = eda.EDAData()
def show(name):
    fig, txt = eda.ALL_FIGURES[name](d)
    fig.tight_layout(); display(fig); display(Markdown('**Business interpretation.** ' + txt))
"""))
C.append(md("## 1. Dimensions, columns and data types"))
C.append(code("""print('Raw shape:', d.raw.drop(columns=['arrival_date','adr_w']).shape)
eda.schema_table(d.raw)"""))
C.append(md("## 2. Missing values"))
C.append(code("show('missing_values')"))
C.append(md("## 3. Duplicates\nThere is no booking ID, so identical rows cannot be proven to be the same booking. The decision (drop exact copies) is tested for sensitivity below."))
C.append(code("show('duplicates_sensitivity')"))
C.append(md("## 4. Cancellations"))
C.append(code("show('cancellation')"))
C.append(md("## 5. Arrival dates and date coverage"))
C.append(code("show('arrival_distribution')"))
C.append(md("## 6. Weekday vs weekend"))
C.append(code("show('weekday_pattern')"))
C.append(md("## 7. Monthly trends and seasonality"))
C.append(code("show('monthly_trend'); show('seasonality_heatmap')"))
C.append(md("## 8. Room categories (reserved vs assigned)"))
C.append(code("show('room_categories')"))
C.append(md("## 9. Lead time"))
C.append(code("show('lead_time')"))
C.append(md("## 10. Average daily rate"))
C.append(code("show('adr')"))
C.append(md("## 11. Correlation"))
C.append(code("show('correlation')"))
C.append(md("## 12. Outliers"))
C.append(code("show('outliers')"))
C.append(md("## 13. Cleaning decisions (reproducible: `python -m src.preprocessing`)"))
C.append(code("""rep = json.load(open(config.CLEAN_REPORT_FILE))
print({k: v for k, v in rep.items() if k != 'decisions'})
print(*rep['decisions'], sep='\\n- ')"""))
C.append(md("""## 14. Target construction
For each stay date *d* and reserved room category *c*:

`demand(d, c)` = number of booked room-nights on night *d* from bookings that were **not cancelled** (1 booking = 1 room, deduplicated).

* Booking-level rows are expanded to one row per night (`expand_to_stay_nights`).
* Cancelled / no-show bookings are **kept in the data**: they feed the *on-the-books* features and the cancellation analysis, but not the target.
* The first 21 days (stays that began before the file starts) are dropped as burn-in; nights after 2017-08-31 are excluded because later arrivals are missing."""))
C.append(code("show('target_series')"))
C.append(md("## 15. Feature leakage audit (live demonstration)\nFeatures for an origin date must be identical when the booking history is truncated at that date."))
C.append(code("""from src.feature_engineering import FeatureBuilder, truncate_as_of, FEATURE_COLUMNS
for origin in ['2016-03-10', '2017-06-15']:
    t = pd.Timestamp(origin)
    full = d.builder.make_rows([t])[FEATURE_COLUMNS]
    hidden = FeatureBuilder(truncate_as_of(d.clean, t)).make_rows([t])[FEATURE_COLUMNS]
    print(origin, 'features identical with future hidden:', full.equals(hidden))"""))
C.append(md("## 16. Chronological split and results summary (from `reports/metrics.json`)"))
C.append(code("""m = json.load(open(config.METRICS_FILE))
print('Split by target date:', m['split'], '| rows:', m['rows'])
rows = [{'model': 'Naive seasonal baseline', **{f'{p}_{k}': m['baseline'][p]['overall_1_14d'][k] for p in ('validation','test') for k in ('MAE','RMSE','WAPE')}}]
rows += [{'model': n, **{f'{p}_{k}': v[p]['overall_1_14d'][k] for p in ('validation','test') for k in ('MAE','RMSE','WAPE')}} for n, v in m['models'].items()]
pd.DataFrame(rows).round(3)"""))
C.append(code("""print('Selected on validation:', m['selected_model'])
print('Without on-the-books features (ablation) test WAPE:', m['ablation_without_on_the_books']['test']['overall_1_14d']['WAPE'])"""))
nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "hotel_eda.ipynb"
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(out.parent)}}).execute()
nbf.write(nb, out)
print("wrote", out)
