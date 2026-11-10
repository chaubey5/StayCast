"""Exploratory data analysis: one function per chart.

Each `fig_*` function returns (matplotlib figure, interpretation string). The
interpretation is built from the numbers on the chart, so the business reading
can never drift out of sync with the data. The notebook just calls these.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data_loader import arrival_dates, load_raw
from src.evaluate import PALETTE as P, _style
from src.feature_engineering import FeatureBuilder
from src.preprocessing import load_clean

COLUMN_DESCRIPTIONS = {
    "hotel": "Resort Hotel or City Hotel (two Portuguese properties)",
    "is_canceled": "1 if the booking was cancelled (known only after the outcome)",
    "lead_time": "Days between booking entry and arrival",
    "arrival_date_year": "Year of arrival", "arrival_date_month": "Month name of arrival",
    "arrival_date_week_number": "ISO week of arrival", "arrival_date_day_of_month": "Day of arrival",
    "stays_in_weekend_nights": "Saturday/Sunday nights booked", "stays_in_week_nights": "Mon-Fri nights booked",
    "adults": "Number of adults", "children": "Number of children", "babies": "Number of babies",
    "meal": "Meal plan booked (BB, HB, FB, SC)", "country": "Guest country (ISO-3, 'NULL' = unknown)",
    "market_segment": "Market segment (Online TA, Direct, Groups...)", "distribution_channel": "Booking channel",
    "is_repeated_guest": "1 if returning guest", "previous_cancellations": "Earlier cancellations by this guest",
    "previous_bookings_not_canceled": "Earlier completed bookings by this guest",
    "reserved_room_type": "Room type RESERVED at booking (anonymised code) - used for forecasting",
    "assigned_room_type": "Room type ASSIGNED at check-in - NOT known at prediction time, not used",
    "booking_changes": "Amendments made to the booking", "deposit_type": "No Deposit / Non Refund / Refundable",
    "agent": "Travel agent ID", "company": "Company ID", "days_in_waiting_list": "Days on waiting list",
    "customer_type": "Contract / Group / Transient / Transient-party", "adr": "Average daily rate (currency units)",
    "required_car_parking_spaces": "Parking spaces requested", "total_of_special_requests": "Count of special requests",
    "reservation_status": "Final status: Check-Out / Canceled / No-Show (outcome - leakage if used)",
    "reservation_status_date": "Date the final status was set (= cancellation date for cancelled bookings)",
}

_DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class EDAData:
    """Loads raw + clean + daily demand once for all charts."""

    def __init__(self):
        self.raw = load_raw()
        self.raw["arrival_date"] = arrival_dates(self.raw)
        self.raw["adr_w"] = self.raw["adr"].clip(upper=self.raw["adr"].quantile(0.995))
        self.clean = load_clean()
        self.builder = FeatureBuilder(self.clean)
        win = slice(config.DEMAND_START, config.DATA_END)
        self.demand = self.builder.demand_frame().loc[win]          # net room-nights (target)
        self.gross = self.builder.demand_frame(gross=True).loc[win]  # incl. later-cancelled


def _fig(w=9, h=4.2):
    plt = _style()
    return plt.subplots(figsize=(w, h))


# ------------------------------------------------------------ tables
def schema_table(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": raw.drop(columns=["arrival_date", "adr_w"], errors="ignore").dtypes.astype(str),
        "missing": raw.drop(columns=["arrival_date", "adr_w"], errors="ignore").isna().sum(),
        "NULL_string": [int((raw[c].astype(str) == "NULL").sum()) if c in raw else 0
                        for c in raw.drop(columns=["arrival_date", "adr_w"], errors="ignore").columns],
        "description": pd.Series(COLUMN_DESCRIPTIONS),
    })


# ------------------------------------------------------------ figures
def fig_missing(d: EDAData):
    raw = d.raw
    miss = {c: int(raw[c].isna().sum() + (raw[c].astype(str) == "NULL").sum()) for c in raw.columns
            if c not in ("arrival_date", "adr_w")}
    s = pd.Series(miss)
    s = s[s > 0].sort_values()
    fig, ax = _fig(8, 3.2)
    ax.barh(s.index, s.values / len(raw) * 100, color=P["teal"])
    for i, (k, v) in enumerate(s.items()):
        ax.text(v / len(raw) * 100 + 0.8, i, f"{v:,}", va="center", fontsize=9)
    ax.set_xlabel("Missing or 'NULL' (% of 119,390 bookings)"); ax.set_title("Missing values by column (raw file)")
    ax.set_xlim(0, 105)
    txt = (f"`company` is missing for {s.get('company', 0)/len(raw):.0%} and `agent` for {s.get('agent', 0)/len(raw):.0%} of bookings, "
           "so neither is used as a forecasting feature. Only 4 `children` and 488 `country` values are missing - harmless for a demand forecast.")
    return fig, txt


def fig_duplicates(d: EDAData):
    """Sensitivity of the daily demand series to the duplicate-handling decision."""
    from src.preprocessing import add_dates, expand_to_stay_nights, map_room_category, standardise

    typed = add_dates(standardise(d.raw.drop(columns=["arrival_date", "adr_w"])))
    typed = typed[(typed.nights > 0)]
    typed["room_category"] = map_room_category(typed["reserved_room_type"])
    typed = typed.rename(columns={"reservation_status_date": "status_date_raw"})
    keep_all = expand_to_stay_nights(typed.assign(status_date=typed["status_date"]))
    ka = keep_all[(keep_all.is_canceled == 0)].groupby("stay_date").size()
    ded = d.demand.sum(axis=1)
    both = ded.rename("dedup").to_frame().join(ka.rename("keep_all"), how="inner")
    corr = both.corr().iloc[0, 1]
    fig, ax = _fig(9, 3.8)
    ax.plot(both.index, both["keep_all"], color=P["grey"], lw=1.2, label="Keep all rows")
    ax.plot(both.index, both["dedup"], color=P["emerald"], lw=1.4, label="Drop exact duplicates (used)")
    ax.set_title("Daily net room-nights: effect of the duplicate decision"); ax.set_ylabel("Room-nights per day")
    ax.set_xlabel("Stay date"); ax.legend()
    ratio = both["dedup"].sum() / both["keep_all"].sum()
    txt = (f"31,994 rows (26.8%) are exact copies of another row. Dropping them lowers total room-nights to {ratio:.0%} of the keep-all level "
           f"but the two daily series have correlation {corr:.3f}, so weekly/seasonal shape - what a forecaster learns - is unchanged. "
           "The absolute level may be understated if some copies are genuine repeat bookings (no booking ID exists to prove either way).")
    return fig, txt


def fig_cancellation(d: EDAData):
    r = d.raw
    fig, axes = _style().subplots(1, 3, figsize=(13, 3.9))
    by_hotel = r.groupby("hotel")["is_canceled"].mean() * 100
    axes[0].bar(by_hotel.index, by_hotel.values, color=[P["navy"], P["teal"]])
    axes[0].set_title("Cancellation rate by hotel"); axes[0].set_ylabel("% of bookings cancelled")
    bins = pd.cut(r["lead_time"], [-1, 7, 30, 90, 180, 365, 800], labels=["0-7", "8-30", "31-90", "91-180", "181-365", "366+"])
    by_lead = r.groupby(bins, observed=True)["is_canceled"].mean() * 100
    axes[1].bar(by_lead.index.astype(str), by_lead.values, color=P["teal"])
    axes[1].set_title("Cancellation rate by lead time"); axes[1].set_xlabel("Lead time (days)")
    by_dep = r.groupby("deposit_type")["is_canceled"].mean() * 100
    axes[2].bar(by_dep.index, by_dep.values, color=P["gold"])
    axes[2].set_title("Cancellation rate by deposit type")
    for a in axes:
        a.tick_params(axis="x", labelsize=8)
    txt = (f"{r['is_canceled'].mean():.1%} of bookings are cancelled - ignoring them would overstate demand by about a third. "
           f"Cancellation climbs from {by_lead.iloc[0]:.0f}% for last-minute bookings to {by_lead.iloc[-1]:.0f}% beyond a year of lead time, "
           f"and 'Non Refund' deposits cancel at {by_dep.get('Non Refund', np.nan):.0f}% (implausibly high, so it likely reflects how the source system recorded these bookings; deposit type is not used). "
           "This is why the model counts bookings 'on the books' at the origin only if not yet cancelled by then.")
    return fig, txt


def fig_arrivals(d: EDAData):
    daily = d.raw.groupby("arrival_date").size()
    fig, ax = _fig(10, 3.8)
    ax.plot(daily.index, daily.values, color=P["navy"], lw=0.9)
    ax.set_title("Bookings per arrival date (raw file)"); ax.set_xlabel("Arrival date"); ax.set_ylabel("Bookings arriving")
    txt = (f"Arrivals span {daily.index.min().date()} to {daily.index.max().date()} ({len(daily)} consecutive days, no gaps). "
           f"Volume peaks in summer (max {daily.max()} arrivals/day) and troughs in winter; 2015 and 2017 are partial years, so the data holds only two full "
           "summers (2016, 2017) plus July-August 2015 - a key limit for learning seasonality.")
    return fig, txt


def fig_weekday(d: EDAData):
    s = d.demand.sum(axis=1)
    by = s.groupby(s.index.dayofweek).mean()
    fig, ax = _fig(7, 3.8)
    ax.bar(_DOW, by.values, color=[P["teal"]] * 4 + [P["gold"]] * 2 + [P["teal"]])
    ax.set_ylim(by.min() * 0.9, by.max() * 1.03)
    ax.set_title("Average net room-nights by weekday (stay date)"); ax.set_ylabel("Room-nights per day")
    top, low = _DOW[int(by.idxmax())], _DOW[int(by.idxmin())]
    wk = s[s.index.dayofweek >= 5].mean() / s[s.index.dayofweek < 5].mean() - 1
    txt = (f"{top} is the busiest night and {low} the quietest; weekend nights average {wk:+.1%} versus weekdays. The weekday pattern is modest "
           "but consistent, which is exactly what the seasonal-naive baseline exploits.")
    return fig, txt


def fig_monthly(d: EDAData):
    s = d.demand.sum(axis=1)
    m = s.groupby([s.index.year, s.index.month]).mean().unstack(0)
    fig, ax = _fig(9, 4)
    for yr, col in zip(m.columns, [P["grey"], P["teal"], P["gold"]]):
        ax.plot(m.index, m[yr], marker="o", color=col, label=str(yr))
    ax.set_xticks(range(1, 13), _MONTHS); ax.set_ylabel("Average room-nights per day")
    ax.set_title("Monthly average daily room-night demand by year"); ax.legend(title="Year")
    peak = int(m.mean(axis=1).idxmax())
    yoy = m.loc[1:8, 2017].mean() / m.loc[1:8, 2016].mean() - 1
    txt = (f"Demand peaks in {_MONTHS[peak-1]} and bottoms out in winter. Comparing Jan-Aug, 2017 averages {yoy:+.1%} versus 2016 "
           "(2015 has July-December only), so there is a level shift on top of the seasonal cycle - a plain 'same as last year' rule would be biased.")
    return fig, txt


def fig_room_categories(d: EDAData):
    r = d.raw
    res = r["reserved_room_type"].value_counts()
    asg = r["assigned_room_type"].value_counts().reindex(res.index.union(r["assigned_room_type"].unique())).fillna(0)
    order = res.index.tolist() + [x for x in asg.index if x not in res.index]
    fig, ax = _fig(9, 4)
    x = np.arange(len(order))
    ax.bar(x - 0.2, res.reindex(order).fillna(0).values, 0.4, color=P["navy"], label="Reserved")
    ax.bar(x + 0.2, asg.reindex(order).fillna(0).values, 0.4, color=P["gold"], label="Assigned at check-in")
    ax.set_xticks(x, order); ax.set_yscale("log"); ax.set_ylabel("Bookings (log scale)")
    ax.set_xlabel("Room type code"); ax.set_title("Room-type distribution: reserved vs assigned"); ax.legend()
    share = (r["reserved_room_type"] != r["assigned_room_type"]).mean()
    txt = (f"Type A holds {res.iloc[0]/len(r):.0%} of reservations. {share:.1%} of guests are placed in a different type from the one reserved, "
           "and assigned types I and K never appear as reservations - so assigned type is a check-in outcome. We forecast by RESERVED type and pool the rare "
           "types B, C, H, L, P into 'Other'.")
    return fig, txt


def fig_lead_time(d: EDAData):
    lt = d.raw["lead_time"]
    fig, ax = _fig(8, 3.8)
    ax.hist(lt, bins=60, color=P["teal"])
    ax.axvline(lt.median(), color=P["gold"], lw=2, label=f"Median {lt.median():.0f} days")
    ax.set_title("Lead-time distribution"); ax.set_xlabel("Lead time (days)"); ax.set_ylabel("Bookings"); ax.legend()
    txt = (f"Median lead time is {lt.median():.0f} days but the mean is {lt.mean():.0f}: a long right tail (max {lt.max()}) of very early bookings. "
           f"{(lt > 90).mean():.0%} of bookings are made more than 90 days ahead, which is what makes 'on-the-books' counts informative 7-14 days out.")
    return fig, txt


def fig_adr(d: EDAData):
    r = d.raw[d.raw["adr"] > 0]
    r = r.assign(month=r["arrival_date"].dt.month)
    fig, axes = _style().subplots(1, 2, figsize=(12, 3.9))
    for hotel, col in (("City Hotel", P["navy"]), ("Resort Hotel", P["gold"])):
        g = r[r["hotel"] == hotel].groupby("month")["adr_w"].mean()
        axes[0].plot(g.index, g.values, marker="o", color=col, label=hotel)
    axes[0].set_xticks(range(1, 13), _MONTHS, fontsize=8); axes[0].set_ylabel("Mean ADR")
    axes[0].set_title("Average daily rate by month"); axes[0].legend()
    axes[1].hist(r["adr_w"], bins=60, color=P["teal"]); axes[1].set_title("ADR distribution (winsorised at 99.5%)")
    axes[1].set_xlabel("ADR")
    peak = r.groupby("month")["adr_w"].mean()
    txt = (f"Price follows demand: mean ADR is highest in {_MONTHS[int(peak.idxmax())-1]} ({peak.max():.0f}) and lowest in {_MONTHS[int(peak.idxmin())-1]} ({peak.min():.0f}), "
           f"with the Resort Hotel far more seasonal than the City Hotel. 1,959 raw rows have ADR = 0 and one has 5,400 (excluded from the chart tail); ADR is not a model feature "
           "because it is only observed with the booking, not forecastable demand.")
    return fig, txt


def fig_seasonality_heatmap(d: EDAData):
    s = d.demand.sum(axis=1)
    h = s.groupby([s.index.month, s.index.dayofweek]).mean().unstack()
    fig, ax = _fig(8, 4.6)
    im = ax.imshow(h.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(7), _DOW); ax.set_yticks(range(12), _MONTHS); ax.grid(False)
    for i in range(12):
        for j in range(7):
            ax.text(j, i, f"{h.values[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if h.values[i, j] > h.values.mean() else P["charcoal"])
    ax.set_title("Seasonality: average room-nights by month x weekday")
    fig.colorbar(im, ax=ax, label="Room-nights per day")
    summer, winter = h.loc[[6, 7, 8]].values.mean(), h.loc[[12, 1, 2]].values.mean()
    within = (h.max(axis=1) / h.min(axis=1)).mean()
    txt = (f"Month matters more than weekday: Jun-Aug averages {summer / winter:.1f}x Dec-Feb, while the busiest weekday in a month is on average only {within - 1:.0%} above the quietest. "
           "Calendar features therefore set the level, and recent history + bookings on the books supply the day-level detail.")
    return fig, txt


def fig_correlation(d: EDAData):
    c = d.clean.copy()
    num = c[["is_canceled", "lead_time", "nights", "adults", "children", "adr"]].copy()
    num["month"] = c["arrival_date"].dt.month
    corr = num.corr()
    fig, ax = _fig(6.2, 5)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right"); ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlation of booking attributes (cleaned data)"); fig.colorbar(im, ax=ax)
    best = corr["is_canceled"].drop("is_canceled").abs().idxmax()
    txt = (f"{best} is the attribute most correlated with cancellation (r = {corr.loc[best, 'is_canceled']:.2f}); ADR vs month r = {corr.loc['adr', 'month']:.2f}. "
           "These are booking-level attributes. For forecasting we correlate demand with its own history instead (see the lag features), and correlation is not used as causal evidence.")
    return fig, txt


def fig_outliers(d: EDAData):
    r = d.raw
    fig, axes = _style().subplots(1, 3, figsize=(12, 3.8))
    nights = r["stays_in_weekend_nights"] + r["stays_in_week_nights"]
    for ax, (name, s) in zip(axes, (("Lead time (days)", r["lead_time"]), ("ADR", r["adr"]), ("Nights per stay", nights))):
        ax.boxplot(s, orientation="vertical", flierprops={"markersize": 2, "markeredgecolor": P["grey"]}, medianprops={"color": P["gold"]})
        q1, q3 = s.quantile([.25, .75]); out = int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())
        ax.set_title(f"{name}\nIQR outliers: {out:,}", fontsize=10); ax.set_xticks([])
    txt = ("Lead time, ADR and length of stay all have long right tails. ADR 5,400 and stays of 69 nights are extreme but still plausible bookings, so they are kept; "
           "they affect room-night counts by at most a handful of rows. Hard errors (zero nights, zero guests, negative ADR) are removed instead. See reports/cleaning_report.json.")
    return fig, txt


def fig_target_series(d: EDAData):
    fig, ax = _fig(10, 4.2)
    cols = [P["navy"], P["teal"], P["emerald"], P["gold"], P["grey"], P["red"]]
    ax.stackplot(d.demand.index, d.demand.T.values, labels=d.demand.columns, colors=cols, alpha=0.9)
    ax.set_title("The forecasting target: daily net room-night demand by reserved room category")
    ax.set_xlabel("Stay date"); ax.set_ylabel("Room-nights per day"); ax.set_ylim(0, d.demand.sum(axis=1).max() * 1.08)
    ax.legend(title="Category", ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    share = d.demand.sum() / d.demand.sum().sum()
    txt = (f"Category A carries {share['A']:.0%} of room-nights, D {share['D']:.0%}, E {share['E']:.0%}; F, G and Other are small ({share[['F','G','Other']].sum():.0%} together), "
           "so their daily counts are noisy and will show the largest relative error. Target = room-nights on the stay date from bookings NOT cancelled; this is booking demand, not physical occupancy.")
    return fig, txt


ALL_FIGURES = {
    "missing_values": fig_missing, "duplicates_sensitivity": fig_duplicates, "cancellation": fig_cancellation,
    "arrival_distribution": fig_arrivals, "weekday_pattern": fig_weekday, "monthly_trend": fig_monthly,
    "room_categories": fig_room_categories, "lead_time": fig_lead_time, "adr": fig_adr,
    "seasonality_heatmap": fig_seasonality_heatmap, "correlation": fig_correlation, "outliers": fig_outliers,
    "target_series": fig_target_series,
}


def main() -> None:
    import json
    d = EDAData()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    notes = {}
    for name, fn in ALL_FIGURES.items():
        fig, txt = fn(d)
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / f"eda_{name}.png", dpi=150)
        notes[name] = txt
        print(f"eda_{name}.png")
    (config.REPORTS_DIR / "eda_interpretations.json").write_text(json.dumps(notes, indent=2))


if __name__ == "__main__":
    main()
