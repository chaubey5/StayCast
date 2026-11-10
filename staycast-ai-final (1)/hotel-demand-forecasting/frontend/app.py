"""STAYCAST AI dashboard (Streamlit + Plotly). Every number comes from the FastAPI backend.

Run:  uvicorn backend.api:app --port 8000      (terminal 1)
      streamlit run frontend/app.py            (terminal 2)
"""
from __future__ import annotations

import datetime as dt
import json
import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("STAYCAST_API_URL", "http://127.0.0.1:8000")

NAVY, CHARCOAL, IVORY = "#0B1F3A", "#2B2F36", "#F6F1E7"
EMERALD, TEAL, GOLD, GREY, CLAY = "#2E7D6B", "#1F8A9E", "#B8985A", "#8A8F98", "#A66A5B"
CAT_COLORS = {"A": NAVY, "D": TEAL, "E": EMERALD, "F": GOLD, "G": GREY, "Other": CLAY}
PAGES = ["Executive Overview", "Forecast Intelligence", "Room Category Analytics", "Model Performance Lab",
         "Prediction Explorer", "Explainable AI", "Data & Model Health"]

st.set_page_config(page_title="STAYCAST AI", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}
h1, h2, h3 {{ font-family: 'Fraunces', Georgia, serif !important; color: {NAVY}; letter-spacing: -0.01em; }}
h1 {{ font-size: 1.9rem !important; margin-bottom: 0.1rem; }}
h3 {{ font-size: 1.15rem !important; }}
section[data-testid="stSidebar"] {{ background: {NAVY}; }}
section[data-testid="stSidebar"] * {{ color: #E9E4D6 !important; }}
section[data-testid="stSidebar"] [role="radiogroup"] label {{ padding: 6px 10px; border-radius: 8px; }}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(184,152,90,0.15); }}
.block-container {{ padding-top: 2rem; max-width: 1280px; }}
.brand {{ font-family: 'Fraunces', serif; font-size: 1.5rem; letter-spacing: 0.04em; color: #F6F1E7 !important; }}
.brand small {{ display:block; font-family:'Inter',sans-serif; font-size:0.68rem; letter-spacing:0.12em; color:{GOLD} !important; text-transform:uppercase; }}
.sub {{ color:#5B6472; margin-bottom: 1.2rem; font-size: 0.95rem; }}
.kpi {{ background:#FFFDF8; border:1px solid #E4DCC8; border-radius:12px; padding:14px 16px; box-shadow:0 1px 3px rgba(11,31,58,.06); height:100%; }}
.kpi .l {{ font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase; color:#6B7280; }}
.kpi .v {{ font-family:'Fraunces',serif; font-size:1.7rem; color:{NAVY}; line-height:1.15; margin-top:4px; }}
.kpi .s {{ font-size:0.78rem; color:#6B7280; margin-top:2px; }}
.kpi.gold {{ border-top:3px solid {GOLD}; }}
.card {{ background:#FFFDF8; border:1px solid #E4DCC8; border-radius:12px; padding:14px 18px; box-shadow:0 1px 3px rgba(11,31,58,.06); }}
.notice {{ background:#EFE8D8; border-left:3px solid {GOLD}; padding:8px 14px; border-radius:6px; font-size:0.85rem; color:#4A4F57; margin:6px 0 14px; }}
.pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.75rem; font-weight:600; }}
.pill.ok {{ background:#DCEFE9; color:#1E5E50; }} .pill.bad {{ background:#F5DEDA; color:#8C3228; }} .pill.warn {{ background:#F4E8C9; color:#7A5E1E; }}
.warnbox {{ background:#FBF1D6; border:1px solid #E6D49B; border-radius:8px; padding:8px 12px; margin:4px 0; font-size:0.88rem; }}
div[data-testid="stDataFrame"] {{ border:1px solid #E4DCC8; border-radius:8px; }}
button[kind="primary"] {{ background:{NAVY}; border:none; }}
section[data-testid="stSidebar"] .pill.ok {{ background:#DCEFE9; color:#1E5E50 !important; }}
section[data-testid="stSidebar"] .pill.bad {{ background:#F5DEDA; color:#8C3228 !important; }}
section[data-testid="stSidebar"] button {{ background:transparent !important; border:1px solid {GOLD} !important; }}
section[data-testid="stSidebar"] button:hover {{ background:rgba(184,152,90,0.2) !important; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ API client
class ApiError(Exception):
    pass


@st.cache_data(ttl=300, show_spinner=False)
def _get(path: str, params_json: str = "{}"):
    try:
        r = httpx.get(f"{API_URL}{path}", params=json.loads(params_json), timeout=60)
    except httpx.HTTPError as exc:
        raise ApiError(f"Cannot reach the API at {API_URL} ({exc.__class__.__name__}). Start it with: uvicorn backend.api:app --port 8000") from exc
    if r.status_code >= 400:
        raise ApiError(r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text)
    return r.json()


def get(path: str, **params):
    return _get(path, json.dumps({k: v for k, v in params.items() if v is not None}, default=str))


def post(path: str, payload: dict):
    try:
        r = httpx.post(f"{API_URL}{path}", json=payload, timeout=60)
    except httpx.HTTPError as exc:
        raise ApiError(f"Cannot reach the API at {API_URL} ({exc.__class__.__name__}).") from exc
    if r.status_code >= 400:
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        detail = body.get("detail", r.text)
        raise ApiError(json.dumps(detail) if not isinstance(detail, str) else detail)
    return r.json()


def safe(fn, *a, **k):
    """Call the API; show a friendly error and return None on failure."""
    try:
        return fn(*a, **k)
    except ApiError as exc:
        st.error(f"**API problem:** {exc}")
        return None


# ---------------------------------------------------------------------- helpers
def kpi(label: str, value: str, sub: str = "", gold: bool = False) -> str:
    return f'<div class="kpi {"gold" if gold else ""}"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def style(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=CHARCOAL, size=12), hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#C9CCD1")
    fig.update_yaxes(gridcolor="#E9E3D2", zeroline=False)
    return fig


def header(title: str, sub: str):
    st.markdown(f"# {title}")
    st.markdown(f'<div class="sub">{sub}</div>', unsafe_allow_html=True)


def notice(text: str):
    st.markdown(f'<div class="notice">{text}</div>', unsafe_allow_html=True)


def points_df(points: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(points)
    if not df.empty:
        df["target_date"] = pd.to_datetime(df["target_date"])
        df["origin_date"] = pd.to_datetime(df["origin_date"])
    return df


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def csv_button(df: pd.DataFrame, name: str, label: str = "Download CSV", key: str | None = None):
    st.download_button(label, df.to_csv(index=False).encode(), file_name=name, mime="text/csv", key=key)


def forecast_figure(df: pd.DataFrame, cats: list[str], total: bool, history: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    if total:
        t = df.groupby("target_date").agg(prediction=("prediction", "sum"), actual=("recorded_actual", "sum")).reset_index()
        if history is not None and not history.empty:
            h = history.groupby("date")["room_nights"].sum().reset_index()
            fig.add_scatter(x=h["date"], y=h["room_nights"], name="Recorded demand", line=dict(color=NAVY, width=2))
        else:
            fig.add_scatter(x=t["target_date"], y=t["actual"], name="Recorded demand", line=dict(color=NAVY, width=2))
        fig.add_scatter(x=t["target_date"], y=t["prediction"], name="Forecast (sum of categories)", line=dict(color=EMERALD, width=2.2))
        fig.update_yaxes(title="Room-nights per day")
        return fig
    for c in cats:
        d = df[df["room_category"] == c].sort_values("target_date")
        col = CAT_COLORS.get(c, TEAL)
        fig.add_scatter(x=pd.concat([d["target_date"], d["target_date"][::-1]]), y=pd.concat([d["upper"], d["lower"][::-1]]),
                        fill="toself", mode="lines", fillcolor=f"rgba({int(col[1:3], 16)},{int(col[3:5], 16)},{int(col[5:7], 16)},0.16)",
                        line=dict(width=0), hoverinfo="skip", showlegend=False, name=f"{c} interval")
        fig.add_scatter(x=d["target_date"], y=d["prediction"], name=f"{c} forecast", line=dict(color=col, width=2.2),
                        customdata=d[["lower", "upper"]], hovertemplate="%{y:.1f} (interval %{customdata[0]:.0f}-%{customdata[1]:.0f})")
        act = d.dropna(subset=["recorded_actual"])
        fig.add_scatter(x=act["target_date"], y=act["recorded_actual"], name=f"{c} recorded", mode="lines+markers",
                        line=dict(color=col, width=1.2, dash="dot"), marker=dict(size=4))
    fig.update_yaxes(title="Room-nights per day")
    return fig


# --------------------------------------------------------------------- sidebar
def sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="brand">STAYCAST AI<small>Hotel demand intelligence</small></div>', unsafe_allow_html=True)
        st.write("")
        default = st.query_params.get("page", PAGES[0])
        idx = PAGES.index(default) if default in PAGES else 0
        page = st.radio("Navigate", PAGES, index=idx, label_visibility="collapsed")
        st.query_params["page"] = page
        st.divider()
        try:
            h = _get("/api/health")
            pill = '<span class="pill ok">API connected</span>' if h["model_loaded"] else '<span class="pill bad">Model not loaded</span>'
            st.markdown(pill, unsafe_allow_html=True)
            if h["model_loaded"]:
                st.caption(f"Model v{h['model_version']} · data to {h['data_last_date']}")
        except ApiError:
            st.markdown('<span class="pill bad">API offline</span>', unsafe_allow_html=True)
            st.caption(f"Expected at {API_URL}")
        if st.button("Refresh data", help="Clear cached API responses"):
            st.cache_data.clear()
            st.rerun()
        st.caption("Decision support only. Forecasts are estimates, not guarantees, and not physical occupancy.")
    return page


# ----------------------------------------------------------------------- pages
def page_overview():
    header("Executive Overview", "Room-night demand outlook for the next 7 and 14 days, with model credibility alongside.")
    o = safe(get, "/api/overview")
    if not o:
        return
    notice(o["note"])
    p = o["performance"]
    c = st.columns(5)
    c[0].markdown(kpi("Next 7 days", f"{o['total_7d']:,.0f}", "forecast room-nights", True), unsafe_allow_html=True)
    c[1].markdown(kpi("Next 14 days", f"{o['total_14d']:,.0f}", "forecast room-nights", True), unsafe_allow_html=True)
    c[2].markdown(kpi("Peak day", o["peak"]["date"][5:], f"{o['peak']['room_nights']:.0f} room-nights"), unsafe_allow_html=True)
    c[3].markdown(kpi("Lowest day", o["lowest"]["date"][5:], f"{o['lowest']['room_nights']:.0f} room-nights"), unsafe_allow_html=True)
    c[4].markdown(kpi("Test WAPE (1-14d)", pct(p["selected_test"]["WAPE"]), f"baseline {pct(p['baseline_test']['WAPE'])}"), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([2.2, 1])
    with left:
        st.markdown("### Demand trend and outlook")
        hist = pd.DataFrame(o["recent_history"]); hist["date"] = pd.to_datetime(hist["date"])
        fc = pd.DataFrame(o["daily"]); fc["date"] = pd.to_datetime(fc["date"])
        fig = go.Figure()
        fig.add_scatter(x=hist["date"], y=hist["room_nights"], name="Recorded (last 28 days to origin)", line=dict(color=NAVY, width=2))
        fig.add_scatter(x=fc["date"], y=fc["prediction"], name="Forecast", line=dict(color=EMERALD, width=2.4))
        fig.add_scatter(x=fc["date"], y=fc["recorded_actual"], name="Recorded afterwards (check)", line=dict(color=GOLD, width=1.4, dash="dot"))
        fig.add_vline(x=pd.Timestamp(o["origin_date"]).timestamp() * 1000, line_dash="dash", line_color=GREY,
                      annotation_text="forecast origin", annotation_position="top left")
        fig.update_yaxes(title="Room-nights per day (all categories)")
        st.plotly_chart(style(fig, 360), use_container_width=True)
    with right:
        st.markdown("### 14-day mix")
        mix = o["by_category_14d"]
        fig = go.Figure(go.Bar(x=list(mix.values()), y=list(mix.keys()), orientation="h",
                               marker_color=[CAT_COLORS.get(k, TEAL) for k in mix], text=[f"{v:,.0f}" for v in mix.values()]))
        fig.update_yaxes(autorange="reversed", title="Room category")
        fig.update_xaxes(title="Forecast room-nights")
        st.plotly_chart(style(fig, 360, legend=False), use_container_width=True)
    st.markdown("### Business insights")
    for line in o["insights"]:
        st.markdown(f"- {line}")
    notice(o["notice"])


def page_forecast():
    header("Forecast Intelligence", "Forecast vs recorded demand with residual-based 90% prediction intervals.")
    info = safe(get, "/api/model/info")
    if not info:
        return
    cats_all = info["categories"]
    last = dt.date.fromisoformat(info["data_last_date"]); first_origin = dt.date.fromisoformat(info["first_valid_origin"])
    mode = st.radio("View", ["Forecast from one origin", "Rolling backtest over a date range"], horizontal=True)
    f1, f2, f3 = st.columns([1, 1.4, 1.6])
    horizon = f1.radio("Forecast horizon", [7, 14], horizontal=True, format_func=lambda x: f"{x} days")
    chosen = f2.multiselect("Room categories", cats_all, default=["A", "D"], help="Choose one or more reserved room types")
    total = f3.checkbox("Show all-category total instead (no interval)", value=False,
                        help="Intervals are per category and cannot be added up, so the total shows the point forecast only.")
    if not chosen and not total:
        st.info("Select at least one room category to display a forecast.")
        return
    cats_param = ",".join(chosen if not total else cats_all)

    if mode.startswith("Forecast from"):
        latest_origin = last - dt.timedelta(days=horizon)
        origin = st.date_input("Forecast origin (end of this day)", value=latest_origin, min_value=first_origin, max_value=latest_origin)
        with st.spinner("Asking the model..."):
            r = safe(get, "/api/forecast", origin_date=str(origin), horizon=horizon, room_category=cats_param)
        if not r:
            return
        df = points_df(r["points"])
        hist = None
        st.plotly_chart(style(forecast_figure(df, chosen, total, hist), 420), use_container_width=True)
        split = df["data_split"].mode().iloc[0]
        notice(f"Origin {origin}: dates fall in the <b>{split}</b> period. "
               + ("Only test-period dates are genuinely held-out; other periods were used to fit the model." if split != "test" else "These dates were never used for training or model selection."))
        table = df[["origin_date", "target_date", "horizon", "room_category", "prediction", "lower", "upper", "bookings_on_the_books", "recorded_actual", "data_split"]]
        st.dataframe(table.assign(origin_date=table.origin_date.dt.date, target_date=table.target_date.dt.date), use_container_width=True, hide_index=True)
        csv_button(table, f"staycast_forecast_{origin}_{horizon}d.csv", "Download forecast as CSV", key="dl_fc")
    else:
        d1, d2 = st.columns(2)
        start = d1.date_input("From (stay date)", value=dt.date(2017, 6, 1), min_value=first_origin, max_value=last)
        end = d2.date_input("To (stay date)", value=last, min_value=first_origin, max_value=last)
        if end < start:
            st.warning("The end date is before the start date. Adjust the date filter.")
            return
        with st.spinner("Generating rolling forecasts..."):
            r = safe(get, "/api/forecast/backtest", start=str(start), end=str(end), horizon=horizon, room_category=cats_param)
        if not r:
            return
        df = points_df(r["points"])
        if df.empty:
            st.info("No forecasts in that range.")
            return
        st.plotly_chart(style(forecast_figure(df, chosen, total), 440), use_container_width=True)
        shares = df["data_split"].value_counts(normalize=True)
        notice("Period mix: " + ", ".join(f"{k} {v:.0%}" for k, v in shares.items()) + ". In-sample (train/validation) dates look better than real life.")
        dd = df.dropna(subset=["recorded_actual"])
        if not total and not dd.empty:
            inside = ((dd["recorded_actual"] >= dd["lower"]) & (dd["recorded_actual"] <= dd["upper"])).mean()
            wape = (dd["recorded_actual"] - dd["prediction"]).abs().sum() / dd["recorded_actual"].sum()
            a, b = st.columns(2)
            a.markdown(kpi("WAPE in this window", pct(wape), f"{horizon}-day-ahead"), unsafe_allow_html=True)
            b.markdown(kpi("Actuals inside interval", pct(inside), "nominal 90%"), unsafe_allow_html=True)
        csv_button(df.drop(columns=["origin_date"]).assign(target_date=df.target_date.dt.date), f"staycast_backtest_{start}_{end}_{horizon}d.csv",
                   "Download forecast as CSV", key="dl_bt")


def page_categories():
    header("Room Category Analytics", "Demand by reserved room type (booking room-nights, cancellations excluded).")
    a = safe(get, "/api/analytics/categories")
    info = safe(get, "/api/model/info")
    if not a or not info:
        return
    cats = pd.DataFrame(a["categories"])
    top = cats.loc[cats["total_room_nights"].idxmax()]
    c = st.columns(4)
    c[0].markdown(kpi("Highest-demand category", top["room_category"], f"{top['share']:.0%} of room-nights", True), unsafe_allow_html=True)
    c[1].markdown(kpi("Avg daily demand (A)", f"{cats.loc[cats.room_category == 'A', 'avg_daily'].iloc[0]:.0f}", "room-nights per day"), unsafe_allow_html=True)
    worst = cats.loc[cats["test_WAPE"].idxmax()]
    c[2].markdown(kpi("Hardest to forecast", worst["room_category"], f"test WAPE {pct(worst['test_WAPE'])}"), unsafe_allow_html=True)
    c[3].markdown(kpi("Categories", str(len(cats)), "B, C, H, L, P pooled as Other"), unsafe_allow_html=True)
    st.write("")
    l, r = st.columns(2)
    with l:
        st.markdown("### Demand by room category")
        fig = go.Figure(go.Bar(x=cats["room_category"], y=cats["total_room_nights"], marker_color=[CAT_COLORS[x] for x in cats["room_category"]],
                               text=[f"{s:.0%}" for s in cats["share"]], textposition="outside"))
        fig.update_xaxes(title="Room category (reserved type)"); fig.update_yaxes(title="Total room-nights, Jul 2015 - Aug 2017")
        st.plotly_chart(style(fig, 330, legend=False), use_container_width=True)
    with r:
        st.markdown("### Category-wise forecasting error (test period)")
        fig = go.Figure()
        fig.add_bar(x=cats["room_category"], y=cats["baseline_test_WAPE"] * 100, name="Naive seasonal baseline", marker_color=GREY)
        fig.add_bar(x=cats["room_category"], y=cats["test_WAPE"] * 100, name="Selected model", marker_color=EMERALD)
        fig.update_layout(barmode="group"); fig.update_xaxes(title="Room category"); fig.update_yaxes(title="WAPE (%)")
        st.plotly_chart(style(fig, 330), use_container_width=True)
    st.markdown("### Category-wise trend and historical comparison")
    f1, f2 = st.columns([1.5, 1])
    sel = f1.multiselect("Room categories", list(cats["room_category"]), default=["A", "D", "E"], key="cat_trend")
    view = f2.radio("Granularity", ["Daily", "Monthly average"], horizontal=True)
    if not sel:
        st.info("Select at least one category to see its trend.")
    elif view == "Daily":
        hist = safe(get, "/api/forecast/history", start="2015-07-22", end=info["data_last_date"], room_category=",".join(sel))
        if hist:
            h = pd.DataFrame(hist["points"]); h["date"] = pd.to_datetime(h["date"])
            fig = go.Figure()
            for k in sel:
                d = h[h.room_category == k]
                fig.add_scatter(x=d["date"], y=d["room_nights"].rolling(7, min_periods=1).mean(), name=f"{k} (7-day avg)", line=dict(color=CAT_COLORS[k], width=2))
            fig.update_yaxes(title="Room-nights per day")
            st.plotly_chart(style(fig, 360), use_container_width=True)
    else:
        m = pd.DataFrame(a["monthly_avg_daily"])
        m = m[m.room_category.isin(sel)]
        fig = go.Figure()
        for k in sel:
            d = m[m.room_category == k]
            fig.add_scatter(x=d["month"], y=d["avg_daily"], name=k, mode="lines+markers", line=dict(color=CAT_COLORS[k], width=2))
        fig.update_yaxes(title="Average room-nights per day"); fig.update_xaxes(title="Month")
        st.plotly_chart(style(fig, 360), use_container_width=True)
    st.markdown("### Category summary")
    show = cats.rename(columns={"room_category": "Category", "total_room_nights": "Room-nights", "share": "Share", "avg_daily": "Avg/day",
                                "peak_day": "Peak day", "last_28d_avg": "Last 28d avg", "change_vs_prior_28d": "vs prior 28d",
                                "test_WAPE": "Test WAPE", "test_MAE": "Test MAE", "baseline_test_WAPE": "Baseline WAPE"})
    st.dataframe(show, use_container_width=True, hide_index=True,
                 column_config={"Share": st.column_config.NumberColumn(format="percent"), "vs prior 28d": st.column_config.NumberColumn(format="percent"),
                                "Test WAPE": st.column_config.NumberColumn(format="percent"), "Baseline WAPE": st.column_config.NumberColumn(format="percent")})
    notice(a["note"] + " Small categories (F, G, Other) have few room-nights per day, so the same absolute error shows as a larger percentage.")


def page_models():
    header("Model Performance Lab", "Naive seasonal baseline vs three candidates. Validation picks the model; test is a single final look.")
    m = safe(get, "/api/model/metrics")
    if not m:
        return
    scope = st.radio("Horizon scope", ["1-14 days", "1-7 days"], horizontal=True)
    key = "overall_1_14d" if scope == "1-14 days" else "horizon_1_7d"
    rows = [{"Model": "Naive seasonal baseline", "Selected": "", "Val MAE": m["baseline"]["validation"][key]["MAE"], "Val RMSE": m["baseline"]["validation"][key]["RMSE"],
             "Val WAPE": m["baseline"]["validation"][key]["WAPE"], "Test MAE": m["baseline"]["test"][key]["MAE"], "Test RMSE": m["baseline"]["test"][key]["RMSE"],
             "Test WAPE": m["baseline"]["test"][key]["WAPE"]}]
    for n, v in m["models"].items():
        rows.append({"Model": n, "Selected": "◀ chosen on validation" if n == m["selected_model"] else "", "Val MAE": v["validation"][key]["MAE"],
                     "Val RMSE": v["validation"][key]["RMSE"], "Val WAPE": v["validation"][key]["WAPE"], "Test MAE": v["test"][key]["MAE"],
                     "Test RMSE": v["test"][key]["RMSE"], "Test WAPE": v["test"][key]["WAPE"]})
    tbl = pd.DataFrame(rows)
    sel_row = tbl[tbl.Model == m["selected_model"]].iloc[0]; base_row = tbl.iloc[0]
    c = st.columns(3)
    c[0].markdown(kpi("Selected model", m["selected_model"], "lowest validation WAPE", True), unsafe_allow_html=True)
    c[1].markdown(kpi(f"Test WAPE ({scope})", pct(sel_row["Test WAPE"]), f"baseline {pct(base_row['Test WAPE'])}"), unsafe_allow_html=True)
    c[2].markdown(kpi("Test MAE", f"{sel_row['Test MAE']:.2f}", f"baseline {base_row['Test MAE']:.2f} room-nights"), unsafe_allow_html=True)
    st.write("")
    st.markdown("### Metrics table")
    num2 = {c: st.column_config.NumberColumn(format="%.2f") for c in ("Val MAE", "Val RMSE", "Test MAE", "Test RMSE")}
    st.dataframe(tbl, use_container_width=True, hide_index=True,
                 column_config={**num2, "Val WAPE": st.column_config.NumberColumn(format="percent"), "Test WAPE": st.column_config.NumberColumn(format="percent")})
    split = m["split"]
    notice(f"Chronological split by target date. Train: {split['first']} to {split['train_end']} · Validation: to {split['val_end']} · Test: to {split['test_end']}. "
           "Validation metrics come from models fitted on train only; test metrics come from the chosen configurations refitted on train+validation.")
    l, r = st.columns(2)
    with l:
        st.markdown("### Validation vs test WAPE")
        fig = go.Figure()
        fig.add_bar(x=tbl["Model"], y=tbl["Val WAPE"] * 100, name="Validation (selection)", marker_color=TEAL)
        fig.add_bar(x=tbl["Model"], y=tbl["Test WAPE"] * 100, name="Test (final)", marker_color=GOLD)
        fig.update_layout(barmode="group"); fig.update_yaxes(title="WAPE (%) - lower is better")
        st.plotly_chart(style(fig, 340), use_container_width=True)
    with r:
        st.markdown("### Test error by forecast horizon")
        hs = sorted(int(h) for h in m["baseline"]["test"]["by_horizon"])
        fig = go.Figure()
        fig.add_scatter(x=hs, y=[m["baseline"]["test"]["by_horizon"][str(h)]["WAPE"] * 100 for h in hs], name="Naive baseline", line=dict(color=GREY, dash="dash"))
        palette = {"Linear Regression": GOLD, "Random Forest": TEAL, "HistGradientBoosting": EMERALD}
        for n, v in m["models"].items():
            fig.add_scatter(x=hs, y=[v["test"]["by_horizon"][str(h)]["WAPE"] * 100 for h in hs], name=n,
                            line=dict(color=palette.get(n, NAVY), width=3 if n == m["selected_model"] else 1.5))
        fig.update_xaxes(title="Days ahead of forecast origin"); fig.update_yaxes(title="WAPE (%)")
        st.plotly_chart(style(fig, 340), use_container_width=True)
    ab = m.get("ablation_without_on_the_books")
    if ab:
        st.markdown("### What if bookings-on-the-books were unavailable?")
        a_t, b_t = ab["test"]["overall_1_14d"]["WAPE"], m["baseline"]["test"]["overall_1_14d"]["WAPE"]
        a_v, b_v = ab["validation"]["overall_1_14d"]["WAPE"], m["baseline"]["validation"]["overall_1_14d"]["WAPE"]
        st.markdown(f"Same model without pickup features: validation WAPE **{pct(a_v)}** (baseline {pct(b_v)}), test WAPE **{pct(a_t)}** (baseline {pct(b_t)}). "
                    "It still beats the baseline, but most of the accuracy comes from knowing what is already booked.")
    with st.expander("Model search log (validation WAPE for every configuration tried)"):
        st.dataframe(pd.DataFrame(m["search_log"]).assign(params=lambda d: d["params"].astype(str)), use_container_width=True, hide_index=True)
    verdict = "beats" if m["selected_beats_baseline_on_test"] else "does NOT beat"
    notice(f"Honest summary: the selected model {verdict} the naive seasonal baseline on the untouched test period. Errors here are on room-night counts per category per day.")


def page_explorer():
    header("Prediction Explorer", "Ask the saved model for one forecast. The response below is the raw API output.")
    info = safe(get, "/api/model/info")
    if not info:
        return
    last = dt.date.fromisoformat(info["data_last_date"])
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        target = c1.date_input("Prediction date (stay date)", value=dt.date(2017, 8, 25), min_value=dt.date(2015, 8, 19), max_value=last)
        cat = c2.selectbox("Room category", info["categories"], help="Reserved room type; B, C, H, L, P are pooled as Other")
        horizon = c3.slider("Forecast horizon (days ahead)", 1, 14, 7, help="The forecast is made at the end of the day this many days before the stay date")
        submitted = st.form_submit_button("Generate forecast", type="primary")
    if submitted:
        with st.spinner("Running the saved model..."):
            try:
                st.session_state["pred"] = post("/api/forecast/predict", {"target_date": str(target), "room_category": cat, "horizon": horizon})
                st.session_state["pred_error"] = None
            except ApiError as exc:
                st.session_state["pred"], st.session_state["pred_error"] = None, str(exc)
    if st.session_state.get("pred_error"):
        st.error(f"The request was rejected: {st.session_state['pred_error']}")
        return
    r = st.session_state.get("pred")
    if not r:
        st.info("Choose a date, room category and horizon, then press **Generate forecast**.")
        return
    st.success(f"Forecast generated by {r['model_name']} v{r['model_version']} (origin: end of {r['origin_date']}).")
    c = st.columns(4)
    c[0].markdown(kpi("Predicted room-nights", f"{r['prediction']:.1f}", f"{r['room_category']} on {r['target_date']}", True), unsafe_allow_html=True)
    c[1].markdown(kpi("90% interval", f"{r['lower']:.0f} - {r['upper']:.0f}", "residual-based, not a guarantee"), unsafe_allow_html=True)
    c[2].markdown(kpi("On the books at origin", f"{r['bookings_on_the_books']}", "room-nights already booked"), unsafe_allow_html=True)
    c[3].markdown(kpi("Period", r["data_split"], "held-out only if 'test'"), unsafe_allow_html=True)
    for w in r["warnings"]:
        st.markdown(f'<div class="warnbox">⚠ {w}</div>', unsafe_allow_html=True)
    st.write("")
    l, rr = st.columns(2)
    with l:
        st.markdown("### Most influential feature groups for this prediction")
        ex = pd.DataFrame(r["explanation"]).sort_values("contribution")
        fig = go.Figure(go.Bar(x=ex["contribution"], y=ex["group"], orientation="h",
                               marker_color=[EMERALD if v >= 0 else CLAY for v in ex["contribution"]]))
        fig.update_xaxes(title="Change in prediction when set to a typical value (room-nights)")
        st.plotly_chart(style(fig, 360, legend=False), use_container_width=True)
        st.caption("Sensitivity of this model, not causal effects. Groups interact, so values need not add up.")
    with rr:
        st.markdown("### Relevant historical demand")
        hc = r["history_context"]
        hist = pd.DataFrame({"date": [d[5:] for d in hc["same_weekday_previous_weeks"]], "room_nights": list(hc["same_weekday_previous_weeks"].values())}).sort_values("date")
        fig = go.Figure()
        fig.add_bar(x=hist["date"], y=hist["room_nights"], name="Same weekday, earlier weeks", marker_color=TEAL)
        fig.add_scatter(x=[r["target_date"][5:]], y=[r["prediction"]], mode="markers", name="Forecast", marker=dict(color=EMERALD, size=13, symbol="diamond"),
                        error_y=dict(type="data", symmetric=False, array=[r["upper"] - r["prediction"]], arrayminus=[r["prediction"] - r["lower"]], color=EMERALD))
        if r["recorded_actual"] is not None:
            fig.add_scatter(x=[r["target_date"][5:]], y=[r["recorded_actual"]], mode="markers", name="Recorded (for review)", marker=dict(color=GOLD, size=11))
        fig.update_yaxes(title="Room-nights"); fig.update_xaxes(type="category")
        st.plotly_chart(style(fig, 360), use_container_width=True)
        st.caption(f"Mean of the 28 days before the stay date: {hc['mean_last_28d']} · all-time max for this category: {hc['category_all_time_max']:.0f}")
    notice(r["notice"])
    with st.expander("Raw API response (POST /api/forecast/predict)"):
        st.json({k: v for k, v in r.items() if k != "feature_values"})
        csv_button(pd.DataFrame([{k: v for k, v in r.items() if k in ("origin_date", "target_date", "horizon", "room_category", "prediction", "lower", "upper")}]),
                   f"staycast_prediction_{r['target_date']}_{r['room_category']}.csv", "Download this prediction as CSV", key="dl_pred")


def page_explain():
    header("Explainable AI", "What the model leans on, how sure it is, and where humans should step in.")
    e = safe(get, "/api/model/explain")
    if not e:
        return
    st.markdown("### Global feature importance")
    imp = pd.DataFrame(e["importance"]).sort_values("mae_increase")
    fig = go.Figure(go.Bar(x=imp["mae_increase"], y=imp["group"], orientation="h", marker_color=TEAL,
                           text=[f"{s:.1%}" for s in imp["share_of_total"]], textposition="outside"))
    fig.update_xaxes(title="Increase in validation MAE when the group is shuffled (room-nights)")
    st.plotly_chart(style(fig, 380, legend=False), use_container_width=True)
    st.caption(e["method"] + " " + e["caveat"])
    ab = e.get("ablation_without_on_the_books")
    if ab:
        st.markdown(f"**Reading it:** bookings on the books dominate. Without them the same model's test WAPE is {pct(ab['test']['overall_1_14d']['WAPE'])} "
                    f"instead of the headline figure, so the forecast is really 'what is booked now, adjusted for typical cancellations and late bookings'.")
    l, r = st.columns(2)
    with l:
        st.markdown("### Uncertainty")
        u = e["uncertainty"]
        st.markdown(f"- Method: {u['method']}\n- Nominal coverage: **{u['nominal_coverage']:.0%}**\n- Observed coverage on test: **{u['test_coverage']:.1%}**\n- Mean interval width on test: **{u['mean_width_test']} room-nights**")
        st.markdown("Intervals widen with horizon and for small categories. A recorded value outside the band is expected roughly 1 time in 10.")
        st.markdown("### Individual prediction explanation")
        st.markdown("Open **Prediction Explorer**, generate a forecast and read the 'most influential feature groups' chart. Each bar shows how far the prediction moves if that group is replaced by a typical training value.")
    with r:
        st.markdown("### Limitations")
        for x in e["limitations"]:
            st.markdown(f"- {x}")
    st.markdown("### Human review recommendations")
    for x in e["human_review"]:
        st.markdown(f"- {x}")


def page_health():
    header("Data & Model Health", "What the model was built from and the state of its artifacts. Nothing here is a live feed.")
    h = safe(get, "/api/data/health")
    if not h:
        return
    d, a, ev = h["dataset"], h["artifact"], h["evaluation"]
    c = st.columns(4)
    c[0].markdown(kpi("Raw records", f"{d['raw_rows']:,}", "hotel_bookings.csv", True), unsafe_allow_html=True)
    c[1].markdown(kpi("After cleaning", f"{d['clean_rows']:,}", f"{d['exact_duplicates_dropped']:,} duplicates removed"), unsafe_allow_html=True)
    c[2].markdown(kpi("Coverage", f"{d['arrival_date_min'][:7]} to {d['arrival_date_max'][:7]}", "arrival dates"), unsafe_allow_html=True)
    ok = a["loaded"]
    c[3].markdown(kpi("Model artifact", "Loaded" if ok else "Not loaded", f"{a['size_kb']} KB · sha {a['sha256_12']}"), unsafe_allow_html=True)
    st.write("")
    l, r = st.columns(2)
    with l:
        st.markdown("### Missing values in the raw file")
        mv = pd.DataFrame({"column": list(d["missing_values_raw"]), "rows": list(d["missing_values_raw"].values())})
        mv["share"] = mv["rows"] / d["raw_rows"]
        st.dataframe(mv, hide_index=True, use_container_width=True, column_config={"share": st.column_config.NumberColumn(format="percent")})
        st.markdown("### Rows dropped as invalid")
        st.json(d["invalid_rows_dropped"])
    with r:
        st.markdown("### Model & artifact")
        m = h.get("model", {})
        st.markdown(f"- Model: **{m.get('name', 'n/a')}** · version **{m.get('version', 'n/a')}**\n- Last trained (UTC): **{m.get('trained_at_utc', 'n/a')}**\n"
                    f"- Fitted on: {m.get('fit_on', 'n/a')}\n- Artifact: `{a['path']}` · exists: **{a['exists']}** · loads: **{a['loaded']}**")
        if a.get("load_error"):
            st.error(a["load_error"])
        st.markdown("### Evaluation status")
        if ev.get("metrics_file_present") and "selected_beats_baseline_on_test" in ev:
            pill = lambda ok: f'<span class="pill {"ok" if ok else "bad"}">{"yes" if ok else "no"}</span>'  # noqa: E731
            st.markdown(f"Metrics file present · beats baseline on validation {pill(ev['selected_beats_baseline_on_validation'])} · on test {pill(ev['selected_beats_baseline_on_test'])}"
                        f"<br>Rows: {ev['rows']}", unsafe_allow_html=True)
        else:
            st.warning("No evaluation results found. Run `python -m src.train`.")
    notice(f"<b>Data freshness:</b> {d['freshness']} <b>Monitoring:</b> {h['monitoring']}")
    st.caption(f"Source: {d['source']}")


PAGE_FUNCS = {"Executive Overview": page_overview, "Forecast Intelligence": page_forecast, "Room Category Analytics": page_categories,
              "Model Performance Lab": page_models, "Prediction Explorer": page_explorer, "Explainable AI": page_explain,
              "Data & Model Health": page_health}

# Streamlit executes the script top to bottom on every interaction.
PAGE_FUNCS[sidebar()]()
