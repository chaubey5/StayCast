"""Metrics, error breakdowns and evaluation figures.

WAPE = sum|error| / sum|actual|. We lead with it because room-night counts
differ a lot between categories (A ~160/day, G ~6/day) and WAPE weights every
room-night equally, which is what drives staffing and housekeeping cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

PALETTE = {
    "navy": "#0B1F3A", "charcoal": "#2B2F36", "ivory": "#F6F1E7", "emerald": "#2E7D6B",
    "teal": "#1F8A9E", "gold": "#B8985A", "grey": "#8A8F98", "red": "#B5483C",
}


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def wape(y, p) -> float:
    y = np.asarray(y, dtype=float)
    denom = np.abs(y).sum()
    return float(np.abs(y - np.asarray(p)).sum() / denom) if denom else float("nan")


def metric_dict(y, p) -> dict:
    return {"MAE": round(mae(y, p), 3), "RMSE": round(rmse(y, p), 3),
            "WAPE": round(wape(y, p), 4), "n": int(len(y))}


def evaluate_predictions(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """Overall, 7-day, 14-day, by-horizon and by-category metrics for one model/period."""
    d = df.assign(pred=pred)
    out = {
        "overall_1_14d": metric_dict(d["y"], d["pred"]),
        "horizon_1_7d": metric_dict(d.loc[d["horizon"] <= 7, "y"], d.loc[d["horizon"] <= 7, "pred"]),
        "by_horizon": {int(h): metric_dict(g["y"], g["pred"]) for h, g in d.groupby("horizon")},
        "by_category": {str(c): metric_dict(g["y"], g["pred"]) for c, g in d.groupby("room_category")},
    }
    return out


def interval_coverage(df: pd.DataFrame, lower, upper) -> float:
    y = df["y"].to_numpy()
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


# --------------------------------------------------------------- figures
def _style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#C9CCD1",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#ECEDEF", "grid.linewidth": 0.8, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10, "font.size": 10, "legend.frameon": False,
    })
    return plt


def make_figures(preds: pd.DataFrame, metrics: dict, final_model: str, out_dir=config.FIGURES_DIR) -> list[str]:
    """Create the evaluation figures from saved out-of-sample predictions.

    `preds` columns: origin_date, target_date, horizon, room_category, y, split,
    baseline, <one column per model name>.
    """
    plt = _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    P = PALETTE
    test = preds[preds["split"] == "test"]
    names = ["Naive seasonal baseline"] + [n for n in metrics["models"]]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=150)
        plt.close(fig)
        saved.append(name)

    # 1 model comparison (validation vs test WAPE)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = names
    val = [metrics["baseline"]["validation"]["overall_1_14d"]["WAPE"]] + [
        metrics["models"][n]["validation"]["overall_1_14d"]["WAPE"] for n in names[1:]]
    tst = [metrics["baseline"]["test"]["overall_1_14d"]["WAPE"]] + [
        metrics["models"][n]["test"]["overall_1_14d"]["WAPE"] for n in names[1:]]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, np.array(val) * 100, 0.4, label="Validation (model selection)", color=P["teal"])
    ax.bar(x + 0.2, np.array(tst) * 100, 0.4, label="Test (final, untouched)", color=P["gold"])
    ax.set_xticks(x, [l.replace(" ", "\n", 1) for l in labels])
    ax.set_ylabel("WAPE (%) - lower is better")
    ax.set_title("Model comparison: WAPE, horizons 1-14 days, all room categories")
    ax.legend()
    save(fig, "model_comparison.png")

    # 2 actual vs predicted (daily total, 7-day-ahead) on test
    t7 = test[test["horizon"] == 7].groupby("target_date")[["y", "baseline", final_model]].sum()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(t7.index, t7["y"], color=P["navy"], lw=2, label="Actual")
    ax.plot(t7.index, t7[final_model], color=P["emerald"], lw=1.8, label=f"{final_model} (7-day ahead)")
    ax.plot(t7.index, t7["baseline"], color=P["grey"], lw=1.2, ls="--", label="Naive seasonal baseline")
    ax.set_title("Test period: actual vs predicted daily room-night demand (all categories, 7 days ahead)")
    ax.set_xlabel("Stay date"); ax.set_ylabel("Room-nights per day"); ax.legend(ncol=3)
    save(fig, "actual_vs_predicted_test.png")

    # 3 residual distribution
    r = (test["y"] - test[final_model])
    rb = (test["y"] - test["baseline"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    lim = np.percentile(np.abs(np.r_[r, rb]), 99)
    bins = np.linspace(-lim, lim, 60)
    ax.hist(rb, bins=bins, alpha=0.5, color=P["grey"], label="Baseline residuals")
    ax.hist(r, bins=bins, alpha=0.7, color=P["emerald"], label=f"{final_model} residuals")
    ax.axvline(0, color=P["navy"], lw=1)
    ax.set_title("Test residual distribution (actual - predicted, room-nights)")
    ax.set_xlabel("Residual (room-nights)"); ax.set_ylabel("Rows"); ax.legend()
    save(fig, "residual_distribution.png")

    # 4 error by forecast horizon
    fig, ax = plt.subplots(figsize=(8, 4.2))
    hs = sorted(int(h) for h in metrics["models"][final_model]["test"]["by_horizon"])
    ax.plot(hs, [metrics["models"][final_model]["test"]["by_horizon"][str(h) if str(h) in metrics["models"][final_model]["test"]["by_horizon"] else h]["WAPE"] * 100 for h in hs],
            marker="o", color=P["emerald"], label=final_model)
    ax.plot(hs, [metrics["baseline"]["test"]["by_horizon"][str(h) if str(h) in metrics["baseline"]["test"]["by_horizon"] else h]["WAPE"] * 100 for h in hs],
            marker="o", color=P["grey"], ls="--", label="Naive seasonal baseline")
    ax.set_title("Test WAPE by forecast horizon"); ax.set_xlabel("Days ahead of forecast origin")
    ax.set_ylabel("WAPE (%)"); ax.legend()
    save(fig, "error_by_horizon.png")

    # 5 error by room category
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cats = config.CATEGORIES
    mc = metrics["models"][final_model]["test"]["by_category"]; bc = metrics["baseline"]["test"]["by_category"]
    x = np.arange(len(cats))
    ax.bar(x - 0.2, [bc[c]["WAPE"] * 100 for c in cats], 0.4, color=P["grey"], label="Naive seasonal baseline")
    ax.bar(x + 0.2, [mc[c]["WAPE"] * 100 for c in cats], 0.4, color=P["emerald"], label=final_model)
    ax.set_xticks(x, cats); ax.set_xlabel("Room category (reserved type)"); ax.set_ylabel("WAPE (%)")
    ax.set_title("Test WAPE by room category"); ax.legend()
    save(fig, "error_by_category.png")

    # 6 prediction trend with interval for category A, 7 days ahead
    if "lower" in test.columns:
        a = test[(test["horizon"] == 7) & (test["room_category"] == "A")].sort_values("target_date")
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.fill_between(a["target_date"], a["lower"], a["upper"], color=P["teal"], alpha=0.18, label="90% prediction interval")
        ax.plot(a["target_date"], a["y"], color=P["navy"], lw=2, label="Actual")
        ax.plot(a["target_date"], a[final_model], color=P["emerald"], lw=1.6, label="Prediction")
        ax.set_title("Category A, 7 days ahead: prediction with residual-based 90% interval (test)")
        ax.set_xlabel("Stay date"); ax.set_ylabel("Room-nights per day"); ax.legend(ncol=3)
        save(fig, "prediction_interval_category_A.png")
    return saved
