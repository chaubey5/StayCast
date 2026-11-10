"""End-to-end training: python -m src.train

1. Build leakage-safe feature rows from data/processed/bookings_clean.csv.gz
2. Chronological 70/15/15 split by target date
3. Select each model's configuration on VALIDATION only
4. Refit the chosen configurations on train+validation, score once on TEST
5. Save artifact (models/best_model.joblib), metrics.json, predictions, figures
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn

from src import config
from src.evaluate import evaluate_predictions, interval_coverage, make_figures, wape
from src.explainability import feature_statistics, group_permutation_importance, reference_row
from src.feature_engineering import FEATURE_COLUMNS, FEATURE_GROUPS, FeatureBuilder
from src.forecasting import (CANDIDATE_GRID, MODEL_NAMES, DemandModel, assign_split,
                             chronological_split, naive_seasonal_predict)
from src.preprocessing import load_clean

MODEL_VERSION = "1.0.0"
HGROUP = lambda h: "1-7" if h <= 7 else "8-14"  # noqa: E731


def residual_quantiles(val: pd.DataFrame, pred: np.ndarray, coverage: float) -> dict:
    """Empirical residual quantiles per (horizon group, room category)."""
    lo_q, hi_q = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    d = val.assign(res=val["y"].to_numpy() - pred, hg=val["horizon"].map(HGROUP))
    table = {}
    for (hg, cat), g in d.groupby(["hg", "room_category"]):
        table[f"{hg}|{cat}"] = [float(g["res"].quantile(lo_q)), float(g["res"].quantile(hi_q)), int(len(g))]
    return table


def apply_intervals(df: pd.DataFrame, pred: np.ndarray, table: dict) -> tuple[np.ndarray, np.ndarray]:
    keys = df["horizon"].map(HGROUP) + "|" + df["room_category"]
    lo = np.array([table[k][0] for k in keys]); hi = np.array([table[k][1] for k in keys])
    return np.clip(pred + lo, 0, None), np.clip(pred + hi, 0, None)


def main() -> None:
    t_start = time.time()
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    clean = load_clean()
    builder = FeatureBuilder(clean)
    rows = builder.all_rows()
    split = chronological_split(rows["target_date"])
    rows["split"] = assign_split(rows, split)
    train, val, test = (rows[rows["split"] == s].reset_index(drop=True) for s in ("train", "validation", "test"))
    trainval = pd.concat([train, val], ignore_index=True)
    print(f"rows: train={len(train):,} val={len(val):,} test={len(test):,}  split={split.to_dict()}")

    # ---------------- baseline
    base_val, base_test = naive_seasonal_predict(val), naive_seasonal_predict(test)
    metrics: dict = {
        "baseline": {"description": "Demand on the same weekday of the latest fully observed week (d - 7*ceil(h/7))",
                     "validation": evaluate_predictions(val, base_val), "test": evaluate_predictions(test, base_test)},
        "models": {},
    }
    print(f"baseline  val WAPE={wape(val.y, base_val):.4f}  test WAPE={wape(test.y, base_test):.4f}")

    # ---------------- model selection on validation only
    search_log, best_cfg, val_preds_trainonly = [], {}, {}
    for name in MODEL_NAMES:
        best = None
        for params in CANDIDATE_GRID[name]:
            for residual in (False, True):
                model = DemandModel(name, params, residual).fit(train[FEATURE_COLUMNS], train["y"])
                pv = model.predict(val[FEATURE_COLUMNS])
                score = wape(val["y"], pv)
                search_log.append({"model": name, "params": params, "residual_target": residual, "val_WAPE": round(score, 4)})
                print(f"  {name:22s} residual={residual!s:5} {params}  val WAPE={score:.4f}")
                if best is None or score < best[0]:
                    best = (score, params, residual, model, pv)
        best_cfg[name] = {"params": best[1], "residual": best[2]}
        val_preds_trainonly[name] = (best[3], best[4])
        metrics["models"][name] = {"config": {"params": best[1], "residual_target": best[2]},
                                   "validation": evaluate_predictions(val, best[4])}

    val_scores = {n: metrics["models"][n]["validation"]["overall_1_14d"]["WAPE"] for n in MODEL_NAMES}
    final_name = min(val_scores, key=val_scores.get)
    baseline_val = metrics["baseline"]["validation"]["overall_1_14d"]["WAPE"]
    beats_baseline = val_scores[final_name] < baseline_val
    print(f"selected on validation: {final_name} (val WAPE {val_scores[final_name]:.4f} vs baseline {baseline_val:.4f})")

    # ---------------- refit on train+validation, single look at TEST
    preds = test[["origin_date", "target_date", "horizon", "room_category", "y", "split"]].copy()
    preds["baseline"] = base_test
    refit = {}
    for name in MODEL_NAMES:
        cfg = best_cfg[name]
        m = DemandModel(name, cfg["params"], cfg["residual"]).fit(trainval[FEATURE_COLUMNS], trainval["y"])
        refit[name] = m
        preds[name] = m.predict(test[FEATURE_COLUMNS])
        metrics["models"][name]["test"] = evaluate_predictions(test, preds[name].to_numpy())
        print(f"  {name:22s} test WAPE={wape(test.y, preds[name]):.4f}")

    # ---------------- ablation: how much of the skill comes from on-the-books pickup features?
    otb_cols = tuple(FEATURE_GROUPS["On-the-books bookings (pickup)"])
    cfg = best_cfg[final_name]
    abl_val = DemandModel(final_name, cfg["params"], cfg["residual"], otb_cols).fit(train, train["y"])
    abl_test = DemandModel(final_name, cfg["params"], cfg["residual"], otb_cols).fit(trainval, trainval["y"])
    metrics["ablation_without_on_the_books"] = {
        "model": final_name,
        "note": "Same model/config trained WITHOUT the on-the-books (pickup) features: demand history + calendar only.",
        "validation": evaluate_predictions(val, abl_val.predict(val)),
        "test": evaluate_predictions(test, abl_test.predict(test)),
    }
    print(f"ablation (no OTB): val WAPE={wape(val.y, abl_val.predict(val)):.4f}  test WAPE={wape(test.y, abl_test.predict(test)):.4f}")

    # ---------------- uncertainty: residual quantiles from validation (train-only model)
    table = residual_quantiles(val, val_preds_trainonly[final_name][1], config.INTERVAL_COVERAGE)
    lo, hi = apply_intervals(test, preds[final_name].to_numpy(), table)
    preds["lower"], preds["upper"] = lo, hi
    coverage = interval_coverage(test, lo, hi)
    metrics["uncertainty"] = {
        "method": "Empirical residual quantiles of validation errors, per (horizon group, room category)",
        "nominal_coverage": config.INTERVAL_COVERAGE, "test_coverage": round(coverage, 4),
        "mean_width_test": round(float(np.mean(hi - lo)), 2),
    }
    print(f"interval coverage on test: {coverage:.3f} (nominal {config.INTERVAL_COVERAGE})")

    # ---------------- explainability (train-only model, held-out validation rows)
    importance = group_permutation_importance(val_preds_trainonly[final_name][0], val)

    # validation-period predictions for the figures / report
    vpreds = val[["origin_date", "target_date", "horizon", "room_category", "y", "split"]].copy()
    vpreds["baseline"] = base_val
    for n in MODEL_NAMES:
        vpreds[n] = val_preds_trainonly[n][1]
    vpreds["lower"], vpreds["upper"] = apply_intervals(val, vpreds[final_name].to_numpy(), table)
    pd.concat([vpreds, preds], ignore_index=True).to_csv(config.REPORTS_DIR / "predictions_eval.csv.gz", index=False)

    # ---------------- bookkeeping + artifact
    metrics.update({
        "split": split.to_dict(),
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "selected_model": final_name, "selection_rule": "lowest validation WAPE, horizons 1-14, all categories",
        "selected_beats_baseline_on_validation": bool(beats_baseline),
        "selected_beats_baseline_on_test": bool(
            metrics["models"][final_name]["test"]["overall_1_14d"]["WAPE"] < metrics["baseline"]["test"]["overall_1_14d"]["WAPE"]),
        "search_log": search_log, "feature_importance": importance,
    })
    metrics = json.loads(json.dumps(metrics))  # normalise keys to JSON types
    config.METRICS_FILE.write_text(json.dumps(metrics, indent=2))

    raw_sha = hashlib.sha256(config.RAW_FILE.read_bytes()).hexdigest()
    artifact = {
        "model": refit[final_name],
        "model_name": final_name,
        "feature_columns": FEATURE_COLUMNS,
        "interval_table": table,
        "interval_coverage": config.INTERVAL_COVERAGE,
        "reference_row": reference_row(trainval),
        "feature_stats": feature_statistics(trainval),
        "category_max_demand": {c: float(builder.demand[:, i].max()) for i, c in enumerate(builder.cats)},
        "split": split.to_dict(),
        "metadata": {
            "version": MODEL_VERSION,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "random_seed": config.RANDOM_SEED,
            "config": refit[final_name].describe(),
            "fit_on": f"train+validation (target dates {split.first.date()} .. {split.val_end.date()})",
            "raw_data_sha256": raw_sha,
            "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                                 "pandas": pd.__version__, "numpy": np.__version__},
            "metrics_summary": {
                "validation_WAPE": val_scores[final_name],
                "test_WAPE": metrics["models"][final_name]["test"]["overall_1_14d"]["WAPE"],
                "baseline_test_WAPE": metrics["baseline"]["test"]["overall_1_14d"]["WAPE"],
            },
        },
    }
    joblib.dump(artifact, config.MODEL_FILE, compress=3)
    figs = make_figures(pd.concat([vpreds, preds], ignore_index=True), metrics, final_name)
    print(f"figures: {figs}")
    print(f"saved {config.MODEL_FILE.relative_to(config.PROJECT_ROOT)}  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
