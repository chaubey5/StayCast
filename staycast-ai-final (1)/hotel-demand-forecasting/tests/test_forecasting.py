import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluate import mae, rmse, wape
from src.feature_engineering import FEATURE_COLUMNS, FeatureBuilder, truncate_as_of
from src.forecasting import DemandModel, assign_split, chronological_split, naive_seasonal_predict
from src.predict import Forecaster, InputValidationError, ModelNotAvailableError


# ------------------------------------------------------------ metrics
def test_metrics_known_values():
    y, p = np.array([10, 20, 30]), np.array([12, 18, 33])
    assert mae(y, p) == pytest.approx((2 + 2 + 3) / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert wape(y, p) == pytest.approx(7 / 60)


# ---------------------------------------------------- chronological split
def test_split_is_chronological_and_disjoint(rows):
    split = chronological_split(rows["target_date"])
    s = assign_split(rows, split)
    tr, va, te = (rows.loc[s == k, "target_date"] for k in ("train", "validation", "test"))
    assert tr.max() < va.min() and va.max() < te.min()
    assert len(tr) + len(va) + len(te) == len(rows)
    share = len(tr) / len(rows)
    assert 0.65 < share < 0.75                       # ~70% by date


def test_split_dates_match_saved_metrics(rows):
    split = chronological_split(rows["target_date"]).to_dict()
    saved = json.loads(config.METRICS_FILE.read_text())["split"]
    assert split == saved


# ------------------------------------------------------ feature / leakage
def test_no_missing_features(rows):
    assert rows[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_reference_dates_never_after_origin(rows):
    """Baseline / same-weekday references must be observable at the origin."""
    ref_gap = (rows["target_date"] - rows["origin_date"]).dt.days            # == horizon
    assert (ref_gap == rows["horizon"]).all()
    m = np.ceil(rows["horizon"] / 7).astype(int)
    ref_date = rows["target_date"] - pd.to_timedelta(7 * m, unit="D")
    assert (ref_date <= rows["origin_date"]).all()


def test_lag_features_match_history(builder, rows):
    r = rows.sample(200, random_state=config.RANDOM_SEED)
    for row in r.itertuples():
        t = builder.pos(row.origin_date)
        c = builder.cats.index(row.room_category)
        assert row.lag_0 == builder.demand[t, c]
        assert row.lag_3 == builder.demand[t - 3, c]
        assert row.roll_mean_7 == pytest.approx(builder.demand[t - 6:t + 1, c].mean())


@pytest.mark.parametrize("origin", ["2016-03-10", "2016-12-25", "2017-06-15", "2017-08-10"])
def test_features_identical_when_future_is_hidden(clean, builder, origin):
    """Leakage proof: history truncated at the origin yields the same features."""
    t = pd.Timestamp(origin)
    hidden = FeatureBuilder(truncate_as_of(clean, t))
    full, trunc = builder.make_rows([t]), hidden.make_rows([t])
    pd.testing.assert_frame_equal(full[FEATURE_COLUMNS].reset_index(drop=True), trunc[FEATURE_COLUMNS].reset_index(drop=True))


def test_changing_the_future_does_not_change_features(clean, builder):
    t = pd.Timestamp("2016-06-15")
    tampered = clean.copy()
    late = tampered["booking_date"] > t
    tampered.loc[late, "lead_time"] = 1             # distort every booking made after the origin
    tampered.loc[late, "is_canceled"] = 1 - tampered.loc[late, "is_canceled"]
    tampered = FeatureBuilder(truncate_as_of(tampered, t))
    a, b = builder.make_rows([t]), tampered.make_rows([t])
    pd.testing.assert_frame_equal(a[FEATURE_COLUMNS], b[FEATURE_COLUMNS])


def test_forbidden_columns_are_not_features():
    banned = {"y", "assigned_room_type", "reservation_status", "is_canceled", "reservation_status_date", "adr"}
    assert banned.isdisjoint(FEATURE_COLUMNS)


def test_invalid_origin_or_target_is_rejected(builder):
    with pytest.raises(Exception):
        builder.make_rows(["2015-07-30"], [7])                                   # before burn-in
    with pytest.raises(Exception):
        builder.make_rows([builder.last_date - pd.Timedelta(days=3)], [7])       # target beyond data


# ---------------------------------------------------------- baseline/model
def test_naive_baseline_is_same_weekday(rows):
    r = rows.iloc[:50]
    assert (naive_seasonal_predict(r) == r["base_same_dow"]).all()


def test_training_is_deterministic_and_non_negative(rows):
    sub = rows.iloc[::7].reset_index(drop=True)
    params = {"max_iter": 40, "learning_rate": 0.1}
    m1 = DemandModel("HistGradientBoosting", params, False).fit(sub[FEATURE_COLUMNS], sub["y"])
    m2 = DemandModel("HistGradientBoosting", params, False).fit(sub[FEATURE_COLUMNS], sub["y"])
    p1, p2 = m1.predict(sub[FEATURE_COLUMNS]), m2.predict(sub[FEATURE_COLUMNS])
    assert np.array_equal(p1, p2) and (p1 >= 0).all()


def test_residual_model_adds_baseline(rows):
    sub = rows.iloc[::11].reset_index(drop=True)
    m = DemandModel("Linear Regression", {}, True).fit(sub[FEATURE_COLUMNS], sub["y"])
    off = sub["base_same_dow"].to_numpy()
    raw = m.pipeline_.predict(sub[m.used_features_]) + off
    assert np.allclose(m.predict(sub[FEATURE_COLUMNS]), np.clip(raw, 0, None))


# ----------------------------------------------------- artifact / loading
def test_artifact_contains_trained_pipeline(forecaster):
    art = forecaster.artifact
    assert hasattr(art["model"], "pipeline_") and art["feature_columns"] == FEATURE_COLUMNS
    assert art["metadata"]["random_seed"] == config.RANDOM_SEED


def test_artifact_metrics_consistent_with_metrics_file(forecaster):
    m = json.loads(config.METRICS_FILE.read_text())
    assert m["selected_model"] == forecaster.artifact["model_name"]
    assert m["selected_beats_baseline_on_validation"] is True


def test_model_beats_baseline_on_test():
    m = json.loads(config.METRICS_FILE.read_text())
    sel = m["selected_model"]
    assert m["models"][sel]["test"]["overall_1_14d"]["WAPE"] < m["baseline"]["test"]["overall_1_14d"]["WAPE"]


def test_missing_artifact_raises_not_fabricates(tmp_path):
    with pytest.raises(ModelNotAvailableError):
        Forecaster.load(tmp_path / "nope.joblib")


def test_corrupt_artifact_raises(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a joblib file")
    with pytest.raises(ModelNotAvailableError):
        Forecaster.load(bad)


def test_incomplete_artifact_raises(tmp_path):
    p = tmp_path / "incomplete.joblib"
    joblib.dump({"model": object()}, p)
    with pytest.raises(ModelNotAvailableError):
        Forecaster.load(p)


# ----------------------------------------------------------- inference
def test_predict_one_uses_saved_model(forecaster, builder):
    res = forecaster.predict_one("2017-08-25", "A", 7)
    rows = builder.make_rows([pd.Timestamp("2017-08-18")], [7], ["A"])
    direct = float(forecaster.model.predict(rows[FEATURE_COLUMNS])[0])
    assert res["prediction"] == pytest.approx(direct, abs=0.05)
    assert res["lower"] <= res["prediction"] <= res["upper"]
    assert res["data_split"] == "test"
    assert res["explanation"] and res["history_context"]["same_weekday_previous_weeks"]


def test_prediction_changes_with_inputs(forecaster):
    a = forecaster.predict_one("2017-08-25", "A", 7)["prediction"]
    f = forecaster.predict_one("2017-08-25", "F", 7)["prediction"]
    assert a != f and a > f


def test_predictions_are_non_negative_over_whole_test_period(forecaster):
    bt = forecaster.backtest("2017-05-15", "2017-08-31", 7)
    assert (bt["prediction"] >= 0).all() and (bt["lower"] >= 0).all() and (bt["lower"] <= bt["upper"]).all()


def test_in_sample_dates_are_flagged(forecaster):
    res = forecaster.predict_one("2016-07-20", "A", 7)
    assert res["data_split"] == "train"
    assert any("fitted on" in w for w in res["warnings"])


@pytest.mark.parametrize("target,cat,h", [
    ("2017-09-20", "A", 7),         # beyond data
    ("2015-08-01", "A", 7),         # before history
    ("2017-08-25", "Z", 7),         # unknown category
    ("2017-08-25", "A", 0),
    ("2017-08-25", "A", 15),
    ("not-a-date", "A", 7),
])
def test_invalid_inputs_raise(forecaster, target, cat, h):
    with pytest.raises(InputValidationError):
        forecaster.predict_one(target, cat, h)


def test_forecast_covers_all_horizon_days(forecaster):
    df = forecaster.forecast("2017-08-17", 14)
    assert df["target_date"].nunique() == 14 and set(df["room_category"]) == set(config.CATEGORIES)


def test_test_interval_coverage_reasonable():
    m = json.loads(config.METRICS_FILE.read_text())
    assert 0.80 <= m["uncertainty"]["test_coverage"] <= 0.99
