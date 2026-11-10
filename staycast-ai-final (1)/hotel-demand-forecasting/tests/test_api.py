import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import api, services
from src.feature_engineering import FEATURE_COLUMNS


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["model_loaded"] is True


def test_model_info_and_metrics(client):
    info = client.get("/api/model/info").json()
    assert info["model_name"] and info["categories"] == ["A", "D", "E", "F", "G", "Other"]
    m = client.get("/api/model/metrics").json()
    assert {"baseline", "models", "split", "uncertainty"} <= set(m)
    assert set(m["models"]) == {"Linear Regression", "Random Forest", "HistGradientBoosting"}


def test_predict_returns_real_model_output(client):
    body = {"target_date": "2017-08-25", "room_category": "A", "horizon": 7}
    r = client.post("/api/forecast/predict", json=body)
    assert r.status_code == 200
    j = r.json()
    # independent check: same number straight from the saved model
    f = api.state.forecaster
    rows = f.builder.make_rows([pd.Timestamp("2017-08-18")], [7], ["A"])
    direct = float(f.model.predict(rows[FEATURE_COLUMNS])[0])
    assert j["prediction"] == pytest.approx(direct, abs=0.05)
    assert j["lower"] <= j["prediction"] <= j["upper"]
    assert "not guaranteed" in j["notice"].lower()


def test_predict_differs_by_input(client):
    a = client.post("/api/forecast/predict", json={"target_date": "2017-08-25", "room_category": "A", "horizon": 7}).json()
    b = client.post("/api/forecast/predict", json={"target_date": "2017-07-04", "room_category": "D", "horizon": 3}).json()
    assert a["prediction"] != b["prediction"]


@pytest.mark.parametrize("body", [
    {"target_date": "2017-08-25", "room_category": "Z", "horizon": 7},
    {"target_date": "2017-10-25", "room_category": "A", "horizon": 7},
    {"target_date": "2017-08-25", "room_category": "A", "horizon": 99},
    {"target_date": "garbage", "room_category": "A", "horizon": 7},
    {"room_category": "A", "horizon": 7},
])
def test_invalid_predict_requests_return_422(client, body):
    r = client.post("/api/forecast/predict", json=body)
    assert r.status_code == 422


def test_forecast_endpoint(client):
    r = client.get("/api/forecast", params={"origin_date": "2017-08-17", "horizon": 14, "room_category": "A,D"})
    assert r.status_code == 200
    pts = r.json()["points"]
    assert len(pts) == 28 and {p["room_category"] for p in pts} == {"A", "D"}
    assert client.get("/api/forecast", params={"origin_date": "2017-08-30", "horizon": 14}).status_code == 422
    assert client.get("/api/forecast", params={"origin_date": "2017-08-17", "horizon": 30}).status_code == 422


def test_backtest_and_history(client):
    r = client.get("/api/forecast/backtest", params={"start": "2017-06-01", "end": "2017-06-10", "horizon": 7, "room_category": "A"})
    assert r.status_code == 200 and len(r.json()["points"]) == 10
    h = client.get("/api/forecast/history", params={"start": "2017-06-01", "end": "2017-06-05", "room_category": "A"})
    assert h.status_code == 200 and len(h.json()["points"]) == 5
    assert client.get("/api/forecast/backtest", params={"start": "2017-06-10", "end": "2017-06-01", "horizon": 7}).status_code == 422


def test_explain_overview_categories_health(client):
    e = client.get("/api/model/explain").json()
    assert e["importance"] and e["limitations"] and e["human_review"]
    o = client.get("/api/overview").json()
    assert o["total_14d"] > o["total_7d"] > 0 and o["insights"]
    c = client.get("/api/analytics/categories").json()
    assert c["highest_demand_category"] == "A"
    d = client.get("/api/data/health").json()
    assert d["artifact"]["loaded"] is True and d["dataset"]["raw_rows"] == 119390


def test_missing_model_gives_503_not_fake_output(tmp_path, monkeypatch):
    broken = services.AppState(model_path=tmp_path / "missing.joblib")
    monkeypatch.setattr(api, "state", broken)
    with TestClient(api.app) as c:                      # lifespan -> broken.load()
        h = c.get("/api/health").json()
        assert h["status"] == "degraded" and h["model_loaded"] is False
        r = c.post("/api/forecast/predict", json={"target_date": "2017-08-25", "room_category": "A", "horizon": 7})
        assert r.status_code == 503 and r.json()["error"] == "model_unavailable"
