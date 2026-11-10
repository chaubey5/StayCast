"""STAYCAST AI FastAPI application.  Run: uvicorn backend.api:app --reload"""
from __future__ import annotations

import datetime as dt
import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import services
from backend.schemas import ErrorResponse, ForecastResponse, HealthResponse, PredictRequest, PredictResponse
from src import config
from src.predict import InputValidationError, ModelNotAvailableError

logging.basicConfig(level=os.getenv("STAYCAST_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("staycast.api")

state = services.AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load()
    yield


app = FastAPI(
    title="STAYCAST AI API",
    description="Hotel room-night demand forecasts from a saved model. Decision support only.",
    version="1.0.0", lifespan=lifespan,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
origins = [o.strip() for o in os.getenv("STAYCAST_ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:5173").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])


# -------------------------------------------------------------- error mapping
@app.exception_handler(InputValidationError)
async def _bad_input(_: Request, exc: InputValidationError):
    return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": str(exc)})


@app.exception_handler(ModelNotAvailableError)
async def _no_model(_: Request, exc: ModelNotAvailableError):
    return JSONResponse(status_code=503, content={"error": "model_unavailable", "detail": str(exc)})


@app.exception_handler(Exception)
async def _unexpected(_: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "Unexpected server error; see server log."})


def _cats(room_category: str | None) -> list[str] | None:
    return [c.strip() for c in room_category.split(",") if c.strip()] if room_category else None


# ------------------------------------------------------------------ endpoints
@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health():
    f = state.forecaster
    if f is None:
        return HealthResponse(status="degraded", model_loaded=False, detail=state.error)
    return HealthResponse(status="ok", model_loaded=True, model_version=f.info()["version"],
                          data_last_date=f.builder.last_date.date())


@app.get("/api/model/metrics", tags=["model"])
def model_metrics():
    state.require()
    return state.metrics()


@app.get("/api/model/info", tags=["model"])
def model_info():
    return state.require().info()


@app.get("/api/model/explain", tags=["model"])
def model_explain():
    return services.explain_global(state)


@app.get("/api/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast(origin_date: dt.date = Query(description="Forecast origin (end of this day)"),
             horizon: int = Query(7, ge=1, le=14), room_category: str | None = Query(None, description="Comma separated; default all")):
    f = state.require()
    df = f.forecast(origin_date, horizon, _cats(room_category))
    return ForecastResponse(origin_date=origin_date, horizon=horizon, interval_coverage=f.artifact["interval_coverage"],
                            model_name=f.artifact["model_name"], model_version=f.artifact["metadata"]["version"],
                            points=services.points_from_frame(df), notice=services.NOTICE)


@app.post("/api/forecast/predict", response_model=PredictResponse, tags=["forecast"])
def predict(req: PredictRequest):
    res = state.require().predict_one(req.target_date, req.room_category, req.horizon)
    res["notice"] = services.NOTICE
    return res


@app.get("/api/forecast/backtest", tags=["forecast"], summary="Rolling h-day-ahead predictions over a date range")
def backtest(start: dt.date, end: dt.date, horizon: int = Query(7, ge=1, le=14), room_category: str | None = None):
    f = state.require()
    df = f.backtest(start, end, horizon, _cats(room_category))
    return {"horizon": horizon, "interval_coverage": f.artifact["interval_coverage"],
            "points": services.points_from_frame(df), "notice": services.NOTICE}


@app.get("/api/forecast/history", tags=["forecast"])
def history(start: dt.date, end: dt.date, room_category: str | None = None):
    df = state.require().history(start, end, _cats(room_category))
    return {"points": [{"date": r.date.date(), "room_category": r.room_category, "room_nights": float(r.room_nights),
                        "gross_room_nights": float(r.gross_room_nights)} for r in df.itertuples()],
            "definition": "room_nights = net demand (non-cancelled bookings); gross includes bookings later cancelled"}


@app.get("/api/overview", tags=["dashboard"])
def overview():
    return services.overview(state)


@app.get("/api/analytics/categories", tags=["dashboard"])
def categories():
    return services.category_analytics(state)


@app.get("/api/data/health", tags=["dashboard"])
def data_health():
    return services.data_health(state)
