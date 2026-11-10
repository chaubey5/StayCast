"""Pydantic request/response models (also drive the OpenAPI docs at /docs)."""
from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {"target_date": "2017-08-25", "room_category": "A", "horizon": 7}})
    target_date: dt.date = Field(description="Stay date to forecast (YYYY-MM-DD)")
    room_category: str = Field(min_length=1, max_length=10, description="A, D, E, F, G or Other (reserved room type)")
    horizon: int = Field(ge=1, le=14, description="Days between the forecast origin (end of day) and the target date")


class ForecastPoint(BaseModel):
    origin_date: dt.date
    target_date: dt.date
    horizon: int
    room_category: str
    prediction: float = Field(description="Predicted net room-night demand (not occupancy)")
    lower: float
    upper: float
    data_split: str = Field(description="train / validation / test - only 'test' is a genuinely held-out date")
    bookings_on_the_books: int
    recorded_actual: float | None = Field(default=None, description="Demand recorded in the historical dataset, for review")


class ForecastResponse(BaseModel):
    origin_date: dt.date
    horizon: int
    interval_coverage: float
    model_name: str
    model_version: str
    points: list[ForecastPoint]
    notice: str


class PredictResponse(BaseModel):
    origin_date: dt.date
    target_date: dt.date
    horizon: int
    room_category: str
    prediction: float
    lower: float
    upper: float
    interval_coverage: float
    data_split: str
    bookings_on_the_books: int
    recorded_actual: float | None
    model_name: str
    model_version: str
    warnings: list[str]
    explanation: list[dict[str, Any]]
    history_context: dict[str, Any]
    feature_values: dict[str, Any]
    notice: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    data_last_date: dt.date | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
