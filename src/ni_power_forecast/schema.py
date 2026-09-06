from __future__ import annotations

from dataclasses import dataclass

TARGET = "price_gbp_mwh"
TIMESTAMP = "timestamp"

REQUIRED_COLUMNS = {
    TIMESTAMP,
    TARGET,
    "demand_actual_mw",
    "demand_forecast_mw",
    "wind_actual_mw",
    "wind_forecast_mw",
}

OPTIONAL_COLUMNS = {
    "temperature_actual_c",
    "temperature_forecast_c",
    "interconnector_flow_mw",
    "source",
}


@dataclass(frozen=True)
class FeatureSpec:
    forecast_horizon_hours: int = 24
    price_lags: tuple[int, ...] = (24, 48, 168)
    actual_lags: tuple[int, ...] = (24, 168)
    rolling_windows: tuple[int, ...] = (24, 168)
