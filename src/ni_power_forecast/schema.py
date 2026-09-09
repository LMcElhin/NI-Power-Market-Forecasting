from __future__ import annotations

from dataclasses import dataclass

TARGET = "price_gbp_mwh"
TIMESTAMP = "timestamp"


REQUIRED_COLUMNS = {
    TIMESTAMP,
    TARGET,
    "demand_actual_mw",
    "wind_actual_mw",
}


OPTIONAL_COLUMNS = {
    "demand_forecast_mw",
    "wind_forecast_mw",
    "temperature_actual_c",
    "temperature_forecast_c",
    "interconnector_flow_mw",
    "source",
}


@dataclass(frozen=True)
class FeatureSpec:
    """Configuration for leakage-safe 24-hour-ahead features."""

    forecast_horizon_hours: int = 24

    # Historical prices known by the forecast origin.
    price_lags: tuple[int, ...] = (
        24,
        25,
        26,
        27,
        28,
        48,
        72,
        168,
        336,
    )

    # Realised system quantities must also be delayed by at least
    # the forecast horizon.
    actual_lags: tuple[int, ...] = (
        24,
        48,
        168,
    )

    # Rolling price windows are calculated only after shifting by
    # the forecast horizon.
    rolling_windows: tuple[int, ...] = (
        24,
        72,
        168,
    )
