from __future__ import annotations

import numpy as np
import pandas as pd

from ni_power_forecast.schema import TARGET, TIMESTAMP, FeatureSpec


def build_features(
    df: pd.DataFrame, spec: FeatureSpec | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Build leakage-aware features for a 24-hour-ahead price model.

    Price and actual-system features are lagged by at least the forecast horizon.
    Target-time demand/wind/weather *forecast* columns may be used contemporaneously.
    """
    spec = spec or FeatureSpec()
    out = df.copy().sort_values(TIMESTAMP).reset_index(drop=True)
    ts = pd.to_datetime(out[TIMESTAMP], utc=True)

    hour = ts.dt.hour.astype(float)
    dow = ts.dt.dayofweek.astype(float)
    month = ts.dt.month.astype(float)

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    for lag in spec.price_lags:
        if lag < spec.forecast_horizon_hours:
            raise ValueError("price lags must be >= forecast horizon")
        out[f"price_lag_{lag}"] = out[TARGET].shift(lag)

    for lag in spec.actual_lags:
        if lag < spec.forecast_horizon_hours:
            raise ValueError("actual lags must be >= forecast horizon")
        out[f"demand_actual_lag_{lag}"] = out["demand_actual_mw"].shift(lag)
        out[f"wind_actual_lag_{lag}"] = out["wind_actual_mw"].shift(lag)

    shifted_price = out[TARGET].shift(spec.forecast_horizon_hours)
    for window in spec.rolling_windows:
        out[f"price_roll_mean_{window}"] = shifted_price.rolling(window).mean()
        out[f"price_roll_std_{window}"] = shifted_price.rolling(window).std()

    out["net_demand_forecast_mw"] = out["demand_forecast_mw"] - out["wind_forecast_mw"]
    out["wind_share_forecast"] = out["wind_forecast_mw"] / out["demand_forecast_mw"].clip(lower=1)

    feature_cols = [
        "demand_forecast_mw",
        "wind_forecast_mw",
        "net_demand_forecast_mw",
        "wind_share_forecast",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
    ]

    if "temperature_forecast_c" in out.columns:
        feature_cols.append("temperature_forecast_c")
    feature_cols.extend(f"price_lag_{lag}" for lag in spec.price_lags)
    for lag in spec.actual_lags:
        feature_cols.extend([f"demand_actual_lag_{lag}", f"wind_actual_lag_{lag}"])
    for window in spec.rolling_windows:
        feature_cols.extend([f"price_roll_mean_{window}", f"price_roll_std_{window}"])

    X = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = out[TARGET].copy()
    return X, y
