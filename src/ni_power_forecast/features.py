from __future__ import annotations

import numpy as np
import pandas as pd

from ni_power_forecast.schema import TARGET, TIMESTAMP, FeatureSpec


def build_features(
    df: pd.DataFrame,
    spec: FeatureSpec | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build leakage-aware features for a 24-hour-ahead price model.

    Realised market/system variables are lagged by at least the
    forecast horizon. Genuine target-period forecast variables are
    included only when they are available in the dataset.
    """

    spec = spec or FeatureSpec()

    out = df.copy().sort_values(TIMESTAMP).reset_index(drop=True)

    # Store timestamps canonically in UTC, but use Northern Ireland
    # local time for calendar features.
    ts_utc = pd.to_datetime(
        out[TIMESTAMP],
        utc=True,
    )

    ts_local = ts_utc.dt.tz_convert("Europe/London")

    hour = ts_local.dt.hour.astype(float)
    dow = ts_local.dt.dayofweek.astype(float)
    month = ts_local.dt.month.astype(float)

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    out["is_weekend"] = (ts_local.dt.dayofweek >= 5).astype(int)

    feature_cols = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
    ]

    # Genuine target-period forecasts are optional.
    has_demand_forecast = "demand_forecast_mw" in out.columns

    has_wind_forecast = "wind_forecast_mw" in out.columns

    if has_demand_forecast:
        feature_cols.append("demand_forecast_mw")

    if has_wind_forecast:
        feature_cols.append("wind_forecast_mw")

    if has_demand_forecast and has_wind_forecast:
        out["net_demand_forecast_mw"] = out["demand_forecast_mw"] - out["wind_forecast_mw"]

        out["wind_share_forecast"] = out["wind_forecast_mw"] / out["demand_forecast_mw"].clip(
            lower=1
        )

        feature_cols.extend(
            [
                "net_demand_forecast_mw",
                "wind_share_forecast",
            ]
        )

    if "temperature_forecast_c" in out.columns:
        feature_cols.append("temperature_forecast_c")

    # Historical price features.
    for lag in spec.price_lags:
        if lag < spec.forecast_horizon_hours:
            raise ValueError("price lags must be >= forecast horizon")

        out[f"price_lag_{lag}"] = out[TARGET].shift(lag)

        feature_cols.append(f"price_lag_{lag}")

    # Historical realised-system variables.
    for lag in spec.actual_lags:
        if lag < spec.forecast_horizon_hours:
            raise ValueError("actual lags must be >= forecast horizon")

        demand_name = f"demand_actual_lag_{lag}"
        wind_name = f"wind_actual_lag_{lag}"

        out[demand_name] = out["demand_actual_mw"].shift(lag)

        out[wind_name] = out["wind_actual_mw"].shift(lag)

        feature_cols.extend([demand_name, wind_name])

    # Price rolling statistics are shifted by the entire
    # forecast horizon before being calculated.
    shifted_price = out[TARGET].shift(spec.forecast_horizon_hours)

    for window in spec.rolling_windows:
        mean_name = f"price_roll_mean_{window}"
        std_name = f"price_roll_std_{window}"

        out[mean_name] = shifted_price.rolling(window).mean()

        out[std_name] = shifted_price.rolling(window).std()

        feature_cols.extend([mean_name, std_name])

    X = out[feature_cols].replace([np.inf, -np.inf], np.nan)

    y = out[TARGET].copy()

    return X, y
