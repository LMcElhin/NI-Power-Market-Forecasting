from __future__ import annotations

import numpy as np
import pandas as pd

from ni_power_forecast.schema import TARGET, TIMESTAMP, FeatureSpec


def build_features(
    df: pd.DataFrame,
    spec: FeatureSpec | None = None,
    include_derived_forecasts: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build leakage-safe features for 24-hour-ahead price forecasting.

    Realised prices and system variables are lagged by at least the
    forecast horizon.

    Genuine target-period forecast variables are included directly when
    present.

    By default, net-demand and wind-share forecast features are derived
    automatically when both demand and wind forecasts are available.
    This can be disabled for feature-ablation experiments with
    ``include_derived_forecasts=False``.
    """

    spec = spec or FeatureSpec()

    out = (
        df.copy()
        .sort_values(TIMESTAMP)
        .reset_index(drop=True)
    )

    feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # Calendar features
    # ------------------------------------------------------------------

    ts_utc = pd.to_datetime(
        out[TIMESTAMP],
        utc=True,
    )

    ts_local = ts_utc.dt.tz_convert(
        "Europe/London"
    )

    hour = ts_local.dt.hour.astype(float)
    dow = ts_local.dt.dayofweek.astype(float)
    month = ts_local.dt.month.astype(float)

    out["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )
    out["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    out["dow_sin"] = np.sin(
        2 * np.pi * dow / 7
    )
    out["dow_cos"] = np.cos(
        2 * np.pi * dow / 7
    )

    out["month_sin"] = np.sin(
        2 * np.pi * (month - 1) / 12
    )
    out["month_cos"] = np.cos(
        2 * np.pi * (month - 1) / 12
    )

    out["is_weekend"] = (
        ts_local.dt.dayofweek >= 5
    ).astype(int)

    feature_cols.extend(
        [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "is_weekend",
        ]
    )

    # ------------------------------------------------------------------
    # Genuine target-period forecasts
    # ------------------------------------------------------------------

    has_demand_forecast = (
        "demand_forecast_mw" in out.columns
    )

    has_wind_forecast = (
        "wind_forecast_mw" in out.columns
    )

    if has_demand_forecast:
        feature_cols.append(
            "demand_forecast_mw"
        )

    if has_wind_forecast:
        feature_cols.append(
            "wind_forecast_mw"
        )

    # Preserve the original behaviour by default.
    #
    # For ablation experiments this can be disabled so that the model
    # receives only the raw demand and wind forecasts.
    if (
        include_derived_forecasts
        and has_demand_forecast
        and has_wind_forecast
    ):
        out["net_demand_forecast_mw"] = (
            out["demand_forecast_mw"]
            - out["wind_forecast_mw"]
        )

        out["wind_share_forecast"] = (
            out["wind_forecast_mw"]
            / out["demand_forecast_mw"].clip(
                lower=1
            )
        )

        feature_cols.extend(
            [
                "net_demand_forecast_mw",
                "wind_share_forecast",
            ]
        )

    if "temperature_forecast_c" in out.columns:
        feature_cols.append(
            "temperature_forecast_c"
        )

    # ------------------------------------------------------------------
    # Historical price features
    # ------------------------------------------------------------------

    for lag in spec.price_lags:
        if lag < spec.forecast_horizon_hours:
            raise ValueError(
                "price lags must be >= forecast horizon"
            )

        name = f"price_lag_{lag}"

        out[name] = out[TARGET].shift(lag)

        feature_cols.append(name)

    if {
        "price_lag_24",
        "price_lag_48",
    }.issubset(out.columns):

        out["price_change_24_48"] = (
            out["price_lag_24"]
            - out["price_lag_48"]
        )

        feature_cols.append(
            "price_change_24_48"
        )

    if {
        "price_lag_24",
        "price_lag_168",
    }.issubset(out.columns):

        out["price_change_24_168"] = (
            out["price_lag_24"]
            - out["price_lag_168"]
        )

        feature_cols.append(
            "price_change_24_168"
        )

    # ------------------------------------------------------------------
    # Historical realised system features
    # ------------------------------------------------------------------

    system_variables = {
        "demand_actual_mw": "demand_actual",
        "wind_actual_mw": "wind_actual",
        "interconnector_flow_mw": "interconnector_flow",
    }

    for lag in spec.actual_lags:
        if lag < spec.forecast_horizon_hours:
            raise ValueError(
                "actual lags must be >= forecast horizon"
            )

        for (
            raw_name,
            feature_prefix,
        ) in system_variables.items():

            if raw_name not in out.columns:
                continue

            feature_name = (
                f"{feature_prefix}_lag_{lag}"
            )

            out[feature_name] = (
                out[raw_name].shift(lag)
            )

            feature_cols.append(
                feature_name
            )

        demand_name = (
            f"demand_actual_lag_{lag}"
        )
        wind_name = (
            f"wind_actual_lag_{lag}"
        )

        if (
            demand_name in out.columns
            and wind_name in out.columns
        ):
            net_name = (
                f"net_demand_lag_{lag}"
            )

            out[net_name] = (
                out[demand_name]
                - out[wind_name]
            )

            feature_cols.append(
                net_name
            )

            wind_share_name = (
                f"wind_share_actual_lag_{lag}"
            )

            out[wind_share_name] = (
                out[wind_name]
                / out[demand_name].clip(
                    lower=1
                )
            )

            feature_cols.append(
                wind_share_name
            )

    # ------------------------------------------------------------------
    # Changes in historical system state
    # ------------------------------------------------------------------

    for prefix in (
        "demand_actual",
        "wind_actual",
        "interconnector_flow",
    ):
        lag_24 = f"{prefix}_lag_24"
        lag_48 = f"{prefix}_lag_48"

        if (
            lag_24 in out.columns
            and lag_48 in out.columns
        ):
            change_name = (
                f"{prefix}_change_24_48"
            )

            out[change_name] = (
                out[lag_24]
                - out[lag_48]
            )

            feature_cols.append(
                change_name
            )

    if {
        "net_demand_lag_24",
        "net_demand_lag_48",
    }.issubset(out.columns):

        out["net_demand_change_24_48"] = (
            out["net_demand_lag_24"]
            - out["net_demand_lag_48"]
        )

        feature_cols.append(
            "net_demand_change_24_48"
        )

    # ------------------------------------------------------------------
    # Historical price distributions
    # ------------------------------------------------------------------

    shifted_price = out[TARGET].shift(
        spec.forecast_horizon_hours
    )

    for window in spec.rolling_windows:
        rolling = shifted_price.rolling(
            window
        )

        mean_name = (
            f"price_roll_mean_{window}"
        )
        std_name = (
            f"price_roll_std_{window}"
        )
        min_name = (
            f"price_roll_min_{window}"
        )
        max_name = (
            f"price_roll_max_{window}"
        )

        out[mean_name] = rolling.mean()
        out[std_name] = rolling.std()
        out[min_name] = rolling.min()
        out[max_name] = rolling.max()

        feature_cols.extend(
            [
                mean_name,
                std_name,
                min_name,
                max_name,
            ]
        )

    if {
        "price_lag_24",
        "price_roll_mean_168",
    }.issubset(out.columns):

        out["price_vs_weekly_mean"] = (
            out["price_lag_24"]
            - out["price_roll_mean_168"]
        )

        feature_cols.append(
            "price_vs_weekly_mean"
        )

    # ------------------------------------------------------------------
    # Final feature matrix / target
    # ------------------------------------------------------------------

    X = out[
        feature_cols
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    y = out[TARGET].copy()

    return X, y