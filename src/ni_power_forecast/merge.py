from __future__ import annotations

import pandas as pd

from ni_power_forecast.schema import TARGET


def merge_price_and_system(
    price_df: pd.DataFrame,
    system_df: pd.DataFrame,
    forecast_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create the hourly modelling table.

    Prices:
        SEMO NI-DA half-hourly -> hourly mean.

    System:
        SONI quarter-hourly -> hourly mean.

    Forecasts:
        Already converted to hourly values by semo_forecasts.py.
    """

    price = price_df.copy()
    system = system_df.copy()

    price["timestamp"] = pd.to_datetime(
        price["timestamp"],
        utc=True,
    )

    system["timestamp"] = pd.to_datetime(
        system["timestamp"],
        utc=True,
    )

    price_hourly = price.set_index("timestamp")[[TARGET]].resample("h").mean().reset_index()

    system_hourly = (
        system.set_index("timestamp").resample("h").mean(numeric_only=True).reset_index()
    )

    merged = price_hourly.merge(
        system_hourly,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    if forecast_df is not None:
        forecasts = forecast_df.copy()

        forecasts["timestamp"] = pd.to_datetime(
            forecasts["timestamp"],
            utc=True,
        )

        merged = merged.merge(
            forecasts,
            on="timestamp",
            how="left",
            validate="one_to_one",
        )

    merged["source"] = "SEMO NI-DA + SONI"

    if forecast_df is not None:
        merged["source"] += " + point-in-time SEMO forecasts"

    return merged.sort_values("timestamp").reset_index(drop=True)
