from __future__ import annotations

import pandas as pd

from ni_power_forecast.schema import TARGET


def merge_price_and_system(
    price_df: pd.DataFrame,
    system_df: pd.DataFrame,
) -> pd.DataFrame:
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

    # SEMO NI-DA prices are half-hourly.
    # Convert to a single hourly price by averaging both
    # half-hour delivery periods.
    price_hourly = price.set_index("timestamp")[[TARGET]].resample("h").mean().reset_index()

    # SONI data are 15-minute average SCADA values.
    system_hourly = (
        system.set_index("timestamp").resample("h").mean(numeric_only=True).reset_index()
    )

    merged = price_hourly.merge(
        system_hourly,
        on="timestamp",
        how="inner",
    )

    merged["source"] = "SEMO NI-DA + SONI"

    return merged.sort_values("timestamp").reset_index(drop=True)
