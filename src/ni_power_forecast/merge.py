from __future__ import annotations

import pandas as pd


def merge_price_and_system(price_df: pd.DataFrame, system_df: pd.DataFrame) -> pd.DataFrame:
    price = price_df.copy()
    system = system_df.copy()
    price["timestamp"] = pd.to_datetime(price["timestamp"], utc=True)
    system["timestamp"] = pd.to_datetime(system["timestamp"], utc=True)

    # The modelling layer is hourly; quarter-hourly SONI data are averaged here.
    system = system.set_index("timestamp").resample("h").mean(numeric_only=True).reset_index()
    merged = price.merge(system, on="timestamp", how="inner")
    return merged.sort_values("timestamp").reset_index(drop=True)
