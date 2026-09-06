from pathlib import Path

import pandas as pd

from ni_power_forecast.data.soni import normalise_soni_excel


def test_soni_normaliser_auto_detects_columns(tmp_path: Path):
    source = tmp_path / "soni.xlsx"
    out = tmp_path / "normalised.csv"
    pd.DataFrame(
        {
            "DateTime": ["2026-01-01 00:00", "2026-01-01 00:15"],
            "System Demand": [800, 805],
            "System Demand Forecast": [790, 800],
            "Wind Generation": [300, 310],
            "Wind Forecast": [295, 305],
        }
    ).to_excel(source, index=False)

    normalise_soni_excel(source, out)
    df = pd.read_csv(out)
    assert list(df.columns) == [
        "timestamp",
        "demand_actual_mw",
        "demand_forecast_mw",
        "wind_actual_mw",
        "wind_forecast_mw",
    ]
